/**
 * opencode-lazy-load
 *
 * Strips ALL tool definitions from every LLM request. The LLM only sees
 * load_tool as a callable tool. To use any other tool (built-in, user-installed,
 * or MCP), the LLM must call load_tool — there is no other path.
 *
 * What gets sent per message:
 *   body.tools   → [load_tool] only (every other tool is REMOVED from the array)
 *   load_tool.description → includes pointer list of available tools
 *
 * Two modes:
 *   load_tool({name: "read"})                    → returns full instructions + schema
 *   load_tool({name: "read", args: {path: "/x"}}) → executes read({path: "/x"})
 *
 * The execute mode is rewritten in the SSE response stream before opencode
 * sees it, so opencode dispatches the real tool normally.
 *
 * INSTALL:
 *   Place this file at .opencode/plugin/lazy-load.ts
 *   Opencode auto-discovers plugins from .opencode/plugin/
 *
 * REMOVE:
 *   Delete the file. Everything returns to normal immediately.
 *
 * ENFORCEMENT: mechanical, not prompt-based. The LLM literally cannot call
 * any tool directly — the tool is not in the array. No throw, no error,
 * no prompt.
 */

import type { Plugin } from "@opencode-ai/plugin"
import { tool } from "@opencode-ai/plugin"

// ─── TEMP DEBUG LOGGING ───
// @ts-ignore - import.meta.dir is a Bun runtime global, not in standard ImportMeta types
const DEBUG_LOG_PATH = `${import.meta.dir}/log/lazy-load-debug.log`
async function debugLog(msg: string) {
  try {
    // @ts-ignore
    const existing = (await Bun.file(DEBUG_LOG_PATH).exists()) ? await Bun.file(DEBUG_LOG_PATH).text() : ""
    // @ts-ignore
    await Bun.write(DEBUG_LOG_PATH, existing + `[${new Date().toISOString()}] ${msg}\n`)
  } catch {}
}

// ─── State ───────────────────────────────────────────────────────────────────

/**
 * Full original descriptions keyed by toolID.
 * Populated when tool.definition fires for each tool.
 * Never cleared — descriptions are static across the process lifetime.
 */
const originals = new Map<string, string>()

/**
 * Original JSON schemas keyed by toolID.
 * Saved from output.jsonSchema in the tool.definition hook so load_tool
 * can return the full parameter info to the LLM on demand.
 */
const originalSchemas = new Map<string, any>()

/**
 * Full original descriptions for MCP tools, keyed by tool name.
 * MCP tools bypass the tool.definition hook (verified at session/tools.ts
 * L117-201). The fetch wrapper identifies MCP tools as: any tool in
 * body.tools that's NOT in `originals` (built-in) and NOT `load_tool`.
 * Saves their description + schema before removing them from the HTTP body.
 */
const mcpOriginals = new Map<string, string>()
const mcpSchemas = new Map<string, any>()

/**
 * Per-turn loaded-tools tracking. Keyed by sessionID. Persists across
 * multiple fetch calls within the SAME turn (one user message = one turn,
 * which may span multiple LLM API calls as the LLM does multi-step tool use).
 * Cleared when the SSE stream ends (finish_reason or [DONE]).
 */
const turnLoaded = new Map<string, Set<string>>()

// ─── Helpers ─────────────────────────────────────────────────────────────────

function isLoadToolName(name: string): boolean {
  return name === "load_tool" || name.endsWith("_load_tool")
}

/**
 * Extract a brief one-line summary from a full tool description.
 * Takes the first sentence or first line (whichever is shorter),
 * cleans up template variable remnants, and truncates to ~80 chars.
 */
function briefOf(description: string): string {
  if (!description) return ""
  const byPeriod = description.split(".")[0]
  const byNewline = description.split("\n")[0].trim()
  let candidate = byPeriod.length <= byNewline.length ? byPeriod : byNewline
  // Clean up any unexpanded template variable remnants like ${intro}
  candidate = candidate.replace(/\$\{[^}]*\}/g, "").trim()
  if (candidate.length < 5) return ""
  return candidate.length > 80 ? candidate.slice(0, 77) + "..." : candidate
}

/**
 * Build the pointer list for load_tool's description.
 * Format: "- toolname - brief description"
 */
function buildPointerList(): string {
  const pointers: string[] = []
  for (const [name, desc] of originals) {
    if (isLoadToolName(name)) continue
    const brief = briefOf(desc)
    pointers.push(brief ? `- ${name} - ${brief}` : `- ${name}`)
  }
  // MCP tools are deliberately EXCLUDED from the pointer list.
  // They are still callable directly (the SSE transform passes them through
  // untouched), but they do NOT appear in load_tool's description.
  // This keeps load_tool's token footprint minimal — MCP tools add zero
  // tokens to the tools array. The LLM discovers MCP tools through other
  // channels (system prompt, etc.) and can call them directly.
  return pointers.sort().join("\n")
}

/**
 * Check if a string is parseable as complete JSON.
 * (Same logic as @ai-sdk/openai-compatible's isParsableJson.)
 */
function isParsableJson(str: string): boolean {
  if (!str) return false
  try { JSON.parse(str); return true } catch { return false }
}

const DSML_TOOL_CALLS_START = "<｜｜DSML｜｜tool_calls>"
const DSML_TOOL_CALLS_END = "</｜｜DSML｜｜tool_calls>"

function parseDSMLAttributes(segment: string | undefined): Record<string, string> | undefined {
  const input = (segment || "").trim()
  if (!input) return {}
  if (!/^[^\s="]+\s*=\s*"[^"]*"(?:\s+[^\s="]+\s*=\s*"[^"]*")*$/.test(input)) return undefined

  const attributes: Record<string, string> = Object.create(null)
  for (const attribute of input.matchAll(/([^\s="]+)\s*=\s*"([^"]*)"/g)) {
    if (Object.prototype.hasOwnProperty.call(attributes, attribute[1])) return undefined
    attributes[attribute[1]] = attribute[2]
  }
  return attributes
}

function parseDSMLCalls(block: string): Array<{ name: string; arguments: string }> {
  const calls: Array<{ name: string; arguments: string }> = []
  const invokePattern = /<｜｜DSML｜｜invoke(?:\s+([^>]*?))?>([\s\S]*?)<\/｜｜DSML｜｜invoke>/g
  const parameterPattern = /<｜｜DSML｜｜parameter(?:\s+([^>]*?))?>([\s\S]*?)<\/｜｜DSML｜｜parameter>/g
  const inner = block.slice(DSML_TOOL_CALLS_START.length, -DSML_TOOL_CALLS_END.length)
  const invokes = Array.from(inner.matchAll(invokePattern))
  let invokeEnd = 0

  for (const invoke of invokes) {
    if (inner.slice(invokeEnd, invoke.index).trim()) return []
    invokeEnd = invoke.index + invoke[0].length
    const name = parseDSMLAttributes(invoke[1])?.name
    if (!name) return []

    const args: Record<string, string> = Object.create(null)
    const parameters = Array.from(invoke[2].matchAll(parameterPattern))
    let parameterEnd = 0
    for (const parameter of parameters) {
      if (invoke[2].slice(parameterEnd, parameter.index).trim()) return []
      parameterEnd = parameter.index + parameter[0].length
      const parameterName = parseDSMLAttributes(parameter[1])?.name
      if (!parameterName) return []
      if (Object.prototype.hasOwnProperty.call(args, parameterName)) return []
      args[parameterName] = parameter[2]
    }
    if (invoke[2].slice(parameterEnd).trim()) return []
    calls.push({ name, arguments: JSON.stringify(args) })
  }

  if (inner.slice(invokeEnd).trim()) return []
  return calls
}

type KnownTool = { name: string; kind: "built-in" | "mcp" }

function resolveKnownTool(name: string): KnownTool | undefined {
  if (originals.has(name)) return { name, kind: "built-in" }
  if (mcpOriginals.has(name)) return { name, kind: "mcp" }

  const lowerName = name.toLowerCase()
  const foldedNames = new Set([
    ...Array.from(originals.keys()).filter((knownName) => knownName.toLowerCase() === lowerName),
    ...Array.from(mcpOriginals.keys()).filter((knownName) => knownName.toLowerCase() === lowerName),
  ])
  if (foldedNames.size !== 1) return undefined

  const [resolvedName] = foldedNames
  if (originals.has(resolvedName)) return { name: resolvedName, kind: "built-in" }
  if (mcpOriginals.has(resolvedName)) return { name: resolvedName, kind: "mcp" }
  return undefined
}

function normalizeSchemaValue(value: any, schema: any): any {
  const types = Array.isArray(schema?.type)
    ? schema.type.filter((type: unknown) => type !== "null")
    : [schema?.type]
  if (types.length !== 1) return value

  switch (types[0]) {
    case "number":
    case "integer": {
      if (typeof value !== "string" || value.trim() === "") return value
      const number = Number(value)
      if (!Number.isFinite(number)) return value
      if (types[0] === "integer" && !Number.isSafeInteger(number)) return value
      return number
    }
    case "boolean":
      if (value === "true") return true
      if (value === "false") return false
      return value
    case "array": {
      let array = value
      if (typeof array === "string") {
        try {
          const parsed = JSON.parse(array)
          if (!Array.isArray(parsed)) return value
          array = parsed
        } catch {
          return value
        }
      }
      if (!Array.isArray(array) || !schema.items) return array
      return array.map((item: any) => normalizeSchemaValue(item, schema.items))
    }
    case "object": {
      let object = value
      if (typeof object === "string") {
        try {
          const parsed = JSON.parse(object)
          if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return value
          object = parsed
        } catch {
          return value
        }
      }
      if (!object || typeof object !== "object" || Array.isArray(object)) return object
      if (!schema.properties || typeof schema.properties !== "object") return object
      const normalized = { ...object }
      for (const [property, propertySchema] of Object.entries(schema.properties)) {
        if (Object.prototype.hasOwnProperty.call(normalized, property)) {
          normalized[property] = normalizeSchemaValue(normalized[property], propertySchema)
        }
      }
      return normalized
    }
    default:
      return value
  }
}

function normalizeToolArguments(argumentsJson: string, schema: any): string {
  if (!schema || !isParsableJson(argumentsJson)) return argumentsJson
  return JSON.stringify(normalizeSchemaValue(JSON.parse(argumentsJson), schema))
}

// ─── Fetch wrapper (request + response interception) ─────────────────────────
//
// REQUEST side: Remove ALL tools except load_tool from body.tools. The LLM
// only sees load_tool. Pointers go into load_tool's description so the LLM
// knows what tools exist but cannot call them directly.
//
// RESPONSE side: When the LLM calls load_tool in execute mode (has "args"
// field), rewrite the tool_call to the real tool name + args before opencode
// parses it. opencode then dispatches the real tool from prepared.tools.
//
// Verified from opencode source:
//   - session/llm.ts line 128: opencode looks up prepared.tools[toolName]
//   - prepared.tools is opencode's internal map, SEPARATE from body.tools
//   - Removing tools from body.tools does NOT affect prepared.tools
//   - The AI SDK serializes tools into body.tools at fetch time
//
// opencode's provider closure uses `const fetchFn = customFetch ?? fetch`
// where `fetch` resolves to globalThis.fetch at CALL TIME, not definition time.
// So wrapping globalThis.fetch BEFORE the first LLM call works.

let _originalFetch: typeof fetch | null = null
let _fetchWrapped = false

function wrapFetch(): void {
  if (_fetchWrapped) return
  _fetchWrapped = true
  _originalFetch = globalThis.fetch

  globalThis.fetch = async function (input: RequestInfo | URL, init?: RequestInit): Promise<Response> {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : (input as Request).url
    // Detect LLM API calls. The AI SDK appends "/chat/completions" (OpenAI-compatible)
    // or "/v1/messages" (Anthropic) to the provider baseURL. We check for both the
    // path AND common provider domains to catch all cases including custom proxies.
    const isLLM = url.includes("/chat/completions") || url.includes("/v1/messages") ||
      url.includes("/messages") && url.includes("anthropic") ||
      url.includes("api.deepseek.com") || url.includes("api.openai.com") ||
      url.includes("anthropic.com") || url.includes("openrouter.ai")
    if (!isLLM || !init) return _originalFetch!.call(globalThis, input, init)

    // Extract sessionID from request headers. NO shared "__unknown__" fallback
    // — that causes cross-session state leaks. Per-request unique ID if missing.
    let sessionID = ""
    try {
      const h = init.headers
      const headers = h instanceof Headers
        ? h
        : Array.isArray(h) ? new Headers(h as any) : h ? new Headers(h as any) : new Headers()
      sessionID = headers.get("x-opencode-session") || headers.get("x-session-id") || headers.get("X-Session-Id") || ""
    } catch {}
    if (!sessionID) {
      sessionID = `__req_${Date.now()}_${Math.random().toString(36).slice(2)}__`
    }
    let loadToolName = "load_tool"

    // ── Request-side: remove ALL tools except load_tool ──
    // The LLM only sees load_tool. Pointers go into load_tool's description.
    // This is the REAL blinding — tools not in the array cannot be called.
    if (init.body) {
      let bodyText = ""
      if (typeof init.body === "string") bodyText = init.body
      else if (init.body instanceof Uint8Array || init.body instanceof ArrayBuffer) bodyText = new TextDecoder().decode(init.body)
      else if (init.body instanceof Blob) bodyText = await init.body.text()

      if (bodyText) {
        try {
          const body = JSON.parse(bodyText)
          if (Array.isArray(body.tools)) {
            const gateway = body.tools.find((t: any) => {
              const name = t?.function?.name || t?.name || ""
              return isLoadToolName(name)
            })
            if (gateway) {
              loadToolName = gateway.function?.name || gateway.name
            }

            // Save MCP tools (not in originals) before removing them
            for (const t of body.tools) {
              const fn = t?.function
              const name = fn?.name || t?.name || ""
              if (!name || isLoadToolName(name)) continue
              if (originals.has(name)) {
                // Built-in tool: capture its JSON schema here (jsonSchema is
                // undefined in the tool.definition hook — it's only generated
                // by the AI SDK at serialization time, which is this point).
                // MCP tools are NOT touched here — they fall through below.
                const params = fn?.parameters || t?.parameters
                if (params && !originalSchemas.has(name)) {
                  originalSchemas.set(name, params)
                }
                continue
              }
              const desc = fn?.description || t?.description || ""
              const params = fn?.parameters || t?.parameters
              if (!mcpOriginals.has(name)) {
                mcpOriginals.set(name, desc)
              }
              if (params && !mcpSchemas.has(name)) {
                mcpSchemas.set(name, params)
              }
            }

            // Keep ONLY load_tool in the tools array
            body.tools = body.tools.filter((t: any) => {
              const name = t?.function?.name || t?.name || ""
              return isLoadToolName(name)
            })

            // STRIP prior load_tool calls AND their results from the messages
            // array — but ONLY those before the LAST user message. This
            // prevents context accumulation across turns while preserving
            // the current turn's load_tool result so the LLM can use it.
            //
            // The API requires every assistant tool_call to be followed by a
            // matching tool-result. So we must remove BOTH sides:
            //   1. The tool-result message (role:"tool", tool_call_id matches)
            //   2. The tool_call entry from the preceding assistant message
            //      (if the assistant message has no other tool_calls and no
            //       text content, remove it entirely)
            if (Array.isArray(body.messages)) {
              // Find index of the last user message — anything before it
              // is prior turns (eligible for stripping); anything from it
              // onward is the current turn (kept intact).
              let lastUserIdx = -1
              for (let i = body.messages.length - 1; i >= 0; i--) {
                if (body.messages[i].role === "user") { lastUserIdx = i; break }
              }
              if (lastUserIdx > 0) {
                const priorMessages = body.messages.slice(0, lastUserIdx)
                // Find tool_call_ids that belong to load_tool in prior turns
                const loadToolCallIds = new Set<string>()
                for (const m of priorMessages) {
                  if (m.role === "assistant" && Array.isArray(m.tool_calls)) {
                    for (const tc of m.tool_calls) {
                      if (isLoadToolName(tc?.function?.name || "") && tc?.id) {
                        loadToolCallIds.add(tc.id)
                      }
                    }
                  }
                }
                // Filter prior messages
                const filteredPrior: any[] = []
                for (const m of priorMessages) {
                  if (m.role === "tool" && loadToolCallIds.has(m.tool_call_id)) {
                    continue
                  }
                  if (m.role === "assistant" && Array.isArray(m.tool_calls)) {
                    m.tool_calls = m.tool_calls.filter((tc: any) => !isLoadToolName(tc?.function?.name || ""))
                    if (m.tool_calls.length === 0) {
                      // Delete the empty tool_calls array — some providers (DeepSeek)
                      // reject "tool_calls: []" with "Expected an array with minimum length 1"
                      delete m.tool_calls
                      const hasText = typeof m.content === "string" && m.content.length > 0
                      if (!hasText) continue
                    }
                  }
                  filteredPrior.push(m)
                }
                body.messages = [...filteredPrior, ...body.messages.slice(lastUserIdx)]

                // Second pass: scan ALL messages for empty tool_calls arrays.
                // Some providers (DeepSeek) reject "tool_calls: []" with
                // "Expected an array with minimum length 1". Delete any empty
                // tool_calls field we find, anywhere in the messages array.
                for (const m of body.messages) {
                  if (Array.isArray(m.tool_calls) && m.tool_calls.length === 0) {
                    delete m.tool_calls
                  }
                }
              }
            }

            // Append pointer list to load_tool's description
            const pointerList = buildPointerList()
            if (pointerList) {
              for (const t of body.tools) {
                const fn = t?.function
                if (fn && isLoadToolName(fn.name)) {
                  fn.description = [
                    "Gateway tool — the only tool you can call directly.",
                    "All other tools are accessed through this tool.",
                    "",
                    "Available tools:",
                    pointerList,
                    "",
                    "Usage:",
                    '  Load instructions: call with {"name": "toolname"}',
                    "  After loading, call the real tool directly on your next turn.",
                  ].join("\n")
                }
              }
            }

            init = { ...init, body: JSON.stringify(body) }
          }
        } catch {
          // Body wasn't valid JSON — send as-is
        }
      }
    }

    const response = await _originalFetch!.call(globalThis, input, init)

    // Only intercept SSE streaming responses
    const contentType = response.headers.get("content-type") || ""
    if (!contentType.includes("text/event-stream") || !response.body) return response

    const transformed = response.body.pipeThrough(createSSETransform(sessionID, loadToolName))
    return new Response(transformed, {
      status: response.status,
      statusText: response.statusText,
      headers: response.headers,
    })
  }
}

/**
 * TransformStream that parses OpenAI-compatible SSE chunks.
 *
 * Since the LLM can only call load_tool (everything else was removed from
 * body.tools), we buffer load_tool calls until arguments are complete JSON,
 * then decide:
 *   - Load mode {name: "X"} → pass through as load_tool (returns instructions)
 *   - Execute mode {name: "X", args: {...}} → rewrite to X({...}) so opencode
 *     dispatches the real tool from prepared.tools
 *
 * The challenge: tool_call arguments arrive in chunks across multiple SSE
 * events. We must buffer load_tool calls until arguments are complete JSON.
 *
 * Verified against @ai-sdk/openai-compatible:
 *   - Line 758: iterates delta.tool_calls[]
 *   - Line 776: reads toolName from delta.tool_calls[].function.name
 *   - Line 826: subsequent chunks APPEND to toolCalls[index].function.arguments
 *   - Line 833: when accumulated args become parseable JSON, emits tool-call
 */
function createSSETransform(sessionID: string, loadToolName: string): TransformStream<Uint8Array, Uint8Array> {
  const decoder = new TextDecoder()
  const encoder = new TextEncoder()
  let buffer = ""
  // Native fragments stay keyed by their upstream index until emission.
  const toolBuffers = new Map<number, {
    id?: string
    name?: string
    arguments: string
    hasFunction: boolean
  }>()
  const nativeToolIndexes = new Map<number, number>()
  const nativeSourceIndex = Symbol("nativeSourceIndex")
  const nativeRewrite = Symbol("nativeRewrite")
  type TextField = "content" | "reasoning_content"
  type TextFragment = { field: TextField; text: string; parsed: any }
  let textField: TextField | undefined
  let textBuffer = ""
  let textFragments: TextFragment[] = []
  let nextEmittedToolIndex = 0
  let hasPendingDSMLCall = false
  // Get or create this session's turn-loaded set. Persists across multiple
  // fetch calls within ONE turn (one user message). Cleared when finish_reason
  // "stop" is seen in the SSE stream — that's the LLM's end-of-turn signal.
  // The AI SDK's loop continues (next fetch) only when finish_reason is
  // "tool-calls"; "stop" means the turn is done.
  function getTurnLoaded(): Set<string> {
    if (!turnLoaded.has(sessionID)) turnLoaded.set(sessionID, new Set())
    return turnLoaded.get(sessionID)!
  }

  function toolCall(index: number, id: string | undefined, name: string, argumentsJson: string): any {
    return {
      index,
      id,
      type: "function",
      function: { name, arguments: argumentsJson },
    }
  }

  function allocateToolIndex(): number {
    return nextEmittedToolIndex++
  }

  function nativeToolIndex(sourceIndex: unknown): number {
    if (typeof sourceIndex !== "number") return allocateToolIndex()
    let emittedIndex = nativeToolIndexes.get(sourceIndex)
    if (emittedIndex === undefined) {
      emittedIndex = allocateToolIndex()
      nativeToolIndexes.set(sourceIndex, emittedIndex)
    }
    return emittedIndex
  }

  function markNativeCall(
    call: any,
    sourceIndex: unknown,
    rewrite?: { id?: string; name: string; arguments: string },
  ): any {
    Object.defineProperty(call, nativeSourceIndex, { value: sourceIndex })
    if (rewrite) Object.defineProperty(call, nativeRewrite, { value: rewrite })
    return call
  }

  function rewriteCompletedCall(index: number, id: string | undefined, name: string, argumentsJson: string): any {
    const callArgs = JSON.parse(argumentsJson)

    if (isLoadToolName(name)) {
      const requestedName = typeof callArgs?.name === "string" ? callArgs.name : ""
      const resolvedName = resolveKnownTool(requestedName)?.name || requestedName
      if (resolvedName) getTurnLoaded().add(resolvedName)
      return toolCall(
        index,
        id,
        loadToolName,
        resolvedName !== requestedName
          ? JSON.stringify({ ...callArgs, name: resolvedName })
          : argumentsJson,
      )
    }

    const knownTool = resolveKnownTool(name)
    if (knownTool?.kind === "built-in") {
      if (getTurnLoaded().has(knownTool.name)) {
        return toolCall(
          index,
          id,
          knownTool.name,
          normalizeToolArguments(argumentsJson, originalSchemas.get(knownTool.name)),
        )
      }

      getTurnLoaded().add(knownTool.name)
      return toolCall(index, id, loadToolName, JSON.stringify({ name: knownTool.name }))
    }

    const mcpName = knownTool?.kind === "mcp" ? knownTool.name : undefined
    return toolCall(
      index,
      id,
      mcpName || name,
      mcpName ? normalizeToolArguments(argumentsJson, mcpSchemas.get(mcpName)) : argumentsJson,
    )
  }

  function enqueueParsed(controller: TransformStreamDefaultController<Uint8Array>, parsed: any): void {
    const calls = parsed?.choices?.[0]?.delta?.tool_calls
    if (Array.isArray(calls)) {
      for (let index = 0; index < calls.length; index++) {
        const call = calls[index]
        if (call && typeof call === "object" && nativeSourceIndex in call) {
          const emittedIndex = nativeToolIndex(call[nativeSourceIndex])
          const rewrite = call[nativeRewrite]
          if (rewrite) {
            calls[index] = rewriteCompletedCall(
              emittedIndex,
              rewrite.id,
              rewrite.name,
              rewrite.arguments,
            )
          } else {
            call.index = emittedIndex
          }
        }
      }
    }
    controller.enqueue(encoder.encode(`data: ${JSON.stringify(parsed)}\n\n`))
  }

  function minimalTextEvent(): any {
    return { choices: [{ delta: {} }] }
  }

  function enqueueWithoutText(
    controller: TransformStreamDefaultController<Uint8Array>,
    parsed: any,
    field: TextField,
  ): void {
    const choice = parsed?.choices?.[0]
    const delta = choice?.delta
    if (delta) delete delta[field]
    const hasData = Object.keys(parsed || {}).some((key) => key !== "choices")
      || (Array.isArray(parsed?.choices) && parsed.choices.length > 1)
      || Object.keys(choice || {}).some((key) => key !== "delta")
      || Object.keys(delta || {}).length > 0
    if (hasData) enqueueParsed(controller, parsed)
  }

  function emitBufferedText(
    controller: TransformStreamDefaultController<Uint8Array>,
    length: number,
  ): void {
    textBuffer = textBuffer.slice(length)
    while (length > 0) {
      const fragment = textFragments[0]
      const consumed = Math.min(length, fragment.text.length)
      const parsed = fragment.parsed
      parsed.choices[0].delta[fragment.field] = fragment.text.slice(0, consumed)
      enqueueParsed(controller, parsed)

      fragment.text = fragment.text.slice(consumed)
      fragment.parsed = minimalTextEvent()
      length -= consumed
      if (!fragment.text) textFragments.shift()
    }
    if (!textBuffer) textField = undefined
  }

  function replaceBufferedText(
    controller: TransformStreamDefaultController<Uint8Array>,
    length: number,
    calls: any[],
  ): void {
    const field = textField!
    const extraEnvelopes: any[] = []
    let callEnvelope: any
    textBuffer = textBuffer.slice(length)

    while (length > 0) {
      const fragment = textFragments[0]
      const consumed = Math.min(length, fragment.text.length)
      if (!callEnvelope) callEnvelope = fragment.parsed
      else extraEnvelopes.push(fragment.parsed)

      fragment.text = fragment.text.slice(consumed)
      fragment.parsed = minimalTextEvent()
      length -= consumed
      if (!fragment.text) textFragments.shift()
    }

    const delta = callEnvelope.choices[0].delta
    delete delta[field]
    delta.tool_calls = [...calls, ...(Array.isArray(delta.tool_calls) ? delta.tool_calls : [])]
    enqueueParsed(controller, callEnvelope)
    for (const parsed of extraEnvelopes) enqueueWithoutText(controller, parsed, field)
    if (!textBuffer) textField = undefined
  }

  function possibleStartSuffixLength(text: string): number {
    for (let length = Math.min(text.length, DSML_TOOL_CALLS_START.length - 1); length > 0; length--) {
      if (DSML_TOOL_CALLS_START.startsWith(text.slice(-length))) return length
    }
    return 0
  }

  function processBufferedText(controller: TransformStreamDefaultController<Uint8Array>): number {
    let convertedCalls = 0
    while (textBuffer) {
      const start = textBuffer.indexOf(DSML_TOOL_CALLS_START)
      if (start < 0) {
        emitBufferedText(controller, textBuffer.length - possibleStartSuffixLength(textBuffer))
        return convertedCalls
      }
      if (start > 0) {
        emitBufferedText(controller, start)
        continue
      }

      const end = textBuffer.indexOf(DSML_TOOL_CALLS_END, DSML_TOOL_CALLS_START.length)
      if (end < 0) return convertedCalls
      const blockLength = end + DSML_TOOL_CALLS_END.length
      const parsedCalls = parseDSMLCalls(textBuffer.slice(0, blockLength))
      if (parsedCalls.length === 0) {
        emitBufferedText(controller, blockLength)
        continue
      }

      const calls = parsedCalls.map((call) => rewriteCompletedCall(
        allocateToolIndex(),
        `call_${globalThis.crypto.randomUUID().replace(/-/g, "")}`,
        call.name,
        call.arguments,
      ))
      convertedCalls += calls.length
      replaceBufferedText(controller, blockLength, calls)
    }
    return convertedCalls
  }

  function appendText(
    controller: TransformStreamDefaultController<Uint8Array>,
    parsed: any,
    field: TextField,
    text: string,
  ): number {
    if (textField && textField !== field) emitBufferedText(controller, textBuffer.length)
    textField = field
    textBuffer += text
    textFragments.push({ field, text, parsed })
    return processBufferedText(controller)
  }

  function flushBufferedText(controller: TransformStreamDefaultController<Uint8Array>): void {
    if (textBuffer) emitBufferedText(controller, textBuffer.length)
  }

  function flushToolBuffers(controller: TransformStreamDefaultController<Uint8Array>): void {
    for (const [sourceIndex, buf] of toolBuffers) {
      if (!buf.hasFunction) continue
      const name = isLoadToolName(buf.name || "") ? loadToolName : buf.name || loadToolName
      enqueueParsed(controller, {
        choices: [{ delta: { tool_calls: [toolCall(
          sourceIndex,
          buf.id,
          name,
          buf.arguments,
        )].map((call) => markNativeCall(call, sourceIndex)) } }],
      })
    }
    toolBuffers.clear()
  }

  return new TransformStream<Uint8Array, Uint8Array>({
    transform(chunk, controller) {
      buffer += decoder.decode(chunk, { stream: true })

      const events = buffer.split(/\n\n|\r\n\r\n/)
      buffer = events.pop() || ""

      for (const event of events) {
        const lines = event.split(/\n|\r\n/)
        for (const line of lines) {
          if (!line.startsWith("data:")) continue
          const data = line.startsWith("data: ") ? line.slice(6) : line.slice(5)
          if (data === "[DONE]") {
            flushBufferedText(controller)
            flushToolBuffers(controller)
            controller.enqueue(encoder.encode("data: [DONE]\n\n"))
            continue
          }

          let resetTurnOnStop = false
          try {
            const parsed = JSON.parse(data)
            const toolCalls = parsed?.choices?.[0]?.delta?.tool_calls
            const finishReason = parsed?.choices?.[0]?.finish_reason
            resetTurnOnStop = false
            const finishEvent = finishReason != null
              ? JSON.parse(JSON.stringify(parsed))
              : undefined
            if (finishEvent) parsed.choices[0].finish_reason = null
            const incomingDelta = parsed?.choices?.[0]?.delta
            const hasIncomingText = (["content", "reasoning_content"] as TextField[])
              .some((field) => typeof incomingDelta?.[field] === "string" && incomingDelta[field].length > 0)

            if ((Array.isArray(toolCalls) || finishReason != null) && !hasIncomingText) {
              flushBufferedText(controller)
            }
            let shouldEmit = true

            if (Array.isArray(toolCalls)) {
              const filtered: any[] = []

              for (const tc of toolCalls) {
                if (!tc || typeof tc !== "object") {
                  filtered.push(tc)
                  continue
                }
                const idx = tc.index

                if (!tc.function) {
                  if (typeof idx === "number") {
                    const buf = toolBuffers.get(idx) || { arguments: "", hasFunction: false }
                    if (tc.id) buf.id = tc.id
                    toolBuffers.set(idx, buf)
                  }
                  filtered.push(markNativeCall({ ...tc }, idx))
                  continue
                }

                // First chunk for this index has the tool name; subsequent
                // chunks only append arguments.
                if (!toolBuffers.has(idx)) {
                  toolBuffers.set(idx, {
                    id: tc.id,
                    name: tc.function.name,
                    arguments: tc.function.arguments || "",
                    hasFunction: true,
                  })
                } else {
                  const buf = toolBuffers.get(idx)!
                  if (tc.id) buf.id = tc.id
                  if (tc.function.name) buf.name = tc.function.name
                  buf.arguments += tc.function.arguments || ""
                  buf.hasFunction = true
                }

                const buf = toolBuffers.get(idx)!
                if (!isParsableJson(buf.arguments)) {
                  // Still buffering — don't emit yet
                  continue
                }

                // Arguments complete — process by name
                const name = buf.name || ""
                toolBuffers.delete(idx)
                filtered.push(markNativeCall(
                  toolCall(0, buf.id, name, buf.arguments),
                  idx,
                  { id: buf.id, name, arguments: buf.arguments },
                ))
              }

              if (filtered.length > 0) {
                parsed.choices[0].delta.tool_calls = filtered
              } else {
                // All tool_calls are buffered — emit chunk without tool_calls
                // (but keep text/finish_reason if present)
                delete parsed.choices[0].delta.tool_calls
                const delta = parsed.choices[0].delta
                shouldEmit = Boolean(delta.content || delta.reasoning || delta.reasoning_content)
              }
            }

            const delta = parsed?.choices?.[0]?.delta
            const fields = (["content", "reasoning_content"] as TextField[])
              .filter((field) => typeof delta?.[field] === "string" && delta[field].length > 0)
            let convertedDSMLCalls = 0
            if (shouldEmit && fields.length > 0) {
              const values = fields.map((field) => delta[field] as string)
              for (let index = 0; index < fields.length; index++) {
                const field = fields[index]
                const textEvent = index === 0 ? parsed : minimalTextEvent()
                if (index === 0) {
                  for (const otherField of fields.slice(1)) delete textEvent.choices[0].delta[otherField]
                }
                textEvent.choices[0].delta[field] = values[index]
                convertedDSMLCalls += appendText(controller, textEvent, field, values[index])
              }
            } else if (shouldEmit && (!finishEvent || Array.isArray(delta?.tool_calls))) {
              enqueueParsed(controller, parsed)
            }
            if (convertedDSMLCalls > 0) hasPendingDSMLCall = true

            if (finishEvent) {
              flushBufferedText(controller)
              flushToolBuffers(controller)
              if (finishReason === "stop" && hasPendingDSMLCall) {
                finishEvent.choices[0].finish_reason = "tool_calls"
                resetTurnOnStop = false
              }
              hasPendingDSMLCall = false
              const finishDelta = finishEvent.choices[0].delta
              if (finishDelta && typeof finishDelta === "object") {
                delete finishDelta.content
                delete finishDelta.reasoning_content
                delete finishDelta.tool_calls
              }
              enqueueParsed(controller, finishEvent)
            }
          } catch {
            // Not valid JSON — pass through unchanged
            flushBufferedText(controller)
            controller.enqueue(encoder.encode(`data: ${data}\n\n`))
          }
          if (resetTurnOnStop) {
            const hadEntry = turnLoaded.has(sessionID)
            turnLoaded.delete(sessionID)
            debugLog(`turnLoaded CLEARED via finish_reason:stop — sessionID=${sessionID} hadEntry=${hadEntry} remainingSize=${turnLoaded.size}`)
          }
        }
      }
    },
    flush(controller) {
      flushBufferedText(controller)
      flushToolBuffers(controller)
      if (buffer) {
        controller.enqueue(encoder.encode(buffer))
      }
    },
  })
}

// ─── Plugin ──────────────────────────────────────────────────────────────────

const LazyLoadPlugin: Plugin = async (_input, _options) => {
  // Wrap fetch BEFORE the first LLM call. The wrapper removes all tools
  // except load_tool from the request body, and rewrites load_tool execute
  // calls to real tool calls in the SSE response — no throw, no error, no prompt.
  wrapFetch()

    return {
    // ── TEMP DEBUG: log real event payloads ──────────────────────────────────
    async event({ event }) {
      if (event.type === "session.error" || event.type === "session.deleted") {
        debugLog(`EVENT FIRED: type=${event.type} rawPayload=${JSON.stringify(event)}`)
        const sessionID = (event as any).properties?.sessionID
        if (sessionID) {
          const hadEntry = turnLoaded.has(sessionID)
          turnLoaded.delete(sessionID)
          debugLog(`turnLoaded CLEARED via event:${event.type} — sessionID=${sessionID} hadEntry=${hadEntry} remainingSize=${turnLoaded.size}`)
        }
      }
      if (event.type === "message.part.updated") {
        const part = (event as any)?.properties?.part
        if (part?.reason === "stop") {
          const sessionID = (event as any)?.properties?.sessionID
          if (sessionID) {
            const hadEntry = turnLoaded.has(sessionID)
            turnLoaded.delete(sessionID)
            debugLog(`turnLoaded CLEARED via reason:stop — sessionID=${sessionID} hadEntry=${hadEntry} remainingSize=${turnLoaded.size}`)
          }
        }
      }
    },

    // ── New tool: load_tool ──────────────────────────────────────────────────

    tool: {
      load_tool: tool({
        description: [
          "Gateway tool — the only tool you can call directly.",
          "All other tools are accessed through this tool.",
          "",
          "Usage:",
          '  Load instructions: call with {"name": "toolname"}',
          "  After loading, call the real tool directly on your next turn.",
        ].join("\n"),
        args: {
          name: tool.schema
            .string()
            .describe("Tool name to load instructions for"),
        },
        async execute(args, context) {
          const full = originals.get(args.name) || mcpOriginals.get(args.name)
          const schema = originalSchemas.get(args.name) || mcpSchemas.get(args.name)

          if (!full) {
            const allKnown = Array.from(new Set([...originals.keys(), ...mcpOriginals.keys()])).sort()
            return {
              title: `Unknown tool: ${args.name}`,
              output: `No instructions found for "${args.name}". Available tools: ${allKnown.join(", ")}`,
            }
          }

          // No global tracking — the SSE transform tracks loaded state per-stream.
          // load_tool just returns the instructions; the LLM calls the real tool next.

          // Build output: full description + parameter schema
          let output = full
          if (schema) {
            try {
              output += "\n\n--- Parameter schema ---\n" + JSON.stringify(schema, null, 2)
            } catch {
              // If schema can't be serialized, skip it
            }
          }

          return {
            title: `Loaded: ${args.name}`,
            output,
          }
        },
      }),
    },

    // ── Hook: tool.definition ────────────────────────────────────────────────
    //
    // Saves the original full description and JSON schema on first encounter.
    // Stripping is no longer needed here — the fetch wrapper removes all tools
    // except load_tool from the HTTP body. But we still need to save originals
    // so load_tool.execute can return them.

    async "tool.definition"(input, output) {
      // Never modify our own tool
      if (isLoadToolName(input.toolID)) return

      if (!originals.has(input.toolID)) {
        originals.set(input.toolID, output.description)
      }

      const outAny = output as any
      if (outAny.jsonSchema !== undefined && !originalSchemas.has(input.toolID)) {
        originalSchemas.set(input.toolID, outAny.jsonSchema)
      }
    },
  }
}

// ─── Export (v1 plugin format) ───────────────────────────────────────────────

export default {
  id: "opencode-lazy-load",
  server: LazyLoadPlugin,
}
