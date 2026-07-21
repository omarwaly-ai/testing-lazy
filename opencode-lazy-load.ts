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
