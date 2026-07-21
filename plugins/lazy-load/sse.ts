// .opencode/plugins/lazy-load/sse.ts
import { 
  originals, 
  originalSchemas, 
  mcpOriginals, 
  mcpSchemas, 
  turnLoaded, 
  debugLog 
} from "./state"
import { 
  isParsableJson, 
  resolveKnownTool, 
  normalizeToolArguments, 
  isLoadToolName, 
  parseDSMLCalls 
} from "./utils"

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

