"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { API_BASE, newMessageId } from "@/lib/api";
import type {
  AssistantMeta,
  BlockedEvent,
  CacheHitEvent,
  ChatMessage,
  ContextsEvent,
  DoneEvent,
  ErrorEvent,
  GraderEvent,
  RedactedEvent,
  RewriteEvent,
  RouteEvent,
  TokenEvent,
  WebContextsEvent,
  WebSearchStartEvent,
} from "@/lib/types";

interface UseChatStreamArgs {
  sessionId: string;
  useCache: boolean;
  initialMessages?: ChatMessage[];
}

interface UseChatStream {
  messages: ChatMessage[];
  isStreaming: boolean;
  error: string | null;
  sendQuery: (text: string) => Promise<void>;
  cancel: () => void;
  reset: () => void;
}

type SseFrame = { event: string; data: string };

/**
 * Parse a raw SSE buffer into individual frames. sse-starlette emits CRLF
 * line endings; we normalize to LF first so the rest stays simple.
 */
function parseSseBuffer(buf: string): { frames: SseFrame[]; rest: string } {
  const frames: SseFrame[] = [];
  let rest = buf.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  while (true) {
    const sep = rest.indexOf("\n\n");
    if (sep === -1) break;
    const raw = rest.slice(0, sep);
    rest = rest.slice(sep + 2);
    let event = "message";
    const dataLines: string[] = [];
    for (const line of raw.split("\n")) {
      if (!line) continue;
      if (line.startsWith(":")) continue;
      const ix = line.indexOf(":");
      if (ix === -1) continue;
      const field = line.slice(0, ix);
      const value = line.slice(ix + 1).replace(/^ /, "");
      if (field === "event") event = value;
      else if (field === "data") dataLines.push(value);
    }
    frames.push({ event, data: dataLines.join("\n") });
  }
  return { frames, rest };
}

function safeJson<T>(s: string): T | null {
  try {
    return JSON.parse(s) as T;
  } catch {
    return null;
  }
}

/**
 * Typewriter strategy:
 *   - `receivedRef`  : full text received so far, per assistant message id.
 *   - `state.content`: text currently visible in the UI.
 *   - `tickerRef`    : single setInterval that drains chars from received
 *                      into state at a constant rate.
 *
 * Backend tends to deliver many tokens per network chunk (uvicorn +
 * sse-starlette buffer ~8-64 KB). Without smoothing, all of them would
 * pop in one React render. With this loop we reveal at ~120 chars/sec.
 *
 * Cache hits use the same typewriter for visual consistency -- the
 * speed-advantage is already telegraphed via the Cache badge and the
 * sub-500ms latency number in the metadata strip.
 */
const TICK_MS = 22; // ~45 fps
const CHARS_PER_TICK = 6; // ~270 chars/sec, ~60 tokens/sec -- doubled from
                          // the original 3 chars/tick because long
                          // EXPLANATORY answers felt slow to read past.

export function useChatStream({
  sessionId,
  useCache,
  initialMessages,
}: UseChatStreamArgs): UseChatStream {
  // initialMessages is only honoured on first mount of this hook instance.
  // The parent rotates `key={sessionId}` on chat switches, so a fresh hook
  // instance picks up the new initial messages cleanly.
  const [messages, setMessages] = useState<ChatMessage[]>(
    () => initialMessages ?? [],
  );
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const receivedRef = useRef<Map<string, string>>(new Map());
  const donePendingRef = useRef<Map<string, DoneEvent>>(new Map());
  const redactedPendingRef = useRef<Map<string, RedactedEvent>>(new Map());
  const tickerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopTicker = () => {
    if (tickerRef.current != null) {
      clearInterval(tickerRef.current);
      tickerRef.current = null;
    }
  };

  const tick = useCallback(() => {
    // Side-effects are deferred until AFTER setMessages dispatches. React
    // Strict Mode (default in Next 16 dev) calls the updater twice -- if we
    // mutated receivedRef / *PendingRef inside the updater, run 1 would
    // delete the pending entry and run 2 would no longer see it, producing
    // a `next` without meta.done. React keeps run 2's result, so the
    // thumbs-feedback strip (which gates on msg.meta.done) would never
    // render. The pure-updater pattern avoids that entirely.
    const sideEffects: Array<() => void> = [];
    setMessages((prev) =>
      prev.map((msg) => {
        if (msg.role !== "assistant") return msg;
        const received = receivedRef.current.get(msg.id) ?? "";
        let displayedLen = msg.content.length;
        let next: ChatMessage = msg;

        // 1) Reveal more chars if needed.
        if (received.length > displayedLen) {
          const newLen = Math.min(displayedLen + CHARS_PER_TICK, received.length);
          next = { ...msg, content: received.slice(0, newLen) };
          displayedLen = newLen;
        }

        // 2) Apply pending redaction once reveal has caught up enough.
        const redaction = redactedPendingRef.current.get(msg.id);
        if (redaction && displayedLen >= received.length) {
          sideEffects.push(() => {
            receivedRef.current.set(msg.id, redaction.redacted_text);
            redactedPendingRef.current.delete(msg.id);
          });
          next = {
            ...next,
            content: redaction.redacted_text.slice(
              0,
              Math.min(displayedLen, redaction.redacted_text.length),
            ),
            meta: {
              ...(next.meta ?? {}),
              redactions: redaction.redactions,
            },
          };
        }

        // 3) Finalize once reveal is fully caught up and a done event is
        //    pending. This is what flips the typing cursor off + makes the
        //    thumbs strip render (it gates on msg.meta.done).
        const finalReceived = receivedRef.current.get(msg.id) ?? "";
        const done = donePendingRef.current.get(msg.id);
        if (
          done &&
          next.isStreaming &&
          (next.content?.length ?? 0) >= finalReceived.length
        ) {
          sideEffects.push(() => {
            donePendingRef.current.delete(msg.id);
          });
          next = {
            ...next,
            isStreaming: false,
            meta: { ...(next.meta ?? {}), done },
          };
        }

        return next;
      }),
    );
    // Strict Mode runs the updater twice -> sideEffects is pushed twice with
    // identical closures, harmless because delete + set are idempotent.
    for (const fx of sideEffects) fx();
    // NOTE: the ticker NEVER stops itself. It's stopped explicitly in
    // sendQuery's finally (after draining), in reset(), in cancel(), and
    // in cleanup.
  }, []);

  const ensureTicker = useCallback(() => {
    if (tickerRef.current != null) return;
    tickerRef.current = setInterval(tick, TICK_MS);
    // Fire one immediately so very short chunks don't wait a full TICK_MS.
    tick();
  }, [tick]);

  useEffect(() => {
    return () => {
      stopTicker();
      receivedRef.current.clear();
      donePendingRef.current.clear();
      redactedPendingRef.current.clear();
      abortRef.current?.abort();
      abortRef.current = null;
    };
  }, []);

  const cancel = useCallback(() => {
    // Hard stop: abort the fetch AND kill the typewriter in the same tick.
    // A soft cancel (just abortRef.abort()) still let the ticker reveal
    // queued chars from receivedRef during the finally drain -- the user
    // would press stop and watch the text keep typing for another second.
    //
    // For each still-streaming message we synthesise a partial DoneEvent
    // so the meta-strip (latency + thumbs) still renders after the abort.
    // We know latency from createdAt; cost stays undefined (the backend
    // never finished accounting on the aborted stream); trace_id is null
    // so feedback is disabled (no Opik trace to attach a score to). A
    // separate errorMessage shows the "Stream gestoppt." note.
    abortRef.current?.abort();
    abortRef.current = null;
    stopTicker();
    donePendingRef.current.clear();
    redactedPendingRef.current.clear();
    const now = Date.now();
    setMessages((m) =>
      m.map((msg) => {
        if (!msg.isStreaming) return msg;
        const syntheticDone: DoneEvent = {
          latency_ms: now - msg.createdAt,
          cache_hit: false,
          query_type: msg.meta?.route?.query_type ?? null,
          trace_id: null,
        };
        return {
          ...msg,
          isStreaming: false,
          meta: {
            ...(msg.meta ?? {}),
            done: syntheticDone,
            errorMessage: "Stream gestoppt.",
          },
        };
      }),
    );
    setIsStreaming(false);
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    stopTicker();
    receivedRef.current.clear();
    donePendingRef.current.clear();
    redactedPendingRef.current.clear();
    setMessages([]);
    setError(null);
    setIsStreaming(false);
  }, []);

  const sendQuery = useCallback(
    async (text: string) => {
      const trimmed = text.trim();
      if (!trimmed || isStreaming) return;

      const userMsg: ChatMessage = {
        id: newMessageId(),
        role: "user",
        content: trimmed,
        createdAt: Date.now(),
      };
      const assistantId = newMessageId();
      const assistantMsg: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        createdAt: Date.now(),
        isStreaming: true,
        meta: {},
      };
      receivedRef.current.set(assistantId, "");
      setMessages((m) => [...m, userMsg, assistantMsg]);
      setIsStreaming(true);
      setError(null);

      const patchAssistant = (fn: (prev: ChatMessage) => ChatMessage) => {
        setMessages((m) =>
          m.map((msg) => (msg.id === assistantId ? fn(msg) : msg)),
        );
      };
      const patchMeta = (fn: (prev: AssistantMeta) => AssistantMeta) => {
        patchAssistant((prev) => ({ ...prev, meta: fn(prev.meta ?? {}) }));
      };

      const ctrl = new AbortController();
      abortRef.current = ctrl;
      let assistantHasTokens = false;
      let cacheKnown = false;

      try {
        const res = await fetch(`${API_BASE}/query/stream`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: trimmed,
            session_id: sessionId,
            use_cache: useCache,
          }),
          signal: ctrl.signal,
        });
        if (!res.ok || !res.body) {
          throw new Error(`HTTP ${res.status} -- ${await res.text()}`);
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const { frames, rest } = parseSseBuffer(buffer);
          buffer = rest;

          for (const frame of frames) {
            switch (frame.event) {
              case "rewrite": {
                const payload = safeJson<RewriteEvent>(frame.data);
                if (payload) patchMeta((m) => ({ ...m, rewrite: payload }));
                break;
              }
              case "cache_hit": {
                const payload = safeJson<CacheHitEvent>(frame.data);
                if (payload) {
                  cacheKnown = true;
                  patchMeta((m) => ({ ...m, cacheHit: payload }));
                }
                break;
              }
              case "contexts": {
                const payload = safeJson<ContextsEvent>(frame.data);
                if (payload)
                  patchMeta((m) => ({
                    ...m,
                    contexts: payload.contexts,
                    // Carried so the Inspector can name the collection that
                    // actually answered, rather than a hard-coded one.
                    collection: payload.collection,
                  }));
                break;
              }
              case "route": {
                const payload = safeJson<RouteEvent>(frame.data);
                if (payload) patchMeta((m) => ({ ...m, route: payload }));
                break;
              }
              case "grader": {
                const payload = safeJson<GraderEvent>(frame.data);
                if (payload) patchMeta((m) => ({ ...m, grader: payload }));
                break;
              }
              case "web_search_start": {
                const payload = safeJson<WebSearchStartEvent>(frame.data);
                if (payload)
                  patchMeta((m) => ({ ...m, webSearchStart: payload }));
                break;
              }
              case "web_contexts": {
                const payload = safeJson<WebContextsEvent>(frame.data);
                if (payload)
                  patchMeta((m) => ({ ...m, webContexts: payload.contexts }));
                break;
              }
              case "token": {
                const payload = safeJson<TokenEvent>(frame.data);
                if (!payload) break;
                const isFirstToken = !assistantHasTokens;
                const prev = receivedRef.current.get(assistantId) ?? "";
                if (isFirstToken && cacheKnown) {
                  // Cache hit emits the full answer in one token frame.
                  receivedRef.current.set(assistantId, payload.text);
                } else {
                  receivedRef.current.set(assistantId, prev + payload.text);
                }
                assistantHasTokens = true;
                ensureTicker();
                break;
              }
              case "answer_reset": {
                // CRAG web-retry: the backend refused from the corpus, then
                // fetched web docs and is about to re-stream a corrected
                // answer. Clear the refusal text (both the buffer and what's
                // already visible) so the retried tokens replace it cleanly.
                receivedRef.current.set(assistantId, "");
                redactedPendingRef.current.delete(assistantId);
                assistantHasTokens = false;
                patchAssistant((prev) => ({ ...prev, content: "" }));
                break;
              }
              case "blocked": {
                const payload = safeJson<BlockedEvent>(frame.data);
                if (payload) {
                  receivedRef.current.set(assistantId, "");
                  donePendingRef.current.delete(assistantId);
                  patchAssistant((prev) => ({
                    ...prev,
                    isStreaming: false,
                    content: "",
                    meta: { ...(prev.meta ?? {}), blocked: payload },
                  }));
                }
                break;
              }
              case "redacted": {
                const payload = safeJson<RedactedEvent>(frame.data);
                if (payload) {
                  redactedPendingRef.current.set(assistantId, payload);
                  ensureTicker();
                }
                break;
              }
              case "done": {
                const payload = safeJson<DoneEvent>(frame.data);
                if (payload) {
                  donePendingRef.current.set(assistantId, payload);
                  ensureTicker();
                  // Flip the hook-level isStreaming OFF as soon as the
                  // backend says it's done -- the typewriter still keeps
                  // ticking in the background to reveal the remaining
                  // chars at its smooth pace, but the Stop-button no
                  // longer makes sense (there's nothing left to stop)
                  // and the user gets the scroll back. The drain loop in
                  // finally still ensures the typewriter completes; it
                  // just doesn't gate the UI any more.
                  setIsStreaming(false);
                }
                break;
              }
              case "error": {
                const payload = safeJson<ErrorEvent>(frame.data);
                const msg = payload?.message ?? "Unbekannter Stream-Fehler.";
                donePendingRef.current.delete(assistantId);
                patchAssistant((prev) => ({
                  ...prev,
                  isStreaming: false,
                  meta: { ...(prev.meta ?? {}), errorMessage: msg },
                }));
                setError(msg);
                break;
              }
            }
          }
        }
      } catch (e) {
        if ((e as Error).name === "AbortError") {
          patchAssistant((prev) => ({ ...prev, isStreaming: false }));
        } else {
          const msg = (e as Error).message || "Verbindungsfehler.";
          setError(msg);
          patchAssistant((prev) => ({
            ...prev,
            isStreaming: false,
            meta: { ...(prev.meta ?? {}), errorMessage: msg },
          }));
        }
        // No more events will arrive on an aborted/errored stream -- clear
        // the pending refs so the finally drain doesn't poll forever for a
        // done event tick() can never finalize.
        donePendingRef.current.delete(assistantId);
        redactedPendingRef.current.delete(assistantId);
      } finally {
        // Drain remaining reveals before stopping the typewriter. The loop
        // exits the moment both pending queues empty, which is the normal
        // end condition (tick() processed the done event after catching up
        // to the full received text). The cap is a backstop sized to THIS
        // message's received length: enough time to reveal every received
        // char at CHARS_PER_TICK / TICK_MS, plus a 4s safety buffer for the
        // final tick chain. A fixed 6s cap was too short for long
        // EXPLANATORY answers -- they would cut off mid-sentence.
        const pendingChars = receivedRef.current.get(assistantId)?.length ?? 0;
        const maxWaitMs = Math.max(
          4000,
          (pendingChars / CHARS_PER_TICK) * TICK_MS + 4000,
        );
        const start = Date.now();
        while (
          (donePendingRef.current.size > 0 ||
            redactedPendingRef.current.size > 0) &&
          Date.now() - start < maxWaitMs
        ) {
          await new Promise((r) => setTimeout(r, TICK_MS));
        }
        stopTicker();
        abortRef.current = null;
        setIsStreaming(false);
      }
    },
    [ensureTicker, isStreaming, sessionId, useCache],
  );

  return { messages, isStreaming, error, sendQuery, cancel, reset };
}
