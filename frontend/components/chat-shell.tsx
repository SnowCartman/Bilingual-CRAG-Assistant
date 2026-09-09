"use client";

import { useEffect, useRef } from "react";
import { ChatPanel } from "@/components/chat-panel";
import { PipelineInspector } from "@/components/pipeline-inspector";
import { useChatStream } from "@/hooks/use-chat-stream";
import type { ChatMessage } from "@/lib/types";

interface ChatShellProps {
  sessionId: string;
  useCache: boolean;
  view: "chat" | "pipeline";
  initialMessages?: ChatMessage[];
  onPersist?: (messages: ChatMessage[]) => void;
}

/**
 * Owns the single useChatStream instance so Chat and Pipeline views share
 * the same message list. Without this, switching views would unmount the
 * hook and drop the last run -- the user would land in an empty inspector.
 *
 * The outer component sets `key={sessionId}`, which remounts ChatShell on
 * "Neuer Chat" / pick-from-history and resets the hook cleanly. No manual
 * reset needed here.
 *
 * onPersist fires once per completed message -- the trigger is the last
 * assistant message flipping its OWN `isStreaming` to false (the typewriter
 * sets that in tick() after revealing every received char). Keying off the
 * hook-level `isStreaming` would persist on the backend's `done` event,
 * while the typewriter is still revealing the last few seconds of text --
 * the saved chat would then re-open with a half-typed answer.
 */
export function ChatShell({
  sessionId,
  useCache,
  view,
  initialMessages,
  onPersist,
}: ChatShellProps) {
  const { messages, isStreaming, sendQuery, cancel } = useChatStream({
    sessionId,
    useCache,
    initialMessages,
  });

  const persistedRef = useRef<Set<string>>(new Set());
  useEffect(() => {
    if (messages.length === 0) return;
    const last = messages[messages.length - 1];
    if (last.role !== "assistant") return;
    const isDone = last.isStreaming === false && last.meta?.done != null;
    if (isDone && !persistedRef.current.has(last.id)) {
      persistedRef.current.add(last.id);
      onPersist?.(messages);
    }
  }, [messages, onPersist]);

  if (view === "pipeline") {
    return (
      <PipelineInspector
        messages={messages}
        isStreaming={isStreaming}
        sendQuery={sendQuery}
        cancel={cancel}
        sessionId={sessionId}
      />
    );
  }
  return (
    <ChatPanel
      messages={messages}
      isStreaming={isStreaming}
      sendQuery={sendQuery}
      cancel={cancel}
      sessionId={sessionId}
    />
  );
}
