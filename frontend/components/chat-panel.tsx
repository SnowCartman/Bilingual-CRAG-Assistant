"use client";

import { useEffect, useRef } from "react";
import { ChatInput } from "@/components/chat-input";
import { Message } from "@/components/message";
import { SuggestionChips } from "@/components/suggestion-chips";
import type { ChatMessage } from "@/lib/types";

interface ChatPanelProps {
  messages: ChatMessage[];
  isStreaming: boolean;
  sendQuery: (text: string) => Promise<void>;
  cancel: () => void;
  sessionId: string;
}

export function ChatPanel({
  messages,
  isStreaming,
  sendQuery,
  cancel,
  sessionId,
}: ChatPanelProps) {
  // Scroll behaviour: SINGLE jump to bottom only when the user has just
  // sent a new query (a new user-role message appears). During the
  // streamed answer the viewport stays anchored where the user left it.
  // This gives "I see what I just asked" without fighting the reader
  // during long EXPLANATORY answers.
  const scrollRef = useRef<HTMLDivElement>(null);
  const prevUserCountRef = useRef(0);
  useEffect(() => {
    const userCount = messages.reduce(
      (n, m) => n + (m.role === "user" ? 1 : 0),
      0,
    );
    if (userCount > prevUserCountRef.current) {
      const el = scrollRef.current;
      if (el) el.scrollTop = el.scrollHeight;
    }
    prevUserCountRef.current = userCount;
  }, [messages]);

  const empty = messages.length === 0;
  return (
    <div className="flex flex-1 flex-col min-h-0">
      <div ref={scrollRef} className="flex-1 overflow-y-auto">
        {empty ? (
          <div className="flex h-full items-center justify-center p-8">
            <SuggestionChips onPick={sendQuery} />
          </div>
        ) : (
          <div className="mx-auto flex max-w-3xl flex-col gap-6 px-4 py-6">
            {messages.map((m) => (
              <Message key={m.id} msg={m} sessionId={sessionId} />
            ))}
          </div>
        )}
      </div>
      <ChatInput
        onSend={sendQuery}
        onCancel={cancel}
        isStreaming={isStreaming}
      />
    </div>
  );
}
