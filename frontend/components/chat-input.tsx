"use client";

import { useState, KeyboardEvent } from "react";
import { Send, Square } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface ChatInputProps {
  onSend: (text: string) => void;
  onCancel?: () => void;
  isStreaming: boolean;
  placeholder?: string;
}

export function ChatInput({
  onSend,
  onCancel,
  isStreaming,
  placeholder,
}: ChatInputProps) {
  const [value, setValue] = useState("");

  const submit = () => {
    if (!value.trim() || isStreaming) return;
    onSend(value);
    setValue("");
  };

  const handleClick = () => {
    if (isStreaming) {
      onCancel?.();
    } else {
      submit();
    }
  };

  const onKey = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="border-t border-border bg-surface p-4">
      <div
        className={cn(
          "mx-auto flex max-w-3xl items-end gap-2 rounded-2xl border border-border bg-background p-2",
          "focus-within:border-accent focus-within:ring-1 focus-within:ring-accent/30",
        )}
      >
        <textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={onKey}
          placeholder={
            placeholder ?? "Frag etwas auf Deutsch oder Russisch..."
          }
          rows={1}
          className={cn(
            "flex-1 resize-none bg-transparent px-2 py-2 text-sm text-foreground placeholder:text-muted",
            "focus:outline-none",
            "max-h-40",
          )}
          style={{ minHeight: "2.25rem" }}
        />
        <Button
          onClick={handleClick}
          disabled={!isStreaming && !value.trim()}
          size="icon"
          aria-label={isStreaming ? "Stoppen" : "Senden"}
          title={isStreaming ? "Stream stoppen" : "Senden"}
        >
          {isStreaming ? (
            <Square className="h-4 w-4 fill-current" />
          ) : (
            <Send className="h-4 w-4" />
          )}
        </Button>
      </div>
      <p className="mx-auto mt-2 max-w-3xl text-[10px] text-muted">
        Enter zum Senden, Shift+Enter fuer neue Zeile. Quellenangaben in eckigen Klammern.
      </p>
    </div>
  );
}
