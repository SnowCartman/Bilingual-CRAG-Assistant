"use client";

import { useState } from "react";
import { Plus, Trash2 } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ChatSummary } from "@/lib/chat-storage";

interface ChatHistoryProps {
  chats: ChatSummary[];
  activeId: string;
  onPick: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
}

export function ChatHistory({
  chats,
  activeId,
  onPick,
  onNew,
  onDelete,
}: ChatHistoryProps) {
  const [confirmId, setConfirmId] = useState<string | null>(null);

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2">
      <button
        type="button"
        onClick={onNew}
        className="flex items-center justify-center gap-2 rounded-lg border border-accent/40 bg-accent/5 px-3 py-2 text-xs font-medium text-accent transition hover:bg-accent/10"
      >
        <Plus className="h-3.5 w-3.5" />
        Neuer Chat
      </button>

      <h2 className="mt-1 text-[11px] uppercase tracking-wider text-muted">
        Chat-Verlauf
      </h2>

      <div className="-mr-1 flex min-h-0 flex-1 flex-col gap-1 overflow-y-auto pr-1">
        {chats.length === 0 && (
          <p className="text-[11px] italic text-muted">
            noch keine vergangenen Chats
          </p>
        )}
        {chats.map((c) => {
          const isActive = c.id === activeId;
          const isConfirm = c.id === confirmId;
          const when = new Date(c.updatedAt).toLocaleString("de-DE", {
            day: "2-digit",
            month: "2-digit",
            hour: "2-digit",
            minute: "2-digit",
          });
          return (
            <div
              key={c.id}
              className={cn(
                "group relative rounded-md border bg-background p-2 text-left text-xs transition",
                isActive
                  ? "border-accent/40 bg-accent/5"
                  : "border-border hover:border-accent/30",
              )}
            >
              <button
                type="button"
                onClick={() => onPick(c.id)}
                className="block w-full text-left"
              >
                <span
                  className={cn(
                    "block truncate font-medium",
                    isActive ? "text-accent" : "text-foreground",
                  )}
                  title={c.title}
                >
                  {c.title}
                </span>
                <span className="mt-0.5 block text-[10px] text-muted">
                  Chat {c.number} -- {when}
                </span>
              </button>
              {isConfirm ? (
                <div className="mt-1 flex items-center justify-end gap-1">
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      setConfirmId(null);
                    }}
                    className="rounded px-1.5 py-0.5 text-[10px] text-muted hover:text-foreground"
                  >
                    Abbrechen
                  </button>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      setConfirmId(null);
                      onDelete(c.id);
                    }}
                    className="rounded bg-rust px-1.5 py-0.5 text-[10px] text-on-accent hover:bg-rust/90"
                  >
                    Loeschen
                  </button>
                </div>
              ) : (
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    setConfirmId(c.id);
                  }}
                  title="Chat loeschen"
                  className={cn(
                    "absolute right-1 top-1 rounded p-1 text-muted transition-opacity",
                    "opacity-0 group-hover:opacity-100 hover:bg-rust/15 hover:text-rust-text",
                  )}
                  aria-label="Chat loeschen"
                >
                  <Trash2 className="h-3 w-3" />
                </button>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
