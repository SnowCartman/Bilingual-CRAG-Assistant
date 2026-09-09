"use client";

import { useEffect } from "react";
import { X } from "lucide-react";
import { cn } from "@/lib/utils";
import { ChatHistory } from "@/components/chat-history";
import type { ChatSummary } from "@/lib/chat-storage";

interface SidebarProps {
  chats: ChatSummary[];
  activeId: string;
  open: boolean;
  onClose: () => void;
  onPickChat: (id: string) => void;
  onNewChat: () => void;
  onDeleteChat: (id: string) => void;
}

export function Sidebar({
  chats,
  activeId,
  open,
  onClose,
  onPickChat,
  onNewChat,
  onDeleteChat,
}: SidebarProps) {
  // Esc closes the mobile drawer. Desktop ignores because sidebar is static.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  // Wrap each callback so picking / new-chat / delete on mobile also closes
  // the drawer; on desktop onClose is a no-op so this is safe.
  const handlePick = (id: string) => {
    onPickChat(id);
    onClose();
  };
  const handleNew = () => {
    onNewChat();
    onClose();
  };
  const handleDelete = (id: string) => {
    onDeleteChat(id);
  };

  return (
    <>
      {/* Backdrop -- mobile only, only while drawer is open */}
      <div
        onClick={onClose}
        className={cn(
          "fixed inset-0 z-30 bg-foreground/40 backdrop-blur-[1px] transition-opacity md:hidden",
          open ? "opacity-100" : "pointer-events-none opacity-0",
        )}
        aria-hidden="true"
      />

      <aside
        className={cn(
          // Mobile: fixed off-canvas drawer with slide-in transform
          "fixed inset-y-0 left-0 z-40 flex w-72 flex-col border-r border-border bg-surface p-4 transition-transform",
          open ? "translate-x-0" : "-translate-x-full",
          // Desktop: revert to static panel, always visible, narrower
          "md:relative md:z-auto md:w-64 md:translate-x-0",
        )}
        aria-label="Chat-Verlauf"
      >
        <button
          type="button"
          onClick={onClose}
          aria-label="Sidebar schliessen"
          title="Schliessen"
          className="absolute right-2 top-2 rounded-md p-1.5 text-muted transition hover:bg-background hover:text-foreground md:hidden"
        >
          <X className="h-4 w-4" />
        </button>

        <ChatHistory
          chats={chats}
          activeId={activeId}
          onPick={handlePick}
          onNew={handleNew}
          onDelete={handleDelete}
        />

        <div className="mt-3 flex flex-col gap-1 text-[11px] text-muted">
          <Legend color="bg-sage" label="DB -- Voyage Hybrid + Rerank" />
          <Legend color="bg-dusty-rose" label="Web -- Tavily Fallback" />
          <Legend color="bg-honey" label="Cache -- Redis HNSW" />
          <Legend color="bg-rust" label="Blocked / Redacted" />
        </div>
      </aside>
    </>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <div className="flex items-center gap-2">
      <span className={cn("h-2 w-2 rounded-full", color)} />
      <span>{label}</span>
    </div>
  );
}
