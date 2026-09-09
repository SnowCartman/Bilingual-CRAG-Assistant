"use client";

import { useEffect, useRef, useState } from "react";
import { Trash2 } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

type Status = "idle" | "confirm" | "clearing" | "done" | "error";

/**
 * Header-bar button that wipes every semantic-cache entry via POST
 * /admin/cache/clear. Click opens a small inline confirm popover (analog to
 * the chat-history delete pattern); a second click confirms. Status text in
 * the popover reflects in-flight + success + error states so the user can
 * see what happened without a separate toast system.
 */
export function CacheClearIcon() {
  const [status, setStatus] = useState<Status>("idle");
  const rootRef = useRef<HTMLDivElement | null>(null);

  // Auto-close popover when the user clicks somewhere else or hits Escape.
  useEffect(() => {
    if (status === "idle") return;
    const onDocClick = (e: MouseEvent) => {
      if (!rootRef.current) return;
      if (!rootRef.current.contains(e.target as Node)) setStatus("idle");
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setStatus("idle");
    };
    document.addEventListener("mousedown", onDocClick);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDocClick);
      document.removeEventListener("keydown", onKey);
    };
  }, [status]);

  // Auto-dismiss the success / error popover after a short moment so the
  // header doesn't sit there showing stale state.
  useEffect(() => {
    if (status !== "done" && status !== "error") return;
    const id = setTimeout(() => setStatus("idle"), 1800);
    return () => clearTimeout(id);
  }, [status]);

  const onClear = async () => {
    setStatus("clearing");
    try {
      const res = await fetch(`${API_BASE}/admin/cache/clear`, {
        method: "POST",
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setStatus("done");
    } catch {
      setStatus("error");
    }
  };

  return (
    <div ref={rootRef} className="relative">
      <Button
        variant="ghost"
        size="icon"
        aria-label="Semantischen Cache leeren"
        title="Semantischen Cache leeren"
        onClick={() => setStatus((s) => (s === "idle" ? "confirm" : "idle"))}
        className={cn(
          "text-muted hover:text-rust-text",
          status !== "idle" && "text-rust-text",
        )}
      >
        <Trash2 className="h-4 w-4" />
      </Button>
      {status !== "idle" && (
        <div
          role="dialog"
          aria-label="Cache leeren?"
          className="absolute right-0 top-full z-50 mt-2 w-56 rounded-lg border border-border bg-background p-3 shadow-xl"
          style={{ animation: "fadeIn 150ms ease-out" }}
        >
          {status === "confirm" && (
            <>
              <div className="text-xs font-semibold text-foreground">
                Cache wirklich leeren?
              </div>
              <div className="mt-1 text-[11px] text-muted">
                L&ouml;scht alle gespeicherten Antworten. N&auml;chste Anfrage
                f&auml;hrt die volle Pipeline.
              </div>
              <div className="mt-3 flex justify-end gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setStatus("idle")}
                  className="h-7 px-2 text-xs"
                >
                  Abbrechen
                </Button>
                <Button
                  size="sm"
                  onClick={onClear}
                  className="h-7 bg-rust px-2 text-xs text-on-accent hover:bg-rust"
                >
                  Leeren
                </Button>
              </div>
            </>
          )}
          {status === "clearing" && (
            <div className="text-xs text-muted">Leere Cache &hellip;</div>
          )}
          {status === "done" && (
            <div className="text-xs text-sage-text">Cache geleert.</div>
          )}
          {status === "error" && (
            <div className="text-xs text-rust-text">Fehler beim Leeren.</div>
          )}
        </div>
      )}
    </div>
  );
}
