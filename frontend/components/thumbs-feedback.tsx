"use client";

import { useState } from "react";
import { ThumbsDown, ThumbsUp } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { cn } from "@/lib/utils";
import { FeedbackDialog } from "@/components/feedback-dialog";

interface ThumbsFeedbackProps {
  traceId: string | null;
  sessionId: string;
  messageId: string;
}

type Rating = "up" | "down" | null;

export function ThumbsFeedback({
  traceId,
  sessionId,
  messageId,
}: ThumbsFeedbackProps) {
  const [rating, setRating] = useState<Rating>(null);
  const [pending, setPending] = useState<Rating>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const disabled = !traceId;

  const post = async (chosen: "up" | "down", note: string | null) => {
    setBusy(true);
    setErr(null);
    try {
      const res = await fetch(`${API_BASE}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          trace_id: traceId,
          rating: chosen,
          session_id: sessionId,
          message_id: messageId,
          note: note || null,
        }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setRating(chosen);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  const openDialog = (chosen: "up" | "down") => {
    if (disabled || busy || rating !== null) return;
    setPending(chosen);
  };

  const submitDialog = (note: string | null) => {
    if (!pending) return;
    const chosen = pending;
    setPending(null);
    post(chosen, note);
  };

  const cancelDialog = () => {
    if (!pending) return;
    const chosen = pending;
    setPending(null);
    post(chosen, null);
  };

  const upActive = rating === "up" || pending === "up";
  const downActive = rating === "down" || pending === "down";

  return (
    <div className="flex items-center gap-1.5">
      <button
        onClick={() => openDialog("up")}
        disabled={disabled || busy || rating !== null}
        title={disabled ? "Kein Feedback nach Stop" : "Hilfreich"}
        className={cn(
          "rounded-md p-1 transition-colors",
          "hover:bg-background hover:text-accent",
          upActive ? "text-accent" : "text-muted",
          (disabled || rating === "down") &&
            "opacity-40 cursor-not-allowed hover:bg-transparent hover:text-muted",
        )}
        aria-label="Daumen hoch"
      >
        <ThumbsUp className="h-3.5 w-3.5" />
      </button>
      <button
        onClick={() => openDialog("down")}
        disabled={disabled || busy || rating !== null}
        title={disabled ? "Kein Feedback nach Stop" : "Nicht hilfreich"}
        className={cn(
          "rounded-md p-1 transition-colors",
          "hover:bg-background hover:text-accent",
          downActive ? "text-accent" : "text-muted",
          (disabled || rating === "up") &&
            "opacity-40 cursor-not-allowed hover:bg-transparent hover:text-muted",
        )}
        aria-label="Daumen runter"
      >
        <ThumbsDown className="h-3.5 w-3.5" />
      </button>
      {err && <span className="text-[10px] text-rust-text">{err}</span>}
      {pending && (
        <FeedbackDialog
          rating={pending}
          onSubmit={submitDialog}
          onCancel={cancelDialog}
        />
      )}
    </div>
  );
}
