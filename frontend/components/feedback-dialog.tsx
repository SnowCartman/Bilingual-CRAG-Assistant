"use client";

import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";

const NEGATIVE_REASONS = [
  "Inhaltlich falsch",
  "Nicht hilfreich / off-topic",
  "Hat Anweisungen nicht befolgt",
  "Zu kurz / unvollstaendig",
  "Zu lang / weitschweifig",
  "Anderes",
] as const;

interface FeedbackDialogProps {
  rating: "up" | "down";
  onSubmit: (note: string | null) => void;
  onCancel: () => void;
}

// Modal feedback collector. Both ratings open this dialog after a click;
// Submit sends the note, Skip (or Esc, or backdrop click) closes without note.
// The rating itself is POSTed by the parent in both cases so the click still
// counts as feedback. Negative reasons are folded into the note string as
// "[Grund] kommentar" so the backend FeedbackRequest schema stays untouched.
export function FeedbackDialog({
  rating,
  onSubmit,
  onCancel,
}: FeedbackDialogProps) {
  const [reason, setReason] = useState<string | null>(null);
  const [comment, setComment] = useState("");

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCancel();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel]);

  const isDown = rating === "down";
  const trimmed = comment.trim();

  const handleSubmit = () => {
    let note: string | null = null;
    if (isDown && reason) {
      note = trimmed ? `[${reason}] ${trimmed}` : `[${reason}]`;
    } else if (trimmed) {
      note = trimmed;
    }
    onSubmit(note);
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      onClick={onCancel}
    >
      <div className="fixed inset-0 bg-foreground/40 backdrop-blur-[2px]" />
      <div
        role="dialog"
        aria-modal="true"
        aria-label={
          isDown ? "Feedback -- Daumen runter" : "Feedback -- Daumen hoch"
        }
        className="relative w-full max-w-md rounded-xl border border-border bg-background p-5 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <h3 className="text-sm font-semibold text-foreground">
          {isDown ? "Was ging schief?" : "Was war besonders gut?"}
        </h3>
        <p className="mt-1 text-[12px] text-muted">
          {isDown
            ? "Optional. Hilft beim Tuning von Retrieval und Templates."
            : "Optional. Hilft bestaetigen, welche Chunks und Routings gut funktionieren."}
        </p>

        {isDown && (
          <div className="mt-4 flex flex-col gap-1.5">
            {NEGATIVE_REASONS.map((r) => (
              <label
                key={r}
                className={cn(
                  "flex items-center gap-2 rounded-md border border-border bg-surface px-3 py-2 text-xs transition cursor-pointer",
                  reason === r
                    ? "border-accent bg-accent/5 text-accent"
                    : "text-foreground hover:border-accent/40",
                )}
              >
                <input
                  type="radio"
                  name="reason"
                  value={r}
                  checked={reason === r}
                  onChange={() => setReason(r)}
                  className="h-3 w-3 accent-[var(--accent)]"
                />
                <span>{r}</span>
              </label>
            ))}
          </div>
        )}

        <div className="mt-4">
          <label className="block text-[11px] uppercase tracking-wider text-muted">
            Kommentar (optional)
          </label>
          <textarea
            value={comment}
            onChange={(e) => setComment(e.target.value.slice(0, 280))}
            rows={3}
            placeholder={
              isDown
                ? "Was hat gefehlt oder genervt?"
                : "Was hat besonders geholfen?"
            }
            className="mt-1 w-full resize-none rounded-md border border-border bg-surface p-2 text-sm text-foreground outline-none focus:border-accent"
          />
          <div className="mt-1 flex justify-end text-[10px] text-muted">
            {comment.length}/280
          </div>
        </div>

        <div className="mt-4 flex items-center justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            className="rounded-md px-3 py-1.5 text-sm text-muted transition hover:text-foreground"
          >
            Skip
          </button>
          <button
            type="button"
            onClick={handleSubmit}
            className="rounded-md bg-accent px-3 py-1.5 text-sm text-on-accent transition hover:bg-accent-deep"
          >
            Senden
          </button>
        </div>
      </div>
    </div>
  );
}
