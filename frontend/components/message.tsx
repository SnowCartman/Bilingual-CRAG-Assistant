"use client";

import { User as UserIcon } from "lucide-react";
import type { ChatMessage } from "@/lib/types";
import { cn, fixMojibake } from "@/lib/utils";
import { SourceCards } from "@/components/source-card";
import { MetaStrip } from "@/components/meta-strip";
import { ThumbsFeedback } from "@/components/thumbs-feedback";
import { Matryoshka } from "@/components/matryoshka";

interface MessageProps {
  msg: ChatMessage;
  sessionId: string;
}

export function Message({ msg, sessionId }: MessageProps) {
  const isUser = msg.role === "user";
  const cleaned = fixMojibake(msg.content);

  return (
    <div className={cn("flex gap-3", isUser ? "flex-row-reverse" : "flex-row")}>
      <Avatar role={msg.role} />
      <div
        className={cn(
          "flex max-w-[80%] flex-col rounded-2xl border px-4 py-3 text-sm leading-relaxed",
          isUser
            ? "bg-accent text-on-accent border-accent/30 rounded-tr-sm"
            : "bg-surface text-foreground border-border rounded-tl-sm",
        )}
      >
        {!isUser && <MetaStrip meta={msg.meta} />}
        {cleaned ? (
          <p className="whitespace-pre-wrap">{cleaned}</p>
        ) : msg.isStreaming ? (
          <p className="italic text-muted">denke nach...</p>
        ) : msg.meta?.blocked ? (
          <p className="italic text-rust-text">
            Anfrage wurde aus Sicherheitsgr&uuml;nden nicht beantwortet.
          </p>
        ) : null}
        {!isUser && msg.meta?.contexts && (
          <SourceCards contexts={msg.meta.contexts} />
        )}
        {!isUser && !msg.isStreaming && msg.meta?.done && (
          <div className="mt-3 flex items-center justify-between gap-3 border-t border-border pt-2 text-[11px] text-muted">
            <div className="flex items-center gap-3 font-mono">
              <span>{msg.meta.done.latency_ms} ms</span>
              {msg.meta.done.cache_hit && <span className="text-honey">cache</span>}
              {msg.meta.done.cost && (
                <span
                  title="Geschaetzt: Gemini + Voyage zu Listenpreisen Mai 2026. Tavily-Web-Suche und Infrastrukturkosten nicht enthalten."
                  className="cursor-help"
                >
                  ~${msg.meta.done.cost.total_usd.toFixed(4)}
                </span>
              )}
            </div>
            <ThumbsFeedback
              traceId={msg.meta.done.trace_id ?? null}
              sessionId={sessionId}
              messageId={msg.id}
            />
          </div>
        )}
      </div>
    </div>
  );
}

function Avatar({ role }: { role: "user" | "assistant" }) {
  const isUser = role === "user";
  return (
    <div
      className={cn(
        "mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full",
        isUser
          ? "bg-accent/15 text-accent ring-2 ring-accent/30"
          : "bg-surface text-foreground border border-border",
      )}
      aria-label={isUser ? "Du" : "Assistent"}
    >
      {isUser ? (
        <UserIcon className="h-4 w-4" />
      ) : (
        <Matryoshka size={26} className="text-accent" ariaLabel="Assistent" />
      )}
    </div>
  );
}
