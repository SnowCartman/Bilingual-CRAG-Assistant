"use client";

import { useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  ExternalLink,
  FileText,
  Globe2,
  Zap,
} from "lucide-react";
import type { ContextItem } from "@/lib/types";
import { cn, fixMojibake, formatScore } from "@/lib/utils";

interface SourceCardsProps {
  contexts: ContextItem[];
}

export function SourceCards({ contexts }: SourceCardsProps) {
  const [open, setOpen] = useState(false);
  if (!contexts || contexts.length === 0) return null;

  const head = contexts.slice(0, 3);
  const rest = contexts.slice(3);

  return (
    <div className="mt-3 flex flex-col gap-2">
      <div className="flex flex-wrap items-center gap-2">
        {head.map((c) => (
          <SourcePill key={c.rank} ctx={c} />
        ))}
        {rest.length > 0 && (
          <button
            onClick={() => setOpen((v) => !v)}
            className={cn(
              "inline-flex items-center gap-1 rounded-md border border-border bg-surface px-2 py-1 text-[11px] text-muted",
              "hover:bg-background hover:text-foreground transition-colors",
            )}
          >
            {open ? (
              <ChevronDown className="h-3 w-3" />
            ) : (
              <ChevronRight className="h-3 w-3" />
            )}
            {open ? "weniger" : `+${rest.length} weitere`}
          </button>
        )}
      </div>
      {open && (
        <div className="flex flex-wrap gap-2">
          {rest.map((c) => (
            <SourcePill key={c.rank} ctx={c} />
          ))}
        </div>
      )}
      <details className="mt-1">
        <summary className="cursor-pointer text-[11px] text-muted hover:text-foreground select-none">
          Volltext der Top-3 Belege ansehen
        </summary>
        <div className="mt-2 flex flex-col gap-2">
          {head.map((c) => (
            <SourceDetail key={c.rank} ctx={c} />
          ))}
        </div>
      </details>
    </div>
  );
}

function originStyle(origin: string | undefined) {
  switch (origin) {
    case "tavily":
      // Switched from clay (warm brown) to dusty-rose so
      // the Web pill no longer reads as "almost cache" next to honey.
      // text-dusty-rose-text auto-darkens in light mode for AA contrast.
      return {
        bg: "bg-dusty-rose/30 text-dusty-rose-text border-dusty-rose/50",
        Icon: Globe2,
        label: "Web",
      };
    case "cache":
      return {
        bg: "bg-honey/25 text-[#8a6a2e] border-honey/50 dark:text-honey",
        Icon: Zap,
        label: "Cache",
      };
    default:
      return {
        bg: "bg-sage/25 text-sage-text border-sage/40",
        Icon: FileText,
        label: "DB",
      };
  }
}

function SourcePill({ ctx }: { ctx: ContextItem }) {
  const o = originStyle(ctx.origin);
  const isWeb = ctx.origin === "tavily";
  const file = ctx.source_file
    ? ctx.source_file.replace(/^.*[\\/]/, "").replace(/\.pdf$/i, "")
    : "Quelle";
  const page = ctx.page
    ? ctx.page_end && ctx.page_end !== ctx.page
      ? `S. ${ctx.page}-${ctx.page_end}`
      : `S. ${ctx.page}`
    : null;
  const tail = isWeb && ctx.url ? domainOf(ctx.url) : page;
  const className = cn(
    "inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-[11px]",
    o.bg,
  );
  const tip = `Score ${formatScore(ctx.rerank_score ?? ctx.score)} -- Rang ${ctx.rank}`;
  const inner = (
    <>
      <o.Icon className="h-3 w-3" />
      <span className="font-medium">{o.label}</span>
      <span>{fixMojibake(file)}</span>
      {tail && <span className="opacity-80">{tail}</span>}
      {isWeb && ctx.url && <ExternalLink className="h-2.5 w-2.5 opacity-80" />}
    </>
  );
  if (isWeb && ctx.url) {
    return (
      <a
        href={ctx.url}
        target="_blank"
        rel="noopener noreferrer"
        className={cn(className, "hover:underline")}
        title={tip}
      >
        {inner}
      </a>
    );
  }
  return (
    <span className={className} title={tip}>
      {inner}
    </span>
  );
}

function domainOf(url: string): string {
  try {
    const host = new URL(url).host;
    return host.replace(/^www\./, "");
  } catch {
    return "";
  }
}

function SourceDetail({ ctx }: { ctx: ContextItem }) {
  const isWeb = ctx.origin === "tavily";
  const titleStr = ctx.source_file
    ? fixMojibake(ctx.source_file.replace(/^.*[\\/]/, ""))
    : "";
  return (
    <div className="rounded-md border border-border bg-surface p-3 text-xs leading-relaxed text-foreground">
      <div className="mb-1 flex flex-wrap items-center justify-between gap-1 text-[11px] text-muted">
        <span>
          #{ctx.rank} {titleStr}
          {!isWeb && ctx.page &&
            ` -- S. ${ctx.page}${ctx.page_end && ctx.page_end !== ctx.page ? `-${ctx.page_end}` : ""}`}
        </span>
        {isWeb ? (
          <span className="font-mono flex items-center gap-2">
            tavily {formatScore(ctx.score)}
            {ctx.url && (
              <a
                href={ctx.url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 text-dusty-rose-text hover:underline"
              >
                {domainOf(ctx.url)}
                <ExternalLink className="h-2.5 w-2.5" />
              </a>
            )}
          </span>
        ) : (
          <span className="font-mono">
            rerank {formatScore(ctx.rerank_score)} | hybrid {formatScore(ctx.score)}
            {ctx.grader_score != null && ` | crag ${formatScore(ctx.grader_score)}`}
          </span>
        )}
      </div>
      <p className="whitespace-pre-wrap">{fixMojibake(ctx.text)}</p>
    </div>
  );
}
