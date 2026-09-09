"use client";

import { ArrowRight, Brain, ShieldAlert, Zap } from "lucide-react";
import type { AssistantMeta } from "@/lib/types";
import { cn } from "@/lib/utils";

interface MetaStripProps {
  meta: AssistantMeta | undefined;
}

export function MetaStrip({ meta }: MetaStripProps) {
  if (!meta) return null;
  const items: React.ReactNode[] = [];

  if (meta.rewrite && meta.rewrite.original !== meta.rewrite.rewritten) {
    items.push(
      <MetaLine key="rewrite" Icon={ArrowRight} color="text-claude">
        umformuliert: <span className="font-mono">{meta.rewrite.rewritten}</span>
      </MetaLine>,
    );
  }
  if (meta.cacheHit) {
    items.push(
      <MetaLine key="cache" Icon={Zap} color="text-honey">
        Cache-Treffer (distance {meta.cacheHit.distance.toFixed(3)})
      </MetaLine>,
    );
  }
  if (meta.route) {
    items.push(
      <MetaLine key="route" Icon={Brain} color="text-sage-text">
        Typ: <span className="font-mono">{meta.route.query_type}</span>
      </MetaLine>,
    );
  }
  if (meta.blocked) {
    items.push(
      <MetaLine key="blocked" Icon={ShieldAlert} color="text-rust-text">
        Geblockt durch InputGuard ({meta.blocked.pattern})
      </MetaLine>,
    );
  }
  if (meta.errorMessage) {
    items.push(
      <MetaLine key="error" Icon={ShieldAlert} color="text-rust-text">
        Fehler: {meta.errorMessage}
      </MetaLine>,
    );
  }
  if (items.length === 0) return null;
  return <div className="mb-2 flex flex-col gap-0.5">{items}</div>;
}

function MetaLine({
  Icon,
  color,
  children,
}: {
  Icon: React.ComponentType<{ className?: string }>;
  color: string;
  children: React.ReactNode;
}) {
  return (
    <div
      className={cn(
        "inline-flex items-center gap-1.5 text-[11px] font-mono italic text-muted",
      )}
    >
      <Icon className={cn("h-3 w-3", color)} />
      <span>{children}</span>
    </div>
  );
}
