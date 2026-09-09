"use client";

import { useState } from "react";
import {
  Brain,
  CheckCircle2,
  Circle,
  ExternalLink,
  Globe2,
  Loader2,
  Scale,
  Search,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Wand2,
  Workflow,
  X,
} from "lucide-react";
import { ChatInput } from "@/components/chat-input";
import { PipelineDiagramDialog } from "@/components/pipeline-diagram";
import { SourceCards } from "@/components/source-card";
import type { ChatMessage } from "@/lib/types";
import { cn, fixMojibake, formatScore } from "@/lib/utils";

interface PipelineInspectorProps {
  messages: ChatMessage[];
  isStreaming: boolean;
  sendQuery: (text: string) => Promise<void>;
  cancel: () => void;
  sessionId: string;
}

type CardStatus = "idle" | "running" | "complete" | "skipped" | "blocked";

interface CardSpec {
  id: CardId;
  label: string;
  Icon: React.ComponentType<{ className?: string }>;
  status: CardStatus;
  metric: string;
  hint?: string;
}

type CardId =
  | "retriever"
  | "grader"
  | "decomposer"
  | "websearch"
  | "refiner"
  | "generator";

/**
 * Live read-only view of the pipeline for the latest assistant message.
 * Consumes the same useChatStream the chat view does -- we just render the
 * events as a 3x2 agent grid instead of a chat bubble. No backend changes;
 * all data is already in msg.meta (rewrite / cacheHit / contexts / route /
 * blocked / redactions / done).
 */
export function PipelineInspector({
  messages,
  isStreaming,
  sendQuery,
  cancel,
  sessionId: _sessionId,
}: PipelineInspectorProps) {
  const [selectedCard, setSelectedCard] = useState<CardId | null>(null);
  const [diagramOpen, setDiagramOpen] = useState(false);

  const lastAssistant = [...messages]
    .reverse()
    .find((m) => m.role === "assistant");
  const lastUser = [...messages].reverse().find((m) => m.role === "user");
  const empty = !lastAssistant;

  const cards = empty ? defaultIdleCards() : deriveCards(lastAssistant);
  const verdicts = deriveSecurityVerdicts(lastAssistant);
  const selected = selectedCard
    ? cards.find((c) => c.id === selectedCard) ?? null
    : null;

  return (
    <div className="flex flex-1 flex-col min-h-0">
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto flex max-w-5xl flex-col gap-5 px-4 py-6">
          <PipelineIntro
            showHelp={empty}
            onOpenDiagram={() => setDiagramOpen(true)}
          />
          {!empty && lastUser && <QueryEcho text={lastUser.content} />}
          <SecurityStrip verdicts={verdicts} />
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {cards.map((card) => (
              <PipelineCard
                key={card.id}
                card={card}
                onClick={() => setSelectedCard(card.id)}
              />
            ))}
          </div>
          <PipelineLegend />
        </div>
      </div>
      <ChatInput
        onSend={sendQuery}
        onCancel={cancel}
        isStreaming={isStreaming}
      />
      {selected && (
        <CardDrawer
          card={selected}
          msg={lastAssistant}
          onClose={() => setSelectedCard(null)}
        />
      )}
      {diagramOpen && (
        <PipelineDiagramDialog onClose={() => setDiagramOpen(false)} />
      )}
    </div>
  );
}

// ---------- card derivation ------------------------------------------------

// Card order = REAL execution order in rag_pipeline.query_stream:
// Query Rewrite -> Retriever (hybrid retrieve + rerank) -> CRAG Grader
// -> Web Search (Tavily fallback) -> Query Router -> Generator. The
// router runs AFTER the web branch, right before the generator
// (rag_pipeline.py `_classify` call site). Card labels match the
// architecture diagram (pipeline-diagram.tsx) and the backend module
// names, not the CRAG-paper stage names (Decomposer / Refiner).
function defaultIdleCards(): CardSpec[] {
  const labels: Array<[CardId, string, React.ComponentType<{ className?: string }>]> = [
    ["decomposer", "Query Rewrite", Wand2],
    ["retriever", "Retriever", Search],
    ["grader", "CRAG Grader", Scale],
    ["websearch", "Web Search", Globe2],
    ["refiner", "Query Router", Brain],
    ["generator", "Generator", Sparkles],
  ];
  return labels.map(([id, label, Icon]) => ({
    id,
    label,
    Icon,
    status: "idle" as CardStatus,
    metric: "wartet",
  }));
}

function deriveCards(m: ChatMessage): CardSpec[] {
  const meta = m.meta ?? {};
  const blocked = !!meta.blocked;
  const cacheHit = !!meta.cacheHit;
  const streaming = !!m.isStreaming;
  const done = !!meta.done;
  const contexts = meta.contexts ?? [];
  const tavilyHits = contexts.filter((c) => c.origin === "tavily").length;

  // Retriever
  const retriever: CardSpec = (() => {
    if (blocked) return base("retriever", "Retriever", Search, "skipped", "Input blockiert");
    if (cacheHit) return base("retriever", "Retriever", Search, "skipped", "Cache umgangen Retrieval");
    if (contexts.length > 0) {
      const topScore = contexts[0]?.rerank_score ?? contexts[0]?.score ?? null;
      return base(
        "retriever",
        "Retriever",
        Search,
        "complete",
        `${contexts.length} Chunks  -  Top ${formatScore(topScore)}`,
      );
    }
    if (streaming) return base("retriever", "Retriever", Search, "running", "Voyage + Qdrant Hybrid");
    return base("retriever", "Retriever", Search, "idle", "wartet");
  })();

  // CRAG Grader -- relevance-scores each retrieved chunk via
  // gemini-2.5-flash. Emits the 'grader' SSE event on cache miss with
  // non-empty retrieval. The DocumentGuard lives in the
  // Security Strip above, not as a card.
  const graderEv = meta.grader;
  const grader: CardSpec = (() => {
    if (blocked) return base("grader", "CRAG Grader", Scale, "skipped", "Input blockiert");
    if (cacheHit) return base("grader", "CRAG Grader", Scale, "skipped", "Cache-Pfad");
    if (graderEv) {
      const verdict = graderEv.needs_web ? "needs web" : "DB sufficient";
      return base(
        "grader",
        "CRAG Grader",
        Scale,
        "complete",
        `avg ${graderEv.avg_score.toFixed(2)}  -  ${graderEv.kept}/${graderEv.kept + graderEv.filtered} kept  -  ${verdict}`,
      );
    }
    if (streaming) return base("grader", "CRAG Grader", Scale, "running", "scoring relevance");
    if (done) return base("grader", "CRAG Grader", Scale, "skipped", "no retrieval");
    return base("grader", "CRAG Grader", Scale, "idle", "wartet");
  })();

  // Query Rewrite (conditional follow-up rewrite)
  const decomposer: CardSpec = (() => {
    if (blocked) return base("decomposer", "Query Rewrite", Wand2, "skipped", "Input blockiert");
    if (meta.rewrite && meta.rewrite.original !== meta.rewrite.rewritten)
      return base("decomposer", "Query Rewrite", Wand2, "complete", "umformuliert");
    if (done || cacheHit)
      return base("decomposer", "Query Rewrite", Wand2, "skipped", "keine Umformulierung noetig");
    if (streaming) return base("decomposer", "Query Rewrite", Wand2, "running", "prueft Kontext");
    return base("decomposer", "Query Rewrite", Wand2, "idle", "wartet");
  })();

  // Web Search -- Tavily fallback fires when CRAG verdict says
  // needs_web. Three live states: running (web_search_start arrived but
  // no web_contexts yet), complete (web_contexts populated), skipped
  // (grader was happy with DB chunks, or Tavily disabled, or cache hit).
  const webDocs = meta.webContexts ?? [];
  const websearch: CardSpec = (() => {
    if (blocked) return base("websearch", "Web Search", Globe2, "skipped", "Input blockiert");
    if (cacheHit) return base("websearch", "Web Search", Globe2, "skipped", "Cache-Pfad");
    if (webDocs.length > 0)
      return base("websearch", "Web Search", Globe2, "complete", `${webDocs.length} Tavily-Treffer`);
    if (meta.webSearchStart)
      return base("websearch", "Web Search", Globe2, "running", "Tavily-Suche laeuft");
    if (tavilyHits > 0)
      return base("websearch", "Web Search", Globe2, "complete", `${tavilyHits} Web-Treffer`);
    if (graderEv && !graderEv.needs_web)
      return base("websearch", "Web Search", Globe2, "skipped", "Korpus ausreichend");
    if (done) return base("websearch", "Web Search", Globe2, "skipped", "nicht ausgeloest");
    return base("websearch", "Web Search", Globe2, "idle", "wartet");
  })();

  // Query Router (template classifier)
  const refiner: CardSpec = (() => {
    if (blocked) return base("refiner", "Query Router", Brain, "skipped", "Input blockiert");
    if (meta.route?.query_type)
      return base("refiner", "Query Router", Brain, "complete", meta.route.query_type);
    if (cacheHit) return base("refiner", "Query Router", Brain, "skipped", "Cache-Pfad");
    if (streaming) return base("refiner", "Query Router", Brain, "running", "klassifiziert");
    return base("refiner", "Query Router", Brain, "idle", "wartet");
  })();

  // Generator (Gemini token stream)
  const generator: CardSpec = (() => {
    if (blocked) return base("generator", "Generator", Sparkles, "skipped", "Input blockiert");
    if (done) {
      const lat = meta.done?.latency_ms;
      const cost = meta.done?.cost?.total_usd;
      const parts = [];
      if (lat != null) parts.push(`${lat} ms`);
      if (cost != null) parts.push(`~$${cost.toFixed(4)}`);
      return base("generator", "Generator", Sparkles, "complete", parts.join("  -  ") || "fertig");
    }
    if (streaming) {
      const chars = m.content.length;
      return base("generator", "Generator", Sparkles, "running", `${chars} Zeichen ...`);
    }
    return base("generator", "Generator", Sparkles, "idle", "wartet");
  })();

  // Same execution order as defaultIdleCards (rewrite -> retrieve ->
  // grade -> web -> route -> generate).
  return [decomposer, retriever, grader, websearch, refiner, generator];
}

function base(
  id: CardId,
  label: string,
  Icon: React.ComponentType<{ className?: string }>,
  status: CardStatus,
  metric: string,
): CardSpec {
  return { id, label, Icon, status, metric };
}

interface SecurityVerdict {
  guard: "Input Guard" | "Document Guard" | "Output Validator";
  verdict: "ok" | "blocked" | "redacted" | "skipped" | "idle";
  detail: string;
}

function deriveSecurityVerdicts(
  m: ChatMessage | undefined,
): SecurityVerdict[] {
  if (!m) {
    return [
      { guard: "Input Guard", verdict: "idle", detail: "wartet" },
      { guard: "Document Guard", verdict: "idle", detail: "wartet" },
      { guard: "Output Validator", verdict: "idle", detail: "wartet" },
    ];
  }
  const meta = m.meta ?? {};
  const inputV: SecurityVerdict = meta.blocked
    ? {
        guard: "Input Guard",
        verdict: "blocked",
        detail: meta.blocked.pattern,
      }
    : { guard: "Input Guard", verdict: "ok", detail: "clean" };

  const docV: SecurityVerdict = meta.blocked
    ? { guard: "Document Guard", verdict: "skipped", detail: "Input blockiert" }
    : meta.cacheHit
      ? { guard: "Document Guard", verdict: "skipped", detail: "Cache-Pfad, kein Retrieval" }
      : (meta.contexts?.length ?? 0) > 0
        ? { guard: "Document Guard", verdict: "ok", detail: "keine Drops" }
        : { guard: "Document Guard", verdict: "idle", detail: "wartet" };

  const outV: SecurityVerdict = (() => {
    if (meta.blocked)
      return { guard: "Output Validator", verdict: "skipped", detail: "kein Output erzeugt" };
    const r = meta.redactions;
    if (r && r.length > 0) {
      const total = r.reduce((acc, x) => acc + x.count, 0);
      return {
        guard: "Output Validator",
        verdict: "redacted",
        detail: `${total}x ${r.map((x) => x.kind).join(", ")}`,
      };
    }
    if (meta.done)
      return { guard: "Output Validator", verdict: "ok", detail: "keine PII" };
    return { guard: "Output Validator", verdict: "idle", detail: "wartet" };
  })();

  return [inputV, docV, outV];
}

// ---------- presentational pieces -----------------------------------------

function SecurityStrip({ verdicts }: { verdicts: SecurityVerdict[] }) {
  return (
    <div className="flex flex-wrap gap-2 rounded-lg border border-border bg-surface p-2">
      {verdicts.map((v) => (
        <VerdictPill key={v.guard} v={v} />
      ))}
    </div>
  );
}

function VerdictPill({ v }: { v: SecurityVerdict }) {
  const style =
    v.verdict === "blocked"
      ? "bg-rust/25 text-rust-text border-rust/45"
      : v.verdict === "redacted"
        ? "bg-honey/25 text-[#8a6a2e] border-honey/50 dark:text-honey"
        : v.verdict === "ok"
          ? "bg-sage/25 text-sage-text border-sage/40"
          : v.verdict === "skipped"
            ? "bg-background text-muted border-dashed border-muted/40 opacity-80"
            : "bg-background text-muted border-border";
  const Icon =
    v.verdict === "blocked"
      ? ShieldAlert
      : v.verdict === "redacted"
        ? ShieldAlert
        : v.verdict === "skipped"
          ? Circle
          : ShieldCheck;
  const label =
    v.verdict === "skipped"
      ? "skipped"
      : v.verdict === "ok"
        ? "ok"
        : v.verdict === "blocked"
          ? "blocked"
          : v.verdict === "redacted"
            ? "redacted"
            : null;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-md border px-2 py-1 text-[11px]",
        style,
      )}
    >
      <Icon className="h-3 w-3" />
      <span className="font-medium">{v.guard}</span>
      {label && (
        <span className="rounded bg-foreground/5 px-1 text-[10px] uppercase tracking-wider">
          {label}
        </span>
      )}
      <span>{v.detail}</span>
    </span>
  );
}

function QueryEcho({ text }: { text: string }) {
  return (
    <div className="rounded-lg border border-border bg-surface px-4 py-3 text-sm">
      <div className="mb-1 text-[10px] uppercase tracking-wider text-muted">
        Anfrage
      </div>
      <div className="font-mono text-foreground">{fixMojibake(text)}</div>
    </div>
  );
}

function PipelineCard({
  card,
  onClick,
}: {
  card: CardSpec;
  onClick: () => void;
}) {
  const { Icon } = card;
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex flex-col gap-2 rounded-lg border border-border bg-surface p-3 text-left",
        "transition-colors hover:bg-background hover:border-accent/40",
        "focus:outline-none focus:ring-2 focus:ring-accent/40",
      )}
    >
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2">
          <Icon className="h-4 w-4 text-muted" />
          <span className="text-sm font-medium text-foreground">{card.label}</span>
        </div>
        <StatusPill status={card.status} />
      </div>
      <div className="text-[11px] font-mono text-muted truncate">
        {card.metric}
      </div>
    </button>
  );
}

function StatusPill({ status }: { status: CardStatus }) {
  const map: Record<CardStatus, { label: string; cls: string; Icon: React.ComponentType<{ className?: string }> }> = {
    idle: {
      label: "idle",
      cls: "bg-background text-muted border-border",
      Icon: Circle,
    },
    running: {
      label: "running",
      cls: "bg-accent/15 text-accent border-accent/30",
      Icon: Loader2,
    },
    complete: {
      label: "complete",
      cls: "bg-sage/25 text-sage-text border-sage/40",
      Icon: CheckCircle2,
    },
    skipped: {
      label: "skipped",
      cls: "bg-background text-muted border-border opacity-70",
      Icon: Circle,
    },
    blocked: {
      label: "blocked",
      cls: "bg-rust/25 text-rust-text border-rust/45",
      Icon: ShieldAlert,
    },
  };
  const cfg = map[status];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[10px] uppercase tracking-wider",
        cfg.cls,
      )}
    >
      <cfg.Icon
        className={cn(
          "h-2.5 w-2.5",
          status === "running" && "animate-spin",
        )}
      />
      {cfg.label}
    </span>
  );
}

function PipelineIntro({
  showHelp,
  onOpenDiagram,
}: {
  showHelp: boolean;
  onOpenDiagram: () => void;
}) {
  // Header row for the Inspector view -- always rendered so the Overview
  // button (animated architecture diagram) stays reachable after a query
  // has run. The explanatory paragraph only shows in the idle empty state;
  // the chat view next door handles "type your first query".
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between gap-2">
        <h2 className="text-base font-semibold text-foreground">CRAG Pipeline</h2>
        <button
          type="button"
          onClick={onOpenDiagram}
          className={cn(
            "inline-flex items-center gap-1.5 rounded-md border border-accent/40 bg-accent/10",
            "px-2.5 py-1 text-xs font-medium text-accent transition-colors",
            "hover:bg-accent/20 focus:outline-none focus:ring-2 focus:ring-accent/40",
          )}
        >
          <Workflow className="h-3.5 w-3.5" />
          Overview
        </button>
      </div>
      {showHelp && (
        <p className="text-sm text-muted leading-relaxed">
          Sobald eine Frage l&auml;uft, zeigt der Inspector hier in Echtzeit,
          wie Query Rewrite, Retriever, CRAG Grader, Web Search, Query Router
          und Generator die Anfrage verarbeiten -- begleitet von den drei
          Security-Guards oben. Der Overview-Button &ouml;ffnet das
          Architektur-Diagramm der Pipeline.
        </p>
      )}
    </div>
  );
}

function PipelineLegend() {
  // HTML entities (&XXXX;) render real umlauts in the browser while the
  // source file stays pure ASCII (ASCII gate). JSX-text-body decodes
  // HTML entities; JS-string-literals do NOT, so card-metric strings still
  // use ASCII translit ("prueft" etc).
  return (
    <div className="text-[11px] text-muted leading-relaxed">
      Die Karten folgen der Ausf&uuml;hrungs-Reihenfolge der Pipeline.
      Klick auf eine Karte &ouml;ffnet die Details. CRAG Grader scort jeden
      retrievten Chunk, und wenn der h&ouml;chste Score unter Threshold
      f&auml;llt, &uuml;bernimmt Tavily die Web-Suche.
    </div>
  );
}

// ---------- drawer ---------------------------------------------------------

function CardDrawer({
  card,
  msg,
  onClose,
}: {
  card: CardSpec;
  msg: ChatMessage | undefined;
  onClose: () => void;
}) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-end justify-center bg-black/40 backdrop-blur-sm sm:items-center"
      onClick={onClose}
    >
      <div
        className="m-4 max-h-[85vh] w-full max-w-2xl overflow-y-auto rounded-lg border border-border bg-surface shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-border px-4 py-3">
          <div className="flex items-center gap-2">
            <card.Icon className="h-4 w-4 text-muted" />
            <h2 className="text-sm font-semibold text-foreground">{card.label}</h2>
            <StatusPill status={card.status} />
          </div>
          <button
            onClick={onClose}
            className="rounded-md p-1 text-muted hover:bg-background hover:text-foreground"
            aria-label="Schliessen"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="px-4 py-4">
          <DrawerBody card={card} msg={msg} />
        </div>
      </div>
    </div>
  );
}

function DrawerBody({ card, msg }: { card: CardSpec; msg: ChatMessage | undefined }) {
  // When no query has run yet (idle inspector), msg is undefined --
  // we still render the static ConfigBox per card so a reviewer
  // can read the architecture before any live data exists.
  const meta = msg?.meta ?? {};
  switch (card.id) {
    case "retriever": {
      const contexts = meta.contexts ?? [];
      return (
        <div className="flex flex-col gap-4">
          <ConfigBox
            title="Configuration"
            rows={[
              ["Chunking", "logical recursive, 2000 chars / 200 overlap"],
              ["Dense Embedder", "voyage-multilingual-2 (1024-dim)"],
              ["Sparse Embedder", "BM25 via fastembed (Qdrant/bm25, IDF)"],
              [
                "Vector Store",
                `Qdrant  -  ${meta.collection ?? "collection from QDRANT_COLLECTION"}`,
              ],
              ["Fusion", "Reciprocal Rank Fusion, prefetch=100"],
              ["Reranker", "voyage rerank-2.5, RRF Top-100 -> Top-10"],
            ]}
          />
          {contexts.length === 0 ? (
            <p className="text-sm text-muted">
              Keine Kontexte aktiv f&uuml;r diese Anfrage -- Cache-Hit oder
              Input-Block.
            </p>
          ) : (
            <div>
              <div className="mb-2 text-[11px] uppercase tracking-wider text-muted">
                {contexts.length} Belege nach Rerank
              </div>
              <SourceCards contexts={contexts} />
            </div>
          )}
        </div>
      );
    }
    case "grader": {
      const g = meta.grader;
      const skipped = meta.cacheHit || meta.blocked;
      return (
        <div className="flex flex-col gap-4 text-sm">
          <p>
            Der CRAG Grader (Corrective RAG) bewertet jeden retrievten
            Chunk auf Relevanz zur Anfrage. Wenn der h&ouml;chste
            Chunk-Score (max) unter den Threshold f&auml;llt,
            f&auml;llt die Pipeline auf Tavily-Web-Suche zur&uuml;ck.
          </p>
          <ConfigBox
            title="Configuration"
            rows={[
              ["Model", "gemini-2.5-flash, JSON-mode"],
              ["Web fallback", "max_score < 0.45 (kalibriert mit temperature=0)"],
              ["Per-chunk filter", "score < 0.50 (only in Tavily-mix mode)"],
              ["Skip conditions", "cache hit, input block, empty corpus"],
            ]}
          />
          {skipped ? (
            <p className="font-mono text-xs text-muted">
              &Uuml;bersprungen -- kein Retrieval auf diesem Pfad.
            </p>
          ) : g ? (
            <div className="grid grid-cols-2 gap-2 text-xs text-muted">
              <Metric label="Avg Score">{g.avg_score.toFixed(3)}</Metric>
              <Metric label="Threshold (auf max)">{g.threshold.toFixed(2)}</Metric>
              <Metric label="Kept">{g.kept}</Metric>
              <Metric label="Filtered">{g.filtered}</Metric>
              <div className="col-span-2 rounded-md border border-border bg-background px-2 py-1.5">
                <div className="text-[10px] uppercase tracking-wider text-muted">
                  Verdict
                </div>
                <div className={cn(
                  "font-mono",
                  g.needs_web ? "text-dusty-rose-text" : "text-sage-text",
                )}>
                  {g.needs_web
                    ? "needs_web=true -- Tavily fallback active"
                    : "DB context sufficient"}
                </div>
              </div>
            </div>
          ) : (
            <p className="font-mono text-xs text-muted">
              Noch kein Verdict -- Retriever l&auml;uft.
            </p>
          )}
        </div>
      );
    }
    case "decomposer": {
      if (meta.rewrite && meta.rewrite.original !== meta.rewrite.rewritten)
        return (
          <div className="flex flex-col gap-3 text-sm">
            <div>
              <div className="text-[11px] uppercase tracking-wider text-muted">
                Original
              </div>
              <div className="font-mono">{fixMojibake(meta.rewrite.original)}</div>
            </div>
            <div>
              <div className="text-[11px] uppercase tracking-wider text-muted">
                Rewritten
              </div>
              <div className="font-mono text-accent">
                {fixMojibake(meta.rewrite.rewritten)}
              </div>
            </div>
            <p className="text-xs text-muted">
              Conditional Rewrite -- gemini-2.5-flash expandiert
              Pronouns und faltet History ein, nur wenn die Anfrage
              kontextabh&auml;ngig ist.
            </p>
          </div>
        );
      return (
        <p className="text-sm text-muted">
          Keine Umformulierung n&ouml;tig -- die Anfrage stand alleine bereits
          retrievable.
        </p>
      );
    }
    case "websearch": {
      const webDocs = meta.webContexts ?? [];
      const startedQuery = meta.webSearchStart?.query;
      const usedWeb = meta.done?.used_web_search ?? webDocs.length > 0;
      return (
        <div className="flex flex-col gap-4 text-sm">
          <p>
            Web Search ist der CRAG-Fallback. Wenn der
            Grader sagt
            <span className="font-mono"> needs_web=true</span>, ruft die
            Pipeline Tavily auf und mischt die Web-Treffer (origin
            <span className="font-mono"> &quot;tavily&quot;</span>) in die
            Kontexte.
          </p>
          <ConfigBox
            title="Configuration"
            rows={[
              ["Provider", "Tavily REST (api.tavily.com/search)"],
              ["Topic", "general"],
              ["Search Depth", "basic (5x cheaper than advanced)"],
              ["Max Results", "3"],
              ["Safety", "DocumentGuard scannt auch Web-Docs"],
            ]}
          />
          {usedWeb && webDocs.length > 0 ? (
            <div className="flex flex-col gap-2">
              <div className="text-[11px] uppercase tracking-wider text-muted">
                {webDocs.length} Treffer von Tavily
                {startedQuery && (
                  <span className="ml-2 font-mono normal-case opacity-80">
                    f&uuml;r &quot;{startedQuery}&quot;
                  </span>
                )}
              </div>
              <div className="flex flex-col gap-2">
                {webDocs.map((c) => (
                  <div
                    key={`${c.rank}-${c.url ?? c.source_file}`}
                    className="rounded-md border border-border bg-background p-3 text-xs"
                  >
                    <div className="mb-1 flex flex-wrap items-center justify-between gap-1 text-[11px] text-muted">
                      <span className="font-medium text-foreground">
                        {c.source_file || "Web-Treffer"}
                      </span>
                      {c.url && (
                        <a
                          href={c.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-1 text-dusty-rose-text hover:underline"
                        >
                          {domainOfUrl(c.url)}
                          <ExternalLink className="h-2.5 w-2.5" />
                        </a>
                      )}
                    </div>
                    <p className="line-clamp-3 text-foreground/90">{c.text}</p>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <p className="text-xs text-muted">
              {meta.cacheHit
                ? "Uebersprungen -- Cache hat den Pfad umgangen."
                : meta.grader?.needs_web === false
                  ? "Uebersprungen -- Grader hat den Korpus fuer ausreichend befunden."
                  : "Bisher nicht ausgeloest."}
            </p>
          )}
        </div>
      );
    }
    case "refiner": {
      const t = meta.route?.query_type;
      const desc: Record<string, string> = {
        FACTUAL: "Kurze Belegantwort mit Verbatim-Zitat-Pflicht.",
        EXPLANATORY: "Strukturierte Erklaerung mit Beispielen.",
        COMPARATIVE: "Tabellarischer Vergleich zweier Konzepte.",
      };
      return (
        <div className="flex flex-col gap-4 text-sm">
          <p>
            Der Query Router klassifiziert die Anfrage und
            w&auml;hlt ein type-spezifisches Prompt-Template aus. Plus
            harte Language-Detection-Direktive (&quot;antworte in der
            Sprache der letzten Frage&quot;) als Fix f&uuml;r einen
            fr&uuml;heren Sprach-Switch-Bug.
          </p>
          <ConfigBox
            title="Configuration"
            rows={[
              ["Classes", "FACTUAL  -  EXPLANATORY  -  COMPARATIVE"],
              ["Classifier", "gemini-2.5-flash, JSON-Schema response"],
              ["Templates", "3 type-specific (no one-size-fits-all)"],
              ["Language", "detected on last query, reply in same"],
            ]}
          />
          {t ? (
            <div className="flex flex-col gap-1 rounded-md border border-border bg-background p-3">
              <div className="text-[11px] uppercase tracking-wider text-muted">
                Classification
              </div>
              <div className="font-mono text-accent">{t}</div>
              <div className="text-xs text-muted">
                {desc[t] ?? "Custom template type."}
              </div>
            </div>
          ) : (
            <p className="text-xs text-muted">
              Noch nicht klassifiziert -- oder Cache hat den Pfad umgangen.
            </p>
          )}
        </div>
      );
    }
    case "generator": {
      const d = meta.done;
      return (
        <div className="flex flex-col gap-4 text-sm">
          <ConfigBox
            title="Configuration"
            rows={[
              ["Model", "gemini-2.5-flash via google-genai SDK"],
              ["Streaming", "token-by-token via generate_content_stream()"],
              ["Safety", "hang-guard + retry, Output-Validator post-LLM"],
            ]}
          />
          <div>
            <div className="mb-1 text-[11px] uppercase tracking-wider text-muted">
              Antwort
            </div>
            <p className="whitespace-pre-wrap font-mono text-xs leading-relaxed text-foreground">
              {fixMojibake(msg?.content ?? "") || "(noch leer)"}
            </p>
          </div>
          {d && (
            <div className="grid grid-cols-2 gap-2 text-xs text-muted">
              <Metric label="Latency">{d.latency_ms} ms</Metric>
              <Metric label="Type">{d.query_type ?? "-"}</Metric>
              <Metric label="Cache">{d.cache_hit ? "Hit" : "Miss"}</Metric>
              <Metric label="Cost">
                <span title="Geschaetzt: Gemini + Voyage zu Listenpreisen Mai 2026. Tavily-Web-Suche und Infrastrukturkosten nicht enthalten.">
                  {d.cost ? `~$${d.cost.total_usd.toFixed(4)}` : "-"}
                </span>
              </Metric>
            </div>
          )}
        </div>
      );
    }
    default:
      return null;
  }
}

function Metric({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-md border border-border bg-background px-2 py-1.5">
      <div className="text-[10px] uppercase tracking-wider text-muted">
        {label}
      </div>
      <div className="font-mono text-foreground">{children}</div>
    </div>
  );
}

function domainOfUrl(url: string): string {
  try {
    return new URL(url).host.replace(/^www\./, "");
  } catch {
    return "";
  }
}

function ConfigBox({
  title,
  rows,
}: {
  title: string;
  rows: Array<[string, string]>;
}) {
  return (
    <div className="rounded-md border border-border bg-background p-3">
      <div className="mb-2 text-[10px] uppercase tracking-wider text-muted">
        {title}
      </div>
      <div className="flex flex-col gap-1.5">
        {rows.map(([k, v]) => (
          <div
            key={k}
            className="flex flex-col sm:flex-row sm:items-baseline gap-0.5 sm:gap-3 text-xs"
          >
            <span className="text-muted shrink-0 sm:w-32">{k}</span>
            <span className="font-mono text-foreground break-all">{v}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

