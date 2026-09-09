"use client";

import { useEffect } from "react";
import { X } from "lucide-react";

// Animated architecture diagram of the shipped CRAG pipeline, opened from
// the "Overview" button next to the "CRAG Pipeline" header in the Inspector.
// Landscape (viewBox 1280x650, modal ~95vw): an OFFLINE INGEST lane on top
// and the live query flow as a serpentine below (row A left-to-right, row B
// right-to-left). The main route is ONE continuous line drawn behind the
// nodes, so the whole path from Query to Answer reads as a single journey.
//
// Node order mirrors the REAL execution order in rag_pipeline.query_stream:
// InputGuard -> rewrite -> cache lookup -> hybrid retrieve -> rerank ->
// DocumentGuard -> CRAG grader -> (Tavily web fallback) -> query router ->
// generator -> OutputValidator -> answer. The router deliberately sits
// AFTER the web branch (rag_pipeline.py: classify runs right before
// generate).
//
// The semantic cache is drawn as a round trip between Answer and Cache:
// the lower honey curve is the HIT path (stored answer served directly),
// the parallel upper curve is the STORE path (every validated answer is
// written back). Tavily sits in the middle band directly under the cache,
// with a short honey connector because web-fallback answers are cached
// too (decision 2026-06-26, rag_pipeline.py "Cache ALL successful
// answers, including web-fallback ones").
//
// Colors come from the app's CSS custom properties (light/dark automatic);
// flow animation = stroke-dashoffset keyframe (globals.css .flow-line) +
// SMIL animateMotion pulse dots. Labels use a paint-order halo AND are
// placed with geometric clearance from every curve.

interface PipelineDiagramDialogProps {
  onClose: () => void;
}

type Tone = "default" | "security" | "cache" | "grader" | "web" | "answer" | "ingest";

// The -text variants render as the original soft hues in dark mode and as
// darker, contrast-safe variants in light mode (light-mode token pattern).
const TONE_STROKE: Record<Tone, string> = {
  default: "var(--border)",
  security: "var(--sage-text)",
  cache: "var(--honey-text)",
  grader: "var(--accent)",
  web: "var(--dusty-rose-text)",
  answer: "var(--accent)",
  ingest: "var(--clay-text)",
};

function Node({
  x,
  y,
  w = 170,
  h = 64,
  title,
  sub,
  sub2,
  tone = "default",
}: {
  x: number;
  y: number;
  w?: number;
  h?: number;
  title: string;
  sub?: string;
  sub2?: string;
  tone?: Tone;
}) {
  const answer = tone === "answer";
  const titleY = y + (sub2 ? 23 : sub ? 27 : h / 2 + 5);
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={10}
        fill={answer ? "var(--accent)" : "var(--pipe-node-fill)"}
        stroke={TONE_STROKE[tone]}
        strokeWidth={tone === "default" ? 1 : 1.8}
      />
      <text
        x={x + w / 2}
        y={titleY}
        textAnchor="middle"
        fontSize={16}
        fontWeight={600}
        fill={answer ? "var(--on-accent)" : "var(--foreground)"}
      >
        {title}
      </text>
      {sub && (
        <text
          x={x + w / 2}
          y={titleY + 17}
          textAnchor="middle"
          fontSize={11}
          fill={answer ? "var(--on-accent)" : "var(--muted)"}
          opacity={answer ? 0.85 : 1}
        >
          {sub}
        </text>
      )}
      {sub2 && (
        <text
          x={x + w / 2}
          y={titleY + 31}
          textAnchor="middle"
          fontSize={11}
          fill={answer ? "var(--on-accent)" : "var(--muted)"}
          opacity={answer ? 0.85 : 1}
        >
          {sub2}
        </text>
      )}
    </g>
  );
}

function FlowEdge({
  d,
  color,
  dotDur,
  width = 2.2,
}: {
  d: string;
  color: string;
  dotDur?: string;
  width?: number;
}) {
  return (
    <g filter="url(#pipeGlow)">
      <path
        d={d}
        className="flow-line"
        fill="none"
        stroke={color}
        strokeWidth={width}
        strokeLinecap="round"
      />
      {dotDur && (
        <circle r={4} fill={color}>
          <animateMotion dur={dotDur} repeatCount="indefinite" path={d} />
        </circle>
      )}
    </g>
  );
}

// Text with a background-colored halo (paint-order: stroke) so labels stay
// readable even where they touch an edge.
function EdgeLabel({
  x,
  y,
  color,
  anchor = "middle",
  children,
}: {
  x: number;
  y: number;
  color: string;
  anchor?: "start" | "middle" | "end";
  children: React.ReactNode;
}) {
  return (
    <text
      x={x}
      y={y}
      textAnchor={anchor}
      fontSize={12.5}
      fontWeight={600}
      fill={color}
      paintOrder="stroke"
      stroke="var(--pipe-halo)"
      strokeWidth={5}
      strokeLinejoin="round"
      strokeLinecap="round"
    >
      {children}
    </text>
  );
}

export function PipelineDiagramDialog({ onClose }: PipelineDiagramDialogProps) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Grid: 6 columns, x = 30 + col * 210 (node w=170, gap 40). Column
  // centers: 115 / 325 / 535 / 745 / 955 / 1165. Ingest lane y=64,
  // row A y=240, Tavily y=400 (middle band), row B y=560.

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div className="fixed inset-0 bg-foreground/40 backdrop-blur-[2px]" />
      <div
        role="dialog"
        aria-modal="true"
        aria-label="CRAG pipeline architecture diagram"
        className="relative flex max-h-[94vh] w-[95vw] max-w-[1400px] flex-col overflow-hidden rounded-xl border border-border bg-background shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-border px-5 py-3">
          <div>
            <h3 className="text-lg font-semibold text-foreground">
              CRAG Pipeline -- Architecture
            </h3>
            <p className="text-xs text-muted">
              22 textbooks &middot; 3,053 pages &middot; 4,648 chunks &middot;
              offline ingest + live query flow (real execution order)
            </p>
          </div>
          <button
            onClick={onClose}
            className="rounded-md p-1 text-muted hover:bg-surface hover:text-foreground"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="overflow-y-auto px-4 py-3">
          <div className="pipe-canvas p-2">
          <svg
            viewBox="0 0 1280 680"
            className="w-full"
            role="img"
            aria-label="Flow diagram of the CRAG pipeline"
          >
            <defs>
              <filter id="pipeGlow" x="-40%" y="-40%" width="180%" height="180%">
                <feGaussianBlur stdDeviation="2.6" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>

            {/* ================= OFFLINE INGEST LANE ================= */}
            {/* frame is vertically centered on the ingest route (y=92):
                y 34..150, center 92 = node center = line y */}
            <rect
              x={20}
              y={34}
              width={1240}
              height={116}
              rx={12}
              fill="none"
              stroke="var(--muted)"
              strokeWidth={1}
              strokeDasharray="4 6"
              opacity={0.55}
            />
            <text
              x={1240}
              y={86}
              textAnchor="end"
              fontSize={13}
              fontWeight={600}
              letterSpacing={2}
              fill="var(--muted)"
            >
              OFFLINE INGEST
            </text>
            <text
              x={1240}
              y={105}
              textAnchor="end"
              fontSize={11}
              fill="var(--muted)"
              opacity={0.8}
            >
              one-time, before shipping
            </text>

            {/* one continuous ingest line: PDFs -> ... -> Qdrant -> retriever */}
            <FlowEdge
              d="M115,92 L955,92 L955,240"
              color="var(--clay-text)"
              dotDur="5s"
            />

            <Node x={30} y={64} h={56} title="22 PDF Books" sub="3,053 pages, DE + RU" tone="ingest" />
            <Node x={240} y={64} h={56} title="Extraction" sub="pypdf + Marker / Surya OCR" tone="ingest" />
            <Node x={450} y={64} h={56} title="Chunking" sub="logical recursive 2000 / 200" tone="ingest" />
            <Node x={660} y={64} h={56} title="Embedding" sub="voyage-multilingual-2 + BM25" tone="ingest" />
            <Node x={870} y={64} h={56} title="Qdrant Cloud" sub="4,648 chunks, hybrid index" tone="ingest" />

            {/* ================= MAIN QUERY ROUTE ======================== */}
            {/* ONE continuous line behind the nodes: row A left-to-right,
                wrap down at column 6, row B right-to-left into Answer. */}
            <FlowEdge
              d="M115,272 L1165,272 L1165,592 L115,592"
              color="var(--accent)"
              dotDur="9s"
            />

            {/* ============ SEMANTIC CACHE ROUND TRIP (honey) ============ */}
            {/* Two parallel curves between Answer and Cache -- no crossing.
                Lower curve = HIT (cache -> answer), upper curve = STORE
                (validated answer -> cache). Labels sit clear of both. */}
            {/* endpoint sides are consistent at BOTH nodes so the curves
                run parallel without crossing: store = left at cache (690)
                AND left at answer (80); hit = right at cache (745) AND
                right at answer (170) */}
            <FlowEdge
              d="M745,304 C620,485 250,510 170,560"
              color="var(--honey-text)"
              dotDur="3.4s"
            />
            <EdgeLabel x={535} y={474} color="var(--honey-text)" anchor="start">
              cache hit -- instant answer
            </EdgeLabel>
            <EdgeLabel x={535} y={491} color="var(--honey-text)" anchor="start">
              (~31x faster)
            </EdgeLabel>
            <FlowEdge
              d="M80,560 C220,500 570,430 690,306"
              color="var(--honey-text)"
              dotDur="4.2s"
            />
            <EdgeLabel x={240} y={464} color="var(--honey-text)">
              every answer is stored
            </EdgeLabel>
            {/* Tavily sits right under the cache -- short connector makes
                "web answers are cached too" explicit */}
            <FlowEdge
              d="M850,400 C848,362 800,330 790,304"
              color="var(--honey-text)"
              dotDur="2.4s"
            />
            <EdgeLabel x={868} y={352} color="var(--honey-text)" anchor="start">
              web answers cached too
            </EdgeLabel>

            {/* ============ CRAG WEB BRANCH (dusty-rose, middle band) ===== */}
            {/* grader sends the query up to Tavily, the web docs re-join
                the main route right before the Query Router */}
            <FlowEdge
              d="M955,560 C955,495 945,458 935,432"
              color="var(--dusty-rose-text)"
              dotDur="2.6s"
            />
            <FlowEdge
              d="M765,428 C748,458 745,510 745,560"
              color="var(--dusty-rose-text)"
              dotDur="2.6s"
            />
            <EdgeLabel x={975} y={505} color="var(--dusty-rose-text)" anchor="start">
              max &lt; 0.45 -&gt; needs web
            </EdgeLabel>
            <EdgeLabel x={852} y={496} color="var(--dusty-rose-text)">
              3 web docs join the contexts
            </EdgeLabel>

            {/* ================= ROW A (left to right) =================== */}
            <Node x={30} y={240} title="Query" sub="German or Russian" />
            <Node
              x={240}
              y={240}
              title="Input Guard"
              sub="prompt-injection filter"
              sub2="DE + RU + EN patterns"
              tone="security"
            />
            <Node
              x={450}
              y={240}
              title="Query Rewrite"
              sub="resolves follow-ups"
              sub2="gemini-2.5-flash, temp=0"
            />
            <Node
              x={660}
              y={240}
              title="Semantic Cache"
              sub="Redis HNSW, threshold 0.20"
              sub2="+ lexical guard (Jaccard 0.7)"
              tone="cache"
            />
            <Node
              x={870}
              y={240}
              title="Hybrid Retrieval"
              sub="voyage dense 100 + BM25 100"
              sub2="RRF fuse -&gt; Top-100"
            />
            <Node
              x={1080}
              y={240}
              title="Reranker"
              sub="voyage rerank-2.5"
              sub2="Top-100 -&gt; Top-10"
            />

            {/* Tavily in the middle band, above CRAG Grader / Query Router
                and right under the Semantic Cache */}
            <Node
              x={765}
              y={400}
              h={56}
              title="Tavily Web Search"
              sub="max. 3 web documents"
              tone="web"
            />

            {/* ================= ROW B (right to left) =================== */}
            <Node
              x={1080}
              y={560}
              title="Document Guard"
              sub="poisoning scan on DB"
              sub2="and web chunks"
              tone="security"
            />
            <Node
              x={870}
              y={560}
              title="CRAG Grader"
              sub="scores every chunk 0.0 - 1.0"
              sub2="gemini-2.5-flash, temp=0"
              tone="grader"
            />
            <Node
              x={660}
              y={560}
              title="Query Router"
              sub="FACTUAL / EXPLANATORY"
              sub2="/ COMPARATIVE templates"
            />
            <Node
              x={450}
              y={560}
              title="Generator"
              sub="gemini-2.5-flash"
              sub2="SSE token stream"
            />
            <Node
              x={240}
              y={560}
              title="Output Validator"
              sub="PII redaction"
              tone="security"
            />
            <Node
              x={30}
              y={560}
              title="Answer"
              sub="[book, page] or web citations"
              tone="answer"
            />
          </svg>
          </div>

          <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-1 px-2 pb-1 text-xs text-muted">
            <LegendDot color="var(--accent)" label="Live query route" />
            <LegendDot color="var(--sage-text)" label="Security guards" />
            <LegendDot color="var(--dusty-rose-text)" label="CRAG web fallback" />
            <LegendDot color="var(--honey-text)" label="Semantic cache (hit + store)" />
            <LegendDot color="var(--clay-text)" label="Offline ingest (one-time)" />
          </div>
        </div>
      </div>
    </div>
  );
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-1.5">
      <span
        className="inline-block h-2 w-2 rounded-full"
        style={{ background: color }}
      />
      {label}
    </span>
  );
}
