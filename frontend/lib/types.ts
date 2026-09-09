// Shapes mirror the project docs + the SSE payloads emitted from
// rag_pipeline.query_stream(). Kept loose where the backend already JSON-serializes;
// strict where the UI needs to render specific fields.

export type Origin = "qdrant" | "tavily" | "cache" | string;

export interface ContextItem {
  rank: number;
  score: number;
  source_file?: string | null;
  page?: number | null;
  page_end?: number | null;
  language?: string | null;
  text: string;
  rerank_score?: number | null;
  origin?: Origin;
  // Tavily web result URL (null on Qdrant + cache chunks).
  url?: string | null;
  // Per-chunk CRAG relevance score (null when CRAG did not run --
  // cache hit, empty context, or input block).
  grader_score?: number | null;
}

export interface CostBreakdown {
  voyage_embed_usd: number;
  voyage_rerank_usd: number;
  gemini_in_usd: number;
  gemini_out_usd: number;
  total_usd: number;
}

export interface Redaction {
  kind: string;
  count: number;
}

// SSE event payloads (data JSON, decoded)
export interface RewriteEvent {
  original: string;
  rewritten: string;
}

export interface CacheHitEvent {
  distance: number;
  cached_query: string;
}

export interface ContextsEvent {
  contexts: ContextItem[];
  model: string;
  collection: string;
}

export interface RouteEvent {
  query_type: "FACTUAL" | "EXPLANATORY" | "COMPARATIVE" | string;
}

export interface TokenEvent {
  text: string;
}

export interface BlockedEvent {
  reason: string;
  pattern: string;
  trace_id?: string | null;
}

export interface RedactedEvent {
  redacted_text: string;
  redactions: Redaction[];
}

export interface DoneEvent {
  latency_ms: number;
  cost?: CostBreakdown;
  cache_hit: boolean;
  query_type: string | null;
  trace_id?: string | null;
  // CRAG critic + Tavily fallback metrics. null/0/false when CRAG
  // skipped the request (cache hit, empty contexts + no Tavily, input block).
  crag_score?: number | null;
  used_web_search?: boolean;
  chunks_filtered?: number;
}

// CRAG SSE events -- emitted only on cache miss between contexts retrieval
// and the final 'contexts' frame. See rag_pipeline.query_stream docstring.

export interface GraderEvent {
  avg_score: number;
  kept: number;
  filtered: number;
  needs_web: boolean;
  threshold: number;
}

export interface WebSearchStartEvent {
  query: string;
}

export interface WebContextsEvent {
  contexts: ContextItem[];
}

export interface ErrorEvent {
  message: string;
}

// UI message model -- assistant messages accumulate the events above.
export type ChatRole = "user" | "assistant";

export interface AssistantMeta {
  rewrite?: RewriteEvent;
  cacheHit?: CacheHitEvent;
  route?: RouteEvent;
  contexts?: ContextItem[];
  collection?: string;
  redactions?: Redaction[];
  blocked?: BlockedEvent;
  done?: DoneEvent;
  errorMessage?: string;
  // CRAG critic + Tavily web search payloads. The final 'contexts'
  // field above is the MIXED list (db kept + web); webContexts here holds the
  // web-only slice so the Inspector WebSearch drawer can render URL/domain.
  grader?: GraderEvent;
  webSearchStart?: WebSearchStartEvent;
  webContexts?: ContextItem[];
}

export interface ChatMessage {
  id: string;
  role: ChatRole;
  content: string;
  createdAt: number;
  isStreaming?: boolean;
  meta?: AssistantMeta;
}
