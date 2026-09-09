"""Pydantic request / response schemas for the FastAPI backend.

The chunk / answer field naming mirrors the retriever output so the Next.js
frontend can render DB / Web / Cache source-cards from the same payload shape
the eval scripts already produce.

Origin tag on ContextItem is one of "qdrant" / "tavily" / "cache"; the
frontend uses it to colour the source-card badges.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """POST /query and POST /query/stream body."""

    query: str = Field(..., min_length=1, max_length=2000)
    session_id: str | None = Field(
        default=None,
        description="Optional session id. Drives conversation memory and "
                    "follow-up rewriting; ignored when Redis is not configured.",
    )
    use_cache: bool = Field(
        default=True,
        description="When False, skip the semantic cache lookup AND store -- the "
                    "request runs the full pipeline and its answer is NOT written "
                    "back to the cache. Toggled from the frontend sidebar.",
    )


class FeedbackRequest(BaseModel):
    """POST /feedback body.

    ``trace_id`` is the Opik trace UUID returned in the QueryResponse for the
    answer the user is rating. Optional so the endpoint stays usable when
    Opik is disabled (the local jsonl audit log is written either way).
    """

    session_id: str
    message_id: str
    rating: Literal["up", "down"]
    note: str | None = None
    trace_id: str | None = None


class FeedbackResponse(BaseModel):
    """POST /feedback response: where the rating was persisted."""

    opik: bool = False     # True iff the Opik client logged the score
    jsonl: bool = False    # True iff the local audit log was appended


# ---------------------------------------------------------------------------
# Response bodies
# ---------------------------------------------------------------------------

class ContextItem(BaseModel):
    """One retrieved chunk, whatever it was retrieved from.

    ``url`` is set on Tavily web results and None for Qdrant and cache hits.
    ``grader_score`` carries the CRAG per-chunk relevance so the Pipeline
    Inspector can show it in the Retriever drawer when the grader ran.
    """

    rank: int
    score: float
    source_file: str
    page: int | None = None
    page_end: int | None = None
    language: str = "?"
    text: str
    rerank_score: float | None = None
    origin: Literal["qdrant", "tavily", "cache"] = "qdrant"
    url: str | None = None              # Tavily result URL
    grader_score: float | None = None   # CRAG per-chunk relevance score


class CostBreakdown(BaseModel):
    """Per-request API cost in USD -- populated by the CostTracker."""

    total_usd: float = 0.0
    voyage_usd: float = 0.0
    gemini_usd: float = 0.0


class QueryResponse(BaseModel):
    """POST /query response body."""

    answer: str
    contexts: list[ContextItem]
    model: str
    collection: str
    latency_ms: int
    cost: CostBreakdown
    # Conversation memory + conditional rewrite
    rewritten_query: str | None = None        # populated only when rewrite fired
    was_rewritten: bool = False
    # Set by the semantic cache and the CRAG web fallback respectively
    cache_hit: bool = False
    used_web_search: bool = False
    # CRAG critic metrics. ``crag_score`` is the avg per-chunk
    # relevance score from the grader (None when skipped: cache hit,
    # empty-context, or input-block). ``chunks_filtered`` is how many DB
    # chunks the grader dropped before generate.
    crag_score: float | None = None
    chunks_filtered: int = 0
    # Query routing -- None on cache hits (no classifier call), otherwise
    # one of {"FACTUAL", "EXPLANATORY", "COMPARATIVE"}.
    query_type: str | None = None
    # Opik trace UUID -- frontend includes this in /feedback so the
    # rating attaches to the correct trace. None when observability is off.
    trace_id: str | None = None
    # OutputValidator redactions applied to the answer. Empty list
    # if no PII was found. Each entry: {"kind": "EMAIL"|"PHONE_DE"|..., "count": N}.
    redactions: list[dict] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """GET /health -- liveness + config snapshot, no secrets."""

    status: Literal["ok", "degraded"] = "ok"
    version: str
    embed_model: str
    rerank_model: str
    gen_model: str
    collection: str
