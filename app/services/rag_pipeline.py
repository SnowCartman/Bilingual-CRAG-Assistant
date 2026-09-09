"""The retrieval-and-generation pipeline, end to end.

The winning configuration from the retrieval benchmark (hybrid search + a
cross-encoder reranker, 100 fused candidates) ported into a single service
module. It stays one file on purpose: the public surface is one class with two
methods, and a reader following a query through the system should not have to
jump between six modules to do it.

Pipeline shape:

    user query
        |-- voyage-multilingual-2 embed (input_type=query, L2-normalised)
        |-- fastembed BM25 sparse embed
        v
    Qdrant hybrid search (dense_prefetch=100 + sparse_prefetch=100 -> RRF -> top-100)
        v
    Voyage rerank-2.5 (top-10)
        v
    app.services.query_router classifies (FACTUAL/EXPLANATORY/COMPARATIVE)
        v
    app.prompts.templates.build_prompt(query_type, ...) -> Gemini 2.5-flash generate (hang-guard'd)

Hang-guard ports from common.py: HTTP timeout on the client + per-call watchdog
daemon thread + retry loop honouring "retry in Xs" hints from 429 errors.
gemini-2.5-flash is frequently overloaded, so this matters in production too.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

import numpy as np
import requests
from fastembed import SparseTextEmbedding
from google import genai
from google.genai import types as genai_types
from qdrant_client import QdrantClient, models as qmodels

from app.config import Settings
from app.models import ContextItem, CostBreakdown
from app.prompts.templates import build_prompt
from app.services import observability as obs
from app.services.security import (
    DocumentGuard,
    InputGuard,
    OutputValidator,
    PromptInjectionBlocked,
)

log = logging.getLogger("rag_pipeline")

# --- Pricing tables (May 2026 list prices, USD per 1M tokens) ---------------
GEMINI_PRICING = {
    "gemini-2.5-flash":      {"in": 0.30, "out": 2.50},
    "gemini-2.5-flash-lite": {"in": 0.10, "out": 0.40},
    "gemini-2.5-pro":        {"in": 1.25, "out": 10.00},
}
VOYAGE_PRICING = {
    "voyage-multilingual-2": 0.12,
    "rerank-2.5":            0.05,
}

# NOTE: There is no single SYSTEM_PROMPT here any more. The prompt is built
# per query by app.prompts.templates.build_prompt(query_type, ...), driven by
# the FACTUAL/EXPLANATORY/COMPARATIVE classifier in app.services.query_router.
# The persona this replaced ("a Russian native speaker who is learning German")
# was the source of a language-switching bug in evaluation: the model read it
# as an instruction to answer in Russian. _BASE_RULES states the language rule
# imperatively instead.


# ---------------------------------------------------------------------------
# Cost tracking (per-request scope -- one tracker lives for one query call)
# ---------------------------------------------------------------------------

@dataclass
class CostTracker:
    """Per-request cost accumulator. Ported from common.CostTracker."""

    gemini: dict[str, dict] = field(default_factory=dict)
    voyage: dict[str, dict] = field(default_factory=dict)

    def add_gemini(self, model: str, prompt_tokens: int, output_tokens: int) -> None:
        rec = self.gemini.setdefault(model, {"calls": 0, "in": 0, "out": 0})
        rec["calls"] += 1
        rec["in"] += prompt_tokens
        rec["out"] += output_tokens

    def add_voyage(self, model: str, tokens: int) -> None:
        rec = self.voyage.setdefault(model, {"calls": 0, "tokens": 0})
        rec["calls"] += 1
        rec["tokens"] += tokens

    def to_breakdown(self) -> CostBreakdown:
        gemini_usd = 0.0
        for model, rec in self.gemini.items():
            price = GEMINI_PRICING.get(model, {"in": 0.0, "out": 0.0})
            gemini_usd += rec["in"] / 1e6 * price["in"] + rec["out"] / 1e6 * price["out"]
        voyage_usd = 0.0
        for model, rec in self.voyage.items():
            rate = VOYAGE_PRICING.get(model, 0.0)
            voyage_usd += rec["tokens"] / 1e6 * rate
        return CostBreakdown(
            total_usd=round(gemini_usd + voyage_usd, 6),
            gemini_usd=round(gemini_usd, 6),
            voyage_usd=round(voyage_usd, 6),
        )


# ---------------------------------------------------------------------------
# Voyage HTTP helpers (no SDK -- explicit and transparent)
# ---------------------------------------------------------------------------

def _voyage_post(url: str, payload: dict, api_key: str, max_attempts: int = 5) -> dict:
    """POST to a Voyage endpoint with retry on transient 429 / 5xx."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    for attempt in range(1, max_attempts + 1):
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < max_attempts:
            backoff = min(30.0, 2 ** attempt)
            log.warning("voyage %s; retry %d/%d in %.0fs",
                        resp.status_code, attempt, max_attempts, backoff)
            time.sleep(backoff)
            continue
        resp.raise_for_status()
    raise RuntimeError(f"Voyage call failed after {max_attempts} attempts: {url}")


def voyage_embed_query(text: str, model: str, api_key: str,
                       cost: CostTracker) -> list[float]:
    """Embed a single query with voyage-multilingual-2 (1024d, L2-normalised)."""
    body = _voyage_post(
        "https://api.voyageai.com/v1/embeddings",
        {"input": [text], "model": model, "input_type": "query"},
        api_key,
    )
    cost.add_voyage(model, body.get("usage", {}).get("total_tokens", 0))
    v = np.array(body["data"][0]["embedding"], dtype=np.float32)
    norm = float(np.linalg.norm(v)) or 1.0
    return (v / norm).tolist()


def voyage_rerank(query: str, documents: list[str], top_k: int,
                  model: str, api_key: str, cost: CostTracker) -> list[dict]:
    """Voyage rerank-2.5. Returns [{index, score}] sorted desc."""
    if not documents:
        return []
    body = _voyage_post(
        "https://api.voyageai.com/v1/rerank",
        {"query": query, "documents": documents, "model": model,
         "top_k": min(top_k, len(documents))},
        api_key,
    )
    cost.add_voyage(model, body.get("usage", {}).get("total_tokens", 0))
    results = [{"index": d["index"], "score": float(d["relevance_score"])} for d in body["data"]]
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Gemini hang-guard (ported from common.gemini_generate)
# ---------------------------------------------------------------------------

def _retry_hint_seconds(message: str) -> float | None:
    m = re.search(r"retry in ([\d.]+)s", message, re.IGNORECASE)
    return float(m.group(1)) if m else None


def gemini_generate_guarded(
    client,
    model: str,
    prompt: str,
    *,
    config=None,
    max_attempts: int = 12,
    call_timeout_s: float = 150.0,
    label: str = "gemini",
) -> dict:
    """Robust Gemini call: retry + per-call hang guard (watchdog daemon thread).

    Returns {text, prompt_tokens, output_tokens, attempts}. Raises RuntimeError
    after max_attempts. Ported 1:1 from an earlier retrieval prototype
    """
    last_err = "unknown"
    for attempt in range(1, max_attempts + 1):
        box: dict = {}

        def _worker():
            try:
                box["resp"] = client.models.generate_content(
                    model=model, contents=prompt, config=config
                )
            except Exception as exc:  # noqa: BLE001
                box["err"] = exc

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=call_timeout_s)

        if t.is_alive():
            last_err = f"call exceeded {call_timeout_s}s hang guard"
        elif "err" in box:
            last_err = str(box["err"])
        else:
            resp = box["resp"]
            um = getattr(resp, "usage_metadata", None)
            prompt_tokens = getattr(um, "prompt_token_count", 0) or 0
            output_tokens = (getattr(um, "candidates_token_count", 0) or 0) + (
                getattr(um, "thoughts_token_count", 0) or 0
            )
            return {
                "text": resp.text or "",
                "prompt_tokens": prompt_tokens,
                "output_tokens": output_tokens,
                "attempts": attempt,
            }

        if attempt >= max_attempts:
            break
        hint = _retry_hint_seconds(last_err)
        wait = (hint + 2) if hint else min(60.0, 5 * 2 ** (attempt - 1))
        log.warning("%s: attempt %d/%d failed (%s); retry in %.0fs",
                    label, attempt, max_attempts, last_err[:110], wait)
        time.sleep(wait)

    raise RuntimeError(f"{label}: all {max_attempts} attempts failed. Last: {last_err}")


# ---------------------------------------------------------------------------
# Qdrant hybrid search (ported from an earlier retrieval prototype)
# ---------------------------------------------------------------------------

def _point_to_dict(point, rank: int) -> dict:
    pl = point.payload or {}
    return {
        "rank": rank,
        "score": round(float(point.score), 5) if point.score is not None else 0.0,
        "source_file": pl.get("source_file", "?"),
        "page": pl.get("page"),
        "page_end": pl.get("page_end", pl.get("page")),
        "language": pl.get("language", "?"),
        "text": pl.get("text", ""),
    }


def hybrid_search(
    qdrant: QdrantClient,
    collection: str,
    dense_vec: list[float],
    sparse_vec,                       # fastembed SparseEmbedding (has .indices / .values)
    top_k: int,
    dense_prefetch: int,
    sparse_prefetch: int,
) -> list[dict]:
    """Dense + sparse with explicit prefetch limits before RRF fusion."""
    qsparse = qmodels.SparseVector(
        indices=sparse_vec.indices.tolist(),
        values=sparse_vec.values.tolist(),
    )
    resp = qdrant.query_points(
        collection_name=collection,
        prefetch=[
            qmodels.Prefetch(query=dense_vec, using="dense",  limit=dense_prefetch),
            qmodels.Prefetch(query=qsparse,   using="sparse", limit=sparse_prefetch),
        ],
        query=qmodels.FusionQuery(fusion=qmodels.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return [_point_to_dict(p, i + 1) for i, p in enumerate(resp.points)]


# ---------------------------------------------------------------------------
# RAGPipeline -- the public class wired into FastAPI lifespan
# ---------------------------------------------------------------------------

class RAGPipeline:
    """Single instance per process, created in FastAPI's lifespan startup."""

    def __init__(self, settings: Settings) -> None:
        self.s = settings
        log.info("RAGPipeline: connecting to Qdrant %s", settings.qdrant_url)
        self.qdrant = QdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=120
        )
        log.info("RAGPipeline: loading fastembed BM25")
        self.bm25 = SparseTextEmbedding(settings.sparse_model)
        log.info("RAGPipeline: building Gemini client (timeout=%.0fs)",
                 settings.gemini_http_timeout_s)
        self.gemini = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=genai_types.HttpOptions(
                timeout=int(settings.gemini_http_timeout_s * 1000)
            ),
        )
        # Shared generation config -- low temperature for stable, language-
        # faithful grounded answers (see config.gen_temperature), and an
        # explicit thinking budget so answer latency stays predictable
        # (see config.gen_thinking_budget). A negative budget means "leave the
        # decision to the model", i.e. send no thinking_config at all.
        self._gen_config = genai_types.GenerateContentConfig(
            temperature=settings.gen_temperature,
            thinking_config=(
                genai_types.ThinkingConfig(
                    thinking_budget=settings.gen_thinking_budget)
                if settings.gen_thinking_budget >= 0 else None
            ),
        )
        # Session memory (Redis sliding window) -- gracefully disabled
        # if REDIS_HOST is not set, so the service still runs stateless.
        from app.services.session_memory import SessionMemory  # noqa: PLC0415
        self.memory = SessionMemory(settings)
        # Semantic cache (Redis HNSW). Same graceful-degradation contract.
        from app.services.semantic_cache import SemanticCache  # noqa: PLC0415
        self.cache = SemanticCache(settings)
        # Regex guards -- cheap to build, no external deps.
        self.input_guard = InputGuard()
        self.doc_guard = DocumentGuard()
        self.output_validator = OutputValidator()
        # CRAG critic + Tavily web-search fallback. Both gracefully
        # degrade -- ContextGrader needs Gemini (already configured), Tavily
        # needs an API key (skipped silently when missing).
        from app.augmentations.crag_critic import ContextGrader  # noqa: PLC0415
        from app.augmentations.web_search import TavilyClient  # noqa: PLC0415
        self.grader = ContextGrader(
            gemini_client=self.gemini,
            model=settings.crag_model,
            crag_threshold=settings.crag_threshold,
            per_chunk_threshold=settings.per_chunk_threshold,
            max_attempts=settings.gemini_max_attempts,
            call_timeout_s=settings.gemini_call_timeout_s,
            thinking_budget=settings.grader_thinking_budget,
        )
        self.tavily = TavilyClient(
            api_key=settings.tavily_api_key,
            max_results=settings.tavily_max_results,
        )
        log.info(
            "RAGPipeline: ready (collection=%s, memory=%s, cache=%s, "
            "guards=on, crag=on, tavily=%s)",
            settings.qdrant_collection,
            "on" if self.memory.enabled else "off",
            "on" if self.cache.enabled else "off",
            "on" if self.tavily.enabled else "off",
        )

    # ---- internal helpers ----------------------------------------------------

    @obs.safe_track(name="embed")
    def _embed_query(self, query: str, cost: CostTracker) -> list[float]:
        """Voyage dense embed -- pulled out of _retrieve so the semantic
        cache can reuse the SAME embedding for the lookup (no extra Voyage call)."""
        return voyage_embed_query(query, self.s.embed_model,
                                  self.s.voyage_api_key, cost)

    @obs.safe_track(name="retrieve")
    def _retrieve_with_embedding(self, query: str, dense: list[float],
                                 cost: CostTracker) -> list[dict]:
        """Hybrid retrieve + rerank, given a pre-computed dense embedding."""
        sparse = next(iter(self.bm25.query_embed([query])))
        candidates = hybrid_search(
            self.qdrant, self.s.qdrant_collection, dense, sparse,
            top_k=self.s.rrf_prefetch,
            dense_prefetch=self.s.dense_prefetch,
            sparse_prefetch=self.s.sparse_prefetch,
        )
        if not candidates:
            obs.update_span(metadata={"candidates": 0, "reranked": 0})
            return []
        ranked = voyage_rerank(
            query, [c["text"] for c in candidates],
            self.s.top_k, self.s.rerank_model, self.s.voyage_api_key, cost,
        )
        out = []
        for new_rank, r in enumerate(ranked, 1):
            ctx = dict(candidates[r["index"]])
            ctx["rank"] = new_rank
            ctx["rerank_score"] = round(r["score"], 5)
            ctx["origin"] = "qdrant"
            out.append(ctx)
        # Helpful at-a-glance metric in Opik: how many candidates the reranker
        # picked from the RRF-fused pool.
        obs.update_span(metadata={"candidates": len(candidates), "reranked": len(out)})
        return out

    def _retrieve(self, query: str, cost: CostTracker) -> list[dict]:
        """Convenience wrapper used by tests / callers that don't want to manage
        the embedding themselves. The cache path calls _embed_query +
        _retrieve_with_embedding directly so it can intercept after the embed."""
        dense = self._embed_query(query, cost)
        return self._retrieve_with_embedding(query, dense, cost)

    @obs.safe_track(name="rewrite")
    def _maybe_rewrite(self, text: str, session_id: str | None,
                       cost: CostTracker) -> tuple[str, bool]:
        """Apply conditional rewrite. Returns (effective_query, was_rewritten).
        Stateless when session_id is None or memory is disabled."""
        if not session_id or not self.memory.enabled:
            return text, False
        # Local import avoids a top-of-file circular ref through CostTracker.
        from app.services.query_rewriter import rewrite  # noqa: PLC0415
        prior = self.memory.get_recent_turns(session_id)
        if not prior:
            return text, False
        rewritten, was = rewrite(text, prior, self.gemini, self.s.rewrite_model, cost)
        if was:
            # Tag with rewrite-prompt version so we can A/B template changes later.
            from app.prompts.registry import get as get_prompt  # noqa: PLC0415
            obs.update_span(tags=[get_prompt("rewrite").tag()])
        return rewritten, was

    @obs.safe_track(name="classify", type="llm")
    def _classify(self, query: str, cost: CostTracker) -> str:
        """Router as a separate span -- emitted only on cache miss."""
        from app.services.query_router import classify  # noqa: PLC0415
        from app.prompts.registry import get as get_prompt  # noqa: PLC0415
        cls = classify(query, self.gemini, self.s.router_model, cost)
        obs.update_span(
            metadata={"query_type": cls},
            tags=[get_prompt("router").tag()],
        )
        return cls

    def _filter_documents(self, contexts: list[dict]) -> list[dict]:
        """Drop retrieval-poisoned chunks via DocumentGuard.

        Threshold is conservative (>=3 weighted points) so textbook content
        that legitimately quotes an attacker phrase does NOT get dropped.
        Attaches counts + dropped source-file names to the parent span so the
        Pipeline Inspector can render a Security Strip.
        """
        if not contexts:
            return contexts
        kept: list[dict] = []
        flagged_sources: list[str] = []
        for ctx in contexts:
            dg = self.doc_guard.scan(ctx.get("text", ""))
            if dg.flagged:
                flagged_sources.append(ctx.get("source_file", "?"))
                log.warning("DocumentGuard flagged chunk: source=%s score=%d patterns=%s",
                            ctx.get("source_file"), dg.score,
                            [m["name"] for m in dg.matched_patterns])
            else:
                kept.append(ctx)
        obs.update_span(metadata={
            "doc_guard_kept":            len(kept),
            "doc_guard_flagged":         len(flagged_sources),
            "doc_guard_flagged_sources": flagged_sources,
        })
        return kept

    def _validate_output(self, answer: str) -> tuple[str, list[dict]]:
        """Redact PII from the final answer; returns (text, redactions)."""
        ov = self.output_validator.scan(answer or "")
        obs.update_span(metadata={
            "output_validator_redactions": ov.redactions,
            "output_validator_is_clean":   ov.is_clean,
        })
        return ov.redacted_text, ov.redactions

    # ---- CRAG + Tavily ---------------------------------------------

    @obs.safe_track(name="crag_grade", type="llm")
    def _grade_contexts(self, query: str, contexts: list[dict],
                        cost: CostTracker):
        """Score retrieved chunks for relevance + decide if web fallback fires.

        Returns the full ``GraderVerdict``. The pipeline reads ``needs_web`` +
        ``kept`` + ``avg_score`` from it. Per-chunk scores are attached back
        to context dicts so the Inspector drawer can render them.
        """
        verdict = self.grader.grade(query, contexts, cost)
        # Attach per-chunk scores onto context dicts so the Inspector can show
        # them in the Retriever drawer. Match by rank (1-based).
        by_rank = {s.rank: s.score for s in verdict.per_chunk}
        for i, ctx in enumerate(contexts, start=1):
            if i in by_rank:
                ctx["grader_score"] = round(by_rank[i], 3)
        obs.update_span(metadata={
            "crag_avg_score":   verdict.avg_score,
            "crag_kept":        len(verdict.kept),
            "crag_filtered":    len(verdict.filtered),
            "crag_needs_web":   verdict.needs_web,
            "crag_threshold":   self.s.crag_threshold,
        })
        return verdict

    async def _remember_failed_turn(self, session_id: str | None,
                                    text: str) -> None:
        """Record a turn whose answer never reached the user.

        Session memory feeds exactly one consumer: the query rewriter. If a
        failed turn is simply absent, a follow-up like "die Antwort wurde
        nicht generiert, mach das nochmal" has no prior question to resolve
        against, so the complaint itself becomes the retrieval query.
        """
        if not session_id:
            return
        await asyncio.to_thread(
            self.memory.add_turn, session_id, text, _FAILED_TURN_NOTE,
        )

    @obs.safe_track(name="web_search")
    def _web_search_fallback(self, query: str) -> list[dict]:
        """Tavily web-search call + DocumentGuard filter on web docs."""
        if not self.tavily.enabled:
            obs.update_span(metadata={"tavily_enabled": False, "web_results": 0})
            return []
        raw = self.tavily.search(query)
        # Same DocumentGuard sweep we run on Qdrant chunks; web
        # docs are higher-risk for retrieval-poisoning. Threshold is the same
        # conservative >=3.
        kept = self._filter_documents(raw)
        obs.update_span(metadata={
            "tavily_enabled":   True,
            "web_results_raw":  len(raw),
            "web_results_kept": len(kept),
        })
        return kept

    # ---- public API ----------------------------------------------------------

    @obs.safe_track(name="query")
    def query(
        self,
        text: str,
        session_id: str | None = None,
        use_cache: bool = True,
    ) -> dict:
        """Sync: full pipeline, returns dict ready for QueryResponse(**...).

        Cache hit short-circuit:
        - rewrite first (so the cache key is the canonical standalone query)
        - embed once (re-used by either cache.lookup OR retrieve, never both)
        - if cache hit: skip retrieve + generate, mark every context origin='cache'
        - if miss: normal hybrid + rerank + Gemini, then cache.store at the end

        ``use_cache=False`` lets the frontend force the full pipeline path
        (lookup AND store skipped). Useful for the cache toggle + live
        demos that show the same query running with/without the cache.
        """
        t0 = time.perf_counter()
        cost = CostTracker()
        # Trace-level metadata up front -- session_id makes it easy to grep
        # multi-turn conversations in the Opik UI.
        obs.update_trace(
            metadata={"session_id": session_id, "original_query": text},
        )
        # InputGuard FIRST -- block known prompt-injection before
        # spending a single Voyage/Qdrant token. The exception is caught by
        # main.py's /query handler and turned into HTTP 422 with a localized
        # error message.
        ig = self.input_guard.scan(text)
        obs.update_span(metadata={"input_guard_verdict": ig.verdict,
                                  "input_guard_patterns": ig.matched_patterns})
        if ig.verdict == "blocked":
            raise PromptInjectionBlocked(ig.matched_patterns[0], text)
        effective_query, was_rewritten = self._maybe_rewrite(text, session_id, cost)

        # Embed once -- reused by lookup (if cache enabled) AND retrieve (if miss).
        dense = self._embed_query(effective_query, cost)
        cache_hit_entry = (
            self.cache.lookup(dense, query_text=effective_query)
            if (self.cache.enabled and use_cache)
            else None
        )

        # Query_type stays None on cache hits (we deliberately skip the
        # classifier so cache lookups remain cheap) and on empty-context refusals.
        query_type: str | None = None
        # Per-request CRAG metrics returned at the end. None / 0 /
        # False means "did not fire" (cache hit, empty corpus + no Tavily,
        # or grader said the DB chunks suffice).
        crag_score: float | None = None
        chunks_filtered = 0
        used_web_search = False
        if cache_hit_entry is not None:
            contexts = [_with_origin(c, "cache") for c in cache_hit_entry["contexts"]]
            answer = cache_hit_entry["answer"]
            log.info("cache HIT: query='%s' distance=%.4f",
                     effective_query[:60], cache_hit_entry["distance"])
            cache_hit = True
        else:
            contexts = self._retrieve_with_embedding(effective_query, dense, cost)
            # DocumentGuard between retrieve and generate -- drop
            # poisoned chunks before they reach the LLM. Conservative threshold
            # (>=3) so legitimate textbook content that quotes an injection
            # phrase (e.g. as an example) does NOT get filtered.
            contexts = self._filter_documents(contexts)
            # CRAG critic + Tavily fallback. Two entry points:
            #   (a) empty contexts after DocumentGuard -> direct Tavily (no
            #       grader call, no chunks to score).
            #   (b) non-empty contexts -> grader decides web-or-not.
            # Both honour the Tavily-disabled case: fall through to the
            # existing refusal / DB-only path so deployments without
            # TAVILY_API_KEY keep behaving like a corpus-only build.
            if not contexts:
                if self.tavily.enabled:
                    web_docs = self._web_search_fallback(effective_query)
                    if web_docs:
                        for i, c in enumerate(web_docs, start=1):
                            c["rank"] = i
                        contexts = web_docs
                        used_web_search = True
            else:
                verdict = self._grade_contexts(effective_query, contexts, cost)
                crag_score = verdict.avg_score
                if verdict.needs_web and self.tavily.enabled:
                    # Mix mode: drop the low-score DB chunks, add web docs.
                    # The per-chunk filter is load-bearing HERE so the
                    # mixed list to the generator stays high-signal.
                    chunks_filtered = len(verdict.filtered)
                    web_docs = self._web_search_fallback(effective_query)
                    merged = list(verdict.kept) + list(web_docs)
                    if merged:
                        for i, c in enumerate(merged, start=1):
                            c["rank"] = i
                        contexts = merged
                    else:
                        contexts = []
                    used_web_search = bool(web_docs)
                else:
                    # DB-sufficient (or Tavily off): keep ALL 10 reranked
                    # chunks. Grader per-chunk scores are informational --
                    # filtering here would drop legitimate Cyrillic-dict
                    # entries that the grader scored conservatively (the
                    # blond p.25 case), which is exactly the reranker
                    # repair the demo relies on.
                    chunks_filtered = 0
            if not contexts:
                answer = "No context retrieved for this query."
            else:
                query_type = self._classify(effective_query, cost)
                prompt = build_prompt(query_type, effective_query, contexts)
                res = gemini_generate_guarded(
                    self.gemini, self.s.gen_model, prompt,
                    config=self._gen_config,
                    max_attempts=self.s.gemini_max_attempts,
                    call_timeout_s=self.s.gemini_call_timeout_s,
                    label="generate",
                )
                cost.add_gemini(self.s.gen_model, res["prompt_tokens"], res["output_tokens"])
                answer = res["text"]
                # CRAG safety net: if the corpus chunks looked good enough to
                # skip web (grader max just above crag_threshold) but the
                # generator still refused, do a one-shot Tavily retry. Without
                # this a borderline-retrievable query (e.g. lowercase "was ist
                # schwarz auf russisch") deterministically dead-ends on a
                # refusal even though the web can answer it.
                if (_is_refusal(answer) and not used_web_search
                        and self.tavily.enabled):
                    web_docs = self._web_search_fallback(effective_query)
                    if web_docs:
                        merged = list(contexts) + list(web_docs)
                        for i, c in enumerate(merged, start=1):
                            c["rank"] = i
                        contexts = merged
                        used_web_search = True
                        prompt = build_prompt(query_type, effective_query, contexts)
                        res = gemini_generate_guarded(
                            self.gemini, self.s.gen_model, prompt,
                            config=self._gen_config,
                            max_attempts=self.s.gemini_max_attempts,
                            call_timeout_s=self.s.gemini_call_timeout_s,
                            label="generate_retry",
                        )
                        cost.add_gemini(self.s.gen_model,
                                        res["prompt_tokens"], res["output_tokens"])
                        answer = res["text"]
                # Tag the generate span with the template versions that
                # produced this answer + record Gemini usage. The generate call
                # itself isn't a separate Opik span (would clutter the tree),
                # but its metadata lives on the parent query span via the tags.
                from app.prompts.registry import template_tags_for  # noqa: PLC0415
                obs.update_span(
                    tags=template_tags_for(query_type),
                    usage={
                        "prompt_tokens": res["prompt_tokens"],
                        "completion_tokens": res["output_tokens"],
                        "total_tokens": res["prompt_tokens"] + res["output_tokens"],
                    },
                )
            cache_hit = False
            # Cache successful answers, including web-fallback ones. The
            # trade-off (potentially stale URLs) was deliberately accepted
            # 2026-06-26 -- the user expects "repeat = instant", and a personal
            # language-learner tolerates the occasional stale Tavily link far
            # better than re-paying retrieval + LLM cost on a repeat query.
            # Refusals are NEVER cached: a single unlucky run must not freeze a
            # wrong "no content" answer for every future similar query.
            if (self.cache.enabled and use_cache and contexts
                    and not _is_refusal(answer)):
                self.cache.store(effective_query, dense, answer, contexts)

        # OutputValidator before persistence + return. The redacted
        # text is what we cache + remember (so future hits stay clean too).
        answer, redactions = self._validate_output(answer)
        # Persist the turn AFTER the answer succeeds. Use the ORIGINAL query
        # so the rewriter on the next turn sees what the user actually typed.
        if session_id:
            self.memory.add_turn(session_id, text, answer)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        cost_break = cost.to_breakdown()
        # Final trace-level metadata so the Opik UI shows total cost+latency at
        # the top of the trace, not buried in a child span.
        obs.update_trace(metadata={
            "latency_ms": latency_ms,
            "cost_usd": cost_break.total_usd,
            "cache_hit": cache_hit,
            "was_rewritten": was_rewritten,
            "query_type": query_type,
            "crag_score": crag_score,
            "used_web_search": used_web_search,
            "chunks_filtered": chunks_filtered,
        })
        return {
            "answer": answer,
            "contexts": [ContextItem(**c) for c in contexts],
            "model": self.s.gen_model,
            "collection": self.s.qdrant_collection,
            "latency_ms": latency_ms,
            "cost": cost_break,
            "rewritten_query": effective_query if was_rewritten else None,
            "was_rewritten": was_rewritten,
            "cache_hit": cache_hit,
            "used_web_search": used_web_search,
            "crag_score": crag_score,
            "chunks_filtered": chunks_filtered,
            "query_type": query_type,
            "trace_id": obs.get_current_trace_id(),
            "redactions": redactions,
        }

    @obs.safe_track(name="query_stream")
    async def query_stream(
        self,
        text: str,
        session_id: str | None = None,
        use_cache: bool = True,
    ) -> AsyncIterator[dict]:
        """Async generator yielding event dicts for sse-starlette EventSourceResponse.

        Event types (the frontend dispatches on these):
            - "rewrite":          query was canonicalised
            - "cache_hit":        semantic cache hit -- skipping retrieval +
                                  generation
            - "grader":           CRAG verdict -- emitted ONLY on
                                  cache miss with non-empty DB retrieval.
                                  Payload: {avg_score, kept, filtered,
                                  needs_web, threshold}.
            - "web_search_start": Tavily fallback fired -- payload
                                  {query}.
            - "web_contexts":     Tavily results -- payload
                                  {contexts: [...]} (also included in the
                                  final 'contexts' mix below).
            - "contexts":         FINAL mixed context list (DB kept + Tavily),
                                  sent ONCE before any token.
            - "route":            query classified into FACTUAL/EXPLANATORY/
                                  COMPARATIVE -- NOT emitted on
                                  cache hits.
            - "token":            one chunk of the answer (cache hits send
                                  the whole cached answer as a single token).
            - "redacted":         OutputValidator applied PII redactions to
                                  the streamed answer; frontend replaces the
                                  displayed body with redacted_text.
            - "done":             final metadata (latency, cost, cache_hit,
                                  query_type, trace_id, crag_score,
                                  used_web_search, chunks_filtered).
            - "blocked":          InputGuard fired -- terminal.
            - "error":            pipeline failure.
        """
        t0 = time.perf_counter()
        cost = CostTracker()
        obs.update_trace(
            metadata={"session_id": session_id, "original_query": text},
        )
        # Capture trace_id once we are inside the trace context -- propagated
        # safely across asyncio.to_thread because Python 3.7+ contextvars copy.
        trace_id = obs.get_current_trace_id()
        # InputGuard before anything else (yield structured error
        # event instead of raising so sse-starlette can close the stream
        # cleanly -- raising mid-stream confuses the EventSourceResponse).
        ig = self.input_guard.scan(text)
        obs.update_span(metadata={"input_guard_verdict": ig.verdict,
                                  "input_guard_patterns": ig.matched_patterns})
        if ig.verdict == "blocked":
            yield _sse_event("blocked", {
                "reason": "prompt_injection",
                "pattern": ig.matched_patterns[0],
                "trace_id": trace_id,
            })
            return
        # Rewrite (off-thread because it does a sync Gemini call).
        effective_query, was_rewritten = await asyncio.to_thread(
            self._maybe_rewrite, text, session_id, cost,
        )
        if was_rewritten:
            yield _sse_event("rewrite", {
                "original": text,
                "rewritten": effective_query,
            })

        # Embed once -- reused by cache.lookup AND, on a miss, by retrieve.
        try:
            dense = await asyncio.to_thread(self._embed_query, effective_query, cost)
        except Exception as exc:  # noqa: BLE001
            await self._remember_failed_turn(session_id, text)
            yield _sse_event("error", {"message": f"embed failed: {exc}"})
            return

        cache_hit_entry = None
        if self.cache.enabled and use_cache:
            cache_hit_entry = await asyncio.to_thread(
                self.cache.lookup, dense, effective_query,
            )

        if cache_hit_entry is not None:
            cached_contexts = [_with_origin(c, "cache")
                               for c in cache_hit_entry["contexts"]]
            cached_answer = cache_hit_entry["answer"]
            # Even cached answers get re-validated -- patterns may
            # have evolved since the entry was stored. Cheap, no LLM.
            cached_answer, cached_redactions = await asyncio.to_thread(
                self._validate_output, cached_answer,
            )
            log.info("stream cache HIT: query='%s' distance=%.4f",
                     effective_query[:60], cache_hit_entry["distance"])
            yield _sse_event("cache_hit", {
                "distance": cache_hit_entry["distance"],
                "cached_query": cache_hit_entry["query_text"],
            })
            yield _sse_event("contexts", {
                "contexts": [ContextItem(**c).model_dump() for c in cached_contexts],
                "model": self.s.gen_model,
                "collection": self.s.qdrant_collection,
            })
            yield _sse_event("token", {"text": cached_answer})
            if session_id:
                await asyncio.to_thread(
                    self.memory.add_turn, session_id, text, cached_answer,
                )
            latency_ms = int((time.perf_counter() - t0) * 1000)
            obs.update_trace(metadata={
                "latency_ms": latency_ms, "cache_hit": True,
                "cost_usd": cost.to_breakdown().total_usd,
            })
            yield _sse_event("done", {
                "latency_ms": latency_ms,
                "cost": cost.to_breakdown().model_dump(),
                "cache_hit": True,
                "query_type": None,
                "trace_id": trace_id,
                "redactions": cached_redactions,
                "crag_score": None,
                "used_web_search": False,
                "chunks_filtered": 0,
            })
            return

        try:
            contexts = await asyncio.to_thread(
                self._retrieve_with_embedding, effective_query, dense, cost,
            )
        except Exception as exc:  # noqa: BLE001
            await self._remember_failed_turn(session_id, text)
            yield _sse_event("error", {"message": f"retrieval failed: {exc}"})
            return

        # DocumentGuard between retrieve and generate.
        contexts = await asyncio.to_thread(self._filter_documents, contexts)

        # CRAG critic + Tavily fallback (mirror of sync `query`).
        # Emitted SSE order is: grader (with verdict) -> web_search_start
        # (only if needs_web) -> web_contexts (only after Tavily returns) ->
        # contexts (final mix). The Inspector updates Grader + WebSearch
        # cards live from these events; the chat SourceCards consume the
        # final 'contexts' event as today.
        crag_score: float | None = None
        chunks_filtered = 0
        used_web_search = False
        if not contexts:
            # No DB chunks survived retrieve+filter. If Tavily is on, skip
            # the grader entirely (nothing to grade) and go direct to web.
            if self.tavily.enabled:
                yield _sse_event("web_search_start", {"query": effective_query})
                web_docs = await asyncio.to_thread(
                    self._web_search_fallback, effective_query,
                )
                if web_docs:
                    for i, c in enumerate(web_docs, start=1):
                        c["rank"] = i
                    contexts = web_docs
                    used_web_search = True
                yield _sse_event("web_contexts", {
                    "contexts": [ContextItem(**c).model_dump() for c in web_docs],
                })
        else:
            verdict = await asyncio.to_thread(
                self._grade_contexts, effective_query, contexts, cost,
            )
            crag_score = verdict.avg_score
            yield _sse_event("grader", {
                "avg_score":  verdict.avg_score,
                "kept":       len(verdict.kept),
                "filtered":   len(verdict.filtered),
                "needs_web":  verdict.needs_web,
                "threshold":  self.s.crag_threshold,
            })
            if verdict.needs_web and self.tavily.enabled:
                # Mix mode: drop low-score DB chunks, add web docs. The
                # per-chunk filter is load-bearing HERE so the mixed list
                # to the generator stays high-signal.
                chunks_filtered = len(verdict.filtered)
                yield _sse_event("web_search_start", {"query": effective_query})
                web_docs = await asyncio.to_thread(
                    self._web_search_fallback, effective_query,
                )
                merged = list(verdict.kept) + list(web_docs)
                if merged:
                    for i, c in enumerate(merged, start=1):
                        c["rank"] = i
                    contexts = merged
                else:
                    contexts = []
                used_web_search = bool(web_docs)
                yield _sse_event("web_contexts", {
                    "contexts": [ContextItem(**c).model_dump() for c in web_docs],
                })
            else:
                # DB-sufficient (or Tavily off): keep ALL 10 reranked chunks.
                # Per-chunk scores are informational only -- filtering would
                # drop legitimate Cyrillic-dict entries that the grader
                # scored conservatively (the blond p.25 case), which is the
                # reranker repair the demo relies on.
                chunks_filtered = 0

        yield _sse_event("contexts", {
            "contexts": [ContextItem(**c).model_dump() for c in contexts],
            "model": self.s.gen_model,
            "collection": self.s.qdrant_collection,
        })

        if not contexts:
            answer = "No context retrieved for this query."
            yield _sse_event("token", {"text": answer})
            if session_id:
                await asyncio.to_thread(self.memory.add_turn, session_id, text, answer)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            obs.update_trace(metadata={
                "latency_ms": latency_ms, "cache_hit": False,
                "cost_usd": cost.to_breakdown().total_usd,
            })
            yield _sse_event("done", {
                "latency_ms": latency_ms,
                "cost": cost.to_breakdown().model_dump(),
                "cache_hit": False,
                "query_type": None,
                "trace_id": trace_id,
                "crag_score": crag_score,
                "used_web_search": used_web_search,
                "chunks_filtered": chunks_filtered,
            })
            return

        # Classify + open a trace span (via self._classify, decorated).
        query_type = await asyncio.to_thread(self._classify, effective_query, cost)
        yield _sse_event("route", {"query_type": query_type})

        prompt = build_prompt(query_type, effective_query, contexts)
        full_answer_chunks: list[str] = []
        last_in_tokens = last_out_tokens = 0
        try:
            async for chunk_text, in_tokens, out_tokens in _gemini_stream(
                self.gemini, self.s.gen_model, prompt,
                self.s.gemini_call_timeout_s,
                max_attempts=self.s.gemini_max_attempts,
                label="generate",
                config=self._gen_config,
            ):
                if chunk_text:
                    full_answer_chunks.append(chunk_text)
                    yield _sse_event("token", {"text": chunk_text})
                if in_tokens or out_tokens:
                    cost.add_gemini(self.s.gen_model, in_tokens, out_tokens)
                    last_in_tokens, last_out_tokens = in_tokens, out_tokens
        except Exception as exc:  # noqa: BLE001
            await self._remember_failed_turn(session_id, text)
            yield _sse_event("error", {"message": f"generation failed: {exc}"})
            return

        full_answer = "".join(full_answer_chunks)
        # CRAG safety net (mirror of sync query()): the grader let us skip web
        # (max just above crag_threshold) but the generator still refused ->
        # one-shot Tavily retry. We emit 'answer_reset' so the frontend clears
        # the streamed refusal, then stream the corrected answer in its place.
        if (_is_refusal(full_answer) and not used_web_search
                and self.tavily.enabled):
            yield _sse_event("web_search_start", {"query": effective_query})
            web_docs = await asyncio.to_thread(
                self._web_search_fallback, effective_query,
            )
            if web_docs:
                merged = list(contexts) + list(web_docs)
                for i, c in enumerate(merged, start=1):
                    c["rank"] = i
                contexts = merged
                used_web_search = True
                yield _sse_event("web_contexts", {
                    "contexts": [ContextItem(**c).model_dump() for c in web_docs],
                })
                yield _sse_event("contexts", {
                    "contexts": [ContextItem(**c).model_dump() for c in contexts],
                    "model": self.s.gen_model,
                    "collection": self.s.qdrant_collection,
                })
                yield _sse_event("answer_reset", {})
                prompt = build_prompt(query_type, effective_query, contexts)
                retry_chunks: list[str] = []
                try:
                    async for chunk_text, in_tokens, out_tokens in _gemini_stream(
                        self.gemini, self.s.gen_model, prompt,
                        self.s.gemini_call_timeout_s,
                        max_attempts=self.s.gemini_max_attempts,
                        label="generate_retry",
                        config=self._gen_config,
                    ):
                        if chunk_text:
                            retry_chunks.append(chunk_text)
                            yield _sse_event("token", {"text": chunk_text})
                        if in_tokens or out_tokens:
                            cost.add_gemini(self.s.gen_model, in_tokens, out_tokens)
                            last_in_tokens, last_out_tokens = in_tokens, out_tokens
                except Exception as exc:  # noqa: BLE001
                    await self._remember_failed_turn(session_id, text)
                    yield _sse_event("error", {"message": f"generation failed: {exc}"})
                    return
                full_answer = "".join(retry_chunks)
        # Tag the parent trace with template versions + record usage on
        # the parent span (the streamed generation has no dedicated child span).
        from app.prompts.registry import template_tags_for  # noqa: PLC0415
        obs.update_span(
            tags=template_tags_for(query_type),
            usage={
                "prompt_tokens": last_in_tokens,
                "completion_tokens": last_out_tokens,
                "total_tokens": last_in_tokens + last_out_tokens,
            } if (last_in_tokens or last_out_tokens) else None,
        )

        # OutputValidator on the full assembled answer. If
        # redactions were applied, yield a 'redacted' event so the frontend
        # can replace the streamed body. Caching + memory both store the
        # *redacted* version so future hits stay clean.
        full_answer, redactions = await asyncio.to_thread(
            self._validate_output, full_answer,
        )
        if redactions:
            yield _sse_event("redacted", {
                "redacted_text": full_answer,
                "redactions":    redactions,
            })

        # Persist the turn AFTER the stream completes successfully. Use the
        # ORIGINAL query so the next turn's rewriter sees the user's actual text.
        if session_id:
            await asyncio.to_thread(
                self.memory.add_turn, session_id, text, full_answer,
            )
        # Cache after generation succeeds (mirrors the sync query() ordering).
        # When use_cache=False we also skip the write, otherwise turning the
        # toggle back ON would suddenly serve a stale entry the user thought
        # had been bypassed. Web-fallback answers ARE cached (decision
        # 2026-06-26 -- see sync query() for the rationale). Refusals are
        # NEVER cached (see _is_refusal) so an unlucky run can't freeze a
        # wrong "no content" answer for every future similar query.
        if (self.cache.enabled and use_cache and contexts
                and not _is_refusal(full_answer)):
            await asyncio.to_thread(
                self.cache.store, effective_query, dense, full_answer, contexts,
            )

        latency_ms = int((time.perf_counter() - t0) * 1000)
        cost_break = cost.to_breakdown()
        obs.update_trace(metadata={
            "latency_ms": latency_ms,
            "cost_usd": cost_break.total_usd,
            "cache_hit": False,
            "was_rewritten": was_rewritten,
            "query_type": query_type,
            "crag_score": crag_score,
            "used_web_search": used_web_search,
            "chunks_filtered": chunks_filtered,
        })
        yield _sse_event("done", {
            "latency_ms": latency_ms,
            "cost": cost_break.model_dump(),
            "cache_hit": False,
            "query_type": query_type,
            "trace_id": trace_id,
            "redactions": redactions,
            "crag_score": crag_score,
            "used_web_search": used_web_search,
            "chunks_filtered": chunks_filtered,
        })


def _with_origin(ctx: dict, origin: str) -> dict:
    """Return a copy of ctx with origin overridden -- cache hits relabel every
    chunk from 'qdrant' to 'cache' so the frontend can show the cache badge."""
    out = dict(ctx)
    out["origin"] = origin
    return out


# Russian refusal markers spelled out via chr() so this source stays pure ASCII
# (some Windows toolchains still read source as cp1252).
def _ru_marker(*codes: int) -> str:
    return "".join(chr(c) for c in codes)


# Phrases that mark a generator "the context does not answer this" refusal.
# German markers use ASCII-only stems (no umlauts): "enthalten keine" covers
# "...Auszuege enthalten keine (direkte) Uebersetzung...", the exact shape the
# faithfulness rule in prompts/templates.py produces. Each phrase must be
# specific enough that a genuine answer almost never contains it; the length
# guard in _is_refusal() is the second line of defence.
_REFUSAL_MARKERS: tuple[str, ...] = (
    "no context retrieved",           # the hard empty-context sentinel
    "enthalten keine",                # excerpts contain no ...
    "enthalten nicht",                # excerpts do not contain ...
    "nicht enthalten",                # ... is NOT contained (reversed word order)
    "nicht enth",                     # truncation-safe stem of the above
    "enthalten leider keine",
    "nicht vorhanden",
    "keine ausreichenden",
    "nicht genug informationen",
    "keine informationen zu",
    "keine angaben",
    "keine direkte",
    "geben keine auskunft",
    "lassen sich nicht beantworten",
    "lasst sich nicht",
    "kann nicht beantwortet werden",
    "do not contain",
    "does not contain",
    "no information",
    _ru_marker(0x043D, 0x0435, 0x20, 0x0441, 0x043E, 0x0434, 0x0435, 0x0440, 0x0436),  # "ne soderzh(at)"
    _ru_marker(0x043D, 0x0435, 0x0434, 0x043E, 0x0441, 0x0442, 0x0430, 0x0442,
               0x043E, 0x0447, 0x043D, 0x043E),                                          # "nedostatochno"
    _ru_marker(0x043D, 0x0435, 0x0442, 0x20, 0x0438, 0x043D, 0x0444, 0x043E,
               0x0440, 0x043C, 0x0430, 0x0446, 0x0438, 0x0438),                          # "net informatsii"
)

_REFUSAL_MAX_CHARS = 400  # refusals are short; longer text is a real answer

# Placeholder stored in session memory when a turn died before producing an
# answer. Only the rewriter ever reads it, and it needs to see that the prior
# turn exists but failed. ASCII-only, like everything else in this file.
_FAILED_TURN_NOTE = "(Antwort konnte nicht generiert werden.)"


def _is_refusal(answer: str) -> bool:
    """True iff ``answer`` looks like a context-insufficient refusal.

    Used to (a) keep refusals OUT of the semantic cache -- otherwise a single
    unlucky run freezes a wrong "no content" answer for every future
    semantically-similar query -- and (b) trigger a one-shot Tavily web retry
    when the corpus fell short but the CRAG max-score sat just above threshold
    so no web fallback fired. Conservative on purpose: a false positive only
    costs a cache-miss or one extra web call, never a wrong answer.
    """
    if not answer:
        return True
    head = answer.strip().lower()[:_REFUSAL_MAX_CHARS]
    if len(answer.strip()) > _REFUSAL_MAX_CHARS:
        # A long, substantive answer that merely *mentions* a gap is not a
        # refusal; only short answers qualify.
        return any(m in head for m in (
            "no context retrieved", "enthalten keine", "nicht enthalten"))
    return any(m in head for m in _REFUSAL_MARKERS)


def _sse_event(event: str, data: dict) -> dict:
    """sse-starlette consumes {event, data}; we json-encode the payload here
    with ensure_ascii, so Cyrillic never reaches the wire as raw UTF-8 and no
    intermediate hop can mangle it."""
    return {"event": event, "data": json.dumps(data, ensure_ascii=True)}


async def _gemini_stream(
    client, model: str, prompt: str, call_timeout_s: float,
    max_attempts: int = 12, label: str = "generate", config=None,
) -> AsyncIterator[tuple[str, int, int]]:
    """Wrap sync Gemini streaming in async via to_thread + asyncio.Queue.

    Carries the same retry pattern as ``gemini_generate_guarded``: if
    the upstream call dies BEFORE any token reached the client (typical
    5xx "high demand" or watchdog timeout on the first chunk), back off and
    re-open the stream. Up to ``max_attempts``, exponential backoff with
    upstream retry-hint parsing. Once we've already emitted real text to
    the caller we MUST surface the failure -- restarting mid-stream would
    show the user duplicated / contradictory output.

    This closes a gap found in container smoke testing: sync `query()` rode
    out a 5xx wave via `gemini_generate_guarded`, while the SSE stream died on
    attempt 1.
    """
    last_err = "unknown"
    for attempt in range(1, max_attempts + 1):
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, int, int] | None | Exception] = asyncio.Queue()
        any_tokens_emitted = False

        def _produce() -> None:
            try:
                stream = client.models.generate_content_stream(
                    model=model, contents=prompt, config=config,
                )
                for chunk in stream:
                    txt = getattr(chunk, "text", "") or ""
                    um = getattr(chunk, "usage_metadata", None)
                    in_t = getattr(um, "prompt_token_count", 0) or 0 if um else 0
                    out_t = ((getattr(um, "candidates_token_count", 0) or 0)
                             + (getattr(um, "thoughts_token_count", 0) or 0)) if um else 0
                    loop.call_soon_threadsafe(queue.put_nowait, (txt, in_t, out_t))
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=_produce, daemon=True).start()
        try:
            while True:
                item = await asyncio.wait_for(queue.get(), timeout=call_timeout_s)
                if item is None:
                    return
                if isinstance(item, Exception):
                    raise item
                # Once we emit any real text the stream is past the point
                # of safe restart. We may still yield empty usage-only
                # chunks before the first text -- those don't count.
                if item[0]:
                    any_tokens_emitted = True
                yield item
            # `return` above is the normal exit; control should not reach here.
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            if any_tokens_emitted:
                # Mid-stream failure -- restart would duplicate output.
                raise
            if attempt >= max_attempts:
                raise RuntimeError(
                    f"{label}: all {max_attempts} stream attempts failed. "
                    f"Last: {last_err}"
                ) from exc
            hint = _retry_hint_seconds(last_err)
            wait = (hint + 2) if hint else min(60.0, 5 * 2 ** (attempt - 1))
            log.warning(
                "%s: stream attempt %d/%d failed (%s); retry in %.0fs",
                label, attempt, max_attempts, last_err[:110], wait,
            )
            await asyncio.sleep(wait)
