"""FastAPI entry point for the Bilingual CRAG Assistant.

Exposes the RAG pipeline over HTTP:

    /health             liveness probe
    /query              synchronous query (full answer + metadata)
    /query/stream       SSE stream of pipeline events + answer tokens
    /feedback           thumbs up/down with an optional note
    /admin/cache/clear  drop the semantic cache

Run locally from the repo root:
    uv run uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from app.config import Settings, get_settings
from app.models import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from app.services import observability as obs
from app.services.rag_pipeline import RAGPipeline
from app.services.security import PromptInjectionBlocked

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the (expensive) RAGPipeline singleton at startup; tear down on stop."""
    settings: Settings = get_settings()
    # Configure Opik BEFORE building the pipeline so the very first
    # @safe_track-decorated method already sees _ENABLED=True. If the keys are
    # missing or the network is unreachable, configure_opik logs a warning and
    # returns False -- the service still serves traffic, just untraced.
    obs.configure_opik(settings)
    log.info("startup: building RAGPipeline (collection=%s, gen_model=%s)",
             settings.qdrant_collection, settings.gen_model)
    app.state.settings = settings
    app.state.pipeline = RAGPipeline(settings)
    log.info("startup: ready (opik=%s)", "on" if obs.is_enabled() else "off")
    yield
    obs.flush()
    log.info("shutdown")


app = FastAPI(
    title="Bilingual CRAG Assistant",
    version=get_settings().app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    s: Settings = request.app.state.settings
    return HealthResponse(
        status="ok",
        version=s.app_version,
        embed_model=s.embed_model,
        rerank_model=s.rerank_model,
        gen_model=s.gen_model,
        collection=s.qdrant_collection,
    )


def _detect_query_lang(text: str) -> str:
    """Naive language detector for security error messages: Cyrillic -> ru,
    else default to de (our primary user base). Good enough for picking the
    right phrasing of the prompt-injection block reply."""
    return "ru" if any(0x0400 <= ord(c) <= 0x04FF for c in text) else "de"


# Localized prompt-injection error messages. RU phrasing is romanized to keep
# this source file ASCII-clean (frontend can render the real RU on its side);
# de + en are native.
_INJECTION_MSG = {
    "de": "Deine Anfrage wurde durch die Sicherheitspruefung blockiert (Verdacht auf Prompt-Injection).",
    "ru": "Vash zapros zablokirovan po prichine podozreniya na prompt-injection. (Bitte normales DE/RU verwenden.)",
    "en": "Your query was blocked by the security check (suspected prompt injection).",
}


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest, request: Request) -> QueryResponse:
    pipeline: RAGPipeline = request.app.state.pipeline
    try:
        # Pipeline calls block on HTTP + Gemini watchdog; run off the event loop.
        import asyncio
        result = await asyncio.to_thread(
            pipeline.query, req.query, req.session_id, req.use_cache,
        )
    except PromptInjectionBlocked as exc:
        lang = _detect_query_lang(req.query)
        log.info("input blocked by InputGuard (pattern=%s, lang=%s)",
                 exc.pattern_name, lang)
        raise HTTPException(status_code=422, detail={
            "code":    "prompt_injection_blocked",
            "pattern": exc.pattern_name,
            "message": _INJECTION_MSG.get(lang, _INJECTION_MSG["en"]),
        })
    except Exception as exc:  # noqa: BLE001
        log.exception("query failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"pipeline error: {exc}")
    return QueryResponse(**result)


@app.post("/query/stream")
async def query_stream(req: QueryRequest, request: Request):
    """SSE event sequence:

      [blocked?] -> [rewrite?] -> [cache_hit?] -> 'contexts' -> [route?] ->
      'token'* -> [redacted?] -> 'done'

    - 'blocked':  InputGuard fired -- no further events. Carries pattern + trace_id.
    - 'redacted': OutputValidator applied PII redactions to the streamed answer;
                  frontend should replace the displayed body with redacted_text.
    """
    pipeline: RAGPipeline = request.app.state.pipeline
    return EventSourceResponse(
        pipeline.query_stream(req.query, req.session_id, req.use_cache),
    )


@app.post("/admin/cache/clear")
async def clear_cache(request: Request) -> dict:
    """Wipe every semantic-cache entry. No auth -- this app is a personal
    demo running on localhost; for a multi-tenant deploy this would need
    a token. Returns {enabled, cleared} so the caller can confirm the cache
    was actually present (not skipped due to missing Redis).
    """
    pipeline: RAGPipeline = request.app.state.pipeline
    if not pipeline.cache.enabled:
        return {"enabled": False, "cleared": False}
    try:
        import asyncio
        await asyncio.to_thread(pipeline.cache.clear)
    except Exception as exc:  # noqa: BLE001
        log.warning("clear_cache failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"cache clear failed: {exc}")
    return {"enabled": True, "cleared": True}


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest) -> FeedbackResponse:
    """Record a thumbs-up/down on a prior answer.

    Always writes a local jsonl audit line (so we never silently lose feedback).
    Additionally posts to Opik when observability is enabled and the request
    carries a trace_id from the original /query response.
    """
    result = obs.record_feedback(
        trace_id=req.trace_id,
        rating=req.rating,
        note=req.note,
        session_id=req.session_id,
        message_id=req.message_id,
    )
    return FeedbackResponse(**result)
