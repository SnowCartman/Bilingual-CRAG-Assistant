"""Opik observability wiring.

Graceful-degradation contract identical to ``session_memory`` and
``semantic_cache``: if ``OPIK_API_KEY`` (or workspace / project) is missing
or ``opik.configure`` fails, the service still serves traffic; only the
tracing + feedback persistence become no-ops, and a single startup warning
is logged.

Public surface (kept small on purpose -- the pipeline imports only these):

    configure_opik(settings) -> bool       -- call once at FastAPI startup
    is_enabled() -> bool
    safe_track(name, type) -> decorator    -- conditional @opik.track wrapper
    update_span(metadata, tags, usage)     -- runtime span enrichment
    update_trace(metadata, tags)           -- runtime trace enrichment
    get_current_trace_id() -> str | None   -- expose UUID for QueryResponse
    record_feedback(...) -> {opik, jsonl}  -- /feedback handler delegate
    flush()                                -- optional, for tests/scripts
"""

from __future__ import annotations

import datetime
import functools
import inspect
import json
import logging
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("observability")

# Module-level flag flipped by configure_opik(). All public helpers short-
# circuit when False -- this is what gives us graceful degradation.
_ENABLED: bool = False
# Project name captured at configure-time; we re-use it as a default for
# safe_track() callers that don't override it (so traces always land in the
# my own workspace's RAG project, not Opik's "Default Project").
_PROJECT_NAME: str | None = None

# Where /feedback writes when Opik is unreachable. Lives in scripts/ because
# scripts/ is already the build tooling-included (we want the file in the repo
# if it ends up populated; otherwise the .jsonl is just missing -- fine).
FEEDBACK_LOG = (
    Path(__file__).resolve().parents[2] / "scripts" / "feedback.jsonl"
)


def configure_opik(settings) -> bool:
    """Configure Opik once at startup. Returns True iff tracing is wired."""
    global _ENABLED, _PROJECT_NAME
    if not (settings.opik_api_key and settings.opik_workspace):
        log.info(
            "Opik disabled (api_key=%s, workspace=%s)",
            bool(settings.opik_api_key), settings.opik_workspace,
        )
        return False
    try:
        import opik  # noqa: PLC0415
        opik.configure(
            api_key=settings.opik_api_key,
            workspace=settings.opik_workspace,
            use_local=False,
            force=True,    # overwrite ~/.opik.config -- predictable on Cloud Run
        )
        _ENABLED = True
        _PROJECT_NAME = settings.opik_project_name
        log.info("Opik enabled -- workspace=%s project=%s",
                 settings.opik_workspace, settings.opik_project_name)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("Opik configure failed (%s) -- continuing without tracing", exc)
        return False


def is_enabled() -> bool:
    return _ENABLED


def safe_track(
    name: str | None = None,
    *,
    type: str = "general",
    project_name: str | None = None,
) -> Callable[[Callable], Callable]:
    """Conditional ``@opik.track`` wrapper. Auto-handles sync, async, and
    async-generator targets (the latter is what ``query_stream`` needs).

    Why we don't apply ``opik.track`` unconditionally: at module-load time
    ``configure_opik`` has not run yet, and we don't want to *force* every
    deploy to have Opik credentials. The wrapper resolves the decision at
    function-CALL time -- if Opik is enabled, the target is lazily wrapped
    with the real ``opik.track`` (once, then cached); otherwise it runs
    untouched.
    """
    def deco(fn: Callable) -> Callable:
        wrapped: dict[str, Callable] = {}

        def _resolve() -> Callable:
            if not _ENABLED:
                return fn
            if "fn" not in wrapped:
                import opik  # noqa: PLC0415
                wrapped["fn"] = opik.track(
                    name=name,
                    type=type,
                    project_name=project_name or _PROJECT_NAME,
                )(fn)
            return wrapped["fn"]

        if inspect.isasyncgenfunction(fn):
            @functools.wraps(fn)
            async def call_agen(*args: Any, **kwargs: Any):
                target = _resolve()
                async for value in target(*args, **kwargs):
                    yield value
            return call_agen

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def call_coro(*args: Any, **kwargs: Any) -> Any:
                target = _resolve()
                return await target(*args, **kwargs)
            return call_coro

        @functools.wraps(fn)
        def call_sync(*args: Any, **kwargs: Any) -> Any:
            target = _resolve()
            return target(*args, **kwargs)
        return call_sync
    return deco


def update_span(
    *,
    metadata: dict | None = None,
    tags: list[str] | None = None,
    usage: dict | None = None,
) -> None:
    """Attach span-level info to the *current* span (no-op if not enabled)."""
    if not _ENABLED:
        return
    try:
        from opik import opik_context  # noqa: PLC0415
        kwargs: dict[str, Any] = {}
        if metadata is not None:
            kwargs["metadata"] = metadata
        if tags is not None:
            kwargs["tags"] = tags
        if usage is not None:
            kwargs["usage"] = usage
        if kwargs:
            opik_context.update_current_span(**kwargs)
    except Exception as exc:  # noqa: BLE001
        log.warning("update_span failed: %s", exc)


def update_trace(
    *,
    metadata: dict | None = None,
    tags: list[str] | None = None,
) -> None:
    """Attach trace-level info (no-op if not enabled)."""
    if not _ENABLED:
        return
    try:
        from opik import opik_context  # noqa: PLC0415
        kwargs: dict[str, Any] = {}
        if metadata is not None:
            kwargs["metadata"] = metadata
        if tags is not None:
            kwargs["tags"] = tags
        if kwargs:
            opik_context.update_current_trace(**kwargs)
    except Exception as exc:  # noqa: BLE001
        log.warning("update_trace failed: %s", exc)


def get_current_trace_id() -> str | None:
    """Return the active Opik trace UUID, or None if outside a trace / disabled."""
    if not _ENABLED:
        return None
    try:
        from opik import opik_context  # noqa: PLC0415
        trace = opik_context.get_current_trace_data()
        return trace.id if trace is not None else None
    except Exception:
        return None


def record_feedback(
    *,
    trace_id: str | None,
    rating: str,
    note: str | None,
    session_id: str,
    message_id: str,
) -> dict:
    """Persist a feedback rating. Always writes the local jsonl (never lose
    data), and additionally calls the Opik client when enabled + we have a
    trace_id to attach to. Returns {opik: bool, jsonl: bool}.
    """
    result = {"opik": False, "jsonl": False}
    # Convention: thumbs-up = 1.0, thumbs-down = 0.0. Float kept simple so
    # Opik's aggregations work directly.
    value = 1.0 if rating == "up" else 0.0

    if _ENABLED and trace_id:
        try:
            import opik  # noqa: PLC0415
            # project_name is REQUIRED for the feedback score to attach to the
            # right project's trace. Without it, opik 2.x routes the score to
            # "Default Project" and the user-facing UI shows nothing under
            # our RAG project. We pass it both per-score (the BatchFeedback-
            # ScoreDict expects it) and as the method-level default.
            client = opik.Opik(project_name=_PROJECT_NAME) if _PROJECT_NAME else opik.Opik()
            client.log_traces_feedback_scores(
                scores=[{
                    "id": trace_id,
                    "name": "user_feedback",
                    "value": value,
                    "reason": note or rating,
                    "project_name": _PROJECT_NAME,
                }],
                project_name=_PROJECT_NAME,
            )
            result["opik"] = True
        except Exception as exc:  # noqa: BLE001
            log.warning("Opik feedback log failed: %s", exc)

    # Local jsonl fallback -- ALWAYS write so we have an audit trail even when
    # Opik is unreachable. ensure_ascii=True keeps the file ASCII-safe.
    try:
        FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "trace_id": trace_id,
            "session_id": session_id,
            "message_id": message_id,
            "rating": rating,
            "value": value,
            "note": note,
        }
        with FEEDBACK_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=True) + "\n")
        result["jsonl"] = True
    except Exception as exc:  # noqa: BLE001
        log.warning("Feedback jsonl write failed: %s", exc)

    return result


def flush() -> None:
    """Force-send buffered traces. Useful in scripts; not required in FastAPI."""
    if not _ENABLED:
        return
    try:
        import opik  # noqa: PLC0415
        opik.flush_tracker()
    except Exception as exc:  # noqa: BLE001
        log.warning("flush failed: %s", exc)
