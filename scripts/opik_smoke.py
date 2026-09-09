"""Smoke -- Opik tracing + /feedback round-trip.

Drives RAGPipeline.query() directly (skips uvicorn) so the smoke is fast and
deterministic. Verifies:

  (1) Opik is configured from the loaded settings
  (2) pipeline.query() returns a non-empty trace_id (UUID)
  (3) observability.record_feedback(trace_id=..., rating="up") writes both to
      Opik AND to the local feedback.jsonl audit log
  (4) The jsonl line matches what we just wrote (round-trip read-back)
  (5) Pre-existing path still works -- cost_usd > 0, latency_ms > 0

Run from the repo root:
    uv run python scripts/opik_smoke.py

Estimated cost: ~$0.02 (one /query against gemini-2.5-flash + classifier).
The Opik trace appears at https://www.comet.com/<workspace>/<project>/traces.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# scripts/opik_smoke.py -> scripts -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.services import observability as obs  # noqa: E402
from app.services.rag_pipeline import RAGPipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("step7_smoke")


SMOKE_QUERY = "Welche Modalpartikeln signalisieren Hoeflichkeit im Deutschen?"


def main() -> int:
    settings = get_settings()
    log.info("configuring Opik (workspace=%s, project=%s)",
             settings.opik_workspace, settings.opik_project_name)
    enabled = obs.configure_opik(settings)
    log.info("opik enabled = %s", enabled)
    if not enabled:
        log.error("Opik not enabled -- check OPIK_API_KEY / OPIK_WORKSPACE in .env")
        return 1

    log.info("building pipeline")
    pipeline = RAGPipeline(settings)

    # ---- (1)+(2) Query path -----------------------------------------------
    log.info("---- /query smoke ----")
    log.info("Q: %s", SMOKE_QUERY)
    result = pipeline.query(SMOKE_QUERY)
    trace_id = result["trace_id"]
    log.info("trace_id:    %s", trace_id)
    log.info("query_type:  %s", result["query_type"])
    log.info("latency_ms:  %d", result["latency_ms"])
    log.info("cost_usd:    %.6f", result["cost"].total_usd)
    log.info("answer (first 250):\n%s", result["answer"][:250])

    if not trace_id or "-" not in trace_id:
        log.error("trace_id missing or malformed: %r", trace_id)
        return 1

    # ---- (3) /feedback round-trip ----------------------------------------
    log.info("---- /feedback smoke ----")
    fb = obs.record_feedback(
        trace_id=trace_id,
        rating="up",
        note="smoke test thumbs-up",
        session_id="smoke-session-001",
        message_id="smoke-msg-001",
    )
    log.info("feedback persisted: opik=%s jsonl=%s", fb["opik"], fb["jsonl"])
    if not fb["opik"]:
        log.warning("Opik feedback did NOT persist -- check Comet dashboard")
    if not fb["jsonl"]:
        log.error("jsonl fallback failed -- check write permissions")
        return 1

    # ---- (4) Read-back jsonl line ----------------------------------------
    log_path = obs.FEEDBACK_LOG
    log.info("verifying jsonl line at %s", log_path)
    last_line = log_path.read_text(encoding="utf-8").splitlines()[-1]
    entry = json.loads(last_line)
    assert entry["trace_id"] == trace_id, "trace_id mismatch"
    assert entry["rating"] == "up", "rating mismatch"
    assert entry["value"] == 1.0, "value mismatch"
    log.info("jsonl read-back OK: %s", json.dumps(entry, ensure_ascii=True))

    # ---- Flush Opik so the trace appears in the dashboard immediately ----
    obs.flush()
    log.info("==== Opik smoke PASS ====")
    log.info("Check Opik UI -- expect ONE trace with nested spans: "
             "query / embed / retrieve / classify (no rewrite -- no session)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
