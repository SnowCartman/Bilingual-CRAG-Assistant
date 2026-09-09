"""Smoke -- CRAG critic + Tavily web-search fallback end-to-end.

Drives RAGPipeline.query() directly across three scenarios that exercise the
three CRAG paths plus the cache-skip:

  (1) DB-sufficient: "Was ist Konjunktiv I?"
        -> grader avg_score >= 0.5 -> DB-only generate -> used_web_search=False
  (2) DB-empty / corpus-miss: "Wie heisst 'Guten Morgen' auf Schwedisch?"
        -> grader avg_score < 0.5 -> Tavily fallback -> used_web_search=True,
           mixed or full-web contexts (each ContextItem.origin == "tavily").
  (3) CACHE SKIP: REPEAT scenario (1)
        -> cache hit (sub-500ms) -> CRAG entirely skipped:
           crag_score=None, chunks_filtered=0, used_web_search=False.

Run from the repo root:
    uv run python scripts/crag_smoke.py

Estimated cost: ~$0.03-0.05.
  - Scenario 1: 1 embed + 1 retrieve + 1 grader (flash, ~10 chunks)
                + 1 generate. ~$0.012.
  - Scenario 2: 1 embed + 1 retrieve + 1 grader + 1 Tavily (free tier)
                + 1 generate on mixed/web contexts. ~$0.015.
  - Scenario 3: 1 embed + 1 cache HIT -> short-circuit. ~$0.0001.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings                                # noqa: E402
from app.services import observability as obs                      # noqa: E402
from app.services.rag_pipeline import RAGPipeline                  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("crag_smoke")


def _print_header(num: int, title: str) -> None:
    print()
    print("=" * 78)
    print(f"  SCENARIO {num}: {title}")
    print("=" * 78)


def _print_summary(result: dict) -> None:
    contexts = result["contexts"]
    origins = sorted({getattr(c, "origin", "?") for c in contexts})
    web_n = sum(1 for c in contexts if getattr(c, "origin", "") == "tavily")
    db_n = sum(1 for c in contexts if getattr(c, "origin", "") == "qdrant")
    cache_n = sum(1 for c in contexts if getattr(c, "origin", "") == "cache")
    print(f"  cache_hit         : {result['cache_hit']}")
    print(f"  used_web_search   : {result['used_web_search']}")
    print(f"  crag_score        : {result['crag_score']}")
    print(f"  chunks_filtered   : {result['chunks_filtered']}")
    print(f"  query_type        : {result['query_type']}")
    print(f"  contexts          : total={len(contexts)} db={db_n} web={web_n} cache={cache_n}")
    print(f"  context origins   : {origins}")
    print(f"  latency_ms        : {result['latency_ms']}")
    print(f"  cost_usd          : {result['cost'].total_usd:.6f}")
    print(f"  trace_id          : {result['trace_id']}")
    if web_n:
        print("  web sources:")
        for c in contexts:
            if getattr(c, "origin", "") == "tavily":
                url = getattr(c, "url", None) or "?"
                title = getattr(c, "source_file", "?")
                print(f"    [{c.rank}] {title}  ({url})")
    print("  answer head       :")
    head = result["answer"][:280].replace("\n", " ")
    print(f"    {head}{'...' if len(result['answer']) > 280 else ''}")


def _validate(scenario: int, result: dict, expectations: dict) -> bool:
    """Print expectations + actual; return True iff every check passes."""
    print()
    print("  Validation:")
    ok = True
    for key, expected in expectations.items():
        actual = result.get(key)
        if callable(expected):
            verdict = expected(actual)
            check = "OK" if verdict else "FAIL"
        else:
            verdict = actual == expected
            check = "OK" if verdict else "FAIL"
        if not verdict:
            ok = False
        print(f"    [{check}] {key}: expected={expected!r}  actual={actual!r}")
    return ok


def main() -> int:
    settings = get_settings()
    obs.configure_opik(settings)
    pipeline = RAGPipeline(settings)
    session_id = "crag_smoke_session"
    failures = 0

    # --- Scenario 1: DB-sufficient -----------------------------------------
    _print_header(1, "Konjunktiv I -- corpus-strong -> DB-only")
    q1 = "Was ist Konjunktiv I und wann wird er verwendet?"
    res1 = pipeline.query(q1, session_id=session_id, use_cache=True)
    _print_summary(res1)
    ok1 = _validate(1, res1, {
        "cache_hit":       False,
        "used_web_search": False,
        # avg per-chunk score should be solidly above 0.5 for a well-covered
        # textbook topic; allow some headroom for grader variance.
        "crag_score":      lambda v: isinstance(v, float) and v >= 0.5,
    })
    if not ok1:
        failures += 1

    # --- Scenario 2: DB-miss -> Tavily fallback ----------------------------
    _print_header(2, "Schwedisch -> Tavily fallback expected")
    q2 = "Wie heisst 'Guten Morgen' auf Schwedisch?"
    res2 = pipeline.query(q2, session_id=session_id + "_swe", use_cache=True)
    _print_summary(res2)
    ok2 = _validate(2, res2, {
        "cache_hit":       False,
        "used_web_search": True,
        "crag_score":      lambda v: isinstance(v, float) and v < 0.5,
    })
    # Additionally: at least one context with origin == "tavily".
    web_n = sum(
        1 for c in res2["contexts"] if getattr(c, "origin", "") == "tavily"
    )
    extra_ok = web_n >= 1
    print(f"    [{'OK' if extra_ok else 'FAIL'}] web_contexts_present: "
          f"expected>=1  actual={web_n}")
    if not (ok2 and extra_ok):
        failures += 1

    # --- Scenario 3: REPEAT Q1 -> cache hit -> CRAG skipped ----------------
    _print_header(3, "REPEAT Konjunktiv I -> cache hit -> CRAG skipped")
    res3 = pipeline.query(q1, session_id=session_id + "_repeat", use_cache=True)
    _print_summary(res3)
    ok3 = _validate(3, res3, {
        "cache_hit":       True,
        "used_web_search": False,
        "crag_score":      None,
        "chunks_filtered": 0,
    })
    if not ok3:
        failures += 1

    print()
    print("=" * 78)
    if failures == 0:
        print(f"  ALL 3 SCENARIOS PASSED  -  CRAG smoke OK")
        rc = 0
    else:
        print(f"  {failures}/3 SCENARIOS FAILED  -  see [FAIL] lines above")
        rc = 1
    print("=" * 78)
    obs.flush()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
