"""Smoke test -- query router + 3 templates + language-detection fix.

Drives RAGPipeline.query() directly (skips FastAPI / uvicorn) with 4 queries
covering all 3 router classes plus the cross-lingual language-detection case:

  1. FACTUAL     (DE) -- "Was ist Funktionsverbgefuege?"
  2. EXPLANATORY (DE) -- "Erklaer mir die Verwendung des Konjunktivs II."
  3. COMPARATIVE (DE) -- "Was ist der Unterschied zwischen Konjunktiv I und II?"
  4. EXPLANATORY (RU) -- "Chto takoe Konjunktiv I?" -- expects RU answer
     (demo 1a-style language-detection check; the old persona prompt sometimes
     produced DE on RU queries)

Queries 1-3 are different enough from the cache smoke set to avoid cache hits;
query 4 is brand new (Russian). For each call we log query_type + cache_hit +
the first 250 chars of the answer so I can eyeball language correctness.

Run from the repo root:
    uv run python scripts/router_smoke.py

Estimated cost: ~$0.05 (4 queries x retrieval + classify + generate).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# scripts/router_smoke.py -> scripts -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings           # noqa: E402
from app.services.rag_pipeline import RAGPipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("step6_smoke")


def _cyr(*codes: int) -> str:
    """ASCII-source helper for Cyrillic strings (same as query_rewriter)."""
    return "".join(chr(c) for c in codes)


# Russian for "What is Konjunktiv I?" -- transliteration: "Chto takoe Konjunktiv I?"
# Codepoints (so the source stays pure ASCII per ):
#   U+0427 U+0442 U+043E -- "Chto"
#   U+0442 U+0430 U+043A U+043E U+0435 -- "takoe"
_RU_QUERY = (
    _cyr(0x0427, 0x0442, 0x043E, 0x20, 0x0442, 0x0430, 0x043A, 0x043E, 0x0435)
    + " Konjunktiv I?"
)


QUERIES: list[tuple[str, str, str]] = [
    ("FACTUAL",     "DE", "Was ist Funktionsverbgefuege?"),
    ("EXPLANATORY", "DE", "Erklaer mir die Verwendung des Konjunktivs II."),
    ("COMPARATIVE", "DE", "Was ist der Unterschied zwischen Konjunktiv I und Konjunktiv II?"),
    ("EXPLANATORY", "RU", _RU_QUERY),
]


def main() -> int:
    settings = get_settings()
    log.info("building pipeline (collection=%s, router_model=%s)",
             settings.qdrant_collection, settings.router_model)
    pipeline = RAGPipeline(settings)

    total_cost = 0.0
    for i, (expected_class, lang, q) in enumerate(QUERIES, 1):
        log.info("---- query %d (%s expected, user lang=%s) ----", i, expected_class, lang)
        log.info("Q: %s", q)
        out = pipeline.query(q)
        ans = out["answer"]
        log.info("classified as: %s   (expected %s)", out["query_type"], expected_class)
        log.info("cache_hit:     %s", out["cache_hit"])
        log.info("latency_ms:    %d", out["latency_ms"])
        log.info("cost_usd:      %.6f", out["cost"].total_usd)
        log.info("answer (first 250 chars):\n%s", ans[:250])
        total_cost += out["cost"].total_usd

    log.info("==== total cost: $%.4f ====", total_cost)
    return 0


if __name__ == "__main__":
    sys.exit(main())
