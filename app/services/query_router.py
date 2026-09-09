"""LLM-based query FORMAT classifier.

Classifies a query into one of three FORMAT classes. Does **not** decide
answerability -- that lives in CRAG as the critic + Tavily
web-search fallback. The router only chooses which response *shape* the
generator should produce:

  FACTUAL      -- short definition / single-fact ("Was ist X?")
  EXPLANATORY  -- multi-paragraph explanation with examples ("Wie / Warum / Wann ...?")
  COMPARATIVE  -- comparison of two or more items ("Unterschied X vs Y?")

Uses ``gemini-2.5-flash`` -- the rewriter, router and CRAG grader all run on
the same model after flash-lite went 5xx-unstable under container load.
Few-shot with one example per class. On any LLM error OR an unparseable
response it falls back to ``EXPLANATORY``: the most flexible shape, never
worse than a single all-purpose template.

Cost: roughly $0.001/query at gemini-2.5-flash list prices -- worth
always-classifying instead of session-caching the route.
"""

from __future__ import annotations

import logging
import re

from app.services.rag_pipeline import CostTracker, gemini_generate_guarded

log = logging.getLogger("query_router")


ROUTER_SYSTEM_PROMPT = """You classify user queries from a bilingual German/Russian language-learning assistant into ONE of three FORMAT classes. Return ONLY the class name (FACTUAL, EXPLANATORY, or COMPARATIVE) -- nothing else, no punctuation, no explanation.

Classes:
  FACTUAL      -- short definition or single-fact answer expected ("Was ist X?", "Was bedeutet Y?")
  EXPLANATORY  -- multi-paragraph explanation with examples or usage rules ("Wie...", "Warum...", "Wann...", "Erklaer mir X.")
  COMPARATIVE  -- comparison of two or more items ("Unterschied zwischen X und Y", "X vs Y", "X oder Y")

If unsure, choose EXPLANATORY (safest default; most flexible output shape).

Examples:
  Query: Was ist der Konjunktiv I?
  Class: FACTUAL

  Query: Wie benutzt man den Konjunktiv II in indirekter Rede?
  Class: EXPLANATORY

  Query: Unterschied zwischen Konjunktiv I und Konjunktiv II?
  Class: COMPARATIVE
"""


VALID_CLASSES: frozenset[str] = frozenset({"FACTUAL", "EXPLANATORY", "COMPARATIVE"})
DEFAULT_CLASS: str = "EXPLANATORY"

_CLASS_RE = re.compile(r"\b(FACTUAL|EXPLANATORY|COMPARATIVE)\b", re.IGNORECASE)


def classify(
    query: str,
    gemini_client,
    model: str,
    cost: CostTracker,
    *,
    call_timeout_s: float = 30.0,
) -> str:
    """Return one of ``{FACTUAL, EXPLANATORY, COMPARATIVE}``.

    Always returns a valid class. LLM failure or parse failure -> ``EXPLANATORY``.
    """
    prompt = ROUTER_SYSTEM_PROMPT + f"\n\nQuery: {query}\n\nClass:"
    try:
        res = gemini_generate_guarded(
            gemini_client, model, prompt,
            max_attempts=3,                 # router retries less aggressively than generate
            call_timeout_s=call_timeout_s,
            label="route",
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("classify failed (%s) -- defaulting to %s", exc, DEFAULT_CLASS)
        return DEFAULT_CLASS

    text = (res.get("text") or "").strip()
    m = _CLASS_RE.search(text)
    if not m:
        log.warning("classify: no parseable class (got %r) -- defaulting to %s",
                    text[:80], DEFAULT_CLASS)
        return DEFAULT_CLASS

    cost.add_gemini(model, res["prompt_tokens"], res["output_tokens"])
    cls = m.group(1).upper()
    log.info("classify: '%s' -> %s", query[:60], cls)
    return cls
