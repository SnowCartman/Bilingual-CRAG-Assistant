"""CRAG Context Grader.

Implements the **Corrective RAG** critic pass on retrieved contexts BEFORE the
generator runs. The grader scores each reranked chunk on a 0.0--1.0 relevance
scale against the user query; if the highest per-chunk score falls below the
threshold, the pipeline triggers the Tavily web-search fallback (see
``app/augmentations/web_search.py``).

Design choices:

- **Context-based grading, not answer-based.** A single LLM call rates the
  retrieved chunks; we decide BEFORE generating whether the corpus suffices.
  Cheaper (1 LLM call vs 2) and easier to reason about than answer-based
  critic patterns.

- **Deterministic threshold on the per-chunk MAX.** ``needs_web`` is purely
  ``max_score < settings.crag_threshold`` (0.45 after the 2026-06-27
  re-calibration that paired this with grader ``temperature=0``; was
  0.65 under noisy ``temperature=1`` defaults).
  We use MAX, not avg, because with the reranker dominating, retrieved
  sets typically look like ``[0.9, 0.2, 0.1, ...]`` -- avg would collapse
  to ~0.2 and needlessly fire Tavily even though the top chunk perfectly
  answers the query. We do NOT use the LLM's own meta-verdict -- a pure
  threshold is reproducible and trivially explainable in a demo.

- **gemini-2.5-flash, JSON-mode.** Standardised alongside the rewriter and
  router -- flash-lite was cheaper but went 5xx-unstable under container
  load; flash holds up at +$0.0004
  per request, acceptable for a demo workload. JSON-mode guarantees a
  parseable shape so we don't have to babysit free-form output.

- **Chunk text truncated to 600 chars in the grader prompt.** Reranker rank-1
  hits routinely run 1500+ chars; we keep the grader cheap by only showing
  the head. The reranker already separated useful-prefix from noise, so the
  head is representative for relevance scoring.

Skip-list (handled in ``rag_pipeline``, NOT here):
- cache hits (answer already trusted)
- empty-context refusals (Web is the only path -- skip grader, go direct)
- InputGuard blocks (no context to grade)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from google import genai
from google.genai import types as genai_types

from app.services.rag_pipeline import CostTracker, gemini_generate_guarded

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ChunkScore:
    """One grader judgement on one chunk."""

    rank: int            # 1-based, matches the reranker rank
    score: float         # 0.0 -- 1.0
    reason: str = ""     # short LLM reason, kept for the Inspector drawer


@dataclass
class GraderVerdict:
    """Verdict returned by ``ContextGrader.grade``.

    - ``kept``       = chunks with per-chunk score >= ``per_chunk_threshold``
    - ``filtered``   = chunks dropped below that threshold
    - ``avg_score``  = mean of ALL per-chunk scores (reported for visibility)
    - ``needs_web``  = ``True`` iff ``max_score < settings.crag_threshold`` OR
                      no contexts to grade at all (see ``grade`` for the
                      max-not-avg rationale)

    The pipeline uses ``kept`` + ``needs_web`` to decide the fallback mode:
    full-web (kept=[]) vs mixed (kept + Tavily docs).
    """

    kept: list[dict] = field(default_factory=list)
    filtered: list[dict] = field(default_factory=list)
    per_chunk: list[ChunkScore] = field(default_factory=list)
    avg_score: float = 0.0
    needs_web: bool = False


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

# Russian examples built via chr() to keep this source file pure ASCII --
# some tools read source as cp1252 on Windows. Same technique as the
# _cyr() helper in query_rewriter.
_PO_NEMETSKI = (
    chr(0x043F) + chr(0x043E) + "-"
    + chr(0x043D) + chr(0x0435) + chr(0x043C) + chr(0x0435)
    + chr(0x0446) + chr(0x043A) + chr(0x0438)
)
_PO_RUSSKI = (
    chr(0x043F) + chr(0x043E) + "-"
    + chr(0x0440) + chr(0x0443) + chr(0x0441) + chr(0x0441)
    + chr(0x043A) + chr(0x0438)
)


_GRADER_SYSTEM = (
    "You are a relevance grader for a bilingual German-Russian language-learning "
    "RAG system. Users ask in either DE or RU; retrieved chunks may be in either "
    "language or mixed (e.g. a Russian-German dictionary entry, or a German "
    "textbook with Russian translations). Score how RELEVANT each chunk is to "
    "ANSWERING the user question, on a 0.0--1.0 scale.\n\n"
    "CRITICAL RULES:\n"
    "1. CROSS-LINGUAL RELEVANCE IS NORMAL AND HIGH. A Russian-language chunk "
    "that explains or translates a German word IS highly relevant (>= 0.8) to "
    "a Russian question asking for the German equivalent (and vice versa). "
    "Do NOT downgrade a chunk because the chunk's language differs from the "
    "question's.\n"
    "2. A chunk containing the exact term or concept asked about -- even if "
    "the surrounding text is truncated -- scores 0.7-1.0.\n"
    "3. A chunk on a related topic without the specific answer scores 0.3-0.5.\n"
    "4. A chunk that is completely off-topic scores 0.0-0.2.\n"
    "5. Dictionary entries, vocab lists, and grammar tables that contain the "
    "queried term count as DIRECT answers (0.8-1.0), not 'partial'.\n"
    "6. TRANSLATION-TARGET CHECK -- third-party language only. The corpus "
    "covers GERMAN and RUSSIAN. If the question asks for a translation "
    "INTO a third language that is NEITHER German NOR Russian -- e.g. "
    "'auf Schwedisch', 'in English', 'po-shvedski', 'na frantsuzskom', "
    "'in Italian' -- and the chunk contains NO content in that target "
    "language, score it 0.0-0.3 even if it shows the source phrase. A "
    "German chunk showing 'Guten Tag' does NOT answer 'How is Guten Tag "
    "said in Swedish'. THIS RULE DOES NOT APPLY when the target language "
    "is German or Russian: 'auf Deutsch', 'auf Russisch', 'po-nemetski' "
    f"('{_PO_NEMETSKI}'), 'po-russki' ('{_PO_RUSSKI}'), 'in German', 'in Russian' "
    "are exactly the directions the corpus IS designed to answer -- score "
    "those chunks per rules 1-5 normally.\n\n"
    "Return STRICT JSON: {\"scores\": [{\"rank\": 1, \"score\": 0.8, "
    "\"reason\": \"<10 words>\"}, ...]}. One entry per chunk. Score floats only."
)

_CHUNK_HEAD_CHARS = 600        # truncate per-chunk text in the prompt
_REASON_CAP_CHARS = 120        # truncate any over-long LLM reasons


def _build_grader_prompt(query: str, contexts: list[dict]) -> str:
    """Assemble the grader user-prompt. Numbered chunks, truncated heads."""
    lines = [f"USER QUESTION:\n{query.strip()}", "", "RETRIEVED CHUNKS:"]
    for i, ctx in enumerate(contexts, start=1):
        text = (ctx.get("text") or "").strip().replace("\n", " ")
        if len(text) > _CHUNK_HEAD_CHARS:
            text = text[:_CHUNK_HEAD_CHARS] + "..."
        src = ctx.get("source_file") or "?"
        page = ctx.get("page")
        loc = f"{src} p.{page}" if page is not None else src
        lines.append(f"[{i}] ({loc}) {text}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

class ContextGrader:
    """Single-call relevance grader. One instance per pipeline."""

    def __init__(
        self,
        gemini_client: genai.Client,
        model: str,
        crag_threshold: float = 0.45,
        per_chunk_threshold: float = 0.5,
        max_attempts: int = 5,
        call_timeout_s: float = 60.0,
        thinking_budget: int = 0,
    ) -> None:
        self.gemini = gemini_client
        self.model = model
        self.crag_threshold = crag_threshold
        self.per_chunk_threshold = per_chunk_threshold
        self.max_attempts = max_attempts
        self.call_timeout_s = call_timeout_s
        self.thinking_budget = thinking_budget

    def grade(
        self,
        query: str,
        contexts: list[dict],
        cost: CostTracker,
    ) -> GraderVerdict:
        """Grade ``contexts`` against ``query``. Returns a ``GraderVerdict``.

        Empty input -> ``needs_web=True`` with no LLM call (cheap pass-through).
        Any Gemini failure -> falls back to ``needs_web=False`` (treat as
        sufficient) so the user still gets an answer; logged as a warning.
        """
        if not contexts:
            return GraderVerdict(needs_web=True)

        user_prompt = _build_grader_prompt(query, contexts)
        full_prompt = _GRADER_SYSTEM + "\n\n" + user_prompt
        # JSON-mode via google-genai config; the helper forwards config=
        # straight to client.models.generate_content.
        # temperature=0 makes the grader deterministic across calls --
        # critical for demo reliability (the demo: Russian "blond" must
        # consistently keep page 25 in top-10. With default temperature=1
        # the grader's scores were noisy; once paired with this temp=0,
        # the deterministic baseline became blond max=0.50 vs schwedisch
        # max=0.20-0.40, which forced crag_threshold from 0.65 down to
        # 0.45 (see config.py). Paired with the rewriter's temperature=0
        # (set 2026-06-27) for end-to-end stability.
        # Scoring ten short chunks needs no chain of thought: with the model's
        # default thinking budget this call took 5-15s, without it 2-3s, and
        # re-running the calibration set showed every query staying on the same
        # side of crag_threshold -- the tightest case (Russian "blond", the one
        # that forced the threshold down to 0.45) moved from max=0.50 to 0.70,
        # i.e. further clear of the line. Set GRADER_THINKING_BUDGET to a
        # negative value to hand the decision back to the model.
        config = genai_types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0,
            thinking_config=(
                genai_types.ThinkingConfig(thinking_budget=self.thinking_budget)
                if self.thinking_budget >= 0 else None
            ),
        )
        try:
            res = gemini_generate_guarded(
                self.gemini, self.model, full_prompt,
                config=config,
                max_attempts=self.max_attempts,
                call_timeout_s=self.call_timeout_s,
                label="crag_grader",
            )
        except Exception as exc:  # noqa: BLE001
            # Grader failure must not block the user -- treat as "sufficient"
            # so the normal generate path runs. Logged so we notice in Opik.
            log.warning("ContextGrader: Gemini failed (%s); skipping CRAG", exc)
            return GraderVerdict(
                kept=list(contexts),
                avg_score=1.0,
                needs_web=False,
            )

        cost.add_gemini(self.model, res["prompt_tokens"], res["output_tokens"])
        per_chunk = _parse_scores(res["text"], expected=len(contexts))
        return self._build_verdict(contexts, per_chunk)

    def _build_verdict(
        self,
        contexts: list[dict],
        per_chunk: list[ChunkScore],
    ) -> GraderVerdict:
        kept: list[dict] = []
        filtered: list[dict] = []
        # per_chunk is aligned by rank to contexts (1-based).
        by_rank = {s.rank: s for s in per_chunk}
        for i, ctx in enumerate(contexts, start=1):
            cs = by_rank.get(i)
            if cs is None or cs.score >= self.per_chunk_threshold:
                kept.append(ctx)
            else:
                filtered.append(ctx)
        if per_chunk:
            avg = sum(s.score for s in per_chunk) / len(per_chunk)
            max_score = max(s.score for s in per_chunk)
        else:
            avg = max_score = 0.0
        # We trigger the Tavily fallback on MAX-score, not avg. Reason: with
        # the reranker dominating, retrieved sets typically look like
        # [0.9, 0.2, 0.1, 0.1, ...] -- one very strong chunk plus noise.
        # The avg collapses to ~0.2 which would needlessly fire the web
        # fallback even though the top chunk perfectly answers the query.
        # Fallback only when EVERY chunk is below threshold ("even the best
        # one is weak"). avg_score is still reported for visibility in the
        # Inspector.
        needs_web = max_score < self.crag_threshold
        log.info(
            "ContextGrader: avg=%.2f max=%.2f kept=%d filtered=%d "
            "needs_web=%s threshold=%.2f",
            avg, max_score, len(kept), len(filtered),
            needs_web, self.crag_threshold,
        )
        return GraderVerdict(
            kept=kept,
            filtered=filtered,
            per_chunk=per_chunk,
            avg_score=round(avg, 3),
            needs_web=needs_web,
        )


# ---------------------------------------------------------------------------
# JSON parsing -- defensive, never raises
# ---------------------------------------------------------------------------

def _parse_scores(raw: str, expected: int) -> list[ChunkScore]:
    """Parse the JSON-mode grader response into ChunkScore objects.

    Defensive: a malformed response yields zero scores rather than crashing
    the request. Pipeline then treats avg_score=0.0 as needs_web=True --
    which is the safe outcome (fall back to web rather than ship a bad
    answer on grader noise).
    """
    if not raw:
        return []
    text = raw.strip()
    # Some models still wrap JSON in ```json ... ``` fences despite JSON-mode.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].lstrip("\n")
        # Drop a trailing fence if present.
        text = text.rsplit("```", 1)[0].strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        log.warning("ContextGrader: JSON parse failed (%s); raw=%r",
                    exc, text[:200])
        return []
    raw_scores = data.get("scores") if isinstance(data, dict) else None
    if not isinstance(raw_scores, list):
        log.warning("ContextGrader: missing 'scores' list; data=%r", data)
        return []
    out: list[ChunkScore] = []
    for entry in raw_scores:
        if not isinstance(entry, dict):
            continue
        try:
            rank = int(entry.get("rank"))
            score = float(entry.get("score"))
        except (TypeError, ValueError):
            continue
        if rank < 1 or rank > expected:
            continue
        score = max(0.0, min(1.0, score))
        reason = str(entry.get("reason") or "")[:_REASON_CAP_CHARS]
        out.append(ChunkScore(rank=rank, score=score, reason=reason))
    # De-duplicate -- keep first score per rank if the model emitted dupes.
    seen: set[int] = set()
    deduped: list[ChunkScore] = []
    for cs in out:
        if cs.rank in seen:
            continue
        seen.add(cs.rank)
        deduped.append(cs)
    return deduped
