"""Evaluation -- CRAG-on vs the pre-CRAG retrieval baseline.

Runs ``pipeline.query()`` with the CRAG layer on, against the two failure
modes an earlier evaluation round had flagged:

    q7 (analytical, in-corpus): 'zu' + Dativ direction vs 'in' + Akk
    q10 (conceptual, mixed):    typical Russian-learner article+Kasus errors

Method B (deterministic semantic metrics, voyage-multilingual-2 cosine):
    P@1, P@3, P@5, P@10, R@3, R@5, R@10, MRR, NDCG@10, answer_coverage.

Method A v2 (decomposed Gemini-2.5-pro judge, hooks A-D):
    faithfulness = (1.0*supported + 0.5*partial) / total_claims
    completeness = satisfied / total_criteria

The baseline numbers are hardcoded from the pre-CRAG run, so the comparison
needs no second stack -- but it DOES need a labelled question set, and that set
is **not part of this repository**: its reference answers and contexts quote the
source books verbatim, which is not ours to publish. Bring your own, in this
shape:

    {"entries": [
      {"question_id": "q7",
       "question":    "...",
       "language":    "de",
       "reference_answer":   "the answer a domain expert would give",
       "reference_contexts": [{"source_file": "...", "page": 12,
                               "text": "the passage that supports it"}],
       "reference_criteria": ["each fact the answer must contain"]}
    ]}

Question ids that match the stored baseline (q3 / q7 / q10) are printed side by
side with it; any other id is reported on its own numbers.

Run from the repo root (host):
    docker compose exec -T backend python /app/scripts/eval_crag_vs_baseline.py \
        /app/scripts/your_golden_set.json

Cost estimate: ~$0.50 per run (2 pipeline runs + Method B embeds + the
Gemini-2.5-pro judge).
"""
from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from pydantic import BaseModel, Field

sys.path.insert(0, "/app")

from app.config import get_settings                                     # noqa: E402
from app.services.rag_pipeline import (                                  # noqa: E402
    RAGPipeline, CostTracker, voyage_embed_query, gemini_generate_guarded,
)
from google.genai import types as genai_types                            # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("crag_eval")

_DEFAULT_GOLDEN = "/app/scripts/golden_set.json"
_DEFAULT_OUT = "/app/scripts/crag_eval_results.json"
GOLDEN_PATH = Path(sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_GOLDEN)
OUTPUT_PATH = Path(sys.argv[2] if len(sys.argv) > 2 else _DEFAULT_OUT)
EMBED_MODEL = "voyage-multilingual-2"
JUDGE_MODEL = "gemini-2.5-pro"
JUDGE_THINKING_BUDGET = 8000
DEFAULT_THRESHOLD = 0.70

# Hardcoded pre-CRAG baseline.
PRE_CRAG_BASELINE = {
    "q3": {
        "method_b": {
            "P@1": 1.0, "P@3": 0.3333, "P@5": 0.2, "P@10": 0.1,
            "R@3": 1.0, "R@5": 1.0, "R@10": 1.0, "MRR": 1.0,
            "NDCG@10": 0.9935, "answer_coverage": 0.8735,
        },
        "method_a_v2": {
            "faithfulness": 1.0, "completeness": 0.3333,
            "total_claims": 2, "supported": 2, "partial": 0, "unsupported": 0,
            "total_criteria": 3, "satisfied": 1,
        },
    },
    "q7": {
        "method_b": {
            "P@1": 1.0, "P@3": 1.0, "P@5": 0.8, "P@10": 0.4,
            "R@3": 1.0, "R@5": 1.0, "R@10": 1.0, "MRR": 1.0,
            "NDCG@10": 0.9949, "answer_coverage": 0.8977,
        },
        "method_a_v2": {
            "faithfulness": 1.0, "completeness": 1.0,
            "total_claims": 20, "supported": 20, "partial": 0, "unsupported": 0,
            "total_criteria": 9, "satisfied": 9,
        },
    },
    "q10": {
        "method_b": {
            "P@1": 1.0, "P@3": 0.3333, "P@5": 0.2, "P@10": 0.1,
            "R@3": 1.0, "R@5": 1.0, "R@10": 1.0, "MRR": 1.0,
            "NDCG@10": 0.9803, "answer_coverage": 0.8370,
        },
        "method_a_v2": {
            "faithfulness": 1.0, "completeness": 1.0,
            "total_claims": 6, "supported": 6, "partial": 0, "unsupported": 0,
            "total_criteria": 6, "satisfied": 6,
        },
    },
}


# -----------------------------------------------------------------------------
# Pydantic schemas for the judge's structured output
# -----------------------------------------------------------------------------

class Claim(BaseModel):
    id: int
    text: str


class ClaimExtraction(BaseModel):
    claims: List[Claim]
    total_claims: int


class ClaimVerdict(BaseModel):
    claim_id: int
    verdict: str = Field(description="supported|partially-supported|unsupported")
    evidence_source: Optional[str] = None
    evidence_quote: Optional[str] = None
    reasoning: str


class ClaimVerification(BaseModel):
    verdicts: List[ClaimVerdict]


class Criterion(BaseModel):
    id: int
    text: str


class CriteriaExtraction(BaseModel):
    criteria: List[Criterion]


class CriterionCheck(BaseModel):
    criterion_id: int
    satisfied: bool
    reasoning: str


class CompletenessCheck(BaseModel):
    checks: List[CriterionCheck]


# -----------------------------------------------------------------------------
# Method A v2 prompts
# -----------------------------------------------------------------------------

CLAIM_EXTRACTOR_PROMPT = """\
You decompose a generated answer into atomic claims. Each claim is a single,
checkable proposition about facts. Do NOT split rhetorical phrases. Do NOT
invent claims the answer does not actually make.

GRANULARITY RULES (Hook A):
  - One claim per simple fact.
  - One claim per listed example.
  - Do NOT split a single definitional phrase into sub-claims.
  - Do NOT extract claims about answer structure.
  - Target between 3 and 25 claims.

If the answer is an honest-refusal ("the corpus does not contain..."), extract
the refusal itself as claim 1.

Generated answer:
---
{answer}
---

Return JSON with claims (id starting at 1) and total_claims.
"""

CLAIM_VERIFIER_PROMPT = """\
Verify each claim against the contexts using a 3-TIER verdict:

  'supported'            -- backed by contexts. Hook B: accept near-verbatim
                            even with formatting differences. Paraphrase that
                            preserves meaning is acceptable.
  'partially-supported'  -- Hook C: core fact backed but a peripheral detail
                            is not. Do NOT use as soft "I'm not sure".
  'unsupported'          -- no context supports OR context contradicts.

HOOK D (refusal): claim is 'supported' when NO context (retrieved OR
reference) contains the claimed-absent information.

For each claim: verdict + evidence_source + evidence_quote + 1-sentence reason.

CLAIMS TO VERIFY:
{claims_block}

RETRIEVED CONTEXTS (the answer was generated from these):
{retrieved_block}

REFERENCE CONTEXTS (curated ground-truth supplement):
{reference_block}
"""

CRITERIA_EXTRACTOR_PROMPT = """\
Decompose a reference answer into discrete criteria a complete answer must
satisfy. Each criterion is a fact, rule, or coverage point. NO stylistic
preferences.

For honest-refusal references, extract the refusal as criterion 1.

Reference answer:
---
{reference}
---

Return JSON: criteria list with id starting at 1.
"""

COMPLETENESS_CHECK_PROMPT = """\
For each criterion, decide if the generated answer satisfies it. A criterion
is satisfied when the answer covers its content (verbatim or paraphrase),
regardless of how it is phrased.

CRITERIA:
{criteria_block}

GENERATED ANSWER:
---
{answer}
---

Return JSON: checks (criterion_id + satisfied:bool + reasoning).
"""


# -----------------------------------------------------------------------------
# Method B helpers
# -----------------------------------------------------------------------------

def _voyage_embed_batch(texts: list[str], settings, cost: CostTracker) -> np.ndarray:
    """Batch-embed via existing Voyage helper (one call per text -- 2 docs total,
    keeping it simple). Returns L2-normalised (n, 1024) array."""
    rows = []
    for t in texts:
        v = voyage_embed_query(t, EMBED_MODEL, settings.voyage_api_key, cost)
        rows.append(v)
    return np.asarray(rows, dtype=np.float32)


def precision_at_k(rel: np.ndarray, threshold: float, k: int) -> float:
    if k <= 0 or len(rel) == 0:
        return 0.0
    return float((rel[:k] >= threshold).sum()) / float(min(k, len(rel)))


def recall_at_k(ref_match_rank: np.ndarray, k: int, n_ref: int) -> float:
    if n_ref == 0:
        return 0.0
    matched = ((ref_match_rank > 0) & (ref_match_rank <= k)).sum()
    return float(matched) / float(n_ref)


def mrr(rel: np.ndarray, threshold: float) -> float:
    for i, s in enumerate(rel, start=1):
        if s >= threshold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(rel: np.ndarray, k: int) -> float:
    if len(rel) == 0 or k <= 0:
        return 0.0
    r = np.clip(rel[:k], 0.0, 1.0)
    disc = 1.0 / np.log2(np.arange(2, len(r) + 2))
    dcg = float((r * disc).sum())
    ideal = np.sort(r)[::-1]
    idcg = float((ideal * disc).sum())
    return dcg / idcg if idcg > 0 else 0.0


def compute_method_b(
    retrieved: list[dict],
    references: list[dict],
    generated_answer: str,
    reference_answer: str,
    settings,
    cost: CostTracker,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Cosine-based IR metrics + answer-side coverage."""
    if not retrieved:
        return {k: 0.0 for k in
                ("P@1", "P@3", "P@5", "P@10", "R@3", "R@5", "R@10",
                 "MRR", "NDCG@10", "answer_coverage")}

    # Embed everything
    ret_vecs = _voyage_embed_batch([c.get("text", "") for c in retrieved], settings, cost)
    if references:
        ref_vecs = _voyage_embed_batch([c.get("text", "") for c in references], settings, cost)
        cos = ret_vecs @ ref_vecs.T            # (n_retrieved, n_ref)
        rel = cos.max(axis=1)
        ref_match_rank = np.zeros(len(references), dtype=int)
        for r in range(len(references)):
            for i in range(len(retrieved)):
                if cos[i, r] >= threshold:
                    ref_match_rank[r] = i + 1
                    break
    else:
        rel = np.zeros(len(retrieved))
        ref_match_rank = np.zeros(0, dtype=int)

    gen_vec = _voyage_embed_batch([generated_answer], settings, cost)[0] if generated_answer else None
    ref_ans_vec = _voyage_embed_batch([reference_answer], settings, cost)[0] if reference_answer else None
    answer_coverage = float(gen_vec @ ref_ans_vec) if gen_vec is not None and ref_ans_vec is not None else None

    n_ref = len(references)
    return {
        "P@1":  precision_at_k(rel, threshold, 1),
        "P@3":  precision_at_k(rel, threshold, 3),
        "P@5":  precision_at_k(rel, threshold, 5),
        "P@10": precision_at_k(rel, threshold, 10),
        "R@3":  recall_at_k(ref_match_rank, 3, n_ref),
        "R@5":  recall_at_k(ref_match_rank, 5, n_ref),
        "R@10": recall_at_k(ref_match_rank, 10, n_ref),
        "MRR":  mrr(rel, threshold),
        "NDCG@10": ndcg_at_k(rel, 10),
        "answer_coverage": answer_coverage,
        "per_chunk_relevance": [round(float(x), 4) for x in rel],
    }


# -----------------------------------------------------------------------------
# Method A v2 driver
# -----------------------------------------------------------------------------

def _judge_call(client, prompt: str, schema, settings, cost: CostTracker, label: str):
    cfg = genai_types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=schema,
        thinking_config=genai_types.ThinkingConfig(thinking_budget=JUDGE_THINKING_BUDGET),
    )
    res = gemini_generate_guarded(
        client, JUDGE_MODEL, prompt,
        config=cfg, max_attempts=settings.gemini_max_attempts,
        call_timeout_s=settings.gemini_call_timeout_s, label=label,
    )
    cost.add_gemini(JUDGE_MODEL, res["prompt_tokens"], res["output_tokens"])
    return schema.model_validate_json(res["text"])


def _format_contexts(contexts: list[dict], kind: str) -> str:
    if not contexts:
        return f"(no {kind} contexts)"
    lines = []
    for c in contexts:
        rank = c.get("rank") if kind == "retrieved" else c.get("relevance_rank")
        source = (c.get("source_file") or "")[:80]
        page = c.get("page")
        text = (c.get("text") or "").strip()
        lines.append(f"[{kind} {rank}] {source} p.{page}\n{text}\n")
    return "\n".join(lines)


def compute_method_a_v2(
    pipeline: RAGPipeline,
    answer: str,
    retrieved: list[dict],
    references: list[dict],
    reference_answer: str,
    settings,
    cost: CostTracker,
) -> dict:
    """Decomposed judge (Hooks A-D) using gemini-2.5-pro."""
    client = pipeline.gemini  # reuse the already-built Gemini client

    # Stage 1 -- extract claims
    extr: ClaimExtraction = _judge_call(
        client,
        CLAIM_EXTRACTOR_PROMPT.format(answer=answer),
        ClaimExtraction, settings, cost, "extract_claims",
    )

    # Stage 2 -- verify each claim against contexts
    if extr.claims:
        claims_block = "\n".join(f"[claim {c.id}] {c.text}" for c in extr.claims)
        ver: ClaimVerification = _judge_call(
            client,
            CLAIM_VERIFIER_PROMPT.format(
                claims_block=claims_block,
                retrieved_block=_format_contexts(retrieved, "retrieved"),
                reference_block=_format_contexts(references, "reference"),
            ),
            ClaimVerification, settings, cost, "verify_claims",
        )
    else:
        ver = ClaimVerification(verdicts=[])

    supported = sum(1 for v in ver.verdicts if v.verdict == "supported")
    partial = sum(1 for v in ver.verdicts if v.verdict == "partially-supported")
    unsupported = sum(1 for v in ver.verdicts if v.verdict == "unsupported")
    total = len(ver.verdicts)
    faith = (supported + 0.5 * partial) / total if total else 0.0

    # Stage 3 -- completeness against reference_answer
    cri: CriteriaExtraction = _judge_call(
        client,
        CRITERIA_EXTRACTOR_PROMPT.format(reference=reference_answer),
        CriteriaExtraction, settings, cost, "extract_criteria",
    )
    if cri.criteria:
        criteria_block = "\n".join(f"[criterion {c.id}] {c.text}" for c in cri.criteria)
        cp: CompletenessCheck = _judge_call(
            client,
            COMPLETENESS_CHECK_PROMPT.format(
                criteria_block=criteria_block, answer=answer,
            ),
            CompletenessCheck, settings, cost, "completeness",
        )
        satisfied = sum(1 for c in cp.checks if c.satisfied)
        n_cri = len(cri.criteria)
        comp = satisfied / n_cri if n_cri else 0.0
    else:
        cp = CompletenessCheck(checks=[])
        satisfied = 0
        n_cri = 0
        comp = 0.0

    return {
        "faithfulness": faith,
        "completeness": comp,
        "total_claims": total,
        "supported": supported,
        "partial": partial,
        "unsupported": unsupported,
        "total_criteria": n_cri,
        "satisfied": satisfied,
        "claims": [c.model_dump() for c in extr.claims],
        "verdicts": [v.model_dump() for v in ver.verdicts],
        "criteria": [c.model_dump() for c in cri.criteria],
        "checks": [c.model_dump() for c in cp.checks],
    }


# -----------------------------------------------------------------------------
# Pretty-print summary
# -----------------------------------------------------------------------------

def _delta(a: float, b: float) -> str:
    """Return a tagged delta string -- '+0.05', '-0.10', '0.00'."""
    if a is None or b is None:
        return "n/a"
    d = a - b
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.3f}"


def _row(label: str, baseline: float, crag: float) -> str:
    bs = f"{baseline:.3f}" if isinstance(baseline, (int, float)) else "n/a"
    cs = f"{crag:.3f}" if isinstance(crag, (int, float)) else "n/a"
    return f"  {label:<18} base: {bs:<8} CRAG: {cs:<8} delta: {_delta(crag, baseline)}"


def print_summary(results: list[dict]) -> None:
    print()
    print("=" * 78)
    print("  CRAG evaluation -- CRAG-on vs the pre-CRAG baseline")
    print("=" * 78)
    for r in results:
        qid = r["question_id"]
        print()
        print(f"  {qid}: {r['question'][:70]}")
        print(f"  CRAG operational: crag_score={r['operational']['crag_score']}, "
              f"used_web_search={r['operational']['used_web_search']}, "
              f"chunks_filtered={r['operational']['chunks_filtered']}, "
              f"cache_hit={r['operational']['cache_hit']}")
        # Baseline numbers only exist for the question ids that were measured
        # before CRAG. Your own set will not have them, so the run still
        # reports its own metrics and simply skips the side-by-side.
        b = PRE_CRAG_BASELINE.get(qid)
        mb_crag = r["method_b"]
        ma_crag = r["method_a_v2"]
        if b is None:
            print("  Method B (deterministic semantic) -- no stored baseline "
                  f"for {qid}, showing this run only:")
            for k in ("P@1", "P@3", "P@5", "P@10", "R@3", "R@5", "R@10",
                      "MRR", "NDCG@10", "answer_coverage"):
                print(f"    {k:<18} {mb_crag[k]}")
            print("  Method A v2 (decomposed judge):")
            print(f"    {'faithfulness':<18} {ma_crag['faithfulness']}")
            print(f"    {'completeness':<18} {ma_crag['completeness']}")
            print(f"  Claims: {ma_crag['total_claims']} total "
                  f"({ma_crag['supported']}sup/{ma_crag['partial']}par/"
                  f"{ma_crag['unsupported']}uns)")
            continue
        mb_base = b["method_b"]
        print("  Method B (deterministic semantic):")
        for k in ("P@1", "P@3", "P@5", "P@10", "R@3", "R@5", "R@10",
                  "MRR", "NDCG@10", "answer_coverage"):
            print(_row(k, mb_base[k], mb_crag[k]))
        ma_base = b["method_a_v2"]
        print("  Method A v2 (decomposed judge):")
        print(_row("faithfulness", ma_base["faithfulness"], ma_crag["faithfulness"]))
        print(_row("completeness", ma_base["completeness"], ma_crag["completeness"]))
        print(f"  Claims: base {ma_base['total_claims']} total ({ma_base['supported']}sup/"
              f"{ma_base['partial']}par/{ma_base['unsupported']}uns) | "
              f"CRAG {ma_crag['total_claims']} total ({ma_crag['supported']}sup/"
              f"{ma_crag['partial']}par/{ma_crag['unsupported']}uns)")
    print()
    print("=" * 78)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    log.info("loading settings + pipeline")
    settings = get_settings()
    pipeline = RAGPipeline(settings)
    log.info("loading golden subset from %s", GOLDEN_PATH)
    if not GOLDEN_PATH.exists():
        log.error(
            "no labelled question set at %s. This repository does not ship one "
            "-- the reference answers quote the source books. Supply your own "
            "(schema in this file's docstring) and pass its path as the first "
            "argument.", GOLDEN_PATH)
        return 1
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    entries = golden["entries"]
    log.info("%d entries to evaluate", len(entries))

    total_cost = CostTracker()
    out_results = []
    t_start = time.perf_counter()

    for entry in entries:
        qid = entry["question_id"]
        question = entry["question"]
        log.info("===== %s : %s", qid, question[:80])

        # Run pipeline with CRAG-on, use_cache=False -> forces fresh DB+grader run
        # for every probe, no cross-question contamination.
        log.info("[%s] running pipeline.query() ...", qid)
        t0 = time.perf_counter()
        result = pipeline.query(question, session_id=None, use_cache=False)
        latency = time.perf_counter() - t0
        log.info("[%s] done in %.1fs cost=%.4f", qid, latency, result["cost"].total_usd)

        contexts_dump = [
            {
                "rank": c.rank,
                "score": c.score,
                "source_file": c.source_file,
                "page": c.page,
                "language": c.language,
                "text": c.text,
                "origin": c.origin,
            }
            for c in result["contexts"]
        ]

        # Method B
        log.info("[%s] Method B ...", qid)
        mb_cost = CostTracker()
        mb = compute_method_b(
            contexts_dump, entry.get("reference_contexts", []),
            result["answer"], entry.get("reference_answer", ""),
            settings, mb_cost,
        )
        log.info("[%s] Method B done: answer_coverage=%.3f P@1=%.2f",
                 qid, mb.get("answer_coverage", 0.0), mb["P@1"])

        # Method A v2
        log.info("[%s] Method A v2 (Gemini-2.5-pro judge) ...", qid)
        ma_cost = CostTracker()
        ma = compute_method_a_v2(
            pipeline, result["answer"], contexts_dump,
            entry.get("reference_contexts", []), entry.get("reference_answer", ""),
            settings, ma_cost,
        )
        log.info("[%s] Method A v2 done: faith=%.3f comp=%.3f",
                 qid, ma["faithfulness"], ma["completeness"])

        # Accumulate eval costs (Method B + Method A v2 only; pipeline cost is
        # tracked separately on each result entry via result["cost"]).
        def _merge_gemini(src: CostTracker) -> None:
            for m, rec in src.gemini.items():
                tgt = total_cost.gemini.setdefault(m, {"calls": 0, "in": 0, "out": 0})
                for k in ("calls", "in", "out"):
                    tgt[k] += rec[k]
        def _merge_voyage(src: CostTracker) -> None:
            for m, rec in src.voyage.items():
                tgt = total_cost.voyage.setdefault(m, {"calls": 0, "tokens": 0})
                for k in ("calls", "tokens"):
                    tgt[k] += rec[k]
        _merge_gemini(mb_cost); _merge_voyage(mb_cost)
        _merge_gemini(ma_cost); _merge_voyage(ma_cost)
        out_results.append({
            "question_id": qid,
            "question": question,
            "operational": {
                "crag_score": result.get("crag_score"),
                "used_web_search": result.get("used_web_search"),
                "chunks_filtered": result.get("chunks_filtered"),
                "cache_hit": result.get("cache_hit"),
                "query_type": result.get("query_type"),
                "latency_s": round(latency, 2),
                "pipeline_cost_usd": result["cost"].total_usd,
            },
            "answer": result["answer"],
            "n_retrieved": len(contexts_dump),
            "method_b": mb,
            "method_a_v2": ma,
        })

    wallclock = time.perf_counter() - t_start
    summary = {
        "step": 10,
        "title": "CRAG-on vs pre-CRAG baseline",
        "judge_model": JUDGE_MODEL,
        "embedder": EMBED_MODEL,
        "method_b_threshold": DEFAULT_THRESHOLD,
        "wallclock_s": round(wallclock, 1),
        "eval_cost_usd": round(total_cost.to_breakdown().total_usd, 4),
        "baseline_source": (
            "an earlier retrieval prototype + "
            "method_a_decomposed_v2.json"
        ),
        "baseline": PRE_CRAG_BASELINE,
        "crag_on_results": out_results,
    }
    OUTPUT_PATH.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="ascii")
    log.info("wrote %s (%d bytes)", OUTPUT_PATH, OUTPUT_PATH.stat().st_size)

    print_summary(out_results)
    print(f"\nWallclock: {wallclock:.1f}s   Eval cost: ${summary['eval_cost_usd']:.4f}")
    print(f"Output: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
