# Evaluation

How retrieval quality and answer quality were measured, what the numbers say,
and — just as important — where the numbers stop being informative.

> The evaluation ran against a private set of questions over a copyrighted
> textbook corpus, so the golden dataset and raw judge transcripts are **not**
> published here (they contain verbatim book excerpts). This document reports the
> metrics and methodology; the harness code lives in [`scripts/`](scripts/) and
> runs against your own corpus and question set.

---

## Two methods, triangulated

No single metric captures a RAG system. Two independent methods were used so
their blind spots don't overlap:

- **Method B — deterministic semantic metrics.** For each question, retrieval is
  scored with standard IR metrics (P@k, R@k, MRR, NDCG@10) against labelled
  relevant chunks, plus an `answer_coverage` cosine between the generated answer
  and the reference. Fully deterministic — no LLM in the scoring path, so it never
  drifts between runs.
- **Method A — decomposed LLM judge.** A stronger model (`gemini-2.5-pro`)
  decomposes each answer into atomic **claims** and scores **faithfulness**
  (is every claim supported by a retrieved context?) and **completeness** (does
  the answer satisfy the question's criteria?). Decomposing into claims makes the
  judgment auditable — each verdict cites the exact supporting context.

Two question sets were used, and every number below says which one it came from,
because they are not interchangeable:

- **The comparison set — 10 questions.** Factual / analytical / conceptual types
  across German, Russian and mixed languages, deliberately including questions
  expected to be *hard* or *out of corpus* so honest-refusal behaviour is tested
  rather than assumed. Used to rank three retrieval configurations.
- **The self-correction set — 3 questions.** The two the comparison round flagged
  as failure modes plus one control, re-run in depth against the shipped
  configuration with and without the CRAG layer.

---

## Retrieval results

Retrieval is effectively solved after reranking. Shipped configuration, on the
10-question comparison set:

| Metric | Result |
|---|---|
| **P@1** | **1.00** — the top chunk is always relevant |
| MRR | 1.00 |
| NDCG@10 | 0.98 (0.98–0.99 per question in the deeper round) |
| Recall@3 / @5 / @10 | 1.00 — measured in the 3-question round |

Two earlier configurations, measured identically, reached P@1 0.70 and 0.40, and
MRR 0.70 and 0.55. The hybrid-plus-reranker configuration is not at ceiling by
accident.

The low P@10 (~0.1–0.4 on some questions) is *expected and correct*: most
questions have only one or two truly relevant pages, so precision at 10
mechanically drops even when recall and ranking are perfect. MRR and P@1 are the
metrics that matter for a cited-answer system, and both are at ceiling.

The reranker earns its place here: on a Russian dictionary query it promoted the
correct page to rank 1, which is what drives P@1 = 1.00.

---

## Answer results

| Metric | 10-question comparison set | 3-question self-correction round |
|---|---|---|
| **Faithfulness** (LLM judge) | **0.94** | **1.00** on every question, both runs |
| Faithfulness (second judge) | 0.96 | — |
| Completeness (LLM judge) | 0.85 | 0.33–1.00 per question |
| Answer coverage (Method B) | 0.85 | 0.83–0.90 |

**Faithfulness is the number to trust.** In the deep round the judge found zero
unsupported claims across every question and both replicate runs; on the broader
10-question set it sat at 0.94, against 0.78 and 0.86 for the two earlier
configurations. The generator does not invent facts, and when the corpus lacks
the answer it says so rather than filling the gap. On a question about "typical
mistakes Russian speakers make with German cases" the corpus genuinely didn't
cover it; the system returned an honest "the sources don't contain this", and the
judge scored that refusal fully faithful *and* complete, because acknowledging
the limit was the correct answer.

Completeness is the weaker metric and is reported as such: in the deep round the
per-question spread runs from 0.33 (a question the corpus answers only partially)
to 1.00, so no configuration decision was made on it alone.

---

## Honest interpretation

- **The metric was already at ceiling.** Faithfulness sat at 1.0 and P@1 at 1.0
  *before* the CRAG self-correction layer was added, so CRAG shows no headroom to
  claim on this in-corpus dataset. That's a finding, not a disappointment: the
  retrieval + generation core was already sound.
- **CRAG's value is operational, not on these numbers.** Its payoff is on
  *out-of-corpus* questions, which the golden dataset deliberately excludes. Live
  runs confirm the behaviour: a Swedish-vocabulary question scores every chunk far
  below threshold, fires the web search, and answers from web documents badged as
  web sources — while in-corpus questions never trigger an unnecessary external
  call.
- **Generator + judge non-determinism dominates small deltas.** Two replicate
  runs of the same question swung completeness by ~20 percentage points
  (0.7 ↔ 1.0). Any sub-20-point difference between configurations on this set is
  inside the noise band and not a real signal — which is exactly why faithfulness
  (stable at 1.0) is the metric to trust here.

---

## What more evaluation budget would buy

- **Replicate each question 5× and report mean ± stdev.** Two replicates already
  bracket the noise band; five would let the completeness numbers carry a
  confidence interval instead of a single point.
- **A curated out-of-corpus reference set** (Swedish vocabulary, English idioms)
  with ground-truth answers, scored by both methods. This is where CRAG could
  demonstrate a *quantitative* gain on its intended use case — deferred because
  hand-authored ground truth carries curator bias that's easy to introduce and
  hard to detect.

---

## Reproducing

The harnesses in [`scripts/`](scripts/) run the smoke tests and the
CRAG/threshold evaluations against your own corpus:

```bash
docker compose exec -T backend python /app/scripts/crag_smoke.py
docker compose exec -T backend python /app/scripts/cache_threshold_experiment.py
```

Bring your own labelled question set in place of the private golden dataset.
