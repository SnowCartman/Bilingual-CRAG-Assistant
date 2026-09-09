# Scripts

Two kinds of thing live here: the **ingest** script that fills the vector
database, and **smoke tests / calibration harnesses** that exercise a running
backend from outside the request path.

## Ingest — run this first

`ingest.py` turns a folder of PDFs into the Qdrant collection the app queries:
extract → chunk (2000/200, structure-aware) → embed (Voyage dense + BM25
sparse) → upsert with `source_file` / `page` / `page_end` / `language` / `text`
payloads.

```bash
pip install -r ../requirements-ingest.txt
python scripts/ingest.py --source ./data/books --collection my_books_hybrid
```

- `--extractor pypdf` (default) reads the PDF text layer. A book that comes back
  with no text is a **scan**, not a script failure.
- `--extractor marker` runs marker-pdf, which OCRs scans and keeps markdown
  structure. Install it separately (`pip install marker-pdf`); it pulls in
  torch, and a CUDA GPU makes it roughly an order of magnitude faster than CPU.
- `--recreate` drops the collection first; without it, chunks are appended.

Put the collection name in `.env` as `QDRANT_COLLECTION`, then start the stack.

## Smoke tests and harnesses

Run with the stack up (`docker compose up -d`), from the repo root:

```bash
docker compose exec -T backend python /app/scripts/crag_smoke.py
```

- `router_smoke.py` — FACTUAL / EXPLANATORY / COMPARATIVE routing, template
  selection, and the cross-lingual language-detection rule.
- `security_smoke.py` — five scenarios across the three guards: EN / DE / RU
  injection attempts (blocked at the input guard), a legitimate DE query that
  must *not* be blocked, and synthetic PII the output validator redacts.
- `crag_smoke.py` — end to end through CRAG: an in-corpus question (grader keeps
  the corpus answer) and an out-of-corpus one (grader fires the web search).
- `opik_smoke.py` — one query plus one feedback round-trip, verifying Opik trace
  persistence and the `/feedback` JSONL fallback. Writes `feedback.jsonl`
  (gitignored — a transient audit log).
- `cache_threshold_experiment.py` — calibrates the semantic-cache distance
  threshold from paraphrase pairs versus distinct questions.
- `eval_crag_vs_baseline.py` — measures the CRAG delta two ways: deterministic
  semantic metrics, and a decomposed `gemini-2.5-pro` judge. Point it at your
  own labelled question set.

> **These tests carry the questions of the corpus they were written for** — a
> German/Russian grammar shelf, so they ask about *Konjunktiv*, *Dativ*, and a
> Swedish phrase chosen because it is deliberately *not* in that corpus. With a
> different corpus the in-corpus and out-of-corpus expectations flip, so swap
> the questions before reading anything into a pass or a fail.
