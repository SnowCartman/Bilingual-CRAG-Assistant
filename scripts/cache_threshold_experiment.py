"""Cache threshold calibration -- 10 eval questions + 10 Gemini paraphrases.

Two phases so I can review the paraphrases before measurement:

    Phase 1 (default, --phase paraphrase):
        Generates a same-language paraphrase for each of the 10 eval questions
        with gemini-2.5-flash-lite. Dumps to cache_paraphrases.json.

    Phase 2 (--phase measure):
        Reads cache_paraphrases.json, embeds original + paraphrase with
        voyage-multilingual-2, computes cosine distance (1 - dot, since both
        vectors are L2-normalised). Sweeps thresholds 0.05/0.10/0.15/0.20/0.25
        and reports True-Positive Rate (positive pair distance <= threshold,
        out of 10) and False-Positive Rate (negative pair distance <= threshold,
        out of 90; negatives = each original paired with each OTHER original).
        Best threshold maximises (2*TPR - FPR) -- biases toward recall, but
        penalises false-hits which corrupt the cache. Dumps to
        cache_threshold_results.json.

Run from the repo root:
    uv run python scripts/cache_threshold_experiment.py
    uv run python scripts/cache_threshold_experiment.py --phase measure

Both phases hit live APIs (Gemini + Voyage). Estimated cost: phase 1 < $0.001,
phase 2 < $0.001. Roundtrip ~30 seconds total.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import requests

# scripts/cache_threshold_experiment.py -> scripts -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("cache_threshold")

QUESTIONS_PATH = PROJECT_ROOT / "evaluations" / "test_questions.json"
PARAPHRASE_PATH = PROJECT_ROOT / "scripts" / "cache_paraphrases.json"
RESULTS_PATH = PROJECT_ROOT / "scripts" / "cache_threshold_results.json"

THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25]

PARAPHRASE_PROMPT = """You paraphrase user questions about German grammar and bilingual
German-Russian vocabulary. Rules:
1. Output exactly ONE paraphrase, no preamble, no quotes, no explanation.
2. Use the SAME language as the input (German -> German, Russian -> Russian, mixed -> mixed).
3. Rephrase using different words and a different structure but preserve the meaning EXACTLY.
4. Keep any technical terms (Konjunktiv, Kasus, Modalpartikel, etc.) intact.
5. The paraphrase should be 80-130 percent of the original length."""


# ---------------------------------------------------------------------------
# Phase 1: paraphrase generation
# ---------------------------------------------------------------------------

def load_questions() -> list[dict]:
    with QUESTIONS_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["questions"]


def generate_paraphrase(client, model: str, query: str) -> str:
    """One-shot Gemini call. Returns the stripped paraphrase text."""
    from google.genai import types as genai_types
    prompt = f"{PARAPHRASE_PROMPT}\n\nOriginal question:\n{query}\n\nParaphrase:"
    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(
            temperature=0.4,
            max_output_tokens=200,
        ),
    )
    text = (resp.text or "").strip()
    # Strip surrounding quotes the model sometimes emits despite the instruction.
    # Cyrillic/typographic quotes rebuilt via chr() so this file stays
    # ASCII-clean.
    _QUOTES = ('"', "'", chr(0x201C), chr(0x201D), chr(0x00AB), chr(0x00BB))
    for quote in _QUOTES:
        if text.startswith(quote) and text.endswith(quote):
            text = text[1:-1].strip()
            break
    return text


def phase_paraphrase() -> None:
    """Generate paraphrases, dump JSON, prompt me to review."""
    from google import genai
    s = get_settings()
    client = genai.Client(api_key=s.gemini_api_key)
    questions = load_questions()

    out = []
    for q in questions:
        original = q["question"]
        log.info("paraphrasing %s (%s)...", q["id"], q["language"])
        para = generate_paraphrase(client, s.rewrite_model, original)
        out.append({
            "id": q["id"],
            "language": q["language"],
            "original": original,
            "paraphrase": para,
        })

    PARAPHRASE_PATH.write_text(
        json.dumps(out, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )
    log.info("wrote %d paraphrases to %s", len(out), PARAPHRASE_PATH)
    print()
    print("=" * 78)
    print("REVIEW THE PARAPHRASES BEFORE RUNNING PHASE 2:")
    print(f"  {PARAPHRASE_PATH}")
    print()
    print("If a paraphrase drifts in meaning, edit the JSON file manually.")
    print("Then run: uv run python scripts/cache_threshold_experiment.py --phase measure")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Phase 2: embed + measure
# ---------------------------------------------------------------------------

def voyage_embed_batch(texts: list[str], model: str, api_key: str) -> list[list[float]]:
    """Batched embed call. Returns L2-normalised float lists, one per input."""
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"input": texts, "model": model, "input_type": "query"}
    for attempt in range(1, 5):
        resp = requests.post("https://api.voyageai.com/v1/embeddings",
                             headers=headers, json=payload, timeout=120)
        if resp.status_code == 200:
            break
        if resp.status_code in (429, 500, 502, 503, 504) and attempt < 4:
            backoff = min(20.0, 2 ** attempt)
            log.warning("voyage %s; retry in %.0fs", resp.status_code, backoff)
            time.sleep(backoff)
            continue
        resp.raise_for_status()
    body = resp.json()
    out = []
    for d in body["data"]:
        v = np.array(d["embedding"], dtype=np.float32)
        n = float(np.linalg.norm(v)) or 1.0
        out.append((v / n).tolist())
    return out


def cosine_distance(a: list[float], b: list[float]) -> float:
    """cosine distance for L2-normalised vectors = 1 - dot product."""
    arr_a = np.asarray(a, dtype=np.float32)
    arr_b = np.asarray(b, dtype=np.float32)
    return float(1.0 - np.dot(arr_a, arr_b))


def phase_measure() -> None:
    if not PARAPHRASE_PATH.exists():
        raise SystemExit(
            f"{PARAPHRASE_PATH} not found -- run phase 1 first (default phase)."
        )
    s = get_settings()
    pairs = json.loads(PARAPHRASE_PATH.read_text(encoding="utf-8"))
    originals = [p["original"] for p in pairs]
    paraphrases = [p["paraphrase"] for p in pairs]

    log.info("embedding %d originals + %d paraphrases (one batched call each)...",
             len(originals), len(paraphrases))
    emb_orig = voyage_embed_batch(originals, s.cache_embed_model, s.voyage_api_key)
    emb_para = voyage_embed_batch(paraphrases, s.cache_embed_model, s.voyage_api_key)

    # Positive pairs: original_i <-> paraphrase_i (10 pairs)
    pos_distances: list[tuple[str, float]] = []
    for i, p in enumerate(pairs):
        d = cosine_distance(emb_orig[i], emb_para[i])
        pos_distances.append((p["id"], d))

    # Negative pairs: original_i <-> original_j for all i < j (45 unique unordered pairs)
    neg_distances: list[tuple[str, str, float]] = []
    for i in range(len(originals)):
        for j in range(i + 1, len(originals)):
            d = cosine_distance(emb_orig[i], emb_orig[j])
            neg_distances.append((pairs[i]["id"], pairs[j]["id"], d))

    # Threshold sweep
    sweep = []
    for tau in THRESHOLDS:
        tp = sum(1 for _, d in pos_distances if d <= tau)
        fp = sum(1 for _, _, d in neg_distances if d <= tau)
        tpr = tp / len(pos_distances)
        fpr = fp / len(neg_distances)
        score = 2.0 * tpr - fpr  # recall-biased
        sweep.append({
            "threshold": tau,
            "tp": tp, "fp": fp,
            "tpr": round(tpr, 4),
            "fpr": round(fpr, 4),
            "score": round(score, 4),
        })

    best = max(sweep, key=lambda r: (r["score"], -r["fpr"]))

    results = {
        "metadata": {
            "embed_model": s.cache_embed_model,
            "embed_dim": s.cache_embed_dimension,
            "n_positive_pairs": len(pos_distances),
            "n_negative_pairs": len(neg_distances),
            "thresholds": THRESHOLDS,
            "selection_criterion": "argmax 2*TPR - FPR; ties broken by lower FPR",
        },
        "positive_pair_distances": [
            {"id": qid, "distance": round(d, 4)} for qid, d in pos_distances
        ],
        "negative_pair_distance_summary": {
            "min": round(min(d for _, _, d in neg_distances), 4),
            "median": round(float(np.median([d for _, _, d in neg_distances])), 4),
            "max": round(max(d for _, _, d in neg_distances), 4),
        },
        "threshold_sweep": sweep,
        "recommended_threshold": best["threshold"],
        "recommended_reason": (
            f"At threshold {best['threshold']}: TPR={best['tpr']} ({best['tp']}/10), "
            f"FPR={best['fpr']} ({best['fp']}/{len(neg_distances)})"
        ),
    }

    RESULTS_PATH.write_text(
        json.dumps(results, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("THRESHOLD SWEEP RESULTS")
    print("=" * 78)
    print(f"{'threshold':>10}  {'TP':>4}  {'FP':>4}  {'TPR':>6}  {'FPR':>6}  {'score':>7}")
    for row in sweep:
        marker = "  <-- recommended" if row["threshold"] == best["threshold"] else ""
        print(f"{row['threshold']:>10.2f}  {row['tp']:>4}  {row['fp']:>4}  "
              f"{row['tpr']:>6.2%}  {row['fpr']:>6.2%}  {row['score']:>7.4f}{marker}")
    print()
    print(f"Positive-pair distances (per question id):")
    for qid, d in pos_distances:
        print(f"  {qid}: {d:.4f}")
    print()
    print(f"Negative-pair distance: "
          f"min {results['negative_pair_distance_summary']['min']}, "
          f"median {results['negative_pair_distance_summary']['median']}, "
          f"max {results['negative_pair_distance_summary']['max']}")
    print()
    print(f"Full report: {RESULTS_PATH}")
    print()
    print(f"To persist: set CACHE_DISTANCE_THRESHOLD={best['threshold']} in .env")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["paraphrase", "measure"], default="paraphrase")
    args = parser.parse_args()
    if args.phase == "paraphrase":
        phase_paraphrase()
    else:
        phase_measure()


if __name__ == "__main__":
    main()
