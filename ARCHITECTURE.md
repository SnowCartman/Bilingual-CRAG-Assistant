# Architecture

This document is the *why-by-why* of the system: the engineering decisions
behind each layer and the tradeoffs weighed to make them. For what the code
does, read the modules; for how well it works, see [`EVALUATION.md`](EVALUATION.md).

The design goal throughout: a production-shaped RAG system for **one demanding
end user** (a professional German–Russian translator) that a **technical
reviewer** can also inspect end to end.

---

## Retrieval: hybrid, reranked, cross-lingual

The corpus is bilingual and the queries come in either language — a Russian
question often needs a German answer from a German book, and vice versa. That
rules out a single-language keyword approach.

- **Dense embeddings — `voyage-multilingual-2` (1024d).** Chosen over a larger
  2048d model after measuring both: P@10 was within **0.002**, so doubling the
  vector size (and the Qdrant storage) bought nothing. Cross-lingual RU↔DE
  retrieval was validated directly.
- **Sparse — BM25.** Language-agnostic IDF catches the exact term (a specific
  grammar form, a proper noun) that a dense model can blur.
- **Fusion + rerank.** Dense and sparse candidates are merged with reciprocal
  rank fusion (prefetch 100 each), then a cross-encoder reranker
  (`rerank-2.5`) lifts the top 10. On one Russian dictionary query this reranking
  step moved the correct page from rank ~5 to rank 1 — the difference between a
  right and a wrong answer.
- **Qdrant as the store.** Hybrid dense+sparse with server-side fusion, and the
  index persists independently of the app container, so a rebuild never
  re-indexes.

---

## Self-correction: CRAG + web fallback

Retrieval alone can't tell you when the answer simply isn't in the corpus. A
**Corrective-RAG grader** scores each retrieved chunk (0–1) with a
temperature-0 LLM call. The decision rule is deliberately simple and **under my
control, not the model's**:

```
max chunk score < 0.45  →  corpus insufficient  →  fire a Tavily web search
otherwise               →  answer from the corpus
```

- **Why a threshold, not an agent.** Agency here is a dial, not a switch. A fixed,
  observable threshold is debuggable and reproducible; handing the "should I
  search the web?" decision to an autonomous planner would trade that away for
  no measured benefit on this corpus.
- **Why web fallback at all.** Out-of-corpus questions are real in daily use
  ("how do you say *good morning* in Swedish?"). A refusal-only system feels
  broken; a web search returns up to 3 documents in the same shape as corpus
  chunks and the generator answers from them, clearly badged as web sources.
- **The per-chunk filter is load-bearing only when mixing.** When corpus and web
  contexts are combined, weak corpus chunks are filtered out so they don't dilute
  the web answer. When the corpus is sufficient, that filter is bypassed —
  otherwise it would wrongly drop conservatively-scored dictionary pages.
- **Threshold calibration.** The grader runs at `temperature=0` so scores are
  deterministic. The 0.45 cutoff was calibrated against that deterministic
  distribution: in-corpus questions land ~0.50, out-of-corpus ~0.20–0.40, and
  0.45 cleanly separates them. An earlier 0.65 (set under a noisy `temperature=1`
  grader) intermittently sent good in-corpus questions to the web.

---

## Generation: type-aware, cited, language-matched

- **Query router.** A lightweight classifier tags each question as factual,
  explanatory, or comparative and selects a matching prompt template — a
  factual lookup and a multi-part comparison want different answer shapes.
- **Language matching.** The generator answers in the *question's* language. A
  hard prompt rule distinguishes "answer in Russian" from "translate *into*
  Russian" — an early version confused the two and switched answer languages.
- **Citations.** Every answer cites book + page from the retrieved chunks, so the
  learner can go verify on the actual page. Web results are cited as
  `[source, web]`; labelling them "page None" was enough to make the model treat
  its own citation as broken and refuse.
- **Translation is composition, not lookup.** Asking for a whole sentence in a
  third language used to fail even when the sources held every piece of it: the
  faithfulness rule ("answer only from the excerpts") read as "only answer if the
  exact phrase is present". The rule now explicitly outranks refusal for
  translation requests — build the phrase from the evidence, cite each piece, and
  name the pattern you followed and any deviation from it. Refusal is reserved
  for excerpts with no usable target-language material, which a regression check
  confirms still fires.
- **The thinking budget is not a free lunch — and where it is spent matters.**
  The *generator* runs with model-side thinking disabled: on a grounded prompt
  the evidence is already in the context block, so that reasoning bought only
  latency (17.2 s and ~2 575 thinking tokens versus 4.0 s without, at equal
  answer quality), and on the longest comparison prompts it pushed the call past
  the upstream deadline, which reached the user as a missing answer. The *CRAG
  grader* keeps it. Disabling it there was ~4× faster and looked fine on a
  five-query check — then broke in real use: its third-language rule ("the
  corpus holds German and Russian; a chunk with no Greek in it does not answer a
  question about Greek") is a conditional the model has to reason through, and
  without a budget German grammar pages about *wohin* scored 0.70–0.80 against
  "what is *wohin geht es zum Hafen* in Greek". No web search fired, and the
  answer was built from chunks that could not contain the answer. Across eight
  calibration queries: budget 0 got two wrong, 128 one wrong, 512 one wrong *and*
  pushed the tightest in-corpus case below the threshold; model-decided got all
  eight right. The lesson generalised: a latency change to a *judging* component
  has to be validated on the decisions it makes, not on the queries that happen
  to be at hand.
- **Resilience.** The generation call is wrapped in a retry loop with a watchdog
  timer; during a real multi-minute upstream 5xx outage the system rode through
  five consecutive failures and succeeded on the retry rather than hanging.

---

## Caching + memory: fast, private, script-safe

- **Semantic cache (Redis + HNSW).** Repeat and near-repeat questions return in
  ~0.4 s instead of the ~14 s a full corrective pass costs — roughly 35× on a
  hit, which is why the cache is worth its complexity here. Distance
  threshold 0.20.
- **A lexical guard on top.** Vector similarity alone would match a German cached
  question to a Russian one (their embeddings are close by design). A same-script
  Jaccard-0.7 check rejects cross-script and typo'd near-matches, so the answer
  language always matches the question and a typo never fetches the wrong cached
  answer.
- **Conversation memory.** A 4-turn Redis sliding window feeds the follow-up
  rewriter — one call per turn, no database schema.
- **Turns that steer instead of ask.** "That didn't answer, try again" is not a
  question, but retrieval treated it as one and confidently answered about
  whichever chunks happened to share a word with the complaint. The rewriter now
  recognises those turns and resolves them back to the previous question, and a
  turn that failed before producing an answer is still written to memory —
  otherwise the retry has nothing to point at.

---

## Security: three regex layers

Deterministic, sub-millisecond, zero-cost first-line defence — no LLM in the
guard path:

- **Input guard** blocks known prompt-injection phrasings (English + German +
  Russian).
- **Document guard** filters retrieval-poisoned chunks from both Qdrant and the
  web results.
- **Output validator** redacts PII from the generated answer.

The three-layer shape was inspired by material studied during an AI engineering
course; the pattern libraries were rebuilt and extended with German and Russian
phrasings for this bilingual user base.

---

## Frontend: two views, one stream

A single Next.js app serves two audiences from the same SSE event stream:

- **Chat view** — clean, for the end user: chat bubble, source pills, optional
  full-text detail. No debug panels.
- **Pipeline Inspector** — for a technical reviewer: the same request rendered as
  a grid of agent cards (rewrite / retriever / CRAG grader / web search / router /
  generator) with live status, a security-verdict strip, and a per-card config
  drawer. An animated diagram walks the whole flow.

Notable choices:

- **Client-side chat history** in `localStorage` (keys per chat, FIFO cap 30):
  no server state to invalidate on redeploy, offline-readable, private by
  default.
- **`NEXT_PUBLIC_API_BASE` is a build-time arg.** The browser runs on the host
  network, so the client bundle is built with the reachable backend URL — the
  same mechanism that would point it at a cloud URL in a hosted deployment.
- **Cost display carries a `~`.** The per-answer figure is exact for the LLM and
  embedding tokens it counts, but excludes web-search and monthly infrastructure
  cost — the tilde and tooltip signal "defensible estimate", not "your invoice".

---

## Offline ingest

The corpus is indexed **once**, offline, before the app runs; the runtime
container never re-extracts. This is why the backend image stays slim (no GPU
libraries) while extraction can use heavy tooling on a workstation.

| Stage | Tooling |
|---|---|
| PDF → Markdown | `marker-pdf` + Surya OCR (GPU) — recovers scanned books plain parsers return empty |
| Chunking | recursive splitter, 2000 / 200 overlap, structure-aware separators |
| Dense embed | `voyage-multilingual-2` (1024d) |
| Sparse embed | BM25 |
| Index | Qdrant (hybrid + RRF) |

OCR is what unlocked scanned-only books that a plain PDF parser could not read.
Because that GPU path is needed only for ingest, the runtime dependency manifest
is deliberately kept separate and torch-free — the deployed image is ~500 MB
instead of multiple GB.

`scripts/ingest.py` implements this pipeline end to end. It defaults to plain
text extraction, which needs no torch at all, and takes `--extractor marker` for
the OCR path; marker is an optional install for exactly that reason. See
[the README](README.md#2-index-your-own-corpus) to run it.

---

## Graceful degradation

Four external services are individually optional at runtime:

| Service down | Behaviour |
|---|---|
| Redis | conversation memory + semantic cache disabled; queries still work |
| Web search (Tavily) | out-of-corpus questions get an honest corpus-only refusal |
| Observability (Opik) | traces fall back to a local JSONL file |
| Cache (toggle off) | both lookup and store are skipped |

None is load-bearing for "ask a question, get a grounded answer". Every external
client is wrapped at its integration point and gated by a config flag or a
sentinel (e.g. a missing API key disables the web search entirely).

---

## Performance shape

Measured over the SSE endpoint — the same path the browser uses — on a warm
container against the reference corpus, medians of repeated runs.

| Path | First token | Total |
|---|---|---|
| Semantic cache hit | ~0.4 s | ~0.4 s |
| Corpus answer | ~14 s | ~14.5 s |
| Web-fallback answer | ~10 s | ~10.5 s |
| Injection blocked | <50 ms | <50 ms (no LLM call) |

Time to first token is dominated by everything that happens *before*
generation, and inside that, by the CRAG grader: 5–15 s of it is the grader
reasoning over ten chunks. That is the price of the corrective step, and the
measurements below say it is not optional — see "the thinking budget is not a
free lunch" under [Generation](#generation-type-aware-cited-language-matched).
The cache exists precisely because this is expensive: a repeat question skips
all of it and returns in ~0.4 s.

---

## Considered, deferred

| Option | Why deferred |
|---|---|
| Parent-child retrieval | Larger effort; CRAG addressed the higher-impact failure modes first |
| Alternate chunk size (1200/120) | Would invalidate the evaluation baseline measured on 2000/200 |
| Cross-lingual cache hits | Technically worked, but disabled on purpose so answer language always matches question language |
| Autonomous adaptive routing | Cross-lingual routing isn't the dominant failure mode here; a controlled threshold fit better |
