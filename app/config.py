"""Application settings -- loaded from a .env file at the repo root.

Required keys:
    VOYAGE_API_KEY, QDRANT_URL, QDRANT_API_KEY, GEMINI_API_KEY

Optional keys (the service still starts without them, with the matching
feature disabled):
    REDIS_URL, REDIS_PASSWORD          -- conversation memory + semantic cache
    OPIK_API_KEY, OPIK_WORKSPACE        -- observability
    TAVILY_API_KEY                      -- CRAG web-search fallback

If code tries to use an optional key that isn't set, Pydantic raises a clear
error -- intentional, so silent fallthroughs don't hide config bugs.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# app/config.py -> app -> repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"


class Settings(BaseSettings):
    """Single source of truth for runtime config.

    Field names are lowercase but Pydantic Settings reads upper-case env-vars
    by default (case_sensitive=False). So VOYAGE_API_KEY in .env -> voyage_api_key.
    """

    model_config = SettingsConfigDict(
        env_file=str(ENV_PATH),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Required: retrieval + generation ----------------------------------
    voyage_api_key: str = Field(..., description="Voyage AI key for embed + rerank")
    qdrant_url: str = Field(..., description="Qdrant Cloud URL")
    qdrant_api_key: str = Field(..., description="Qdrant Cloud API key")
    gemini_api_key: str = Field(..., description="Google AI Studio key for Gemini")

    # --- Pipeline config (defaults = the configuration that won the
    # --- retrieval benchmark; see EVALUATION.md) ---------------------------
    qdrant_collection: str = "my_books_hybrid"   # override via QDRANT_COLLECTION
    embed_model: str = "voyage-multilingual-2"        # 1024d, benchmark winner
    rerank_model: str = "rerank-2.5"                   # Voyage rerank-2.5
    gen_model: str = "gemini-2.5-flash"
    # Low (not zero) generation temperature: grounded RAG answers should be
    # stable across identical inputs. gemini's default temperature=1 made the
    # generator occasionally (a) refuse on context that DID contain the answer
    # and (b) ignore the "reply in the question's language" rule and answer in
    # the other language. 0.3 keeps phrasing natural while removing that
    # flakiness. Set GEN_TEMPERATURE to override. (rewriter/router/grader are
    # already pinned to 0.0.)
    gen_temperature: float = 0.3
    # gemini-2.5-flash reasons before answering by default. On a grounded RAG
    # prompt that reasoning is largely wasted -- the evidence is already in the
    # context block -- but it dominates latency and, on the longest COMPARATIVE
    # prompts, pushed the call past the upstream deadline (measured: 17.2s and
    # ~2 575 thinking tokens with thinking on vs 4.0s without, same answer
    # quality and citations; one run died on a 504 DEADLINE_EXCEEDED after
    # ~120s and surfaced to the user as "no answer"). 0 disables it, a positive
    # value caps it, -1 hands the decision back to the model.
    # Set GEN_THINKING_BUDGET to override.
    gen_thinking_budget: int = 0

    top_k: int = 10                  # final contexts handed to the generator
    dense_prefetch: int = 100        # dense candidates before RRF
    sparse_prefetch: int = 100       # sparse (BM25) candidates before RRF
    rrf_prefetch: int = 100          # RRF-fused candidates handed to reranker

    # --- Gemini hang-guard (ported from an earlier retrieval prototype) ----------
    gemini_http_timeout_s: float = 120.0
    gemini_max_attempts: int = 12
    gemini_call_timeout_s: float = 150.0

    # --- Conversation Memory + Query Rewriting ---------------------
    redis_host: str | None = None
    redis_port: int = 6379
    redis_username: str | None = None
    redis_password: str | None = None
    redis_ssl: bool = False                   # some managed Redis instances need this false; flip if the SSL handshake times out
    conversation_window_size: int = 4         # number of (q, a) turns kept per session
    conversation_session_ttl: int = 3600      # seconds; refreshed on every write
    # All three helper LLMs (rewriter / router / crag-grader) standardised on
    # gemini-2.5-flash. flash-lite was cheaper but went 5xx-unstable under
    # container load; flash holds up + costs
    # ~+$0.0004/request, acceptable for a demo workload.
    rewrite_model: str = "gemini-2.5-flash"

    # --- Query Routing (FACTUAL / EXPLANATORY / COMPARATIVE) -------
    router_model: str = "gemini-2.5-flash"   # standardised w/ rewriter + crag

    # --- Semantic cache ----------------------------------------------------
    sparse_model: str = "Qdrant/bm25"
    cache_embed_model: str = "voyage-multilingual-2"
    cache_embed_dimension: int = 1024
    # 0.20 is the calibrated value, not a guess: over 10 paraphrase pairs and
    # 45 distinct-question pairs it was the lowest threshold reaching TPR 1.0 at
    # FPR 0.0 (scripts/cache_threshold_experiment.py). The nearest distinct
    # question sat at 0.396, so there is a wide margin above it.
    cache_distance_threshold: float = 0.20
    cache_ttl: int = 3600

    # --- Observability (Opik) + CRAG web fallback (Tavily) -----------------
    opik_api_key: str | None = None
    opik_workspace: str | None = None
    opik_project_name: str | None = None
    tavily_api_key: str | None = None
    # CRAG grader thresholds, calibrated against the deterministic grader at
    # temperature=0 -- see ARCHITECTURE.md "Self-correction: CRAG + web fallback".
    #  crag_threshold        -- when the MAX per-chunk relevance is below
    #                           this, the Tavily web-search fallback fires.
    #                           With the grader at temperature=0 (set in
    #                           crag_critic.py for demo reliability), the
    #                           deterministic scores are blond q3 max=0.50
    #                           (in-corpus) vs schwedisch max=~0.20-0.40
    #                           (out-of-corpus). 0.45 separates them: blond
    #                           stays DB-sufficient, schwedisch fires
    #                           Tavily. Original temperature=1 reading was
    #                           0.65 but noisy -- intermittent demo
    #                           failures when blond's max randomly rolled
    #                           below 0.65.
    #  per_chunk_threshold   -- chunks with per-chunk score below this are
    #                           dropped from the MIXED context (kept-DB +
    #                           Tavily) handed to gen. NOT applied when the
    #                           grader says DB sufficient. Kept at 0.5
    #                           (calibrated) so schwedisch mix-mode
    #                           shows only Tavily web cards (all DB chunks
    #                           score < 0.5 against an out-of-corpus query
    #                           and get correctly dropped).
    #  tavily_max_results    -- 5, not 3. Phrase-translation questions ("was
    #                           heisst <ganzer Satz> auf X") need a phrasebook
    #                           or dictionary page carrying a sentence pattern,
    #                           and those consistently rank 4th-5th behind bare
    #                           single-word dictionary hits: at 3 the answer was
    #                           a refusal, at 5 the pattern was present in every
    #                           probe run. Costs one extra source card in the UI
    #                           row and ~3k prompt characters.
    crag_threshold: float = 0.45
    per_chunk_threshold: float = 0.5
    #  grader_thinking_budget -- LEFT AT -1 (model decides) on purpose, unlike
    #                           the generator. Disabling it made the grader
    #                           ~4x faster but broke rule 6 of _GRADER_SYSTEM,
    #                           the third-language check: German grammar pages
    #                           about "wohin" scored 0.70-0.80 against "was
    #                           heisst <phrase> auf Griechisch", so no web
    #                           search fired and the answer came from corpus
    #                           chunks that cannot contain Greek. Measured over
    #                           8 calibration queries -- budget 0: 2 wrong,
    #                           128: 1 wrong, 512: 1 wrong (and it pushed the
    #                           tight in-corpus case below the threshold),
    #                           model-decided: 0 wrong at ~10.7s. That rule is
    #                           a conditional the model has to reason through;
    #                           it is not something the prompt can shorten.
    grader_thinking_budget: int = -1
    tavily_max_results: int = 5
    crag_model: str = "gemini-2.5-flash"      # standardised w/ rewriter + router

    # --- App-level ---------------------------------------------------------
    app_version: str = "0.1.0"
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton accessor -- Pydantic re-reads the env every call otherwise."""
    return Settings()
