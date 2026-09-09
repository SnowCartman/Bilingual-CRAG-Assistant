"""Semantic answer cache backed by Redis HNSW (RediSearch vector index).

Storage layout:
    INDEX:   cache:idx (HNSW, COSINE distance, 1024d float32)
    KEY:     cache:entry:{sha256(query)[:16]}    -- a Redis Hash
    FIELDS:  query_text    TEXT (the canonicalized standalone query)
             embedding     VECTOR (1024d float32, L2-normalised)
             answer        TEXT (generator output)
             contexts      TEXT (JSON-serialized ContextItem list, ensure_ascii=True)
             timestamp     NUMERIC (unix epoch seconds)
    TTL:     per-entry EXPIRE (refreshed on every store of the same key)

Lookup semantics:
    KNN top-1 over the index; if the returned cosine *distance* is <= threshold,
    that's a cache hit. RediSearch's `__score` from `KNN` IS the cosine distance
    when DISTANCE_METRIC=COSINE, so smaller = more similar (0 = identical).

Embedding reuse:
    Caller is expected to pass the SAME embedding that will be used for hybrid
    retrieval on a miss -- so the cache lookup is free (no extra Voyage call).

Graceful degradation:
    If redis_host isn't set OR the cache_distance_threshold is non-positive,
    .enabled is False and lookup/store become no-ops. Service still serves
    queries, just without the cache speedup.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time

import numpy as np
import redis
from redis.commands.search.field import VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.exceptions import ResponseError

from app.config import Settings

log = logging.getLogger("semantic_cache")

INDEX_NAME = "cache:idx"
KEY_PREFIX = "cache:entry:"

# Lexical-overlap guard. The semantic cache uses cosine distance on
# voyage-multilingual-2 embeddings; for short queries that share a frame
# ("Wie heisst X auf Schwedisch?") the embeddings collapse the X away --
# "guten Morgen" vs "guten Abend" land within the distance threshold.
# After a vector hit we re-check that the new and cached queries share
# enough content tokens. We only apply this guard when BOTH queries are
# in the same script (Latin OR Cyrillic) so cross-lingual RU<->DE
# paraphrases -- which the vector cache catches by design -- still hit.
_LATIN_TOKEN_RE = re.compile(r"[a-z\u00e4\u00f6\u00fc]{3,}")
_CYRILLIC_TOKEN_RE = re.compile(r"[\u0430-\u044f\u0451]{3,}")
_ESZETT = chr(0x00df)
_LEXICAL_JACCARD_MIN = 0.7


def _u(*codes: int) -> str:
    """Codepoint -> string. Used to keep non-ASCII stopwords out of source."""
    return "".join(chr(c) for c in codes)


# Stopwords are FUNCTION words only. Pronouns (mir/dir/ihm/ich) and short
# adjectives (gut/neu/alt) intentionally stay -- they carry meaning in a
# translation query ("Wie heisst 'mir geht es gut' auf Schwedisch" vs.
# "Wie heisst 'dir geht es gut' auf Schwedisch" should NOT collapse).
_STOPWORDS_DE = frozenset({
    "der", "die", "das", "den", "dem", "des",
    "ein", "eine", "einen", "einem", "einer",
    "ist", "sind", "war", "waren", "wird", "werden", "hat", "haben",
    "und", "oder", "aber", "denn", "doch", "noch",
    "auf", "von", "vor", "zur", "zum", "ueber",
    _u(0x00fc, 0x62, 0x65, 0x72),
    "bei", "mit", "nach", "aus", "ohne", "hin", "weg",
    "wie", "was", "wer", "wann", "wenn",
})
_STOPWORDS_RU = frozenset({
    _u(0x044d, 0x0442, 0x043e),
    _u(0x043a, 0x0430, 0x043a),
    _u(0x0447, 0x0442, 0x043e),
    _u(0x0433, 0x0434, 0x0435),
    _u(0x043a, 0x043e, 0x0433, 0x0434, 0x0430),
    _u(0x043f, 0x043e, 0x0447, 0x0435, 0x043c, 0x0443),
    _u(0x043f, 0x043e),
    _u(0x043d, 0x0430),
    _u(0x0432, 0x043e),
    _u(0x0438, 0x0437),
    _u(0x043e, 0x0442),
    _u(0x0434, 0x043e),
    _u(0x0437, 0x0430),
    _u(0x0438, 0x043b, 0x0438),
    _u(0x0442, 0x0430, 0x043a),
    _u(0x0442, 0x043e, 0x0436, 0x0435),
    _u(0x0442, 0x0430, 0x043a, 0x0436, 0x0435),
    _u(0x0435, 0x0449, 0x0451),
    _u(0x0443, 0x0436, 0x0435),
})
_STOPWORDS = _STOPWORDS_DE | _STOPWORDS_RU


def _key_for(query_text: str) -> str:
    """16 hex chars of sha256 -- short enough for logging, collision-safe enough
    for a TTL'd cache that maxes out at a few hundred live entries."""
    h = hashlib.sha256(query_text.encode("utf-8")).hexdigest()[:16]
    return f"{KEY_PREFIX}{h}"


def _to_bytes(vec) -> bytes:
    """Serialize a 1024d float vector to the raw float32 bytes RediSearch wants."""
    arr = np.asarray(vec, dtype=np.float32)
    return arr.tobytes()


class SemanticCache:
    """Redis-backed semantic answer cache (RediSearch HNSW + COSINE).

    Construct once at FastAPI lifespan startup. Lookup is a single FT.SEARCH
    KNN call (sub-millisecond on Redis Cloud); store is a HSET + EXPIRE.
    """

    def __init__(self, settings: Settings) -> None:
        self.dim = settings.cache_embed_dimension
        self.threshold = settings.cache_distance_threshold
        self.ttl = settings.cache_ttl

        if not settings.redis_host or self.threshold <= 0:
            log.warning(
                "SemanticCache: disabled (host=%s, threshold=%s)",
                bool(settings.redis_host), self.threshold,
            )
            self.client = None
            return

        self.client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            username=settings.redis_username or "default",
            password=settings.redis_password,
            ssl=settings.redis_ssl,
            decode_responses=False,    # vectors are raw bytes; we decode strings ourselves
            socket_timeout=5,
            socket_connect_timeout=5,
        )
        self._ensure_index()
        log.info(
            "SemanticCache: connected to redis %s:%s (ssl=%s, dim=%d, threshold=%.3f, ttl=%ds)",
            settings.redis_host, settings.redis_port, settings.redis_ssl,
            self.dim, self.threshold, self.ttl,
        )

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def ping(self) -> bool:
        if self.client is None:
            return False
        try:
            return bool(self.client.ping())
        except Exception:  # noqa: BLE001
            return False

    # ---- index management ---------------------------------------------------

    def _ensure_index(self) -> None:
        """Create the HNSW index if it doesn't exist. Idempotent.

        Only embedding is indexed -- query_text/answer/contexts/timestamp are
        stored as plain hash fields and returned via return_fields on lookup
        (RediSearch returns any hash field listed in RETURN, indexed or not).
        Keeping the schema minimal avoids RediSearch's restriction that
        non-indexable text fields must be SORTABLE."""
        assert self.client is not None
        schema = (
            VectorField(
                "embedding",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": self.dim,
                    "DISTANCE_METRIC": "COSINE",
                },
            ),
        )
        definition = IndexDefinition(prefix=[KEY_PREFIX], index_type=IndexType.HASH)
        try:
            self.client.ft(INDEX_NAME).create_index(schema, definition=definition)
            log.info("SemanticCache: created index %s", INDEX_NAME)
        except ResponseError as exc:
            if "Index already exists" in str(exc):
                log.info("SemanticCache: index %s already exists", INDEX_NAME)
                return
            raise

    # ---- public API ---------------------------------------------------------

    def lookup(self, embedding, query_text: str | None = None) -> dict | None:
        """KNN top-1 lookup. Returns the cached entry as dict if distance is
        within threshold; otherwise None. Never raises -- on Redis errors we
        log and return None so the pipeline falls through to a normal retrieve.

        Uses execute_command rather than the high-level Query/Result classes
        because redis-py 8.0 returns FT.SEARCH as a RESP3 dict that those
        classes don't parse correctly (res.docs comes back empty even when
        Redis found the doc -- verified against raw FT.SEARCH 2026-06-17).

        When ``query_text`` is provided we additionally enforce a same-script
        lexical-overlap guard against the cached query: voyage-multilingual-2
        embeddings collapse short same-frame queries ("guten Morgen auf
        Schwedisch" vs "guten Abend auf Schwedisch") into the threshold,
        producing wrong-answer hits. Cross-script paraphrases (RU<->DE)
        bypass the guard so the cross-lingual cache still works.

        Returns:
            None on miss / disabled / error / lexical-guard reject.
            {"query_text": str, "answer": str, "contexts": list[dict],
             "distance": float, "cached_at": float} on hit.
        """
        if self.client is None:
            return None
        try:
            raw = self.client.execute_command(
                "FT.SEARCH", INDEX_NAME,
                "*=>[KNN 1 @embedding $vec AS distance]",
                "PARAMS", "2", "vec", _to_bytes(embedding),
                "RETURN", "4", "query_text", "answer", "contexts", "distance",
                "SORTBY", "distance",
                "DIALECT", "2",
                "LIMIT", "0", "1",
            )
        except ResponseError as exc:
            log.warning("SemanticCache: lookup ResponseError: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001
            log.warning("SemanticCache: lookup failed: %s", exc)
            return None

        attrs = _first_doc_attributes(raw)
        if attrs is None:
            return None
        try:
            distance = float(_decode(attrs.get(b"distance") or attrs.get("distance")))
        except (TypeError, ValueError):
            log.warning("SemanticCache: lookup result missing distance field")
            return None
        if distance > self.threshold:
            return None

        # The store path keeps contexts in the hash but not in the index --
        # FT.SEARCH RETURN can still surface it, but on some RESP3 builds the
        # non-indexed fields don't round-trip. Fall back to a plain HGET on
        # the doc id, which always works.
        doc_id = _decode(_first_doc_id(raw))
        cached_query_text = _decode(
            attrs.get(b"query_text") or attrs.get("query_text") or b""
        )
        # Lexical-overlap guard: reject "guten Morgen" vs "guten Abend"
        # style false positives. Only fires when both queries share script.
        if query_text and cached_query_text and not _lexical_overlap_ok(
            query_text, cached_query_text,
        ):
            log.info(
                "SemanticCache: lexical guard rejected hit "
                "(new='%s', cached='%s', distance=%.4f)",
                query_text[:60], cached_query_text[:60], distance,
            )
            return None
        query_text = cached_query_text
        answer_raw = attrs.get(b"answer") or attrs.get("answer")
        contexts_raw = attrs.get(b"contexts") or attrs.get("contexts")
        if (answer_raw is None or contexts_raw is None) and doc_id:
            fallback = self.client.hmget(doc_id, ["answer", "contexts", "query_text"])
            answer_raw = answer_raw or fallback[0]
            contexts_raw = contexts_raw or fallback[1]
            if not query_text:
                query_text = _decode(fallback[2] or b"")
        try:
            contexts = json.loads(_decode(contexts_raw))
        except (TypeError, json.JSONDecodeError):
            contexts = []
        cached_at = 0.0
        if doc_id:
            ts = self.client.hget(doc_id, "timestamp")
            try:
                cached_at = float(_decode(ts)) if ts is not None else 0.0
            except ValueError:
                cached_at = 0.0
        return {
            "query_text": query_text,
            "answer": _decode(answer_raw or b""),
            "contexts": contexts,
            "distance": distance,
            "cached_at": cached_at,
        }

    def store(self, query_text: str, embedding, answer: str,
              contexts: list[dict]) -> None:
        """Persist (query, embedding, answer, contexts) with TTL. Never raises."""
        if self.client is None:
            return
        key = _key_for(query_text)
        try:
            mapping = {
                "query_text": query_text.encode("utf-8"),
                "embedding": _to_bytes(embedding),
                "answer": answer.encode("utf-8"),
                "contexts": json.dumps(contexts, ensure_ascii=True).encode("utf-8"),
                "timestamp": str(int(time.time())).encode("utf-8"),
            }
            pipe = self.client.pipeline()
            pipe.hset(key, mapping=mapping)
            pipe.expire(key, self.ttl)
            pipe.execute()
        except Exception as exc:  # noqa: BLE001
            log.warning("SemanticCache: store failed for %s: %s", key, exc)

    def clear(self) -> None:
        """Delete every cache entry (matches KEY_PREFIX*). Test helper."""
        if self.client is None:
            return
        cursor = 0
        while True:
            cursor, keys = self.client.scan(cursor=cursor, match=f"{KEY_PREFIX}*", count=200)
            if keys:
                self.client.delete(*keys)
            if cursor == 0:
                break


def _decode(value) -> str:
    """Tolerate both bytes and str return shapes from FT.SEARCH."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _first_doc_id(raw):
    """Pull the first doc id out of FT.SEARCH's response. Handles both the
    RESP3 dict shape (redis-py 8.0 default) and the RESP2 flat-array shape."""
    if isinstance(raw, dict):
        results = raw.get(b"results") or raw.get("results") or []
        if not results:
            return None
        first = results[0]
        if isinstance(first, dict):
            return first.get(b"id") or first.get("id")
        return None
    if isinstance(raw, list) and len(raw) >= 2:
        # [total, key, [field, value, ...], key, [field, value, ...], ...]
        return raw[1]
    return None


def _content_tokens(text: str) -> tuple[set[str], set[str]]:
    """Return (latin_tokens, cyrillic_tokens) of length >= 3, lowercased,
    normalised, and filtered through _STOPWORDS.

    Normalisation:
      lowercase + eszett->ss. The rewriter occasionally produces "heisst"
      and "heisst-with-eszett" interchangeably even with temperature=0,
      so the lexical key has to collapse them.
    Length 3 (not 4) keeps content-bearing short words like "mir", "dir",
    "gut"; the stopword filter handles function-words separately.
    """
    lowered = text.lower().replace(_ESZETT, "ss")
    latin = {t for t in _LATIN_TOKEN_RE.findall(lowered) if t not in _STOPWORDS}
    cyr = {t for t in _CYRILLIC_TOKEN_RE.findall(lowered) if t not in _STOPWORDS}
    return latin, cyr


def _lexical_overlap_ok(new_q: str, cached_q: str) -> bool:
    """Same-script Jaccard guard + cross-script rejection.

    Returns True iff:
      - either query has no content tokens (nothing to compare), OR
      - both queries use the SAME dominant script AND share enough content
        tokens (Jaccard >= _LEXICAL_JACCARD_MIN).

    Cross-script hits are REJECTED on purpose. The semantic cache stores
    the answer in whatever language was first asked; serving a Russian
    answer to a German question (or vice versa) is the wrong UX even if
    the embeddings agree that the meaning matches. The generator's
    language-detection rule (_BASE_RULES) only fires on a fresh
    pipeline run -- so a cross-language follow-up MUST miss the cache so
    that rule re-runs.
    """
    new_lat, new_cyr = _content_tokens(new_q)
    cached_lat, cached_cyr = _content_tokens(cached_q)
    if not (new_lat or new_cyr) or not (cached_lat or cached_cyr):
        return True
    new_is_latin = len(new_lat) >= len(new_cyr)
    cached_is_latin = len(cached_lat) >= len(cached_cyr)
    if new_is_latin != cached_is_latin:
        return False
    a, b = (new_lat, cached_lat) if new_is_latin else (new_cyr, cached_cyr)
    if not a or not b:
        return True
    jaccard = len(a & b) / len(a | b)
    return jaccard >= _LEXICAL_JACCARD_MIN


def _first_doc_attributes(raw):
    """Pull the first doc's returned attributes from FT.SEARCH. Returns a dict
    keyed by attribute name (bytes), or None on empty results."""
    if isinstance(raw, dict):
        results = raw.get(b"results") or raw.get("results") or []
        if not results:
            return None
        first = results[0]
        if isinstance(first, dict):
            return first.get(b"extra_attributes") or first.get("extra_attributes") or {}
        return None
    if isinstance(raw, list) and len(raw) >= 3:
        flat = raw[2]
        # flat is [field, value, field, value, ...]
        attrs = {}
        for i in range(0, len(flat) - 1, 2):
            attrs[flat[i]] = flat[i + 1]
        return attrs
    return None
