"""Tavily web-search fallback for CRAG.

Triggered by ``ContextGrader.grade`` when the retrieved DB chunks score below
``settings.crag_threshold``. Returns up to ``max_results`` web docs in the
SAME shape as Qdrant chunks (so the downstream filter / classify / generate
pipeline does not need to know the origin).

Design choices:

- **Direct REST via ``requests``**, no ``tavily-python`` dependency. One HTTP
  POST, transparent retry, no version-drift across the package ecosystem
  (one less line in ``requirements-backend.txt``).

- **``topic='general'`` + ``search_depth='basic'``.** Language-learning
  questions are not news; basic depth is ~5x cheaper than ``advanced`` and
  returns the same domain coverage for our queries (Wiktionary, language
  blogs, dictionaries).

- **Output shape mirrors Qdrant chunks.** The pipeline's existing
  ``_filter_documents`` (DocumentGuard) operates on ``{"text": ...,
  "source_file": ...}`` dicts -- we hand it the same shape, just with
  ``origin="tavily"``. No special-casing needed downstream of the fallback.

- **Graceful degradation when ``TAVILY_API_KEY`` is missing or the request
  fails.** Returns ``[]``, logs a warning. CRAG fallback path then degrades
  to a corpus-only honest refusal -- the behaviour of a build without CRAG.
  Means dev / Cloud-Run / Compose can all run without Tavily configured,
  same graceful-degradation contract as Redis and Opik.
"""

from __future__ import annotations

import logging
from urllib.parse import urlparse

import requests

log = logging.getLogger(__name__)


_TAVILY_URL = "https://api.tavily.com/search"
_TAVILY_TIMEOUT_S = 15.0
_CONTENT_HEAD_CHARS = 1500     # cap per-result text we hand to Gemini


class TavilyClient:
    """Single Tavily client per pipeline instance.

    ``api_key=None`` means Tavily is disabled -- ``search()`` returns ``[]``
    silently. The CRAG fallback path then has no web docs to mix in, and
    the answer falls back to the regular generator with whatever DB chunks
    survived the grader.
    """

    def __init__(
        self,
        api_key: str | None,
        max_results: int = 3,
        topic: str = "general",
        search_depth: str = "basic",
        timeout_s: float = _TAVILY_TIMEOUT_S,
    ) -> None:
        self.api_key = api_key
        self.max_results = max_results
        self.topic = topic
        self.search_depth = search_depth
        self.timeout_s = timeout_s
        self.enabled = bool(api_key)
        if not self.enabled:
            log.info("TavilyClient: disabled (no TAVILY_API_KEY)")

    def search(self, query: str) -> list[dict]:
        """Run a single Tavily search, return Qdrant-shaped chunks."""
        if not self.enabled:
            return []
        payload = {
            "api_key": self.api_key,
            "query": query,
            "topic": self.topic,
            "search_depth": self.search_depth,
            "max_results": self.max_results,
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }
        try:
            resp = requests.post(_TAVILY_URL, json=payload, timeout=self.timeout_s)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as exc:
            log.warning("TavilyClient: request failed (%s)", exc)
            return []
        except ValueError as exc:
            # JSON decode failed -- log raw status for diagnosis.
            log.warning("TavilyClient: non-JSON response (%s)", exc)
            return []

        raw_results = data.get("results") if isinstance(data, dict) else None
        if not isinstance(raw_results, list):
            log.warning("TavilyClient: response missing 'results' list; data=%r",
                        data)
            return []
        out: list[dict] = []
        for i, r in enumerate(raw_results, start=1):
            if not isinstance(r, dict):
                continue
            text = (r.get("content") or "").strip()
            if not text:
                continue
            if len(text) > _CONTENT_HEAD_CHARS:
                text = text[:_CONTENT_HEAD_CHARS] + "..."
            url = r.get("url") or ""
            title = (r.get("title") or "").strip() or _domain_from(url) or "Web result"
            try:
                score = float(r.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            out.append({
                "rank": i,
                "score": round(score, 5),
                "source_file": title,
                "page": None,
                "page_end": None,
                "language": "?",        # Tavily does not report language
                "text": text,
                "rerank_score": None,
                "origin": "tavily",
                "url": url or None,
            })
        log.info("TavilyClient: query=%r -> %d results", query[:60], len(out))
        return out


def _domain_from(url: str) -> str:
    """Best-effort netloc extractor used as a title fallback."""
    if not url:
        return ""
    try:
        parsed = urlparse(url)
    except ValueError:
        return ""
    host = parsed.netloc or ""
    return host.lstrip("www.")
