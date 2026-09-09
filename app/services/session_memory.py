"""Per-session conversation history backed by Redis (sliding window N=4).

Storage layout:
    KEY:    session:{session_id}:turns     -- a Redis List
    VALUE:  JSON-encoded {"q": "...", "a": "..."} per turn, ensure_ascii=True
    ORDER:  LPUSH at head, so list[0] = newest turn

On every write: LTRIM 0 (N-1) caps the list to N entries, EXPIRE refreshes the
TTL so a live conversation doesn't expire mid-thread.

Why one entry per turn (instead of separate q/a messages): keeps the rewriter
prompt simple ("here are prior turns") and matches the demo story
("conversation memory" not "message history").

The semantic cache uses a separate key pattern (a Redis HNSW index), so this
file stays focused on the conversation-memory concern only.
"""

from __future__ import annotations

import json
import logging

import redis

from app.config import Settings

log = logging.getLogger("session_memory")


def _key(session_id: str) -> str:
    return f"session:{session_id}:turns"


class SessionMemory:
    """Redis-backed sliding-window conversation memory.

    Constructed once at FastAPI lifespan startup. Falls back to a no-op
    instance (client=None) if redis_host isn't configured -- so the service
    can still serve stateless queries without Redis. add_turn/get_recent_turns
    on a no-op instance simply skip the operation.
    """

    def __init__(self, settings: Settings) -> None:
        self.window_size = settings.conversation_window_size
        self.session_ttl = settings.conversation_session_ttl
        if not settings.redis_host:
            log.warning("SessionMemory: REDIS_HOST not set -- memory disabled")
            self.client = None
            return
        self.client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            username=settings.redis_username or "default",
            password=settings.redis_password,
            ssl=settings.redis_ssl,
            decode_responses=True,
            socket_timeout=5,
            socket_connect_timeout=5,
        )
        log.info("SessionMemory: connected to redis %s:%s (ssl=%s, window=%d, ttl=%ds)",
                 settings.redis_host, settings.redis_port, settings.redis_ssl,
                 self.window_size, self.session_ttl)

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def ping(self) -> bool:
        """Health probe -- True if Redis responds. Never raises."""
        if self.client is None:
            return False
        try:
            return bool(self.client.ping())
        except Exception:  # noqa: BLE001
            return False

    def add_turn(self, session_id: str, question: str, answer: str) -> None:
        """Append (q, a) at the head, trim to window_size, refresh TTL."""
        if self.client is None or not session_id:
            return
        entry = json.dumps({"q": question, "a": answer}, ensure_ascii=True)
        key = _key(session_id)
        pipe = self.client.pipeline()
        pipe.lpush(key, entry)
        pipe.ltrim(key, 0, self.window_size - 1)
        pipe.expire(key, self.session_ttl)
        pipe.execute()

    def get_recent_turns(self, session_id: str) -> list[dict]:
        """Return the most recent up-to-window_size turns, oldest first.

        Oldest-first reads more naturally in the rewrite prompt
        ("Earlier in the conversation: ... -> User now asks: ...").
        Returns [] if Redis is disabled or the session has no history.
        """
        if self.client is None or not session_id:
            return []
        raw = self.client.lrange(_key(session_id), 0, self.window_size - 1)
        out: list[dict] = []
        # LPUSH writes newest at head, so raw[0] = newest; reverse for chrono.
        for r in reversed(raw):
            try:
                out.append(json.loads(r))
            except json.JSONDecodeError:
                log.warning("session_memory: skipping malformed entry in %s", session_id)
        return out

    def clear(self, session_id: str) -> None:
        if self.client is None or not session_id:
            return
        self.client.delete(_key(session_id))
