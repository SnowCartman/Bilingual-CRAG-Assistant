"""Versioned prompt registry.

Tags every Opik trace/span with the *exact* prompt body that was used to
produce it -- so when we tweak a template later we can A/B answer quality
without losing the link between answer and prompt.

The bodies live next to where they are USED (in ``query_rewriter.py``,
``query_router.py``, ``templates.py``), so the registry imports those
constants rather than re-declaring them. Single source of truth.

A change to any prompt body MUST be paired with a version bump here --
the SHA-256 hash is just a backup signal in case someone forgets.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from app.prompts.templates import (
    _BASE_RULES,
    _FORMAT_COMPARATIVE,
    _FORMAT_EXPLANATORY,
    _FORMAT_FACTUAL,
)
from app.services.query_rewriter import REWRITE_SYSTEM_PROMPT
from app.services.query_router import ROUTER_SYSTEM_PROMPT


@dataclass(frozen=True)
class PromptVersion:
    name: str
    version: str
    body: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.body.encode("utf-8")).hexdigest()[:12]

    def tag(self) -> str:
        """Compact Opik-tag form: ``prompt:rewrite:v1:ab12cd34ef56``"""
        return f"prompt:{self.name}:{self.version}:{self.sha256}"


_REGISTRY: dict[str, PromptVersion] = {
    "rewrite":     PromptVersion("rewrite",     "v1", REWRITE_SYSTEM_PROMPT),
    "router":      PromptVersion("router",      "v1", ROUTER_SYSTEM_PROMPT),
    # Template entries store the FORMAT block only (not the full prompt) --
    # the full prompt is assembled at request time from _BASE_RULES + format
    # + context block, so storing the *whole* assembled prompt is misleading.
    # Versioning the format block + base rules separately lets us answer
    # "did the FACTUAL format change?" independently of "did the shared rules
    # change?".
    "base_rules":  PromptVersion("base_rules",  "v1", _BASE_RULES),
    "tpl_factual":     PromptVersion("tpl_factual",     "v1", _FORMAT_FACTUAL),
    "tpl_explanatory": PromptVersion("tpl_explanatory", "v1", _FORMAT_EXPLANATORY),
    "tpl_comparative": PromptVersion("tpl_comparative", "v1", _FORMAT_COMPARATIVE),
}


def get(name: str) -> PromptVersion:
    """Look up a prompt by name. Raises KeyError if unknown -- callers should
    only request prompts that exist (the registry is the contract)."""
    return _REGISTRY[name]


def template_tags_for(query_type: str) -> list[str]:
    """Return Opik-tags identifying which prompts produced a generated answer.

    Every generate-span gets base_rules + the type-specific format tag, so
    queries can be partitioned by template version in the Opik UI.
    """
    tpl_key = {
        "FACTUAL":     "tpl_factual",
        "EXPLANATORY": "tpl_explanatory",
        "COMPARATIVE": "tpl_comparative",
    }.get(query_type, "tpl_explanatory")  # mirrors templates.build_prompt default
    return [get("base_rules").tag(), get(tpl_key).tag()]


def all_versions() -> list[PromptVersion]:
    return list(_REGISTRY.values())
