"""Conditional query rewriting: makes a follow-up turn standalone.

Retrieval sees one query at a time, so "Und was ist mit Konjunktiv II?"
retrieves nothing useful on its own. Two steps:

  1. should_rewrite() -- cheap heuristic over the query. Decides whether to
     spend an LLM call at all; roughly 80% of turns in a typical session are
     already standalone and skip it. Fires on anaphora (conjunction openers,
     pronouns referring back) and on meta turns that steer the previous
     answer rather than asking something new.

  2. rewrite() -- ask the small model for a STANDALONE reformulation in the
     input's language. Returns (text, was_rewritten). Any LLM failure falls
     back to the original query: a slightly worse query beats a 500. The
     rewriter, router and CRAG grader all run on flash rather than
     flash-lite, which turned out to be 5xx-unstable under container load.

"Und was ist mit Konjunktiv II?" prefix-matches "und was" -> rewritten to
"Was ist Konjunktiv II?". The rewrite is emitted on the SSE stream, so the UI
can show the user what was actually searched for.
"""

from __future__ import annotations

import logging
from typing import Iterable

from google.genai import types as genai_types

from app.services.rag_pipeline import CostTracker, gemini_generate_guarded

# temperature=0 makes the rewriter deterministic. Two identical user inputs
# produce the same rewrite -> same canonical cache key -> the semantic cache
# can actually hit on repeat questions. Without this, gemini-2.5-flash's
# default temperature=1 produces different rewrites for the same input
# (observed live 2026-06-27: "und wie heist X auf schwedisch" rewrote to
# "Was ist X auf schwedisch?" then "Wie heisst X auf schwedisch?" -- the
# lexical-overlap cache guard correctly rejected the second as a different
# query even though semantically identical).
_REWRITE_CONFIG = genai_types.GenerateContentConfig(temperature=0.0)

log = logging.getLogger("query_rewriter")

# Cyrillic Russian markers are spelled out via chr() so the source stays
# pure ASCII (some Windows toolchains still read source as cp1252).
# Romanised forms appear in trailing comments.
def _cyr(*codes: int) -> str:
    return "".join(chr(c) for c in codes)


# Anaphora prefixes -- queries starting with one of these strongly imply a
# follow-up. Lowercased. Order doesn't matter; we check each.
_ANAPHORA_PREFIXES: tuple[str, ...] = (
    # German
    "und ", "und was", "und wie", "und warum",
    "aber ", "oder ", "auch ", "doch ",
    "wie steht es", "was ist mit", "wie ist es mit", "wie sieht's",
    "und wenn", "und im",
    # Russian
    _cyr(0x0430) + " ",                                                # "a "
    _cyr(0x0438) + " ",                                                # "i "
    _cyr(0x0438, 0x043B, 0x0438) + " ",                                # "ili "
    _cyr(0x0430, 0x20, 0x0447, 0x0442, 0x043E),                         # "a chto"
    _cyr(0x0430, 0x20, 0x043A, 0x0430, 0x043A),                         # "a kak"
    _cyr(0x0430, 0x20, 0x043F, 0x043E, 0x0447, 0x0435, 0x043C, 0x0443), # "a pochemu"
    _cyr(0x0438, 0x20, 0x0447, 0x0442, 0x043E),                         # "i chto"
    _cyr(0x0438, 0x20, 0x043A, 0x0430, 0x043A),                         # "i kak"
)

# Meta turns: the user is not asking a NEW question, they are steering the
# previous one ("that didn't answer", "do it again"). Without this the raw
# instruction became the retrieval query -- observed live 2026-09-09: "die
# Antwort wurde nicht generiert, bitte mache das noch einmal" retrieved
# chunks about the modal particle "doch" (lexical overlap with "noch") and
# the generator produced a confident answer about the wrong topic.
# Matched as SUBSTRINGS anywhere in the query, unlike the prefix list above,
# because the marker often sits mid-sentence. Guarded by _META_MAX_CHARS so
# a long genuine question that happens to contain "nochmal" is unaffected.
_META_MAX_CHARS = 120

_META_RETRY_MARKERS: tuple[str, ...] = (
    # German -- retry / repeat
    "nochmal", "noch mal", "noch einmal", "wiederhol", "erneut",
    "versuch es", "versuche es", "probier es",
    # German -- "you did not answer"
    "nicht generiert", "nicht beantwortet", "keine antwort",
    "nichts geantwortet", "hast nicht geantwortet", "kam keine",
    # English
    "try again", "once more", "repeat that", "no answer",
    # Russian (chr() for ASCII source safety)
    _cyr(0x043F, 0x043E, 0x0432, 0x0442, 0x043E, 0x0440),               # "povtor(i)"
    _cyr(0x0435, 0x0449, 0x0435, 0x20, 0x0440, 0x0430, 0x0437),         # "eshche raz"
    _cyr(0x0441, 0x043D, 0x043E, 0x0432, 0x0430),                       # "snova"
)


# Pronouns/demonstratives that often refer to the prior turn.
_ANAPHORA_TOKENS: frozenset[str] = frozenset({
    # German
    "es", "das", "dies", "diese", "dieser", "dieses",
    "sie", "er", "ihn", "ihm", "ihr", "deren", "dessen",
    # Russian (via chr() for ASCII source safety)
    _cyr(0x043E, 0x043D),                            # "on"
    _cyr(0x043E, 0x043D, 0x0430),                    # "ona"
    _cyr(0x043E, 0x043D, 0x043E),                    # "ono"
    _cyr(0x043E, 0x043D, 0x0438),                    # "oni"
    _cyr(0x044D, 0x0442, 0x043E),                    # "eto"
    _cyr(0x0442, 0x043E, 0x0442),                    # "tot"
    _cyr(0x0442, 0x0430),                            # "ta"
    _cyr(0x0442, 0x0435),                            # "te"
    _cyr(0x044D, 0x0442, 0x043E, 0x0442),            # "etot"
    _cyr(0x044D, 0x0442, 0x0430),                    # "eta"
})


REWRITE_SYSTEM_PROMPT = """You rewrite follow-up questions in a multilingual (German/Russian) language-learning chat into STANDALONE questions a retrieval system can use without the prior turns.

Rules:
- If the new query starts with a conjunction (Und / Aber / Oder / Auch / A / I / Ili), STRIP IT and reformulate as a full question. Example: "Und Konjunktiv II?" -> "Was ist Konjunktiv II?"
- Resolve pronouns (es, das, dies, sie, er, on, ona, eto, etc.) and elided subjects from the conversation.
- Inherit the grammatical structure of the PRIOR turn when applicable: if prior was "Was ist X?", the rewrite of "Und Y?" should be "Was ist Y?".
- Carry forward topical/scope qualifiers from the prior turn unless the new query explicitly overrides them. Examples of such qualifiers: "auf Schwedisch", "auf Englisch", "auf Russisch", "im Plural", "im Akkusativ", "im Praeteritum", "in der gehobenen Sprache", "po-russki", "po-anglijski".
- If the new message is NOT a question about the language material but an instruction ABOUT the previous answer -- it failed, it was empty, please repeat it, try again -- return the PREVIOUS user question unchanged. Never turn the complaint itself into a question.
- If such an instruction also asks for a modification (shorter, with examples, in Russian), apply that modification to the previous question.
- Keep the same language as the new query (German stays German, Russian stays Russian).
- Do NOT answer the question. Do NOT add commentary. Return ONLY the rewritten question, one line, no quotes.

Examples (German):
  Prior: "Was ist Konjunktiv I?"
  User:  "Und was ist mit Konjunktiv II?"
  -> Was ist Konjunktiv II?

  Prior: "Wie konjugiert man 'sein' im Praesens?"
  User:  "Und im Praeteritum?"
  -> Wie konjugiert man 'sein' im Praeteritum?

  Prior: "Erklaer mir das Passiv."
  User:  "Wann benutzt man es?"
  -> Wann benutzt man das Passiv?

  Prior: "Wie heisst 'Guten Morgen' auf Schwedisch?"
  User:  "Und was ist Guten Abend?"
  -> Wie heisst 'Guten Abend' auf Schwedisch?

  Prior: "Was heisst Hund auf Russisch?"
  User:  "Und Katze?"
  -> Was heisst Katze auf Russisch?

  Prior: "Erklaer mir den Akkusativ mit Beispielen."
  User:  "Und beim Dativ?"
  -> Erklaer mir den Dativ mit Beispielen.

Examples (instructions about the previous answer):
  Prior: "Was ist der Unterschied zwischen wenn, wann und warum?"
  User:  "die Antwort wurde nicht generiert, bitte mache das noch einmal"
  -> Was ist der Unterschied zwischen wenn, wann und warum?

  Prior: "Erklaer mir den Konjunktiv II."
  User:  "Das war zu lang, bitte kuerzer nochmal."
  -> Erklaer mir den Konjunktiv II kurz.
"""


def should_rewrite(query: str, prior_turns: Iterable[dict]) -> bool:
    """Heuristic gate: True iff we should burn an LLM call to rewrite."""
    turns = list(prior_turns)
    if not turns:
        return False
    q = query.strip().lower()
    if not q:
        return False
    if len(q) <= _META_MAX_CHARS and any(m in q for m in _META_RETRY_MARKERS):
        return True
    for prefix in _ANAPHORA_PREFIXES:
        if q.startswith(prefix):
            return True
    first_tokens = q.split()[:5]
    if any(tok.strip("?.,!:;\"'") in _ANAPHORA_TOKENS
           for tok in first_tokens):
        return True
    return False


def _format_prior_turns(turns: list[dict]) -> str:
    lines = []
    for i, t in enumerate(turns, 1):
        q = (t.get("q") or "").strip()
        a = (t.get("a") or "").strip()
        # Trim long answers -- the rewriter only needs the topic, not the answer body.
        a_short = a[:400] + ("..." if len(a) > 400 else "")
        lines.append(f"Turn {i}:\n  User: {q}\n  Assistant: {a_short}")
    return "\n\n".join(lines)


def rewrite(
    query: str,
    prior_turns: list[dict],
    gemini_client,
    model: str,
    cost: CostTracker,
    *,
    call_timeout_s: float = 60.0,
) -> tuple[str, bool]:
    """Conditional rewrite. Returns (text, was_rewritten).

    Heuristic-skipped -> (query, False). LLM-failed -> (query, False) with
    a warning log. Successful rewrite -> (rewritten_text, True).
    """
    if not should_rewrite(query, prior_turns):
        return query, False

    prompt = (
        REWRITE_SYSTEM_PROMPT
        + "\n\nConversation so far:\n"
        + _format_prior_turns(prior_turns)
        + f"\n\nNew query: {query}\n\nStandalone rewrite:"
    )
    try:
        res = gemini_generate_guarded(
            gemini_client, model, prompt,
            config=_REWRITE_CONFIG,
            max_attempts=3,                 # rewriter retries less aggressively
            call_timeout_s=call_timeout_s,
            label="rewrite",
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("rewrite failed (%s) -- using original query", exc)
        return query, False

    rewritten = (res.get("text") or "").strip().strip('"').strip("'")
    if not rewritten or len(rewritten) > 500:
        # Sanity: empty or absurdly long => model misbehaved, drop back.
        log.warning("rewrite produced suspicious output (%d chars) -- using original",
                    len(rewritten))
        return query, False

    cost.add_gemini(model, res["prompt_tokens"], res["output_tokens"])
    log.info("rewrite: '%s' -> '%s'", query[:60], rewritten[:60])
    return rewritten, True
