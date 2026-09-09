"""Regex-based security guards.

Implements 3 defence layers as plain classes (no framework decorators) and
extends the InputGuard / DocumentGuard pattern libraries with German and
Russian phrasings so this product's bilingual user base is covered. The
three-layer approach was inspired by material studied during an AI
engineering course.

Three layers:

    InputGuard       -- block obvious prompt-injection at /query entry
    DocumentGuard    -- filter retrieval-poisoned chunks (Qdrant + Tavily)
    OutputValidator  -- redact PII from the generated answer

Design choices:

- **Pure regex, no LLM.** Deterministic, sub-millisecond, $0/scan. The
  point of these guards is *first-line defence*; richer LLM-based content
  moderation would belong on a separate dimension.
- **Cyrillic patterns via \\uXXXX escapes** inside raw-string regexes so the
  source file stays pure-ASCII (some Windows toolchains still read source as
  cp1252). The ``re`` engine interprets ``\\u0437`` as the Cyrillic codepoint at
  compile time.
- **InputGuard scoring 1-hit-blocks** (single confirmed injection wins),
  **DocumentGuard scoring weighted** (subtle hints alone don't block; only
  combinations do -- avoids false-positives on textbook content that
  legitimately *quotes* phrases like "ignoriere alle ...").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


def _ru(*codes: int) -> str:
    """Build a Cyrillic substring from codepoints so the source stays pure
    ASCII -- same pattern as ``query_rewriter._cyr``. The ``re`` engine sees
    real Cyrillic characters at compile time, but the tooling never has to."""
    return "".join(chr(c) for c in codes)


# Russian word constants used in InputGuard patterns. Translit in comment.
_RU_FORGET      = _ru(0x0437, 0x0430, 0x0431, 0x0443, 0x0434, 0x044c)               # "zabud'"
_RU_IGNORE      = _ru(0x0438, 0x0433, 0x043d, 0x043e, 0x0440, 0x0438, 0x0440, 0x0443, 0x0439)  # "ignoriruj"
_RU_TY          = _ru(0x0442, 0x044b)                                                # "ty"
_RU_TEPER       = _ru(0x0442, 0x0435, 0x043f, 0x0435, 0x0440, 0x044c)               # "teper'"
_RU_NEW         = _ru(0x043d, 0x043e, 0x0432, 0x0430, 0x044f)                        # "novaya"
_RU_INSTRUCTION = _ru(0x0438, 0x043d, 0x0441, 0x0442, 0x0440, 0x0443, 0x043a, 0x0446, 0x0438, 0x044f)  # "instrukciya"
_RU_SHOW        = _ru(0x043f, 0x043e, 0x043a, 0x0430, 0x0436, 0x0438)               # "pokazhi"
_RU_SYS_ROOT    = _ru(0x0441, 0x0438, 0x0441, 0x0442, 0x0435, 0x043c, 0x043d)       # "sistemn" (root)
_RU_PROMPT_WORD = _ru(0x043f, 0x0440, 0x043e, 0x043c, 0x043f, 0x0442)               # "promt"
_RU_VSE         = _ru(0x0432, 0x0441, 0x0435)                                        # "vse"
_RU_VSYE        = _ru(0x0432, 0x0441, 0x0451)                                        # "vsye" (with ye)
_RU_PREVIOUS    = _ru(0x043f, 0x0440, 0x0435, 0x0434, 0x044b, 0x0434, 0x0443, 0x0449, 0x0438, 0x0435)  # "predyduschie"


class PromptInjectionBlocked(Exception):
    """Raised by InputGuard when the InputGuard finds a known injection.

    The FastAPI layer catches this and returns HTTP 422 with the localized
    error message (DE if the user typed German, RU if Russian, EN otherwise).
    """

    def __init__(self, pattern_name: str, query: str, score: int = 1) -> None:
        super().__init__(f"prompt injection detected: {pattern_name}")
        self.pattern_name = pattern_name
        self.query = query
        self.score = score


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class InputScanResult:
    verdict: str  # "ok" | "blocked"
    score: int = 0
    matched_patterns: list[str] = field(default_factory=list)


@dataclass
class DocumentScanResult:
    flagged: bool = False
    score: int = 0
    matched_patterns: list[dict] = field(default_factory=list)


@dataclass
class OutputScanResult:
    redacted_text: str
    redactions: list[dict] = field(default_factory=list)  # [{"kind": ..., "count": ...}]

    @property
    def is_clean(self) -> bool:
        return not self.redactions


# ---------------------------------------------------------------------------
# InputGuard -- prompt injection (pre-retrieval)
# ---------------------------------------------------------------------------

class InputGuard:
    """Single-hit blocker for known prompt-injection phrasings.

    13 English patterns cover the common prompt-injection phrasings; 5 DE +
    5 RU patterns are added on top (the bilingual users may try anything).
    """

    # Note: Cyrillic patterns use \uXXXX escapes -- the ``re`` engine
    # interprets these as Unicode codepoints at compile time, while the
    # source byte stream stays pure ASCII.
    PATTERNS: list[tuple[str, str]] = [
        # --- English (ported from a reference implementation) ----
        ("en_instruction_override",     r"ignore\s+(all\s+)?previous\s+(instructions|context|rules)"),
        ("en_instruction_override_alt", r"disregard\s+(all\s+)?(prior|previous|above)"),
        ("en_system_prompt_extract",    r"(output|reveal|show|print|display)\s+(me\s+)?(the\s+)?system\s+prompt"),
        ("en_role_hijack",              r"you\s+are\s+now\s+(?!able|going|ready)"),
        ("en_delimiter_injection",      r"---\s*\n\s*(SYSTEM|USER|ASSISTANT)\s*:"),
        ("en_template_injection",       r"\{\{.*?(system|prompt|config|admin).*?\}\}"),
        ("en_context_extraction",       r"(list|dump|show|reveal)\s+(all\s+)?(api\s+keys?|credentials?|secrets?|tokens?)"),
        ("en_context_dump",             r"(dump|output|print)\s+(all\s+)?(your\s+)?(context|documents|memory)"),
        ("en_encoding_bypass",          r"(base64|hex)\s*(encode|decode|convert)"),
        ("en_jailbreak_dan",            r"\bDAN\b.*?(mode|persona|character)"),
        ("en_jailbreak_developer",      r"developer\s+mode\s+(enabled|activated|on)"),
        ("en_forget_instruction",       r"forget\s+(that|everything|all|previous)"),
        ("en_new_instruction",          r"new\s+(instruction|directive|rule)\s*:"),
        # --- Deutsch -------------------------------------------------------
        ("de_instruction_override",     r"ignorier\w*\s+(alle\s+)?(vorherig\w*|vorhergehend\w*|bisherig\w*)\s+(anweisung\w*|instruktion\w*|befehl\w*|regel\w*)"),
        ("de_forget",                   r"vergiss\s+(alles|alle|jede)\s+(davor|bisherig\w*|vorherig\w*|vorhergehend\w*)"),
        ("de_role_hijack",              r"du\s+bist\s+(jetzt|nun|ab\s+jetzt)\s+(?!fertig|bereit|in\s+der\s+lage)"),
        ("de_new_instruction",          r"neue\s+(anweisung|regel|direktive|instruktion)\s*:"),
        ("de_system_prompt_extract",    r"(zeig|gib|nenn|verrate?)\s+(mir\s+)?(den\s+|deinen\s+)?(system|systemprompt|systemanweisung)"),
        # --- Russkij (Cyrillic chars built from codepoints; see _ru helper) ---
        # zabud' (vse|vsye|predyduschie ...) -- "forget (everything|previous ...)"
        ("ru_forget",
         _RU_FORGET + r"\s+(" + _RU_VSE + r"|" + _RU_VSYE + r"|" + _RU_PREVIOUS + r"?)"),
        # ignoriruj (vse|predyduschie ...) -- "ignore (all|previous ...)"
        ("ru_ignore",
         _RU_IGNORE + r"\s+(" + _RU_VSE + r"|" + _RU_PREVIOUS + r"?)"),
        # ty teper' -- "you are now"
        ("ru_role_hijack",
         _RU_TY + r"\s+" + _RU_TEPER + r"\s+"),
        # novaya instrukciya: -- "new instruction:"
        ("ru_new_instruction",
         _RU_NEW + r"\s+" + _RU_INSTRUCTION + r"\s*:"),
        # pokazhi sistemnyy promt -- "show the system prompt"
        ("ru_system_prompt_extract",
         _RU_SHOW + r"\s+" + _RU_SYS_ROOT + r"\w+\s+" + _RU_PROMPT_WORD),
    ]

    def __init__(self) -> None:
        self._compiled = [
            (name, re.compile(pattern, re.IGNORECASE | re.DOTALL))
            for name, pattern in self.PATTERNS
        ]

    def scan(self, query: str) -> InputScanResult:
        """Single-hit blocker. Score = number of distinct pattern matches.
        First match wins for the matched_patterns list (we don't continue
        after a hit -- cheap)."""
        if not query:
            return InputScanResult(verdict="ok", score=0)
        for name, regex in self._compiled:
            if regex.search(query):
                return InputScanResult(
                    verdict="blocked", score=1, matched_patterns=[name],
                )
        return InputScanResult(verdict="ok", score=0)


# ---------------------------------------------------------------------------
# DocumentGuard -- retrieval poisoning (post-retrieval, pre-generate)
# ---------------------------------------------------------------------------

class DocumentGuard:
    """Weighted content scanner. Score >= threshold marks a doc as flagged.

    Threshold defaults to 3 so a single low-weight hit (e.g. "deprecated")
    on legitimate technical content does NOT trip the guard. Combinations
    (high+medium, medium+medium+low) do.
    """

    PATTERNS: list[tuple[int, str, str]] = [
        # --- Weight 3: direct manipulation attempts ------------------------
        (3, "instruction_override",  r"(disregard|ignore)\s+(all\s+)?(other|previous)\s+(documents?|context|instructions?)"),
        (3, "system_update",         r"(important|critical)\s+(system\s+)?update\s*:"),
        (3, "note_to_ai",            r"\[?\s*note\s+to\s+(ai|model|assistant|llm)\s*:?\s*\]?"),
        (3, "correct_answer_is",     r"the\s+correct\s+answer\s+is\s*:"),
        (3, "role_override",         r"you\s+(must|should)\s+(always|never)\s+(say|state|claim|mention)"),
        (3, "de_system_update",      r"(wichtige?s?|kritische?s?)\s+(system\s*)?update\s*:"),
        (3, "de_correct_answer",     r"die\s+korrekte\s+antwort\s+(ist|lautet)\s*:"),
        # --- Weight 2: behavioural steering --------------------------------
        (2, "always_recommend",      r"always\s+recommend"),
        (2, "never_mention",         r"never\s+mention"),
        (2, "do_not_reveal",         r"do\s+not\s+(reveal|disclose|share|tell)"),
        (2, "disable_security",      r"disabl(e|ing)\s+(all\s+)?(authentication|security|verification|encryption)"),
        (2, "unencrypted",           r"recommend.{0,30}(unencrypted|plaintext|http\b(?!s))"),
        (2, "de_never_mention",      r"erw[ae]hn\w*\s+(niemals|nie)"),
        # --- Weight 1: misinformation markers ------------------------------
        (1, "deprecated_claim",      r"deprecated\s+and\s+should\s+not\s+be\s+used"),
        (1, "actually_incorrect",    r"actually\s+(is\s+)?incorrect"),
        (1, "unsafe_recommendation", r"(recommended|best)\s+(production\s+)?approach.{0,50}plaintext"),
    ]

    def __init__(self, threshold: int = 3) -> None:
        self.threshold = threshold
        self._compiled = [
            (weight, name, re.compile(pattern, re.IGNORECASE | re.DOTALL))
            for weight, name, pattern in self.PATTERNS
        ]

    def scan(self, text: str) -> DocumentScanResult:
        if not text:
            return DocumentScanResult()
        score = 0
        matched: list[dict] = []
        for weight, name, regex in self._compiled:
            if regex.search(text):
                score += weight
                matched.append({"name": name, "weight": weight})
        return DocumentScanResult(
            flagged=score >= self.threshold,
            score=score,
            matched_patterns=matched,
        )


# ---------------------------------------------------------------------------
# OutputValidator -- PII redaction (post-generate)
# ---------------------------------------------------------------------------

class OutputValidator:
    """Scan generated answers for PII and replace with ``[REDACTED-<KIND>]``."""

    # Pattern ORDER matters: each is substituted in sequence, so a later
    # pattern sees the text *after* earlier redactions. IBAN must run BEFORE
    # CREDIT_CARD because IBAN starts with 16+ digits in 4-digit groups (the
    # credit-card shape would otherwise eat part of the IBAN body).
    REDACTION_PATTERNS: list[tuple[str, str]] = [
        ("SSN",             r"\b\d{3}-\d{2}-\d{4}\b"),
        ("EMAIL",           r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        # German phone first (longer + more specific than the US 10-digit shape).
        # Leading "+" is non-word, so \b doesn't anchor it -- use a non-word
        # lookbehind instead.
        ("PHONE_DE",        r"(?<!\w)\+49\s?\(?0?\)?\s?\d{2,5}[\s.-]?\d{4,12}"),
        # IBAN: 2-letter country + 2 check digits + 4..7 four-char groups +
        # optional trailing partial group. Country letters must be uppercase.
        ("IBAN",            r"\b[A-Z]{2}\d{2}(?:[\s]?[A-Z0-9]{4}){3,7}(?:[\s]?[A-Z0-9]{1,4})?\b"),
        # Stricter credit-card: forbid digits on either side so we don't bite
        # off the middle of an IBAN or other long number run.
        ("CREDIT_CARD",     r"(?<!\d)(?:\d{4}[-\s]?){3}\d{4}(?!\d)"),
        ("PHONE_US",        r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        ("API_KEY_SK",      r"\bsk-[A-Za-z0-9]{20,}\b"),
        ("API_KEY_AWS",     r"\bAKIA[A-Z0-9]{16}\b"),
        ("API_KEY_GENERIC", r"\b[A-Za-z0-9]{32,}\b(?=.*(?:key|token|secret))"),
    ]

    def __init__(self) -> None:
        self._compiled = [
            (kind, re.compile(pattern))   # NOT IGNORECASE -- SK/AKIA are case-significant
            for kind, pattern in self.REDACTION_PATTERNS
        ]

    def scan(self, text: str) -> OutputScanResult:
        if not text:
            return OutputScanResult(redacted_text="")
        cleaned = text
        redactions: list[dict] = []
        for kind, regex in self._compiled:
            matches = regex.findall(cleaned)
            if matches:
                cleaned = regex.sub(f"[REDACTED-{kind}]", cleaned)
                redactions.append({"kind": kind, "count": len(matches)})
        return OutputScanResult(redacted_text=cleaned, redactions=redactions)
