"""Type-aware DE-RU prompt templates.

Three FORMAT-specific templates sharing one ``_BASE_RULES`` block, selected by
:mod:`app.services.query_router` (FACTUAL / EXPLANATORY / COMPARATIVE) and
assembled by :class:`~app.services.rag_pipeline.RAGPipeline` between a cache
miss and the generation call.

Why three templates instead of one persona prompt
-------------------------------------------------
The first version of this system used a single prompt built around a persona
("a Russian native speaker who is learning German"). That persona was
ambiguous: the model sometimes read it as "speak Russian to her" and answered
German questions in Russian. ``_BASE_RULES`` replaces it with an imperative --
detect the language of the last user question and reply exclusively in that
language -- which leaves no room for that reading.

Faithfulness ("if the context is insufficient, say so") is a rule in every
template, deliberately outranked by the translation rule: a phrase the
excerpts let us assemble is not a refusal case. Correcting the retrieval
itself happens one layer up, in the CRAG critic.
"""

from __future__ import annotations

from typing import Iterable, Literal

QueryType = Literal["FACTUAL", "EXPLANATORY", "COMPARATIVE"]


# Shared across all 3 templates so a behaviour change (citation format,
# language rule, level instruction) updates in one place.
_BASE_RULES = """\
- Detect the language of the LAST user question and reply EXCLUSIVELY in
  that language (German question -> German answer; Russian question ->
  Russian answer). Do NOT switch languages mid-answer, even if the
  textbook excerpts are in the other language.
- A phrase like "auf Russisch", "auf Schwedisch", "auf Deutsch", "po-russki"
  names the TARGET language of the translation the user wants -- it does NOT
  change the language you answer in. Example: "Was ist schwarz auf Russisch?"
  is a GERMAN question, so answer in German and simply give the Russian word
  inside that German sentence (e.g.: Auf Russisch heisst "schwarz" ...).
  Never flip the whole answer into the target language.
- Answer using ONLY the provided context excerpts. Do NOT invent facts.
- TRANSLATION REQUESTS ("Was heisst X auf <Sprache>?") are a COMPOSITION
  task, not a lookup task. If the excerpts carry the words or a close
  phrase pattern of X in the target language, BUILD the phrase from those
  pieces and give it. This rule outranks the refusal rule below: a
  buildable phrase is never a refusal. Requirements when you build one:
  (a) every target-language element you use must appear in the excerpts,
  (b) cite each piece, (c) name the pattern you followed and any deviation
  from it in one short sentence. Worked example -- excerpts hold
  "Wo ist...?" -> "Pou einai...;" and "Hafen" -> "limani": answer
  "Pou einai to limani;", cite both pieces, and note that the sources give
  the location pattern ("wo") rather than the direction pattern ("wohin").
- SCRIPT PLUS PRONUNCIATION. Whenever you give a word or phrase in a
  language that does not use the Latin alphabet (Greek, Russian, ...),
  write it FIRST in its own script, spelled exactly as the excerpts spell
  it, and immediately after it a pronunciation a German reader can say out
  loud, in round brackets -- shape: <original script> (deutsche
  Aussprache). Never give only the script, never give only the
  transcription. If the excerpts contain a transliteration but not the
  original spelling, give the transliteration and say that the excerpts do
  not show the original script -- do NOT reconstruct the spelling
  yourself.
- Refuse ONLY when the excerpts hold no usable target-language material at
  all, or when a non-translation question is genuinely uncovered. Then say
  so plainly in the user's language. Do not fabricate citations.
- Cite every claim immediately after the claim it supports: book excerpts
  as [source_file, page N], web excerpts as [source_file, web].
- Use precise grammatical terminology (Konjunktiv I/II, Modalpartikel,
  Funktionsverbgefuege, Passiversatzformen, erweitertes Partizip).
  Do NOT dumb things down to A1 -- match the user's level."""


_FORMAT_FACTUAL = """\
Format: A short definition in 1-3 sentences, followed by its citation --
one citation per excerpt you actually used (a phrase assembled from two
excerpts carries two). Be precise and terminologically exact. No examples
unless the user explicitly asks for one."""


_FORMAT_EXPLANATORY = """\
Format: 2-4 paragraphs covering (a) the rule, (b) 1-2 concrete examples
drawn from the excerpts, (c) any edge cases or exceptions visible in
the excerpts. Cite each claim."""


_FORMAT_COMPARATIVE = """\
Format: A compact table or bullet list comparing the items. The table
header row (or bullet labels) must be in the user's reply language.
Cite each row."""


_FORMATS: dict[str, str] = {
    "FACTUAL":     _FORMAT_FACTUAL,
    "EXPLANATORY": _FORMAT_EXPLANATORY,
    "COMPARATIVE": _FORMAT_COMPARATIVE,
}


_INTRO = (
    "You are a precise bilingual (German/Russian) grammar assistant for a "
    "learner working through A1-C1 textbook material."
)


# Deliberately redundant with the translation rule in _BASE_RULES. The rule
# block states the requirements; this restates the decision immediately before
# the question, where instructions carry the most weight. With the generator's
# thinking budget at 0 the rule block alone was not enough: the model listed
# the pieces it had found and then refused anyway. With this line it assembles
# the phrase on every probe run.
_TRANSLATION_DIRECTIVE = """\
Before answering, check whether this question asks what a word or phrase is
called in another language. If it does, and the excerpts contain the words or
a close phrase pattern, you MUST assemble the phrase and give it in your FIRST
sentence, then cite the pieces and note in one clause how the pattern differs
from what was asked. A pattern with the same communicative purpose counts as
close -- asking where something IS versus how to GET there is a difference you
name, not a reason to refuse. For such a question, refusing while listing the
pieces you found is a WRONG answer. Write every foreign word or phrase in its
own script followed by a German-readable pronunciation in round brackets."""


def _build_context_block(contexts: Iterable[dict]) -> str:
    """Flatten retrieved chunks into the inline context block.

    Web docs (CRAG fallback) have no page number -- labelling them
    "page None" told the model the citation was broken and nudged it toward
    refusing. They get a ``web`` locator plus their URL instead, which is
    also the citation shape the rules ask for.
    """
    blocks = []
    for c in contexts:
        if c.get("origin") == "tavily":
            url = c.get("url")
            locator = f"web: {url}" if url else "web"
        else:
            locator = f"page {c.get('page')}"
        blocks.append(f"[{c['source_file']}, {locator}]\n{c['text']}")
    return "\n\n".join(blocks)


def build_prompt(query_type: str, query: str, contexts: list[dict]) -> str:
    """Assemble the final Gemini prompt for one query.

    Unknown ``query_type`` falls back to ``EXPLANATORY`` -- the safest default
    (most flexible output shape, never worse than a single all-purpose
    template). The router already does the same fallback on parse failure,
    so this is belt-and-suspenders for callers that build prompts directly.
    """
    fmt = _FORMATS.get(query_type, _FORMAT_EXPLANATORY)
    return (
        f"{_INTRO}\n\n"
        f"Rules:\n{_BASE_RULES}\n\n"
        f"{fmt}\n\n"
        f"Context:\n{_build_context_block(contexts)}\n\n"
        f"{_TRANSLATION_DIRECTIVE}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
