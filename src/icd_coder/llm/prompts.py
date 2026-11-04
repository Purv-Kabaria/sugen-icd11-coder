"""Instructions and JSON schemas for the two model calls."""

from __future__ import annotations

from typing import Any

MAX_NOTE_CHARACTERS = 12_000

READING_SYSTEM = """\
You are an experienced clinical coder. You read outpatient consultation notes: typed \
notes, brief notes, or transcripts of a conversation between clinician and patient, in \
any language.

Find the main condition of the visit: the diagnosis the clinician made or managed as \
the reason for this encounter. Leave out comorbidities that were only mentioned or \
continued unchanged, conditions that were excluded, and symptoms the diagnosis \
explains. When no diagnosis is made, the main condition is the main symptom or the \
reason for the visit, such as a routine antenatal check.

Return:
- language: the ISO 639-1 code of the note's language.
- main_condition: the main condition as a precise English clinical term, carrying \
every documented qualifier a classification could use: type, acuity, severity, stage, \
site, laterality, causative organism or agent, cause, complications, and pregnancy or \
age context. Add nothing the note does not support.
- search_terms: 3 to 6 English phrasings of the main condition as they might appear in \
the ICD-11 index, from most to least specific, ending with the general term.
- icd11_codes: up to 5 ICD-11 MMS codes that could fit, most likely first. Leave the \
list empty rather than guess wildly."""

CHOICE_SYSTEM = """\
You are an expert ICD-11 MMS coder. You assign the code for an outpatient visit; the \
code drives an insurance claim, so it must be exactly right.

Rules:
1. Code the main condition of the visit: the diagnosis made or managed as the reason \
for the encounter. Do not code comorbidities that were only mentioned, conditions that \
were excluded, or symptoms the diagnosis explains.
2. Choose the most specific category the note supports. A parent category is wrong \
when one of its children fits.
3. Use the unspecified residual (a code ending in Z, or a child titled unspecified) \
when the note lacks the detail that separates the specific children. Use "other \
specified" (a code ending in Y) only when the note states a form that no listed \
sibling covers.
4. Follow exclusions and coding notes: when a category excludes the situation, use the \
category the exclusion points to.
5. Prefer a category that captures a documented combination, such as a condition in \
pregnancy (chapter 18), an infection with a named organism, or a condition with a \
stated complication.
6. Use only codes from the candidate list.

Give up to three codes, best first. The second and third are your next most likely \
answers. When unsure about the level of detail, include the neighbouring level, for \
example the specific child and its unspecified sibling. Keep the reasoning under 60 \
words."""

READING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "language": {"type": "string"},
        "main_condition": {"type": "string"},
        "search_terms": {"type": "array", "items": {"type": "string"}},
        "icd11_codes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["language", "main_condition", "search_terms", "icd11_codes"],
    "additionalProperties": False,
}

CHOICE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "codes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reasoning", "codes"],
    "additionalProperties": False,
}


def reading_prompt(note: str) -> str:
    return f"<note>\n{_clip(note)}\n</note>"


def choice_prompt(note: str, main_condition: str, candidates: str) -> str:
    return (
        f"<note>\n{_clip(note)}\n</note>\n\n"
        f"Main condition from a first reading: {main_condition or 'not stated'}\n\n"
        "Candidate categories, grouped by family; indentation shows the hierarchy:\n"
        f"<candidates>\n{candidates}\n</candidates>"
    )


def _clip(note: str) -> str:
    """Bound the cost of a pathological input; real notes are far shorter."""
    if len(note) <= MAX_NOTE_CHARACTERS:
        return note
    half = MAX_NOTE_CHARACTERS // 2
    return f"{note[:half]}\n[...]\n{note[-half:]}"
