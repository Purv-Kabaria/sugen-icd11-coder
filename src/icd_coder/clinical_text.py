"""Find the diagnostic statements in a consult note."""

from __future__ import annotations

import re
from dataclasses import dataclass

from icd_coder.abbreviations import expand

DIAGNOSTIC_WEIGHT = 1.0
CONTEXT_WEIGHT = 0.25
PLAN_WEIGHT = 0.05

SENTENCE_BREAK = re.compile(r"\n+|(?<=[.!?;])\s+(?=[A-Z\"'(])")
CLAUSE_BREAK = re.compile(r"[,;()\[\]]|\s-\s|\s\+\s|\s/\s|:\s")

CLINICIAN_PREFIX = re.compile(r"^\s*(?:doctor|dr|clinician|nurse|co|mo)\s*:\s*", re.IGNORECASE)
LAY_PREFIX = re.compile(
    r"^\s*(?:patient|pt|mother|mum|mom|father|dad|parent|caregiver|guardian|son|daughter|"
    r"husband|wife|relative)\s*:\s*",
    re.IGNORECASE,
)
DIAGNOSTIC_HEADER = re.compile(
    r"""^\W*(?:a|ax|imp|impression|dx|ddx|diagnos[ie]s|assessment|assess|final\ diagnosis|
    diagnostic|diagnostique|conclusion|diagn[oó]stico|impresi[oó]n(?:\ diagn[oó]stica)?|
    juicio\ cl[ií]nico|hip[oó]tese(?:\ diagn[oó]stica)?|diagnose|beurteilung|utambuzi)
    \b\s*[:.\-]\s*""",
    re.IGNORECASE | re.VERBOSE,
)
INLINE_DIAGNOSIS = re.compile(
    r"\b(?:dx|diagnosis|diagnosed\ with|imp|impression|assessment|diagnostic|diagn[oó]stico)\b"
    r"\s*[:\-]?\s*",
    re.IGNORECASE | re.VERBOSE,
)
SPOKEN_DIAGNOSIS = re.compile(
    r"\b(?:this\ is|this\ looks\ like|it'?s|it\ is|you\ have|you've\ got|looks\ like|"
    r"consistent\ with|we\ call\ it|what\ you\ have\ is|suggests?|the\ diagnosis\ is|"
    r"c'est|vous\ avez|usted\ tiene|voc[eê]\ tem)\b",
    re.IGNORECASE | re.VERBOSE,
)
PLAN_HEADER = re.compile(
    r"^\W*(?:p|plan|rx|mx|tx|management|treatment|advice|traitement|tratamiento|conduta)"
    r"\b\s*[:.\-]",
    re.IGNORECASE,
)
NEGATES_FOLLOWING = re.compile(
    r"\b(?:no|not|nil|without|denies|denied|absence\ of|negative\ for|free\ of|pas\ de|sans|"
    r"aucune?|sin|sem|n[aã]o|hakuna)\b",
    re.IGNORECASE | re.VERBOSE,
)
NEGATES_PRECEDING = re.compile(
    r"\b(?:excluded|ruled\ out|negative|neg|unlikely|absent|normal|n[eé]gatif|descartad[oa]|"
    r"negativo)\b",
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True, slots=True)
class WeightedClause:
    text: str
    weight: float


def sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in SENTENCE_BREAK.split(text) if sentence.strip()]


def classify(sentence: str) -> tuple[str, float]:
    """The sentence without speaker or header cues, and its diagnostic weight."""
    if LAY_PREFIX.match(sentence):
        return LAY_PREFIX.sub("", sentence), CONTEXT_WEIGHT
    sentence = CLINICIAN_PREFIX.sub("", sentence)
    if PLAN_HEADER.match(sentence):
        return PLAN_HEADER.sub("", sentence), PLAN_WEIGHT
    header = DIAGNOSTIC_HEADER.match(sentence)
    if header:
        return sentence[header.end() :], DIAGNOSTIC_WEIGHT
    if INLINE_DIAGNOSIS.search(sentence):
        return INLINE_DIAGNOSIS.sub(" ", sentence), DIAGNOSTIC_WEIGHT
    if SPOKEN_DIAGNOSIS.search(sentence):
        return sentence, DIAGNOSTIC_WEIGHT
    return sentence, CONTEXT_WEIGHT


def affirmed_part(clause: str) -> str:
    """The part of a clause that no negation cue covers."""
    following = NEGATES_FOLLOWING.search(clause)
    if following:
        clause = clause[: following.start()]
    preceding = list(NEGATES_PRECEDING.finditer(clause))
    if preceding:
        clause = clause[preceding[-1].end() :]
    return clause.strip()


def weighted_clauses(text: str) -> list[WeightedClause]:
    """Affirmed clauses of the note with their diagnostic weight."""
    clauses: list[WeightedClause] = []
    for sentence in sentences(text):
        content, weight = classify(sentence)
        for clause in CLAUSE_BREAK.split(expand(content)):
            affirmed = affirmed_part(clause)
            if affirmed:
                clauses.append(WeightedClause(affirmed, weight))
    return clauses
