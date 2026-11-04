"""Build labelled training notes from WHO's own ICD-11 text, with no model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from icd_coder.abbreviations import ABBREVIATIONS
from icd_coder.catalogue import Catalogue, Category

KB_DIR = Path("data/kb")
OUT_DIR = Path("cache/who-notes")
HELD_OUT_BUCKETS = 5
TRANSLATED = ("fr", "es", "pt", "ru", "ar", "zh")
KIND_WEIGHTS = {"definition": 0.35, "term": 0.35, "parent": 0.2, "translated": 0.1}
FIRST_SENTENCE = re.compile(r"^(.+?[.;])(?:\s|$)")
NOT_OTHERWISE_SPECIFIED = re.compile(r",?\s*\bNOS\b")
UNSPECIFIED_SUFFIX = re.compile(r",\s*unspecified$", re.IGNORECASE)

COMPLAINTS = (
    "fever",
    "cough",
    "pain",
    "fatigue",
    "headache",
    "rash",
    "swelling",
    "dizziness",
    "poor appetite",
    "weight loss",
    "shortness of breath",
    "vomiting",
)
DURATIONS = ("2 days", "3 days", "1 week", "2 weeks", "a month", "several months")
FINDINGS = (
    "BP 128/82, pulse 88, temp 37.4",
    "vitals stable",
    "chest clear, abdomen soft",
    "alert, mildly unwell",
    "no focal neurology",
    "afebrile, well hydrated",
)
PLANS = (
    "review in two weeks",
    "bloods sent",
    "advised rest and fluids",
    "refer to specialist clinic",
    "start treatment as per local guideline",
    "safety-netting advice given",
)
DIAGNOSIS_LINE = {
    "fr": "Diagnostic : {term}. Antécédents : {history}.",
    "es": "Diagnóstico: {term}. Antecedentes: {history}.",
    "pt": "Diagnóstico: {term}. Antecedentes: {history}.",
    "ru": "Диагноз: {term}. В анамнезе: {history}.",  # noqa: RUF001
    "ar": "التشخيص: {term}. السوابق: {history}.",
    "zh": "诊断：{term}。既往史：{history}。",  # noqa: RUF001
}


@dataclass(frozen=True, slots=True)
class Draft:
    kind: str
    gold: str
    text: str


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--count", type=int, default=12_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    catalogue = Catalogue.load(KB_DIR, TRANSLATED)
    writer = NoteWriter(catalogue, random.Random(args.seed))
    drafts: list[Draft] = []
    while len(drafts) < args.count:
        draft = writer.write()
        if draft is not None:
            drafts.append(draft)
    write_split(drafts, catalogue)


class NoteWriter:
    """Writes one templated note per call, choosing its kind by `KIND_WEIGHTS`."""

    def __init__(self, catalogue: Catalogue, rng: random.Random) -> None:
        self._catalogue = catalogue
        self._rng = rng
        self._leaves = [category for category in catalogue if category.is_leaf]
        self._defined = [category for category in self._leaves if category.definition]
        self._translated = [category for category in self._leaves if category.translations]
        self._defaults = [
            (category, catalogue[resolved])
            for category in catalogue
            if not category.is_leaf
            and (resolved := catalogue.most_specific_default(category.code)) != category.code
            and catalogue[resolved].is_leaf
        ]

    def write(self) -> Draft | None:
        kind = self._rng.choices(list(KIND_WEIGHTS), list(KIND_WEIGHTS.values()))[0]
        if kind == "definition":
            return self._from_definition()
        if kind == "term":
            return self._from_term()
        if kind == "parent":
            return self._from_parent()
        return self._from_translation()

    def _from_definition(self) -> Draft | None:
        target = self._rng.choice(self._defined)
        match = FIRST_SENTENCE.match(target.definition or "")
        if match is None:
            return None
        description = f"picture in keeping with {lower_first(match.group(1).rstrip('.;'))}"
        return Draft("definition", target.code, self._english_note(target, description))

    def _from_term(self) -> Draft:
        target = self._rng.choice(self._leaves)
        term = noisy(self._rng.choice(labels_of(target)), self._rng)
        return Draft("term", target.code, self._english_note(target, term))

    def _from_parent(self) -> Draft:
        parent, target = self._rng.choice(self._defaults)
        term = noisy(self._rng.choice([parent.title, *parent.index_terms]), self._rng)
        return Draft("parent", target.code, self._english_note(parent, term))

    def _from_translation(self) -> Draft:
        target = self._rng.choice(self._translated)
        language = self._rng.choice(sorted(target.translations))
        line = DIAGNOSIS_LINE[language].format(
            term=target.translations[language].title,
            history=self._other_chapter(target, language),
        )
        return Draft("translated", target.code, line)

    def _english_note(self, subject: Category, description: str) -> str:
        """A consult around the description, in one of three styles."""
        rng = self._rng
        complaints = ", ".join(rng.sample(COMPLAINTS, rng.randint(1, 3)))
        duration = rng.choice(DURATIONS)
        history = self._other_chapter(subject)
        findings = rng.choice(FINDINGS)
        plan = rng.choice(PLANS)
        negated = self._negated_sibling(subject)
        style = rng.choice(("soap", "narrative", "transcript"))
        if style == "soap":
            ruled_out = f" No {negated}." if negated else ""
            lines = [
                f"S: {complaints} for {duration}. PMH: {history}.",
                f"O: {findings}.{ruled_out}",
                f"A: {description}.",
                f"P: {plan}.",
            ]
        elif style == "narrative":
            ruled_out = f" There is no evidence of {negated}." if negated else ""
            lines = [
                f"Seen today with {complaints} for {duration}, background of {history}. "
                f"On examination {findings}.{ruled_out} Impression: {description}. "
                f"Plan: {plan}."
            ]
        else:
            lines = [
                "Doctor: What brings you in today?",
                f"Patient: I have had {complaints} for {duration}.",
                "Doctor: Any past problems?",
                f"Patient: Only {history}.",
                f"Doctor: From what I see this is {description}. We will {plan}.",
            ]
        return "\n".join(lines)

    def _other_chapter(self, subject: Category, language: str = "en") -> str:
        """A random condition from another chapter, as background history."""
        while True:
            candidate = self._rng.choice(self._leaves)
            if candidate.chapter == subject.chapter:
                continue
            if language == "en":
                return lower_first(UNSPECIFIED_SUFFIX.sub("", candidate.title))
            if language in candidate.translations:
                return candidate.translations[language].title

    def _negated_sibling(self, subject: Category) -> str | None:
        """A sibling condition the note rules out, the hardest kind of distractor."""
        if subject.parent is None or self._rng.random() < 0.5:
            return None
        siblings = [
            sibling
            for sibling in self._catalogue.children(subject.parent)
            if sibling.code != subject.code and not sibling.is_residual
        ]
        return lower_first(self._rng.choice(siblings).title) if siblings else None


def labels_of(category: Category) -> list[str]:
    """Index terms and inclusions other than the title, or the title when there are none."""
    terms = sorted({*category.index_terms, *category.inclusions} - {category.title})
    return terms or [category.title]


def noisy(term: str, rng: random.Random) -> str:
    """The term as a clinician might write it: shortened, abbreviated, lower case."""
    text = NOT_OTHERWISE_SPECIFIED.sub("", term).strip()
    words = text.split()
    droppable = [position for position in range(1, len(words) - 1) if len(words[position]) > 3]
    if droppable and rng.random() < 0.3:
        del words[rng.choice(droppable)]
        text = " ".join(words)
    if rng.random() < 0.5:
        for abbreviation, expansion in ABBREVIATIONS.items():
            if len(abbreviation) > 3 and expansion in text.lower():
                text = re.sub(re.escape(expansion), abbreviation.upper(), text, flags=re.I)
                break
    return text.lower() if rng.random() < 0.5 else text


def write_split(drafts: Sequence[Draft], catalogue: Catalogue) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    splits: dict[str, list[str]] = {"train": [], "heldout": []}
    for number, draft in enumerate(drafts, 1):
        record = {
            "id": f"who-{number:05d}",
            "text": draft.text,
            "gold": [draft.gold],
            "slice": "W",
            "style": draft.kind,
        }
        split = family_split(catalogue.stem(draft.gold))
        splits[split].append(json.dumps(record, ensure_ascii=False))
    for name, lines in splits.items():
        (OUT_DIR / f"{name}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{name}: {len(lines)} notes")


def family_split(stem: str) -> str:
    bucket = int(hashlib.sha256(stem.encode()).hexdigest(), 16) % HELD_OUT_BUCKETS
    return "heldout" if bucket == 0 else "train"


def lower_first(text: str) -> str:
    """Lower-case a leading capital unless the first word is an acronym."""
    return text[:1].lower() + text[1:] if text[1:2].islower() else text


if __name__ == "__main__":
    main()
