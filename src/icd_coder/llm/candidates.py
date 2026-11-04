"""Pick and present the categories a model chooses between."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from icd_coder.catalogue import Catalogue, Category

RECIPROCAL_RANK_OFFSET = 10
MAX_FAMILY_LINES = 40
MAX_TERMS_PER_CODE = 4
MAX_NOTE_CHARACTERS = 240


@dataclass(frozen=True, slots=True)
class CandidateSet:
    families: tuple[str, ...]
    codes: frozenset[str]
    text: str


def fuse(rankings: Iterable[Sequence[str]]) -> list[str]:
    """Codes ordered by reciprocal rank fusion over several rankings."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, code in enumerate(dict.fromkeys(ranking)):
            scores[code] = scores.get(code, 0.0) + 1.0 / (RECIPROCAL_RANK_OFFSET + rank)
    return sorted(scores, key=lambda code: -scores[code])


def build_candidates(fused: Sequence[str], catalogue: Catalogue, max_families: int) -> CandidateSet:
    """The top families of the fused ranking, rendered as trees."""
    families: list[str] = []
    focus: dict[str, set[str]] = {}
    for code in fused:
        if code not in catalogue:
            continue
        stem = catalogue.stem(code)
        if stem not in focus:
            if len(families) == max_families:
                continue
            families.append(stem)
            focus[stem] = set()
        focus[stem].add(code)

    blocks: list[str] = []
    shown: set[str] = set()
    for stem in families:
        members = _family_members(stem, focus[stem], catalogue)
        shown.update(category.code for category in members)
        blocks.append("\n".join(_render(category, catalogue) for category in members))
    return CandidateSet(tuple(families), frozenset(shown), "\n\n".join(blocks))


def _family_members(stem: str, focus: set[str], catalogue: Catalogue) -> list[Category]:
    """The whole family when small, otherwise the retrieved codes with their context."""
    family = list(catalogue.family(stem))
    if len(family) <= MAX_FAMILY_LINES:
        return family
    keep: set[str] = {stem}
    for code in focus:
        current: str | None = code
        while current is not None:
            keep.add(current)
            parent = catalogue[current].parent
            if parent is not None:
                keep.update(child.code for child in catalogue.children(parent))
            current = parent
        keep.update(child.code for child in catalogue.children(code))
    return [category for category in family if category.code in keep]


def _render(category: Category, catalogue: Catalogue) -> str:
    indent = "  " * _depth_in_family(category, catalogue)
    parts = [f"{indent}{category.code} {category.title}"]
    terms = [term for term in category.index_terms if term.lower() != category.title.lower()]
    extras = list(dict.fromkeys([*category.inclusions, *terms]))[:MAX_TERMS_PER_CODE]
    if extras:
        parts.append(f"[also: {'; '.join(extras)}]")
    exclusions = [f"{item.label} -> {item.code}" for item in category.exclusions if item.code]
    if exclusions:
        parts.append(f"[excludes: {'; '.join(exclusions[:MAX_TERMS_PER_CODE])}]")
    if category.coding_note:
        parts.append(f"[note: {_shorten(category.coding_note)}]")
    return " ".join(parts)


def _depth_in_family(category: Category, catalogue: Catalogue) -> int:
    depth = 0
    current = category
    while current.parent is not None:
        depth += 1
        current = catalogue[current.parent]
    return depth


def _shorten(text: str) -> str:
    flat = " ".join(text.split())
    if len(flat) <= MAX_NOTE_CHARACTERS:
        return flat
    return flat[: MAX_NOTE_CHARACTERS - 3].rstrip() + "..."
