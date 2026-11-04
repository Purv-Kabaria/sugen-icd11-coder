"""The ICD-11 MMS categories the scorer accepts, with WHO knowledge attached."""

from __future__ import annotations

import gzip
import json
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

POSTCOORDINATION_SEPARATOR = re.compile(r"[&/]")
KB_PREFIX = "mms-2026-01"
UNSPECIFIED_TITLE = re.compile(r"^unspecified\b|\bunspecified$", re.IGNORECASE)
UNCOMPLICATED_TITLE = re.compile(
    r"\bwithout (?:mention of )?complications?\b|\buncomplicated\b", re.IGNORECASE
)


@dataclass(frozen=True, slots=True)
class Exclusion:
    label: str
    code: str | None


@dataclass(frozen=True, slots=True)
class Translation:
    title: str
    terms: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Category:
    code: str
    title: str
    chapter: str
    is_leaf: bool
    is_residual: bool
    depth: int
    parent: str | None
    ancestors: tuple[str, ...]
    definition: str | None
    coding_note: str | None
    index_terms: tuple[str, ...]
    inclusions: tuple[str, ...]
    exclusions: tuple[Exclusion, ...]
    translations: Mapping[str, Translation]

    @property
    def is_unspecified(self) -> bool:
        return self.is_residual and self.code.endswith("Z")

    @property
    def is_other_specified(self) -> bool:
        return self.is_residual and self.code.endswith("Y")


def normalise_code(raw: object) -> str | None:
    """Uppercase stem of a code as the scorer reads it, or None for non-strings."""
    if not isinstance(raw, str):
        return None
    stem = POSTCOORDINATION_SEPARATOR.split(raw.strip())[0].strip().upper()
    return stem or None


class Catalogue:
    def __init__(self, categories: Iterable[Category]) -> None:
        self._by_code = {category.code: category for category in categories}
        children: dict[str, list[str]] = {}
        for category in self._by_code.values():
            if category.parent is not None:
                children.setdefault(category.parent, []).append(category.code)
        self._children = {code: tuple(sorted(kids)) for code, kids in children.items()}
        self._descendants: dict[str, tuple[Category, ...]] = {}
        self._stems: dict[str, str] = {}

    @classmethod
    def load(cls, kb_dir: Path, languages: Iterable[str] = ()) -> Catalogue:
        """Read the English core and the requested translations written by `tools/build_kb.py`."""
        translations: dict[str, dict[str, Translation]] = {}
        for language in languages:
            for record in _read_jsonl_gz(kb_dir / f"{KB_PREFIX}.{language}.jsonl.gz"):
                translations.setdefault(record["code"], {})[language] = Translation(
                    record["title"], tuple(record["terms"])
                )
        return cls(
            _category_from_record(record, translations.get(record["code"], {}))
            for record in _read_jsonl_gz(kb_dir / f"{KB_PREFIX}.en.jsonl.gz")
        )

    def __contains__(self, code: object) -> bool:
        return code in self._by_code

    def __iter__(self) -> Iterator[Category]:
        return iter(self._by_code.values())

    def __len__(self) -> int:
        return len(self._by_code)

    def __getitem__(self, code: str) -> Category:
        return self._by_code[code]

    def validate(self, raw: object) -> str | None:
        """The scoreable code a raw model output refers to, or None if it names none."""
        code = normalise_code(raw)
        return code if code in self._by_code else None

    def children(self, code: str) -> tuple[Category, ...]:
        return tuple(self._by_code[child] for child in self._children.get(code, ()))

    def default_child(self, code: str) -> Category | None:
        """The child to code when the note adds nothing beyond the parent."""
        children = self.children(code)
        rules = (
            lambda child: child.is_unspecified,
            lambda child: bool(UNSPECIFIED_TITLE.search(child.title)),
            lambda child: bool(UNCOMPLICATED_TITLE.search(child.title)),
        )
        for rule in rules:
            matching = [child for child in children if rule(child)]
            if len(matching) == 1:
                return matching[0]
        return None

    def most_specific_default(self, code: str) -> str:
        """Follow default children down from a code until none applies."""
        current = code
        while (child := self.default_child(current)) is not None:
            current = child.code
        return current

    def stem(self, code: str) -> str:
        """The depth-one ancestor that heads the code's family."""
        if code not in self._stems:
            current = self._by_code[code]
            while current.parent is not None:
                current = self._by_code[current.parent]
            self._stems[code] = current.code
        return self._stems[code]

    def descendants(self, code: str) -> tuple[Category, ...]:
        """Every category below the code, in code order. Memoised: the tree never changes."""
        if code not in self._descendants:
            found: list[Category] = []
            pending = list(self.children(code))
            while pending:
                current = pending.pop()
                found.append(current)
                pending.extend(self.children(current.code))
            self._descendants[code] = tuple(sorted(found, key=lambda category: category.code))
        return self._descendants[code]

    def family(self, code: str) -> tuple[Category, ...]:
        """The family stem and every descendant, in code order."""
        root = self.stem(code)
        return (self._by_code[root], *self.descendants(root))


def _read_jsonl_gz(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def _category_from_record(
    record: Mapping[str, Any], translations: Mapping[str, Translation]
) -> Category:
    return Category(
        code=record["code"],
        title=record["title"],
        chapter=record["chapter"],
        is_leaf=record["leaf"],
        is_residual=record["residual"],
        depth=record["depth"],
        parent=record["parent"],
        ancestors=tuple(record["ancestors"]),
        definition=record["definition"],
        coding_note=record["coding_note"],
        index_terms=tuple(record["index_terms"]),
        inclusions=tuple(record["inclusions"]),
        exclusions=tuple(Exclusion(item["label"], item["code"]) for item in record["exclusions"]),
        translations=translations,
    )
