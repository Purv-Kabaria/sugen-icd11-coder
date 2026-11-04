"""Candidate codes for a note and the features a learned ranker scores them by.

Training and inference call the same `FeatureExtractor`, so the two can never drift.

Candidates are codes that can be submitted: leaves, or parents that have no default
child. They come from three lexical views of the note (the whole weighted note, only
its diagnostic clauses, and each diagnostic clause alone) and are expanded through
the hierarchy: a retrieved parent contributes its default child and its leaves, and a
retrieved code contributes its siblings, so the right level of specificity is always
among the candidates even when the note never names it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from icd_coder.catalogue import Catalogue
from icd_coder.clinical_text import DIAGNOSTIC_WEIGHT, WeightedClause, weighted_clauses
from icd_coder.language import detect
from icd_coder.lexical import LabelIndex, QueryScores, merge_queries, query_from_text

FEATURES = (
    "full_score",
    "full_coverage",
    "full_covered",
    "full_rank",
    "full_score_ratio",
    "diag_score",
    "diag_coverage",
    "diag_rank",
    "diag_score_ratio",
    "clause_score",
    "clause_coverage",
    "parent_full_score",
    "parent_diag_score",
    "family_full_max",
    "family_share",
    "children_diag_max",
    "retrieved",
    "via_default",
    "via_child",
    "via_sibling",
    "is_leaf",
    "is_unspecified",
    "is_other",
    "is_default_child",
    "depth",
    "siblings",
    "chapter",
    "note_english",
    "diagnostic_clauses",
    "note_length",
)
CATEGORICAL_FEATURES = ("chapter",)

RETRIEVED_PER_VIEW = 40
RETRIEVED_PER_CLAUSE = 10
MAX_CLAUSES = 6
MAX_CANDIDATES = 160
MAX_SIBLINGS = 15
MAX_DESCENDANTS = 25
UNRANKED = RETRIEVED_PER_VIEW + 1

FloatMatrix = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class Candidates:
    codes: tuple[str, ...]
    features: FloatMatrix


@dataclass(frozen=True, slots=True)
class _Views:
    full: QueryScores
    diagnostic: QueryScores
    clauses: tuple[QueryScores, ...]
    full_rank: dict[str, int]
    diagnostic_rank: dict[str, int]
    retrieved: tuple[str, ...]


class FeatureExtractor:
    def __init__(self, catalogue: Catalogue, index: LabelIndex) -> None:
        self._catalogue = catalogue
        self._index = index

    def extract(self, text: str) -> Candidates:
        language = detect(text)
        languages = {"en", language}
        clauses = weighted_clauses(text)
        diagnostic = [clause for clause in clauses if clause.weight == DIAGNOSTIC_WEIGHT]
        views = self._views(clauses, diagnostic, languages)
        origins = self._expand(views.retrieved)
        note = (
            float(language == "en"),
            float(len(diagnostic)),
            math.log1p(len(text)),
        )
        family_max: dict[str, float] = {}
        rows = [
            self._row(code, origin, views, note, family_max) for code, origin in origins.items()
        ]
        matrix = np.array(rows, dtype=np.float32).reshape(len(rows), len(FEATURES))
        _add_ratios(matrix)
        return Candidates(tuple(origins), matrix)

    def _views(
        self,
        clauses: Sequence[WeightedClause],
        diagnostic: Sequence[WeightedClause],
        languages: set[str],
    ) -> _Views:
        full_query = merge_queries(
            *(query_from_text(clause.text, clause.weight) for clause in clauses)
        )
        diagnostic_query = merge_queries(*(query_from_text(clause.text) for clause in diagnostic))
        full = self._index.score(full_query, languages)
        diagnostic_scores = self._index.score(diagnostic_query, languages)
        clause_scores = tuple(
            self._index.score(query_from_text(clause.text), languages)
            for clause in diagnostic[:MAX_CLAUSES]
        )
        full_top = full.top(RETRIEVED_PER_VIEW)
        diagnostic_top = diagnostic_scores.top(RETRIEVED_PER_VIEW)
        retrieved = [scored.code for scored in diagnostic_top] + [
            scored.code for scored in full_top
        ]
        for scores in clause_scores:
            retrieved.extend(scored.code for scored in scores.top(RETRIEVED_PER_CLAUSE))
        return _Views(
            full=full,
            diagnostic=diagnostic_scores,
            clauses=clause_scores,
            full_rank={scored.code: rank for rank, scored in enumerate(full_top, 1)},
            diagnostic_rank={scored.code: rank for rank, scored in enumerate(diagnostic_top, 1)},
            retrieved=tuple(dict.fromkeys(retrieved)),
        )

    def _expand(self, retrieved: Sequence[str]) -> dict[str, str]:
        """Submittable codes with how each entered: retrieved, default, child or sibling."""
        catalogue = self._catalogue
        origins: dict[str, str] = {}

        def add(code: str, origin: str) -> None:
            if code not in origins and len(origins) < MAX_CANDIDATES:
                origins[code] = origin

        for code in retrieved:
            category = catalogue[code]
            resolved = catalogue.most_specific_default(code)
            add(resolved, "retrieved" if resolved == code else "default")
            if not category.is_leaf:
                leaves = [item for item in catalogue.descendants(code) if item.is_leaf]
                for leaf in leaves[:MAX_DESCENDANTS]:
                    add(leaf.code, "child")
            if category.parent is not None:
                siblings = catalogue.children(category.parent)
                if len(siblings) <= MAX_SIBLINGS:
                    for sibling in siblings:
                        add(catalogue.most_specific_default(sibling.code), "sibling")
        return origins

    def _row(
        self,
        code: str,
        origin: str,
        views: _Views,
        note: tuple[float, ...],
        family_max: dict[str, float],
    ) -> list[float]:
        catalogue = self._catalogue
        category = catalogue[code]
        full = views.full.evidence(code)
        diagnostic = views.diagnostic.evidence(code)
        clause_best = max(
            (scores.evidence(code) for scores in views.clauses),
            key=lambda evidence: evidence.score,
            default=None,
        )
        parent = category.parent
        parent_full = views.full.code_score(parent) if parent else 0.0
        parent_diagnostic = views.diagnostic.code_score(parent) if parent else 0.0
        stem = catalogue.stem(code)
        if stem not in family_max:
            family_max[stem] = max(
                views.full.code_score(member.code) for member in catalogue.family(stem)
            )
        best_in_family = family_max[stem]
        children_max = max(
            (views.diagnostic.code_score(child.code) for child in catalogue.children(code)),
            default=0.0,
        )
        siblings = len(catalogue.children(parent)) if parent else 0
        default_child = catalogue.default_child(parent) if parent else None
        return [
            full.score,
            full.coverage,
            full.covered,
            math.log(views.full_rank.get(code, UNRANKED)),
            0.0,
            diagnostic.score,
            diagnostic.coverage,
            math.log(views.diagnostic_rank.get(code, UNRANKED)),
            0.0,
            clause_best.score if clause_best else 0.0,
            clause_best.coverage if clause_best else 0.0,
            parent_full,
            parent_diagnostic,
            best_in_family,
            full.score / best_in_family if best_in_family > 0 else 0.0,
            children_max,
            float(origin == "retrieved"),
            float(origin == "default"),
            float(origin == "child"),
            float(origin == "sibling"),
            float(category.is_leaf),
            float(category.is_unspecified),
            float(category.is_other_specified),
            float(default_child is not None and default_child.code == code),
            float(category.depth),
            float(siblings),
            float(int(category.chapter)),
            *note,
        ]


def _add_ratios(matrix: FloatMatrix) -> None:
    """Scores relative to the note's best candidate, so notes of any length compare."""
    if matrix.shape[0] == 0:
        return
    for score, ratio in (("full_score", "full_score_ratio"), ("diag_score", "diag_score_ratio")):
        column = matrix[:, FEATURES.index(score)]
        best = float(column.max())
        matrix[:, FEATURES.index(ratio)] = column / best if best > 0 else 0.0
