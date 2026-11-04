"""Lexical retrieval over WHO labels.

Every title, index term, inclusion and translation is a separate label pointing at
one code. A code scores as its best-matching label. Labels are short, so a label is
scored by how much of its information the query covers, weighted by inverse document
frequency: a fully mentioned specific label ("acute amoebic dysentery") beats a
partially mentioned one, and longer fully mentioned labels beat shorter ones.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import sparse

from icd_coder.catalogue import Catalogue
from icd_coder.text import tokens

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Label:
    code: str
    text: str


@dataclass(frozen=True, slots=True)
class ScoredCode:
    code: str
    score: float


def catalogue_labels(catalogue: Catalogue, languages: Iterable[str] = ()) -> list[Label]:
    """Titles, index terms and inclusions, plus the given languages' translations."""
    wanted = set(languages)
    labels: list[Label] = []
    for category in catalogue:
        texts = {category.title, *category.index_terms, *category.inclusions}
        for language, translation in category.translations.items():
            if language in wanted:
                texts.update((translation.title, *translation.terms))
        labels.extend(Label(category.code, text) for text in sorted(texts))
    return labels


class LabelIndex:
    def __init__(self, labels: Sequence[Label], coverage_power: float = 1.0) -> None:
        self._coverage_power = coverage_power
        self._codes = sorted({label.code for label in labels})
        code_position = {code: position for position, code in enumerate(self._codes)}
        self._label_code = np.array([code_position[label.code] for label in labels], dtype=np.int64)

        tokenised = [set(tokens(label.text)) for label in labels]
        vocabulary = sorted(set().union(*tokenised))
        self._term_position = {term: position for position, term in enumerate(vocabulary)}
        document_frequency = np.zeros(len(vocabulary))
        rows, columns = [], []
        for row, terms in enumerate(tokenised):
            for term in terms:
                column = self._term_position[term]
                rows.append(row)
                columns.append(column)
                document_frequency[column] += 1
        count = len(labels)
        self._idf = np.log(1.0 + (count - document_frequency + 0.5) / (document_frequency + 0.5))
        weights = self._idf[columns]
        self._matrix = sparse.csr_matrix(
            (weights, (rows, columns)), shape=(count, len(vocabulary)), dtype=np.float64
        )
        self._label_mass: FloatArray = np.asarray(self._matrix.sum(axis=1)).ravel()

    def search(self, query: Mapping[str, float], limit: int = 50) -> list[ScoredCode]:
        """Best codes for a query given as term weights in [0, 1]."""
        weights = np.zeros(len(self._term_position))
        for term, weight in query.items():
            position = self._term_position.get(term)
            if position is not None:
                weights[position] = max(weights[position], weight)
        if not weights.any():
            return []
        covered: FloatArray = self._matrix @ weights
        with np.errstate(divide="ignore", invalid="ignore"):
            coverage = np.where(self._label_mass > 0, covered / self._label_mass, 0.0)
        label_scores = covered * coverage**self._coverage_power
        code_scores = np.zeros(len(self._codes))
        np.maximum.at(code_scores, self._label_code, label_scores)
        best = np.argsort(-code_scores)[:limit]
        return [
            ScoredCode(self._codes[position], float(code_scores[position]))
            for position in best
            if code_scores[position] > 0
        ]


def query_from_text(text: str, weight: float = 1.0) -> dict[str, float]:
    return dict.fromkeys(tokens(text), weight)


def merge_queries(*queries: Mapping[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for query in queries:
        for term, weight in query.items():
            merged[term] = max(merged.get(term, 0.0), weight)
    return merged
