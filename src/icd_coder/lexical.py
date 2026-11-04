"""Lexical retrieval over WHO labels.

Every title, index term, inclusion and translation is a separate label pointing at
one code. A code scores as its best-matching label. Labels are short, so a label is
scored by how much of its information the query covers, weighted by inverse document
frequency: a fully mentioned specific label ("acute amoebic dysentery") beats a
partially mentioned one, and longer fully mentioned labels beat shorter ones.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy import sparse

from icd_coder.catalogue import Catalogue
from icd_coder.text import tokens

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]
TOKENISE_CHUNK = 20_000
PARALLEL_THRESHOLD = 50_000


@dataclass(frozen=True, slots=True)
class Label:
    code: str
    text: str
    language: str = "en"


@dataclass(frozen=True, slots=True)
class ScoredCode:
    code: str
    score: float


def catalogue_labels(catalogue: Catalogue, languages: Iterable[str] = ()) -> list[Label]:
    """Titles, index terms and inclusions plus the given translations, grouped by code."""
    wanted = set(languages)
    labels: list[Label] = []
    for category in sorted(catalogue, key=lambda item: item.code):
        english = {category.title, *category.index_terms, *category.inclusions}
        labels.extend(Label(category.code, text) for text in sorted(english))
        for language, translation in sorted(category.translations.items()):
            if language in wanted:
                texts = {translation.title, *translation.terms}
                labels.extend(Label(category.code, text, language) for text in sorted(texts))
    return labels


@dataclass(frozen=True, slots=True)
class Evidence:
    """How well a code's best label matches a query."""

    score: float
    coverage: float
    covered: float


class QueryScores:
    """One query scored against the labels it touches, kept so codes can be inspected cheaply.

    Only labels sharing a term with the query can score, so they are the only ones
    stored: `labels` holds their row numbers in ascending order, and the other arrays
    are aligned with it.
    """

    def __init__(
        self,
        index: LabelIndex,
        *,
        labels: IntArray,
        covered: FloatArray,
        coverage: FloatArray,
        label_scores: FloatArray,
        code_scores: FloatArray,
    ) -> None:
        self._index = index
        self._labels = labels
        self._covered = covered
        self._coverage = coverage
        self._label_scores = label_scores
        self._code_scores = code_scores

    def top(self, limit: int) -> list[ScoredCode]:
        count = min(limit, int(np.count_nonzero(self._code_scores)))
        if count == 0:
            return []
        best = np.argpartition(-self._code_scores, count - 1)[:count]
        best = best[np.argsort(-self._code_scores[best], kind="stable")]
        codes = self._index.codes
        return [
            ScoredCode(codes[position], float(self._code_scores[position])) for position in best
        ]

    def code_score(self, code: str) -> float:
        return float(self._code_scores[self._index.position(code)])

    def evidence(self, code: str) -> Evidence:
        start, end = self._index.code_range(code)
        low, high = np.searchsorted(self._labels, (start, end))
        if low == high:
            return Evidence(0.0, 0.0, 0.0)
        best = int(low) + int(np.argmax(self._label_scores[low:high]))
        return Evidence(
            score=float(self._label_scores[best]),
            coverage=float(self._coverage[best]),
            covered=float(self._covered[best]),
        )


class LabelIndex:
    def __init__(
        self,
        codes: Sequence[str],
        code_starts: IntArray,
        vocabulary: Sequence[str],
        matrix: sparse.csr_matrix,
        *,
        label_languages: npt.NDArray[np.str_],
        coverage_power: float = 1.0,
    ) -> None:
        """Use `build` or `load`; labels must be grouped by code, `code_starts` marks each group."""
        self._codes = list(codes)
        self._code_position = {code: position for position, code in enumerate(self._codes)}
        self._code_starts = code_starts
        self._label_languages = label_languages
        self._language_masks = {
            language: label_languages == language for language in np.unique(label_languages)
        }
        self._vocabulary = list(vocabulary)
        self._term_position = {term: position for position, term in enumerate(vocabulary)}
        self._matrix = matrix
        self._columns = matrix.tocsc()
        self._label_mass: FloatArray = np.asarray(matrix.sum(axis=1), dtype=np.float32).ravel()
        self._label_code: IntArray = np.repeat(
            np.arange(len(code_starts), dtype=np.int64),
            np.diff(np.append(code_starts, matrix.shape[0])),
        )
        self._coverage_power = coverage_power

    @classmethod
    def build(
        cls, labels: Sequence[Label], coverage_power: float = 1.0, workers: int | None = None
    ) -> LabelIndex:
        ordered = sorted(labels, key=lambda label: label.code)
        token_sets = _tokenise_all([label.text for label in ordered], workers)
        vocabulary = sorted(set().union(*token_sets))
        position = {term: index for index, term in enumerate(vocabulary)}

        rows = np.repeat(
            np.arange(len(ordered), dtype=np.int64), [len(terms) for terms in token_sets]
        )
        columns = np.fromiter(
            (position[term] for terms in token_sets for term in terms), dtype=np.int64
        )
        document_frequency = np.bincount(columns, minlength=len(vocabulary)).astype(np.float64)
        count = len(ordered)
        idf = np.log1p((count - document_frequency + 0.5) / (document_frequency + 0.5))
        matrix = sparse.csr_matrix(
            (idf[columns].astype(np.float32), (rows, columns)),
            shape=(count, len(vocabulary)),
            dtype=np.float32,
        )
        matrix.indices = matrix.indices.astype(np.int32)
        matrix.indptr = matrix.indptr.astype(np.int32)

        codes, code_starts = _code_groups([label.code for label in ordered])
        languages = np.array([label.language for label in ordered])
        return cls(
            codes,
            code_starts,
            vocabulary,
            matrix,
            label_languages=languages,
            coverage_power=coverage_power,
        )

    def _mask(self, languages: Collection[str]) -> npt.NDArray[np.bool_]:
        mask = np.zeros(len(self._label_languages), dtype=bool)
        for language in languages:
            known = self._language_masks.get(language)
            if known is not None:
                mask |= known
        return mask

    def save(self, path: Path) -> None:
        """Write the index atomically, so a reader never sees a half-written file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".partial.npz")
        np.savez(
            partial,
            data=self._matrix.data,
            indices=self._matrix.indices,
            indptr=self._matrix.indptr,
            shape=np.array(self._matrix.shape),
            code_starts=self._code_starts,
            codes=np.array(self._codes),
            vocabulary=np.array(self._vocabulary),
            label_languages=self._label_languages,
        )
        partial.replace(path)

    @classmethod
    def load(cls, path: Path, coverage_power: float = 1.0) -> LabelIndex:
        with np.load(path, allow_pickle=False) as stored:
            matrix = sparse.csr_matrix(
                (stored["data"], stored["indices"], stored["indptr"]),
                shape=tuple(stored["shape"]),
            )
            return cls(
                stored["codes"].tolist(),
                stored["code_starts"],
                stored["vocabulary"].tolist(),
                matrix,
                label_languages=stored["label_languages"],
                coverage_power=coverage_power,
            )

    def search(
        self,
        query: Mapping[str, float],
        limit: int = 50,
        languages: Collection[str] | None = None,
    ) -> list[ScoredCode]:
        """Best codes for a query given as term weights in [0, 1].

        With `languages`, only labels in those languages count, so a French note is
        matched against English and French labels and not against Arabic ones.
        """
        return self.score(query, languages).top(limit)

    def score(
        self, query: Mapping[str, float], languages: Collection[str] | None = None
    ) -> QueryScores:
        """Scores of the labels and codes a query touches, for ranking and for evidence.

        Work is proportional to the postings of the query's terms, not to the number of
        labels, because only labels that share a term with the query can score.
        """
        terms = {
            self._term_position[term]: weight
            for term, weight in query.items()
            if term in self._term_position and weight > 0
        }
        code_scores = np.zeros(len(self._codes), dtype=np.float32)
        if not terms:
            empty_rows = np.zeros(0, dtype=np.int64)
            empty = np.zeros(0, dtype=np.float32)
            return QueryScores(
                self,
                labels=empty_rows,
                covered=empty,
                coverage=empty,
                label_scores=empty,
                code_scores=code_scores,
            )

        block = self._columns[:, list(terms)]
        contributions = block.data * np.repeat(
            np.fromiter(terms.values(), dtype=np.float32), np.diff(block.indptr)
        )
        labels, positions = np.unique(block.indices, return_inverse=True)
        covered = np.bincount(positions, weights=contributions).astype(np.float32)
        coverage: FloatArray = (covered / np.maximum(self._label_mass[labels], 1e-9)).astype(
            np.float32
        )
        label_scores: FloatArray = (covered * coverage**self._coverage_power).astype(np.float32)
        if languages is not None:
            label_scores *= self._mask(languages)[labels]
        np.maximum.at(code_scores, self._label_code[labels], label_scores)
        return QueryScores(
            self,
            labels=labels.astype(np.int64),
            covered=covered,
            coverage=coverage,
            label_scores=label_scores,
            code_scores=code_scores,
        )

    def position(self, code: str) -> int:
        return self._code_position[code]

    def code_range(self, code: str) -> tuple[int, int]:
        """The label rows of a code."""
        position = self._code_position[code]
        end = (
            int(self._code_starts[position + 1])
            if position + 1 < len(self._code_starts)
            else self._matrix.shape[0]
        )
        return int(self._code_starts[position]), end

    @property
    def codes(self) -> Sequence[str]:
        return self._codes

    @property
    def code_starts(self) -> IntArray:
        return self._code_starts


def query_from_text(text: str, weight: float = 1.0) -> dict[str, float]:
    return dict.fromkeys(tokens(text), weight)


def merge_queries(*queries: Mapping[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for query in queries:
        for term, weight in query.items():
            merged[term] = max(merged.get(term, 0.0), weight)
    return merged


def _tokenise_chunk(texts: Sequence[str]) -> list[set[str]]:
    return [set(tokens(text)) for text in texts]


def _tokenise_all(texts: Sequence[str], workers: int | None) -> list[set[str]]:
    if len(texts) < PARALLEL_THRESHOLD or workers == 1:
        return _tokenise_chunk(texts)
    chunks = [
        texts[start : start + TOKENISE_CHUNK] for start in range(0, len(texts), TOKENISE_CHUNK)
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return [token_set for chunk in pool.map(_tokenise_chunk, chunks) for token_set in chunk]


def _code_groups(label_codes: Sequence[str]) -> tuple[list[str], IntArray]:
    codes: list[str] = []
    starts: list[int] = []
    for position, code in enumerate(label_codes):
        if not codes or codes[-1] != code:
            codes.append(code)
            starts.append(position)
    return codes, np.array(starts, dtype=np.int64)
