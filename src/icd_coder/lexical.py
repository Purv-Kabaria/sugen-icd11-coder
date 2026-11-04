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
        self._code_starts = code_starts
        self._label_languages = label_languages
        self._language_masks = {
            language: label_languages == language for language in np.unique(label_languages)
        }
        self._vocabulary = list(vocabulary)
        self._term_position = {term: position for position, term in enumerate(vocabulary)}
        self._matrix = matrix
        self._label_mass: FloatArray = np.asarray(matrix.sum(axis=1), dtype=np.float32).ravel()
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
        weights = np.zeros(len(self._vocabulary), dtype=np.float32)
        for term, weight in query.items():
            position = self._term_position.get(term)
            if position is not None:
                weights[position] = max(weights[position], weight)
        if not weights.any():
            return []
        covered = self._matrix @ weights
        coverage = np.divide(
            covered, self._label_mass, out=np.zeros_like(covered), where=self._label_mass > 0
        )
        label_scores = covered * coverage**self._coverage_power
        if languages is not None:
            label_scores *= self._mask(languages)
        code_scores = np.maximum.reduceat(label_scores, self._code_starts)
        top = min(limit, int(np.count_nonzero(code_scores)))
        if top == 0:
            return []
        best = np.argpartition(-code_scores, top - 1)[:top]
        best = best[np.argsort(-code_scores[best], kind="stable")]
        return [ScoredCode(self._codes[index], float(code_scores[index])) for index in best]


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
