"""The coder that needs no network: lexical retrieval ranked by a learned model or by rules."""

from __future__ import annotations

from collections.abc import Sequence

from icd_coder.catalogue import Catalogue
from icd_coder.clinical_text import weighted_clauses
from icd_coder.language import detect
from icd_coder.lexical import LabelIndex, ScoredCode, merge_queries, query_from_text
from icd_coder.reranker import Reranker

MAX_CODES = 3


class OfflineCoder:
    def __init__(
        self, catalogue: Catalogue, index: LabelIndex, reranker: Reranker | None = None
    ) -> None:
        self._catalogue = catalogue
        self._index = index
        self._reranker = reranker

    def rank(self, text: str, limit: int = 50) -> list[ScoredCode]:
        query = merge_queries(
            *(query_from_text(clause.text, clause.weight) for clause in weighted_clauses(text))
        )
        return self._index.search(query, limit=limit, languages={"en", detect(text)})

    def predict(self, text: str) -> tuple[str, ...]:
        by_rules = specific_codes(self.rank(text), self._catalogue)
        if self._reranker is None:
            return by_rules
        learned = [scored.code for scored in self._reranker.rank(text)[:MAX_CODES]]
        return tuple(dict.fromkeys([*learned, *by_rules]))[:MAX_CODES]


def specific_codes(ranked: Sequence[ScoredCode], catalogue: Catalogue) -> tuple[str, ...]:
    """Top distinct codes, each resolved to the most specific code the note supports."""
    score = {scored.code: scored.score for scored in ranked}
    codes: list[str] = []
    for scored in ranked:
        code = scored.code
        if not catalogue[code].is_leaf:
            supported = [
                child.code
                for child in catalogue.descendants(code)
                if score.get(child.code, 0.0) > scored.score
            ]
            code = (
                max(supported, key=score.__getitem__)
                if supported
                else catalogue.most_specific_default(code)
            )
        if code not in codes:
            codes.append(code)
        if len(codes) == MAX_CODES:
            break
    return tuple(codes)
