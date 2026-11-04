"""The coder that needs no network: weighted lexical retrieval plus specificity rules."""

from __future__ import annotations

from icd_coder.catalogue import Catalogue
from icd_coder.clinical_text import weighted_clauses
from icd_coder.lexical import LabelIndex, ScoredCode, merge_queries, query_from_text

MAX_CODES = 3


class OfflineCoder:
    def __init__(self, catalogue: Catalogue, index: LabelIndex) -> None:
        self._catalogue = catalogue
        self._index = index

    def rank(self, text: str, limit: int = 50) -> list[ScoredCode]:
        query = merge_queries(
            *(query_from_text(clause.text, clause.weight) for clause in weighted_clauses(text))
        )
        return self._index.search(query, limit=limit)

    def predict(self, text: str) -> tuple[str, ...]:
        return specific_codes(self.rank(text), self._catalogue)


def specific_codes(ranked: list[ScoredCode], catalogue: Catalogue) -> tuple[str, ...]:
    """Top distinct codes, each resolved to the most specific code the ranking supports.

    A non-leaf code is replaced by its best-ranked descendant when one was retrieved,
    otherwise by its default child chain.
    """
    position = {scored.code: rank for rank, scored in enumerate(ranked)}
    codes: list[str] = []
    for scored in ranked:
        code = scored.code
        if not catalogue[code].is_leaf:
            retrieved = [
                child.code for child in catalogue.descendants(code) if child.code in position
            ]
            code = (
                min(retrieved, key=position.__getitem__)
                if retrieved
                else catalogue.most_specific_default(code)
            )
        if code not in codes:
            codes.append(code)
        if len(codes) == MAX_CODES:
            break
    return tuple(codes)
