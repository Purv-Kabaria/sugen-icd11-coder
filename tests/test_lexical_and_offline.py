from __future__ import annotations

from icd_coder.catalogue import Catalogue
from icd_coder.lexical import Label, LabelIndex, ScoredCode, catalogue_labels, query_from_text
from icd_coder.offline import OfflineCoder, specific_codes


def test_fully_mentioned_specific_label_beats_partial_match() -> None:
    index = LabelIndex(
        [
            Label("A1", "acute bronchitis"),
            Label("A2", "chronic bronchitis"),
            Label("A3", "acute"),
        ]
    )
    ranked = index.search(query_from_text("Imp: acute bronchitis, likely post-viral"))
    assert ranked[0].code == "A1"


def test_code_scores_as_its_best_label() -> None:
    index = LabelIndex([Label("B1", "tungiasis"), Label("B1", "jiggers"), Label("B2", "fleas")])
    ranked = index.search(query_from_text("Dx jiggers"))
    assert [scored.code for scored in ranked] == ["B1"]


def test_empty_query_returns_nothing() -> None:
    index = LabelIndex([Label("C1", "asthma")])
    assert index.search({}) == []
    assert index.search(query_from_text("the and of")) == []


def test_specific_codes_prefers_retrieved_descendants(catalogue: Catalogue) -> None:
    ranked = [ScoredCode("CA23", 5.0), ScoredCode("CA23.30", 4.0), ScoredCode("1F40", 3.0)]
    assert specific_codes(ranked, catalogue) == ("CA23.30", "1F40.Z")


def test_specific_codes_falls_back_to_default_child(catalogue: Catalogue) -> None:
    ranked = [ScoredCode("1D80", 2.0), ScoredCode("5A11", 1.0)]
    assert specific_codes(ranked, catalogue) == ("1D80.0", "5A11")


def test_offline_coder_end_to_end(catalogue: Catalogue) -> None:
    coder = OfflineCoder(catalogue, LabelIndex(catalogue_labels(catalogue)))
    note = "S: fever x3/7.\nIx: BS for MPS positive.\nA: Uncomplicated falciparum malaria.\nP: AL."
    assert coder.predict(note)[0] == "1F40.Z"
