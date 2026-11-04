from __future__ import annotations

from pathlib import Path

from icd_coder import index_store
from icd_coder.catalogue import Catalogue
from icd_coder.lexical import Label, LabelIndex, ScoredCode, catalogue_labels, query_from_text
from icd_coder.offline import OfflineCoder, specific_codes


def test_fully_mentioned_specific_label_beats_partial_match() -> None:
    index = LabelIndex.build(
        [
            Label("A1", "acute bronchitis"),
            Label("A2", "chronic bronchitis"),
            Label("A3", "acute"),
        ]
    )
    ranked = index.search(query_from_text("Imp: acute bronchitis, likely post-viral"))
    assert ranked[0].code == "A1"


def test_code_scores_as_its_best_label() -> None:
    index = LabelIndex.build(
        [Label("B1", "tungiasis"), Label("B1", "jiggers"), Label("B2", "fleas")]
    )
    ranked = index.search(query_from_text("Dx jiggers"))
    assert [scored.code for scored in ranked] == ["B1"]


def test_saved_index_ranks_like_the_original(tmp_path: Path) -> None:
    labels = [
        Label("D1", "acute otitis media"),
        Label("D2", "otitis externa"),
        Label("D2", "swimmer ear"),
    ]
    built = LabelIndex.build(labels)
    built.save(tmp_path / "index.npz")
    loaded = LabelIndex.load(tmp_path / "index.npz")
    query = query_from_text("Dx: acute otitis media, L ear")
    assert loaded.search(query) == built.search(query)


def test_index_cache_rebuilds_when_unreadable(tmp_path: Path) -> None:
    labels = [Label("E1", "gastritis")]
    path = tmp_path / f"labels-{index_store.fingerprint(labels)}.npz"
    path.write_bytes(b"not an npz file")
    index = index_store.load_or_build(labels, tmp_path)
    assert index.search(query_from_text("gastritis"))[0].code == "E1"
    assert LabelIndex.load(path).search(query_from_text("gastritis"))[0].code == "E1"


def test_empty_query_returns_nothing() -> None:
    index = LabelIndex.build([Label("C1", "asthma")])
    assert index.search({}) == []
    assert index.search(query_from_text("the and of")) == []


def test_specific_codes_keeps_a_descendant_that_outscores_its_parent(catalogue: Catalogue) -> None:
    ranked = [ScoredCode("CA23.30", 6.0), ScoredCode("CA23", 5.0), ScoredCode("1F40", 3.0)]
    assert specific_codes(ranked, catalogue) == ("CA23.30", "1F40.Z")


def test_specific_codes_ignores_weaker_descendants(catalogue: Catalogue) -> None:
    ranked = [ScoredCode("1D80", 5.0), ScoredCode("1D80.1", 4.0)]
    assert specific_codes(ranked, catalogue) == ("1D80.0", "1D80.1")


def test_specific_codes_falls_back_to_default_child(catalogue: Catalogue) -> None:
    ranked = [ScoredCode("1D80", 2.0), ScoredCode("5A11", 1.0)]
    assert specific_codes(ranked, catalogue) == ("1D80.0", "5A11")


def test_offline_coder_end_to_end(catalogue: Catalogue) -> None:
    coder = OfflineCoder(catalogue, LabelIndex.build(catalogue_labels(catalogue)))
    note = "S: fever x3/7.\nIx: BS for MPS positive.\nA: Uncomplicated falciparum malaria.\nP: AL."
    assert coder.predict(note)[0] == "1F40.Z"
