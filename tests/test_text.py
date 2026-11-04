from __future__ import annotations

from icd_coder.text import fold, tokens, unify_spelling


def test_fold_strips_latin_accents_but_keeps_other_scripts() -> None:
    assert fold("Diarrhée AIGUË") == "diarrhee aigue"
    assert fold("बुखार") == "बुखार"
    assert fold("pain � worse") == "pain   worse"


def test_unify_spelling_maps_british_to_american() -> None:
    assert unify_spelling("haemorrhoids") == "hemorrhoids"
    assert unify_spelling("oedema") == "edema"
    assert unify_spelling("tumour") == "tumor"
    assert unify_spelling("toe") == "toe"


def test_tokens_match_across_spelling_and_inflection() -> None:
    assert tokens("Haemorrhoids") == tokens("hemorrhoid")
    assert tokens("infections of the bladder") == tokens("bladder infection")[::-1]


def test_numbers_only_survive_next_to_a_qualifier() -> None:
    assert tokens("Type 2 diabetes, metformin 500 mg BD x 2") == [
        "type",
        "type2",
        "diabet",
        "metformin",
        "mg",
        "bd",
    ]


def test_cjk_runs_become_bigrams() -> None:
    assert tokens("急性阑尾炎") == ["急性", "性阑", "阑尾", "尾炎"]
