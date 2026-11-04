from __future__ import annotations

from icd_coder.clinical_text import (
    CONTEXT_WEIGHT,
    DIAGNOSTIC_WEIGHT,
    PLAN_WEIGHT,
    affirmed_part,
    weighted_clauses,
)


def weights(text: str) -> dict[str, float]:
    return {clause.text: clause.weight for clause in weighted_clauses(text)}


def test_assessment_header_marks_the_diagnosis_and_is_dropped() -> None:
    result = weights("O: T 38.9, PR 104.\nA: Uncomplicated falciparum malaria.\nP: Coartem.")
    assert result["Uncomplicated falciparum malaria."] == DIAGNOSTIC_WEIGHT
    assert result["Coartem."] == PLAN_WEIGHT
    assert result["T 38.9"] == CONTEXT_WEIGHT
    assert not any("A:" in text for text in result)


def test_inline_diagnosis_and_negated_alternatives() -> None:
    result = weights("mRDT NEG. Dx viral URTI, malaria excluded.")
    assert result["viral URTI upper respiratory tract infection"] == DIAGNOSTIC_WEIGHT
    assert not any("malaria" in text for text in result)


def test_negation_scope_keeps_the_affirmed_words() -> None:
    assert affirmed_part("Functional constipation (no alarm features") == (
        "Functional constipation ("
    )
    assert affirmed_part("malaria excluded") == ""
    assert affirmed_part("No cough") == ""


def test_lay_speakers_never_count_as_diagnosis() -> None:
    result = weights("Patient: I think it's my thyroid.\nDoctor: This is acute sinusitis.")
    assert result["I think it's my thyroid."] == CONTEXT_WEIGHT
    assert result["This is acute sinusitis."] == DIAGNOSTIC_WEIGHT
