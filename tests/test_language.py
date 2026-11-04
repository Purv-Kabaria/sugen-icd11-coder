from __future__ import annotations

import pytest

from icd_coder.language import detect
from icd_coder.lexical import Label, LabelIndex, query_from_text


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("S: 28M c/o fever with chills x3/7. A: falciparum malaria. P: AL BD.", "en"),
        ("Patient de 34 ans avec fièvre et toux depuis 5 jours. Pas de dyspnée.", "fr"),
        ("Paciente con dolor abdominal desde hace dos días, sin fiebre.", "es"),
        ("Paciente com dor torácica há dois dias, sem febre. Não fuma.", "pt"),
        ("Mgonjwa ana homa na kikohozi kwa siku tatu, hana kuhara.", "sw"),
        ("المريض يعاني من حمى وسعال منذ ثلاثة أيام", "ar"),
        ("Пациент жалуется на кашель и температуру три дня", "ru"),
        ("患者咳嗽发热三天", "zh"),
        ("मरीज़ को तीन दिन से बुखार और खांसी है", "hi"),
        ("BP 160/100, RBS 9.1", "en"),
    ],
)
def test_detect(text: str, expected: str) -> None:
    assert detect(text) == expected


def test_language_filter_ignores_other_languages_labels() -> None:
    index = LabelIndex.build(
        [
            Label("A1", "malaria"),
            Label("A2", "paludisme grave", "fr"),
            Label("A3", "severe malaria"),
        ]
    )
    query = query_from_text("paludisme grave")
    assert [scored.code for scored in index.search(query, languages={"en", "fr"})] == ["A2"]
    assert index.search(query, languages={"en"}) == []
    assert index.search(query, languages={"en", "es"}) == []
