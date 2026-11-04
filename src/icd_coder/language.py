"""Cheap language identification, enough to pick which WHO translations to search.

Non-Latin scripts decide by character counts. Latin-script text decides by the share
of common function words, which separates the languages WHO translates from English
and from Swahili even in short, abbreviation-heavy notes.
"""

from __future__ import annotations

import re

WORD = re.compile(r"[^\W\d_]+")
SCRIPTS = (
    ("ar", re.compile(r"[؀-ۿ]")),
    ("ru", re.compile(r"[Ѐ-ӿ]")),
    ("zh", re.compile(r"[一-鿿]")),
    ("hi", re.compile(r"[ऀ-ॿ]")),
)
FUNCTION_WORDS = {
    "en": set("the and of with for was is to in on no not has have this you your patient".split()),
    "fr": set("le la les des et avec pour est une dans sur pas sans du au aux patient".split()),
    "es": set("el los las del con para una por que sin paciente dolor desde hace".split()),
    "pt": set("os das dos com para uma por que sem paciente dor desde há não".split()),
    "sw": set("na ya wa kwa ni la za katika mgonjwa siku homa hana ana kama".split()),
}
MIN_SCRIPT_SHARE = 0.2


def detect(text: str) -> str:
    """An ISO 639-1 code, or `en` when nothing else is clearly indicated."""
    letters = sum(character.isalpha() for character in text) or 1
    for language, pattern in SCRIPTS:
        if len(pattern.findall(text)) / letters >= MIN_SCRIPT_SHARE:
            return language
    words = [word.lower() for word in WORD.findall(text)]
    counts = {
        language: sum(word in vocabulary for word in words)
        for language, vocabulary in FUNCTION_WORDS.items()
    }
    best = max(counts, key=counts.__getitem__)
    return best if counts[best] > counts["en"] else "en"
