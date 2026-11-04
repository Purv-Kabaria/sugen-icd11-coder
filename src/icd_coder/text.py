"""Normalisation and tokenisation shared by clinical notes and WHO labels."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import snowballstemmer

TOKEN_PATTERN = re.compile(r"[^\W_]+")
LATIN_LIMIT = 0x0250
MIN_VARIANT_LENGTH = 5

STOPWORDS = frozenset(
    """
    a an and are as at be been but by for from had has have he her his i if in into is it its
    me my no not of on or our she so than that the their them then there these they this to
    was we were what when which who will with you your
    le la les de des du un une et en au aux pour par sur avec dans est
    el los las del y con por para una uno es
    o os da do das dos em um uma com
    der die das und mit ein eine ist von zu
    nos sai nep
    """.split()
)
NUMBER_QUALIFIERS = frozenset(
    "type stage grade class degree phase trimester tipo stade grado estadio".split()
)
CJK_PATTERN = re.compile(r"[぀-ヿ㐀-鿿豈-﫿]")


def _latin_folding_table() -> dict[int, str]:
    table = {0xFFFD: " "}
    for code_point in range(0x80, LATIN_LIMIT):
        decomposed = unicodedata.normalize("NFKD", chr(code_point))
        stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
        if stripped != chr(code_point):
            table[code_point] = stripped
    return table


LATIN_FOLDING = _latin_folding_table()
STEMMER = snowballstemmer.stemmer("english")


def fold(text: str) -> str:
    """Lowercase, strip Latin diacritics and drop replacement characters.

    Only Latin letters lose their marks: in Devanagari or Arabic the combining marks
    carry vowels, so removing them would destroy the word.
    """
    return text.translate(LATIN_FOLDING).lower()


@lru_cache(maxsize=200_000)
def _matching_form(word: str) -> str:
    return str(STEMMER.stemWord(unify_spelling(word))) if word.isascii() else word


def unify_spelling(token: str) -> str:
    """Map British spellings onto American ones so `haemorrhoids` meets `hemorrhoids`."""
    if len(token) < MIN_VARIANT_LENGTH or not token.isascii():
        return token
    token = token.replace("ae", "e").replace("oe", "e")
    if token.endswith("our"):
        token = token[:-3] + "or"
    return token


def tokens(text: str) -> list[str]:
    """Content tokens in matching form: folded, spelling-unified, stemmed, no stopwords.

    A bare number only matters next to a qualifier ("type 2", "stage 1"), so it is fused
    with that word; any other number would match doses and vital signs. Runs of CJK
    characters, which have no spaces, become overlapping character bigrams.
    """
    result: list[str] = []
    previous = ""
    for token in TOKEN_PATTERN.findall(fold(text)):
        if token.isdigit():
            if previous in NUMBER_QUALIFIERS:
                result.append(f"{previous}{token}")
        elif CJK_PATTERN.search(token):
            result.extend(_bigrams(token))
        elif token not in STOPWORDS and len(token) > 1:
            result.append(_matching_form(token))
        previous = token
    return result


def _bigrams(token: str) -> list[str]:
    if len(token) < 2:
        return [token]
    return [token[index : index + 2] for index in range(len(token) - 1)]
