"""The model coder: read the note, gather candidates, choose among them."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

from icd_coder.catalogue import Catalogue
from icd_coder.guards import Budget, FailureStreak
from icd_coder.lexical import LabelIndex, ScoredCode, query_from_text
from icd_coder.llm.candidates import build_candidates, fuse
from icd_coder.llm.prompts import (
    CHOICE_SCHEMA,
    CHOICE_SYSTEM,
    READING_SCHEMA,
    READING_SYSTEM,
    choice_prompt,
    reading_prompt,
)
from icd_coder.llm.providers import Provider, ProviderError
from icd_coder.notes import Note

LOG = logging.getLogger(__name__)
READING_MAX_TOKENS = 600
CHOICE_MAX_TOKENS = 800
RESULTS_PER_QUERY = 15
MAX_GUESSES = 5
MAX_SEARCH_TERMS = 6


class BudgetExhaustedError(ProviderError):
    """The next call could push spend past the ceiling."""


class ModelCoder:
    def __init__(
        self,
        provider: Provider,
        catalogue: Catalogue,
        index: LabelIndex,
        budget: Budget,
        failures: FailureStreak,
        *,
        max_families: int = 10,
    ) -> None:
        self._provider = provider
        self._catalogue = catalogue
        self._index = index
        self._budget = budget
        self._failures = failures
        self._max_families = max_families

    def code(self, note: Note, ranked: Sequence[ScoredCode]) -> Sequence[str]:
        if self._failures.tripped():
            raise ProviderError("provider disabled after repeated failures")
        reading = self._call(
            READING_SYSTEM, reading_prompt(note.text), READING_SCHEMA, READING_MAX_TOKENS
        )
        main_condition = _text(reading.get("main_condition"))
        fused = fuse(self._rankings(reading, main_condition, ranked))
        candidates = build_candidates(fused, self._catalogue, self._max_families)
        choice = self._call(
            CHOICE_SYSTEM,
            choice_prompt(note.text, main_condition, candidates.text),
            CHOICE_SCHEMA,
            CHOICE_MAX_TOKENS,
        )
        chosen = _strings(choice.get("codes"))
        return [*chosen, *fused]

    def _rankings(
        self, reading: Mapping[str, Any], main_condition: str, ranked: Sequence[ScoredCode]
    ) -> list[list[str]]:
        queries = [main_condition, *_strings(reading.get("search_terms"))[:MAX_SEARCH_TERMS]]
        rankings = [
            [
                scored.code
                for scored in self._index.search(query_from_text(query), RESULTS_PER_QUERY)
            ]
            for query in queries
            if query
        ]
        guesses = [self._catalogue.validate(code) for code in _strings(reading.get("icd11_codes"))]
        rankings.append([code for code in guesses[:MAX_GUESSES] if code is not None])
        rankings.append([scored.code for scored in ranked[:RESULTS_PER_QUERY]])
        return rankings

    def _call(
        self, system: str, prompt: str, schema: Mapping[str, Any], max_tokens: int
    ) -> Mapping[str, Any]:
        worst_case = self._provider.price.worst_case(len(system) + len(prompt), max_tokens)
        if not self._budget.can_afford(worst_case):
            raise BudgetExhaustedError(f"spend ceiling reached at ${self._budget.spent:.2f}")
        try:
            reply = self._provider.complete_json(system, prompt, schema, max_tokens)
        except ProviderError:
            self._failures.record(succeeded=False)
            raise
        self._failures.record(succeeded=True)
        self._budget.charge(reply.cost_usd)
        return reply.data


def _text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _strings(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]
