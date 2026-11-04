from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from icd_coder.catalogue import Catalogue
from icd_coder.guards import Budget, FailureStreak
from icd_coder.lexical import LabelIndex, catalogue_labels
from icd_coder.llm.candidates import build_candidates, fuse
from icd_coder.llm.coder import BudgetExhaustedError, ModelCoder
from icd_coder.llm.prompts import CHOICE_SCHEMA, READING_SCHEMA
from icd_coder.llm.providers import (
    CachingProvider,
    Price,
    ProviderError,
    Reply,
    Usage,
    anthropic_effort,
)
from icd_coder.notes import Note

PRICE = Price(1.0, 1.0, 0.1, 1.0)


class FakeProvider:
    model = "fake"
    price = PRICE

    def __init__(self, replies: list[dict[str, Any] | Exception]) -> None:
        self._replies = replies
        self.prompts: list[str] = []

    def complete_json(
        self, system: str, prompt: str, schema: Mapping[str, Any], max_output_tokens: int
    ) -> Reply:
        self.prompts.append(prompt)
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return Reply(reply, Usage(1000, 100), 0.0011)


READING = {
    "language": "en",
    "main_condition": "uncomplicated falciparum malaria",
    "search_terms": ["falciparum malaria", "malaria"],
    "icd11_codes": ["1F40.Z", "made-up"],
}


def coder_for(
    catalogue: Catalogue, provider: FakeProvider, budget: float = 1.0, failure_limit: int = 3
) -> ModelCoder:
    index = LabelIndex.build(catalogue_labels(catalogue))
    return ModelCoder(provider, catalogue, index, Budget(budget), FailureStreak(failure_limit))


def test_two_calls_choose_among_rendered_candidates(catalogue: Catalogue) -> None:
    provider = FakeProvider([READING, {"reasoning": "no severe features", "codes": ["1F40.Z"]}])
    codes = coder_for(catalogue, provider).code(Note("n", "A: malaria, P. falciparum"), [])
    assert codes[0] == "1F40.Z"
    choice_prompt = provider.prompts[1]
    assert "uncomplicated falciparum malaria" in choice_prompt
    assert "1F40.Y Other severe or complicated" in choice_prompt
    assert "\n  1F40.Z" in choice_prompt


def test_budget_stops_calls_before_they_are_made(catalogue: Catalogue) -> None:
    provider = FakeProvider([READING])
    with pytest.raises(BudgetExhaustedError):
        coder_for(catalogue, provider, budget=0.0001).code(Note("n", "malaria"), [])
    assert provider.prompts == []


def test_failure_streak_disables_the_provider(catalogue: Catalogue) -> None:
    failures: list[dict[str, Any] | Exception] = [ProviderError("down")] * 2
    provider = FakeProvider(failures)
    coder = coder_for(catalogue, provider, failure_limit=2)
    for _ in range(2):
        with pytest.raises(ProviderError):
            coder.code(Note("n", "malaria"), [])
    with pytest.raises(ProviderError, match="disabled"):
        coder.code(Note("n", "malaria"), [])
    assert len(provider.prompts) == 2


def test_caching_provider_replays_without_cost(tmp_path: Path) -> None:
    inner = FakeProvider([{"answer": 1}])
    cached = CachingProvider(inner, tmp_path)
    first = cached.complete_json("system", "prompt", READING_SCHEMA, 100)
    second = cached.complete_json("system", "prompt", READING_SCHEMA, 100)
    assert first.data == second.data == {"answer": 1}
    assert second.cost_usd == 0.0
    assert len(inner.prompts) == 1


def test_fuse_rewards_agreement_across_rankings() -> None:
    assert fuse([["A", "B", "C"], ["B", "D"], ["B"]])[0] == "B"
    assert fuse([["A", "A", "B"]]) == ["A", "B"]


def test_candidates_group_by_family_and_cap_families(catalogue: Catalogue) -> None:
    candidates = build_candidates(["1F40.0", "CA23.30", "1D80.1", "5A11"], catalogue, 2)
    assert candidates.families == ("1F40", "CA23")
    assert {"1F40", "1F40.Z", "CA23.3", "CA23.32"} <= candidates.codes
    assert "1D80" not in candidates.text


def test_schemas_are_strict() -> None:
    for schema in (READING_SCHEMA, CHOICE_SCHEMA):
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])


def test_unsupported_effort_fails_at_construction() -> None:
    assert anthropic_effort("low") == "low"
    with pytest.raises(ValueError, match="effort"):
        anthropic_effort("turbo")
