"""Assemble the model coder from settings."""

from __future__ import annotations

import logging

from icd_coder.catalogue import Catalogue
from icd_coder.config import Settings
from icd_coder.guards import Budget, FailureStreak
from icd_coder.lexical import LabelIndex
from icd_coder.llm.coder import ModelCoder
from icd_coder.llm.providers import AnthropicProvider, CachingProvider, OpenAIProvider, Provider

LOG = logging.getLogger(__name__)


def _provider(settings: Settings, name: str) -> Provider:
    if name == "anthropic":
        return AnthropicProvider(
            settings.anthropic_model,
            effort=settings.anthropic_effort,
            timeout_seconds=settings.request_timeout_seconds,
        )
    return OpenAIProvider(
        settings.openai_model,
        effort=settings.openai_effort,
        timeout_seconds=settings.request_timeout_seconds,
    )


def build_model_coder(
    settings: Settings, catalogue: Catalogue, index: LabelIndex, budget: Budget
) -> ModelCoder | None:
    """The model coder for the available provider, or None to run offline only."""
    name = settings.chosen_provider()
    if name is None:
        LOG.info("no model provider available; running offline only")
        return None
    try:
        provider = _provider(settings, name)
    except (ValueError, OSError) as error:
        LOG.error("cannot set up provider %s (%s); running offline only", name, error)
        return None
    if settings.response_cache_dir is not None:
        provider = CachingProvider(provider, settings.response_cache_dir)
    LOG.info("model provider %s, model %s, budget $%.2f", name, provider.model, settings.budget_usd)
    return ModelCoder(
        provider,
        catalogue,
        index,
        budget,
        FailureStreak(settings.failure_limit),
        max_families=settings.max_families,
    )
