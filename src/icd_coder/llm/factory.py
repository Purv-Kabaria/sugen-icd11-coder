"""Assemble the model coder from settings."""

from __future__ import annotations

import logging

import anthropic

from icd_coder.catalogue import Catalogue
from icd_coder.config import Settings
from icd_coder.guards import Budget, FailureStreak
from icd_coder.lexical import LabelIndex
from icd_coder.llm.coder import ModelCoder
from icd_coder.llm.providers import (
    UNKNOWN_MODEL_PRICE,
    AnthropicProvider,
    CachingProvider,
    ChatCompletionsProvider,
    OpenAIProvider,
    Price,
    Provider,
)
from icd_coder.llm.rate_limit import TokenRateLimiter

LOG = logging.getLogger(__name__)


def _provider(settings: Settings, name: str) -> Provider:
    if name == "anthropic":
        return AnthropicProvider(
            settings.anthropic_model,
            effort=settings.anthropic_effort,
            timeout_seconds=settings.request_timeout_seconds,
        )
    if name == "bedrock":
        client = anthropic.AnthropicBedrock(
            aws_region=settings.bedrock_region,
            timeout=settings.request_timeout_seconds,
            max_retries=2,
        )
        return AnthropicProvider(
            settings.bedrock_model,
            effort=settings.bedrock_effort,
            timeout_seconds=settings.request_timeout_seconds,
            client=client,
        )
    if name == "openai":
        return OpenAIProvider(
            settings.openai_model,
            effort=settings.openai_effort,
            timeout_seconds=settings.request_timeout_seconds,
        )
    endpoint = settings.compatible
    if endpoint is None:
        msg = "no compatible endpoint configured"
        raise ValueError(msg)
    if endpoint.price is None:
        LOG.warning("no price for %s; counting spend at a cautious default", endpoint.model)
    return ChatCompletionsProvider(
        endpoint.model,
        base_url=endpoint.base_url,
        api_key=endpoint.api_key,
        timeout_seconds=settings.request_timeout_seconds,
        price=Price(*endpoint.price, 0.0, 0.0) if endpoint.price else UNKNOWN_MODEL_PRICE,
        effort=endpoint.effort,
        limiter=TokenRateLimiter(endpoint.tokens_per_minute)
        if endpoint.tokens_per_minute
        else None,
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
    except (ValueError, OSError, ImportError) as error:
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
