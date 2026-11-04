"""One interface over the Anthropic and OpenAI SDKs: a JSON answer plus its cost.

Both SDKs read their key from the environment and honour `HTTPS_PROXY`, which is the
only route to the APIs on the grading box. Errors surface as `ProviderError` so the
caller can fall back without knowing which SDK raised.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast, get_args

import anthropic
import openai
from anthropic.types import MessageParam, OutputConfigParam
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.completion_create_params import ResponseFormat
from openai.types.responses import ResponseTextConfigParam
from openai.types.shared_params import Reasoning

LOG = logging.getLogger(__name__)
CHARACTERS_PER_TOKEN = 3.0

AnthropicEffort = Literal["low", "medium", "high", "xhigh", "max"]
OpenAIEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]


class ProviderError(Exception):
    """The provider gave no usable answer."""


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


@dataclass(frozen=True, slots=True)
class Price:
    """US dollars per million tokens."""

    input: float
    output: float
    cache_read: float
    cache_write: float

    def cost(self, usage: Usage) -> float:
        return (
            usage.input_tokens * self.input
            + usage.output_tokens * self.output
            + usage.cache_read_tokens * self.cache_read
            + usage.cache_write_tokens * self.cache_write
        ) / 1_000_000

    def worst_case(self, prompt_characters: int, max_output_tokens: int) -> float:
        """An upper bound before the call, used only to decide whether to make it."""
        input_tokens = prompt_characters / CHARACTERS_PER_TOKEN * 1.5
        return (input_tokens * self.input + max_output_tokens * self.output) / 1_000_000


PRICES: Mapping[str, Price] = {
    "claude-haiku-5-5": Price(0.10, 0.50, 0.01, 0.125),
    "claude-sonnet-5-5": Price(2.00, 10.00, 0.20, 2.50),
    "claude-opus-5-5": Price(4.00, 20.00, 0.20, 5.00),
    "gpt-5.4-nano": Price(0.20, 1.25, 0.02, 0.20),
    "gpt-5.4-mini": Price(0.75, 4.50, 0.075, 0.75),
    "gpt-5-mini": Price(0.25, 2.00, 0.025, 0.25),
    "gpt-5-nano": Price(0.05, 0.40, 0.005, 0.05),
}
UNKNOWN_MODEL_PRICE = Price(5.00, 25.00, 0.50, 6.25)


def price_of(model: str) -> Price:
    return PRICES.get(model, UNKNOWN_MODEL_PRICE)


@dataclass(frozen=True, slots=True)
class Reply:
    data: dict[str, Any]
    usage: Usage
    cost_usd: float


class Provider(Protocol):
    model: str
    price: Price

    def complete_json(
        self, system: str, prompt: str, schema: Mapping[str, Any], max_output_tokens: int
    ) -> Reply:
        """A JSON object matching the schema. Raises `ProviderError` when there is none."""
        ...


class AnthropicProvider:
    def __init__(
        self, model: str, *, effort: str, timeout_seconds: float, max_retries: int = 2
    ) -> None:
        self.model = model
        self.price = price_of(model)
        self._effort = anthropic_effort(effort)
        self._client = anthropic.Anthropic(timeout=timeout_seconds, max_retries=max_retries)

    def complete_json(
        self, system: str, prompt: str, schema: Mapping[str, Any], max_output_tokens: int
    ) -> Reply:
        messages: list[MessageParam] = [{"role": "user", "content": prompt}]
        output_config: OutputConfigParam = {
            "effort": self._effort,
            "format": {"type": "json_schema", "schema": dict(schema)},
        }
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=max_output_tokens,
                system=system,
                messages=messages,
                output_config=output_config,
            )
        except anthropic.APIError as error:
            raise ProviderError(f"anthropic {type(error).__name__}") from error
        if response.stop_reason != "end_turn":
            raise ProviderError(f"anthropic stop_reason {response.stop_reason}")
        text = next((block.text for block in response.content if block.type == "text"), "")
        usage = Usage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=response.usage.cache_read_input_tokens or 0,
            cache_write_tokens=response.usage.cache_creation_input_tokens or 0,
        )
        return Reply(_parse_object(text), usage, self.price.cost(usage))


class OpenAIProvider:
    def __init__(
        self, model: str, *, effort: str, timeout_seconds: float, max_retries: int = 2
    ) -> None:
        self.model = model
        self.price = price_of(model)
        self._effort = openai_effort(effort)
        self._client = openai.OpenAI(timeout=timeout_seconds, max_retries=max_retries)

    def complete_json(
        self, system: str, prompt: str, schema: Mapping[str, Any], max_output_tokens: int
    ) -> Reply:
        reasoning: Reasoning = {"effort": self._effort}
        text: ResponseTextConfigParam = {
            "format": {
                "type": "json_schema",
                "name": "answer",
                "schema": dict(schema),
                "strict": True,
            }
        }
        try:
            response = self._client.responses.create(
                model=self.model,
                instructions=system,
                input=prompt,
                max_output_tokens=max_output_tokens,
                reasoning=reasoning,
                text=text,
            )
        except openai.APIError as error:
            raise ProviderError(f"openai {type(error).__name__}") from error
        if response.status != "completed":
            raise ProviderError(f"openai status {response.status}")
        details = response.usage.input_tokens_details if response.usage else None
        cached = details.cached_tokens if details else 0
        usage = Usage(
            input_tokens=(response.usage.input_tokens if response.usage else 0) - cached,
            output_tokens=response.usage.output_tokens if response.usage else 0,
            cache_read_tokens=cached,
        )
        return Reply(_parse_object(response.output_text), usage, self.price.cost(usage))


class ChatCompletionsProvider:
    """Any OpenAI-compatible Chat Completions endpoint, such as Groq, for development.

    Not reachable from the grading box; used to generate data and to exercise the model
    path with open-weight models. Strict schema decoding is tried first; models that
    cannot decode a schema fall back to plain JSON mode with the schema in the prompt.
    """

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str,
        timeout_seconds: float,
        effort: str | None = None,
        max_retries: int = 6,
    ) -> None:
        self.model = model
        self.price = Price(0.0, 0.0, 0.0, 0.0)
        self._effort = openai_effort(effort) if effort else None
        self._client = openai.OpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout_seconds, max_retries=max_retries
        )
        self._strict = True

    def complete_json(
        self, system: str, prompt: str, schema: Mapping[str, Any], max_output_tokens: int
    ) -> Reply:
        if self._strict:
            try:
                return self._request(system, prompt, max_output_tokens, _schema_format(schema))
            except ProviderError as error:
                if "json_validate_failed" not in str(error):
                    raise
                LOG.info("%s cannot decode the schema strictly; using JSON mode", self.model)
                self._strict = False
        instructions = (
            f"{system}\n\nAnswer with one JSON object matching this schema:\n{json.dumps(schema)}"
        )
        return self._request(instructions, prompt, max_output_tokens, {"type": "json_object"})

    def _request(
        self, system: str, prompt: str, max_output_tokens: int, response_format: ResponseFormat
    ) -> Reply:
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_completion_tokens=max_output_tokens,
                response_format=response_format,
                reasoning_effort=self._effort,
            )
        except openai.BadRequestError as error:
            raise ProviderError(f"chat completions {error.code or error.message}") from error
        except openai.APIError as error:
            raise ProviderError(f"chat completions {type(error).__name__}") from error
        choice = response.choices[0]
        if choice.finish_reason != "stop" or not choice.message.content:
            raise ProviderError(f"chat completions finish_reason {choice.finish_reason}")
        usage = Usage(
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )
        return Reply(_parse_object(choice.message.content), usage, 0.0)


def _schema_format(schema: Mapping[str, Any]) -> ResponseFormat:
    return {
        "type": "json_schema",
        "json_schema": {"name": "answer", "strict": True, "schema": dict(schema)},
    }


class CachingProvider:
    """Replays stored replies for identical requests. For experiments, never for grading.

    Re-running an evaluation after changing only post-processing should cost nothing.
    Cached replies are charged at zero.
    """

    def __init__(self, inner: Provider, cache_dir: Path) -> None:
        self.model = inner.model
        self.price = inner.price
        self._inner = inner
        self._cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

    def complete_json(
        self, system: str, prompt: str, schema: Mapping[str, Any], max_output_tokens: int
    ) -> Reply:
        key = hashlib.sha256(
            json.dumps([self.model, system, prompt, schema, max_output_tokens]).encode()
        ).hexdigest()
        path = self._cache_dir / f"{key}.json"
        if path.exists():
            try:
                stored = json.loads(path.read_text(encoding="utf-8"))
                return Reply(stored["data"], Usage(**stored["usage"]), 0.0)
            except (OSError, ValueError, KeyError, TypeError):
                LOG.warning("ignoring unreadable cached reply %s", path.name)
        reply = self._inner.complete_json(system, prompt, schema, max_output_tokens)
        record = {"data": reply.data, "usage": vars_of(reply.usage), "cost_usd": reply.cost_usd}
        partial = path.with_suffix(".partial")
        partial.write_text(json.dumps(record), encoding="utf-8")
        partial.replace(path)
        return reply


def vars_of(usage: Usage) -> dict[str, int]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
    }


def anthropic_effort(value: str) -> AnthropicEffort:
    """The effort level if Anthropic accepts it; a configuration error fails at startup."""
    if value not in get_args(AnthropicEffort):
        raise ValueError(f"unsupported Anthropic effort {value!r}")
    return cast(AnthropicEffort, value)


def openai_effort(value: str) -> OpenAIEffort:
    """The effort level if OpenAI accepts it; a configuration error fails at startup."""
    if value not in get_args(OpenAIEffort):
        raise ValueError(f"unsupported OpenAI effort {value!r}")
    return cast(OpenAIEffort, value)


def _parse_object(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ProviderError("reply is not JSON") from error
    if not isinstance(data, dict):
        raise ProviderError("reply is not a JSON object")
    return data
