from types import SimpleNamespace
from typing import Any

from icd_coder.catalogue import Catalogue
from icd_coder.config import Settings
from icd_coder.guards import Budget
from icd_coder.lexical import LabelIndex, catalogue_labels
from icd_coder.llm.factory import build_model_coder
from icd_coder.llm.providers import UNKNOWN_MODEL_PRICE, ChatCompletionsProvider, Price

ENDPOINT = {
    "ICD_CODER_COMPATIBLE_BASE_URL": "http://localhost:11434/v1",
    "ICD_CODER_COMPATIBLE_MODEL": "local-model",
    "ICD_CODER_COMPATIBLE_API_KEY": "secret-value",
}


def test_compatible_endpoint_needs_a_url_and_a_model() -> None:
    assert Settings.from_environment({}).compatible is None
    only_url = {"ICD_CODER_COMPATIBLE_BASE_URL": "http://localhost/v1"}
    assert Settings.from_environment(only_url).compatible is None
    assert Settings.from_environment(ENDPOINT).chosen_provider() == "compatible"


def test_official_providers_win_automatic_choice() -> None:
    settings = Settings.from_environment({**ENDPOINT, "ANTHROPIC_API_KEY": "k"})
    assert settings.chosen_provider() == "anthropic"


def test_endpoint_key_never_appears_in_settings_repr() -> None:
    assert "secret-value" not in repr(Settings.from_environment(ENDPOINT))


def test_endpoint_price_and_limits_are_parsed() -> None:
    environ = {
        **ENDPOINT,
        "ICD_CODER_COMPATIBLE_PRICE": "0.15,0.60",
        "ICD_CODER_COMPATIBLE_TOKENS_PER_MINUTE": "8000",
    }
    endpoint = Settings.from_environment(environ).compatible
    assert endpoint is not None
    assert endpoint.price == (0.15, 0.60)
    assert endpoint.tokens_per_minute == 8000


def test_factory_builds_a_coder_for_the_endpoint(catalogue: Catalogue) -> None:
    index = LabelIndex.build(catalogue_labels(catalogue))
    coder = build_model_coder(Settings.from_environment(ENDPOINT), catalogue, index, Budget(1.0))
    assert coder is not None


def fake_completion(content: str, prompt_tokens: int, completion_tokens: int) -> Any:
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    choice = SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))
    return SimpleNamespace(choices=[choice], usage=usage)


def test_replies_are_billed_from_reported_usage() -> None:
    provider = ChatCompletionsProvider(
        "local-model",
        base_url="http://localhost/v1",
        api_key="k",
        timeout_seconds=5,
        price=Price(2.0, 10.0, 0.0, 0.0),
    )
    completions = SimpleNamespace(create=lambda **_: fake_completion('{"codes": []}', 1000, 100))
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))  # type: ignore[assignment]

    reply = provider.complete_json("system", "prompt", {"type": "object"}, 200)

    assert reply.data == {"codes": []}
    assert reply.cost_usd == (1000 * 2.0 + 100 * 10.0) / 1_000_000


def test_unpriced_endpoints_are_billed_cautiously() -> None:
    provider = ChatCompletionsProvider(
        "local-model", base_url="http://localhost/v1", api_key="k", timeout_seconds=5
    )
    assert provider.price == UNKNOWN_MODEL_PRICE
