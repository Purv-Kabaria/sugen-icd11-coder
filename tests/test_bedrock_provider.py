from types import SimpleNamespace
from typing import Any

from icd_coder.catalogue import Catalogue
from icd_coder.config import Settings
from icd_coder.guards import Budget
from icd_coder.lexical import LabelIndex, catalogue_labels
from icd_coder.llm.factory import build_model_coder
from icd_coder.llm.providers import PRICES, UNKNOWN_MODEL_PRICE, AnthropicProvider, price_of


class RecordingClient:
    """Stands in for an Anthropic or Bedrock client and records the request."""

    def __init__(self) -> None:
        self.request: dict[str, Any] = {}
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **request: Any) -> Any:
        self.request = request
        usage = SimpleNamespace(
            input_tokens=1000,
            output_tokens=100,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        block = SimpleNamespace(type="text", text='{"codes": ["1F40.Z"]}')
        return SimpleNamespace(stop_reason="end_turn", content=[block], usage=usage)


def test_bedrock_is_only_used_when_chosen() -> None:
    assert Settings.from_environment({}).chosen_provider() is None
    chosen = Settings.from_environment({"ICD_CODER_PROVIDER": "bedrock"})
    assert chosen.chosen_provider() == "bedrock"
    assert chosen.bedrock_model == "us.anthropic.claude-haiku-5-5"


def test_bedrock_effort_can_be_omitted_for_older_models() -> None:
    environ = {"ICD_CODER_PROVIDER": "bedrock", "ICD_CODER_BEDROCK_EFFORT": "none"}
    assert Settings.from_environment(environ).bedrock_effort is None


def test_bedrock_ids_resolve_to_claude_prices() -> None:
    assert price_of("us.anthropic.claude-haiku-5-5") == PRICES["claude-haiku-5-5"]
    assert price_of("us.anthropic.claude-haiku-4-5-20251001-v1:0") == PRICES["claude-haiku-4-5"]
    assert price_of("some-other-model") == UNKNOWN_MODEL_PRICE


def test_effort_is_sent_only_when_set() -> None:
    client = RecordingClient()
    with_effort = AnthropicProvider(
        "claude-haiku-5-5",
        effort="low",
        timeout_seconds=5,
        client=client,  # type: ignore[arg-type]
    )
    with_effort.complete_json("system", "prompt", {"type": "object"}, 100)
    assert client.request["output_config"]["effort"] == "low"

    without_effort = AnthropicProvider(
        "claude-haiku-4-5",
        effort=None,
        timeout_seconds=5,
        client=client,  # type: ignore[arg-type]
    )
    reply = without_effort.complete_json("system", "prompt", {"type": "object"}, 100)
    assert "effort" not in client.request["output_config"]
    assert reply.data == {"codes": ["1F40.Z"]}
    assert reply.cost_usd == (1000 * 1.00 + 100 * 5.00) / 1_000_000


def test_factory_builds_a_bedrock_coder(catalogue: Catalogue) -> None:
    index = LabelIndex.build(catalogue_labels(catalogue))
    settings = Settings.from_environment({"ICD_CODER_PROVIDER": "bedrock"})
    assert build_model_coder(settings, catalogue, index, Budget(1.0)) is not None
