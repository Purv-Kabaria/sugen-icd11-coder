"""Models that Amazon Bedrock sells itself, through its model-neutral Converse API.

One adapter reaches every Bedrock text model (DeepSeek, Kimi, GLM, Qwen, Mistral,
gpt-oss, Nova). These are billed by AWS directly, unlike Claude on Bedrock, which is
sold through AWS Marketplace. Converse has no schema-constrained decoding for most of
these models, so the schema goes into the instructions and the JSON object is taken
from the reply. For development only: boto3 is a development dependency and the
grading box cannot reach Bedrock.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from icd_coder.llm.providers import Price, ProviderError, Reply, Usage, price_of

JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


class ConverseProvider:
    def __init__(self, model: str, *, region: str, timeout_seconds: float) -> None:
        import boto3  # noqa: PLC0415
        from botocore.config import Config  # noqa: PLC0415

        self.model = model
        self.price: Price = price_of(model)
        self._client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(
                connect_timeout=10,
                read_timeout=timeout_seconds,
                retries={"max_attempts": 3, "mode": "adaptive"},
            ),
        )

    def complete_json(
        self, system: str, prompt: str, schema: Mapping[str, Any], max_output_tokens: int
    ) -> Reply:
        from botocore.exceptions import BotoCoreError, ClientError  # noqa: PLC0415

        instructions = (
            f"{system}\n\nAnswer with one JSON object matching this schema and nothing "
            f"else:\n{json.dumps(schema)}"
        )
        try:
            response = self._client.converse(
                modelId=self.model,
                system=[{"text": instructions}],
                messages=[{"role": "user", "content": [{"text": prompt}]}],
                inferenceConfig={"maxTokens": max_output_tokens},
            )
        except ClientError as error:
            raise ProviderError(f"bedrock {error.response['Error']['Code']}") from error
        except BotoCoreError as error:
            raise ProviderError(f"bedrock {type(error).__name__}") from error
        if response["stopReason"] != "end_turn":
            raise ProviderError(f"bedrock stop reason {response['stopReason']}")
        text = "".join(
            block["text"] for block in response["output"]["message"]["content"] if "text" in block
        )
        usage = Usage(
            input_tokens=response["usage"]["inputTokens"],
            output_tokens=response["usage"]["outputTokens"],
        )
        return Reply(json_object(text), usage, self.price.cost(usage))


def json_object(text: str) -> dict[str, Any]:
    """The JSON object in a reply, ignoring code fences or words around it."""
    match = JSON_OBJECT.search(text)
    if match is None:
        raise ProviderError("reply has no JSON object")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        raise ProviderError("reply is not JSON") from error
    if not isinstance(data, dict):
        raise ProviderError("reply is not a JSON object")
    return data
