"""Runtime settings, read once from the environment at startup."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

PREFIX = "ICD_CODER_"


@dataclass(frozen=True, slots=True)
class CompatibleEndpoint:
    """An OpenAI-compatible model server: Groq, OpenRouter, vLLM, Ollama and the like."""

    base_url: str
    model: str
    api_key: str = field(repr=False)
    price: tuple[float, float] | None
    effort: str | None
    tokens_per_minute: int | None


@dataclass(frozen=True, slots=True)
class Settings:
    kb_dir: Path
    cache_dir: Path
    model_dir: Path | None
    languages: tuple[str, ...]
    deadline_seconds: float
    budget_usd: float
    concurrency: int
    provider: str
    anthropic_model: str
    anthropic_effort: str
    openai_model: str
    openai_effort: str
    max_families: int
    request_timeout_seconds: float
    failure_limit: int
    response_cache_dir: Path | None
    compatible: CompatibleEndpoint | None
    bedrock_model: str
    bedrock_region: str
    bedrock_effort: str | None
    has_anthropic_key: bool
    has_openai_key: bool

    @classmethod
    def from_environment(cls, environ: Mapping[str, str]) -> Settings:
        def setting(name: str, default: str) -> str:
            return environ.get(PREFIX + name, "").strip() or default

        languages = setting("LANGUAGES", "fr,es,pt,ar,ru,zh")
        return cls(
            kb_dir=Path(setting("KB_DIR", "data/kb")),
            cache_dir=Path(setting("CACHE_DIR", "cache/index")),
            model_dir=None
            if (models := setting("MODEL_DIR", "data/models")) == "none"
            else Path(models),
            languages=tuple(language for language in languages.split(",") if language),
            deadline_seconds=float(setting("DEADLINE_SECONDS", "2700")),
            budget_usd=float(setting("BUDGET_USD", "4.50")),
            concurrency=int(setting("CONCURRENCY", "16")),
            provider=setting("PROVIDER", "auto").lower(),
            anthropic_model=setting("ANTHROPIC_MODEL", "claude-haiku-5-5"),
            anthropic_effort=setting("ANTHROPIC_EFFORT", "low"),
            openai_model=setting("OPENAI_MODEL", "gpt-5.4-mini"),
            openai_effort=setting("OPENAI_EFFORT", "low"),
            max_families=int(setting("MAX_FAMILIES", "10")),
            request_timeout_seconds=float(setting("REQUEST_TIMEOUT_SECONDS", "60")),
            failure_limit=int(setting("FAILURE_LIMIT", "12")),
            response_cache_dir=Path(cache) if (cache := setting("RESPONSE_CACHE", "")) else None,
            compatible=_compatible_endpoint(setting),
            bedrock_model=setting("BEDROCK_MODEL", "us.anthropic.claude-haiku-5-5"),
            bedrock_region=setting("BEDROCK_REGION", environ.get("AWS_REGION", "us-east-1")),
            bedrock_effort=None
            if (bedrock_effort := setting("BEDROCK_EFFORT", "low")) == "none"
            else bedrock_effort,
            has_anthropic_key=bool(environ.get("ANTHROPIC_API_KEY", "").strip()),
            has_openai_key=bool(environ.get("OPENAI_API_KEY", "").strip()),
        )

    def chosen_provider(self) -> str | None:
        """The provider to call, or None to run offline only."""
        if self.provider == "bedrock":
            return "bedrock"
        available = {
            "anthropic": self.has_anthropic_key,
            "openai": self.has_openai_key,
            "compatible": self.compatible is not None,
        }
        if self.provider == "none":
            return None
        if self.provider in available:
            return self.provider if available[self.provider] else None
        return next((name for name, present in available.items() if present), None)


def _compatible_endpoint(setting: Callable[[str, str], str]) -> CompatibleEndpoint | None:
    base_url, model = setting("COMPATIBLE_BASE_URL", ""), setting("COMPATIBLE_MODEL", "")
    if not base_url or not model:
        return None
    price = setting("COMPATIBLE_PRICE", "")
    input_price, _, output_price = price.partition(",")
    tokens_per_minute = setting("COMPATIBLE_TOKENS_PER_MINUTE", "")
    return CompatibleEndpoint(
        base_url=base_url,
        model=model,
        api_key=setting("COMPATIBLE_API_KEY", "none"),
        price=(float(input_price), float(output_price)) if price else None,
        effort=setting("COMPATIBLE_EFFORT", "") or None,
        tokens_per_minute=int(tokens_per_minute) if tokens_per_minute else None,
    )


def load_dotenv(path: Path) -> None:
    """Set variables from a `KEY=value` file without overriding the real environment."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and not key.strip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
