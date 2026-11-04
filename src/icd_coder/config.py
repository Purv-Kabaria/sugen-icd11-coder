"""Runtime settings, read once from the environment at startup."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

PREFIX = "ICD_CODER_"


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
            model_dir=Path(models) if (models := setting("MODEL_DIR", "data/models")) else None,
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
            has_anthropic_key=bool(environ.get("ANTHROPIC_API_KEY", "").strip()),
            has_openai_key=bool(environ.get("OPENAI_API_KEY", "").strip()),
        )

    def chosen_provider(self) -> str | None:
        """The provider to call, or None to run offline only."""
        available = {"anthropic": self.has_anthropic_key, "openai": self.has_openai_key}
        if self.provider == "none":
            return None
        if self.provider in available:
            return self.provider if available[self.provider] else None
        return next((name for name, present in available.items() if present), None)


def load_dotenv(path: Path) -> None:
    """Set variables from a `KEY=value` file without overriding the real environment.

    For development tools only; the container takes its keys from the environment.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and not key.strip().startswith("#"):
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))
