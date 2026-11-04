FROM python:3.12.9-slim-bookworm AS base

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app


FROM base AS build

COPY --from=ghcr.io/astral-sh/uv:0.11.13 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PYTHON_DOWNLOADS=never

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project --no-editable

COPY src ./src
RUN uv sync --locked --no-editable

COPY data/kb ./data/kb
ENV ICD_CODER_KB_DIR=/app/data/kb \
    ICD_CODER_CACHE_DIR=/app/cache/index
RUN /app/.venv/bin/icd-coder-prepare


FROM base

COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/data/kb /app/data/kb
COPY --from=build /app/cache /app/cache
COPY data/models /app/data/models

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ICD_CODER_KB_DIR=/app/data/kb \
    ICD_CODER_CACHE_DIR=/app/cache/index \
    ICD_CODER_MODEL_DIR=/app/data/models \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

ENTRYPOINT ["icd-coder"]
