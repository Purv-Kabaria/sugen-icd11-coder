FROM python:3.12.9-slim-bookworm AS build

COPY --from=ghcr.io/astral-sh/uv:0.11.13 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PYTHON_DOWNLOADS=never
WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project --no-editable

COPY src ./src
RUN uv sync --locked --no-editable

COPY data/kb ./data/kb
ENV ICD_CODER_KB_DIR=/app/data/kb \
    ICD_CODER_CACHE_DIR=/app/cache/index
RUN /app/.venv/bin/icd-coder-prepare


FROM python:3.12.9-slim-bookworm

WORKDIR /app
COPY --from=build /app/.venv /app/.venv
COPY --from=build /app/data/kb /app/data/kb
COPY --from=build /app/cache /app/cache

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ICD_CODER_KB_DIR=/app/data/kb \
    ICD_CODER_CACHE_DIR=/app/cache/index \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

ENTRYPOINT ["icd-coder"]
