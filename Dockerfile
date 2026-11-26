# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.13.10@sha256:a3ea1357342c1a4bf9a0c53a3deb1b55379aefa037338d8113d5771efa01774f AS uv
FROM python:3.11-slim-bookworm@sha256:0a310eeecf4e1f5a0743f9a6520c90c88d089c903ca5fd283f501e3a805f5f89 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    XDG_CACHE_HOME=/tmp/.cache \
    HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface

COPY --from=uv /uv /uvx /usr/local/bin/

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /app /cache/huggingface \
    && chown -R app:app /home/app /cache

WORKDIR /app

# Install locked third-party dependencies first for effective layer caching.
COPY --chown=app:app pyproject.toml uv.lock README.md LICENSE THIRD_PARTY_NOTICES.md ./
RUN uv sync --locked --no-dev

COPY --chown=app:app src ./src
COPY --chown=app:app scripts ./scripts
COPY --chown=app:app data/episodes.json data/diagram_captions.json data/diagram_captions_provenance.json ./data/
COPY --chown=app:app youtube_show.json ./

USER app

ENTRYPOINT [".venv/bin/python", "scripts/rag.py"]
CMD ["--help"]
