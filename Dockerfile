# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.12.24@sha256:3af4716e991d6956a41e573eab705d0ee08500cd829ed30293eb8472f372c65a AS uv
FROM python:3.14-slim-bookworm@sha256:48b13b003dda20b16f9442b8475aa05fe21bf6579a8c881db92ffb4d8fd20f83 AS runtime

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
