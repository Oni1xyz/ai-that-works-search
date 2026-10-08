UPSTREAM_REF ?= 3274d8b09e99de6df5fa251b159b8fcfafa72b66

.PHONY: setup sources manifest transcripts corpus index test check docker-build docker-test

setup:
	uv sync --locked --group dev --python 3.11

sources:
	@if test -d source/ai-that-works/.git; then \
		test -z "$$(git -C source/ai-that-works status --porcelain)" || (echo "upstream clone is dirty; refusing to update" && exit 1); \
	else \
		git clone https://github.com/ai-that-works/ai-that-works.git source/ai-that-works; \
	fi
	git -C source/ai-that-works fetch origin --tags
	git -C source/ai-that-works checkout --detach "$(UPSTREAM_REF)"
	@raw="$$(mktemp)"; clean="$$(mktemp)"; trap 'rm -f "$$raw" "$$clean"' EXIT; \
		uvx --from 'yt-dlp==2026.8.19' yt-dlp --flat-playlist --dump-single-json 'https://www.youtube.com/show/VLPLi60mUelRAbFqfgymVfZttlkIyt0XHZjt?sbp=KgtiR01pUm1YYlJVc0AB' > "$$raw"; \
		uv run python scripts/sanitize_show.py "$$raw" --output "$$clean"; \
		mv "$$clean" youtube_show.json; trap - EXIT

manifest:
	uv run python scripts/build_manifest.py

transcripts:
	uv run python scripts/fetch_transcripts.py youtube_show.json --output data/youtube

corpus:
	uv run python scripts/build_corpus.py

index:
	uv run python scripts/build_index.py --chunks data/chunks.jsonl --model google/embeddinggemma-2 --output index --batch-size 8

test:
	uv run pytest -q tests

check: test
	uv lock --check
	uv run ruff check scripts src tests
	uv run bandit -q -lll -r scripts src
	uv run python -m compileall -q scripts src
	git diff --check

docker-build:
	docker build --tag ai-that-works-search:local .

docker-test:
	docker run --rm ai-that-works-search:local --help
