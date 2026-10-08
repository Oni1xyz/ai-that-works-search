# Contributing

Thanks for helping improve AI That Works Search.

## Before opening a change

1. Read `README.md`, `DATA_RIGHTS.md`, and `SECURITY.md`.
2. Open an issue for substantial changes to chunking, source mapping, model choice, or artifact publication.
3. Do not commit third-party transcripts, audio/video, cloned source repositories, model weights, or generated indexes.

## Development setup

```bash
git clone <your-fork-url> ai-that-works-search
cd ai-that-works-search
uv sync --locked --group dev --python 3.11
make test
```

The repository is an executable research project, not a PyPI package. Run commands through the checked-in scripts:

```bash
uv run python scripts/search.py --help
uv run python scripts/rag.py --help
uv run python scripts/build_index.py --help
```

## Source refresh workflow

```bash
make sources
make manifest
make transcripts
make corpus
```

`make sources` refuses to update a dirty upstream clone. YouTube caption retrieval is resumable and may be rate-limited. Do not bypass access controls or use personal browser cookies in automation.

The Apple-Silicon Whisper fallback is optional and local-only:

```bash
uv run python scripts/backfill_transcripts.py \
  --show youtube_show.json \
  --output data/youtube \
  --audio-dir data/audio-cache
```

## Tests and quality gates

Run before every pull request:

```bash
make check
```

For Docker changes:

```bash
make docker-build
make docker-test
```

For retrieval/index changes, build a smoke index and manually inspect representative searches for memory/context, adversarial review, and agent observability. Similarity scores alone are not sufficient evidence.

## Pull requests

Keep pull requests focused. Include:

- problem and motivation;
- files and behavior changed;
- exact test commands and results;
- whether chunking/index compatibility changed;
- generated corpus/index counts when applicable;
- new network calls, model downloads, or data-rights implications; and
- screenshots or example query results when useful.

## Mapping and caption standards

- Repository metadata is authoritative for episode IDs and exact linked videos.
- Do not guess missing episode-folder mappings.
- Any inferred mapping must be documented in `scripts/build_manifest.py` with evidence and confidence.
- Diagram captions must describe visible content and flag illegible or uncertain material.
- Locally transcribed content must record the transcription model.

## Security and privacy

Never commit credentials or `.env` files. Do not enable remote model code by default. Remote RAG synthesis must remain explicitly opt-in through `--generate`.

Report suspected vulnerabilities using the process in `SECURITY.md`, not a public issue.
