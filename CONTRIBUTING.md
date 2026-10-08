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

Before you run `make sources`, make sure that the upstream clone has no uncommitted changes. The command does not update a dirty clone. YouTube caption retrieval can resume after an interruption. YouTube can limit the request rate. Do not bypass access controls. Do not use personal browser cookies for automation.

```bash
make sources
make manifest
make transcripts
make corpus
```

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

If you change retrieval or indexing, build a smoke index. Manually inspect representative searches. Include searches about memory and context, adversarial review, and agent observability. Do not use similarity scores as the only evidence.

## Pull requests

Keep pull requests focused. Include this information:

- Describe the problem and motivation.
- Identify the changed files and behavior.
- Report the exact test commands and results.
- State whether chunking or index compatibility changed.
- Provide corpus and index counts when applicable.
- Identify new network calls, model downloads, and data-rights implications.
- Add screenshots or example query results when useful.

## Mapping and caption standards

- Repository metadata is authoritative for episode IDs and exact linked videos.
- Do not guess missing episode-folder mappings.
- If you infer a mapping, document the mapping, evidence, and confidence in `scripts/build_manifest.py`.
- Diagram captions must describe visible content and flag illegible or uncertain material.
- Locally transcribed content must record the transcription model.

## Security and privacy

Never commit credentials or `.env` files. Do not enable remote model code by default. Remote RAG synthesis must remain explicitly opt-in through `--generate`.

Report suspected vulnerabilities using the process in `SECURITY.md`, not a public issue.
