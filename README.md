# AI That Works Semantic Search

Local semantic search across the **AI That Works** YouTube show and its companion GitHub repository.

This repository contains a research tool. It is not a published Python package.

1. Clone the repository.
2. Use the checked-in `uv` environment.
3. Run the scripts directly.

The corpus combines:

- YouTube transcripts with timestamps
- repository notes with commit-pinned provenance
- repository transcripts when YouTube captions are unavailable
- diagram/whiteboard asset paths and AI-assisted, unverified captions
- source URLs and repository provenance on every search result

## Source material

- YouTube show/playlist: `PLi60mUelRAbFqfgymVfZttlkIyt0XHZjt`
- GitHub: <https://github.com/ai-that-works/ai-that-works>
- Repository commit captured in each built index

The cloned upstream repository lives under `source/` and is intentionally ignored by this project's Git history.

## Model choice

The referenced **Gemini Embedding 2** model is available through the Gemini API, not as downloadable Hugging Face weights. For local indexing, this project uses Google's current official open model, [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2), from Hugging Face.

EmbeddingGemma 2 does not contain Gemini weights. It is a Gemma 4-derived, Apache-2.0 multimodal embedding model from Google DeepMind. Hugging Face provides the model without an access gate. The model accepts inputs of up to 8,192 tokens. It produces 768-dimensional vectors and can reduce them to 512, 256, or 128 dimensions. This project uses the text path for transcripts and AI-assisted diagram captions.

The older `google/embeddinggemma-300m` model is text-only and manually license-gated; it is not used here.

A small MiniLM index can be built as a fast pipeline smoke test, but it is not the intended production model.

## Setup

```bash
git clone <your-fork-or-repository-url> ai-that-works-search
cd ai-that-works-search
uv sync --locked --python 3.11
```

## Docker quick start

The image contains the code and locked runtime dependencies, but intentionally excludes source clones, transcripts, indexes, model weights, and credentials.

```bash
docker build -t ai-that-works-search:local .
docker run --rm ai-that-works-search:local --help
```

Run retrieval against a host-built index:

```bash
mkdir -p index .cache/huggingface
HOST_UID="$(id -u)" HOST_GID="$(id -g)" docker compose run --rm rag \
  'what does BAML actually solve?' --index index --top-k 8
```

Build an index in the container after collecting `source/` and `data/chunks.jsonl` on the host:

```bash
mkdir -p index .cache/huggingface
HOST_UID="$(id -u)" HOST_GID="$(id -g)" docker compose run --rm build-index
```

On Linux, create the bind-mounted directories as the host user. Pass the host UID and GID. These steps make index files portable without Docker Desktop permission mapping. The first index build downloads model weights into the ignored `.cache/huggingface` directory. Docker uses CPU PyTorch wheels and can be slower than Apple Silicon or a CUDA host. The Linux image does not include the Apple-Silicon MLX transcript fallback.

If you want remote answer synthesis:

1. Store credentials outside the repository.
2. Pass credentials only when you run generation.

Ordinary retrieval and `--prompt-only` containers do not receive an API key:

```bash
HOST_UID="$(id -u)" HOST_GID="$(id -g)" docker compose run --rm \
  -e OPENAI_API_KEY -e OPENAI_BASE_URL -e AITW_RAG_MODEL \
  rag 'Compare harness engineering and context engineering' --index index --generate
```


## Rebuild the corpus

```bash
# Refresh the upstream clone and YouTube show snapshot atomically, then map sources.
make sources
make manifest

# Retrieve public captions. The script is resumable.
uv run python scripts/fetch_transcripts.py youtube_show.json --output data/youtube

# If YouTube blocks caption requests, backfill missing entries from audio locally on Apple Silicon.
uv run python scripts/backfill_transcripts.py \
  --show youtube_show.json \
  --output data/youtube \
  --audio-dir data/audio-cache

# Assemble transcript, note, and diagram chunks.
uv run python scripts/build_corpus.py
```

`make sources` checks out the upstream commit that `UPSTREAM_REF` specifies in the `Makefile`.

To inspect a newer upstream revision:

1. Pass the revision explicitly. For example, use `make sources UPSTREAM_REF=origin/main`.
2. Review the revision.
3. Update the pin.
4. Commit the regenerated manifests.

The checked-in show snapshot and generated manifest deliberately exclude third-party episode descriptions. Corpus construction supports only source types with immutable, validated citation provenance.

The backfill uses `mlx-community/distil-whisper-large-v3`, records the transcription model in every generated transcript, and deletes downloaded audio after successful processing.

Corpus generation requires a clean Git checkout for repository-backed sources. It writes `data/chunks.provenance.json`. The `data/chunks.provenance.json` file binds each chunk to the repository commit. It also records checksums for the manifest, YouTube snapshot, captions, chunker code, and transcript inputs. Index construction rejects missing, stale, dirty, or mismatched provenance.

## Build the requested index

```bash
uv run python scripts/build_index.py \
  --chunks data/chunks.jsonl \
  --model google/embeddinggemma-2 \
  --output index \
  --batch-size 16
```

Artifacts:

- `index/CURRENT` — atomically replaced pointer to the active immutable version
- `index/versions/<build-id>/embeddings.npy` — normalized float32 vectors
- `index/versions/<build-id>/chunks.jsonl` — cited source chunks in vector order
- `index/versions/<build-id>/corpus_provenance.json` — copied, checksum-bound corpus provenance
- `index/versions/<build-id>/metadata.json` — model ID/revision, dimensions, dtype, corpus count, checksums, and upstream commit

Publishing retains older immutable versions and switches readers with one atomic pointer replacement, so a failed build cannot remove the prior active index. When online, the build resolves the requested Hugging Face model reference to an immutable commit and subsequent searches reuse that exact revision. Offline `--model-revision` values must be full 40-character commit SHAs; mutable branches and tags are rejected.

By default, query commands allow only the documented EmbeddingGemma 2 and MiniLM smoke-test IDs. If an index names an unlisted model, first review the model ID and pinned revision. Then use `--allow-unlisted-model`. The application keeps remote model code disabled.

## Search

```bash
uv run python scripts/search.py \
  'how should coding agents manage long-term memory and context compaction?' \
  --index index --top-k 10
```

Results include episode title, source type, timestamped YouTube URL when available, repository path, similarity score, and a text preview.

## Retrieval-augmented generation (RAG)

Retrieve a diversified context pack from the embedding index:

```bash
uv run python scripts/rag.py \
  'What patterns do the hosts recommend for agent memory and context compaction?' \
  --index index --top-k 8
```

Emit a citation-constrained prompt for Hermes or another model:

```bash
uv run python scripts/rag.py \
  'What patterns do the hosts recommend for agent memory and context compaction?' \
  --index index --top-k 8 --prompt-only
```

Optionally synthesize through any OpenAI-compatible chat-completions endpoint:

```bash
export OPENAI_API_KEY='set this outside the repository'
export OPENAI_BASE_URL='https://api.openai.com/v1'  # or a trusted local endpoint
export AITW_RAG_MODEL='your-generation-model'
uv run python scripts/rag.py 'Compare harness engineering and context engineering' --index index --generate
```

The RAG prompt requires numbered inline citations. It prohibits claims outside the retrieved context. The application validates each generated citation number. It appends trusted YouTube links with timestamps or repository URLs with pinned commits. It does not trust source lists that the model writes. By default, retrieval limits the number of results from each episode.

The application emits generated answers as inert terminal text. It does not emit trusted Markdown. The application replaces Markdown metacharacters in model output and source titles with similar Unicode characters. This replacement keeps the text readable. It also prevents a Markdown renderer from activating links, images, or HTML.

## Provenance and limitations

- `data/episodes.json` is the union of repository episodes and every video in the supplied YouTube show.
- Repository episode metadata is preferred for episode numbers and dates.
- YouTube-only videos remain searchable even when there is no repository folder.
- Repo-only/upcoming episodes remain searchable through commit-pinned repository notes.
- YouTube captions are preferred. Missing captions fall back to repository transcripts, then local Whisper transcription.
- Whiteboard captions are AI-assisted summaries whose per-caption human review was not recorded. Treat them as unverified leads, not factual substitutes for the underlying images.
- Both Gemini Embedding 2 and EmbeddingGemma 2 are multimodal, but this repository currently uses only EmbeddingGemma 2's text path. Diagram pixels are represented through AI-assisted captions rather than native image vectors.

## Project policy

- Original code is MIT licensed; see `LICENSE`.
- Third-party source/model notices are listed in `THIRD_PARTY_NOTICES.md`.
- Third-party data and derived-artifact boundaries are documented in `DATA_RIGHTS.md`.
- Contribution workflow is documented in `CONTRIBUTING.md`.
- Vulnerability reporting and remote-generation warnings are in `SECURITY.md`.
- Hugging Face publication guidance is in `docs/HUGGING_FACE.md`.
