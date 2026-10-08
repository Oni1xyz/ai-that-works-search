# AI That Works Semantic Search

Local semantic search across the **AI That Works** YouTube show and its companion GitHub repository.

The corpus combines:

- YouTube transcripts with timestamps
- repository episode descriptions and notes
- repository transcripts when YouTube captions are unavailable
- diagram/whiteboard asset paths and curated captions
- source URLs and repository provenance on every search result

## Source material

- YouTube show/playlist: `PLi60mUelRAbFqfgymVfZttlkIyt0XHZjt`
- GitHub: <https://github.com/ai-that-works/ai-that-works>
- Repository commit captured in each built index

The cloned upstream repository lives under `source/` and is intentionally ignored by this project's Git history.

## Model choice

The referenced **Gemini Embedding 2** model is available through the Gemini API, not as downloadable Hugging Face weights. For local indexing, this project uses Google's current official open model, [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2), from Hugging Face.

EmbeddingGemma 2 is not Gemini weights: it is a Gemma 4-derived, Apache-2.0 multimodal embedding model released by Google DeepMind. It is public and ungated on Hugging Face, supports 8,192-token inputs, emits 768-dimensional vectors, and supports Matryoshka reductions to 512/256/128 dimensions. This project uses its text path for transcripts and curated diagram captions.

The older `google/embeddinggemma-300m` model is text-only and manually license-gated; it is not used here.

A small MiniLM index is retained only as a fast pipeline smoke test, not the requested final model.

## Setup

```bash
cd /Users/gama/projects/ai-that-works-search
uv sync --python 3.11
```

## Rebuild the corpus

```bash
# Refresh YouTube and repository manifests
uvx yt-dlp --flat-playlist --dump-single-json \
  'https://www.youtube.com/show/VLPLi60mUelRAbFqfgymVfZttlkIyt0XHZjt?sbp=KgtiR01pUm1YYlJVc0AB' \
  > youtube_show.json
uv run python scripts/build_manifest.py

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

The backfill uses `mlx-community/distil-whisper-large-v3`, records the transcription model in every generated transcript, and deletes downloaded audio after successful processing.

## Build the requested index

```bash
uv run aitw-build \
  --chunks data/chunks.jsonl \
  --model google/embeddinggemma-2 \
  --output index \
  --batch-size 16
```

Artifacts:

- `index/embeddings.npy` — normalized float32 vectors
- `index/chunks.jsonl` — cited source chunks in vector order
- `index/metadata.json` — model, dimensions, corpus count, and upstream commit

## Search

```bash
uv run aitw-search \
  'how should coding agents manage long-term memory and context compaction?' \
  --index index --top-k 10
```

Results include episode title, source type, timestamped YouTube URL when available, repository path, similarity score, and a text preview.

## Provenance and limitations

- `data/episodes.json` is the union of repository episodes and every video in the supplied YouTube show.
- Repository episode metadata is preferred for episode numbers and dates.
- YouTube-only videos remain searchable even when there is no repository folder.
- Repo-only/upcoming episodes remain searchable through descriptions and notes.
- YouTube-generated captions are preferred. Missing captions fall back to repository transcripts, then local Whisper transcription.
- Whiteboard captions are curated summaries. Uncaptioned images remain indexed by filename/path but are not treated as fully understood visual content.
- Gemini Embedding 2 is multimodal; EmbeddingGemma is text-only. Diagram pixels are therefore represented through captions, not native image embeddings.
