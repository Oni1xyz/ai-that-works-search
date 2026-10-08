# Publishing Embeddings on Hugging Face

## Recommendation

Technically, yes: Hugging Face is a good distribution channel for a versioned embedding corpus. Operationally, publish only after receiving explicit permission from the AI That Works content owners and documenting the transcript rights.

Use a **[dataset repository](https://huggingface.co/docs/hub/en/datasets-adding)**, not a model repository. The vectors are outputs of `google/embeddinggemma-2`; they are not a new embedding model.

## Recommended dataset layout

```text
README.md                 dataset card
metadata.json             corpus/model/build provenance
corpus_provenance.json    checksums of corpus inputs and chunker code
chunks.parquet            optional text + source metadata
embeddings-768.parquet    chunk_id + fixed-size vector
checksums.sha256
checksums.sha256.sig      optional maintainer signature
LICENSE or LICENSES/
```

Preferred row fields:

- `chunk_id`
- `episode_guid`, season, episode, title, date
- `youtube_id`, timestamp start/end, source URL
- repository commit/path
- source type
- chunk hash
- transcript provenance (`youtube_generated`, `repository`, `local_whisper`)
- transcription model when applicable
- embedding model ID and revision
- vector dimensions, dtype, normalization
- chunker version/commit

## Artifact options

### Option A — code only; users build locally

Lowest rights risk and strongest reproducibility. Publish no vectors or third-party text. This is the recommended initial public release.

### Option B — vectors plus metadata, no raw text

Smaller rights surface, but RAG users still need to reconstruct the same chunks locally. Embeddings may retain information about source content and should still be treated as derived data.

### Option C — vectors plus chunk text

Best user experience and easiest RAG setup. Highest redistribution risk. Use only with explicit content-owner permission and a clear dataset license/takedown process.

### Option D — private or gated dataset

Useful for collaboration while permissions and quality review are pending. This is the recommended intermediate step if sharing is needed before a public release.

## Dataset card checklist

Document:

- intended use and prohibited use;
- source URLs and collection date;
- upstream repository commit;
- exact video count and mapping exceptions;
- transcript acquisition/fallback process;
- known automatic-caption and Whisper errors;
- diagram-caption methodology;
- chunk size/overlap/deduplication;
- [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2) revision and its declared Apache-2.0 model license;
- vector dimension/dtype/normalization;
- content licensing and permission statement;
- privacy review;
- takedown/correction contact; and
- checksums, an independently verifiable signature or release digest, and reproducibility commands. Embedded checksums detect corruption but are not authenticity proof when distributed beside the files they cover.

## Versioning

Treat the artifact as an immutable snapshot. Tag releases using the corpus date and index format, for example:

```text
2026-10-08-eg2-768-v1
```

A new upstream video, transcript correction, diagram caption, chunking change, model revision, or dimension change should produce a new dataset revision rather than silently replacing vectors.
