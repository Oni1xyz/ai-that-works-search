# Data Rights and Provenance

The repository's MIT license applies only to the original source code and documentation created for this project.

It does **not** grant rights to redistribute or relicense third-party material, including:

- AI That Works videos, audio, transcripts, episode descriptions, notes, code, or diagrams;
- YouTube captions or locally generated Whisper transcripts;
- files from `ai-that-works/ai-that-works`;
- Google or Hugging Face model weights; or
- embedding/index artifacts derived from third-party content.

## Default repository policy

The following are intentionally excluded from Git:

- `source/` — upstream repository clones
- `data/youtube/` — fetched/generated transcript content
- `data/audio-cache/` — temporary audio
- `data/chunks.jsonl` — derived text chunks
- `index*/` — vectors and chunk maps
- `models/` and model caches

Tracked manifests contain a minimized show snapshot. The snapshot includes IDs, titles, URLs, and durations. The manifests also contain repository paths, factual metadata, and unverified AI-assisted diagram descriptions. This information lets users reproduce the corpus locally. By default, the manifests exclude third-party episode descriptions.

## User responsibility

Users are responsible for complying with:

- [YouTube's Terms of Service](https://www.youtube.com/static?template=terms) and applicable caption/transcript rights;
- the upstream repository's license or absence of a license;
- model-specific licenses and acceptable-use policies; and
- laws applicable to text-and-data mining and redistribution in their jurisdiction.

The upstream [`ai-that-works/ai-that-works`](https://github.com/ai-that-works/ai-that-works) repository did not show a license when this snapshot was inspected. Public access does not mean that the content has an open license.

## Publishing derived embeddings

Do not publish the embeddings or their paired chunk text by default. Before publishing:

1. obtain explicit permission from the AI That Works content owners;
2. determine whether YouTube-generated captions may be redistributed;
3. decide whether raw chunk text will be included, separately downloadable, or reconstructed locally;
4. document the exact upstream commit, YouTube snapshot date, chunking code commit, model ID/revision, dimensions, dtype, and normalization;
5. publish a model/dataset card with intended use, known omissions, transcription quality, takedown contact, and licensing scope; and
6. consider a private or gated Hugging Face dataset until permissions are settled.

Vectors without chunk text can reduce copyright exposure. However, these vectors are less useful for RAG. Embeddings can still encode information about source material. Treat embeddings as derived content, not rights-free numbers.

## Takedowns and corrections

If a rights holder requests removal or correction, remove the affected source mapping, caption, or derived artifact from future builds. Document the affected source IDs. Before release, artifact publishers must provide a clear takedown contact.
