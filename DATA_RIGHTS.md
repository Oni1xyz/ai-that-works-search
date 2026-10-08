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

Tracked manifests contain a minimized show snapshot (IDs, titles, URLs, and durations), repository paths, factual metadata, and AI-assisted, unverified diagram descriptions so users can reproduce the corpus locally. Third-party episode descriptions are excluded by default.

## User responsibility

Users are responsible for complying with:

- [YouTube's Terms of Service](https://www.youtube.com/static?template=terms) and applicable caption/transcript rights;
- the upstream repository's license or absence of a license;
- model-specific licenses and acceptable-use policies; and
- laws applicable to text-and-data mining and redistribution in their jurisdiction.

The upstream [`ai-that-works/ai-that-works`](https://github.com/ai-that-works/ai-that-works) repository did not expose a repository license when this snapshot was inspected. Do not assume its content is open-licensed merely because it is publicly accessible.

## Publishing derived embeddings

Do not publish the embeddings or their paired chunk text by default. Before publishing:

1. obtain explicit permission from the AI That Works content owners;
2. determine whether YouTube-generated captions may be redistributed;
3. decide whether raw chunk text will be included, separately downloadable, or reconstructed locally;
4. document the exact upstream commit, YouTube snapshot date, chunking code commit, model ID/revision, dimensions, dtype, and normalization;
5. publish a model/dataset card with intended use, known omissions, transcription quality, takedown contact, and licensing scope; and
6. consider a private or gated Hugging Face dataset until permissions are settled.

Vectors without chunk text reduce copyright exposure but are substantially less useful for RAG, and embeddings can still encode information about source material. Treat them as derived content, not rights-free numbers.

## Takedowns and corrections

If a rights holder requests removal or correction of a source mapping, caption, or derived artifact, remove it from future builds and document the affected source IDs. Public artifact hosts should include a clear takedown contact before release.
