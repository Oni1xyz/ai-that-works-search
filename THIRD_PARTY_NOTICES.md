# Third-Party Notices

The MIT license in `LICENSE` applies only to original code and documentation in this repository. It does not relicense the materials listed below.

## AI That Works

- Source repository: https://github.com/ai-that-works/ai-that-works
- Pinned source revision: `3274d8b09e99de6df5fa251b159b8fcfafa72b66`
- YouTube show: https://www.youtube.com/playlist?list=PLi60mUelRAbFqfgymVfZttlkIyt0XHZjt

Episode descriptions, notes, transcripts, diagrams, video, audio, and other upstream materials remain subject to their owners' rights and applicable platform terms. The upstream GitHub repository did not expose a repository license when checked for this release preparation. Generated manifests are factual source maps; fetched transcripts, source clones, chunks, indexes, and media are excluded from Git.

See `DATA_RIGHTS.md` for the project's collection, redistribution, and takedown policy.

## YouTube

Caption and media access is subject to the YouTube Terms of Service:
https://www.youtube.com/static?template=terms

The local scripts are intended for user-initiated reproducible research. They do not grant redistribution rights.

## EmbeddingGemma 2

- Model: https://huggingface.co/google/embeddinggemma-2
- Declared model license: Apache-2.0

Model weights are downloaded from Hugging Face at runtime and are not distributed in this repository or its Docker image.

## Other dependencies

Python dependencies and their exact resolved versions are recorded in `uv.lock`. Each dependency remains under its own license. The Docker image starts from the Python slim image and copies the `uv` binary from Astral's published container image; those components retain their respective licenses.
