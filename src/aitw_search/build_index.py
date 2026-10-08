from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .corpus import iter_jsonl


def encode(model: SentenceTransformer, texts: list[str], batch_size: int, kind: str) -> np.ndarray:
    kwargs = {"batch_size": batch_size, "show_progress_bar": True, "normalize_embeddings": True}
    prompts = getattr(model, "prompts", {}) or {}
    if kind in prompts:
        kwargs["prompt_name"] = kind
    return np.asarray(model.encode(texts, **kwargs), dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed the AI That Works corpus")
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks.jsonl"))
    parser.add_argument("--model", default="google/embeddinggemma-300m")
    parser.add_argument("--output", type=Path, default=Path("index"))
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    chunks = list(iter_jsonl(args.chunks))
    if not chunks:
        raise SystemExit("no chunks found")
    model = SentenceTransformer(args.model, trust_remote_code=True)
    vectors = encode(model, [item["text"] for item in chunks], args.batch_size, "document")
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "embeddings.npy", vectors)
    with (args.output / "chunks.jsonl").open("w") as handle:
        for item in chunks:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    try:
        commit = subprocess.check_output(["git", "-C", "source/ai-that-works", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = None
    metadata = {
        "model": args.model,
        "dimensions": int(vectors.shape[1]),
        "chunks": len(chunks),
        "normalized": True,
        "query_prompt": bool("query" in (getattr(model, "prompts", {}) or {})),
        "document_prompt": bool("document" in (getattr(model, "prompts", {}) or {})),
        "repository_commit": commit,
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
