from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .corpus import iter_jsonl


def format_time(seconds: float | None) -> str:
    if seconds is None:
        return ""
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Semantic search across AI That Works")
    parser.add_argument("query", nargs="+")
    parser.add_argument("--index", type=Path, default=Path("index"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-per-episode", type=int, default=2)
    parser.add_argument("--source-type", action="append")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    metadata = json.loads((args.index / "metadata.json").read_text())
    chunks = list(iter_jsonl(args.index / "chunks.jsonl"))
    embeddings = np.load(args.index / "embeddings.npy")
    model = SentenceTransformer(metadata["model"], trust_remote_code=True)
    query = " ".join(args.query)
    kwargs = {"normalize_embeddings": True}
    if metadata.get("query_prompt"):
        kwargs["prompt_name"] = "query"
    query_vector = np.asarray(model.encode([query], **kwargs)[0], dtype=np.float32)
    scores = embeddings @ query_vector
    order = np.argsort(-scores)
    per_episode: dict[str, int] = defaultdict(int)
    results = []
    for index in order:
        item = chunks[int(index)]
        if args.source_type and item["source_type"] not in args.source_type:
            continue
        episode_key = item.get("episode_guid") or item.get("youtube_id") or item["title"]
        if args.max_per_episode > 0 and per_episode[episode_key] >= args.max_per_episode:
            continue
        stamp = format_time(item.get("start_s"))
        timestamp_url = item.get("youtube_url")
        if timestamp_url and item.get("start_s") is not None:
            timestamp_url = f"{timestamp_url}&t={int(item['start_s'])}s"
        preview = " ".join(item["text"].split())
        results.append({
            "score": float(scores[index]),
            "title": item["title"],
            "episode_guid": item.get("episode_guid"),
            "source_type": item["source_type"],
            "timestamp": stamp or None,
            "url_or_path": timestamp_url or item.get("source_path"),
            "text": item["text"],
            "preview": preview[:500] + ("…" if len(preview) > 500 else ""),
        })
        per_episode[episode_key] += 1
        if len(results) >= args.top_k:
            break
    if args.json:
        print(json.dumps({"query": query, "results": results}, indent=2, ensure_ascii=False))
    else:
        for shown, result in enumerate(results, 1):
            print(f"\n{shown}. {result['score']:.4f} — {result['title']}")
            print(f"   {result['source_type']} {result['timestamp'] or ''} {result['url_or_path']}")
            print(f"   {result['preview']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
