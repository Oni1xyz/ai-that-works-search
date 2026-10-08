from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import httpx
import numpy as np
from sentence_transformers import SentenceTransformer

from .corpus import iter_jsonl


def retrieve(query: str, index_dir: Path, top_k: int, max_per_episode: int) -> tuple[list[dict], dict]:
    metadata = json.loads((index_dir / "metadata.json").read_text())
    chunks = list(iter_jsonl(index_dir / "chunks.jsonl"))
    vectors = np.load(index_dir / "embeddings.npy")
    model = SentenceTransformer(metadata["model"], trust_remote_code=True)
    kwargs = {"normalize_embeddings": True}
    if metadata.get("query_prompt"):
        kwargs["prompt_name"] = "query"
    query_vector = np.asarray(model.encode([query], **kwargs)[0], dtype=np.float32)
    scores = vectors @ query_vector
    counts: dict[str, int] = defaultdict(int)
    results = []
    for vector_index in np.argsort(-scores):
        item = chunks[int(vector_index)]
        episode_key = item.get("episode_guid") or item.get("youtube_id") or item["title"]
        if max_per_episode > 0 and counts[episode_key] >= max_per_episode:
            continue
        source = item.get("youtube_url") or item.get("source_path")
        if source and item.get("start_s") is not None and source.startswith("http"):
            source = f"{source}&t={int(item['start_s'])}s"
        results.append({
            "rank": len(results) + 1,
            "score": float(scores[vector_index]),
            "title": item["title"],
            "episode_guid": item.get("episode_guid"),
            "source_type": item["source_type"],
            "source": source,
            "start_s": item.get("start_s"),
            "end_s": item.get("end_s"),
            "text": item["text"],
        })
        counts[episode_key] += 1
        if len(results) >= top_k:
            break
    return results, metadata


def build_prompt(query: str, results: list[dict], max_context_chars: int) -> str:
    sections = []
    used = 0
    for result in results:
        header = f"[{result['rank']}] {result['title']} | {result['source_type']} | {result['source']}"
        block = f"{header}\n{result['text'].strip()}"
        if sections and used + len(block) > max_context_chars:
            break
        sections.append(block)
        used += len(block)
    context = "\n\n---\n\n".join(sections)
    return f"""You answer questions about the AI That Works corpus.
Use only the retrieved context below. Cite factual claims inline with source numbers like [1].
If the context is insufficient or conflicting, say so explicitly. Do not invent episode claims.
End with a compact Sources list containing the cited title and URL/path.

QUESTION
{query}

RETRIEVED CONTEXT
{context}

ANSWER
"""


def generate_answer(prompt: str, model: str, base_url: str, api_key: str) -> str:
    response = httpx.post(
        f"{base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieve and optionally answer against the AI That Works index")
    parser.add_argument("query", nargs="+")
    parser.add_argument("--index", type=Path, default=Path("index"))
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-per-episode", type=int, default=2)
    parser.add_argument("--max-context-chars", type=int, default=24000)
    parser.add_argument("--prompt-only", action="store_true", help="print the grounded generation prompt")
    parser.add_argument("--generate", action="store_true", help="call an OpenAI-compatible chat-completions API")
    parser.add_argument("--generator-model", default=os.getenv("AITW_RAG_MODEL"))
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    query = " ".join(args.query)
    results, metadata = retrieve(query, args.index, args.top_k, args.max_per_episode)
    prompt = build_prompt(query, results, args.max_context_chars)
    answer = None
    if args.generate:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or not args.generator_model:
            raise SystemExit("--generate requires OPENAI_API_KEY and --generator-model (or AITW_RAG_MODEL)")
        answer = generate_answer(prompt, args.generator_model, args.base_url, api_key)

    if args.json:
        print(json.dumps({"query": query, "index": metadata, "results": results, "prompt": prompt if args.prompt_only else None, "answer": answer}, indent=2, ensure_ascii=False))
    elif answer:
        print(answer)
    elif args.prompt_only:
        print(prompt)
    else:
        print(f"Retrieved {len(results)} chunks with {metadata['model']}:\n")
        for item in results:
            print(f"[{item['rank']}] {item['score']:.4f} — {item['title']}")
            print(f"    {item['source_type']} — {item['source']}")
            print(f"    {' '.join(item['text'].split())[:420]}\n")
        print("Use --prompt-only to emit a grounded synthesis prompt, or --generate with an OpenAI-compatible endpoint.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
