#!/usr/bin/env python3
"""Assemble searchable chunks from YouTube transcripts and repository episode assets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aitw_search.corpus import chunk_id, chunk_text, transcript_chunks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/episodes.json"))
    parser.add_argument("--repo", type=Path, default=Path("source/ai-that-works"))
    parser.add_argument("--youtube", type=Path, default=Path("data/youtube"))
    parser.add_argument("--diagram-captions", type=Path, default=Path("data/diagram_captions.json"))
    parser.add_argument("--output", type=Path, default=Path("data/chunks.jsonl"))
    args = parser.parse_args()

    episodes = json.loads(args.manifest.read_text())["episodes"]
    captions = json.loads(args.diagram_captions.read_text()) if args.diagram_captions.exists() else {}
    chunks: list[dict] = []

    def add(episode: dict, source_type: str, source_path: str, text: str, **extra: object) -> None:
        metadata = {
            "episode_guid": episode.get("guid"),
            "season": episode.get("season"),
            "episode": episode.get("episode"),
            "title": episode.get("title"),
            "event_date": episode.get("event_date"),
            "youtube_id": episode.get("youtube_id"),
            "youtube_url": episode.get("youtube_url"),
            "repository_folder": episode.get("repository_folder"),
            "source_type": source_type,
            "source_path": source_path,
            **extra,
        }
        clean = text.strip()
        if not clean:
            return
        chunks.append({"id": chunk_id(metadata, clean), "text": clean, **metadata})

    for episode in episodes:
        description = episode.get("description", "").strip()
        if description:
            add(episode, "episode_description", "data/episodes.json", description)

        transcript_found = False
        video_id = episode.get("youtube_id")
        youtube_json = args.youtube / f"{video_id}.json" if video_id else None
        if youtube_json and youtube_json.exists():
            document = json.loads(youtube_json.read_text())
            for index, item in enumerate(transcript_chunks(document)):
                add(
                    episode,
                    "youtube_transcript",
                    str(youtube_json),
                    item["text"],
                    chunk_index=index,
                    start_s=item["start_s"],
                    end_s=item["end_s"],
                )
            transcript_found = True

        # Repository transcript is a fallback to prevent duplicate search hits.
        if not transcript_found:
            for relative in episode.get("repo_transcripts", []):
                path = args.repo / relative
                if path.exists():
                    for index, text in enumerate(chunk_text(path.read_text(errors="replace"))):
                        add(episode, "repository_transcript", relative, text, chunk_index=index)

        for relative in episode.get("repo_notes", []):
            path = args.repo / relative
            if not path.exists() or path.stat().st_size > 2_000_000:
                continue
            for index, text in enumerate(chunk_text(path.read_text(errors="replace"))):
                add(episode, "repository_note", relative, text, chunk_index=index)

        for relative in episode.get("repo_images", []):
            caption = captions.get(relative)
            if caption:
                add(episode, "diagram_caption", relative, caption)
            else:
                # Preserve discoverability and provenance even when no OCR/caption exists.
                add(episode, "repository_image", relative, f"Repository image asset: {Path(relative).name}")

    # Deduplicate exact IDs while preserving deterministic order.
    unique = {item["id"]: item for item in chunks}
    ordered = sorted(unique.values(), key=lambda item: (item.get("event_date") or "", item.get("episode_guid") or "", item["source_type"], item.get("chunk_index", 0)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for item in ordered:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for item in ordered:
        counts[item["source_type"]] = counts.get(item["source_type"], 0) + 1
    print(json.dumps({"chunks": len(ordered), "episodes": len({item['episode_guid'] for item in ordered}), "sources": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
