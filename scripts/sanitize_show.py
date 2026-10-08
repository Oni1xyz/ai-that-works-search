#!/usr/bin/env python3
"""Reduce yt-dlp output to the metadata required by this repository."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _shared import require_youtube_video_id

ENTRY_FIELDS = ("title", "url", "duration", "timestamp", "release_timestamp", "live_status")


def sanitize(raw: dict) -> dict:
    entries = []
    for entry in raw.get("entries", []):
        video_id = require_youtube_video_id(entry["id"])
        clean = {"id": video_id}
        clean.update({field: entry.get(field) for field in ENTRY_FIELDS if entry.get(field) is not None})
        clean["url"] = f"https://www.youtube.com/watch?v={video_id}"
        entries.append(clean)
    return {
        "id": raw.get("id"),
        "title": raw.get("title"),
        "webpage_url": raw.get("webpage_url") or raw.get("original_url"),
        "entries": entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw = json.loads(args.input.read_text())
    document = sanitize(raw)
    args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    print(f"sanitized {len(document['entries'])} show entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
