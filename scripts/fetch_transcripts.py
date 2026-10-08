#!/usr/bin/env python3
"""Fetch every transcript from a yt-dlp show/playlist manifest, resumably."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from _shared import require_youtube_video_id
from youtube_transcript_api import YouTubeTranscriptApi


def timestamp(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/youtube"))
    parser.add_argument("--sleep", type=float, default=0.35)
    args = parser.parse_args()

    show = json.loads(args.manifest.read_text())
    entries = show.get("entries", [])
    args.output.mkdir(parents=True, exist_ok=True)
    api = YouTubeTranscriptApi()
    status = []

    for order, entry in enumerate(entries, 1):
        video_id = require_youtube_video_id(entry["id"])
        path = args.output / f"{video_id}.json"
        text_path = args.output / f"{video_id}.txt"
        if path.exists() and text_path.exists():
            try:
                saved = json.loads(path.read_text())
                if not isinstance(saved.get("segments"), list):
                    raise ValueError("cached transcript has no segment list")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                print(f"[{order}/{len(entries)}] invalid cache {video_id}: {exc}; refetching", flush=True)
            else:
                status.append(
                    {
                        "order": order,
                        "video_id": video_id,
                        "title": entry.get("title"),
                        "status": "cached",
                        "segments": len(saved["segments"]),
                    }
                )
                print(f"[{order}/{len(entries)}] cached {video_id}", flush=True)
                continue
        try:
            fetched = api.fetch(video_id, languages=["en"])
            segments = [
                {"start": float(item.start), "duration": float(item.duration), "text": item.text} for item in fetched
            ]
            document = {
                "order": order,
                "video_id": video_id,
                "title": entry.get("title"),
                "url": entry.get("url") or f"https://www.youtube.com/watch?v={video_id}",
                "language": getattr(fetched, "language_code", "en"),
                "is_generated": getattr(fetched, "is_generated", None),
                "retrieved_at": datetime.now(UTC).isoformat(),
                "segments": segments,
            }
            path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
            text_path.write_text("\n".join(f"[{timestamp(item['start'])}] {item['text']}" for item in segments) + "\n")
            status.append(
                {
                    "order": order,
                    "video_id": video_id,
                    "title": entry.get("title"),
                    "status": "ok",
                    "segments": len(segments),
                }
            )
            print(f"[{order}/{len(entries)}] ok {video_id}: {len(segments)} segments", flush=True)
        except Exception as exc:
            status.append(
                {
                    "order": order,
                    "video_id": video_id,
                    "title": entry.get("title"),
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[{order}/{len(entries)}] ERROR {video_id}: {exc}", flush=True)
        (args.output / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
        time.sleep(args.sleep)

    ok = sum(item["status"] in {"ok", "cached"} for item in status)
    print(f"complete: {ok}/{len(entries)} transcripts", flush=True)
    return 0 if ok == len(entries) else 2


if __name__ == "__main__":
    raise SystemExit(main())
