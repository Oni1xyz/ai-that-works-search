#!/usr/bin/env python3
"""Build a union manifest from the YouTube show and repository episode metadata."""
from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

VIDEO_RE = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})")
TEXT_SUFFIXES = {".md", ".txt"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}
VIDEO_OVERRIDES = {
    # Repository episode 59 lacks a YouTube link; this show entry is the exact No Vibes
    # performance-engineering episode and is otherwise unmapped.
    "aitw-059": "mm6n4n09RaU",
}


def video_id(url: str | None) -> str | None:
    if not url:
        return None
    match = VIDEO_RE.search(url)
    return match.group(1) if match else None


def normalize_title(title: str) -> str:
    title = re.sub(r"(?i)\b(s\d+e\d+|episode|ep|ai that works|no vibes allowed)\b", " ", title)
    title = re.sub(r"[^a-z0-9]+", " ", title.lower())
    return " ".join(title.split())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("source/ai-that-works"))
    parser.add_argument("--youtube", type=Path, default=Path("youtube_show.json"))
    parser.add_argument("--output", type=Path, default=Path("data/episodes.json"))
    args = parser.parse_args()

    repository = json.loads((args.repo / "data.json").read_text())["episodes"]
    show_entries = json.loads(args.youtube.read_text()).get("entries", [])
    show = {entry["id"]: entry for entry in show_entries}
    manifest = []
    mapped_ids: set[str] = set()

    for episode in repository:
        if episode.get("event_type") != "episode" or episode.get("isWorkshop"):
            continue
        links = episode.get("links") or {}
        media = episode.get("media") or {}
        vid = VIDEO_OVERRIDES.get(episode.get("guid")) or video_id(links.get("youtube")) or video_id(media.get("url"))
        mapping_note = "manual title/topic match" if episode.get("guid") in VIDEO_OVERRIDES else "repository metadata"
        if vid is None and episode.get("isPast"):
            target = normalize_title(episode["title"])
            candidates = sorted(
                (
                    (SequenceMatcher(None, target, normalize_title(item.get("title", ""))).ratio(), item)
                    for item in show_entries
                    if item["id"] not in mapped_ids
                ),
                reverse=True,
                key=lambda pair: pair[0],
            )
            if candidates and candidates[0][0] >= 0.62:
                vid = candidates[0][1]["id"]
                mapping_note = f"fuzzy title match ({candidates[0][0]:.3f})"
        folder_name = episode.get("folder")
        folder = args.repo / folder_name if folder_name else None
        files = [path for path in folder.rglob("*") if path.is_file()] if folder and folder.exists() else []
        relative_files = [str(path.relative_to(args.repo)) for path in files]
        transcript_files = [path for path in relative_files if "transcript" in Path(path).name.lower() or "trasncript" in Path(path).name.lower()]
        note_files = [
            path for path in relative_files
            if Path(path).suffix.lower() in TEXT_SUFFIXES
            and path not in transcript_files
            and (Path(path).parent == Path(folder_name) or "/thoughts/" in f"/{path}")
        ]
        image_files = [path for path in relative_files if Path(path).suffix.lower() in IMAGE_SUFFIXES]
        item = {
            "guid": episode.get("guid"),
            "season": episode.get("season"),
            "episode": episode.get("episode"),
            "title": episode.get("title"),
            "description": episode.get("description", "").strip(),
            "event_date": episode.get("eventDate"),
            "is_past": episode.get("isPast"),
            "repository_folder": folder_name,
            "repository_url": links.get("code"),
            "youtube_id": vid,
            "youtube_url": f"https://www.youtube.com/watch?v={vid}" if vid else None,
            "in_youtube_show": bool(vid and vid in show),
            "mapping_note": mapping_note,
            "repo_transcripts": sorted(transcript_files),
            "repo_notes": sorted(note_files),
            "repo_images": sorted(image_files),
        }
        manifest.append(item)
        if vid:
            mapped_ids.add(vid)

    # Preserve every video from the user's show URL, even when the repository has no matching entry.
    for order, entry in enumerate(show_entries, 1):
        if entry["id"] in mapped_ids:
            continue
        manifest.append({
            "guid": f"youtube-{entry['id']}",
            "season": None,
            "episode": None,
            "title": entry.get("title"),
            "description": "",
            "event_date": None,
            "is_past": True,
            "repository_folder": None,
            "repository_url": None,
            "youtube_id": entry["id"],
            "youtube_url": entry.get("url") or f"https://www.youtube.com/watch?v={entry['id']}",
            "in_youtube_show": True,
            "youtube_show_order": order,
            "mapping_note": "YouTube show entry without repository mapping",
            "repo_transcripts": [],
            "repo_notes": [],
            "repo_images": [],
        })

    manifest.sort(key=lambda item: (item.get("event_date") or "", item.get("guid") or ""))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"episodes": manifest}, indent=2, ensure_ascii=False) + "\n")
    print(f"episodes={len(manifest)} youtube={sum(bool(item['youtube_id']) for item in manifest)} repo={sum(bool(item['repository_folder']) for item in manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
