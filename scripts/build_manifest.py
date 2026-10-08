#!/usr/bin/env python3
"""Build a union manifest from the YouTube show and repository episode metadata."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from _shared import require_youtube_video_id, safe_child_path

VIDEO_RE = re.compile(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})")
TEXT_SUFFIXES = {".md", ".txt"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}
VIDEO_OVERRIDES = {
    # Repository episode 59 lacks a YouTube link; this show entry is the exact No Vibes
    # performance-engineering episode and is otherwise unmapped.
    "aitw-059": "mm6n4n09RaU",
}
SHOW_FOLDER_OVERRIDES = {
    # The show labels this as Part 1; repository metadata identifies the companion video
    # sCScFZB4Am8 in the same multimodality episode folder.
    "sqJrl09dDmI": "2025-07-22-multimodality",
}


def validate_mapping_invariants(manifest: list[dict], show_ids: set[str]) -> None:
    guids = [item["guid"] for item in manifest]
    if len(guids) != len(set(guids)):
        raise SystemExit("manifest contains duplicate episode GUIDs")
    video_ids = [item["youtube_id"] for item in manifest if item.get("youtube_id")]
    if len(video_ids) != len(set(video_ids)):
        raise SystemExit("manifest contains duplicate YouTube video ownership")
    unknown_overrides = (set(VIDEO_OVERRIDES.values()) | set(SHOW_FOLDER_OVERRIDES)) - show_ids
    if unknown_overrides:
        raise SystemExit(f"mapping overrides are absent from the show snapshot: {sorted(unknown_overrides)}")
    folder_owners: dict[str, list[str]] = {}
    for item in manifest:
        if item.get("repository_folder"):
            folder_owners.setdefault(item["repository_folder"], []).append(item["guid"])
    for folder, owners in folder_owners.items():
        if len(owners) > 1 and sum(guid.startswith("aitw-") for guid in owners) != 1:
            raise SystemExit(f"ambiguous canonical repository-folder ownership for {folder}: {sorted(owners)}")


def video_id(url: str | None) -> str | None:
    if not url:
        return None
    match = VIDEO_RE.search(url)
    return match.group(1) if match else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("source/ai-that-works"))
    parser.add_argument("--youtube", type=Path, default=Path("youtube_show.json"))
    parser.add_argument("--output", type=Path, default=Path("data/episodes.json"))

    args = parser.parse_args()

    repo_root = args.repo.resolve()
    repository = json.loads((repo_root / "data.json").read_text())["episodes"]
    show_entries = json.loads(args.youtube.read_text()).get("entries", [])
    show = {require_youtube_video_id(entry["id"]): entry for entry in show_entries}
    manifest = []
    mapped_ids: set[str] = set()

    for episode in repository:
        if episode.get("event_type") != "episode" or episode.get("isWorkshop"):
            continue
        links = episode.get("links") or {}
        media = episode.get("media") or {}
        vid = VIDEO_OVERRIDES.get(episode.get("guid")) or video_id(links.get("youtube")) or video_id(media.get("url"))
        mapping_note = "manual title/topic match" if episode.get("guid") in VIDEO_OVERRIDES else "repository metadata"
        if vid:
            vid = require_youtube_video_id(vid)
        folder_name = episode.get("folder")
        folder = safe_child_path(repo_root, folder_name) if folder_name else None
        files = [path for path in folder.rglob("*") if path.is_file()] if folder and folder.exists() else []
        relative_files = [str(path.relative_to(repo_root)) for path in files]
        transcript_files = [
            path
            for path in relative_files
            if "transcript" in Path(path).name.lower() or "trasncript" in Path(path).name.lower()
        ]
        note_files = [
            path
            for path in relative_files
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
        entry_id = require_youtube_video_id(entry["id"])
        if entry_id in mapped_ids:
            continue
        folder_name = SHOW_FOLDER_OVERRIDES.get(entry_id)
        folder = safe_child_path(repo_root, folder_name) if folder_name else None
        files = [path for path in folder.rglob("*") if path.is_file()] if folder and folder.exists() else []
        relative_files = [str(path.relative_to(repo_root)) for path in files]
        transcript_files = [
            path
            for path in relative_files
            if "transcript" in Path(path).name.lower() or "trasncript" in Path(path).name.lower()
        ]
        note_files = (
            [
                path
                for path in relative_files
                if Path(path).suffix.lower() in TEXT_SUFFIXES
                and path not in transcript_files
                and (Path(path).parent == Path(folder_name) or "/thoughts/" in f"/{path}")
            ]
            if folder_name
            else []
        )
        image_files = [path for path in relative_files if Path(path).suffix.lower() in IMAGE_SUFFIXES]
        manifest.append(
            {
                "guid": f"youtube-{entry_id}",
                "season": None,
                "episode": None,
                "title": entry.get("title"),
                "event_date": None,
                "is_past": True,
                "repository_folder": folder_name,
                "repository_url": f"https://github.com/ai-that-works/ai-that-works/tree/main/{folder_name}"
                if folder_name
                else None,
                "youtube_id": entry_id,
                "youtube_url": entry.get("url") or f"https://www.youtube.com/watch?v={entry_id}",
                "in_youtube_show": True,
                "youtube_show_order": order,
                "mapping_note": "Inferred companion Part 1 repository mapping"
                if folder_name
                else "YouTube show entry without repository mapping",
                "repo_transcripts": sorted(transcript_files),
                "repo_notes": sorted(note_files),
                "repo_images": sorted(image_files),
            }
        )

    manifest.sort(key=lambda item: (item.get("event_date") or "", item.get("guid") or ""))
    validate_mapping_invariants(manifest, set(show))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"episodes": manifest}, indent=2, ensure_ascii=False) + "\n")
    youtube_count = sum(bool(item["youtube_id"]) for item in manifest)
    repo_count = sum(bool(item["repository_folder"]) for item in manifest)
    print(f"episodes={len(manifest)} youtube={youtube_count} repo={repo_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
