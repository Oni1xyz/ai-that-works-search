from __future__ import annotations

from urllib.parse import quote

YOUTUBE_TRANSCRIPT_TYPES = {"youtube_transcript", "local_whisper_transcript"}
REPOSITORY_SOURCE_TYPES = {"repository_transcript", "repository_note", "repository_image", "diagram_caption"}
CANONICAL_REPOSITORY_URL = "https://github.com/ai-that-works/ai-that-works"


def repository_file_url(item: dict) -> str | None:
    commit = item.get("repository_commit")
    source_path = item.get("source_path")
    if commit and source_path:
        return f"{CANONICAL_REPOSITORY_URL}/blob/{quote(str(commit), safe='')}/{quote(str(source_path), safe='/')}"
    return None


def source_reference(item: dict, *, timestamp: bool = True) -> str | None:
    source_type = item.get("source_type")
    if source_type in YOUTUBE_TRANSCRIPT_TYPES:
        video_id = item.get("youtube_id")
        source = f"https://www.youtube.com/watch?v={video_id}" if video_id else None
        if timestamp and source and item.get("start_s") is not None:
            return f"{source}&t={int(item['start_s'])}s"
        return source
    if source_type in REPOSITORY_SOURCE_TYPES:
        return repository_file_url(item)
    return None
