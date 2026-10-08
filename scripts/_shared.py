from __future__ import annotations

import re
from pathlib import Path

VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


def require_youtube_video_id(value: object) -> str:
    video_id = str(value)
    if not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise ValueError(f"invalid YouTube video ID: {video_id!r}")
    return video_id


def safe_child_path(root: Path, relative: object) -> Path:
    root = root.resolve()
    candidate = (root / str(relative)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes source root: {relative!r}") from exc
    return candidate
