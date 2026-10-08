import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from _shared import require_youtube_video_id, safe_child_path  # noqa: E402
from build_manifest import SHOW_FOLDER_OVERRIDES, VIDEO_OVERRIDES, validate_mapping_invariants  # noqa: E402
from sanitize_show import sanitize  # noqa: E402


def test_accepts_standard_youtube_video_id():
    assert require_youtube_video_id("dQw4w9WgXcQ") == "dQw4w9WgXcQ"


@pytest.mark.parametrize("value", ["../outside", "short", "has a space", "a" * 12])
def test_rejects_unsafe_or_malformed_youtube_video_id(value):
    with pytest.raises(ValueError, match="invalid YouTube video ID"):
        require_youtube_video_id(value)


def test_safe_child_path_rejects_parent_escape(tmp_path):
    with pytest.raises(ValueError, match="escapes source root"):
        safe_child_path(tmp_path, "../outside.txt")


def test_safe_child_path_accepts_nested_path(tmp_path):
    assert safe_child_path(tmp_path, "episode/notes.md") == tmp_path / "episode" / "notes.md"


def test_sanitize_show_removes_descriptions_and_forwarded_ip():
    raw = {
        "id": "show",
        "title": "Show",
        "original_url": "https://youtube.test/show",
        "entries": [
            {
                "id": "dQw4w9WgXcQ",
                "title": "Episode",
                "url": "https://attacker.example/watch?v=dQw4w9WgXcQ",
                "duration": 42,
                "description": "third-party text",
                "__x_forwarded_for_ip": "192.0.2.1",
            }
        ],
    }

    cleaned = sanitize(raw)

    assert cleaned["webpage_url"] == "https://youtube.test/show"
    assert cleaned["entries"] == [
        {
            "id": "dQw4w9WgXcQ",
            "title": "Episode",
            "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "duration": 42,
        }
    ]


def override_show_ids():
    return set(VIDEO_OVERRIDES.values()) | set(SHOW_FOLDER_OVERRIDES)


def test_manifest_mapping_rejects_duplicate_video_ownership():
    manifest = [
        {"guid": "aitw-001", "youtube_id": "dQw4w9WgXcQ", "repository_folder": "one"},
        {"guid": "aitw-002", "youtube_id": "dQw4w9WgXcQ", "repository_folder": "two"},
    ]
    with pytest.raises(SystemExit, match="duplicate YouTube"):
        validate_mapping_invariants(manifest, override_show_ids() | {"dQw4w9WgXcQ"})


def test_manifest_mapping_rejects_ambiguous_folder_ownership():
    manifest = [
        {"guid": "youtube-one", "youtube_id": "dQw4w9WgXcQ", "repository_folder": "shared"},
        {"guid": "youtube-two", "youtube_id": "aaaaaaaaaaa", "repository_folder": "shared"},
    ]
    with pytest.raises(SystemExit, match="ambiguous canonical"):
        validate_mapping_invariants(manifest, override_show_ids() | {"dQw4w9WgXcQ", "aaaaaaaaaaa"})
