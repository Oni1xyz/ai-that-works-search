from aitw_search.sources import source_reference


def test_youtube_transcript_uses_timestamped_video_url():
    item = {
        "source_type": "youtube_transcript",
        "youtube_id": "dQw4w9WgXcQ",
        "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "start_s": 61.9,
        "source_path": "local.json",
    }
    assert source_reference(item).endswith("&t=61s")


def test_repository_note_uses_repository_path_not_youtube_url():
    item = {
        "source_type": "repository_note",
        "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "source_path": "episode/notes.md",
        "repository_url": "https://github.com/example/repo/tree/main/episode",
    }
    assert source_reference(item) is None


def test_repository_source_builds_commit_pinned_url():
    item = {
        "source_type": "repository_note",
        "source_path": "episode/my notes.md",
        "repository_url": "https://github.com/example/repo/tree/main/episode",
        "repository_commit": "a" * 40,
    }
    assert source_reference(item) == (
        f"https://github.com/ai-that-works/ai-that-works/blob/{'a' * 40}/episode/my%20notes.md"
    )


def test_repository_root_url_builds_commit_pinned_url():
    item = {
        "source_type": "repository_note",
        "source_path": "episode/README.md",
        "repository_url": "https://github.com/example/repo",
        "repository_commit": "a" * 40,
    }
    assert source_reference(item) == (
        f"https://github.com/ai-that-works/ai-that-works/blob/{'a' * 40}/episode/README.md"
    )


def test_unknown_source_type_has_no_reference():
    assert source_reference({"source_type": "unknown", "repository_url": "repo"}) is None
