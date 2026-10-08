import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_corpus.py"
VIDEO_ID = "dQw4w9WgXcQ"


def initialize_git_repo(repo):
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "--allow-empty", "-m", "fixture"], check=True)


def build_fixture(tmp_path, include_repository_transcript):
    repo = tmp_path / "repo"
    youtube = tmp_path / "youtube"
    repo.mkdir()
    youtube.mkdir()
    relative_transcript = "episode/transcript.md"
    if include_repository_transcript:
        transcript_path = repo / relative_transcript
        transcript_path.parent.mkdir()
        transcript_path.write_text("edited repository transcript")
    initialize_git_repo(repo)

    manifest = {
        "episodes": [
            {
                "guid": "test-episode",
                "title": "Test episode",
                "youtube_id": VIDEO_ID,
                "youtube_url": f"https://www.youtube.com/watch?v={VIDEO_ID}",
                "repo_transcripts": [relative_transcript],
                "repo_notes": [],
                "repo_images": [],
            }
        ]
    }
    manifest_path = tmp_path / "episodes.json"
    manifest_path.write_text(json.dumps(manifest))
    captions_path = tmp_path / "captions.json"
    captions_path.write_text("{}")
    (youtube / f"{VIDEO_ID}.json").write_text(
        json.dumps(
            {
                "transcription_model": "test-whisper",
                "segments": [{"start": 0, "duration": 1, "text": "local whisper transcript"}],
            }
        )
    )
    output = tmp_path / "chunks.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest_path),
            "--repo",
            str(repo),
            "--youtube",
            str(youtube),
            "--diagram-captions",
            str(captions_path),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    chunks = [json.loads(line) for line in output.read_text().splitlines()]
    provenance = json.loads(output.with_suffix(".provenance.json").read_text())
    return chunks, provenance


def test_repository_transcript_precedes_local_whisper(tmp_path):
    chunks, provenance = build_fixture(tmp_path, include_repository_transcript=True)
    assert {chunk["source_type"] for chunk in chunks} == {"repository_transcript"}
    assert chunks[0]["text"] == "edited repository transcript"
    assert provenance["repository_clean"] is True
    assert provenance["chunks"] == len(chunks)


def test_local_whisper_is_used_when_repository_transcript_is_missing(tmp_path):
    chunks, _ = build_fixture(tmp_path, include_repository_transcript=False)
    assert {chunk["source_type"] for chunk in chunks} == {"local_whisper_transcript"}
    assert chunks[0]["transcription_model"] == "test-whisper"
    assert chunks[0]["source_path"] == f"data/youtube/{VIDEO_ID}.json"


def test_shared_repository_assets_are_indexed_once(tmp_path):
    repo = tmp_path / "repo"
    note = repo / "shared" / "notes.md"
    note.parent.mkdir(parents=True)
    note.write_text("shared note")
    initialize_git_repo(repo)
    manifest = {
        "episodes": [
            {
                "guid": "youtube-alias",
                "title": "Alias",
                "repository_folder": "shared",
                "repo_notes": ["shared/notes.md"],
            },
            {
                "guid": "aitw-001",
                "title": "Canonical",
                "repository_folder": "shared",
                "repo_notes": ["shared/notes.md"],
            },
        ]
    }
    manifest_path = tmp_path / "episodes.json"
    manifest_path.write_text(json.dumps(manifest))
    captions_path = tmp_path / "captions.json"
    captions_path.write_text("{}")
    output = tmp_path / "chunks.jsonl"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--manifest",
            str(manifest_path),
            "--repo",
            str(repo),
            "--youtube",
            str(tmp_path / "youtube"),
            "--diagram-captions",
            str(captions_path),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    chunks = [json.loads(line) for line in output.read_text().splitlines()]
    assert [(chunk["episode_guid"], chunk["source_path"]) for chunk in chunks] == [
        ("aitw-001", "shared/notes.md")
    ]
