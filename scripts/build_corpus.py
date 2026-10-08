#!/usr/bin/env python3
"""Assemble searchable chunks from YouTube transcripts and repository episode assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _shared import require_youtube_video_id, safe_child_path

from aitw_search.corpus import chunk_id, chunk_text, transcript_chunks  # noqa: E402


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/episodes.json"))
    parser.add_argument("--repo", type=Path, default=Path("source/ai-that-works"))
    parser.add_argument("--youtube", type=Path, default=Path("data/youtube"))
    parser.add_argument("--diagram-captions", type=Path, default=Path("data/diagram_captions.json"))
    parser.add_argument(
        "--diagram-caption-provenance", type=Path, default=Path("data/diagram_captions_provenance.json")
    )
    parser.add_argument("--youtube-snapshot", type=Path, default=Path("youtube_show.json"))
    parser.add_argument("--output", type=Path, default=Path("data/chunks.jsonl"))
    parser.add_argument("--provenance-output", type=Path)
    args = parser.parse_args()

    episodes = json.loads(args.manifest.read_text())["episodes"]
    captions = json.loads(args.diagram_captions.read_text()) if args.diagram_captions.exists() else {}
    caption_provenance = (
        json.loads(args.diagram_caption_provenance.read_text()) if args.diagram_caption_provenance.exists() else None
    )
    repo_root = args.repo.resolve()
    has_repository_assets = any(
        episode.get(key) for episode in episodes for key in ("repo_transcripts", "repo_notes", "repo_images")
    )
    repository_commit = None
    if has_repository_assets:
        try:
            repository_commit = subprocess.check_output(
                ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True, stderr=subprocess.PIPE
            ).strip()
            repository_status = subprocess.check_output(
                ["git", "-C", str(repo_root), "status", "--porcelain"], text=True, stderr=subprocess.PIPE
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SystemExit(f"repository-backed corpus requires a Git source repository: {repo_root}: {exc}") from exc
        if repository_status:
            raise SystemExit(
                f"source repository is dirty; commit or clean changes before corpus generation: {repo_root}"
            )
    chunks: list[dict] = []
    transcript_checksums: dict[str, str] = {}
    seen_repository_paths: set[str] = set()
    canonical_repository_guid: dict[str, str] = {}
    for episode in episodes:
        folder = episode.get("repository_folder")
        guid = episode.get("guid")
        if not folder or not guid:
            continue
        current = canonical_repository_guid.get(folder)
        if current is None or (guid.startswith("aitw-") and not current.startswith("aitw-")):
            canonical_repository_guid[folder] = guid

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
            "repository_url": episode.get("repository_url"),
            "repository_commit": repository_commit,
            "source_type": source_type,
            "source_path": source_path,
            **extra,
        }
        clean = text.strip()
        if not clean:
            return
        chunks.append({"id": chunk_id(metadata, clean), "text": clean, **metadata})

    for episode in episodes:
        folder = episode.get("repository_folder")
        owns_repository_assets = not folder or canonical_repository_guid.get(folder) == episode.get("guid")
        transcript_found = False
        local_transcript = None
        raw_video_id = episode.get("youtube_id")
        video_id = require_youtube_video_id(raw_video_id) if raw_video_id else None
        youtube_json = args.youtube / f"{video_id}.json" if video_id else None
        if youtube_json and youtube_json.exists():
            document = json.loads(youtube_json.read_text())
            transcript_hash = sha256_file(youtube_json)
            transcript_checksums[f"{video_id}.json"] = transcript_hash
            if document.get("transcription_model"):
                local_transcript = document
            else:
                transcript_chunks_added = 0
                for index, item in enumerate(transcript_chunks(document)):
                    add(
                        episode,
                        "youtube_transcript",
                        str(youtube_json),
                        item["text"],
                        chunk_index=index,
                        start_s=item["start_s"],
                        end_s=item["end_s"],
                        transcript_origin=(
                            "youtube_generated"
                            if document.get("is_generated") is True
                            else "youtube_manual"
                            if document.get("is_generated") is False
                            else "youtube_caption"
                        ),
                        transcript_language=document.get("language"),
                        caption_is_generated=document.get("is_generated"),
                        transcript_retrieved_at=document.get("retrieved_at"),
                        transcript_sha256=transcript_hash,
                    )
                    transcript_chunks_added += 1
                transcript_found = transcript_chunks_added > 0

        # Prefer an edited repository transcript over a local speech-to-text fallback.
        if not transcript_found and owns_repository_assets:
            for relative in episode.get("repo_transcripts", []):
                if relative in seen_repository_paths:
                    continue
                path = safe_child_path(args.repo, relative)
                if path.exists():
                    seen_repository_paths.add(relative)
                    transcript_chunks_added = 0
                    for index, text in enumerate(chunk_text(path.read_text(errors="replace"))):
                        add(episode, "repository_transcript", relative, text, chunk_index=index)
                        transcript_chunks_added += 1
                    transcript_found = transcript_found or transcript_chunks_added > 0

        if not transcript_found and local_transcript is not None and youtube_json is not None:
            for index, item in enumerate(transcript_chunks(local_transcript)):
                add(
                    episode,
                    "local_whisper_transcript",
                    f"data/youtube/{video_id}.json",
                    item["text"],
                    chunk_index=index,
                    start_s=item["start_s"],
                    end_s=item["end_s"],
                    transcription_model=local_transcript.get("transcription_model"),
                    transcript_origin="local_whisper",
                    transcript_language=local_transcript.get("language"),
                    transcript_retrieved_at=local_transcript.get("generated_at"),
                    transcript_sha256=sha256_file(youtube_json),
                )

        for relative in episode.get("repo_notes", []) if owns_repository_assets else []:
            if relative in seen_repository_paths:
                continue
            path = safe_child_path(args.repo, relative)
            if not path.exists() or path.stat().st_size > 2_000_000:
                continue
            seen_repository_paths.add(relative)
            for index, text in enumerate(chunk_text(path.read_text(errors="replace"))):
                add(episode, "repository_note", relative, text, chunk_index=index)

        for relative in episode.get("repo_images", []) if owns_repository_assets else []:
            if relative in seen_repository_paths:
                continue
            seen_repository_paths.add(relative)
            caption = captions.get(relative)
            if caption:
                add(episode, "diagram_caption", relative, caption, caption_provenance=caption_provenance)
            else:
                # Preserve discoverability and provenance even when no OCR/caption exists.
                add(episode, "repository_image", relative, f"Repository image asset: {Path(relative).name}")

    # Deduplicate exact IDs while preserving deterministic order.
    unique = {item["id"]: item for item in chunks}
    ordered = sorted(
        unique.values(),
        key=lambda item: (
            item.get("event_date") or "",
            item.get("episode_guid") or "",
            item["source_type"],
            item.get("chunk_index", 0),
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for item in ordered:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    provenance_output = args.provenance_output or args.output.with_suffix(".provenance.json")
    provenance = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "repository_commit": repository_commit,
        "repository_clean": True if repository_commit else None,
        "manifest_sha256": sha256_file(args.manifest),
        "youtube_snapshot_sha256": sha256_file(args.youtube_snapshot) if args.youtube_snapshot.exists() else None,
        "diagram_captions_sha256": sha256_file(args.diagram_captions) if args.diagram_captions.exists() else None,
        "diagram_caption_provenance_sha256": (
            sha256_file(args.diagram_caption_provenance) if args.diagram_caption_provenance.exists() else None
        ),
        "build_corpus_sha256": sha256_file(Path(__file__)),
        "corpus_module_sha256": sha256_file(Path(chunk_text.__code__.co_filename)),
        "transcript_sha256": dict(sorted(transcript_checksums.items())),
        "chunks_sha256": sha256_file(args.output),
        "chunks": len(ordered),
    }
    provenance_output.parent.mkdir(parents=True, exist_ok=True)
    provenance_output.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    counts: dict[str, int] = {}
    for item in ordered:
        counts[item["source_type"]] = counts.get(item["source_type"], 0) + 1
    print(
        json.dumps(
            {"chunks": len(ordered), "episodes": len({item["episode_guid"] for item in ordered}), "sources": counts},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
