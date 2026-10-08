#!/usr/bin/env python3
"""Backfill blocked YouTube captions by downloading audio and transcribing locally on Apple Silicon."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", type=Path, default=Path("youtube_show.json"))
    parser.add_argument("--output", type=Path, default=Path("data/youtube"))
    parser.add_argument("--audio-dir", type=Path, default=Path("data/audio-cache"))
    parser.add_argument("--model", default="mlx-community/distil-whisper-large-v3")
    parser.add_argument("--keep-audio", action="store_true")
    args = parser.parse_args()

    entries = json.loads(args.show.read_text()).get("entries", [])
    args.output.mkdir(parents=True, exist_ok=True)
    args.audio_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.output / "local_transcription_status.json"
    status = []

    for order, entry in enumerate(entries, 1):
        video_id = entry["id"]
        destination = args.output / f"{video_id}.json"
        text_destination = args.output / f"{video_id}.txt"
        if destination.exists() and text_destination.exists():
            status.append({"video_id": video_id, "status": "existing"})
            continue
        audio_template = args.audio_dir / f"{video_id}.%(ext)s"
        audio_path = args.audio_dir / f"{video_id}.m4a"
        whisper_json = args.audio_dir / f"{video_id}.json"
        try:
            if not audio_path.exists():
                print(f"[{order}/{len(entries)}] downloading {video_id}", flush=True)
                run([
                    "uvx", "--from", "yt-dlp", "yt-dlp",
                    "--no-write-subs", "-f", "bestaudio[ext=m4a]/bestaudio",
                    "-o", str(audio_template),
                    f"https://www.youtube.com/watch?v={video_id}",
                ])
            print(f"[{order}/{len(entries)}] transcribing {video_id}", flush=True)
            run([
                "uvx", "--from", "mlx-whisper", "mlx_whisper", str(audio_path),
                "--model", args.model,
                "--language", "en",
                "--output-dir", str(args.audio_dir),
                "--output-name", video_id,
                "--output-format", "json",
                "--verbose", "False",
            ])
            raw = json.loads(whisper_json.read_text())
            segments = [
                {
                    "start": float(segment["start"]),
                    "duration": float(segment["end"] - segment["start"]),
                    "text": segment["text"].strip(),
                }
                for segment in raw.get("segments", [])
                if segment.get("text", "").strip()
            ]
            document = {
                "order": order,
                "video_id": video_id,
                "title": entry.get("title"),
                "url": entry.get("url") or f"https://www.youtube.com/watch?v={video_id}",
                "language": raw.get("language", "en"),
                "is_generated": True,
                "transcription_model": args.model,
                "segments": segments,
            }
            destination.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
            text_destination.write_text("\n".join(f"[{int(segment['start']) // 60}:{int(segment['start']) % 60:02d}] {segment['text']}" for segment in segments) + "\n")
            status.append({"video_id": video_id, "status": "ok", "segments": len(segments)})
            print(f"[{order}/{len(entries)}] ok {video_id}: {len(segments)} segments", flush=True)
        except Exception as exc:
            status.append({"video_id": video_id, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
            print(f"[{order}/{len(entries)}] ERROR {video_id}: {exc}", flush=True)
        finally:
            status_path.write_text(json.dumps(status, indent=2) + "\n")
            if not args.keep_audio:
                audio_path.unlink(missing_ok=True)
                whisper_json.unlink(missing_ok=True)

    failures = [item for item in status if item["status"] == "error"]
    print(f"complete: {len(status) - len(failures)}/{len(entries)} available; failures={len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
