from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path


def chunk_text(text: str, max_chars: int = 3200, overlap_chars: int = 350) -> list[str]:
    if max_chars < 1 or not 0 <= overlap_chars < max_chars:
        raise ValueError("require max_chars >= 1 and 0 <= overlap_chars < max_chars")
    normalized = "\n\n".join(part.strip() for part in text.replace("\r\n", "\n").split("\n\n") if part.strip())
    if not normalized:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + max_chars, len(normalized))
        if end < len(normalized):
            search_from = start + max_chars // 2
            boundary = normalized.rfind("\n\n", search_from, end)
            if boundary < 0:
                boundary = normalized.rfind(" ", search_from, end)
            if boundary > start:
                end = boundary
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        next_start = max(end - overlap_chars, start + 1)
        while next_start < len(normalized) and normalized[next_start].isspace():
            next_start += 1
        start = next_start
    return chunks


def transcript_chunks(document: dict, max_chars: int = 3200, overlap_chars: int | None = None) -> list[dict]:
    if overlap_chars is None:
        overlap_chars = min(350, max_chars // 5)
    if max_chars < 1 or not 0 <= overlap_chars < max_chars:
        raise ValueError("require max_chars >= 1 and 0 <= overlap_chars < max_chars")

    expanded: list[dict] = []
    for segment in document.get("segments", []):
        text = segment.get("text", "").strip()
        if not text:
            continue
        pieces = chunk_text(text, max_chars=max_chars, overlap_chars=0)
        start = float(segment.get("start", 0))
        duration = float(segment.get("duration", 0))
        for index, piece in enumerate(pieces):
            piece_start = start + duration * index / len(pieces)
            piece_end = start + duration * (index + 1) / len(pieces)
            expanded.append({"text": piece, "start": piece_start, "duration": piece_end - piece_start})

    def render(items: list[dict]) -> dict:
        return {
            "text": " ".join(item["text"] for item in items),
            "start_s": items[0]["start"],
            "end_s": items[-1]["start"] + items[-1]["duration"],
        }

    output: list[dict] = []
    current: list[dict] = []
    current_size = 0
    for segment in expanded:
        addition = len(segment["text"]) + (1 if current else 0)
        if current and current_size + addition > max_chars:
            output.append(render(current))
            overlap: list[dict] = []
            overlap_size = 0
            for prior in reversed(current):
                prior_size = len(prior["text"]) + (1 if overlap else 0)
                if overlap_size + prior_size > overlap_chars:
                    break
                overlap.insert(0, prior)
                overlap_size += prior_size
            current = overlap
            current_size = len(" ".join(item["text"] for item in current))
            addition = len(segment["text"]) + (1 if current else 0)
            if current_size + addition > max_chars:
                current = []
                current_size = 0
                addition = len(segment["text"])
        current.append(segment)
        current_size += addition
    if current:
        output.append(render(current))
    return output


def chunk_id(metadata: dict, text: str) -> str:
    identity = json.dumps(metadata, sort_keys=True, ensure_ascii=False) + "\n" + text
    return hashlib.sha256(identity.encode()).hexdigest()[:20]


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
