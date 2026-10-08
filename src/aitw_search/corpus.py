from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def chunk_text(text: str, max_chars: int = 3200, overlap_chars: int = 350) -> list[str]:
    paragraphs = [part.strip() for part in text.replace("\r\n", "\n").split("\n\n") if part.strip()]
    if not paragraphs:
        paragraphs = [line.strip() for line in text.splitlines() if line.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            pieces = [paragraph[i : i + max_chars] for i in range(0, len(paragraph), max_chars - overlap_chars)]
        else:
            pieces = [paragraph]
        for piece in pieces:
            candidate = f"{current}\n\n{piece}".strip() if current else piece
            if current and len(candidate) > max_chars:
                chunks.append(current)
                current = (current[-overlap_chars:] + "\n\n" + piece).strip()
            else:
                current = candidate
    if current:
        chunks.append(current)
    return chunks


def transcript_chunks(document: dict, max_chars: int = 3200) -> list[dict]:
    output: list[dict] = []
    current: list[dict] = []
    size = 0
    for segment in document.get("segments", []):
        text = segment.get("text", "").strip()
        if current and size + len(text) + 1 > max_chars:
            output.append({
                "text": " ".join(item["text"].strip() for item in current),
                "start_s": current[0]["start"],
                "end_s": current[-1]["start"] + current[-1].get("duration", 0),
            })
            current = current[-2:]
            size = sum(len(item.get("text", "")) + 1 for item in current)
        current.append(segment)
        size += len(text) + 1
    if current:
        output.append({
            "text": " ".join(item["text"].strip() for item in current),
            "start_s": current[0]["start"],
            "end_s": current[-1]["start"] + current[-1].get("duration", 0),
        })
    return output


def chunk_id(metadata: dict, text: str) -> str:
    identity = json.dumps(metadata, sort_keys=True, ensure_ascii=False) + "\n" + text
    return hashlib.sha1(identity.encode()).hexdigest()[:20]


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
