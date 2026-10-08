from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from datetime import datetime
from io import BytesIO
from pathlib import Path, PurePosixPath

import numpy as np

from .sources import REPOSITORY_SOURCE_TYPES, YOUTUBE_TRANSCRIPT_TYPES

REQUIRED_CHUNK_FIELDS = frozenset({"id", "title", "source_type", "text"})
ALLOWED_SOURCE_TYPES = REPOSITORY_SOURCE_TYPES | YOUTUBE_TRANSCRIPT_TYPES
YOUTUBE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
ALLOWED_REPOSITORY_ROOTS = (
    "https://github.com/ai-that-works/ai-that-works",
    "https://github.com/hellovai/ai-that-works",
)
MAX_METADATA_BYTES = 1024 * 1024
MAX_PROVENANCE_BYTES = 4 * 1024 * 1024
MAX_CHUNKS_BYTES = 64 * 1024 * 1024
MAX_EMBEDDINGS_BYTES = 128 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
REQUIRED_CORPUS_PROVENANCE = frozenset(
    {
        "build_corpus_sha256",
        "chunks",
        "chunks_sha256",
        "corpus_module_sha256",
        "created_at",
        "diagram_caption_provenance_sha256",
        "diagram_captions_sha256",
        "manifest_sha256",
        "repository_clean",
        "repository_commit",
        "schema_version",
        "transcript_sha256",
        "youtube_snapshot_sha256",
    }
)
REQUIRED_V1_METADATA = frozenset(
    {
        "model",
        "model_revision",
        "created_at",
        "dimensions",
        "dtype",
        "chunks",
        "normalized",
        "query_prompt",
        "document_prompt",
        "repository_commit",
        "input_chunks_sha256",
        "corpus_provenance_sha256",
        "chunks_sha256",
        "embeddings_sha256",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_chunks(chunks: list[dict], repository_commit: str | None = None) -> None:
    seen_ids: set[str] = set()
    for position, chunk in enumerate(chunks, 1):
        if not isinstance(chunk, dict):
            raise ValueError(f"chunk {position} must be an object")
        missing = sorted(REQUIRED_CHUNK_FIELDS - chunk.keys())
        if missing:
            raise ValueError(f"chunk {position} is missing required fields: {', '.join(missing)}")
        for field in REQUIRED_CHUNK_FIELDS:
            if not isinstance(chunk[field], str) or not chunk[field].strip():
                raise ValueError(f"chunk {position} field {field} must be a nonempty string")
        if len(chunk["id"]) > 256 or len(chunk["title"]) > 1000 or len(chunk["text"]) > 100_000:
            raise ValueError(f"chunk {position} exceeds a supported string length")
        chunk_id = chunk["id"]
        if chunk_id in seen_ids:
            raise ValueError(f"duplicate chunk ID at chunk {position}: {chunk_id}")
        seen_ids.add(chunk_id)
        source_type = chunk["source_type"]
        if source_type not in ALLOWED_SOURCE_TYPES:
            raise ValueError(f"chunk {position} has unsupported source_type: {source_type}")
        source_path = chunk.get("source_path")
        if not isinstance(source_path, str) or not source_path or "\\" in source_path:
            raise ValueError(f"chunk {position} source_path must be a normalized relative POSIX path")
        if len(source_path) > 4096:
            raise ValueError(f"chunk {position} source_path exceeds the supported length")
        parsed_path = PurePosixPath(source_path)
        if (
            parsed_path.is_absolute()
            or str(parsed_path) != source_path
            or any(part in {"", ".", ".."} for part in parsed_path.parts)
        ):
            raise ValueError(f"chunk {position} source_path must be a normalized relative POSIX path")
        chunk_commit = chunk.get("repository_commit")
        if repository_commit is not None and chunk_commit != repository_commit:
            raise ValueError(f"chunk {position} repository_commit does not match index metadata")
        repository_url = chunk.get("repository_url")
        if repository_url is not None:
            if not isinstance(repository_url, str) or not any(
                repository_url == root or repository_url.startswith(root + "/")
                for root in ALLOWED_REPOSITORY_ROOTS
            ):
                raise ValueError(f"chunk {position} repository_url is not an approved repository origin")
        youtube_id = chunk.get("youtube_id")
        if youtube_id is not None and (not isinstance(youtube_id, str) or not YOUTUBE_ID_RE.fullmatch(youtube_id)):
            raise ValueError(f"chunk {position} youtube_id is malformed")
        if source_type in YOUTUBE_TRANSCRIPT_TYPES and not isinstance(youtube_id, str):
            raise ValueError(f"chunk {position} YouTube transcript is missing youtube_id")
        youtube_url = chunk.get("youtube_url")
        expected_youtube_url = f"https://www.youtube.com/watch?v={youtube_id}" if youtube_id else None
        if youtube_url is not None and youtube_url != expected_youtube_url:
            raise ValueError(f"chunk {position} youtube_url does not match youtube_id")
        if source_type in REPOSITORY_SOURCE_TYPES:
            if not isinstance(chunk_commit, str) or not COMMIT_SHA_RE.fullmatch(chunk_commit):
                raise ValueError(f"chunk {position} repository source is missing an immutable commit")
        if "transcript_sha256" in chunk:
            validate_sha256(chunk["transcript_sha256"], f"chunk {position} transcript_sha256")
        episode_guid = chunk.get("episode_guid")
        if episode_guid is not None and (not isinstance(episode_guid, str) or not episode_guid.strip()):
            raise ValueError(f"chunk {position} episode_guid must be a nonempty string or null")
        numeric_times: dict[str, float] = {}
        for field in ("start_s", "end_s"):
            value = chunk.get(field)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"chunk {position} {field} must be a finite nonnegative number or null")
            numeric_times[field] = float(value)
        if numeric_times.get("end_s", float("inf")) < numeric_times.get("start_s", 0.0):
            raise ValueError(f"chunk {position} end_s must not precede start_s")


def validate_sha256(value: object, field: str, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")


def validate_corpus_provenance(provenance: dict) -> None:
    if provenance.get("schema_version") != 1:
        raise ValueError("unsupported or missing corpus provenance schema_version")
    missing = sorted(REQUIRED_CORPUS_PROVENANCE - provenance.keys())
    if missing:
        raise ValueError(f"corpus provenance is missing required fields: {', '.join(missing)}")
    if not isinstance(provenance["repository_commit"], str) or not COMMIT_SHA_RE.fullmatch(
        provenance["repository_commit"]
    ):
        raise ValueError("corpus provenance repository_commit must be a lowercase 40-character commit SHA")
    if provenance["repository_clean"] is not True:
        raise ValueError("corpus provenance requires repository_clean=true")
    if not isinstance(provenance["chunks"], int) or isinstance(provenance["chunks"], bool) or provenance["chunks"] < 1:
        raise ValueError("corpus provenance chunks must be a positive integer")
    if not isinstance(provenance["created_at"], str):
        raise ValueError("corpus provenance created_at must be an ISO-8601 string")
    try:
        created_at = datetime.fromisoformat(provenance["created_at"])
    except ValueError as exc:
        raise ValueError("corpus provenance created_at must be an ISO-8601 timestamp") from exc
    if created_at.tzinfo is None:
        raise ValueError("corpus provenance created_at must include a timezone")
    for field in ("build_corpus_sha256", "chunks_sha256", "corpus_module_sha256", "manifest_sha256"):
        validate_sha256(provenance[field], f"corpus provenance {field}")
    for field in (
        "diagram_caption_provenance_sha256",
        "diagram_captions_sha256",
        "youtube_snapshot_sha256",
    ):
        validate_sha256(provenance[field], f"corpus provenance {field}", optional=True)
    transcript_hashes = provenance["transcript_sha256"]
    if not isinstance(transcript_hashes, dict):
        raise ValueError("corpus provenance transcript_sha256 must be an object")
    for name, digest in transcript_hashes.items():
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ValueError("corpus provenance transcript_sha256 keys must be plain filenames")
        validate_sha256(digest, f"corpus provenance transcript_sha256[{name!r}]")


def validate_index_metadata(metadata: dict) -> None:
    if not isinstance(metadata["model"], str) or not metadata["model"].strip():
        raise ValueError("index model must be a nonempty string")
    for field in ("dimensions", "chunks"):
        if not isinstance(metadata[field], int) or isinstance(metadata[field], bool) or metadata[field] < 1:
            raise ValueError(f"index {field} must be a positive integer")
    if metadata["dimensions"] > 8192 or metadata["chunks"] > 100_000:
        raise ValueError("index dimensions or chunk count exceeds the supported limit")
    if metadata["dtype"] != "float32":
        raise ValueError("schema version 1 requires dtype=float32")
    if metadata["normalized"] is not True:
        raise ValueError("schema version 1 requires normalized=true")
    if not isinstance(metadata["repository_commit"], str) or not COMMIT_SHA_RE.fullmatch(
        metadata["repository_commit"]
    ):
        raise ValueError("index repository_commit must be a lowercase 40-character commit SHA")
    if not isinstance(metadata["created_at"], str):
        raise ValueError("index created_at must be an ISO-8601 timestamp")
    try:
        created_at = datetime.fromisoformat(metadata["created_at"])
    except ValueError as exc:
        raise ValueError("index created_at must be an ISO-8601 timestamp") from exc
    if created_at.tzinfo is None:
        raise ValueError("index created_at must include a timezone")
    for field in ("query_prompt", "document_prompt"):
        if not isinstance(metadata[field], bool):
            raise ValueError(f"index {field} must be a boolean")
    for field in (
        "input_chunks_sha256",
        "corpus_provenance_sha256",
        "chunks_sha256",
        "embeddings_sha256",
    ):
        validate_sha256(metadata[field], f"index {field}")


def read_current_pointer(pointer: Path) -> str:
    if pointer.is_symlink():
        raise ValueError("index CURRENT must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(pointer, flags)
    except OSError as exc:
        raise ValueError("cannot safely open index CURRENT pointer") from exc
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1 or details.st_size > 256:
            raise ValueError("index CURRENT must be a small, singly linked regular file")
        payload = os.read(descriptor, 257)
        if len(payload) > 256:
            raise ValueError("index CURRENT exceeds the 256-byte limit")
        try:
            return payload.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise ValueError("index CURRENT must contain UTF-8 text") from exc
    finally:
        os.close(descriptor)


def resolve_index_dir(index_dir: Path) -> Path:
    index_dir = index_dir.resolve()
    pointer = index_dir / "CURRENT"
    if not pointer.exists() and not pointer.is_symlink():
        return index_dir
    relative = Path(read_current_pointer(pointer))
    if relative.is_absolute() or len(relative.parts) != 2 or relative.parts[0] != "versions":
        raise ValueError("invalid index CURRENT pointer layout")
    versions = index_dir / "versions"
    candidate = index_dir / relative
    if versions.is_symlink() or candidate.is_symlink():
        raise ValueError("index CURRENT pointer may not target a symlink")
    resolved = (index_dir / relative).resolve()
    if resolved.parent != versions.resolve() or not resolved.is_dir():
        raise ValueError("index CURRENT pointer does not name a version directory")
    return resolved


def _read_artifact(directory_fd: int, name: str, max_bytes: int) -> bytearray:
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
            dir_fd=directory_fd,
        )
    except OSError as exc:
        raise ValueError(f"cannot safely open index artifact: {name}") from exc
    try:
        details = os.fstat(descriptor)
        if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
            raise ValueError(f"index artifact must be a singly linked regular file: {name}")
        if details.st_size > max_bytes:
            raise ValueError(f"index artifact exceeds its size limit: {name}")
        payload = bytearray()
        while len(payload) <= max_bytes:
            block = os.read(descriptor, min(1024 * 1024, max_bytes + 1 - len(payload)))
            if not block:
                break
            payload.extend(block)
        if len(payload) > max_bytes:
            raise ValueError(f"index artifact exceeds its size limit: {name}")
        return payload
    finally:
        os.close(descriptor)


def _open_selected_index(index_dir: Path) -> int:
    if index_dir.is_symlink():
        raise ValueError("index root may not be a symlink")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    root_fd = os.open(index_dir, directory_flags)
    try:
        root_stat = os.fstat(root_fd)
        path_stat = os.stat(index_dir, follow_symlinks=False)
        if (root_stat.st_dev, root_stat.st_ino) != (path_stat.st_dev, path_stat.st_ino):
            raise ValueError("index root changed while opening")
        try:
            pointer_payload = _read_artifact(root_fd, "CURRENT", 256)
        except ValueError as exc:
            try:
                os.stat("CURRENT", dir_fd=root_fd, follow_symlinks=False)
            except FileNotFoundError:
                return os.dup(root_fd)
            raise exc
        try:
            relative = pointer_payload.decode("utf-8").strip()
        except UnicodeDecodeError as exc:
            raise ValueError("index CURRENT must contain UTF-8 text") from exc
        parts = PurePosixPath(relative).parts
        if len(parts) != 2 or parts[0] != "versions" or parts[1] in {"", ".", ".."}:
            raise ValueError("invalid index CURRENT pointer layout")
        versions_fd = os.open("versions", directory_flags, dir_fd=root_fd)
        try:
            return os.open(parts[1], directory_flags, dir_fd=versions_fd)
        finally:
            os.close(versions_fd)
    finally:
        os.close(root_fd)


def load_index(index_dir: Path) -> tuple[dict, list[dict], np.ndarray]:
    directory_fd = _open_selected_index(index_dir)
    try:
        metadata_bytes = _read_artifact(directory_fd, "metadata.json", MAX_METADATA_BYTES)
        metadata = json.loads(metadata_bytes)
        if not isinstance(metadata, dict):
            raise ValueError("index metadata must be an object")
        if metadata.get("schema_version") != 1:
            raise ValueError(
                f"unsupported or missing index schema_version: {metadata.get('schema_version')!r}; rebuild the index"
            )
        missing_metadata = sorted(REQUIRED_V1_METADATA - metadata.keys())
        if missing_metadata:
            raise ValueError(f"index metadata is missing required fields: {', '.join(missing_metadata)}")
        model_revision = metadata["model_revision"]
        if not isinstance(model_revision, str) or not COMMIT_SHA_RE.fullmatch(model_revision):
            raise ValueError("index model_revision must be a lowercase 40-character commit SHA")
        validate_index_metadata(metadata)
        chunks_bytes = _read_artifact(directory_fd, "chunks.jsonl", MAX_CHUNKS_BYTES)
        expected_embedding_bytes = metadata["chunks"] * metadata["dimensions"] * 4 + 64 * 1024
        embeddings_bytes = _read_artifact(
            directory_fd, "embeddings.npy", min(MAX_EMBEDDINGS_BYTES, expected_embedding_bytes)
        )
        provenance_bytes = _read_artifact(directory_fd, "corpus_provenance.json", MAX_PROVENANCE_BYTES)
    finally:
        os.close(directory_fd)
    for name, payload in (("chunks_sha256", chunks_bytes), ("embeddings_sha256", embeddings_bytes)):
        actual = hashlib.sha256(payload).hexdigest()
        if metadata[name] != actual:
            raise ValueError(f"{name.removesuffix('_sha256')} checksum mismatch")
    actual_provenance_sha = hashlib.sha256(provenance_bytes).hexdigest()
    if metadata["corpus_provenance_sha256"] != actual_provenance_sha:
        raise ValueError("corpus_provenance.json checksum mismatch")
    corpus_provenance = json.loads(provenance_bytes)
    if not isinstance(corpus_provenance, dict):
        raise ValueError("corpus provenance must be an object")
    validate_corpus_provenance(corpus_provenance)
    if corpus_provenance.get("chunks_sha256") != metadata["chunks_sha256"]:
        raise ValueError("corpus provenance chunks checksum does not match index chunks")
    if corpus_provenance.get("repository_commit") != metadata["repository_commit"]:
        raise ValueError("corpus provenance repository commit does not match index metadata")
    if metadata["input_chunks_sha256"] != metadata["chunks_sha256"]:
        raise ValueError("index input_chunks_sha256 does not match indexed chunks")
    try:
        chunks = [json.loads(line) for line in chunks_bytes.decode("utf-8").splitlines() if line.strip()]
    except UnicodeDecodeError as exc:
        raise ValueError("chunks.jsonl must contain UTF-8 text") from exc
    validate_chunks(chunks, metadata["repository_commit"])
    embeddings = np.load(BytesIO(embeddings_bytes), allow_pickle=False)
    if embeddings.ndim != 2:
        raise ValueError(f"expected a 2D embedding matrix, got shape {embeddings.shape}")
    if not np.issubdtype(embeddings.dtype, np.floating):
        raise ValueError(f"expected floating-point embeddings, got {embeddings.dtype}")
    if embeddings.shape[0] < 1 or embeddings.shape[1] < 1:
        raise ValueError(f"embedding matrix must be non-empty, got shape {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings contain NaN or infinite values")
    if len(chunks) != embeddings.shape[0]:
        raise ValueError(f"chunk/vector count mismatch: {len(chunks)} chunks vs {embeddings.shape[0]} vectors")
    expected_chunks = metadata.get("chunks")
    if expected_chunks is not None and int(expected_chunks) != len(chunks):
        raise ValueError(f"chunk count mismatch: metadata={expected_chunks}, chunks={len(chunks)}")
    if corpus_provenance["chunks"] != len(chunks):
        raise ValueError(f"corpus provenance chunk count mismatch: {corpus_provenance['chunks']} vs {len(chunks)}")
    dimensions = metadata.get("dimensions")
    if dimensions is not None and int(dimensions) != embeddings.shape[1]:
        raise ValueError(f"dimension mismatch: metadata={dimensions}, vectors={embeddings.shape[1]}")
    expected_dtype = metadata.get("dtype")
    if expected_dtype is not None and str(expected_dtype) != str(embeddings.dtype):
        raise ValueError(f"dtype mismatch: metadata={expected_dtype}, vectors={embeddings.dtype}")
    norms = np.linalg.norm(embeddings, axis=1)
    if np.any(norms <= 1e-12):
        raise ValueError("embeddings contain zero-norm vectors")
    if metadata["normalized"] is not True:
        raise ValueError("schema version 1 requires normalized=true")
    if np.any(np.abs(norms - 1.0) > 1e-5):
        raise ValueError("embeddings are not unit-normalized as declared")
    return metadata, chunks, embeddings
