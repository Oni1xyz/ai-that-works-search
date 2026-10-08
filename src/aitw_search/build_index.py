from __future__ import annotations

import argparse
import fcntl
import json
import os
import secrets
import shutil
import stat
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
from huggingface_hub import model_info
from sentence_transformers import SentenceTransformer

from .cli import positive_int
from .corpus import iter_jsonl
from .index import COMMIT_SHA_RE, SHA256_RE, load_index, sha256_file, validate_chunks, validate_corpus_provenance


def encode(model: SentenceTransformer, texts: list[str], batch_size: int, kind: str) -> np.ndarray:
    kwargs = {"batch_size": batch_size, "show_progress_bar": True, "normalize_embeddings": True}
    prompts = getattr(model, "prompts", {}) or {}
    if kind in prompts:
        kwargs["prompt_name"] = kind
    embeddings = np.asarray(model.encode(texts, **kwargs), dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.isfinite(embeddings).all() or not np.isfinite(norms).all() or np.any(norms <= 1e-12):
        raise ValueError("embedding model returned non-finite or zero-norm vectors")
    # Some bfloat16-backed models return vectors a few parts per thousand away
    # from unit norm even when normalize_embeddings=True. Normalize once more
    # in float32 so dot products and the persisted normalization claim agree.
    return embeddings / norms[:, None]


def git_output(repo: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), *arguments], text=True, stderr=subprocess.PIPE
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"cannot read source repository provenance from {repo}: {exc}") from exc


def resolve_model_revision(model: str, requested_revision: str | None) -> str:
    try:
        revision = model_info(model, revision=requested_revision).sha
    except Exception as exc:
        if not requested_revision:
            raise SystemExit(
                "could not resolve an immutable model revision; pass --model-revision when offline"
            ) from exc
        if not COMMIT_SHA_RE.fullmatch(requested_revision):
            raise SystemExit("offline --model-revision must be a full 40-character Hugging Face commit SHA") from exc
        revision = requested_revision
    if not revision or not COMMIT_SHA_RE.fullmatch(revision):
        raise SystemExit(f"model revision is not an immutable 40-character commit SHA: {revision!r}")
    return revision


def fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def copy_descriptor(source: int, destination: int) -> None:
    os.lseek(source, 0, os.SEEK_SET)
    while True:
        block = os.read(source, 1024 * 1024)
        if not block:
            break
        offset = 0
        while offset < len(block):
            written = os.write(destination, block[offset:])
            if written == 0:
                raise OSError("short write while publishing index artifact")
            offset += written
    os.fsync(destination)


def require_private_directory(descriptor: int, label: str) -> os.stat_result:
    details = os.fstat(descriptor)
    if not stat.S_ISDIR(details.st_mode):
        raise ValueError(f"{label} must be a real directory")
    if details.st_uid != os.geteuid() or details.st_mode & 0o022:
        raise ValueError(f"{label} must be owned by the current user and not group/world writable")
    return details


def require_trusted_directory_chain(path: Path) -> None:
    for directory in (path, *path.parents):
        details = os.lstat(directory)
        if stat.S_ISLNK(details.st_mode) or not stat.S_ISDIR(details.st_mode):
            raise ValueError(f"index output ancestor must be a real directory: {directory}")
        if details.st_uid not in {0, os.geteuid()}:
            raise ValueError(f"index output ancestor is owned by an untrusted user: {directory}")
        if details.st_mode & 0o022 and not details.st_mode & stat.S_ISVTX:
            raise ValueError(f"index output ancestor is writable by another user: {directory}")


def open_or_create_output(output: Path) -> tuple[int, int]:
    output = output.absolute()
    require_trusted_directory_chain(output.parent)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_fd = os.open(output.parent, directory_flags)
    try:
        try:
            os.mkdir(output.name, 0o755, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except FileExistsError:
            pass
        output_fd = os.open(output.name, directory_flags, dir_fd=parent_fd)
        require_private_directory(output_fd, "index output")
        return parent_fd, output_fd
    except Exception:
        os.close(parent_fd)
        raise


def create_staging_directory(output: Path) -> Path:
    output = output.absolute()
    parent_fd, output_fd = open_or_create_output(output)
    try:
        for _ in range(10):
            name = f".staging-{secrets.token_hex(16)}"
            try:
                os.mkdir(name, 0o700, dir_fd=output_fd)
                os.fsync(output_fd)
                return output / name
            except FileExistsError:
                continue
        raise FileExistsError("could not create a unique index staging directory")
    finally:
        os.close(output_fd)
        os.close(parent_fd)


def publish_index(staging: Path, output: Path) -> None:
    output = output.absolute()
    staging = staging.absolute()
    if staging.parent != output or not staging.name.startswith(".staging-"):
        raise ValueError("index staging directory must be a hidden child of the output directory")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    parent_fd, output_fd = open_or_create_output(output)
    versions_fd = staging_fd = lock_fd = version_tmp_fd = None
    artifact_fds: dict[str, int] = {}
    pointer_tmp_name = None
    version_tmp_name = None
    version_artifacts: list[str] = []
    try:
        output_stat = require_private_directory(output_fd, "index output")
        path_stat = os.stat(output, follow_symlinks=False)
        if (output_stat.st_dev, output_stat.st_ino) != (path_stat.st_dev, path_stat.st_ino):
            raise ValueError("index output changed while publication was starting")
        staging_fd = os.open(staging.name, directory_flags, dir_fd=output_fd)
        staging_stat = require_private_directory(staging_fd, "index staging entry")
        try:
            os.mkdir("versions", dir_fd=output_fd)
        except FileExistsError:
            pass
        versions_fd = os.open("versions", directory_flags, dir_fd=output_fd)
        require_private_directory(versions_fd, "index versions entry")
        os.fsync(output_fd)

        artifact_names = sorted(os.listdir(staging_fd))
        for name in artifact_names:
            artifact_fd = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=staging_fd)
            artifact_stat = os.fstat(artifact_fd)
            if not stat.S_ISREG(artifact_stat.st_mode) or artifact_stat.st_nlink != 1:
                os.close(artifact_fd)
                raise ValueError(f"index staging contains unsupported artifact: {name}")
            os.fsync(artifact_fd)
            artifact_fds[name] = artifact_fd
        if "metadata.json" not in artifact_fds:
            raise ValueError("index staging is missing metadata.json")
        with os.fdopen(os.dup(artifact_fds["metadata.json"]), "r") as handle:
            metadata = json.load(handle)
        created_at = metadata.get("created_at")
        embeddings_sha256 = metadata.get("embeddings_sha256")
        if not isinstance(created_at, str) or not isinstance(embeddings_sha256, str):
            raise ValueError("index publication metadata is missing its creation time or embedding hash")
        try:
            created = datetime.fromisoformat(created_at)
        except ValueError as exc:
            raise ValueError("index publication created_at must be an ISO-8601 timestamp") from exc
        if created.tzinfo is None or not SHA256_RE.fullmatch(embeddings_sha256):
            raise ValueError("index publication metadata must use a timezone and a SHA-256 embedding hash")
        build_id = created.astimezone(UTC).strftime("%Y%m%dT%H%M%S_%f%z") + "-" + embeddings_sha256[:12]
        os.fsync(staging_fd)

        lock_fd = os.open(
            ".build.lock",
            os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=output_fd,
        )
        lock_stat = os.fstat(lock_fd)
        if not stat.S_ISREG(lock_stat.st_mode) or lock_stat.st_nlink != 1:
            raise ValueError("index build lock must be a singly linked regular file")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        current_staging = os.stat(staging.name, dir_fd=output_fd, follow_symlinks=False)
        if (current_staging.st_dev, current_staging.st_ino) != (staging_stat.st_dev, staging_stat.st_ino):
            raise ValueError("index staging pathname changed before publication")
        try:
            os.stat(build_id, dir_fd=versions_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"index version already exists: versions/{build_id}")

        for _ in range(10):
            version_tmp_name = f".{build_id}.tmp-{secrets.token_hex(16)}"
            try:
                os.mkdir(version_tmp_name, 0o700, dir_fd=versions_fd)
                break
            except FileExistsError:
                continue
        else:
            raise FileExistsError("could not create a unique private version directory")
        version_tmp_fd = os.open(version_tmp_name, directory_flags, dir_fd=versions_fd)
        require_private_directory(version_tmp_fd, "private version staging entry")
        for name, source_fd in artifact_fds.items():
            destination_fd = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=version_tmp_fd,
            )
            version_artifacts.append(name)
            try:
                before = os.fstat(source_fd)
                copy_descriptor(source_fd, destination_fd)
                after = os.fstat(source_fd)
                if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                ):
                    raise ValueError(f"index staging artifact changed during publication: {name}")
            finally:
                os.close(destination_fd)
        os.fsync(version_tmp_fd)
        os.rename(version_tmp_name, build_id, src_dir_fd=versions_fd, dst_dir_fd=versions_fd)
        version_tmp_name = None
        os.fsync(versions_fd)
        try:
            current_staging = os.stat(staging.name, dir_fd=output_fd, follow_symlinks=False)
        except FileNotFoundError:
            current_staging = None
        if current_staging is not None and (current_staging.st_dev, current_staging.st_ino) == (
            staging_stat.st_dev,
            staging_stat.st_ino,
        ):
            for name in artifact_names:
                os.unlink(name, dir_fd=staging_fd)
            os.rmdir(staging.name, dir_fd=output_fd)
        pointer_payload = f"versions/{build_id}\n".encode()
        for _ in range(10):
            pointer_tmp_name = f".CURRENT.{secrets.token_hex(16)}.tmp"
            try:
                pointer_fd = os.open(
                    pointer_tmp_name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=output_fd,
                )
                break
            except FileExistsError:
                continue
        else:
            raise FileExistsError("could not create a unique temporary CURRENT pointer")
        try:
            offset = 0
            while offset < len(pointer_payload):
                written = os.write(pointer_fd, pointer_payload[offset:])
                if written == 0:
                    raise OSError("short write while creating CURRENT pointer")
                offset += written
            pointer_stat = os.fstat(pointer_fd)
            if not stat.S_ISREG(pointer_stat.st_mode) or pointer_stat.st_nlink != 1:
                raise ValueError("temporary CURRENT pointer must be a singly linked regular file")
            os.fsync(pointer_fd)
        finally:
            os.close(pointer_fd)
        os.replace(pointer_tmp_name, "CURRENT", src_dir_fd=output_fd, dst_dir_fd=output_fd)
        pointer_tmp_name = None
        os.fsync(output_fd)
        visible_output = os.stat(output.name, dir_fd=parent_fd, follow_symlinks=False)
        if (visible_output.st_dev, visible_output.st_ino) != (output_stat.st_dev, output_stat.st_ino):
            raise ValueError("index output pathname changed during publication")
    finally:
        if pointer_tmp_name is not None:
            try:
                os.unlink(pointer_tmp_name, dir_fd=output_fd)
            except FileNotFoundError:
                pass
        if version_tmp_name is not None and versions_fd is not None:
            if version_tmp_fd is not None:
                for name in version_artifacts:
                    try:
                        os.unlink(name, dir_fd=version_tmp_fd)
                    except FileNotFoundError:
                        pass
            try:
                os.rmdir(version_tmp_name, dir_fd=versions_fd)
            except FileNotFoundError:
                pass
        for descriptor in artifact_fds.values():
            os.close(descriptor)
        for descriptor in (version_tmp_fd, lock_fd, versions_fd, staging_fd, output_fd, parent_fd):
            if descriptor is not None:
                os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed the AI That Works corpus")
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks.jsonl"))
    parser.add_argument("--corpus-provenance", type=Path)
    parser.add_argument("--model", default="google/embeddinggemma-2")
    parser.add_argument(
        "--model-revision", help="Hugging Face branch, tag, or commit (resolved to a commit when online)"
    )
    parser.add_argument("--output", type=Path, default=Path("index"))
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--source-repo", type=Path, default=Path("source/ai-that-works"))
    args = parser.parse_args()

    chunks = list(iter_jsonl(args.chunks))
    if not chunks:
        raise SystemExit("no chunks found")
    validate_chunks(chunks)
    source_repo = args.source_repo.resolve()
    commit = git_output(source_repo, "rev-parse", "HEAD")
    if git_output(source_repo, "status", "--porcelain"):
        raise SystemExit(f"source repository is dirty; commit or clean changes before indexing: {source_repo}")
    provenance_path = args.corpus_provenance or args.chunks.with_suffix(".provenance.json")
    if not provenance_path.is_file():
        raise SystemExit(f"missing corpus provenance sidecar: {provenance_path}; rebuild the corpus")
    corpus_provenance = json.loads(provenance_path.read_text())
    try:
        validate_corpus_provenance(corpus_provenance)
    except ValueError as exc:
        raise SystemExit(f"invalid corpus provenance; rebuild the corpus: {exc}") from exc
    input_chunks_sha256 = sha256_file(args.chunks)
    if corpus_provenance.get("chunks_sha256") != input_chunks_sha256:
        raise SystemExit("corpus provenance does not match the input chunks; rebuild the corpus")
    if corpus_provenance.get("chunks") != len(chunks):
        raise SystemExit("corpus provenance chunk count does not match the input chunks")
    if corpus_provenance.get("repository_commit") != commit or corpus_provenance.get("repository_clean") is not True:
        raise SystemExit("corpus provenance does not match the current clean source repository")
    corpus_module = Path(__file__).with_name("corpus.py")
    build_corpus_script = Path(__file__).parents[2] / "scripts" / "build_corpus.py"
    if corpus_provenance["corpus_module_sha256"] != sha256_file(corpus_module):
        raise SystemExit("corpus provenance names different chunker code; rebuild the corpus")
    if (
        not build_corpus_script.is_file()
        or corpus_provenance["build_corpus_sha256"] != sha256_file(build_corpus_script)
    ):
        raise SystemExit("corpus provenance names a different build_corpus script; rebuild the corpus")
    repository_chunk_commits = {
        item.get("repository_commit")
        for item in chunks
        if str(item.get("source_type", "")).startswith("repository_") or item.get("source_type") == "diagram_caption"
    }
    if repository_chunk_commits and repository_chunk_commits != {commit}:
        raise SystemExit("repository-backed chunks do not match corpus/source repository provenance")
    model_revision = resolve_model_revision(args.model, args.model_revision)
    # Keep remote model code disabled by default. EmbeddingGemma 2 and the
    # documented smoke-test models are supported directly by pinned deps.
    model = SentenceTransformer(args.model, revision=model_revision, trust_remote_code=False)
    vectors = encode(model, [item["text"] for item in chunks], args.batch_size, "document")
    output = args.output.absolute()
    staging = create_staging_directory(output)
    try:
        embeddings_path = staging / "embeddings.npy"
        chunks_path = staging / "chunks.jsonl"
        np.save(embeddings_path, vectors)
        shutil.copyfile(args.chunks, chunks_path)
        metadata = {
            "schema_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "model": args.model,
            "model_revision": model_revision,
            "dimensions": int(vectors.shape[1]),
            "dtype": str(vectors.dtype),
            "chunks": len(chunks),
            "normalized": True,
            "query_prompt": bool("query" in (getattr(model, "prompts", {}) or {})),
            "document_prompt": bool("document" in (getattr(model, "prompts", {}) or {})),
            "repository_commit": commit,
            "source_repository_dirty": False,
            "input_chunks_sha256": input_chunks_sha256,
            "corpus_provenance_sha256": sha256_file(provenance_path),
            "chunks_sha256": sha256_file(chunks_path),
            "embeddings_sha256": sha256_file(embeddings_path),
            "batch_size": args.batch_size,
            "runtime_versions": {
                "numpy": version("numpy"),
                "sentence-transformers": version("sentence-transformers"),
                "torch": version("torch"),
            },
        }
        shutil.copy2(provenance_path, staging / "corpus_provenance.json")
        (staging / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        load_index(staging)
        publish_index(staging, output)
    finally:
        # publish_index removes the original staging directory only after
        # descriptor-relative publication and an inode-identity check. Avoid
        # path-based cleanup here: a hostile writer could substitute the name.
        pass
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
