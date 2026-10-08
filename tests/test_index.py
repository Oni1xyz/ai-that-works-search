import json
import os

import numpy as np
import pytest

from aitw_search import build_index as build_index_module
from aitw_search.build_index import create_staging_directory, encode, publish_index, resolve_model_revision
from aitw_search.index import load_index, resolve_index_dir, sha256_file

VALID_CHUNK = {
    "id": "one",
    "title": "Test",
    "source_type": "repository_note",
    "text": "hello",
    "source_path": "episode/note.md",
    "repository_commit": "c" * 40,
    "repository_url": "https://github.com/ai-that-works/ai-that-works",
}


def test_encode_renormalizes_low_precision_model_output():
    class Model:
        prompts = {}

        def encode(self, texts, **kwargs):
            return np.asarray([[3.01, 4.0], [0.0, 2.02]], dtype=np.float32)

    embeddings = encode(Model(), ["one", "two"], 2, "document")
    assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-6)


def write_index(tmp_path, chunks, vectors, dimensions=None):
    chunks_path = tmp_path / "chunks.jsonl"
    embeddings_path = tmp_path / "embeddings.npy"
    with chunks_path.open("w") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk) + "\n")
    np.save(embeddings_path, np.asarray(vectors, dtype=np.float32))
    repository_commit = "c" * 40
    provenance_path = tmp_path / "corpus_provenance.json"
    provenance_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "created_at": "2026-10-08T13:00:00+00:00",
                "repository_commit": repository_commit,
                "repository_clean": True,
                "manifest_sha256": "1" * 64,
                "youtube_snapshot_sha256": "2" * 64,
                "diagram_captions_sha256": "3" * 64,
                "diagram_caption_provenance_sha256": "4" * 64,
                "build_corpus_sha256": "5" * 64,
                "corpus_module_sha256": "6" * 64,
                "transcript_sha256": {"video.json": "7" * 64},
                "chunks": len(chunks),
                "chunks_sha256": sha256_file(chunks_path),
            }
        )
    )
    metadata = {
        "schema_version": 1,
        "created_at": "2026-10-08T13:00:00+00:00",
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "model_revision": "a" * 40,
        "dimensions": dimensions if dimensions is not None else len(vectors[0]),
        "dtype": "float32",
        "chunks": len(chunks),
        "normalized": True,
        "query_prompt": False,
        "document_prompt": False,
        "repository_commit": repository_commit,
        "input_chunks_sha256": sha256_file(chunks_path),
        "corpus_provenance_sha256": sha256_file(provenance_path),
        "chunks_sha256": sha256_file(chunks_path),
        "embeddings_sha256": sha256_file(embeddings_path),
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))


def test_load_index_validates_counts(tmp_path):
    write_index(tmp_path, [VALID_CHUNK], [[1.0, 0.0], [0.0, 1.0]], dimensions=2)
    with pytest.raises(ValueError, match="count mismatch"):
        load_index(tmp_path)


def test_load_index_rejects_dimension_mismatch(tmp_path):
    write_index(tmp_path, [VALID_CHUNK], [[1.0, 0.0]], dimensions=3)
    with pytest.raises(ValueError, match="dimension mismatch"):
        load_index(tmp_path)


def test_load_index_reads_complete_artifact(tmp_path):
    write_index(tmp_path, [VALID_CHUNK], [[1.0, 0.0]], dimensions=2)
    metadata, chunks, vectors = load_index(tmp_path)
    assert metadata["dimensions"] == 2
    assert chunks[0]["id"] == "one"
    assert vectors.shape == (1, 2)


def test_load_index_rejects_incomplete_chunk_schema(tmp_path):
    write_index(tmp_path, [{"id": "one"}], [[1.0, 0.0]], dimensions=2)
    with pytest.raises(ValueError, match="missing required fields"):
        load_index(tmp_path)


def test_load_index_rejects_duplicate_chunk_ids(tmp_path):
    write_index(tmp_path, [VALID_CHUNK, VALID_CHUNK], [[1.0, 0.0], [0.0, 1.0]], dimensions=2)
    with pytest.raises(ValueError, match="duplicate chunk ID"):
        load_index(tmp_path)


def test_publish_index_replaces_current_pointer_without_removing_previous_version(tmp_path):
    output = tmp_path / "index"
    old_version = output / "versions" / "old"
    old_version.mkdir(parents=True)
    (old_version / "old.txt").write_text("old")
    (output / "CURRENT").write_text("versions/old\n")
    staging = output / ".staging-test"
    staging.mkdir()
    (staging / "new.txt").write_text("new")
    (staging / "metadata.json").write_text(
        json.dumps({"created_at": "2026-10-08T12:00:00+00:00", "embeddings_sha256": "b" * 64})
    )

    publish_index(staging, output)

    current = resolve_index_dir(output)
    assert not staging.exists()
    assert (old_version / "old.txt").read_text() == "old"
    assert (current / "new.txt").read_text() == "new"
    assert current != old_version


def test_publish_index_rejects_symlinked_versions_without_writing_outside(tmp_path):
    output = tmp_path / "index"
    output.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (output / "versions").symlink_to(outside, target_is_directory=True)
    staging = output / ".staging-hostile-versions"
    staging.mkdir()
    (staging / "artifact.txt").write_text("artifact")
    (staging / "metadata.json").write_text(
        json.dumps({"created_at": "2026-10-08T12:00:00+00:00", "embeddings_sha256": "b" * 64})
    )

    with pytest.raises(OSError):
        publish_index(staging, output)
    assert list(outside.iterdir()) == []


def test_publish_index_does_not_follow_precreated_pointer_temp_symlink(tmp_path, monkeypatch):
    output = tmp_path / "index"
    output.mkdir()
    staging = output / ".staging-hostile-pointer"
    staging.mkdir()
    (staging / "artifact.txt").write_text("artifact")
    (staging / "metadata.json").write_text(
        json.dumps({"created_at": "2026-10-08T12:00:00+00:00", "embeddings_sha256": "b" * 64})
    )
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("DO-NOT-OVERWRITE")
    monkeypatch.setattr(build_index_module.secrets, "token_hex", lambda _: "predictable")
    (output / ".CURRENT.predictable.tmp").symlink_to(sentinel)

    with pytest.raises(FileExistsError, match="unique temporary"):
        publish_index(staging, output)
    assert sentinel.read_text() == "DO-NOT-OVERWRITE"


def test_publish_index_rejects_path_components_in_creation_time(tmp_path):
    output = tmp_path / "index"
    output.mkdir()
    staging = output / ".staging-hostile-build-id"
    staging.mkdir()
    (staging / "metadata.json").write_text(
        json.dumps({"created_at": "../../outside", "embeddings_sha256": "b" * 64})
    )

    with pytest.raises(ValueError, match="ISO-8601"):
        publish_index(staging, output)
    assert not (tmp_path / "outside").exists()


def test_publish_index_uses_retained_staging_descriptors_during_name_substitution(tmp_path, monkeypatch):
    output = tmp_path / "index"
    output.mkdir()
    staging = output / ".staging-race"
    staging.mkdir()
    (staging / "validated.txt").write_text("validated")
    (staging / "metadata.json").write_text(
        json.dumps({"created_at": "2026-10-08T12:00:00+00:00", "embeddings_sha256": "b" * 64})
    )
    real_copy = build_index_module.copy_descriptor
    substituted = False

    def substitute_then_copy(source, destination):
        nonlocal substituted
        if not substituted:
            substituted = True
            staging.rename(output / ".staging-original-renamed")
            staging.mkdir()
            (staging / "attacker.txt").write_text("attacker")
        return real_copy(source, destination)

    monkeypatch.setattr(build_index_module, "copy_descriptor", substitute_then_copy)
    publish_index(staging, output)
    current = resolve_index_dir(output)
    assert (current / "validated.txt").read_text() == "validated"
    assert not (current / "attacker.txt").exists()


@pytest.mark.parametrize(
    "artifact", ["metadata.json", "chunks.jsonl", "embeddings.npy", "corpus_provenance.json"]
)
def test_load_index_rejects_symlinked_artifact(tmp_path, artifact):
    write_index(tmp_path, [VALID_CHUNK], [[1.0, 0.0]], dimensions=2)
    path = tmp_path / artifact
    outside = tmp_path.parent / f"outside-{artifact}"
    path.rename(outside)
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="safely open"):
        load_index(tmp_path)


@pytest.mark.parametrize(
    "artifact", ["metadata.json", "chunks.jsonl", "embeddings.npy", "corpus_provenance.json"]
)
def test_load_index_rejects_fifo_artifact_without_blocking(tmp_path, artifact):
    write_index(tmp_path, [VALID_CHUNK], [[1.0, 0.0]], dimensions=2)
    path = tmp_path / artifact
    path.unlink()
    os.mkfifo(path)
    with pytest.raises(ValueError, match="regular file"):
        load_index(tmp_path)


def test_load_index_rejects_fifo_current_without_blocking(tmp_path):
    os.mkfifo(tmp_path / "CURRENT")
    with pytest.raises(ValueError, match="regular file"):
        load_index(tmp_path)


def test_load_index_rejects_attacker_controlled_youtube_url(tmp_path):
    chunk = {
        **VALID_CHUNK,
        "source_type": "youtube_transcript",
        "youtube_id": "dQw4w9WgXcQ",
        "youtube_url": "https://evil.example/phish",
        "source_path": "youtube/video.json",
    }
    write_index(tmp_path, [chunk], [[1.0, 0.0]], dimensions=2)
    with pytest.raises(ValueError, match="youtube_url"):
        load_index(tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_type", "unknown", "unsupported source_type"),
        ("source_path", "../escape", "normalized relative"),
        ("repository_commit", "d" * 40, "does not match"),
        ("repository_url", "https://evil.example/repo", "approved repository"),
    ],
)
def test_load_index_rejects_untrusted_source_metadata(tmp_path, field, value, message):
    chunk = {**VALID_CHUNK, field: value}
    write_index(tmp_path, [chunk], [[1.0, 0.0]], dimensions=2)
    with pytest.raises(ValueError, match=message):
        load_index(tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("episode_guid", {}, "episode_guid"),
        ("start_s", {}, "start_s"),
        ("start_s", float("nan"), "start_s"),
        ("end_s", float("inf"), "end_s"),
        ("start_s", -1, "start_s"),
    ],
)
def test_load_index_rejects_malformed_retrieval_metadata(tmp_path, field, value, message):
    chunk = {**VALID_CHUNK, field: value}
    write_index(tmp_path, [chunk], [[1.0, 0.0]], dimensions=2)
    with pytest.raises(ValueError, match=message):
        load_index(tmp_path)


def test_load_index_rejects_reversed_timestamps(tmp_path):
    chunk = {**VALID_CHUNK, "start_s": 2.0, "end_s": 1.0}
    write_index(tmp_path, [chunk], [[1.0, 0.0]], dimensions=2)
    with pytest.raises(ValueError, match="end_s"):
        load_index(tmp_path)


def test_staging_directory_is_created_on_output_filesystem(tmp_path):
    output = tmp_path / "mounted-index"
    staging = create_staging_directory(output)
    assert staging.parent == output
    assert staging.name.startswith(".staging-")
    assert os.stat(staging).st_dev == os.stat(output).st_dev


def test_create_staging_rejects_output_symlink(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    output = tmp_path / "index"
    output.symlink_to(target, target_is_directory=True)
    with pytest.raises(OSError):
        create_staging_directory(output)


def test_publish_detects_output_entry_substitution(tmp_path, monkeypatch):
    output = tmp_path / "index"
    output.mkdir()
    staging = output / ".staging-output-race"
    staging.mkdir()
    (staging / "metadata.json").write_text(
        json.dumps({"created_at": "2026-10-08T12:00:00+00:00", "embeddings_sha256": "b" * 64})
    )
    original = tmp_path / "original-output"
    real_open = os.open
    substituted = False

    def substitute_output(path, flags, *args, **kwargs):
        nonlocal substituted
        if path == staging.name and kwargs.get("dir_fd") is not None and not substituted:
            substituted = True
            output.rename(original)
            output.mkdir()
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(build_index_module.os, "open", substitute_output)
    with pytest.raises(ValueError, match="output pathname changed"):
        publish_index(staging, output)
    assert not (output / "CURRENT").exists()


def test_publish_failure_preserves_previous_current_pointer(tmp_path, monkeypatch):
    output = tmp_path / "index"
    old_version = output / "versions" / "old"
    old_version.mkdir(parents=True)
    (output / "CURRENT").write_text("versions/old\n")
    staging = output / ".staging-test"
    staging.mkdir()
    (staging / "metadata.json").write_text(
        json.dumps({"created_at": "2026-10-08T13:00:00+00:00", "embeddings_sha256": "e" * 64})
    )
    real_replace = os.replace

    def fail_pointer_replace(source, destination, **kwargs):
        if destination == "CURRENT":
            raise OSError("injected pointer failure")
        return real_replace(source, destination, **kwargs)

    monkeypatch.setattr("aitw_search.build_index.os.replace", fail_pointer_replace)
    with pytest.raises(OSError, match="injected"):
        publish_index(staging, output)

    assert (output / "CURRENT").read_text() == "versions/old\n"
    assert resolve_index_dir(output) == old_version


def test_resolve_index_rejects_pointer_escape(tmp_path):
    (tmp_path / "CURRENT").write_text("../outside\n")
    with pytest.raises(ValueError, match="invalid index CURRENT"):
        resolve_index_dir(tmp_path)


def test_load_index_rejects_checksum_mismatch(tmp_path):
    write_index(tmp_path, [VALID_CHUNK], [[1.0, 0.0]], dimensions=2)
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(chunks_path.read_text() + "\n")

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_index(tmp_path)


def test_load_index_rejects_missing_schema_version(tmp_path):
    write_index(tmp_path, [VALID_CHUNK], [[1.0, 0.0]])
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    del metadata["schema_version"]
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="schema_version"):
        load_index(tmp_path)


@pytest.mark.parametrize(
    ("vectors", "message"),
    [
        ([[float("nan"), 0.0]], "NaN or infinite"),
        ([[float("inf"), 0.0]], "NaN or infinite"),
        ([[0.0, 0.0]], "zero-norm"),
        ([[2.0, 0.0]], "not unit-normalized"),
        ([[1.00002, 0.0]], "not unit-normalized"),
    ],
)
def test_load_index_rejects_invalid_vectors(tmp_path, vectors, message):
    write_index(tmp_path, [VALID_CHUNK], vectors)
    with pytest.raises(ValueError, match=message):
        load_index(tmp_path)


def test_offline_model_revision_requires_full_commit(monkeypatch):
    def offline(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr("aitw_search.build_index.model_info", offline)
    with pytest.raises(SystemExit, match="40-character"):
        resolve_model_revision("model", "main")
    commit = "d" * 40
    assert resolve_model_revision("model", commit) == commit


def test_online_model_revision_must_resolve_to_commit(monkeypatch):
    class Info:
        sha = "main"

    monkeypatch.setattr("aitw_search.build_index.model_info", lambda *args, **kwargs: Info())
    with pytest.raises(SystemExit, match="not an immutable"):
        resolve_model_revision("model", None)


@pytest.mark.parametrize("target", ["arbitrary", "versions/build/nested", "/tmp/outside"])
def test_resolve_index_requires_exact_version_layout(tmp_path, target):
    (tmp_path / "CURRENT").write_text(target + "\n")
    with pytest.raises(ValueError, match="invalid index CURRENT"):
        resolve_index_dir(tmp_path)


def test_resolve_index_rejects_symlinked_version(tmp_path):
    versions = tmp_path / "versions"
    target = versions / "real"
    target.mkdir(parents=True)
    (versions / "link").symlink_to(target, target_is_directory=True)
    (tmp_path / "CURRENT").write_text("versions/link\n")
    with pytest.raises(ValueError, match="symlink"):
        resolve_index_dir(tmp_path)


def test_resolve_index_rejects_symlinked_current_without_disclosing_target(tmp_path):
    sentinel = "TOP-SECRET-SENTINEL"
    secret = tmp_path / "secret.txt"
    secret.write_text(sentinel)
    (tmp_path / "CURRENT").symlink_to(secret)
    with pytest.raises(ValueError) as error:
        resolve_index_dir(tmp_path)
    assert sentinel not in str(error.value)


def test_resolve_index_rejects_oversized_current(tmp_path):
    (tmp_path / "CURRENT").write_text("x" * 257)
    with pytest.raises(ValueError, match="singly linked"):
        resolve_index_dir(tmp_path)


def test_resolve_index_rejects_hardlinked_current_without_disclosing_target(tmp_path):
    sentinel = "HARDLINK-SECRET-SENTINEL"
    secret = tmp_path / "secret-hardlink.txt"
    secret.write_text(sentinel)
    os.link(secret, tmp_path / "CURRENT")
    with pytest.raises(ValueError) as error:
        resolve_index_dir(tmp_path)
    assert sentinel not in str(error.value)


def test_publish_pre_pointer_fsync_failure_preserves_current(tmp_path, monkeypatch):
    output = tmp_path / "index"
    old_version = output / "versions" / "old"
    old_version.mkdir(parents=True)
    (output / "CURRENT").write_text("versions/old\n")
    staging = output / ".staging-fsync"
    staging.mkdir()
    (staging / "metadata.json").write_text(
        json.dumps({"created_at": "2026-10-08T14:00:00+00:00", "embeddings_sha256": "f" * 64})
    )
    real_fsync = os.fsync
    versions_stat = os.stat(output / "versions")

    def fail_versions_fsync(descriptor):
        descriptor_stat = os.fstat(descriptor)
        if (descriptor_stat.st_dev, descriptor_stat.st_ino) == (versions_stat.st_dev, versions_stat.st_ino):
            raise OSError("injected versions fsync failure")
        return real_fsync(descriptor)

    monkeypatch.setattr("aitw_search.build_index.os.fsync", fail_versions_fsync)
    with pytest.raises(OSError, match="injected"):
        publish_index(staging, output)
    assert (output / "CURRENT").read_text() == "versions/old\n"


@pytest.mark.parametrize("revision", ["main", "v1.0", "abc123", "A" * 40, 123])
def test_load_index_rejects_mutable_or_malformed_model_revision(tmp_path, revision):
    write_index(tmp_path, [VALID_CHUNK], [[1.0, 0.0]], dimensions=2)
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["model_revision"] = revision
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="model_revision"):
        load_index(tmp_path)


def test_load_index_rejects_incomplete_corpus_provenance(tmp_path):
    write_index(tmp_path, [VALID_CHUNK], [[1.0, 0.0]], dimensions=2)
    provenance_path = tmp_path / "corpus_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance.pop("manifest_sha256")
    provenance_path.write_text(json.dumps(provenance))
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["corpus_provenance_sha256"] = sha256_file(provenance_path)
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="missing required fields"):
        load_index(tmp_path)


@pytest.mark.parametrize("field", ["query_prompt", "document_prompt"])
def test_load_index_requires_prompt_semantics(tmp_path, field):
    write_index(tmp_path, [VALID_CHUNK], [[1.0, 0.0]], dimensions=2)
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.pop(field)
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=field):
        load_index(tmp_path)


def test_load_index_rejects_mismatched_input_chunks_hash(tmp_path):
    write_index(tmp_path, [VALID_CHUNK], [[1.0, 0.0]], dimensions=2)
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["input_chunks_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="input_chunks_sha256"):
        load_index(tmp_path)


def test_load_index_rejects_malformed_corpus_provenance_hash(tmp_path):
    write_index(tmp_path, [VALID_CHUNK], [[1.0, 0.0]], dimensions=2)
    provenance_path = tmp_path / "corpus_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["manifest_sha256"] = "not-a-hash"
    provenance_path.write_text(json.dumps(provenance))
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["corpus_provenance_sha256"] = sha256_file(provenance_path)
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="manifest_sha256"):
        load_index(tmp_path)


@pytest.mark.parametrize("field", ["id", "title", "source_type", "text"])
def test_load_index_rejects_empty_required_chunk_strings(tmp_path, field):
    chunk = dict(VALID_CHUNK)
    chunk[field] = "  "
    write_index(tmp_path, [chunk], [[1.0, 0.0]], dimensions=2)
    with pytest.raises(ValueError, match=field):
        load_index(tmp_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chunks", True),
        ("dimensions", "2"),
        ("model", ""),
        ("query_prompt", "yes"),
        ("created_at", "not-a-timestamp"),
    ],
)
def test_load_index_rejects_malformed_metadata_scalars(tmp_path, field, value):
    write_index(tmp_path, [VALID_CHUNK], [[1.0, 0.0]], dimensions=2)
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=field):
        load_index(tmp_path)


@pytest.mark.parametrize(("field", "value"), [("dimensions", 8193), ("chunks", 100_001)])
def test_load_index_rejects_metadata_allocation_limits(tmp_path, field, value):
    write_index(tmp_path, [VALID_CHUNK], [[1.0, 0.0]], dimensions=2)
    metadata_path = tmp_path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="supported limit"):
        load_index(tmp_path)


def test_load_index_rejects_embedding_larger_than_declared_shape(tmp_path):
    write_index(tmp_path, [VALID_CHUNK], [[1.0, 0.0]], dimensions=2)
    with (tmp_path / "embeddings.npy").open("r+b") as handle:
        handle.truncate(64 * 1024 + 9)
    with pytest.raises(ValueError, match="size limit"):
        load_index(tmp_path)


@pytest.mark.parametrize(
    ("artifact", "size"),
    [("chunks.jsonl", 64 * 1024 * 1024 + 1), ("corpus_provenance.json", 4 * 1024 * 1024 + 1)],
)
def test_load_index_rejects_oversized_text_artifacts_before_reading(tmp_path, artifact, size):
    write_index(tmp_path, [VALID_CHUNK], [[1.0, 0.0]], dimensions=2)
    with (tmp_path / artifact).open("r+b") as handle:
        handle.truncate(size)
    with pytest.raises(ValueError, match="size limit"):
        load_index(tmp_path)


def test_create_staging_rejects_nonsticky_world_writable_ancestor(tmp_path):
    ancestor = tmp_path / "shared"
    ancestor.mkdir(mode=0o777)
    ancestor.chmod(0o777)
    with pytest.raises(ValueError, match="writable by another user"):
        create_staging_directory(ancestor / "index")
