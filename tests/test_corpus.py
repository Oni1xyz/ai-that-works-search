from aitw_search.corpus import chunk_id, chunk_text, transcript_chunks


def test_chunk_text_is_bounded_and_overlapping():
    text = "\n\n".join(["alpha " * 80, "beta " * 80, "gamma " * 80])
    chunks = chunk_text(text, max_chars=500, overlap_chars=50)
    assert len(chunks) >= 3
    assert all(len(chunk) <= 600 for chunk in chunks)
    assert any("alpha" in chunk for chunk in chunks)
    assert any("gamma" in chunk for chunk in chunks)


def test_transcript_chunks_preserve_time_ranges():
    segments = [
        {"start": 0.0, "duration": 2.0, "text": "first phrase"},
        {"start": 2.0, "duration": 3.0, "text": "second phrase"},
        {"start": 5.0, "duration": 1.0, "text": "third phrase"},
    ]
    chunks = transcript_chunks({"segments": segments}, max_chars=28)
    assert chunks[0]["start_s"] == 0.0
    assert chunks[-1]["end_s"] == 6.0
    assert "third phrase" in chunks[-1]["text"]


def test_chunk_id_is_deterministic():
    metadata = {"episode": 1, "source": "x"}
    assert chunk_id(metadata, "hello") == chunk_id(metadata, "hello")
    assert chunk_id(metadata, "hello") != chunk_id(metadata, "world")
