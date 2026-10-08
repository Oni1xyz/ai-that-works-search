from aitw_search.rag import build_prompt


def test_build_prompt_has_numbered_sources_and_guardrails():
    results = [
        {
            "rank": 1,
            "title": "Episode One",
            "source_type": "youtube_transcript",
            "source": "https://youtube.test/watch?v=one&t=10s",
            "text": "Agents need explicit verification loops.",
        },
        {
            "rank": 2,
            "title": "Episode Two",
            "source_type": "repository_note",
            "source": "notes/two.md",
            "text": "Memory should be curated over time.",
        },
    ]
    prompt = build_prompt("How should agents work?", results, 10_000)
    assert "[1] Episode One" in prompt
    assert "[2] Episode Two" in prompt
    assert "Use only the retrieved context" in prompt
    assert "How should agents work?" in prompt
