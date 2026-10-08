from __future__ import annotations

DEFAULT_ALLOWED_MODELS = frozenset(
    {
        "google/embeddinggemma-2",
        "sentence-transformers/all-MiniLM-L6-v2",
    }
)


def validate_model_id(model_id: object, *, allow_unlisted: bool = False) -> str:
    model_id = str(model_id)
    if not allow_unlisted and model_id not in DEFAULT_ALLOWED_MODELS:
        allowed = ", ".join(sorted(DEFAULT_ALLOWED_MODELS))
        raise ValueError(
            f"index requests unlisted model {model_id!r}; allowed models: {allowed}. "
            "Use --allow-unlisted-model only after reviewing the model ID and pinned revision."
        )
    return model_id
