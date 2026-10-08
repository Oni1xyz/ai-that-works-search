import pytest

from aitw_search.models import validate_model_id


def test_default_model_allowlist():
    assert validate_model_id("google/embeddinggemma-2") == "google/embeddinggemma-2"
    with pytest.raises(ValueError, match="unlisted model"):
        validate_model_id("unknown/model")


def test_unlisted_model_requires_explicit_opt_in():
    assert validate_model_id("unknown/model", allow_unlisted=True) == "unknown/model"
