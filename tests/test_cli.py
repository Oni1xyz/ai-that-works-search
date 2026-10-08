import argparse

import pytest

from aitw_search.cli import nonnegative_int, positive_int


def test_positive_int_rejects_zero():
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int("0")


def test_nonnegative_int_accepts_zero_and_rejects_negative():
    assert nonnegative_int("0") == 0
    with pytest.raises(argparse.ArgumentTypeError):
        nonnegative_int("-1")
