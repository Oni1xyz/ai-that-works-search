from __future__ import annotations

import argparse
import re

CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|[@-_])")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be at least 0")
    return parsed


def terminal_safe(value: object) -> str:
    return CONTROL_CHARACTERS.sub("", ANSI_ESCAPE.sub("", str(value)))


def terminal_safe_line(value: object) -> str:
    return " ".join(terminal_safe(value).split())
