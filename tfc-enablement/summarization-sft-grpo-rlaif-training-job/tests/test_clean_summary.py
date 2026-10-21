"""Tests for the generation-hygiene cleaner (src/data_utils.clean_summary)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import data_utils as D  # noqa: E402


def test_strips_think_block():
    assert "think" not in D.clean_summary("<think>reasoning here</think>The cat sat.").lower()
    assert D.clean_summary("<think>x</think>The cat sat.") == "The cat sat."


def test_cuts_leaked_chat_turn():
    txt = "A good summary.\nuser\nWrite another one\nassistant\nSomething else"
    assert D.clean_summary(txt) == "A good summary."


def test_cuts_echoed_instruction():
    txt = "The real summary. Write a concise TL;DR summary of the following post."
    assert D.clean_summary(txt) == "The real summary."


def test_dedupes_repeated_lines():
    txt = "I was a jerk.\nI was a jerk.\nI was a jerk."
    assert D.clean_summary(txt) == "I was a jerk."


def test_clean_passthrough_and_empty():
    assert D.clean_summary("A clean one-line summary.") == "A clean one-line summary."
    assert D.clean_summary("") == ""
    assert D.clean_summary(None) == ""


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print(f"\nAll {len(fns)} tests passed.")
