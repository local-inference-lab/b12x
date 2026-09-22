"""Sanitizer completion cannot be inferred from a quiet or partial log."""

from scripts.qualify_hybrid_sanitizer import classify


def test_requires_completion_from_every_rank_and_zero_summaries():
    assert classify(0, False, [0, 0], {0, 1}, 2, False) == "whole_program_pass"
    assert classify(0, False, [0, 0], {0}, 2, False) == "incomplete"
    assert classify(0, False, [], {0, 1}, 2, False) == "incomplete"
    assert classify(99, False, [120, 120], {0, 1}, 2, False) == "failed"
    assert classify(124, True, [0], {0}, 2, False) == "timeout"


def test_filter_never_qualifies_whole_program():
    assert classify(0, False, [0, 0], {0, 1}, 2, True) == "component_pass"
