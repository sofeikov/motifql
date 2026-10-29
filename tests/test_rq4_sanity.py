"""Tests for RQ4 sanity sampling and feedback rules (#15 #7)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.rq4 import sanity_background_ids, sanity_feedback  # noqa: E402


def test_sanity_background_is_disjoint_and_seeded() -> None:
    pool = np.arange(1000)
    used = {1, 2, 3}
    first = sanity_background_ids(pool, used, n=100, seed=7)
    assert len(first) == 100 and not (set(first) & used)
    again = sanity_background_ids(pool, used, n=100, seed=7)
    assert first == again
    assert sanity_background_ids(pool, used, n=100, seed=8) != first


def test_sanity_feedback_rules() -> None:
    # No truth fires: shape failure. Zero background alone: silence
    # (the expected outcome for a rare-pattern motif).
    assert "0/10" in (sanity_feedback(0, 10, 0, 1000) or "")
    assert sanity_feedback(3, 10, 0, 1000) is None
    assert sanity_feedback(3, 10, 50, 1000) is None
    assert "too broad" in (sanity_feedback(3, 10, 300, 1000) or "")
    # The shape failure outranks the broad one.
    assert "shape is wrong" in (sanity_feedback(0, 10, 300, 1000) or "")
