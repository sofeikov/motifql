"""Tests for the issue-25 stability metrics (synthetic rankings, no data).

Pins the pure functions in `eval/stability.py`: top-k overlap, the
score-drop distributions, and the eviction-cost accounting including
its censored/survived edge cases.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from eval.stability import (evasion_costs, level_name, mean_overlap_at_k,  # noqa: E402
                            overlap_at_k, score_drop_stats)


def test_overlap_counts_shared_topk() -> None:
    assert overlap_at_k([1, 2, 3, 4], [1, 2, 3, 4], 2) == 1.0
    assert overlap_at_k([1, 2, 3, 4], [3, 4, 1, 2], 2) == 0.0
    assert overlap_at_k([1, 2, 3, 4], [2, 5, 6, 7], 2) == 0.5
    # Order inside top-k does not matter; members below k do not count.
    assert overlap_at_k([1, 2, 3], [2, 1, 9], 2) == 1.0


def test_mean_overlap_macro_averages_shared_days() -> None:
    clean = {"d1": [1, 2, 3], "d2": [4, 5, 6], "d3": [7, 8, 9]}
    noisy = {"d1": [1, 2, 3], "d2": [9, 9, 9]}
    mean, per_day, n_days = mean_overlap_at_k(clean, noisy, 2)
    assert per_day == {"d1": 1.0, "d2": 0.0}
    assert mean == 0.5
    assert n_days == 2  # d3 has no noisy ranking: skipped, counted


def test_score_drops_split_truth_and_background() -> None:
    clean = {1: 0.9, 2: 0.8, 3: 0.1, 4: 0.0}
    noisy = {1: 0.7, 2: 0.8, 3: 0.0, 5: 0.5}
    stats = score_drop_stats(clean, noisy, {1, 2})
    assert stats["n_matched"] == 3  # tid 4/5 match nothing
    assert stats["truth"]["drop"]["n"] == 2
    assert stats["truth"]["drop"]["mean"] == pytest.approx(-0.1)
    assert stats["truth"]["abs"]["p90"] == pytest.approx(0.18)
    assert stats["background"]["drop"]["n"] == 1
    assert stats["background"]["drop"]["mean"] == pytest.approx(-0.1)


def test_score_drops_empty_group_is_none_not_crash() -> None:
    stats = score_drop_stats({1: 0.5}, {1: 0.4}, set())
    assert stats["truth"]["drop"] == {"n": 0, "mean": None, "p50": None, "p90": None}
    assert stats["background"]["drop"]["mean"] == pytest.approx(-0.1)


def test_evasion_costs_first_eviction_rung() -> None:
    # t1 evicted at rung 1, t2 at rung 2, t3 survives, t4 censored
    # (drifts out at rung 2), t5 never in the clean top-2: out of scope.
    levels = [{1: 1, 2: 2, 3: 1, 4: 2, 5: 9},
              {1: 9, 2: 2, 3: 1, 4: 2, 5: 1},
              {1: 9, 2: 9, 3: 2, 5: 1}]
    per_anchor, summary = evasion_costs(levels, 2)
    assert per_anchor == {1: (1, "evicted"), 2: (2, "evicted"),
                          3: (None, "survived"), 4: (None, "censored")}
    assert summary["n_pop"] == 4
    assert (summary["n_evicted"], summary["n_survived"],
            summary["n_censored"]) == (2, 1, 1)
    curve = summary["curve"]
    assert [(c["rung"], c["n_evicted_cum"]) for c in curve] == [(1, 1), (2, 2)]
    # Fractions exclude the censored anchor (denominator 3, not 4).
    assert curve[0]["frac_evicted_cum"] == 1 / 3
    assert curve[1]["frac_evicted_cum"] == 2 / 3


def test_level_name_is_stable() -> None:
    assert level_name(0.0, 1.0) == "level_p0_g1.parquet"
    assert level_name(0.05, 1.0) == "level_p0.05_g1.parquet"
    assert level_name(0.0, 1.5) == "level_p0_g1.5.parquet"
