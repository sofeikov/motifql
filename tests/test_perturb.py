"""Tests for the RQ2 perturbation generator (Nov 1).

`perturb_frame` is pure (frames in, frame out), so these run on a toy
frame with no data dependency: level-0 identity, seed determinism,
within-instance order preservation, gap scaling, non-member
immutability, and day recomputation.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from eval.perturb import perturb_frame  # noqa: E402


def _toy() -> pl.DataFrame:
    ts = [
        datetime(2022, 9, 8, 10, 0),    # 0 inst 1
        datetime(2022, 9, 8, 11, 0),    # 1 inst 1
        datetime(2022, 9, 8, 12, 0),    # 2 inst 1
        datetime(2022, 9, 8, 10, 30),   # 3 background
        datetime(2022, 9, 8, 23, 0),    # 4 inst 2 (t_min)
        datetime(2022, 9, 8, 23, 30),   # 5 inst 2
        datetime(2022, 9, 8, 23, 30),   # 6 inst 2 (tie with 5)
        datetime(2022, 9, 8, 9, 0),     # 7 background
    ]
    return pl.DataFrame({
        "transfer_id": list(range(8)),
        "timestamp": ts,
        "amount_usd": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0],
        "day": [t.date() for t in ts],
    })


MEMBERS = {1: [0, 1, 2], 2: [4, 5, 6]}


def test_level_zero_is_identity() -> None:
    df = _toy()
    assert_frame_equal(perturb_frame(df, MEMBERS, 0.0, 1.0, 12345), df)


def test_deterministic_and_seed_sensitive() -> None:
    df = _toy()
    a = perturb_frame(df, MEMBERS, 0.2, 2.0, 99)
    b = perturb_frame(df, MEMBERS, 0.2, 2.0, 99)
    assert_frame_equal(a, b)
    c = perturb_frame(df, MEMBERS, 0.2, 2.0, 100)
    assert (a["amount_usd"] != c["amount_usd"]).any()


def test_gaps_scale_and_order_holds() -> None:
    df = _toy()
    out = perturb_frame(df, MEMBERS, 0.0, 2.0, 1)
    ts = out["timestamp"].to_list()
    assert ts[0] == datetime(2022, 9, 8, 10, 0)  # t_min fixed
    assert ts[1] == datetime(2022, 9, 8, 12, 0)  # +1h -> +2h
    assert ts[2] == datetime(2022, 9, 8, 14, 0)  # +2h -> +4h
    assert ts[5] == ts[6]  # ties stay tied
    assert ts[5] == datetime(2022, 9, 9, 0, 0)  # +30m -> +60m
    # Non-members untouched (values and row order).
    for i in (3, 7):
        assert out.row(i) == df.row(i)
    assert out["transfer_id"].to_list() == df["transfer_id"].to_list()


def test_amounts_within_pct_and_day_recomputed() -> None:
    df = _toy()
    out = perturb_frame(df, MEMBERS, 0.1, 3.0, 5)
    for t in (0, 1, 2, 4, 5, 6):
        f = out["amount_usd"][t] / df["amount_usd"][t]
        assert 0.9 <= f <= 1.1
    # Inst-2 members cross midnight at x3: day follows the timestamp.
    assert out["timestamp"][5] == datetime(2022, 9, 9, 0, 30)
    assert out["day"][5].isoformat() == "2022-09-09"
    assert out["day"][4].isoformat() == "2022-09-08"
