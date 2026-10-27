"""Trailing-window features against a brute-force oracle.

Every later result is compared with the aggregate baselines, so the 18
features they share get a committed regression test. Fast: a hand-built
edge-case frame, a seeded random frame, and a seeded single-account hub.
"""

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from eval.features import trailing_count_sum, trailing_distinct, WINDOWS


def brute(df, acct_col, peer_col, prefix):
    """Reference implementation: for each row, filter strictly prior
    same-account rows inside each window. O(n*w); obviously correct."""
    rows = df.to_dicts()
    out = {}
    for i, r in enumerate(rows):
        res = {}
        for w, wus in WINDOWS.items():
            lo = r["timestamp"] - timedelta(microseconds=wus)
            prior = [q for q in rows[:i]
                     if q[acct_col] == r[acct_col] and q["timestamp"] >= lo]
            res[f"{prefix}_count_{w}"] = len(prior)
            res[f"{prefix}_distinct_{w}"] = len({q[peer_col] for q in prior})
            res[f"{prefix}_sum_{w}"] = sum((q["amount_usd"] or 0) for q in prior)
        out[r["transfer_id"]] = res
    return out


def check(df):
    """Both directions, all windows: implementation equals brute force.

    Counts are exact integers; sums allow prefix-subtraction rounding.
    """
    for acct, peer, prefix in (("a", "p", "out"), ("p", "a", "in")):
        got = (trailing_count_sum(df, acct, prefix)
               .join(trailing_distinct(df, acct, peer, prefix),
                     on="transfer_id", how="inner", maintain_order="left")
               .sort("transfer_id").to_dicts())
        exp = brute(df, acct, peer, prefix)
        assert len(got) == len(exp) == len(df)
        for g in got:
            for k, v in exp[g["transfer_id"]].items():
                tol = 0.0 if k.split("_")[1] != "sum" else 1e-6 + 1e-9 * abs(v)
                assert abs(float(g[k]) - float(v)) <= tol, (prefix, g["transfer_id"], k)


def test_edge_cases():
    """Ties, exact window boundaries, empty windows, null amounts."""
    t0 = datetime(2022, 9, 1)
    df = pl.DataFrame({
        "transfer_id": list(range(9)),
        "timestamp": [t0, t0, t0 + timedelta(minutes=30), t0 + timedelta(hours=1),
                      t0 + timedelta(hours=1), t0 + timedelta(hours=2),
                      t0 + timedelta(days=1), t0 + timedelta(days=8),
                      t0 + timedelta(days=8)],
        "a": ["x", "x", "x", "x", "y", "x", "x", "x", "z"],
        "p": ["m", "m", "n", "m", "m", "n", "o", "m", "m"],
        "amount_usd": [10.0, 20.0, 5.0, 1.0, 100.0, 7.0, 3.0, 9.0, None],
    })
    check(df)


def test_random_frame():
    """Seeded multi-account frame with ties and nulls."""
    rng = np.random.default_rng(11)
    t0 = datetime(2022, 9, 1)
    n = 300
    df = pl.DataFrame({
        "transfer_id": range(n),
        "timestamp": sorted(t0 + timedelta(minutes=int(x))
                            for x in rng.integers(0, 10 * 24 * 60, n)),
        "a": [f"a{rng.integers(0, 4)}" for _ in range(n)],
        "p": [f"p{rng.integers(0, 9)}" for _ in range(n)],
        "amount_usd": [None if r < 0.05 else round(float(v), 2)
                       for r, v in zip(rng.uniform(size=n), rng.uniform(1, 9999, n))],
    })
    check(df)


def test_hub():
    """Seeded single-account burst: 800 rows, 60 peers, minute ties."""
    rng = np.random.default_rng(7)
    t0 = datetime(2022, 9, 1)
    n = 800
    df = pl.DataFrame({
        "transfer_id": range(n),
        "timestamp": [t0 + timedelta(minutes=int(x))
                      for x in sorted(rng.integers(0, 30 * 24 * 60, n))],
        "a": ["hub"] * n,
        "p": [f"p{rng.integers(0, 60)}" for _ in range(n)],
        "amount_usd": rng.uniform(1, 10000, n).round(2),
    })
    check(df)
