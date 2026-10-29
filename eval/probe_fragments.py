#!/usr/bin/env python3
"""Fragment probe for the Nov 1 discovery diagnosis (report section 6).

Why fan fragments are (mostly) unrecoverable: background senders hit
>=2 distinct receivers per 4d window ~40% of the time (the flood any
tiny-count motif drowns in), while truth multi-fragments ARE unusual
vs their own history — but singletons show no count signal at all.

History window honesty: data starts Sep 1 and train starts Sep 3,
so "trailing history" is at most 6 days, not 30. Seeds looped for
the background fraction (n=200 has wide spread).

Usage: uv run eval/probe_fragments.py
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

MISS_KEYS = {"out": [123, 138, 223, 224], "in": [105, 107, 160, 208]}


def main() -> None:
    df = pl.read_parquet("data/hi-small/transfers.parquet")
    memb = pl.read_parquet("data/hi-small/pattern_members.parquet")
    print("== truth fragment keys: all-history distinct vs win4d distinct ==")
    for kind, col, ccol in (("out", "from_account", "to_account"),
                            ("in", "to_account", "from_account")):
        for key in MISS_KEYS[kind]:
            tids = memb.filter(pl.col("instance_id") == key)["transfer_id"].to_list()
            rows = df.filter(pl.col("transfer_id").is_in(tids)).sort("timestamp")
            acc, t0 = rows[col][0], rows["timestamp"][0]
            hist = df.filter((pl.col(col) == acc) & (pl.col("timestamp") < t0))[ccol].n_unique()
            win = df.filter((pl.col(col) == acc) & (pl.col("timestamp") >= t0)
                            & (pl.col("timestamp") < t0 + timedelta(days=4)))[ccol].n_unique()
            print(f"  {kind.upper()} inst {key}: hist_all={hist} win4d={win}")
    print("== background senders (n=200, 3 seeds): win4d>=2 fraction ==")
    pool = sorted(df.filter(pl.col("day").is_between(date(2022, 9, 3), date(2022, 9, 7)))
                  ["from_account"].unique().to_list())
    t0 = datetime(2022, 9, 5)
    for seed in (3, 4, 5):
        senders = np.random.default_rng(seed).choice(pool, 200, replace=False)
        ws = [df.filter((pl.col("from_account") == acc) & (pl.col("timestamp") >= t0)
                        & (pl.col("timestamp") < t0 + timedelta(days=4)))["to_account"].n_unique()
              for acc in senders]
        print(f"  seed {seed}: frac(win4d>=2)={np.mean(np.array(ws) >= 2):.3f}")
    print("== agent-family ceiling: sender-days >=2 distinct AND min>=5k ==")
    g = (df.filter(pl.col("day") == date(2022, 9, 5))
         .group_by(["from_account"]).agg(pl.col("to_account").n_unique().alias("d"),
                                          pl.col("amount_usd").min().alias("m")))
    hit = g.filter((pl.col("d") >= 2) & (pl.col("m") >= 5000)).height
    print(f"  {hit}/{g.height} = {hit / g.height:.4f}")


if __name__ == "__main__":
    main()
