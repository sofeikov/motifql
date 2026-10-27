#!/usr/bin/env python3
"""Windowed-aggregate baseline: the RQ1 comparison point from DESIGN.md section 8.

For each transfer, computes trailing-window aggregates over strictly prior
transfers — count, distinct counterparties and sum of USD amounts over
1h / 24h / 7d, from both the sender and receiver side (18 features) — in the
style of ezrules' windowed aggregates. Each feature is rank-normalised against
its training-split empirical distribution (no labels used); the alert score is
the max over features. Holdout transfers are ranked per day and recall is
reported per pattern type at fixed alert budgets.

Usage:
    uv run eval/baseline.py --data data/hi-small --out eval/results
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import polars as pl
import numpy as np
import numba

US = 1_000_000
WINDOWS = {"1h": 3600 * US, "24h": 24 * 3600 * US, "7d": 7 * 24 * 3600 * US}


@numba.njit(cache=True)
def _sweep_account(ts, peer, amt, w_us, freq, out_c, out_d, out_s):
    """One account's trailing-window count/distinct/sum at every row.

    Rows are in time order. Two pointers plus a frequency table make the sweep
    linear; `freq` is a scratch array shared across accounts, reset through
    the codes this account touched.
    """
    n = ts.shape[0]
    touched = np.empty(n, dtype=np.int64)
    n_touched = 0
    lo = 0
    run_sum = 0.0
    distinct = 0
    for i in range(n):
        cutoff = ts[i] - w_us
        while ts[lo] < cutoff:
            c = peer[lo]
            freq[c] -= 1
            if freq[c] == 0:
                distinct -= 1
            run_sum -= amt[lo]
            lo += 1
        out_c[i] = np.float32(i - lo)
        out_d[i] = np.float32(distinct)
        out_s[i] = np.float32(run_sum)
        c = peer[i]
        if freq[c] == 0:
            distinct += 1
            touched[n_touched] = c
            n_touched += 1
        freq[c] += 1
        run_sum += amt[i]
    for k in range(n_touched):
        freq[touched[k]] = 0


def sliding_window_features(df: pl.DataFrame, acct_col: str, peer_col: str,
                            prefix: str) -> pl.DataFrame:
    """Trailing-window count / distinct / sum over strictly prior transfers.

    For every transfer, aggregates the same account's earlier transfers (in
    timestamp, then file order) whose timestamp falls within 1h / 24h / 7d
    before it. Implemented as one linear sweep per account: the sender side
    has 100K+ transfer hubs (bank 070), so pair-materialising joins (tens of
    billions of pairs, measured) do not fit in memory, and windowed distinct
    over such hubs has no window-function formulation — it needs freq-map
    state. Count and sum ride the same pass for consistency. Returns
    transfer_id plus 9 columns, in frame order.
    """
    enc = df.select(
        pl.col("timestamp").dt.timestamp("us").alias("ts"),
        pl.col(acct_col).cast(pl.Categorical).to_physical().alias("acct"),
        pl.col(peer_col).cast(pl.Categorical).to_physical().alias("peer"),
        pl.col("amount_usd").fill_null(0.0).alias("amt"),
    )
    ts = enc["ts"].to_numpy()
    acct = enc["acct"].to_numpy()
    peer = enc["peer"].to_numpy().astype(np.int64)
    amt = enc["amt"].to_numpy()
    n = len(df)
    order = np.argsort(acct, kind="stable")
    oacct = acct[order]
    bounds = np.concatenate(([0], np.flatnonzero(oacct[1:] != oacct[:-1]) + 1, [n]))
    inv = np.argsort(order)
    sizes = np.diff(bounds)
    print(f"  {prefix}: {len(bounds) - 1} accounts, largest {sizes.max()} transfers")
    freq = np.zeros(int(peer.max()) + 1, dtype=np.int64)
    cols: dict[str, np.ndarray] = {}
    for w, wus in WINDOWS.items():
        C = np.empty(n, dtype=np.float32)
        D = np.empty(n, dtype=np.float32)
        S = np.empty(n, dtype=np.float32)
        for b in range(len(bounds) - 1):
            s0, s1 = int(bounds[b]), int(bounds[b + 1])
            idx = order[s0:s1]
            _sweep_account(ts[idx], peer[idx], amt[idx], np.int64(wus), freq,
                           C[s0:s1], D[s0:s1], S[s0:s1])
        cols[f"{prefix}_count_{w}"] = C[inv]
        cols[f"{prefix}_distinct_{w}"] = D[inv]
        cols[f"{prefix}_sum_{w}"] = S[inv]
    return pl.DataFrame({"transfer_id": df["transfer_id"], **cols})


def main() -> None:
    ap = argparse.ArgumentParser(description="Windowed-aggregate baseline.")
    ap.add_argument("--data", required=True, type=Path, help="prepped dataset dir")
    ap.add_argument("--train", default="2022-09-03,2022-09-07")
    ap.add_argument("--holdout", default="2022-09-08,2022-09-10")
    ap.add_argument("--budgets", default="50,100,500,5000",
                    help="alerts per day, comma-separated")
    ap.add_argument("--out", default=Path("eval/results"), type=Path)
    args = ap.parse_args()
    tr0, tr1 = (date.fromisoformat(s) for s in args.train.split(","))
    ho0, ho1 = (date.fromisoformat(s) for s in args.holdout.split(","))
    budgets = [int(b) for b in args.budgets.split(",")]
    args.out.mkdir(parents=True, exist_ok=True)

    print("[1/4] loading transfers and pattern labels")
    df = pl.read_parquet(args.data / "transfers.parquet")
    inst = pl.read_parquet(args.data / "pattern_instances.parquet")
    memb = pl.read_parquet(args.data / "pattern_members.parquet")
    labels = memb.join(inst.select(["instance_id", "pattern_type"]), on="instance_id")
    multi = labels.group_by("transfer_id").len().filter(pl.col("len") > 1).select(pl.len()).item()
    print(f"  rows: {len(df)}, transfers in >1 instance: {multi}")
    df = df.with_columns(
        (pl.col("from_bank") + "/" + pl.col("from_account")).alias("from_key"),
        (pl.col("to_bank") + "/" + pl.col("to_account")).alias("to_key"))

    print("[2/4] computing trailing-window features")
    feats = sliding_window_features(df, "from_key", "to_key", "out").join(
        sliding_window_features(df, "to_key", "from_key", "in"),
        on="transfer_id", how="inner", maintain_order="left")
    feat_names = [c for c in feats.columns if c != "transfer_id"]
    X = (df.select("transfer_id")
           .join(feats, on="transfer_id", how="left", maintain_order="left")
           .select(feat_names).to_numpy())
    print(f"  {len(feat_names)} features")

    print("[3/4] rank-normalising on train, scoring holdout")
    is_train = df.select(pl.col("day").is_between(tr0, tr1)).to_series().to_numpy()
    is_hold = df.select(pl.col("day").is_between(ho0, ho1)).to_series().to_numpy()
    print(f"  train rows: {is_train.sum()}, holdout rows: {is_hold.sum()}")
    cdfs = np.sort(X[is_train], axis=0)
    n_tr = is_train.sum()
    cdf_idx = np.stack([np.searchsorted(cdfs[:, j], X[is_hold, j], side="right")
                        for j in range(len(feat_names))], axis=1)
    hold_scores = (cdf_idx / n_tr).max(axis=1)
    hold_argmax = cdf_idx.argmax(axis=1)
    labels_one = labels.group_by("transfer_id").agg(pl.col("pattern_type").first())
    hold = (df.filter(pl.col("day").is_between(ho0, ho1))
            .with_columns(pl.Series("score", hold_scores),
                          pl.Series("driver", [feat_names[j] for j in hold_argmax]))
            .join(labels_one, on="transfer_id", how="left", maintain_order="left"))
    hold.select(["transfer_id", "timestamp", "day", "score", "driver",
                 "is_laundering", "pattern_type"]
                ).write_parquet(args.data / "baseline_scores.parquet")
    print(f"  full holdout ranking -> {args.data / 'baseline_scores.parquet'}")

    print("[4/4] recall at fixed alerts/day budgets")
    hold_days = sorted(hold["day"].unique())
    rows = []
    for k in budgets:
        topk = hold.with_columns(
            pl.col("score").rank("ordinal", descending=True).over("day").alias("rank")
        ).filter(pl.col("rank") <= k)
        groups: dict[str, pl.DataFrame] = {"ALL_FLAGGED": hold.filter(pl.col("is_laundering") == 1),
                                           "UNATTRIBUTED": hold.filter(
                                               (pl.col("is_laundering") == 1) &
                                               pl.col("pattern_type").is_null())}
        for pt in sorted(hold["pattern_type"].drop_nulls().unique()):
            groups[pt] = hold.filter(pl.col("pattern_type") == pt)
        for gname, gdf in groups.items():
            hit = topk.join(gdf.select("transfer_id"), on="transfer_id", how="semi"
                            ).select(pl.len()).item()
            rows.append({"budget_per_day": k, "group": gname,
                         "n_total": len(gdf), "n_hit": hit,
                         "recall": hit / len(gdf) if len(gdf) else None})
        drivers = (topk.group_by("driver").len().sort("len", descending=True).head(5))
        print(f"  top-{k}/day alert drivers: " +
              ", ".join(f"{r['driver']}={r['len']}" for r in drivers.iter_rows(named=True)))
    res = pl.DataFrame(rows)
    res.write_csv(args.out / "baseline_recall.csv")
    with pl.Config(tbl_rows=-1):
        print(res.pivot(index="group", on="budget_per_day", values="recall").sort("group"))
    print(f"done -> {args.out / 'baseline_recall.csv'} "
          f"({len(hold_days)} holdout days: {hold_days[0]}..{hold_days[-1]})")


if __name__ == "__main__":
    main()