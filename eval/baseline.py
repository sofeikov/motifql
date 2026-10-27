#!/usr/bin/env python3
"""Windowed-aggregate baseline: the RQ1 comparison point from DESIGN.md section 8.

Pipeline
--------
1. Load prepped transfers plus pattern labels.
2. Per transfer, compute trailing-window aggregates over strictly prior
   transfers of the same account: count, distinct counterparties and sum of
   USD amounts over 1h / 24h / 7d, sender side and receiver side (18 features).
3. Rank-normalise each feature against its training-split empirical
   distribution (no labels used); the alert score is the max over features,
   ties broken by the second-largest value, then transfer id.
4. Rank holdout transfers per day and report recall per pattern type at
   fixed alert budgets.

Implementation notes
--------------------
Count and sum are plain Polars: per-account cumulative sums and row numbers
(`.over`), with each row's window start found by a forward as-of join (the
first same-account row at or after timestamp minus window), then prefix
subtraction. Distinct counterparties use a small numba sweep instead: exact
trailing distinct needs freq-map state per window, which no composition of
window functions expresses, and the sender side has 100K+ transfer hubs
(bank 070), where pair-materialising joins mean tens of billions of pairs
(measured: OOM). The sweep is linear in total transfers. All features are
verified against a brute-force oracle (see the day report).

Usage:
    uv run eval/baseline.py --data data/hi-small --out eval/results
"""

from __future__ import annotations

import logging
import warnings
from datetime import date
from pathlib import Path
from typing import Annotated

import polars as pl
import numpy as np
import numba
import typer

log = logging.getLogger("baseline")

US = 1_000_000
WINDOWS = {"1h": 3600 * US, "24h": 24 * 3600 * US, "7d": 7 * 24 * 3600 * US}

# Warm-up (Sep 1-2) fills the trailing windows; the tail (Sep 11+) is too
# thin for per-day alert budgets. See the day report for the split rationale.
DEFAULT_TRAIN = "2022-09-03,2022-09-07"
DEFAULT_HOLDOUT = "2022-09-08,2022-09-10"
DEFAULT_BUDGETS = "50,100,500,5000"


def _ordered_work_table(df: pl.DataFrame, acct_col: str) -> pl.DataFrame:
    """transfer_id, account, timestamp (µs) and amount, sorted by time.

    Every feature function starts here, so none of them depends on the input
    frame's row order: "strictly prior" always means earlier in
    (timestamp, transfer_id). The `pos` column makes that order explicit for
    the partitioned expressions below.
    """
    return (df.select(
        "transfer_id",
        pl.col("timestamp").dt.timestamp("us").alias("ts"),
        pl.col(acct_col).alias("acct"),
        pl.col("amount_usd").fill_null(0.0).alias("amt"),
    ).sort("ts", "transfer_id")
     .with_columns(pl.int_range(pl.len()).alias("pos")))


def trailing_count_sum(df: pl.DataFrame, acct_col: str, prefix: str) -> pl.DataFrame:
    """Trailing-window count and sum over strictly prior transfers.

    Method, per account: number the rows and prefix-sum the amounts (both
    `.over` the account in time order). The window start of row `i` is the
    first same-account row at or after `ts[i] - w`, found with one forward
    as-of join per window; count and sum are prefix differences, which
    restricts them to strictly prior rows inside the window. Exact, including
    same-timestamp ties: tied rows share `ts`, so every earlier-file-order
    row with `ts >= cutoff` falls between the window start and row `i`.

    Returns transfer_id plus count/sum for 1h/24h/7d.
    """
    w = _ordered_work_table(df, acct_col).with_columns(
        pl.int_range(pl.len()).over("acct", order_by="pos").alias("nr"),
        (pl.col("amt").cum_sum().over("acct", order_by="pos")
         - pl.col("amt")).alias("psum"),
    )
    out = w.select("transfer_id")
    mine = w.select("transfer_id", pl.col("nr").alias("my_nr"),
                    pl.col("psum").alias("my_psum"))
    right = w.select("acct", pl.col("ts").alias("key"), "nr", "psum").sort("key")
    for name, wus in WINDOWS.items():
        lo = w.select("transfer_id", "acct", (pl.col("ts") - wus).alias("key")).sort("key")
        with warnings.catch_warnings():
            # Both sides are sorted by key; Polars just can't verify
            # sortedness inside `by` groups.
            warnings.filterwarnings("ignore", message="Sortedness of columns.*",
                                    category=UserWarning)
            hit = lo.join_asof(right, on="key", by="acct", strategy="forward")
        if hit.filter(pl.col("nr").is_null()).select(pl.len()).item():
            # Every row matches at least itself (ts[i] >= ts[i] - w), so a
            # miss means the join or the ordering is broken: fail, don't fill.
            raise RuntimeError(f"{prefix}: window start missing for some rows ({name})")
        feat = (hit.select("transfer_id", pl.col("nr").alias("lo_nr"),
                           pl.col("psum").alias("lo_psum"))
                   .join(mine, on="transfer_id", how="left", maintain_order="left")
                   .select("transfer_id",
                           (pl.col("my_nr") - pl.col("lo_nr")).alias(f"{prefix}_count_{name}"),
                           (pl.col("my_psum") - pl.col("lo_psum")).alias(f"{prefix}_sum_{name}")))
        out = out.join(feat, on="transfer_id", how="left", maintain_order="left")
    return out


@numba.njit(cache=True)
def _distinct_sweep(ts, peer, w_us, freq, out):
    """One account's trailing-window distinct count at every row.

    Rows are in time order. Two pointers plus a frequency table make the
    sweep linear; `freq` is a scratch array shared across accounts, reset
    through the codes this account touched. See `trailing_distinct` for why
    this is a kernel and not window functions.
    """
    n = ts.shape[0]
    touched = np.empty(n, dtype=np.int64)
    n_touched = 0
    lo = 0
    distinct = 0
    for i in range(n):
        cutoff = ts[i] - w_us
        while ts[lo] < cutoff:
            c = peer[lo]
            freq[c] -= 1
            if freq[c] == 0:
                distinct -= 1
            lo += 1
        out[i] = np.float32(distinct)
        c = peer[i]
        if freq[c] == 0:
            distinct += 1
            touched[n_touched] = c
            n_touched += 1
        freq[c] += 1
    for k in range(n_touched):
        freq[touched[k]] = 0


def trailing_distinct(df: pl.DataFrame, acct_col: str, peer_col: str,
                      prefix: str) -> pl.DataFrame:
    """Trailing-window distinct counterparties over strictly prior transfers.

    Why a kernel: exact trailing distinct needs freq-map state per window,
    which no composition of Polars window functions expresses (each row's
    query is a stabbing query over peer occurrences: inherently sequential
    or pair-materialising). The sender side has 168K-transfer hubs, where
    the pair join means tens of billions of pairs (measured: OOM), so the
    linear two-pointer sweep is the algorithm, not a shortcut. The frame is
    sorted explicitly first, so the result never depends on input row order.

    Returns transfer_id plus distinct counts for 1h/24h/7d.
    """
    enc = (df.select(
        "transfer_id",
        pl.col("timestamp").dt.timestamp("us").alias("ts"),
        pl.col(acct_col).cast(pl.Categorical).to_physical().alias("acct"),
        pl.col(peer_col).cast(pl.Categorical).to_physical().alias("peer"),
    ).sort("ts", "transfer_id"))
    ts = enc["ts"].to_numpy()
    acct = enc["acct"].to_numpy()
    peer = enc["peer"].to_numpy().astype(np.int64)
    n = len(enc)
    order = np.argsort(acct, kind="stable")
    oacct = acct[order]
    bounds = np.concatenate(([0], np.flatnonzero(oacct[1:] != oacct[:-1]) + 1, [n]))
    inv = np.argsort(order)
    log.info("  %s: %d accounts, largest %d transfers",
             prefix, len(bounds) - 1, np.diff(bounds).max())
    freq = np.zeros(int(peer.max()) + 1, dtype=np.int64)
    cols: dict[str, np.ndarray] = {}
    for name, wus in WINDOWS.items():
        D = np.empty(n, dtype=np.float32)
        for b in range(len(bounds) - 1):
            s0, s1 = int(bounds[b]), int(bounds[b + 1])
            idx = order[s0:s1]
            _distinct_sweep(ts[idx], peer[idx], np.int64(wus), freq, D[s0:s1])
        cols[f"{prefix}_distinct_{name}"] = D[inv]
    return pl.DataFrame({"transfer_id": enc["transfer_id"], **cols})


def main(
    data: Annotated[Path, typer.Option(help="Prepped dataset directory.")],
    train: Annotated[str, typer.Option(help="Train day range, inclusive.")] = DEFAULT_TRAIN,
    holdout: Annotated[str, typer.Option(help="Holdout day range, inclusive.")] = DEFAULT_HOLDOUT,
    budgets: Annotated[str, typer.Option(help="Alerts per day, comma-separated.")] = DEFAULT_BUDGETS,
    out: Annotated[Path, typer.Option(help="Directory for the recall CSV.")] = Path("eval/results"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tr0, tr1 = (date.fromisoformat(s) for s in train.split(","))
    ho0, ho1 = (date.fromisoformat(s) for s in holdout.split(","))
    budget_list = [int(b) for b in budgets.split(",")]
    out.mkdir(parents=True, exist_ok=True)

    log.info("[1/4] loading transfers and pattern labels")
    df = pl.read_parquet(data / "transfers.parquet")
    inst = pl.read_parquet(data / "pattern_instances.parquet")
    memb = pl.read_parquet(data / "pattern_members.parquet")
    labels = memb.join(inst.select(["instance_id", "pattern_type"]), on="instance_id")
    multi = labels.group_by("transfer_id").len().filter(pl.col("len") > 1).select(pl.len()).item()
    log.info("  rows: %d, transfers in >1 instance: %d", len(df), multi)
    df = df.with_columns(
        (pl.col("from_bank") + "/" + pl.col("from_account")).alias("from_key"),
        (pl.col("to_bank") + "/" + pl.col("to_account")).alias("to_key"))

    log.info("[2/4] computing trailing-window features")
    feats = (trailing_count_sum(df, "from_key", "out")
             .join(trailing_count_sum(df, "to_key", "in"),
                   on="transfer_id", how="inner", maintain_order="left")
             .join(trailing_distinct(df, "from_key", "to_key", "out"),
                   on="transfer_id", how="inner", maintain_order="left")
             .join(trailing_distinct(df, "to_key", "from_key", "in"),
                   on="transfer_id", how="inner", maintain_order="left"))
    feat_names = [c for c in feats.columns if c != "transfer_id"]
    X = (df.select("transfer_id")
           .join(feats, on="transfer_id", how="left", maintain_order="left")
           .select(feat_names).with_columns(pl.all().cast(pl.Float32)).to_numpy())
    log.info("  %d features", len(feat_names))

    log.info("[3/4] rank-normalising on train, scoring holdout")
    is_train = df.select(pl.col("day").is_between(tr0, tr1)).to_series().to_numpy()
    is_hold = df.select(pl.col("day").is_between(ho0, ho1)).to_series().to_numpy()
    log.info("  train rows: %d, holdout rows: %d", is_train.sum(), is_hold.sum())
    cdfs = np.sort(X[is_train], axis=0)
    n_tr = is_train.sum()
    cdf_idx = np.stack([np.searchsorted(cdfs[:, j], X[is_hold, j], side="right")
                        for j in range(len(feat_names))], axis=1)
    hold_scores = (cdf_idx / n_tr).max(axis=1)
    # Deterministic tie-break: rows whose score ties (common at 1.0, where a
    # feature exceeds every training value) order by their second-largest
    # feature value, then transfer id, instead of by input order.
    secondary = np.partition(cdf_idx, -2, axis=1)[:, -2] / n_tr
    hold_argmax = cdf_idx.argmax(axis=1)
    labels_one = labels.group_by("transfer_id").agg(pl.col("pattern_type").first())
    hold = (df.filter(pl.col("day").is_between(ho0, ho1))
            .with_columns(pl.Series("score", hold_scores),
                          pl.Series("secondary", secondary),
                          pl.Series("driver", [feat_names[j] for j in hold_argmax]))
            .join(labels_one, on="transfer_id", how="left", maintain_order="left")
            .sort("day", "score", "secondary", "transfer_id",
                  descending=[False, True, True, False])
            .with_columns((pl.int_range(pl.len()).over("day") + 1).alias("rank")))
    hold.select(["transfer_id", "timestamp", "day", "score", "secondary", "driver",
                 "is_laundering", "pattern_type"]
                ).write_parquet(data / "baseline_scores.parquet")
    log.info("  full holdout ranking -> %s", data / "baseline_scores.parquet")

    log.info("[4/4] recall at fixed alerts/day budgets")
    hold_days = sorted(hold["day"].unique())
    rows = []
    for k in budget_list:
        topk = hold.filter(pl.col("rank") <= k)
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
        log.info("  top-%d/day alert drivers: %s", k,
                 ", ".join(f"{r['driver']}={r['len']}" for r in drivers.iter_rows(named=True)))
    res = pl.DataFrame(rows)
    res.write_csv(out / "baseline_recall.csv")
    with pl.Config(tbl_rows=-1):
        log.info("\n%s", res.pivot(index="group", on="budget_per_day", values="recall").sort("group"))
    log.info("done -> %s (%d holdout days: %s..%s)",
             out / "baseline_recall.csv", len(hold_days), hold_days[0], hold_days[-1])


if __name__ == "__main__":
    typer.run(main)
