"""Trailing-window transfer features shared by the baselines.

For each transfer, aggregates over strictly prior transfers of the same
account (in timestamp, then file order): count, distinct counterparties and
sum of USD amounts over 1h / 24h / 7d, sender side and receiver side
(18 features). Count and sum are plain Polars (cumulative sums plus as-of
window starts); distinct uses a linear sweep (see `trailing_distinct` for
why). Covered by `tests/test_baseline_features.py` against a brute-force
oracle.
"""

from __future__ import annotations

import logging
import warnings

import polars as pl
import numpy as np
import numba

log = logging.getLogger("features")

US = 1_000_000
WINDOWS = {"1h": 3600 * US, "24h": 24 * 3600 * US, "7d": 7 * 24 * 3600 * US}


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


def compute_all(df: pl.DataFrame) -> pl.DataFrame:
    """All 18 trailing-window features (both directions, all windows).

    Expects transfer_id, timestamp, from_key, to_key, amount_usd. Returns
    transfer_id plus 18 float columns, one row per input transfer.
    """
    return (trailing_count_sum(df, "from_key", "out")
            .join(trailing_count_sum(df, "to_key", "in"),
                  on="transfer_id", how="inner", maintain_order="left")
            .join(trailing_distinct(df, "from_key", "to_key", "out"),
                  on="transfer_id", how="inner", maintain_order="left")
            .join(trailing_distinct(df, "to_key", "from_key", "in"),
                  on="transfer_id", how="inner", maintain_order="left"))
