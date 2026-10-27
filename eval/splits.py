"""Instance-to-split assignment for pattern-level evaluation.

Day-cutoff splits slice multi-day laundering instances in two: an instance
that starts in train and ends in holdout would leak across the boundary, and
one that completes in the unscored tail can never be caught (motifs fire on
the transfer that completes a pattern). So instances are assigned wholly to
one bucket by their member span, and per-instance recall is reported over
the holdout buckets only. Transfer-level training labels are unaffected:
this governs instance eval sets and agent-visible instances, not rows.
"""

from __future__ import annotations

import logging
from datetime import date

import polars as pl

log = logging.getLogger("splits")

# Single source of truth for the split; CLI defaults derive from these.
WARMUP_START = date(2022, 9, 1)
WARMUP_END = date(2022, 9, 2)
TRAIN_START = date(2022, 9, 3)
TRAIN_END = date(2022, 9, 7)
HOLDOUT_START = date(2022, 9, 8)
HOLDOUT_END = date(2022, 9, 10)
TAIL_START = date(2022, 9, 11)


def assign_instance_splits(instances: pl.DataFrame, members: pl.DataFrame,
                           transfers: pl.DataFrame) -> pl.DataFrame:
    """Assign every pattern instance to exactly one split bucket.

    Buckets: `train` (wholly in Sep 3-7), `holdout` (wholly in scored Sep
    8-10, completable), `holdout_partial` (starts Sep 8+, runs into the
    tail, but has scored members: catchable, not completable), `tail_only`
    and `warmup_only` (outside every eval set), `crosser` (spans a scored
    boundary: dropped from both sets, count reported).

    Returns the instance frame plus a `split` column.
    """
    member_days = (members.join(transfers.select("transfer_id", "day"),
                                on="transfer_id", how="left")
                   .group_by("instance_id")
                   .agg(pl.len().alias("n_members"),
                        pl.col("day").is_between(HOLDOUT_START, HOLDOUT_END)
                        .sum().alias("n_scored")))
    out = (instances.with_columns(
        pl.col("t_min").dt.date().alias("dmin"),
        pl.col("t_max").dt.date().alias("dmax"))
        .join(member_days, on="instance_id", how="left")
        .with_columns(pl.col("n_scored").fill_null(0))
        .with_columns(
            pl.when(pl.col("dmax") <= WARMUP_END).then(pl.lit("warmup_only"))
            .when((pl.col("dmin") >= TRAIN_START) & (pl.col("dmax") <= TRAIN_END))
            .then(pl.lit("train"))
            .when((pl.col("dmin") >= HOLDOUT_START) & (pl.col("dmax") <= HOLDOUT_END))
            .then(pl.lit("holdout"))
            .when(pl.col("dmin") >= TAIL_START).then(pl.lit("tail_only"))
            .when((pl.col("dmin") >= HOLDOUT_START) & (pl.col("n_scored") > 0))
            .then(pl.lit("holdout_partial"))
            .otherwise(pl.lit("crosser")).alias("split")))
    orphans = out.filter(pl.col("n_members").fill_null(0) == 0).select(pl.len()).item()
    log.info("  instances with no joined members: %d", orphans)
    for row in out.group_by("split").len().sort("split").iter_rows(named=True):
        log.info("    %-15s %d", row["split"], row["len"])
    return out
