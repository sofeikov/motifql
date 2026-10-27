"""Day-wise ranking and recall tables shared by the baselines.

Every baseline funnels through here, so the numbers are comparable by
construction: same ranking rule, same budgets, same denominators, at both
transfer and instance granularity.
"""

from __future__ import annotations

import polars as pl


def rank_per_day(scored: pl.DataFrame, secondary_col: str | None = None) -> pl.DataFrame:
    """Rank transfers 1..N within each day: score desc, then the secondary
    key desc (breaks the ties that saturate at 1.0), then transfer id asc.
    Fully deterministic: no input-order dependence anywhere in the key.
    """
    keys, desc = (["day", "score", "transfer_id"], [False, True, False])
    if secondary_col is not None:
        keys, desc = (["day", "score", secondary_col, "transfer_id"],
                      [False, True, True, False])
    return scored.sort(keys, descending=desc).with_columns(
        (pl.int_range(pl.len()).over("day") + 1).alias("rank"))


def transfer_recall(ranked: pl.DataFrame, budgets: list[int]) -> pl.DataFrame:
    """Per-transfer recall per budget for ALL_FLAGGED, UNATTRIBUTED and each
    pattern type. Expects rank, is_laundering and pattern_type columns.
    """
    rows = []
    groups: dict[str, pl.DataFrame] = {
        "ALL_FLAGGED": ranked.filter(pl.col("is_laundering") == 1),
        "UNATTRIBUTED": ranked.filter((pl.col("is_laundering") == 1) &
                                      pl.col("pattern_type").is_null())}
    for pt in sorted(ranked["pattern_type"].drop_nulls().unique()):
        groups[pt] = ranked.filter(pl.col("pattern_type") == pt)
    for k in budgets:
        topk = ranked.filter(pl.col("rank") <= k).select("transfer_id")
        for gname, gdf in groups.items():
            hit = topk.join(gdf.select("transfer_id"), on="transfer_id",
                            how="semi").select(pl.len()).item()
            rows.append({"budget": k, "group": gname, "n_total": len(gdf),
                         "n_hit": hit, "recall": hit / len(gdf) if len(gdf) else None})
    return pl.DataFrame(rows)


def instance_recall(ranked: pl.DataFrame, members: pl.DataFrame,
                    inst: pl.DataFrame, budgets: list[int]) -> pl.DataFrame:
    """Per-instance recall per budget and pattern: an instance is caught if
    any alert in the scored days hits one of its members. Covers the holdout
    buckets; the `set` column separates completable instances from the
    combined set. Expects rank on `ranked` and split on `inst`.
    """
    hits = (ranked.select("transfer_id", "rank")
            .join(members.select("instance_id", "transfer_id"),
                  on="transfer_id", how="inner"))
    rows = []
    for set_name, splits in (("all", ("holdout", "holdout_partial")),
                             ("completable", ("holdout",))):
        sub = inst.filter(pl.col("split").is_in(splits))
        for k in budgets:
            caught = hits.filter(pl.col("rank") <= k).select("instance_id").unique()
            for pt in sorted(sub["pattern_type"].unique()):
                tot = sub.filter(pl.col("pattern_type") == pt).select(pl.len()).item()
                hit = (sub.filter(pl.col("pattern_type") == pt)
                       .join(caught, on="instance_id", how="semi").select(pl.len()).item())
                rows.append({"budget": k, "set": set_name, "group": pt,
                             "n_total": tot, "n_hit": hit,
                             "recall": hit / tot if tot else None})
    return pl.DataFrame(rows)
