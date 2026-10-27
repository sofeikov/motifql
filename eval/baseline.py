#!/usr/bin/env python3
"""Unsupervised windowed-aggregate baseline (DESIGN.md RQ1 comparison point).

Scores each transfer by its most extreme trailing-window aggregate: 18
features (count / distinct / sum over 1h/24h/7d, sender + receiver side),
rank-normalised against the training distribution (no labels used), score =
max. Shares features, splits and reporting with the supervised baseline
(`eval/supervised.py`) via `eval/features.py`, `eval/splits.py` and
`eval/evaluate.py`, so the rows are comparable by construction.

Usage:
    uv run eval/baseline.py --data data/hi-small --out eval/results
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path
from typing import Annotated

import polars as pl
import numpy as np
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval import features, splits, evaluate

log = logging.getLogger("baseline")

DEFAULT_TRAIN = f"{splits.TRAIN_START},{splits.TRAIN_END}"
DEFAULT_HOLDOUT = f"{splits.HOLDOUT_START},{splits.HOLDOUT_END}"
DEFAULT_BUDGETS = "50,100,500,5000"
COLS = ["model", "granularity", "set", "budget", "group", "n_total", "n_hit", "recall"]


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
    inst = splits.assign_instance_splits(
        pl.read_parquet(data / "pattern_instances.parquet"),
        pl.read_parquet(data / "pattern_members.parquet"), df)
    memb = pl.read_parquet(data / "pattern_members.parquet")
    df = df.with_columns(
        (pl.col("from_bank") + "/" + pl.col("from_account")).alias("from_key"),
        (pl.col("to_bank") + "/" + pl.col("to_account")).alias("to_key"))

    log.info("[2/4] computing trailing-window features")
    feats = features.compute_all(df)
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
    labels_one = (memb.join(inst.select(["instance_id", "pattern_type"]), on="instance_id")
                  .group_by("transfer_id").agg(pl.col("pattern_type").first()))
    hold = evaluate.rank_per_day(
        df.filter(pl.col("day").is_between(ho0, ho1))
        .with_columns(pl.Series("score", hold_scores),
                      pl.Series("secondary", secondary),
                      pl.Series("driver", [feat_names[j] for j in hold_argmax]))
        .join(labels_one, on="transfer_id", how="left", maintain_order="left"),
        secondary_col="secondary")
    hold.select(["transfer_id", "timestamp", "day", "score", "secondary", "driver",
                 "is_laundering", "pattern_type"]
                ).write_parquet(data / "baseline_scores.parquet")
    log.info("  full holdout ranking -> %s", data / "baseline_scores.parquet")

    log.info("[4/4] recall at fixed alerts/day budgets")
    tr = (evaluate.transfer_recall(hold, budget_list)
          .with_columns(pl.lit("unsupervised").alias("model"),
                        pl.lit("transfer").alias("granularity"),
                        pl.lit("all").alias("set")).select(COLS))
    ir = (evaluate.instance_recall(hold, memb, inst, budget_list)
          .with_columns(pl.lit("unsupervised").alias("model"),
                        pl.lit("instance").alias("granularity")).select(COLS))
    res = pl.concat([tr, ir])
    res.write_csv(out / "recall_unsupervised.csv")
    with pl.Config(tbl_rows=-1):
        log.info("\ntransfer recall:\n%s",
                 tr.pivot(index="group", on="budget", values="recall").sort("group"))
        log.info("\ninstance recall:\n%s",
                 ir.pivot(index=["set", "group"], on="budget", values="recall")
                 .sort(["set", "group"]))
    log.info("done -> %s", out / "recall_unsupervised.csv")


if __name__ == "__main__":
    typer.run(main)
