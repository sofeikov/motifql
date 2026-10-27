#!/usr/bin/env python3
"""Supervised windowed-aggregate baseline (LightGBM over the same features).

Trains gradient-boosted trees on the training split with flag labels: the 18
trailing-window aggregates plus one-hot payment_format (and a no-format
variant, per the shortcut policy in the day report). Same per-day ranking
and recall protocol as the unsupervised baseline, shared via
`eval/features.py`, `eval/splits.py` and `eval/evaluate.py`. The last train
day doubles as the early-stopping validation set. No account or bank ids
anywhere (banned features).

Usage:
    uv run eval/supervised.py --data data/hi-small --out eval/results
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
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval import features, splits, evaluate

log = logging.getLogger("supervised")

DEFAULT_TRAIN = f"{splits.TRAIN_START},{splits.TRAIN_END}"
DEFAULT_HOLDOUT = f"{splits.HOLDOUT_START},{splits.HOLDOUT_END}"
DEFAULT_BUDGETS = "50,100,500,5000"
COLS = ["model", "granularity", "set", "budget", "group", "n_total", "n_hit", "recall"]
FORMATS = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]


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
    full = (df.select("transfer_id", "day", "is_laundering", "payment_format")
              .join(feats, on="transfer_id", how="left", maintain_order="left")
              .with_columns(pl.col(feat_names).cast(pl.Float32),
                            *[(pl.col("payment_format") == f).cast(pl.Float32).alias(f"fmt_{f}")
                              for f in FORMATS]))
    day = full["day"].to_numpy()
    y = full["is_laundering"].to_numpy()
    X18 = full.select(feat_names).to_numpy()
    Xfmt = full.select([f"fmt_{f}" for f in FORMATS]).to_numpy()
    is_fit = (day >= np.datetime64(tr0)) & (day < np.datetime64(tr1))
    is_val = day == np.datetime64(tr1)
    is_hold = (day >= np.datetime64(ho0)) & (day <= np.datetime64(ho1))
    log.info("  fit rows: %d, valid rows: %d, holdout rows: %d",
             is_fit.sum(), is_val.sum(), is_hold.sum())

    log.info("[3/4] training gradient boosting (with and without format)")
    tables, scores = [], {}
    for model, use_fmt in (("gbm", True), ("gbm_noformat", False)):
        X = np.concatenate([X18, Xfmt], axis=1) if use_fmt else X18
        names = feat_names + [f"fmt_{f}" for f in FORMATS] if use_fmt else feat_names
        neg, pos = (y[is_fit] == 0).sum(), (y[is_fit] == 1).sum()
        clf = lgb.LGBMClassifier(objective="binary", scale_pos_weight=neg / pos,
                                 num_leaves=63, min_child_samples=200,
                                 learning_rate=0.05, n_estimators=1000,
                                 importance_type="gain",
                                 random_state=7, deterministic=True, verbosity=-1)
        # Early stopping on valid AUC (not logloss): the evaluation ranks
        # transfers per day, so ranking quality is the objective.
        clf.fit(X[is_fit], y[is_fit], eval_X=X[is_val], eval_y=y[is_val],
                eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])
        auc = roc_auc_score(y[is_val], clf.predict_proba(X[is_val])[:, 1])
        log.info("  %s: best iteration %d, valid AUC %.4f",
                 model, clf.best_iteration_, auc)
        top = sorted(zip(names, clf.feature_importances_),
                     key=lambda t: -t[1])[:5]
        log.info("    top features: %s", ", ".join(f"{n}={v:.0f}" for n, v in top))
        scores[model] = clf.predict_proba(X[is_hold])[:, 1]

    hold_base = (df.filter(pl.col("day").is_between(ho0, ho1))
                 .join(memb.join(inst.select(["instance_id", "pattern_type"]), on="instance_id")
                       .group_by("transfer_id").agg(pl.col("pattern_type").first()),
                       on="transfer_id", how="left", maintain_order="left"))
    wide = hold_base.select("transfer_id", "timestamp", "day", "is_laundering", "pattern_type")
    log.info("[4/4] recall at fixed alerts/day budgets")
    for model, s in scores.items():
        hold = evaluate.rank_per_day(hold_base.with_columns(pl.Series("score", s)))
        wide = (wide.with_columns(pl.Series(f"score_{model}", s))
                    .join(hold.select("transfer_id", pl.col("rank").alias(f"rank_{model}")),
                          on="transfer_id", how="left", maintain_order="left"))
        tr = (evaluate.transfer_recall(hold, budget_list)
              .with_columns(pl.lit(model).alias("model"),
                            pl.lit("transfer").alias("granularity"),
                            pl.lit("all").alias("set")).select(COLS))
        ir = (evaluate.instance_recall(hold, memb, inst, budget_list)
              .with_columns(pl.lit(model).alias("model"),
                            pl.lit("instance").alias("granularity")).select(COLS))
        tables.extend([tr, ir])
        with pl.Config(tbl_rows=-1):
            log.info("\n%s transfer recall:\n%s", model,
                     tr.pivot(index="group", on="budget", values="recall").sort("group"))
            log.info("\n%s instance recall:\n%s", model,
                     ir.pivot(index=["set", "group"], on="budget", values="recall")
                     .sort(["set", "group"]))
    wide.write_parquet(data / "supervised_scores.parquet")
    pl.concat(tables).write_csv(out / "recall_supervised.csv")
    log.info("done -> %s", out / "recall_supervised.csv")


if __name__ == "__main__":
    typer.run(main)
