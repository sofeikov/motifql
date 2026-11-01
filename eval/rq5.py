#!/usr/bin/env python3
"""RQ5-lite: hybrid motif + GBM model (#21).

Two fusion arms over saved per-anchor scores, both ranked per day through
the shared `eval/evaluate.py` protocol and reported at operational budgets:

* min-rank union of {3 motifs + GBM} via `eval/union_rank.min_rank_union`
* stacked GBM: `eval/supervised` gbm_own features plus 3 motif top_scores,
  retrained train-only with the same hyperparameters/seed policy
  (`eval/supervised.train_gbm`)

Train/holdout separation: motif texts frozen (`examples/amlworld.motif`,
α=emit); no fitting on holdout. Train motif scores are replayed fresh with
the current engine (rq1_train_scores.parquet predates the Nov 1 tuning and
the issue-15 engine fixes, so reuse would skew train vs holdout). Splits
come from `eval/splits.py`; instance splits are recomputed from the current
frame. The last train day (Sep 7) is the early-stopping validation set.

Usage:
    uv run eval/rq5.py --data data/hi-small --workers 16
    uv run eval/rq5.py --data data/hi-small --train-scores eval/results/rq5_train_scores.parquet
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import eval.rq1 as rq1  # noqa: E402
from eval import evaluate, features, splits, supervised, union_rank  # noqa: E402
from motifql.degrees import libs_from_doc  # noqa: E402
from motifql.graph import TemporalGraph  # noqa: E402
from motifql.matcher import EXACT_FLOOR  # noqa: E402
from motifql.parser import parse_file  # noqa: E402
from motifql.planner import compile_motif  # noqa: E402

log = logging.getLogger("rq5")

MOTIFS = ["round_trip", "layering_chain", "scatter_gather"]
COLS = ["model", "granularity", "set", "budget", "group", "n_total", "n_hit", "recall"]
GBM_COL = "gbm_own"


def check_coverage(have: set[int], want: set[int], name: str) -> None:
    """Fail unless a score file covers exactly the expected transfer ids."""
    if have != want:
        missing = len(want - have)
        extra = len(have - want)
        raise SystemExit(
            f"{name}: id mismatch (missing={missing} extra={extra}); "
            "score file does not cover the split it claims")


def replay_train_motifs(df: pl.DataFrame, motifs: list[str], tr0: date, tr1: date,
                        workers: int, out_path: Path) -> pl.DataFrame:
    """Fresh train motif scores (fuzzy exact-floor, current engine/texts).

    Mirrors the RQ2 fuzzy arm: pruned search from EXACT_FLOOR for path
    motifs (exact best witness), exact-by-construction groups. The graph
    covers all days like RQ2 (queries look back only, so future edges are
    inert). Writes transfer_id + {motif}__fuzzy to `out_path`.
    """
    graph = TemporalGraph.from_transfers(df)
    days = df.get_column("day").to_numpy()
    train_pos = np.flatnonzero(
        (days >= np.datetime64(str(tr0))) & (days <= np.datetime64(str(tr1))))
    log.info("  graph edges: %d, train anchors: %d", graph.n_edges, len(train_pos))
    doc = parse_file(Path("examples/amlworld.motif"))
    libs = libs_from_doc(doc)
    rq1._GRAPH, rq1._LIBS = graph, libs
    cols: dict[str, np.ndarray] = {
        "transfer_id": df.get_column("transfer_id").to_numpy()[train_pos]}
    n_days = (tr1 - tr0).days + 1
    for name in motifs:
        rq1._PLAN = compile_motif(doc, name)
        rq1._PRUNE, rq1._CRISP, rq1._CALPHA, rq1._FLOOR = True, False, 0.0, EXACT_FLOOR
        top, found, exps, trunc = rq1._replay(train_pos, workers, f"{name} train")
        cols[f"{name}__fuzzy"] = top
        log.info("  %s: fires %d (%.1f/day) nonzero %d mean expansions %.1f "
                 "truncated %.3f%%", name, int(found.sum()),
                 found.sum() / n_days, int((top > 0).sum()),
                 float(exps.mean()), 100 * float(trunc.mean()))
    frame = pl.DataFrame(cols)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(out_path)
    log.info("  wrote %s", out_path)
    return frame


def main(
    data: Annotated[Path, typer.Option(help="Prepped dataset directory.")] = Path("data/hi-small"),
    motif_scores: Annotated[Path, typer.Option(help="Holdout motif scores.")] = Path(
        "eval/results/rq2_holdout_scores.parquet"),
    gbm_scores: Annotated[Path | None, typer.Option(
        help="GBM holdout scores (default: <data>/supervised_scores.parquet).")] = None,
    train_scores: Annotated[Path, typer.Option(
        help="Train motif scores: reused when present, else replayed fresh.")] = Path(
        "eval/results/rq5_train_scores.parquet"),
    gbm_model: Annotated[str, typer.Option(help="GBM variant (score_<name> column).")] = "gbm_own",
    motifs: Annotated[str, typer.Option(help="Comma-separated motif names.")] = ",".join(MOTIFS),
    budgets: Annotated[str, typer.Option(help="Alerts/day budgets, comma-separated.")] = "50,100,500",
    workers: Annotated[int, typer.Option(help="Worker processes for train replay.")] = 16,
    out: Annotated[Path, typer.Option(help="Directory for the recall CSVs.")] = Path("eval/results"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    motifs_wanted = [m.strip() for m in motifs.split(",") if m.strip()]
    motif_cols = [f"{m}__fuzzy" for m in motifs_wanted]
    budget_list = [int(b) for b in budgets.split(",")]
    gbm_path = gbm_scores or (data / "supervised_scores.parquet")
    gbm_col = f"score_{gbm_model}"
    out.mkdir(parents=True, exist_ok=True)
    replayed_train = False

    log.info("[1/5] loading transfers, labels and holdout scores")
    df = pl.read_parquet(data / "transfers.parquet")
    memb = pl.read_parquet(data / "pattern_members.parquet")
    inst = splits.assign_instance_splits(
        pl.read_parquet(data / "pattern_instances.parquet"), memb, df)
    tr0, tr1 = splits.TRAIN_START, splits.TRAIN_END
    ho0, ho1 = splits.HOLDOUT_START, splits.HOLDOUT_END
    wide_motif = pl.read_parquet(motif_scores)
    for c in motif_cols:
        if c not in wide_motif.columns:
            raise SystemExit(f"{motif_scores} has no column {c}")
    wide_gbm = pl.read_parquet(gbm_path)
    if gbm_col not in wide_gbm.columns:
        raise SystemExit(f"{gbm_path} has no column {gbm_col}")
    want_hold = set(df.filter(pl.col("day").is_between(ho0, ho1))["transfer_id"].to_list())
    check_coverage(set(wide_motif["transfer_id"].to_list()), want_hold,
                   f"motif scores {motif_scores.name}")
    check_coverage(set(wide_gbm["transfer_id"].to_list()), want_hold,
                   f"gbm scores {gbm_path.name}")
    log.info("  holdout anchors: %d (motif + gbm ids match, days %s..%s)", len(want_hold), ho0, ho1)

    labels_one = (memb.join(inst.select(["instance_id", "pattern_type"]), on="instance_id")
                  .group_by("transfer_id").agg(pl.col("pattern_type").first()))
    hold_base = (df.filter(pl.col("day").is_between(ho0, ho1))
                 .join(labels_one, on="transfer_id", how="left", maintain_order="left")
                 .join(wide_motif.select(["transfer_id", *motif_cols]),
                       on="transfer_id", how="left", maintain_order="left")
                 .join(wide_gbm.select("transfer_id", pl.col(gbm_col).alias(GBM_COL)),
                       on="transfer_id", how="left", maintain_order="left"))
    null_scores = hold_base.filter(
        pl.any_horizontal([pl.col(c).is_null() for c in [*motif_cols, GBM_COL]])
    ).select(pl.len()).item()
    if null_scores:
        raise SystemExit(f"holdout join left {null_scores} anchors without scores")
    # Saved GBM days must agree with transfers (key/day misalignment check).
    gbm_day = wide_gbm.join(df.select("transfer_id", "day"),
                            on="transfer_id", how="left", maintain_order="left")
    if "day_right" in gbm_day.columns and (gbm_day["day"] != gbm_day["day_right"]).any():
        raise SystemExit(f"{gbm_path.name}: saved days disagree with transfers.parquet")

    log.info("[2/5] train motif scores (reuse or fresh replay)")
    want_train = set(df.filter(pl.col("day").is_between(tr0, tr1))["transfer_id"].to_list())
    if train_scores.exists():
        train_motif = pl.read_parquet(train_scores)
        for c in motif_cols:
            if c not in train_motif.columns:
                raise SystemExit(f"{train_scores} has no column {c}")
        check_coverage(set(train_motif["transfer_id"].to_list()), want_train,
                       f"train scores {train_scores.name}")
        log.info("  reusing %s (%d anchors)", train_scores, len(train_motif))
    else:
        train_motif = replay_train_motifs(df, motifs_wanted, tr0, tr1, workers, train_scores)
        replayed_train = True

    log.info("[3/5] training stacked GBM (own-amount features + 3 motif scores)")
    df_keys = df.with_columns(
        (pl.col("from_bank") + "/" + pl.col("from_account")).alias("from_key"),
        (pl.col("to_bank") + "/" + pl.col("to_account")).alias("to_key"))
    feats = features.compute_all(df_keys)
    feat_names = [c for c in feats.columns if c != "transfer_id"]
    own = supervised.own_transfer_features(df_keys)
    full = (df_keys.select("transfer_id", "day", "is_laundering", "payment_format")
            .join(feats, on="transfer_id", how="left", maintain_order="left")
            .join(own, on="transfer_id", how="left", maintain_order="left")
            .with_columns(pl.col(feat_names).cast(pl.Float32),
                          pl.col("own_log_amount").cast(pl.Float32),
                          *[(pl.col("payment_format") == f).cast(pl.Float32).alias(f"fmt_{f}")
                            for f in supervised.FORMATS]))
    motif_all = pl.concat([train_motif.select(["transfer_id", *motif_cols]),
                           wide_motif.select(["transfer_id", *motif_cols])],
                          how="vertical")
    full = (full.join(motif_all, on="transfer_id", how="left", maintain_order="left")
            .with_columns(pl.col(motif_cols).fill_null(0.0).cast(pl.Float32)))
    day = full["day"].to_numpy()
    y = full["is_laundering"].to_numpy()
    parts = [full.select(feat_names).to_numpy(),
             full.select([f"fmt_{f}" for f in supervised.FORMATS]).to_numpy(),
             full.select(supervised.OWN_FEATS).to_numpy(),
             full.select(motif_cols).to_numpy()]
    names = (list(feat_names) + [f"fmt_{f}" for f in supervised.FORMATS]
             + list(supervised.OWN_FEATS) + list(motif_cols))
    X = np.concatenate(parts, axis=1)
    is_fit = (day >= np.datetime64(str(tr0))) & (day < np.datetime64(str(tr1)))
    is_val = day == np.datetime64(str(tr1))
    is_hold = (day >= np.datetime64(str(ho0))) & (day <= np.datetime64(str(ho1)))
    log.info("  fit rows: %d, valid rows: %d, holdout rows: %d",
             is_fit.sum(), is_val.sum(), is_hold.sum())
    # Train/holdout rows must carry real motif scores, never the 0.0 fill
    # for unscored warmup/tail days.
    scored_ids = set(motif_all["transfer_id"].to_list())
    for mask, split in ((is_fit | is_val, "train"), (is_hold, "holdout")):
        missing = int((~np.isin(full["transfer_id"].to_numpy()[mask],
                                np.fromiter(scored_ids, dtype=np.int64))).sum())
        if missing:
            raise SystemExit(f"stacked {split}: {missing} rows lack motif scores")
    clf = supervised.train_gbm(X[is_fit], y[is_fit], X[is_val], y[is_val])
    auc = roc_auc_score(y[is_val], clf.predict_proba(X[is_val])[:, 1])
    log.info("  stacked: best iteration %d, valid AUC %.4f", clf.best_iteration_, auc)
    top = sorted(zip(names, clf.feature_importances_), key=lambda t: -t[1])[:8]
    log.info("    top features: %s", ", ".join(f"{n}={v:.0f}" for n, v in top))
    stacked_scores = clf.predict_proba(X[is_hold])[:, 1]
    stacked_ids = full.filter(pl.Series("mask", is_hold))["transfer_id"].to_list()
    hold_base = hold_base.join(
        pl.DataFrame({"transfer_id": stacked_ids, "stacked": stacked_scores}),
        on="transfer_id", how="left", maintain_order="left")

    log.info("[4/5] ranking fusion arms per day")
    hold_keyed = evaluate.with_tiebreak(hold_base)
    arms: dict[str, pl.DataFrame] = {}
    arms["motif_union"] = union_rank.min_rank_union(hold_base, motif_cols)
    arms["gbm"] = evaluate.rank_per_day(
        hold_keyed.with_columns(pl.col(GBM_COL).alias("score")))
    arms["fusion_minrank"] = union_rank.min_rank_union(hold_base, [*motif_cols, GBM_COL])
    arms["stacked"] = evaluate.rank_per_day(
        hold_keyed.with_columns(pl.col("stacked").alias("score")))
    # Best-single headroom for the dilution check (union vs best single).
    for c in motif_cols:
        arms[f"single_{c.replace('__fuzzy', '')}"] = evaluate.rank_per_day(
            hold_keyed.with_columns(pl.col(c).alias("score")))

    log.info("[5/5] recall at fixed alerts/day budgets")
    tables = []
    for model, hold in arms.items():
        tr = (evaluate.transfer_recall(hold, budget_list)
              .with_columns(pl.lit(model).alias("model"),
                            pl.lit("transfer").alias("granularity"),
                            pl.lit("all").alias("set")).select(COLS))
        ir = (evaluate.instance_recall(hold, memb, inst, budget_list)
              .with_columns(pl.lit(model).alias("model"),
                            pl.lit("instance").alias("granularity")).select(COLS))
        tables.extend([tr, ir])
        hits = {r["budget"]: r["n_hit"] for r in
                tr.filter(pl.col("group") == "ALL_FLAGGED")
                .select("budget", "n_hit").iter_rows(named=True)}
        log.info("  %-22s ALL_FLAGGED: %s", model,
                 ", ".join(f"@{b}={hits.get(b, 0)}" for b in budget_list))
    res = pl.concat(tables)
    res.filter(pl.col("granularity") == "transfer").write_csv(out / "rq5_recall_transfer.csv")
    res.filter(pl.col("granularity") == "instance").write_csv(out / "rq5_recall_instance.csv")
    hold_base.select("transfer_id", pl.col("stacked").alias("score_stacked")).write_parquet(
        out / "rq5_stacked_scores.parquet")

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "unknown"
    motif_text = Path("examples/amlworld.motif").read_bytes()
    provenance = {
        "commit": commit,
        "cmd": sys.argv,
        "data": str(data),
        "motif_scores": {"path": str(motif_scores), "rows": len(wide_motif)},
        "gbm_scores": {"path": str(gbm_path), "column": gbm_col, "rows": len(wide_gbm)},
        "train_scores": {"path": str(train_scores), "rows": len(train_motif),
                         "replayed": replayed_train},
        "splits": {"train": [str(tr0), str(tr1)], "holdout": [str(ho0), str(ho1)],
                   "fit": f"{tr0}..day-before-{tr1}", "valid": str(tr1)},
        "motifs": motifs_wanted,
        "motif_file_sha256": hashlib.sha256(motif_text).hexdigest(),
        "budgets": budget_list,
        "gbm_params": {k: (v if isinstance(v, (str, int, float, bool)) else str(v))
                       for k, v in supervised.GBM_PARAMS.items()},
        "early_stopping_rounds": supervised.GBM_EARLY_STOPPING_ROUNDS,
        "stacked": {"best_iteration": int(clf.best_iteration_), "valid_auc": float(auc),
                    "top_features": [(n, float(v)) for n, v in top]},
    }
    (out / "rq5_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    log.info("done -> %s, %s, rq5_provenance.json",
             out / "rq5_recall_transfer.csv", out / "rq5_recall_instance.csv")


if __name__ == "__main__":
    typer.run(main)
