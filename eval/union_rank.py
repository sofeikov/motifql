#!/usr/bin/env python3
"""Motif-score unions over saved per-anchor scores (#15).

Replaces the lost /tmp/unionrank.py behind the RQ2 report §7. Two ways
to combine per-motif top_scores, both ranked per day through the shared
`eval/evaluate.py` protocol:

* minrank: per-motif `rank_per_day`, alert by best (min) rank — the
  standard fix for uncalibrated cross-motif scores.
* max: naive max-of-raw-scores, which collapses (fan background groups
  score 1.0 on counts and bury path truth at 0.5-0.9).

Inputs are score parquets with transfer_id plus `{motif}__{arm}` columns
(`rq2_holdout_scores.parquet`). Ranking only — no replay.

Usage:
    uv run eval/union_rank.py --scores eval/results/rq2_holdout_scores.parquet
    uv run eval/union_rank.py --motifs round_trip,layering_chain
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval import evaluate, splits  # noqa: E402

log = logging.getLogger("union_rank")


def main(
    data: Annotated[Path, typer.Option(help="Prepped dataset directory.")] = Path("data/hi-small"),
    scores: Annotated[Path, typer.Option(help="Score parquet (transfer_id + {motif}__{arm}).")] = Path(
        "eval/results/rq2_holdout_scores.parquet"),
    motifs: Annotated[str, typer.Option(help="Comma-separated motif names.")] = (
        "round_trip,layering_chain,scatter_gather"),
    arms: Annotated[str, typer.Option(help="Comma-separated arms (column suffixes).")] = "fuzzy,crisp",
    modes: Annotated[str, typer.Option(help="Comma-separated: minrank,max.")] = "minrank,max",
    budgets: Annotated[str, typer.Option(help="Alerts/day budgets, comma-separated.")] = "50,100,500,1000,5000",
    out: Annotated[Path, typer.Option(help="Directory for the recall CSVs.")] = Path("eval/results"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    motifs_wanted = [m.strip() for m in motifs.split(",") if m.strip()]
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    mode_list = [m.strip() for m in modes.split(",") if m.strip()]
    if any(m not in ("minrank", "max") for m in mode_list):
        raise SystemExit(f"--modes {modes!r}: use minrank,max")
    budget_list = [int(b) for b in budgets.split(",")]
    out.mkdir(parents=True, exist_ok=True)

    wide = pl.read_parquet(scores)
    for m in motifs_wanted:
        for a in arm_list:
            if f"{m}__{a}" not in wide.columns:
                raise SystemExit(f"{scores} has no column {m}__{a}")
    df = pl.read_parquet(data / "transfers.parquet")
    memb = pl.read_parquet(data / "pattern_members.parquet")
    inst = splits.assign_instance_splits(
        pl.read_parquet(data / "pattern_instances.parquet"), memb, df)
    labels_one = (memb.join(inst.select(["instance_id", "pattern_type"]), on="instance_id")
                  .group_by("transfer_id").agg(pl.col("pattern_type").first()))
    base = (wide.join(df.select("transfer_id", "day", "is_laundering"),
                         on="transfer_id", how="left")
            .join(labels_one, on="transfer_id", how="left"))

    tr_rows, inst_rows = [], []
    for arm in arm_list:
        cols = [f"{m}__{arm}" for m in motifs_wanted]
        for mode in mode_list:
            if mode == "max":
                combined = base.with_columns(
                    pl.max_horizontal(cols).alias("score"))
                hold = evaluate.rank_per_day(combined)
            else:
                ranks = base.select("transfer_id")
                for c in cols:
                    ranked = evaluate.rank_per_day(
                        base.select("transfer_id", "day", pl.col(c).alias("score")))
                    ranks = ranks.join(
                        ranked.select("transfer_id", pl.col("rank").alias(f"rank_{c}")),
                        on="transfer_id", how="left", maintain_order="left")
                best = ranks.with_columns(
                    pl.min_horizontal([f"rank_{c}" for c in cols]).alias("best"))
                # score desc == best-rank asc; the keyed hash still
                # breaks best-rank ties, as in every other ranking.
                hold = evaluate.rank_per_day(
                    base.join(best.select("transfer_id", "best"), on="transfer_id",
                              how="left", maintain_order="left")
                    .with_columns((-pl.col("best")).alias("score")))
            tag = {"mode": mode, "arm": arm,
                   "motifs": "+".join(motifs_wanted)}
            tr_rows.append(
                evaluate.transfer_recall(hold, budget_list)
                .with_columns(pl.lit(tag["mode"]).alias("mode"),
                              pl.lit(tag["arm"]).alias("arm"),
                              pl.lit(tag["motifs"]).alias("motifs"),
                              pl.lit("transfer").alias("granularity")))
            inst_rows.append(
                evaluate.instance_recall(hold, memb, inst, budget_list)
                .with_columns(pl.lit(tag["mode"]).alias("mode"),
                              pl.lit(tag["arm"]).alias("arm"),
                              pl.lit(tag["motifs"]).alias("motifs")))
            hits = {r["budget"]: r["n_hit"] for r in
                    tr_rows[-1].filter(pl.col("group") == "ALL_FLAGGED")
                    .select("budget", "n_hit").iter_rows(named=True)}
            log.info("  %s %s ALL_FLAGGED: %s", mode, arm,
                     ", ".join(f"@{b}={hits.get(b, 0)}" for b in budget_list))
    pl.concat(tr_rows).write_csv(out / "union_recall_transfer.csv")
    pl.concat(inst_rows).write_csv(out / "union_recall_instance.csv")
    log.info("  wrote union_recall_{transfer,instance}.csv")
    log.info("done")


if __name__ == "__main__":
    typer.run(main)
