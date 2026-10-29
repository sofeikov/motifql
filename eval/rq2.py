#!/usr/bin/env python3
"""RQ2 harness: fuzzy vs exact (crisp) recall at fixed alert budgets (Nov 1).

Level-0 table: replays the fuzzy-sensitive motifs (round_trip,
layering_chain, scatter_gather) over the holdout split twice — fuzzy
unpruned (ranking needs sub-emit top_scores) and crisp pruned (exact
arm: binary scores, so every crisp-1 witness completes under pruning;
--verify-prune checks pruned == unpruned on a sample empirically).

Ranking and recall go through the shared `eval/evaluate.py` protocol
at the RQ1 budgets, so the rows are comparable by construction. The
harness also asserts the crisp differential — crisp_top ==
(fuzzy_top > 0) anchor-wise. Any mismatch is an evaluator bug, not a
finding: crisping thresholds the same leaves the fuzzy score combines.

Usage:
    uv run eval/rq2.py --data data/hi-small --workers 16
    uv run eval/rq2.py --data data/hi-small --max-anchors 20000 --workers 4
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import eval.rq1 as rq1  # noqa: E402
from eval import evaluate, splits  # noqa: E402
from motifql.degrees import libs_from_doc  # noqa: E402
from motifql.graph import TemporalGraph  # noqa: E402
from motifql.parser import parse_file  # noqa: E402
from motifql.planner import PathPlan, compile_motif  # noqa: E402

log = logging.getLogger("rq2")

MOTIFS = ["round_trip", "layering_chain", "scatter_gather"]


def main(
    data: Annotated[Path, typer.Option(help="Prepped dataset directory.")] = Path("data/hi-small"),
    holdout: Annotated[str, typer.Option(help="Holdout day range, inclusive.")] = f"{splits.HOLDOUT_START},{splits.HOLDOUT_END}",
    motifs: Annotated[str, typer.Option(help="Comma-separated motif names.")] = ",".join(MOTIFS),
    budgets: Annotated[str, typer.Option(help="Alerts/day budgets, comma-separated.")] = "50,100,500,1000,5000",
    max_anchors: Annotated[int, typer.Option(help="Cap holdout anchors (0 = all).")] = 0,
    verify_prune: Annotated[int, typer.Option(help="Crisp pruned-vs-unpruned check anchors (0 = off).")] = 20000,
    workers: Annotated[int, typer.Option(help="Worker processes (1 = serial).")] = 16,
    out: Annotated[Path, typer.Option(help="Directory for the recall CSVs.")] = Path("eval/results"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ho0, ho1 = (date.fromisoformat(s) for s in holdout.split(","))
    motifs_wanted = [m.strip() for m in motifs.split(",") if m.strip()]
    budget_list = [int(b) for b in budgets.split(",")]
    out.mkdir(parents=True, exist_ok=True)

    log.info("[1/4] loading transfers and building graph (queries look back)")
    df = pl.read_parquet(data / "transfers.parquet")
    graph = TemporalGraph.from_transfers(df)
    days = df.get_column("day").to_numpy()
    hold_pos = np.flatnonzero(
        (days >= np.datetime64(str(ho0))) & (days <= np.datetime64(str(ho1))))
    if max_anchors > 0:
        hold_pos = hold_pos[:max_anchors]
    n_ho_days = (ho1 - ho0).days + 1
    log.info("  graph edges: %d, holdout anchors: %d", graph.n_edges, len(hold_pos))

    log.info("[2/4] compiling motifs from examples/amlworld.motif")
    doc = parse_file(Path("examples/amlworld.motif"))
    libs = libs_from_doc(doc)
    # No stats: the cost check is a live-engine guard, and the refit
    # layering (*5..8) trips E_COST under it (est ~1e8 vs 1e7 budget)
    # while measuring ~1e3 expansions in practice — the estimate is
    # loose by design (LANGUAGE.md section 5). The replay records the
    # true per-anchor cost in rq2_summary.csv instead.
    plans = {m: compile_motif(doc, m) for m in motifs_wanted}

    rq1._GRAPH, rq1._LIBS = graph, libs
    scored: dict[str, dict[str, np.ndarray]] = {}
    summary_rows: list[dict] = []
    log.info("[3/4] replaying holdout: crisp pruned full, fuzzy via screen")
    for name, plan in plans.items():
        rq1._PLAN = plan
        # Crisp arm first, full holdout, pruned. Every crisp-1 witness
        # completes under pruning (all its leaves read 1 along the
        # way), so pruned == unpruned for the crisp arm.
        rq1._PRUNE, rq1._CRISP = True, True
        cr_top, cr_found, cr_exps, cr_tr = rq1._replay(hold_pos, workers, f"{name} crisp")
        if isinstance(plan, PathPlan):
            # Two-phase fuzzy: the crisp screen is exact on completed
            # anchors (crisp 0 + completed ⟹ no all-positive-leaf
            # witness exists ⟹ fuzzy top is 0), so only crisp-positive
            # or crisp-truncated anchors earn a fuzzy replay. Unpruned
            # *5..8 costs ~10ms/anchor and truncates ~5%; the screen
            # cuts the full replay to a few hundred anchors.
            sub_idx = np.flatnonzero((cr_top > 0) | cr_tr)
            fz_top = np.zeros(len(hold_pos))
            fz_found = np.zeros(len(hold_pos), dtype=bool)
            fz_exps = np.zeros(len(hold_pos))
            fz_tr = np.zeros(len(hold_pos), dtype=bool)
            if len(sub_idx):
                rq1._PRUNE, rq1._CRISP = False, False
                s_top, s_found, s_exps, s_tr = rq1._replay(
                    hold_pos[sub_idx], workers, f"{name} fuzzy-sub")
                fz_top[sub_idx], fz_found[sub_idx] = s_top, s_found
                fz_exps[sub_idx], fz_tr[sub_idx] = s_exps, s_tr
        else:
            # Group executors never prune: run the fuzzy arm fully.
            rq1._PRUNE, rq1._CRISP = False, False
            fz_top, fz_found, fz_exps, fz_tr = rq1._replay(
                hold_pos, workers, f"{name} fuzzy")
        arms = {"fuzzy": {"top": fz_top, "found": fz_found},
                "crisp": {"top": cr_top, "found": cr_found}}
        scored[name] = arms
        all_idx = np.arange(len(hold_pos))
        for arm, top, found, exps, trunc, ridx in (
                ("fuzzy", fz_top, fz_found, fz_exps, fz_tr,
                 sub_idx if isinstance(plan, PathPlan) else all_idx),
                ("crisp", cr_top, cr_found, cr_exps, cr_tr, all_idx)):
            # Expansions average over replayed anchors only (shortcut
            # anchors cost nothing, so including them would lie down).
            exps_mean = float(exps[ridx].mean()) if len(ridx) else 0.0
            replayed = len(ridx)
            summary_rows.append({
                "motif": name, "arm": arm, "anchors": len(top),
                "replayed": replayed,
                "fires": int(found.sum()), "fires_per_day": float(found.sum()) / n_ho_days,
                "nonzero": int((top > 0).sum()),
                "nonzero_per_day": float((top > 0).sum()) / n_ho_days,
                "top_p50": float(np.percentile(top, 50)),
                "top_p99": float(np.percentile(top, 99)),
                "top_p999": float(np.percentile(top, 99.9)),
                "mean_expansions": exps_mean,
                "truncated_rate": float(trunc.mean()),
            })
            log.info("  %s %s: fires %d (%.1f/day) nonzero %.1f/day "
                     "mean expansions %.1f truncated %.2f%% replayed %d",
                     name, arm, int(found.sum()), found.sum() / n_ho_days,
                     (top > 0).sum() / n_ho_days, exps_mean,
                     100 * trunc.mean(), replayed)

        # Crisp differential, scoped to anchors where both arms
        # completed: binary scores, positive exactly where the fuzzy
        # top is positive. Truncation strikes the two searches
        # differently (unpruned fuzzy truncates far more), so anchors
        # where either truncated are reported, not asserted.
        nonbinary = int(((cr_top > 0) & (cr_top < 1)).sum())
        both_ok = ~cr_tr & ~fz_tr
        mismatch = int((((cr_top > 0) != (fz_top > 0)) & both_ok).sum())
        trunc_mm = int((((cr_top > 0) != (fz_top > 0)) & ~both_ok).sum())
        log.info("  %s differential: nonbinary=%d mismatch=%d "
                 "(+%d on truncated anchors)", name, nonbinary, mismatch, trunc_mm)
        if nonbinary or mismatch:
            raise SystemExit(f"crisp differential failed for {name}: "
                             f"nonbinary={nonbinary} mismatch={mismatch}")

        if 0 < verify_prune <= len(hold_pos):
            sub = hold_pos[:verify_prune]
            rq1._PRUNE, rq1._CRISP = True, True
            cr_sub, _, _, _ = rq1._replay(sub, min(workers, 4), f"{name} verify-cp")
            rq1._PRUNE, rq1._CRISP = False, True
            cu_sub, _, _, cu_tr = rq1._replay(sub, min(workers, 4), f"{name} verify-cu")
            # Pruning only skips zero-leaf branches and truncates less
            # (subset of expansions), so pruned ⊇ unpruned: every
            # unpruned-1 must be pruned-1. The reverse direction may
            # differ exactly where unpruned truncated.
            lost = int(((cu_sub > 0) & ~(cr_sub > 0)).sum())
            gained_trunc = int(((cr_sub > 0) & ~(cu_sub > 0) & cu_tr).sum())
            gained_ok = int(((cr_sub > 0) & ~(cu_sub > 0) & ~cu_tr).sum())
            if lost or gained_ok:
                raise SystemExit(f"crisp prune check failed for {name}: "
                                 f"lost={lost} gained_completed={gained_ok}")
            log.info("  %s verify: pruned superset holds (+%d on unpruned-truncated)",
                     name, gained_trunc)
            # Fuzzy shortcut check: direct unpruned tops on the sample
            # must equal the two-phase assembly (deterministic per
            # anchor, so truncation included).
            rq1._PRUNE, rq1._CRISP = False, False
            fz_sub, _, _, _ = rq1._replay(sub, min(workers, 4), f"{name} verify-fz")
            if not np.array_equal(fz_sub, fz_top[:verify_prune]):
                raise SystemExit(f"fuzzy shortcut disagrees for {name}")
            log.info("  %s verify: %d-anchor sample agrees", name, verify_prune)

    pl.from_dicts(summary_rows).write_csv(out / "rq2_summary.csv")
    hold_ids = df.get_column("transfer_id").to_numpy()[hold_pos]
    cols = {"transfer_id": hold_ids}
    for m in motifs_wanted:
        cols[f"{m}__fuzzy"] = scored[m]["fuzzy"]["top"]
        cols[f"{m}__crisp"] = scored[m]["crisp"]["top"]
    pl.DataFrame(cols).write_parquet(out / "rq2_holdout_scores.parquet")
    log.info("  wrote %s and rq2_holdout_scores.parquet", out / "rq2_summary.csv")

    log.info("[4/4] ranking holdout per day and scoring recall at fixed budgets")
    memb = pl.read_parquet(data / "pattern_members.parquet")
    inst = splits.assign_instance_splits(
        pl.read_parquet(data / "pattern_instances.parquet"), memb, df)
    labels_one = (memb.join(inst.select(["instance_id", "pattern_type"]), on="instance_id")
                  .group_by("transfer_id").agg(pl.col("pattern_type").first()))
    hold_base = (df.filter(pl.col("day").is_between(ho0, ho1))
                 .head(len(hold_pos))
                 .join(labels_one, on="transfer_id", how="left", maintain_order="left"))
    tr_rows, inst_rows = [], []
    for name in motifs_wanted:
        for arm in ("fuzzy", "crisp"):
            hold = evaluate.rank_per_day(
                hold_base.with_columns(pl.Series("score", scored[name][arm]["top"])))
            tr_rows.append(
                evaluate.transfer_recall(hold, budget_list)
                .with_columns(pl.lit(name).alias("motif"), pl.lit(arm).alias("arm"),
                              pl.lit("transfer").alias("granularity")))
            inst_rows.append(
                evaluate.instance_recall(hold, memb, inst, budget_list)
                .with_columns(pl.lit(name).alias("motif"), pl.lit(arm).alias("arm")))
    pl.concat(tr_rows).write_csv(out / "rq2_recall_transfer.csv")
    pl.concat(inst_rows).write_csv(out / "rq2_recall_instance.csv")

    log.info("RQ2 table (holdout instance recall, completable set)")
    inst_all = pl.concat(inst_rows).filter(pl.col("set") == "completable")
    for row in inst_all.sort(["motif", "arm", "budget", "group"]).iter_rows(named=True):
        log.info("  %-14s %-5s budget %5d %-8s recall %.3f (%d/%d)",
                 row["motif"], row["arm"], row["budget"], row["group"],
                 row["recall"] or 0.0, row["n_hit"], row["n_total"])
    log.info("done")


if __name__ == "__main__":
    typer.run(main)
