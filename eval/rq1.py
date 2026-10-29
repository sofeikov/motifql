#!/usr/bin/env python3
"""RQ1 harness: hand-written motif recall at fixed alert budgets (Oct 30).

Replays every v0 motif in `examples/amlworld.motif` (2 path + 5 group +
butterfly) over the train and holdout splits and reports, per motif:

* fires/day on train and holdout (background rate next to every recall),
* transfer recall and instance recall at fixed alerts/day budgets on holdout,
* mean expansions and truncation rate (hub-cap accounting).

Ranking is by top_score (best group/witness even below emit) with transfer
id as the deterministic tie-break, shared with the baselines through
`eval/evaluate.py`, so the rows are comparable by construction. The tuning
guardrail holds: breakpoints ship as written; any tuning is Nov 1 work fit
on train only.

Usage:
    uv run eval/rq1.py --data data/hi-small --workers 16
    uv run eval/rq1.py --data data/hi-small --motifs fan_out,fan_in --max-anchors 50000
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import sys
import time
from datetime import date
from pathlib import Path
from typing import Annotated

import numpy as np
import numpy.typing as npt
import polars as pl
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from eval import evaluate, splits  # noqa: E402
from motifql.degrees import libs_from_doc  # noqa: E402
from motifql.graph import TemporalGraph  # noqa: E402
from motifql.groups import score_batch  # noqa: E402
from motifql.parser import parse_file  # noqa: E402
from motifql.planner import compile_motif  # noqa: E402

log = logging.getLogger("rq1")

MOTIFS = ["round_trip", "layering_chain", "fan_out", "fan_in",
          "scatter_gather", "gather_scatter", "structuring_fan_in",
          "bipartite_butterfly"]

_GRAPH: TemporalGraph | None = None
_PLAN = None
_LIBS = None
_PRUNE = False
_CRISP = False
_CALPHA = 0.0
_FLOOR = None


def _score_job(args: tuple[npt.NDArray[np.int64], bool]
               ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64],
                          npt.NDArray[bool], npt.NDArray[np.float64],
                          npt.NDArray[bool]]:
    """Score one anchor shard; positions first so results realign."""
    positions, _ = args
    assert _GRAPH is not None and _PLAN is not None and _LIBS is not None
    # Unpruned: RQ1 ranks by the best witness even below emit, and
    # pruning zeroes every sub-emit top_score. (RQ2 sets the globals for
    # its arms, including floor=EXACT_FLOOR for the fuzzy arm; RQ1
    # defaults are unchanged.) score_batch equals a score_any loop over
    # the shard and runs serially in this process (no numba threads, so
    # the fork Pool stays safe).
    top, found, exps, trunc = score_batch(
        _GRAPH, positions, _PLAN, _LIBS, prune=_PRUNE, crisp=_CRISP,
        crisp_alpha=_CALPHA, floor=_FLOOR)
    return positions, top, found, exps, trunc


def _replay(positions: npt.NDArray[np.int64], workers: int,
            label: str) -> tuple[npt.NDArray[np.float64], npt.NDArray[bool],
                                 npt.NDArray[np.float64], npt.NDArray[bool]]:
    t0 = time.perf_counter()
    if workers <= 1:
        _, *out = _score_job((positions, True))
    else:
        # Zero anchors: no scoring and no side effects, but the compiled
        # kernels are loaded here once and inherited by every forked
        # worker, instead of each worker loading them again (~0.6 s).
        _score_job((positions[:0], True))
        n_shards = 8 * workers
        shards = [positions[i::n_shards] for i in range(n_shards)]
        shards = [s for s in shards if len(s)]
        ctx = mp.get_context("fork")
        parts = []
        with ctx.Pool(processes=workers) as pool:
            for done, part in enumerate(
                pool.imap_unordered(_score_job, [(s, True) for s in shards]), start=1
            ):
                parts.append(part)
                if done % workers == 0:
                    log.info("  ... %s %d/%d shards", label, done, len(shards))
        key = np.concatenate([p[0] for p in parts])
        back = np.searchsorted(positions, key)
        out = tuple(
            _scatter(np.concatenate([p[k] for p in parts]), back, len(positions), dt)
            for k, dt in enumerate([np.float64, bool, np.float64, bool], start=1))
    log.info("  %s: %.1fs wall", label, time.perf_counter() - t0)
    return out


def _scatter(vals: npt.NDArray, back: npt.NDArray[np.int64], n: int, dt) -> npt.NDArray:
    aligned = np.empty(n, dtype=dt)
    aligned[back] = vals
    return aligned


def main(
    data: Annotated[Path, typer.Option(help="Prepped dataset directory.")] = Path("data/hi-small"),
    train: Annotated[str, typer.Option(help="Train day range, inclusive.")] = f"{splits.TRAIN_START},{splits.TRAIN_END}",
    holdout: Annotated[str, typer.Option(help="Holdout day range, inclusive.")] = f"{splits.HOLDOUT_START},{splits.HOLDOUT_END}",
    motifs: Annotated[str, typer.Option(help="Comma-separated motif names.")] = ",".join(MOTIFS),
    budgets: Annotated[str, typer.Option(help="Alerts/day budgets, comma-separated.")] = "50,100,500,1000,5000",
    max_anchors: Annotated[int, typer.Option(help="Cap anchors per split (0 = all).")] = 0,
    workers: Annotated[int, typer.Option(help="Worker processes (1 = serial).")] = 16,
    out: Annotated[Path, typer.Option(help="Directory for the recall CSVs.")] = Path("eval/results"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tr0, tr1 = (date.fromisoformat(s) for s in train.split(","))
    ho0, ho1 = (date.fromisoformat(s) for s in holdout.split(","))
    motifs_wanted = [m.strip() for m in motifs.split(",") if m.strip()]
    budget_list = [int(b) for b in budgets.split(",")]
    out.mkdir(parents=True, exist_ok=True)

    log.info("[1/5] loading transfers and building graph (all days; queries look back)")
    df = pl.read_parquet(data / "transfers.parquet")
    graph = TemporalGraph.from_transfers(df)
    days = df.get_column("day").to_numpy()
    train_pos = np.flatnonzero(
        (days >= np.datetime64(str(tr0))) & (days <= np.datetime64(str(tr1))))
    hold_pos = np.flatnonzero(
        (days >= np.datetime64(str(ho0))) & (days <= np.datetime64(str(ho1))))
    if max_anchors > 0:
        train_pos = train_pos[:max_anchors]
        hold_pos = hold_pos[:max_anchors]
    n_tr_days = (tr1 - tr0).days + 1
    n_ho_days = (ho1 - ho0).days + 1
    log.info("  graph edges: %d, train anchors: %d, holdout anchors: %d",
             graph.n_edges, len(train_pos), len(hold_pos))

    log.info("[2/5] compiling motifs from examples/amlworld.motif")
    doc = parse_file(Path("examples/amlworld.motif"))
    libs = libs_from_doc(doc)
    plans = {m: compile_motif(doc, m) for m in motifs_wanted}
    for m, p in plans.items():
        log.info("  %s: %s window=%.0fs emit=%.2f",
                 m, type(p).__name__, p.window_s, p.emit)

    global _GRAPH, _PLAN, _LIBS, _PRUNE, _CRISP, _CALPHA, _FLOOR
    _GRAPH, _LIBS = graph, libs
    _PRUNE, _CRISP, _CALPHA, _FLOOR = False, False, 0.0, None

    log.info("[3/5] replaying (top_score ranking) over train + holdout")
    scored: dict[str, dict[str, npt.NDArray]] = {}
    summary_rows: list[dict] = []
    for name, plan in plans.items():
        _PLAN = plan
        t_top, t_found, t_exps, t_trunc = _replay(train_pos, workers, f"{name} train")
        h_top, h_found, h_exps, h_trunc = _replay(hold_pos, workers, f"{name} holdout")
        scored[name] = {"train_top": t_top, "train_found": t_found,
                        "hold_top": h_top, "hold_found": h_found}
        for split, top, found, exps, trunc, n_days in (
                ("train", t_top, t_found, t_exps, t_trunc, n_tr_days),
                ("holdout", h_top, h_found, h_exps, h_trunc, n_ho_days)):
            summary_rows.append({
                "motif": name, "split": split, "anchors": len(top),
                "fires": int(found.sum()), "fires_per_day": float(found.sum()) / n_days,
                "nonzero": int((top > 0).sum()),
                "nonzero_per_day": float((top > 0).sum()) / n_days,
                "top_p50": float(np.percentile(top, 50)),
                "top_p99": float(np.percentile(top, 99)),
                "top_p999": float(np.percentile(top, 99.9)),
                "mean_expansions": float(exps.mean()),
                "truncated_rate": float(trunc.mean()),
            })
        log.info("  %s: train fires %d (%.1f/day) holdout fires %d (%.1f/day) "
                 "nonzero %.1f/%.1f per day "
                 "mean expansions %.1f/%.1f truncated %.3f%%/%.3f%%",
                 name, int(t_found.sum()), t_found.sum() / n_tr_days,
                 int(h_found.sum()), h_found.sum() / n_ho_days,
                 (t_top > 0).sum() / n_tr_days, (h_top > 0).sum() / n_ho_days,
                 t_exps.mean(), h_exps.mean(),
                 100 * t_trunc.mean(), 100 * h_trunc.mean())
    pl.from_dicts(summary_rows).write_csv(out / "rq1_summary.csv")
    log.info("  wrote %s", out / "rq1_summary.csv")
    # Per-anchor scores for Nov 1 tuning (train-only fitting guardrail).
    train_ids = df.get_column("transfer_id").to_numpy()[train_pos]
    hold_ids = df.get_column("transfer_id").to_numpy()[hold_pos]
    pl.DataFrame({"transfer_id": train_ids, **{f"{m}": scored[m]["train_top"] for m in motifs_wanted}}
                 ).write_parquet(out / "rq1_train_scores.parquet")
    pl.DataFrame({"transfer_id": hold_ids, **{f"{m}": scored[m]["hold_top"] for m in motifs_wanted}}
                 ).write_parquet(out / "rq1_holdout_scores.parquet")
    log.info("  wrote per-anchor score parquets")

    log.info("[4/5] ranking holdout per day and scoring recall at fixed budgets")
    memb = pl.read_parquet(data / "pattern_members.parquet")
    inst = splits.assign_instance_splits(
        pl.read_parquet(data / "pattern_instances.parquet"), memb, df)
    labels_one = (memb.join(inst.select(["instance_id", "pattern_type"]), on="instance_id")
                  .group_by("transfer_id").agg(pl.col("pattern_type").first()))
    # Same frame order as hold_pos (both filter the full frame in order);
    # head() keeps them aligned when --max-anchors truncates the replay.
    hold_base = (df.filter(pl.col("day").is_between(ho0, ho1))
                 .head(len(hold_pos))
                 .join(labels_one, on="transfer_id", how="left", maintain_order="left"))
    # The tie-break hash depends only on transfer_id: compute it once here
    # rather than once per motif (rank_per_day reuses the column).
    hold_base = evaluate.with_tiebreak(hold_base)
    # Anchor order is the frame order filtered to holdout days; row positions
    # below index into the holdout replay arrays in that same order.
    tr_rows, inst_rows = [], []
    for name in motifs_wanted:
        hold = evaluate.rank_per_day(
            hold_base.with_columns(pl.Series("score", scored[name]["hold_top"])))
        tr_rows.append(
            evaluate.transfer_recall(hold, budget_list)
            .with_columns(pl.lit(name).alias("motif"), pl.lit("transfer").alias("granularity")))
        inst_rows.append(
            evaluate.instance_recall(hold, memb, inst, budget_list)
            .with_columns(pl.lit(name).alias("motif")))
        log.info("  %s: holdout anchors %d, top_score p99 %.3f, >=0.5: %d (%.1f/day)",
                 name, len(hold), float(np.percentile(scored[name]["hold_top"], 99)),
                 int((scored[name]["hold_top"] >= 0.5).sum()),
                 (scored[name]["hold_top"] >= 0.5).sum() / n_ho_days)
    transfer_csv = out / "rq1_recall_transfer.csv"
    instance_csv = out / "rq1_recall_instance.csv"
    pl.concat(tr_rows).write_csv(transfer_csv)
    pl.concat(inst_rows).write_csv(instance_csv)
    log.info("  wrote %s and %s", transfer_csv, instance_csv)

    log.info("[5/5] RQ1 table (holdout instance recall, completable set)")
    inst_all = pl.concat(inst_rows).filter(pl.col("set") == "completable")
    for row in inst_all.sort(["motif", "budget", "group"]).iter_rows(named=True):
        log.info("  %-18s budget %5d %-8s recall %.3f (%d/%d)",
                 row["motif"], row["budget"], row["group"],
                 row["recall"] or 0.0, row["n_hit"], row["n_total"])
    log.info("done")


if __name__ == "__main__":
    typer.run(main)
