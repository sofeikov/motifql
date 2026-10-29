#!/usr/bin/env python3
"""Scored replay: compile reference path motifs and replay the train split.

End-to-end Oct 29 exit check: `.motif` text -> parse -> check -> plan ->
scored witnesses over every train anchor. Reports per motif: fires at the
`emit` threshold (per day), score deciles, mean expansions and truncation
rate, per-instance top scores on train CYCLE/RANDOM (anchor = last member),
and recall at emit plus at the score threshold that yields ~500 alerts/day.

`--unpruned N` replays the first N anchors without pruning and compares:
expansion ratio, top_score agreement, and whether recall at emit changes.
Pruning is sound for the emit-filtered query, so the recall must match
exactly; top scores below emit may differ (documented in matcher.score).

Usage:
    uv run eval/score_replay.py --data data/hi-small --motifs round_trip
    uv run eval/score_replay.py --data data/hi-small --max-anchors 50000
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
from eval import splits  # noqa: E402
from motifql.degrees import libs_from_doc  # noqa: E402
from motifql.graph import TemporalGraph  # noqa: E402
from motifql.matcher import ScoredResult, score  # noqa: E402
from motifql.parser import parse_file  # noqa: E402
from motifql.planner import PathPlan, compile_motif  # noqa: E402

log = logging.getLogger("score_replay")

_GRAPH: TemporalGraph | None = None
_PLAN: PathPlan | None = None
_LIBS = None


def _score_job(args: tuple[npt.NDArray[np.int64], bool]
             ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64],
                        npt.NDArray[bool], npt.NDArray[np.float64],
                        npt.NDArray[bool]]:
    """Score one anchor shard; returns positions first for realignment."""
    positions, prune = args
    assert _GRAPH is not None and _PLAN is not None and _LIBS is not None
    n = len(positions)
    top = np.empty(n, dtype=np.float64)
    found = np.empty(n, dtype=bool)
    exps = np.empty(n, dtype=np.float64)
    trunc = np.empty(n, dtype=bool)
    for i, pos in enumerate(positions.tolist()):
        res = score(_GRAPH, int(pos), _PLAN, _LIBS, prune=prune)
        top[i] = res.top_score
        found[i] = res.found
        exps[i] = res.expansions
        trunc[i] = res.truncated
    return positions, top, found, exps, trunc


def _replay(positions: npt.NDArray[np.int64], prune: bool, workers: int,
            label: str) -> tuple[npt.NDArray[np.float64], npt.NDArray[bool],
                                 npt.NDArray[np.float64], npt.NDArray[bool]]:
    """Replay shards and realign results to `positions` order (shards finish
    out of order under imap_unordered)."""
    t0 = time.perf_counter()
    if workers <= 1:
        _, *out = _score_job((positions, prune))
    else:
        n_shards = 8 * workers
        shards = [positions[i::n_shards] for i in range(n_shards)]
        shards = [s for s in shards if len(s)]
        ctx = mp.get_context("fork")
        parts = []
        with ctx.Pool(processes=workers) as pool:
            for done, part in enumerate(
                pool.imap_unordered(_score_job, [(s, prune) for s in shards]), start=1
            ):
                parts.append(part)
                if done % workers == 0:
                    log.info("  ... %s %d/%d shards", label, done, len(shards))
        key = np.concatenate([p[0] for p in parts])
        back = np.searchsorted(positions, key)
        out = tuple(
            _scatter(np.concatenate([p[k] for p in parts]), back, len(positions), dt)
            for k, dt in enumerate(
                [np.float64, bool, np.float64, bool], start=1))
    log.info("  %s: %.1fs wall", label, time.perf_counter() - t0)
    return out


def _scatter(vals: npt.NDArray, back: npt.NDArray[np.int64], n: int, dt) -> npt.NDArray:
    aligned = np.empty(n, dtype=dt)
    aligned[back] = vals
    return aligned


def main(
    data: Annotated[Path, typer.Option(help="Prepped dataset directory.")] = Path("data/hi-small"),
    train: Annotated[str, typer.Option(help="Anchor day range, inclusive.")] = f"{splits.TRAIN_START},{splits.TRAIN_END}",
    motifs: Annotated[str, typer.Option(help="Comma-separated motif names.")] = "round_trip,layering_chain",
    max_anchors: Annotated[int, typer.Option(help="Cap anchors (0 = all).")] = 0,
    workers: Annotated[int, typer.Option(help="Worker processes (1 = serial).")] = 16,
    unpruned: Annotated[int, typer.Option(help="Head anchors to also replay unpruned.")] = 2000,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tr0, tr1 = (date.fromisoformat(s) for s in train.split(","))
    motifs_wanted = [m.strip() for m in motifs.split(",") if m.strip()]

    log.info("[1/4] loading transfers and building graph (warmup+train)")
    df = pl.read_parquet(data / "transfers.parquet")
    scoped = df.filter(pl.col("day") <= tr1)
    graph = TemporalGraph.from_transfers(scoped)
    days = scoped.get_column("day").to_numpy()
    anchor_pos = np.flatnonzero(
        (days >= np.datetime64(str(tr0))) & (days <= np.datetime64(str(tr1)))
    )
    if max_anchors > 0:
        anchor_pos = anchor_pos[:max_anchors]
    log.info("  graph edges: %d, anchors: %d", graph.n_edges, len(anchor_pos))

    log.info("[2/4] compiling motifs from examples/amlworld.motif")
    doc = parse_file(Path("examples/amlworld.motif"))
    libs = libs_from_doc(doc)
    plans = {m: compile_motif(doc, m) for m in motifs_wanted}
    for m, p in plans.items():
        log.info("  %s: closed=%s hops=%d..%d window=%.0fs prunable=%d deferred=%s emit=%.2f",
                 m, p.closed, p.min_hops, p.max_hops, p.window_s,
                 len(p.prunable), p.deferred is not None, p.emit)

    global _GRAPH, _PLAN, _LIBS
    _GRAPH, _LIBS = graph, libs
    n_days = (tr1 - tr0).days + 1
    order = np.argsort(graph.transfer_id)
    sorted_tids = graph.transfer_id[order]

    def edge_pos(tid: int) -> int:
        i = int(np.searchsorted(sorted_tids, tid))
        return int(order[i])

    for name, plan in plans.items():
        _PLAN = plan
        log.info("[3/4] replaying %s (pruned) over %d anchors", name, len(anchor_pos))
        top, found, exps, trunc = _replay(anchor_pos, True, workers, f"{name} pruned")
        fires = int(found.sum())
        log.info("  %s: fires %d (%.1f/day), mean expansions %.1f, truncated %.4f%%",
                 name, fires, fires / n_days, exps.mean(), 100 * trunc.mean())
        for q in (50, 90, 99):
            log.info("  %s: top_score p%d = %.3f", name, q, float(np.percentile(top, q)))
        for t in (0.2, 0.5, 0.6, 0.9):
            c = int((top >= t).sum())
            log.info("  %s: anchors with top_score >= %.1f: %d (%.1f/day)", name, t, c, c / n_days)
        budget_t = float(np.percentile(top, 100 * (1 - 500 * n_days / len(anchor_pos))))
        log.info("  %s: threshold for ~500/day = %.3f", name, budget_t)

        if unpruned > 0:
            # Uniform random sample, sorted back into time order (the pool
            # realigns by position, and a head slice would only cover Sep 3).
            rng = np.random.default_rng(42)
            sample = np.sort(rng.choice(len(anchor_pos),
                                        min(unpruned, len(anchor_pos)),
                                        replace=False))
            head = anchor_pos[sample]
            log.info("[3b/4] replaying %s (unpruned) over %d sampled anchors", name, len(head))
            utop, ufound, uexps, _ = _replay(head, False, workers, f"{name} unpruned")
            agree_found = bool((found[sample] == ufound).all())
            agree_top = float((np.abs(top[sample] - utop) < 1e-9).mean())
            log.info("  %s: pruned/unpruned expansions %.1f/%.1f (ratio %.1fx), "
                     "found agreement %s, top_score agreement %.4f",
                     name, exps[sample].mean(), uexps.mean(),
                     uexps.mean() / max(exps[sample].mean(), 1e-9),
                     agree_found, agree_top)

        log.info("[4/4] per-instance top scores on train CYCLE/RANDOM")
        inst = splits.assign_instance_splits(
            pl.read_parquet(data / "pattern_instances.parquet"),
            pl.read_parquet(data / "pattern_members.parquet"), df)
        memb = pl.read_parquet(data / "pattern_members.parquet")
        ptype = "CYCLE" if name == "round_trip" else "RANDOM"
        sub = inst.filter((pl.col("pattern_type") == ptype) & (pl.col("split") == "train"))
        m = (memb.join(sub.select("instance_id"), on="instance_id")
             .join(df.select("transfer_id", "timestamp"), on="transfer_id"))
        per = m.group_by("instance_id").agg(
            pl.col("transfer_id").sort_by("timestamp").alias("tids"))
        n_eval = 0
        for row in per.sort("instance_id").iter_rows(named=True):
            if len(row["tids"]) < plan.min_hops:
                log.info("  %s %d: excluded (%d members < min_hops %d)",
                         ptype, row["instance_id"], len(row["tids"]), plan.min_hops)
                continue
            pos = edge_pos(row["tids"][-1])
            r = score(graph, pos, plan, libs)
            u = score(graph, pos, plan, libs, prune=False)
            n_eval += 1
            log.info("  %s %d: top_score=%.3f/%.3f (pruned/unpruned) %s",
                     ptype, row["instance_id"], r.top_score, u.top_score,
                     "FIRES" if r.found else "")
    log.info("done")


if __name__ == "__main__":
    typer.run(main)
