#!/usr/bin/env python3
"""Replay benchmark for the exact path matchers (DESIGN.md Oct 28 exit check).

Replays every train-split anchor through `ROUND_TRIP` and `LAYERING_CHAIN`
in time order and reports: wall time, transfers/sec, p50/p99 per-anchor
latency, truncation rate, fire rate (alerts/day) and per-instance recall on
train CYCLE/RANDOM instances (anchor = last member; caught = witness found).
The graph is built from warmup+train so every train anchor's window is
complete; workers skip eviction (live/batch parity is pinned by
tests/test_graph.py::test_evict_parity_with_windowed_replay instead).
`--workers N` shards anchors round-robin across forked processes sharing
the graph copy-on-write; `--workers 1` runs the identical code serially.

Usage:
    uv run eval/replay_bench.py --data data/hi-small
    uv run eval/replay_bench.py --data data/hi-small --max-anchors 50000
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import sys
import time
from datetime import date
from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from eval import splits  # noqa: E402
from motifql.graph import TemporalGraph  # noqa: E402
from motifql.matcher import LAYERING_CHAIN, ROUND_TRIP, PathMotif, match  # noqa: E402

log = logging.getLogger("replay_bench")

MOTIFS: dict[str, PathMotif] = {
    "round_trip": ROUND_TRIP,
    "layering_chain": LAYERING_CHAIN,
}

# Fork-inherited by workers: the graph (~260MB of edge arrays) is shared
# copy-on-write, so tasks only pickle small anchor-index shards. The match
# path never mutates the big arrays (only the few-MB lazy eviction floors),
# and batch replay uses complete windows, so workers skip eviction entirely.
_GRAPH: TemporalGraph | None = None


def _score_shard(args: tuple[np.ndarray, PathMotif]) -> tuple[np.ndarray, int, int, int]:
    """Score one anchor shard; runs in a worker process."""
    positions, motif = args
    assert _GRAPH is not None
    lat_us = np.empty(len(positions), dtype=np.float64)
    found = 0
    trunc = 0
    expansions = 0
    for i, pos in enumerate(positions.tolist()):
        t0 = time.perf_counter()
        res = match(_GRAPH, int(pos), motif)
        lat_us[i] = (time.perf_counter() - t0) * 1e6
        found += res.found
        trunc += res.truncated
        expansions += res.expansions
    return lat_us, found, trunc, expansions


def main(
    data: Annotated[Path, typer.Option(help="Prepped dataset directory.")] = Path("data/hi-small"),
    train: Annotated[str, typer.Option(help="Anchor day range, inclusive.")] = f"{splits.TRAIN_START},{splits.TRAIN_END}",
    max_anchors: Annotated[int, typer.Option(help="Cap anchors per motif (0 = all).")] = 0,
    workers: Annotated[int, typer.Option(help="Worker processes (1 = serial).")] = 1,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tr0, tr1 = (date.fromisoformat(s) for s in train.split(","))

    log.info("[1/4] loading transfers and building graph (warmup+train)")
    df = pl.read_parquet(data / "transfers.parquet")
    scoped = df.filter(pl.col("day") <= tr1)
    assert bool(scoped.select(pl.col("timestamp").is_sorted()).item()), "prep guarantees time order"
    graph = TemporalGraph.from_transfers(scoped)
    global _GRAPH
    _GRAPH = graph  # fork-shared with workers; also serves the serial path
    days = scoped.get_column("day").to_numpy()
    anchor_pos = np.flatnonzero(
        (days >= np.datetime64(str(tr0))) & (days <= np.datetime64(str(tr1)))
    )
    if max_anchors > 0:
        anchor_pos = anchor_pos[:max_anchors]
    log.info("  graph edges: %d, anchors: %d", graph.n_edges, len(anchor_pos))
    order = np.argsort(graph.transfer_id)
    sorted_tids = graph.transfer_id[order]

    def edge_pos(tid: int) -> int:
        i = int(np.searchsorted(sorted_tids, tid))
        assert int(sorted_tids[i]) == tid, f"transfer {tid} not in graph"
        return int(order[i])

    for name, motif in MOTIFS.items():
        log.info("[2/4] replaying %s over %d anchors (%s)", name, len(anchor_pos),
                 "serial" if workers <= 1 else f"{workers} workers")
        t_start = time.perf_counter()
        if workers <= 1:
            lat_us, found, trunc, expansions = _score_shard((anchor_pos, motif))
        else:
            # Round-robin shards, oversubscribed: hub senders cluster in time,
            # so contiguous shards would idle-balance the pool.
            n_shards = 8 * workers
            shards = [anchor_pos[i::n_shards] for i in range(n_shards)]
            shards = [s for s in shards if len(s)]
            ctx = mp.get_context("fork")
            lat_parts: list[np.ndarray] = []
            found, trunc, expansions = 0, 0, 0
            with ctx.Pool(processes=workers) as pool:
                for done, (lat, f, t, e) in enumerate(
                    pool.imap_unordered(_score_shard, [(s, motif) for s in shards]),
                    start=1,
                ):
                    lat_parts.append(lat)
                    found += f
                    trunc += t
                    expansions += e
                    if done % workers == 0:
                        log.info("  ... %d/%d shards", done, len(shards))
            lat_us = np.concatenate(lat_parts)
        wall = time.perf_counter() - t_start
        n_days = (tr1 - tr0).days + 1
        p50, p99 = np.percentile(lat_us, [50, 99])
        rate = len(anchor_pos) / wall if wall > 0 else 0.0
        log.info(
            f"  {name}: wall {wall:.1f}s, {rate:,.0f} anchors/s, "
            f"latency p50 {p50:.0f}us p99 {p99:.0f}us, fires {found} "
            f"({found / n_days:.1f}/day), truncated {trunc} "
            f"({100 * trunc / len(anchor_pos):.3f}%), "
            f"mean expansions {expansions / len(anchor_pos):.1f}"
        )

    log.info("[3/4] per-instance recall on train CYCLE/RANDOM (anchor = last member)")
    inst = splits.assign_instance_splits(
        pl.read_parquet(data / "pattern_instances.parquet"),
        pl.read_parquet(data / "pattern_members.parquet"), df)
    memb = pl.read_parquet(data / "pattern_members.parquet")
    for ptype, motif in (("CYCLE", ROUND_TRIP), ("RANDOM", LAYERING_CHAIN)):
        sub = inst.filter((pl.col("pattern_type") == ptype) & (pl.col("split") == "train"))
        m = (memb.join(sub.select("instance_id"), on="instance_id")
             .join(df.select("transfer_id", "timestamp"), on="transfer_id")
             .sort(["instance_id", "timestamp"]))
        last = m.group_by("instance_id").agg(
            pl.col("transfer_id").last().alias("anchor_tid"),
            pl.len().alias("n"))
        caught = 0
        for row in last.iter_rows(named=True):
            res = match(graph, edge_pos(row["anchor_tid"]), motif)
            caught += res.found
        log.info("  %s: %d/%d train instances caught (anchor = last member)",
                 ptype, caught, last.height)
    log.info("[4/4] done")


if __name__ == "__main__":
    typer.run(main)
