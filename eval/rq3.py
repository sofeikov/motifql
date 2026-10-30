#!/usr/bin/env python3
"""RQ3 harness: live latency, throughput, truncation, pruning effect (Nov 2).

Serial per-anchor timing over a seeded holdout sample, per motif: wall
time, anchors/sec, p50/p99 latency, truncation rate, mean expansions.
Path motifs run three arms on the same sample — pruned from emit (the
live query), exact-floor branch-and-bound (the ranking query), and
unpruned on a smaller sub-sample (full unpruned replay is infeasible;
that is itself the finding). Groups ignore prune/floor, so they run one
arm. A SQLite bounded-BFS arm stands in for ezrules' SQL graph feature
(ezrules is not wired in this pass; labeled as a proxy).

Usage:
    uv run eval/rq3.py --data data/hi-small --sample 5000
    uv run eval/rq3.py --data data/hi-small --sample 200 --bfs-sample 50
"""

from __future__ import annotations

import logging
import sqlite3
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
from motifql.degrees import libs_from_doc  # noqa: E402
from motifql.graph import TemporalGraph  # noqa: E402
from motifql.groups import score_any, warm_up  # noqa: E402
from motifql.matcher import EXACT_FLOOR  # noqa: E402
from motifql.parser import parse_file  # noqa: E402
from motifql.planner import PathPlan, compile_motif  # noqa: E402

log = logging.getLogger("rq3")

MOTIFS = ["round_trip", "layering_chain", "fan_out", "fan_in",
          "scatter_gather", "gather_scatter", "structuring_fan_in",
          "bipartite_butterfly"]

BFS_DEPTH = 3
BFS_WINDOW_S = 7 * 86400
BFS_CAP = 10_000


def _summarize(lat_us: np.ndarray, top: np.ndarray, found: np.ndarray,
               exps: np.ndarray, trunc: np.ndarray, wall: float) -> dict:
    n = len(lat_us)
    return {
        "n": n, "wall_s": wall, "anchors_per_s": n / wall if wall else 0.0,
        "p50_us": float(np.percentile(lat_us, 50)),
        "p99_us": float(np.percentile(lat_us, 99)),
        "trunc_rate": float(trunc.mean()), "mean_expansions": float(exps.mean()),
        "fires": int(found.sum()),
    }


def _run_arm(graph: TemporalGraph, positions: np.ndarray, plan, libs,
             label: str, **kw) -> tuple[dict, np.ndarray]:
    lat = np.empty(len(positions))
    top = np.empty(len(positions))
    found = np.empty(len(positions), dtype=bool)
    exps = np.empty(len(positions))
    trunc = np.empty(len(positions), dtype=bool)
    t0 = time.perf_counter()
    for i, pos in enumerate(positions.tolist()):
        a = time.perf_counter()
        res = score_any(graph, int(pos), plan, libs, **kw)
        lat[i] = (time.perf_counter() - a) * 1e6
        top[i] = res.top_score
        found[i] = res.found
        exps[i] = res.expansions
        trunc[i] = res.truncated
    wall = time.perf_counter() - t0
    s = _summarize(lat, top, found, exps, trunc, wall)
    log.info("  %-30s: %6d anchors, %8.1f/s, p50 %7.0fus p99 %9.0fus, "
             "trunc %6.3f%%, mean exp %8.1f, fires %d",
             label, len(positions), s["anchors_per_s"],
             s["p50_us"], s["p99_us"], 100 * s["trunc_rate"],
             s["mean_expansions"], s["fires"])
    return s, top


def _build_bfs_db(df: pl.DataFrame) -> sqlite3.Connection:
    """Edge table with per-endpoint indexes (the ezrules-SQL shape)."""
    con = sqlite3.connect(":memory:")
    con.execute("CREATE TABLE edges (frm TEXT, too TEXT, t INTEGER)")
    con.execute("CREATE INDEX i_frm ON edges (frm, t)")
    con.execute("CREATE INDEX i_to ON edges (too, t)")
    con.executemany(
        "INSERT INTO edges VALUES (?, ?, ?)",
        ((f"{r['from_bank']}/{r['from_account']}",
          f"{r['to_bank']}/{r['to_account']}",
          int(r["timestamp"].timestamp()))
         for r in df.iter_rows(named=True)))
    con.commit()
    return con


def _bfs_count(con: sqlite3.Connection, start: str, hi_t: int) -> tuple[int, bool]:
    """Distinct accounts within BFS_DEPTH undirected hops in the window.

    Bounded work like the real feature: stops at BFS_CAP nodes and reports
    the cap instead of raising (DESIGN.md 6.5 policy).
    """
    lo_t = hi_t - BFS_WINDOW_S
    seen = {start}
    frontier = {start}
    for _ in range(BFS_DEPTH):
        if not frontier:
            break
        q = ",".join("?" for _ in frontier)
        rows = con.execute(
            f"SELECT frm, too FROM edges WHERE t >= ? AND t < ? "  # noqa: S608
            f"AND (frm IN ({q}) OR too IN ({q}))",
            (lo_t, hi_t, *frontier, *frontier)).fetchall()
        nxt = set()
        for a, b in rows:
            if a not in seen:
                seen.add(a)
                nxt.add(a)
            if b not in seen:
                seen.add(b)
                nxt.add(b)
            if len(seen) >= BFS_CAP:
                return len(seen), True
        frontier = nxt
    return len(seen), len(seen) >= BFS_CAP


def main(
    data: Annotated[Path, typer.Option(help="Prepped dataset directory.")] = Path("data/hi-small"),
    holdout: Annotated[str, typer.Option(help="Holdout day range, inclusive.")] = f"{splits.HOLDOUT_START},{splits.HOLDOUT_END}",
    motifs: Annotated[str, typer.Option(help="Comma-separated motif names.")] = ",".join(MOTIFS),
    sample: Annotated[int, typer.Option(help="Seeded holdout anchors, serial.")] = 5000,
    unpruned_sample: Annotated[int, typer.Option(help="Sub-sample for the unpruned arm.")] = 500,
    bfs_sample: Annotated[int, typer.Option(help="Anchors for the SQLite BFS arm.")] = 1000,
    seed: Annotated[int, typer.Option(help="Sample seed.")] = 42,
    out: Annotated[Path, typer.Option(help="Directory for the latency CSV.")] = Path("eval/results"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ho0, ho1 = (date.fromisoformat(s) for s in holdout.split(","))
    motifs_wanted = [m.strip() for m in motifs.split(",") if m.strip()]
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    log.info("[1/4] loading transfers and building graph (all days; queries look back)")
    df = pl.read_parquet(data / "transfers.parquet")
    graph = TemporalGraph.from_transfers(df)
    days = df.get_column("day").to_numpy()
    hold_pos = np.flatnonzero(
        (days >= np.datetime64(str(ho0))) & (days <= np.datetime64(str(ho1))))
    positions = np.sort(rng.choice(len(hold_pos), min(sample, len(hold_pos)),
                                   replace=False))
    positions = hold_pos[positions]
    log.info("  graph edges: %d, holdout anchors: %d, sample: %d",
             graph.n_edges, len(hold_pos), len(positions))

    log.info("[2/4] compiling motifs from examples/amlworld.motif")
    doc = parse_file(Path("examples/amlworld.motif"))
    libs = libs_from_doc(doc)
    plans = {m: compile_motif(doc, m) for m in motifs_wanted}

    log.info("[3/4] serial latency per motif (pruned + exact-floor arms)")
    rows: list[dict] = []
    for name, plan in plans.items():
        # Load compiled kernels before timing (one-time ~1s per process).
        warm_up(plan, libs)
        if isinstance(plan, PathPlan):
            warm_up(plan, libs, floor=EXACT_FLOOR)
            warm_up(plan, libs, prune=False)
        s, _ = _run_arm(graph, positions, plan, libs, f"{name} pruned")
        rows.append({"motif": name, "arm": "pruned", **s})
        if isinstance(plan, PathPlan):
            s, _ = _run_arm(graph, positions, plan, libs, f"{name} exact_floor",
                            floor=EXACT_FLOOR)
            rows.append({"motif": name, "arm": "exact_floor", **s})
            sub = np.sort(rng.choice(len(positions),
                                     min(unpruned_sample, len(positions)),
                                     replace=False))
            s, utop = _run_arm(graph, positions[sub], plan, libs, f"{name} unpruned",
                               prune=False)
            rows.append({"motif": name, "arm": "unpruned", **s})
            _, ftop = _run_arm(graph, positions[sub], plan, libs, f"{name} exact_floor(sub)",
                               floor=EXACT_FLOOR)
            agree = float((np.abs(utop - ftop) < 1e-9).mean())
            log.info("  %-22s exact-floor vs unpruned top agreement: %.4f",
                     name, agree)
            rows[-1]["floor_agreement"] = agree

    log.info("[4/4] SQLite bounded-BFS proxy baseline (%d anchors)", bfs_sample)
    con = _build_bfs_db(df)
    bpos = np.sort(rng.choice(len(hold_pos), min(bfs_sample, len(hold_pos)),
                              replace=False))
    lat, capped = [], 0
    t0 = time.perf_counter()
    keys = df.get_column("to_bank") + "/" + df.get_column("to_account")
    ts = df.get_column("timestamp").dt.epoch("s")
    for pos in hold_pos[bpos].tolist():
        a = time.perf_counter()
        _, cap = _bfs_count(con, keys[pos], int(ts[pos]))
        lat.append((time.perf_counter() - a) * 1e6)
        capped += cap
    wall = time.perf_counter() - t0
    lat = np.array(lat)
    log.info("  sqlite_bfs: %d anchors, %.1f/s, p50 %.0fus p99 %.0fus, capped %.3f%%",
             len(lat), len(lat) / wall, float(np.percentile(lat, 50)),
             float(np.percentile(lat, 99)), 100 * capped / len(lat))
    rows.append({"motif": "sqlite_bfs_proxy", "arm": "bfs",
                 "n": len(lat), "wall_s": wall,
                 "anchors_per_s": len(lat) / wall,
                 "p50_us": float(np.percentile(lat, 50)),
                 "p99_us": float(np.percentile(lat, 99)),
                 "trunc_rate": capped / len(lat), "mean_expansions": float("nan"),
                 "fires": 0})

    csv = out / "rq3_latency.csv"
    pl.from_dicts(rows).write_csv(csv)
    log.info("  wrote %s", csv)
    log.info("done")


if __name__ == "__main__":
    typer.run(main)
