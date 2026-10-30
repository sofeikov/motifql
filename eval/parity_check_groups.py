#!/usr/bin/env python3
"""Fast-vs-reference parity for the group/butterfly executors.

Scores the same anchors through the compiled fast path
(``motifql.groups_fast``) and the pure-Python reference executors
(``motifql.groups.PURE_PYTHON = True``) and counts any difference in
found / witness / score / top_score / degrees / truncated / expansions.
Floats are compared by ``float.hex`` and every scalar's Python type is
compared too, so a -0.0, a ULP or an np.int64-for-int slip all count.

Anchors per motif: ``--n`` random train anchors plus the ``--hubs``
anchors with the highest reference expansions in a pre-scan of
``--prescan`` further random train anchors (hub stress). Modes: every
combination of prune True/False, floor None/EXACT_FLOOR and crisp
(off / alpha 0 / alpha = plan.emit), plus an eviction replay (anchors in
time order with ``graph.evict_before(t - retention)``, crisp off).

The reference runs on forked worker processes sharing the graph
copy-on-write; the fast path runs in the parent.

Usage:
    uv run eval/parity_check_groups.py --data data/hi-small
    uv run eval/parity_check_groups.py --data data/hi-small --motifs scatter_gather --n 2000
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from eval import splits  # noqa: E402
from motifql import groups  # noqa: E402
from motifql.degrees import libs_from_doc  # noqa: E402
from motifql.graph import SECS_PER_DAY, TemporalGraph  # noqa: E402
from motifql.matcher import EXACT_FLOOR  # noqa: E402
from motifql.parser import parse_file  # noqa: E402
from motifql.planner import compile_motif  # noqa: E402

GROUP_MOTIFS = ("fan_out", "fan_in", "structuring_fan_in", "scatter_gather",
                "gather_scatter", "bipartite_butterfly")

_G: dict = {}  # graph / doc / libs, inherited by forked workers
# Eviction replay lag: shorter than every group window (2d-7d), so the
# watermark really cuts into windows and both paths must agree on it.
EVICT_LAG_S = 2 * SECS_PER_DAY


def canon(x):
    """Type-strict, bit-exact comparable form of a result field."""
    if isinstance(x, bool):
        return ("bool", x)
    if isinstance(x, float):
        return (type(x).__name__, x.hex())
    if isinstance(x, int):
        return (type(x).__name__, int(x))
    if isinstance(x, (tuple, list)):
        return (type(x).__name__, tuple(canon(i) for i in x))
    if x is None or isinstance(x, str):
        return x
    return (type(x).__name__, repr(x))


def result_key(r) -> tuple:
    return canon((r.found, r.witness, r.score, r.top_score, r.degrees,
                  r.truncated, r.expansions))


def modes_for(plan) -> list[tuple]:
    out = []
    for prune in (True, False):
        for floor in (None, EXACT_FLOOR):
            for crisp, alpha in ((False, 0.0), (True, 0.0), (True, float(plan.emit))):
                out.append((prune, floor, crisp, alpha))
    return out


def score_modes(motif: str, anchors: list[int], pure: bool = True) -> list[list[tuple]]:
    """Per anchor, the canonical result in every static mode."""
    groups.PURE_PYTHON = pure
    g, libs = _G["graph"], _G["libs"]
    plan = compile_motif(_G["doc"], motif)
    modes = modes_for(plan)
    out = []
    for a in anchors:
        row = []
        for prune, floor, crisp, alpha in modes:
            r = groups.score_any(g, a, plan, libs, prune=prune, crisp=crisp,
                                 crisp_alpha=alpha, floor=floor)
            row.append(result_key(r))
        out.append(row)
    return out


def score_evict(motif: str, anchors: list[int], pure: bool) -> list[tuple]:
    """Time-ordered replay with lazy eviction on a private graph view."""
    groups.PURE_PYTHON = pure
    g0 = _G["graph"]
    g = TemporalGraph(
        times=g0.times, frm=g0.frm, to=g0.to, amount=g0.amount,
        transfer_id=g0.transfer_id, nodes=g0.nodes,
        out_ptr=g0.out_ptr, out_edges=g0.out_edges, out_times=g0.out_times,
        in_ptr=g0.in_ptr, in_edges=g0.in_edges, in_times=g0.in_times,
        retention_s=g0.retention_s)
    plan = compile_motif(_G["doc"], motif)
    out = []
    for a in sorted(anchors):
        g.evict_before(int(g.times[a]) - EVICT_LAG_S)
        out.append(result_key(groups.score_any(g, a, plan, _G["libs"])))
    return out


def prescan_expansions(motif: str, anchors: list[int]) -> list[int]:
    groups.PURE_PYTHON = True
    plan = compile_motif(_G["doc"], motif)
    return [groups.score_any(_G["graph"], a, plan, _G["libs"]).expansions for a in anchors]


def _chunks(xs: list[int], k: int) -> list[list[int]]:
    return [xs[i::k] for i in range(k)]


def _run_pool(pool, fn, motif: str, anchors: list[int], workers: int):
    parts = _chunks(anchors, workers * 4)
    res = pool.starmap(fn, [(motif, p) for p in parts])
    flat: dict[int, object] = {}
    for p, r in zip(parts, res):
        for a, v in zip(p, r):
            flat[a] = v
    return flat


def main(
    data: Annotated[Path, typer.Option(help="Prepped dataset dir")] = Path("data/hi-small"),
    motif_file: Annotated[Path, typer.Option(help="Motif file")] = Path("examples/amlworld.motif"),
    motifs: Annotated[str, typer.Option(help="Comma-separated motifs")] = ",".join(GROUP_MOTIFS),
    n: Annotated[int, typer.Option(help="Random train anchors per motif")] = 20_000,
    prescan: Annotated[int, typer.Option(help="Pre-scan pool for hub stress")] = 60_000,
    hubs: Annotated[int, typer.Option(help="Top-expansion anchors added")] = 200,
    workers: Annotated[int, typer.Option(help="Forked reference workers")] = 16,
    seed: Annotated[int, typer.Option(help="Sampling seed")] = 0,
) -> None:
    df = pl.read_parquet(data / "transfers.parquet")
    g = TemporalGraph.from_transfers(df)
    doc = parse_file(motif_file)
    _G.update(graph=g, doc=doc, libs=libs_from_doc(doc))
    days = df.get_column("day").to_numpy()
    pool_pos = np.flatnonzero(days >= np.datetime64(str(splits.TRAIN_START)))
    rng = np.random.default_rng(seed)
    picked = rng.choice(pool_pos, n + prescan, replace=False)
    sample = sorted(int(a) for a in picked[:n])
    scan = sorted(int(a) for a in picked[n:])
    print(f"graph {g.n_edges} edges; {len(pool_pos)} train anchors; "
          f"sample {len(sample)} + hub pre-scan {len(scan)} -> top {hubs}", flush=True)
    # Warm the numba kernels before forking (the workers run pure Python).
    for m in motifs.split(","):
        groups.PURE_PYTHON = False
        groups.score_any(g, sample[0], compile_motif(doc, m), _G["libs"])
    total_bad = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for m in motifs.split(","):
            t0 = time.perf_counter()
            exp = _run_pool(pool, prescan_expansions, m, scan, workers)
            hub = sorted(sorted(scan, key=lambda a: (-exp[a], a))[:hubs])
            anchors = sorted(set(sample) | set(hub))
            t1 = time.perf_counter()
            ref = _run_pool(pool, score_modes, m, anchors, workers)
            t2 = time.perf_counter()
            fast = dict(zip(anchors, score_modes(m, anchors, pure=False)))
            t3 = time.perf_counter()
            plan = compile_motif(doc, m)
            modes = modes_for(plan)
            bad_by_mode = [0] * len(modes)
            examples = []
            for a in anchors:
                for k, (rv, fv) in enumerate(zip(ref[a], fast[a])):
                    if rv != fv:
                        bad_by_mode[k] += 1
                        if len(examples) < 3:
                            examples.append((a, modes[k], rv, fv))
            # Eviction replay: whole anchor set in time order per process.
            ev_ref = score_evict(m, anchors, pure=True)
            ev_fast = score_evict(m, anchors, pure=False)
            ev_bad = sum(1 for x, y in zip(ev_ref, ev_fast) if x != y)
            bad = sum(bad_by_mode) + ev_bad
            total_bad += bad
            n_trunc = sum(1 for a in anchors if ref[a][0][1][5][1])
            print(f"{m:20s} anchors={len(anchors)} (hub max exp {max(exp.values())}, "
                  f"truncated {n_trunc}) modes={len(modes)}+evict "
                  f"mismatches={bad} [per mode {bad_by_mode}, evict {ev_bad}] "
                  f"ref {t2 - t1:.0f}s/{workers}w fast {t3 - t2:.1f}s", flush=True)
            for ex in examples:
                print("   example", ex, flush=True)
    groups.PURE_PYTHON = False
    print(f"TOTAL mismatches: {total_bad}", flush=True)
    raise typer.Exit(1 if total_bad else 0)


if __name__ == "__main__":
    typer.run(main)
