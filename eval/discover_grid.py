#!/usr/bin/env python3
"""Shared harness for issue #26 search arms (enumerative + hybrid).

Builds the SAME train-only DiscoveryCtx as eval/discover.py (same
seeds, same exact-floor scoring, same gate constants) so only the
proposer differs across arms. Two-stage protocol: coarse scan on a
seeded 2k background subsample, exact re-score of frontier
candidates on the full 20k sample. Import-only; drivers are
eval/discover_enum.py and eval/discover_hybrid.py.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import eval.rq1 as rq1  # noqa: E402
from eval import splits  # noqa: E402
from eval.discover import BG_N, MOTIFS, _batch_factory  # noqa: E402
from motifql.degrees import libs_from_doc  # noqa: E402
from motifql.graph import TemporalGraph  # noqa: E402
from motifql.models import discovery as D  # noqa: E402
from motifql.models import repair  # noqa: E402
from motifql.parser import parse_file  # noqa: E402
from motifql.planner import compile_motif  # noqa: E402

log = logging.getLogger("discover_grid")

BG_SEED = 11  # must match eval/discover.py
SUB_SEED = 26
SUB_N = 2000


def build_ctx(data: Path, workers: int):
    """Train-only ctx mirroring eval/discover.py steps [1/4]-[2/4]."""
    log.info("loading transfers, ground truth, splits")
    df = pl.read_parquet(data / "transfers.parquet")
    memb = pl.read_parquet(data / "pattern_members.parquet")
    inst = splits.assign_instance_splits(
        pl.read_parquet(data / "pattern_instances.parquet"), memb, df)
    graph = TemporalGraph.from_transfers(df)
    tids = df.get_column("transfer_id").to_numpy()
    pos_of = {int(t): i for i, t in enumerate(tids.tolist())}
    rq1._GRAPH = graph

    train_ids = (df.filter(pl.col("day").is_between(splits.TRAIN_START, splits.TRAIN_END))
                 .filter(pl.col("is_laundering") == 0)
                 .select("transfer_id").to_series().to_list())
    bg_tids = sorted(np.random.default_rng(BG_SEED).choice(
        train_ids, size=min(BG_N, len(train_ids)), replace=False).tolist())
    bg_pos = [pos_of[t] for t in bg_tids]

    train_inst = inst.filter(pl.col("split") == "train")
    member_pos: dict[int, list[int]] = {}
    member_tid: dict[int, list[int]] = {}
    mm = memb.join(train_inst.select("instance_id"), on="instance_id", how="inner")
    for iid, tid in zip(mm["instance_id"].to_list(), mm["transfer_id"].to_list()):
        iid, tid = int(iid), int(tid)
        member_pos.setdefault(iid, []).append(pos_of[tid])
        member_tid.setdefault(iid, []).append(tid)

    log.info("compiling %d motifs and scoring member+bg anchors", len(MOTIFS))
    doc = parse_file(Path("examples/amlworld.motif"))
    libs = libs_from_doc(doc)
    plans = {m: compile_motif(doc, m) for m in MOTIFS}
    batch = _batch_factory(workers, "base")
    want = sorted({p for ps in member_pos.values() for p in ps} | set(bg_pos))
    ctx = D.DiscoveryCtx(graph=graph, transfer_ids=tids, pos_of=pos_of,
                         member_pos=member_pos, member_tid=member_tid, bg_pos=bg_pos)
    for m in MOTIFS:
        tops = batch(want, plans[m], libs)
        ctx.base_scores[m] = tops
        ctx.base_emit[m] = plans[m].emit
        ctx.base_bg_fire[m] = sum(tops[p] >= plans[m].emit for p in bg_pos) / len(bg_pos)
        log.info("  %-18s emit %.2f bg fire %.4f", m, plans[m].emit, ctx.base_bg_fire[m])
    n_train = df.filter(pl.col("day").is_between(splits.TRAIN_START, splits.TRAIN_END)).height
    train_days = (splits.TRAIN_END - splits.TRAIN_START).days + 1
    info = {"df": df, "batch": batch, "plans": plans, "libs": libs,
            "train_per_day": n_train / train_days}
    return ctx, info


def gate_bars(ctx: D.DiscoveryCtx, cluster: list[int]):
    """Pre-registered gate bars: need + allowed bg rate + base median."""
    med = float(np.median([ctx.base_bg_fire[m] for m in MOTIFS]))
    allowed = max(med * D.GATE_BG_MULT, 2 / len(ctx.bg_pos))
    need = min(D.GATE_MIN_CATCHES, len(cluster))
    return need, allowed, med


def sub_bg(ctx: D.DiscoveryCtx) -> list[int]:
    """Seeded 2k subsample of the ctx background (stage-1 scan)."""
    idx = np.random.default_rng(SUB_SEED).choice(
        len(ctx.bg_pos), size=min(SUB_N, len(ctx.bg_pos)), replace=False)
    return sorted(ctx.bg_pos[i] for i in idx.tolist())


def score_candidate(ctx: D.DiscoveryCtx, text: str, batch_score,
                    cluster: list[int], bg_positions: list[int]) -> dict | None:
    """Exact-floor score of one candidate; None when it does not compile."""
    compiled, _ = repair.try_compile(text)
    if compiled is None:
        return None
    plan = compiled.plan
    want = sorted({p for iid in cluster for p in ctx.member_pos[iid]} | set(bg_positions))
    tops = batch_score(want, plan, compiled.libs)
    caught = D.caught_instances(tops, plan.emit, ctx.member_pos, cluster)
    bg_hits = sum(tops[p] >= plan.emit for p in bg_positions)
    marg = sum(1 for p in bg_positions
               if tops[p] >= plan.emit
               and not any(ctx.base_scores[m].get(p, 0.0) >= ctx.base_emit[m]
                           for m in MOTIFS)) / len(bg_positions)
    return {"emit": plan.emit, "caught": caught, "recall": len(caught),
            "bg_hits": bg_hits, "bg_n": len(bg_positions),
            "bg_rate": bg_hits / len(bg_positions), "marg_rate": marg}


def select_frontier(rows: list[dict]) -> list[dict]:
    """Per-recall net for exact re-score: within 3x of the level best.

    The coarse scan only SELECTS; all reported numbers come from the
    exact stage-2 re-score, so the net is deliberately generous.
    """
    by_recall: dict[int, list[dict]] = {}
    for r in rows:
        by_recall.setdefault(r["recall"], []).append(r)
    keep = []
    for level in sorted(by_recall):
        cands = sorted(by_recall[level], key=lambda r: r["bg_hits"])
        best = cands[0]["bg_hits"]
        cap = max(3 * best, 2)
        keep.extend(c for c in cands if c["bg_hits"] <= cap)
    return keep


def pareto(rows: list[dict]) -> list[dict]:
    """Non-dominated rows: no other row has >= recall and <= bg_hits."""
    out = []
    for r in rows:
        if not any(o is not r and o["recall"] >= r["recall"]
                   and o["bg_hits"] <= r["bg_hits"]
                   and (o["recall"] > r["recall"] or o["bg_hits"] < r["bg_hits"])
                   for o in rows):
            out.append(r)
    return sorted(out, key=lambda r: (r["bg_hits"], -r["recall"]))


def run_grid(ctx, info, cluster, templates, out_dir, label) -> Path:
    """Two-stage grid run; writes coarse + exact CSVs, returns exact path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    batch, train_per_day = info["batch"], info["train_per_day"]
    need, allowed, med = gate_bars(ctx, cluster)
    log.info("%s: %d templates, gate need=%d allowed=%.4f (base median %.4f)",
             label, len(templates), need, allowed, med)

    log.info("[coarse] %d templates on %d-bg subsample", len(templates), SUB_N)
    bg_sub = sub_bg(ctx)
    coarse, fails = [], 0
    t0 = time.time()
    for i, t in enumerate(templates):
        r = score_candidate(ctx, t["text"], batch, cluster, bg_sub)
        if r is None:
            fails += 1
            continue
        r.update({"id": t["id"], "text": t["text"]})
        coarse.append(r)
        if (i + 1) % 50 == 0:
            log.info("  %d/%d (%.1fs)", i + 1, len(templates), time.time() - t0)
    log.info("  coarse done: %d scored, %d compile fails (%.1fs)",
             len(coarse), fails, time.time() - t0)
    pl.from_dicts([{k: (str(v) if k in ("caught", "text") else v)
                    for k, v in r.items()} for r in coarse]
                  ).write_csv(out / "grid_coarse.csv")

    selected = select_frontier(coarse)
    log.info("[exact] %d selected on full %d-bg sample", len(selected), len(ctx.bg_pos))
    t0 = time.time()
    exact = []
    for t in selected:
        r = score_candidate(ctx, t["text"], batch, cluster, ctx.bg_pos)
        assert r is not None  # compiled in stage 1
        r.update({"id": t["id"], "text": t["text"],
                  "fires_day": r["bg_rate"] * train_per_day,
                  "pass_absolute": r["recall"] >= need and r["bg_rate"] <= allowed + 1e-9,
                  "pass_marginal": r["recall"] >= need and r["marg_rate"] <= allowed + 1e-9})
        exact.append(r)
    log.info("  exact done (%.1fs)", time.time() - t0)
    for r in pareto(exact):
        r["pareto"] = True
    for r in exact:
        r.setdefault("pareto", False)
    pl.from_dicts([{k: (str(v) if k in ("caught", "text") else v)
                    for k, v in sorted(r.items())} for r in exact]
                  ).write_csv(out / "frontier_exact.csv")
    return out / "frontier_exact.csv"


def write_scatter(exact_csvs: dict[str, Path], out_svg: Path,
                  allowed: float, train_per_day: float) -> None:
    """Combined recall-vs-bg scatter (log-x) across arms, no deps."""
    series = {name: pl.read_csv(p) for name, p in exact_csvs.items()}
    W, H, L, B, T, R = 640, 400, 64, 48, 24, 16
    all_rates = [r for df in series.values() for r in df["bg_rate"].to_list()]
    lo = max(min(all_rates) / 2, 1e-5)
    hi = max(all_rates) * 1.5
    import math
    x = lambda v: L + (math.log10(max(v, lo)) - math.log10(lo)) / (
        math.log10(hi) - math.log10(lo)) * (W - L - R)
    max_recall = max(df["recall"].max() for df in series.values())
    y = lambda r: H - B - r / (max_recall + 0.5) * (H - B - T)
    colors = {"llm": "#d62728", "enum": "#1f77b4", "hybrid": "#2ca02c"}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}">',
             f'<rect width="{W}" height="{H}" fill="white"/>']
    for exp in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1):
        v = 10 ** exp
        if lo <= v <= hi:
            parts.append(f'<line x1="{x(v):.0f}" y1="{T}" x2="{x(v):.0f}" '
                         f'y2="{H - B}" stroke="#eee"/>')
            parts.append(f'<text x="{x(v):.0f}" y="{H - B + 16}" font-size="10" '
                         f'text-anchor="middle">{v:g} ({v * train_per_day:.0f}/d)</text>')
    parts.append(f'<line x1="{x(allowed):.0f}" y1="{T}" x2="{x(allowed):.0f}" '
                 f'y2="{H - B}" stroke="#999" stroke-dasharray="4,3"/>')
    for r in range(max_recall + 1):
        parts.append(f'<line x1="{L}" y1="{y(r):.0f}" x2="{W - R}" '
                     f'y2="{y(r):.0f}" stroke="#eee"/>')
        parts.append(f'<text x="{L - 8}" y="{y(r) + 4:.0f}" font-size="11" '
                     f'text-anchor="end">{r}/4</text>')
    for name, df in series.items():
        c = colors.get(name, "#9467bd")
        for row in df.iter_rows(named=True):
            rad = 6 if row.get("pareto") else 3.5
            op = 1.0 if row.get("pareto") else 0.55
            parts.append(f'<circle cx="{x(row["bg_rate"]):.1f}" cy="{y(row["recall"]):.1f}" '
                         f'r="{rad}" fill="{c}" opacity="{op}"/>')
    lx = W - R - 150
    for i, name in enumerate(series):
        parts.append(f'<circle cx="{lx}" cy="{T + 12 + 20 * i}" r="5" '
                     f'fill="{colors.get(name, "#9467bd")}"/>')
        parts.append(f'<text x="{lx + 10}" y="{T + 16 + 20 * i}" font-size="12">{name}</text>')
    parts.append(f'<text x="{(W + L) / 2:.0f}" y="{H - 6}" font-size="12" '
                 f'text-anchor="middle">train bg rate (fires/day at train volume)</text>')
    parts.append("</svg>")
    out_svg.write_text("\n".join(parts))
