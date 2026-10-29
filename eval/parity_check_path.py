#!/usr/bin/env python3
"""Large-scale parity: numba path kernel vs the pure-Python reference walk.

For every path motif checked, scores the same anchors with the fast path
(``matcher.score`` per anchor, plus ``score_path_batch`` serial and
``parallel=True``) and with the pure-Python walk (``matcher.PURE_PYTHON``,
run in forked worker processes), across all 12 modes the harnesses use:
prune on/off x crisp off / crisp at alpha 0 / crisp at alpha=emit x floor
None / EXACT_FLOOR. Every field must match bit for bit: found, witness,
score, top_score, degrees (floats compared by their hex), truncated and
expansions (the batch rows have no degrees).

Anchor sets per motif:
* ``random``: ``--n-random`` random train anchors (seeded);
* ``hub_pure``: the ``--n-hub`` anchors with the most expansions in a
  pure-Python pre-scan (unpruned, fuzzy, emit floor) over ``--prescan``
  further random train anchors -- hub stress, mostly truncated;
* ``hub_all``: the ``--n-hub`` anchors with the most expansions over *all*
  train anchors, found with the fast batch kernel (the pure side then
  re-derives every field, expansions included).

The pre-scan itself is also a parity check (pure vs fast expansions on
the whole pre-scan pool, one mode). Default watermark, so per-anchor
results do not depend on evaluation order and pure workers may split the
anchors freely.

Usage:
    uv run eval/parity_check_path.py                      # amlworld path motifs
    uv run eval/parity_check_path.py --rq4 --n-random 2000  # + RQ4 path references
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval import splits  # noqa: E402
from motifql import matcher as M  # noqa: E402
from motifql.degrees import libs_from_doc  # noqa: E402
from motifql.graph import TemporalGraph  # noqa: E402
from motifql.parser import parse_file, parse_text  # noqa: E402
from motifql.planner import PathPlan, compile_motif  # noqa: E402

log = logging.getLogger("parity_check")

MODES = [(prune, crisp, alpha, floor)
         for prune in (True, False)
         for crisp, alpha in ((False, "0"), (True, "0"), (True, "emit"))
         for floor in (None, M.EXACT_FLOOR)]

# Fork-inherited worker state: graph plus {plan key: (plan, libs)}.
_G: TemporalGraph | None = None
_PLANS: dict[str, tuple] = {}


def _alpha(plan, alpha: str) -> float:
    return float(plan.emit) if alpha == "emit" else float(alpha)


def _key(r: M.ScoredResult) -> tuple:
    """Bit-exact comparison key for one ScoredResult."""
    degs = tuple((lab, float(d).hex(), w) for lab, d, w in r.degrees)
    return (bool(r.found), tuple(int(e) for e in r.witness), float(r.score).hex(),
            float(r.top_score).hex(), degs, bool(r.truncated), int(r.expansions),
            type(r.expansions).__name__)


def _batch_key(r: M.ScoredResult) -> tuple:
    return (bool(r.found), tuple(int(e) for e in r.witness), float(r.score).hex(),
            float(r.top_score).hex(), bool(r.truncated), int(r.expansions))


def _batch_row(b: M.PathBatch, i: int) -> tuple:
    return (bool(b.found[i]), tuple(b.witness[i, :b.witness_len[i]].tolist()),
            float(b.score[i]).hex(), float(b.top_score[i]).hex(), bool(b.truncated[i]),
            int(b.expansions[i]))


def _pure_chunk(args) -> list:
    """Worker: pure-Python results for (plan key, mode index, anchors)."""
    pkey, mi, anchors = args
    plan, libs = _PLANS[pkey]
    prune, crisp, alpha, floor = MODES[mi]
    M.PURE_PYTHON = True
    return [M.score(_G, int(a), plan, libs, prune, crisp, _alpha(plan, alpha), floor)
            for a in anchors]


def _pure_all(pool, pkey: str, mi: int, anchors: np.ndarray, chunk: int) -> list:
    # Small jobs for small (hub) sets, so every worker gets some.
    chunk = max(1, min(chunk, len(anchors) // (4 * pool._processes)))
    jobs =[(pkey, mi, anchors[i:i + chunk]) for i in range(0, len(anchors), chunk)]
    out: list = []
    for part in pool.imap(_pure_chunk, jobs):  # imap keeps anchor order
        out.extend(part)
    return out


def _plans(doc_path: Path, motifs: list[str], rq4: bool) -> dict[str, tuple]:
    doc = parse_file(doc_path)
    libs = libs_from_doc(doc)
    plans = {m: (compile_motif(doc, m), libs) for m in motifs}
    if rq4:
        from eval.rq4_items import ITEMS

        for item in ITEMS:
            rdoc = parse_text(item["reference"])
            plan = compile_motif(rdoc, "ref")
            if isinstance(plan, PathPlan):
                plans[f"rq4:{item['id']}"] = (plan, libs_from_doc(rdoc))
    return plans


def main(
    data: Annotated[Path, typer.Option(help="Prepped dataset directory.")] = Path("data/hi-small"),
    motif_file: Annotated[Path, typer.Option(help="Motif file.")] = Path("examples/amlworld.motif"),
    motifs: Annotated[str, typer.Option(help="Comma-separated path motifs.")] = "round_trip,layering_chain",
    rq4: Annotated[bool, typer.Option(help="Also check every RQ4 path reference.")] = False,
    n_random: Annotated[int, typer.Option(help="Random train anchors per motif.")] = 20_000,
    prescan: Annotated[int, typer.Option(help="Pure pre-scan pool size for hub_pure.")] = 60_000,
    n_hub: Annotated[int, typer.Option(help="Hub anchors per hub set.")] = 200,
    workers: Annotated[int, typer.Option(help="Forked pure-Python workers.")] = 16,
    chunk: Annotated[int, typer.Option(help="Anchors per worker job.")] = 250,
    seed: Annotated[int, typer.Option(help="Anchor sampling seed.")] = 0,
    fallback_anchors: Annotated[int, typer.Option(
        help="Random anchors for plans the kernel does not run (pure on both sides).")] = 300,
) -> None:
    global _G, _PLANS
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    t0 = time.perf_counter()
    df = pl.read_parquet(data / "transfers.parquet")
    _G = TemporalGraph.from_transfers(df)
    days = df.get_column("day").to_numpy()
    train = np.flatnonzero((days >= np.datetime64(str(splits.TRAIN_START)))
                           & (days <= np.datetime64(str(splits.TRAIN_END))))
    _PLANS = _plans(motif_file, [m.strip() for m in motifs.split(",") if m.strip()], rq4)
    log.info("graph %d edges, %d train anchors, %d plans (%.1fs)", _G.n_edges, len(train),
             len(_PLANS), time.perf_counter() - t0)
    rng = np.random.default_rng(seed)
    pool = mp.get_context("fork").Pool(workers)
    total_checked = total_bad = 0
    summary: list[str] = []
    try:
        for pkey, (plan, libs) in _PLANS.items():
            fast_ok = M._compiled_plan(plan, libs, False, 0.0) is not None
            perm = rng.permutation(train)
            rand = np.sort(perm[:n_random if fast_ok else min(n_random, fallback_anchors)])
            sets = {"random": rand}
            if n_hub > 0 and fast_ok and not pkey.startswith("rq4:"):
                pool_anchors = np.sort(perm[n_random:n_random + prescan])
                t = time.perf_counter()
                pre = _pure_all(pool, pkey, MODES.index((False, False, "0", None)),
                                pool_anchors, chunk)
                pre_exp = np.array([r.expansions for r in pre])
                fb = M.score_path_batch(_G, pool_anchors, plan, libs, prune=False)
                pre_bad = int((fb.expansions != pre_exp).sum())
                total_bad += pre_bad
                total_checked += len(pool_anchors)
                sel = np.argsort(-pre_exp, kind="stable")[:n_hub]
                sets["hub_pure"] = np.sort(pool_anchors[sel])
                allb = M.score_path_batch(_G, train, plan, libs, prune=False, parallel=True)
                sel_all = np.argsort(-allb.expansions, kind="stable")[:n_hub]
                sets["hub_all"] = np.sort(train[sel_all])
                log.info("%s: pre-scan %d anchors (%.0fs): expansion mismatches %d; "
                         "hub_pure exp %d..%d (%d truncated); hub_all exp %d..%d (%d truncated)",
                         pkey, len(pool_anchors), time.perf_counter() - t, pre_bad,
                         pre_exp[sel].min(), pre_exp[sel].max(),
                         sum(pre[i].truncated for i in sel), allb.expansions[sel_all].min(),
                         allb.expansions[sel_all].max(), int(allb.truncated[sel_all].sum()))
                summary.append(f"{pkey:28s} prescan    n={len(pool_anchors):6d} "
                               f"mode=unpruned/fuzzy/emit-floor expansion mismatches={pre_bad}")
            for sname, anchors in sets.items():
                for mi, mode in enumerate(MODES):
                    prune, crisp, alpha, floor = mode
                    a = _alpha(plan, alpha)
                    t = time.perf_counter()
                    M.PURE_PYTHON = False
                    fast = [M.score(_G, int(x), plan, libs, prune, crisp, a, floor)
                            for x in anchors]
                    t_fast = time.perf_counter() - t
                    bser = M.score_path_batch(_G, anchors, plan, libs, prune, crisp, a, floor)
                    bpar = M.score_path_batch(_G, anchors, plan, libs, prune, crisp, a, floor,
                                              parallel=True)
                    t = time.perf_counter()
                    pure = _pure_all(pool, pkey, mi, anchors, chunk)
                    t_pure = time.perf_counter() - t
                    bad = bad_b = 0
                    for i, (f, p) in enumerate(zip(fast, pure, strict=True)):
                        if _key(f) != _key(p):
                            bad += 1
                            if bad <= 3:
                                log.error("MISMATCH %s/%s/%s anchor %d:\n fast %s\n pure %s",
                                          pkey, sname, mode, int(anchors[i]), f, p)
                        pk = _batch_key(p)
                        if _batch_row(bser, i) != pk or _batch_row(bpar, i) != pk:
                            bad_b += 1
                            if bad_b <= 3:
                                log.error("BATCH MISMATCH %s/%s/%s anchor %d", pkey, sname,
                                          mode, int(anchors[i]))
                    total_checked += len(anchors)
                    total_bad += bad + bad_b
                    trunc = sum(r.truncated for r in pure)
                    line = (f"{pkey:28s} {sname:9s} n={len(anchors):6d} prune={prune!s:5s} "
                            f"crisp={crisp!s:5s} alpha={alpha:4s} floor={floor!s:5s} "
                            f"mismatch score={bad} batch={bad_b} trunc={trunc} "
                            f"fast={t_fast / len(anchors) * 1e6:.0f}us/anchor "
                            f"pure(wall/{workers})={t_pure / len(anchors) * 1e6:.0f}us/anchor"
                            f"{'' if fast_ok else ' [pure fallback plan]'}")
                    log.info(line)
                    summary.append(line)
    finally:
        pool.close()
        pool.join()
    log.info("\n".join(["", "SUMMARY"] + summary))
    log.info("TOTAL: %d anchor-mode comparisons, %d mismatches (%.0fs)", total_checked,
             total_bad, time.perf_counter() - t0)
    if total_bad:
        raise SystemExit(1)


if __name__ == "__main__":
    typer.run(main)
