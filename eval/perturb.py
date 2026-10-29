#!/usr/bin/env python3
"""RQ2 perturbation runs: evasion-robustness curves (Nov 1).

Takes real train+holdout instances of each motif's target pattern,
perturbs ONLY their member transfers (amount jitter, order-preserving
gap scaling), and replays truth-member anchors plus a fixed background
sample at every level. Metrics are threshold-based (recall at the
motif's emit threshold + background fire rate + mean top_score), so no
full-day ranking is needed — this stays a targeted replay, not a batch.

Perturbation semantics (pure function `perturb_frame`, unit-tested):
* amount: member amount_usd *= 1 + U(-p, +p), drawn per transfer_id
  from SeedSequence(seed, tid) — independent of which other
  instances share the run.
* gaps: per instance, t'_i = t_min + f * (t_i - t_min) in integer
  microseconds — within-instance order preserved by construction
  (ties stay tied), all within-instance gaps scale by f.
* level (0, x1) is bit-identical to the base frame.

Pool (review #14.2): every TARGET-pattern instance with a member on
a scored (train+holdout) day is perturbed whole, whatever its
original split — restricting to train+holdout splits left most
scored targets (partial/crosser members) clean. Splits are
recomputed from the perturbed frame, so drifted instances
reclassify instead of haunting the wrong denominator; transfer
labels (is_laundering, pattern_type) are timeless. No fitting
happens here (robustness characterization, not model selection).

Usage:
    uv run eval/perturb.py --data data/hi-small --workers 16
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
from motifql.matcher import EXACT_FLOOR  # noqa: E402
from motifql.planner import compile_motif  # noqa: E402

log = logging.getLogger("perturb")

LEVELS = [(0.0, 1.0), (0.05, 1.0), (0.10, 1.0), (0.20, 1.0), (0.0, 1.5), (0.0, 2.0)]
TARGETS = {"round_trip": "CYCLE", "layering_chain": "RANDOM",
           "scatter_gather": "SCATTER-GATHER"}
BG_SEED = 7
BG_N = 20000


def perturb_frame(df: pl.DataFrame, inst_members: dict[int, list[int]],
                  amount_pct: float, gap_factor: float, seed: int) -> pl.DataFrame:
    """Perturb member transfers; same row order, bit-identical at (0, x1).

    Amount draws are per transfer_id (SeedSequence(seed, tid)), so one
    instance's noise never depends on which others share the run.
    """
    member_ids = sorted({t for ts in inst_members.values() for t in ts})
    if not member_ids:
        return df
    factor = {t: 1.0 + float(np.random.default_rng([seed, int(t)]).uniform(
        -amount_pct, amount_pct)) for t in member_ids}
    # Per-instance gap scaling in integer microseconds.
    ts_us = (df.select("transfer_id", "timestamp")
             .with_columns(pl.col("timestamp").dt.epoch(time_unit="us").alias("us")))
    id2us = dict(zip(ts_us["transfer_id"].to_list(), ts_us["us"].to_list()))
    new_ts: dict[int, int] = {}
    for tids in inst_members.values():
        us = [id2us[t] for t in tids]
        t0 = min(us)
        for t, u in zip(tids, us):
            new_ts[t] = t0 + int(round(gap_factor * (u - t0)))
    pert = pl.DataFrame({
        "transfer_id": member_ids,
        "new_amount": [None] * len(member_ids),  # filled below (needs old amount)
        "new_us": [new_ts[t] for t in member_ids],
    })
    old_amount = dict(zip(df["transfer_id"].to_list(), df["amount_usd"].to_list()))
    pert = pert.with_columns(
        pl.Series("new_amount", [old_amount[t] * factor[t] for t in member_ids]))
    out = (df.join(pert, on="transfer_id", how="left", maintain_order="left")
           .with_columns(
               pl.coalesce("new_amount", "amount_usd").alias("amount_usd"),
               pl.coalesce(pl.col("new_us").cast(pl.Datetime("us")), "timestamp").alias("timestamp"),
           )
           .drop("new_amount", "new_us")
           # Members may drift across midnight: recompute the day.
           .with_columns(pl.col("timestamp").dt.date().alias("day")))
    return out


def _positions(sorted_ids: np.ndarray, want: list[int]) -> np.ndarray:
    lut = np.full(int(sorted_ids.max()) + 1, -1, dtype=np.int64)
    lut[sorted_ids] = np.arange(len(sorted_ids))
    pos = lut[np.asarray(want)]
    assert (pos >= 0).all(), "wanted transfer_id missing from frame"
    return pos


def _resplit(pdf: pl.DataFrame, memb: pl.DataFrame, inst: pl.DataFrame) -> pl.DataFrame:
    """Instance splits recomputed from perturbed member timestamps.

    Drifted instances reclassify (completable → partial, train →
    crosser...) instead of haunting their original split's
    denominator (review #14.2/#14.3). At level (0, x1) the frame is
    bit-identical, so splits reproduce exactly.
    """
    spans = (memb.join(pdf.select("transfer_id", "timestamp"), on="transfer_id", how="left")
             .group_by("instance_id").agg(pl.col("timestamp").min().alias("t_min"),
                                           pl.col("timestamp").max().alias("t_max")))
    inst2 = inst.drop("t_min", "t_max").join(spans, on="instance_id", how="left")
    return splits.assign_instance_splits(inst2, memb, pdf)


BUDGET_LEVELS = [(0.0, 1.0), (0.10, 1.0), (0.20, 1.0), (0.0, 2.0), (0.20, 2.0)]


def main(
    data: Annotated[Path, typer.Option(help="Prepped dataset directory.")] = Path("data/hi-small"),
    levels: Annotated[str, typer.Option(help="Levels as pct:factor,pct:factor.")] = ",".join(f"{p}:{f}" for p, f in LEVELS),
    motifs: Annotated[str, typer.Option(help="Comma-separated motif names.")] = ",".join(TARGETS),
    bg_n: Annotated[int, typer.Option(help="Background sample size.")] = BG_N,
    seed: Annotated[int, typer.Option(help="Perturbation seed.")] = 12345,
    workers: Annotated[int, typer.Option(help="Worker processes (1 = serial).")] = 16,
    out: Annotated[Path, typer.Option(help="Directory for the CSV.")] = Path("eval/results"),
    budget: Annotated[bool, typer.Option(help="Budget-recall mode: full holdout replay + ranking per level.")] = False,
    budget_levels: Annotated[str, typer.Option(help="Levels for budget mode.")] = ",".join(f"{p}:{f}" for p, f in BUDGET_LEVELS),
    budgets: Annotated[str, typer.Option(help="Alerts/day budgets (budget mode).")] = "50,100,500,1000,5000",
    crisp_alpha: Annotated[str, typer.Option(help="Exact-arm α-cut: `emit`, `support` or float.")] = "emit",
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    level_list = [tuple(float(x) for x in lv.split(":")) for lv in
                  (budget_levels if budget else levels).split(",")]
    motifs_wanted = [m.strip() for m in motifs.split(",") if m.strip()]
    budget_list = [int(b) for b in budgets.split(",")]
    out.mkdir(parents=True, exist_ok=True)

    log.info("[1/3] loading transfers, ground truth, background sample")
    df = pl.read_parquet(data / "transfers.parquet")
    memb = pl.read_parquet(data / "pattern_members.parquet")
    inst = splits.assign_instance_splits(
        pl.read_parquet(data / "pattern_instances.parquet"), memb, df)
    pool = df.filter(pl.col("day").is_between(splits.TRAIN_START, splits.HOLDOUT_END))
    bg_ids = (pool.filter(pl.col("is_laundering") == 0)
              .select("transfer_id").to_series().to_list())
    bg_ids = sorted(np.random.default_rng(BG_SEED).choice(bg_ids, size=min(bg_n, len(bg_ids)),
                                                          replace=False).tolist())
    # Pool-day member tids (original frame): an instance joins the
    # pool when ANY member sits on a scored day, whatever its split.
    pool_tids = set(pool.get_column("transfer_id").to_list())
    mm_pat = memb.join(inst.select(["instance_id", "pattern_type"]),
                       on="instance_id", how="left")
    log.info("  transfers %d, background sample %d", len(df), len(bg_ids))

    log.info("[2/3] compiling motifs")
    doc = parse_file(Path("examples/amlworld.motif"))
    libs = libs_from_doc(doc)
    plans = {m: compile_motif(doc, m) for m in motifs_wanted}
    alphas = {m: evaluate.resolve_crisp_alpha(crisp_alpha, plans[m].emit)
              for m in motifs_wanted}
    log.info("  crisp α: %s",
             ", ".join(f"{m}={a:g}" for m, a in alphas.items()))

    rows: list[dict] = []
    btr_rows, binst_rows = [], []
    for pct, gf in level_list:
        log.info("[3/3] level pct=%.2f gap=x%.1f: perturbing + rebuilding graph", pct, gf)
        target_members: dict[str, dict[int, list[int]]] = {}
        for m in motifs_wanted:
            sub = mm_pat.filter(pl.col("pattern_type") == TARGETS[m])
            touched = {int(i) for i, t in zip(sub["instance_id"].to_list(),
                                              sub["transfer_id"].to_list())
                       if int(t) in pool_tids}
            d: dict[int, list[int]] = {}
            for iid, tid in zip(sub["instance_id"].to_list(), sub["transfer_id"].to_list()):
                if int(iid) in touched:
                    d.setdefault(int(iid), []).append(int(tid))
            target_members[m] = d
            log.info("  %s: %d instances (%d members) touch scored days",
                     m, len(d), sum(len(v) for v in d.values()))
        union: dict[int, list[int]] = {}
        for d in target_members.values():
            for iid, tids in d.items():
                union.setdefault(iid, []).extend(tids)
        pdf = perturb_frame(df, union, pct, gf, seed)
        # Deterministic total order: ties by transfer_id.
        pdf = pdf.sort("timestamp", "transfer_id")
        graph = TemporalGraph.from_transfers(pdf)
        tids = pdf.get_column("transfer_id").to_numpy()

        rq1._GRAPH, rq1._LIBS = graph, libs
        if budget:
            hdays = pdf.get_column("day").to_numpy()
            hold_pos = np.flatnonzero(
                (hdays >= np.datetime64(str(splits.HOLDOUT_START)))
                & (hdays <= np.datetime64(str(splits.HOLDOUT_END))))
            log.info("  holdout anchors: %d (drifted members leave the set)", len(hold_pos))
            scores: dict[str, np.ndarray] = {}
            for m in motifs_wanted:
                plan = plans[m]
                rq1._PLAN = plan
                rq1._PRUNE, rq1._CRISP, rq1._CALPHA, rq1._FLOOR = True, True, alphas[m], None
                cr_top, _, _, _ = rq1._replay(hold_pos, workers, f"{m} crisp")
                scores[f"{m}__crisp"] = cr_top
                # Exact branch-and-bound (review #14.1): no screen.
                rq1._PRUNE, rq1._CRISP, rq1._CALPHA, rq1._FLOOR = True, False, 0.0, EXACT_FLOOR
                fz_top, _, _, _ = rq1._replay(hold_pos, workers, f"{m} fuzzy")
                scores[f"{m}__fuzzy"] = fz_top
            hold_base = evaluate.holdout_frame(
                pdf, memb, inst, splits.HOLDOUT_START, splits.HOLDOUT_END, len(hold_pos))
            # Splits from the perturbed frame (review #14.2/#14.3);
            # pattern labels stay timeless. Log the completable mix
            # so drift across levels is visible, not silent.
            inst2 = _resplit(pdf, memb, inst)
            n_comp = inst2.filter(pl.col("split") == "holdout").height
            log.info("  holdout-completable instances after drift: %d", n_comp)
            # Hash tie-break only: at a calibrated α the emitter set fits
            # inside the budgets, so ordering barely matters (rq2 proves
            # the sensitivity separately with --exact-order).
            btr, binst = evaluate.recall_at_budgets(hold_base, memb, inst2, scores, budget_list)
            tag = {"amount_pct": pct, "gap_factor": gf}
            btr_rows.append(btr.with_columns(pl.lit(tag["amount_pct"]).alias("amount_pct"),
                                             pl.lit(tag["gap_factor"]).alias("gap_factor")))
            binst_rows.append(binst.with_columns(pl.lit(tag["amount_pct"]).alias("amount_pct"),
                                                pl.lit(tag["gap_factor"]).alias("gap_factor")))
            continue
        for m in motifs_wanted:
            plan = plans[m]
            rq1._PLAN = plan
            tm = target_members[m]
            truth_ids = sorted({t for ts in tm.values() for t in ts})
            truth_set = set(truth_ids)
            anchor_ids = truth_ids + [b for b in bg_ids if b not in truth_set]
            # _replay realigns by searchsorted: positions must be sorted.
            apos = _positions(tids, anchor_ids)
            anchor_ids = [anchor_ids[i] for i in np.argsort(apos)]
            apos = _positions(tids, anchor_ids)
            apos_of = {t: i for i, t in enumerate(anchor_ids)}
            is_truth = np.array([t in truth_set for t in anchor_ids])
            rq1._PRUNE, rq1._CRISP, rq1._CALPHA, rq1._FLOOR = True, True, alphas[m], None
            cr_top, _, _, _ = rq1._replay(apos, workers, f"{m} crisp")
            # Exact branch-and-bound (review #14.1): no screen.
            rq1._PRUNE, rq1._CRISP, rq1._CALPHA, rq1._FLOOR = True, False, 0.0, EXACT_FLOOR
            fz_top, _, _, _ = rq1._replay(apos, workers, f"{m} fuzzy")
            emit = plan.emit
            inst_idx = {iid: np.array([apos_of[t] for t in tids_i])
                        for iid, tids_i in tm.items()}
            for arm, top in (("fuzzy", fz_top), ("crisp", cr_top)):
                t_top = top[is_truth]
                b_top = top[~is_truth]
                caught = {iid for iid, idx in inst_idx.items()
                          if (top[idx] >= emit).any()}
                rows.append({
                    "motif": m, "arm": arm, "alpha": alphas[m] if arm == "crisp" else None,
                    "amount_pct": pct, "gap_factor": gf,
                    "n_truth": len(truth_ids),
                    "transfer_recall": float((t_top >= emit).mean()) if len(t_top) else 0.0,
                    "n_inst": len(tm), "inst_recall": len(caught) / len(tm) if tm else 0.0,
                    "mean_truth_top": float(t_top.mean()) if len(t_top) else 0.0,
                    "n_bg": len(b_top),
                    "bg_fire_rate": float((b_top >= emit).mean()) if len(b_top) else 0.0,
                    "mean_bg_top": float(b_top.mean()) if len(b_top) else 0.0,
                })
                log.info("  %-14s %-5s transfer %.3f instance %.3f (%d/%d) "
                         "mean_top %.3f bg_fire %.4f",
                         m, arm, rows[-1]["transfer_recall"], rows[-1]["inst_recall"],
                         len(caught), len(tm), rows[-1]["mean_truth_top"],
                         rows[-1]["bg_fire_rate"])
    if budget:
        split = [pl.col("key").str.split("__").list.get(0).alias("motif"),
                 pl.col("key").str.split("__").list.get(1).alias("arm")]
        alpha_of = pl.col("motif").replace_strict(alphas, default=None).alias("alpha")
        pl.concat(btr_rows).with_columns([*split, alpha_of]).write_csv(
            out / "rq2_perturb_budget_transfer.csv")
        pl.concat(binst_rows).with_columns([*split, alpha_of]).write_csv(
            out / "rq2_perturb_budget_instance.csv")
        log.info("  wrote rq2_perturb_budget_{transfer,instance}.csv")
    else:
        pl.from_dicts(rows).write_csv(out / "rq2_perturb.csv")
        log.info("  wrote %s", out / "rq2_perturb.csv")
    log.info("done")


if __name__ == "__main__":
    typer.run(main)
