#!/usr/bin/env python3
"""RQ2 harness: fuzzy vs exact (α-cut crisp) recall at fixed budgets.

Level-0 table: replays the fuzzy-sensitive motifs (round_trip,
layering_chain, scatter_gather) over the holdout split twice, both
pruned — crisp from emit (exact arm: binary scores, every crisp-1
witness completes) and fuzzy from EXACT_FLOOR (branch-and-bound for
the exact best witness; unpruned search truncates and loses
positives). --verify-prune checks pruned ⊇ unpruned on a seeded
spread sample. Binarity/differential asserts are scoped per motif
by the safety predicates (most/few/named grade under crisp;
`not` breaks the differential) and decided before any replay.

The exact arm is a per-leaf α-cut (#15 A): the old support boundary
(α=0) is the loosest exact rule a motif allows, so every candidate
tied at 1.0 and ranked by lottery. The default α is each motif's own
emit threshold — no new fitting — with `support` reproducing the old
arm. At α>0 the differential is crisp_top == (fuzzy_top >= α) on
`min` motifs (product has no such equality); any mismatch is an
evaluator bug, not a finding.

--exact-order ranks the binary exact arm under several label-blind
tie-breaks (hash, amount desc/asc): at a calibrated α the emitter
set fits inside operational budgets, so the lottery should barely
matter — the sensitivity rows prove it instead of assuming it.

Usage:
    uv run eval/rq2.py --data data/hi-small --workers 16
    uv run eval/rq2.py --data data/hi-small --max-anchors 20000 --workers 4
    uv run eval/rq2.py --data data/hi-small --crisp-alpha support
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
from motifql.degrees import crisp_binary_safe, crisp_differential_safe, libs_from_doc  # noqa: E402
from motifql.graph import TemporalGraph  # noqa: E402
from motifql.matcher import EXACT_FLOOR  # noqa: E402
from motifql.parser import parse_file  # noqa: E402
from motifql.planner import PathPlan, compile_motif  # noqa: E402

log = logging.getLogger("rq2")

MOTIFS = ["round_trip", "layering_chain", "scatter_gather"]


def main(
    data: Annotated[Path, typer.Option(help="Prepped dataset directory.")] = Path("data/hi-small"),
    holdout: Annotated[str, typer.Option(help="Holdout day range, inclusive.")] = f"{splits.HOLDOUT_START},{splits.HOLDOUT_END}",
    motifs: Annotated[str, typer.Option(help="Comma-separated motif names.")] = ",".join(MOTIFS),
    budgets: Annotated[str, typer.Option(help="Alerts/day budgets, comma-separated.")] = "50,100,500,1000,5000",
    crisp_alpha: Annotated[str, typer.Option(help="Exact-arm α-cut: `emit`, `support` or float.")] = "emit",
    exact_order: Annotated[str, typer.Option(help="Exact tie-breaks: hash,amount_desc,amount_asc,all.")] = "all",
    max_anchors: Annotated[int, typer.Option(help="Cap holdout anchors (0 = all).")] = 0,
    verify_prune: Annotated[int, typer.Option(help="Crisp pruned-vs-unpruned check anchors (0 = off).")] = 20000,
    workers: Annotated[int, typer.Option(help="Worker processes (1 = serial).")] = 16,
    out: Annotated[Path, typer.Option(help="Directory for the recall CSVs.")] = Path("eval/results"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ho0, ho1 = (date.fromisoformat(s) for s in holdout.split(","))
    motifs_wanted = [m.strip() for m in motifs.split(",") if m.strip()]
    budget_list = [int(b) for b in budgets.split(",")]
    orders = [o.strip() for o in exact_order.split(",") if o.strip()]
    if orders == ["all"]:
        orders = ["hash", "amount_desc", "amount_asc"]
    if not orders or any(o not in ("hash", "amount_desc", "amount_asc") for o in orders):
        raise SystemExit(f"--exact-order {exact_order!r}: use hash,amount_desc,amount_asc,all")
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
    plans = {m: compile_motif(doc, m) for m in motifs_wanted}

    # Exact-arm α per motif, decided BEFORE any replay: the motif's
    # own emit by default (no new fitting), `support` for the old arm.
    alphas = {m: evaluate.resolve_crisp_alpha(crisp_alpha, plans[m].emit)
              for m in motifs_wanted}
    log.info("  crisp α: %s",
             ", ".join(f"{m}={a:g}" for m, a in alphas.items()))

    # Check scope per motif, decided BEFORE any replay (review
    # #14.6 + Codex P2): binarity needs no mean-grading quants, the
    # differential additionally no `not` (and `min` at α>0).
    # Out-of-scope checks are skipped with a log line, never abort
    # after a full replay.
    scopes: dict[str, tuple[bool, bool]] = {}
    for name, plan in plans.items():
        if isinstance(plan, PathPlan):
            parts: list = [a for grp in plan.prunable for a in grp.atoms]
            if plan.deferred is not None:
                parts.append(plan.deferred)
        else:
            parts = [plan.soft] if plan.soft is not None else []
        scopes[name] = (
            all(crisp_binary_safe(p) for p in parts),
            all(crisp_differential_safe(p, alpha=alphas[name],
                                        tnorm=plan.score_tnorm) for p in parts))
        log.info("  %s: checks binary=%s differential=%s", name, *scopes[name])

    rq1._GRAPH, rq1._LIBS = graph, libs
    scored: dict[str, dict[str, np.ndarray]] = {}
    summary_rows: list[dict] = []
    log.info("[3/4] replaying holdout: crisp pruned, fuzzy exact floor")
    for name, plan in plans.items():
        rq1._PLAN = plan
        alpha = alphas[name]
        # Crisp arm, pruned from emit: every crisp-1 witness completes
        # (all its leaves read 1 along the way).
        rq1._PRUNE, rq1._CRISP, rq1._CALPHA, rq1._FLOOR = True, True, alpha, None
        cr_top, cr_found, cr_exps, cr_tr = rq1._replay(hold_pos, workers, f"{name} crisp")
        # Fuzzy arm, branch-and-bound from ~0: exact best witness
        # (mod 1e-12) without unpruned search, which truncates and
        # LOSES positives (review #14.1). Groups ignore prune/floor
        # and are exact by construction.
        rq1._PRUNE, rq1._CRISP, rq1._CALPHA, rq1._FLOOR = True, False, 0.0, EXACT_FLOOR
        fz_top, fz_found, fz_exps, fz_tr = rq1._replay(hold_pos, workers, f"{name} fuzzy")
        arms = {"fuzzy": {"top": fz_top, "found": fz_found},
                "crisp": {"top": cr_top, "found": cr_found}}
        scored[name] = arms
        for arm, top, found, exps, trunc in (
                ("fuzzy", fz_top, fz_found, fz_exps, fz_tr),
                ("crisp", cr_top, cr_found, cr_exps, cr_tr)):
            summary_rows.append({
                "motif": name, "arm": arm,
                "alpha": alpha if arm == "crisp" else None,
                "anchors": len(top),
                "fires": int(found.sum()), "fires_per_day": float(found.sum()) / n_ho_days,
                "nonzero": int((top > 0).sum()),
                "nonzero_per_day": float((top > 0).sum()) / n_ho_days,
                "top_p50": float(np.percentile(top, 50)),
                "top_p99": float(np.percentile(top, 99)),
                "top_p999": float(np.percentile(top, 99.9)),
                "mean_expansions": float(exps.mean()),
                "truncated_rate": float(trunc.mean()),
            })
            log.info("  %s %s: fires %d (%.1f/day) nonzero %.1f/day "
                     "mean expansions %.1f truncated %.2f%%",
                     name, arm, int(found.sum()), found.sum() / n_ho_days,
                     (top > 0).sum() / n_ho_days, float(exps.mean()),
                     100 * trunc.mean())

        # Crisp differential on anchors where both arms completed.
        # Truncation can still strike the two searches differently,
        # so truncated anchors are reported, not asserted. At α=0 the
        # rule is crisp>0 ⟺ fuzzy>0; at α>0 (min motifs) it is
        # crisp=1 ⟺ fuzzy>=α — exact, since min selects (no rounding).
        binary_ok, diff_ok = scopes[name]
        if binary_ok:
            nonbinary = int(((cr_top > 0) & (cr_top < 1)).sum())
            if nonbinary:
                raise SystemExit(f"crisp nonbinary for {name}: {nonbinary}")
            log.info("  %s binary check: clean", name)
        else:
            log.info("  %s binary check: skipped (mean-grading quantifiers)", name)
        if diff_ok:
            expect = (fz_top > 0) if alpha <= 0 else (fz_top >= alpha)
            both_ok = ~cr_tr & ~fz_tr
            mismatch = int((((cr_top > 0) != expect) & both_ok).sum())
            trunc_mm = int((((cr_top > 0) != expect) & ~both_ok).sum())
            log.info("  %s differential (α=%g): mismatch=%d (+%d on truncated)",
                     name, alpha, mismatch, trunc_mm)
            if mismatch:
                raise SystemExit(f"crisp differential failed for {name}: {mismatch}")
        else:
            log.info("  %s differential: skipped (not/mean-grading/product present)", name)

        if 0 < verify_prune <= len(hold_pos):
            # Seeded spread sample (review #14.5): the old head slice
            # covered 24 minutes with single-digit positives.
            vs = np.sort(np.random.default_rng(999).choice(
                len(hold_pos), size=verify_prune, replace=False))
            sub = hold_pos[vs]
            if isinstance(plan, PathPlan):
                rq1._PRUNE, rq1._CRISP, rq1._CALPHA, rq1._FLOOR = True, True, alpha, None
                cr_sub, _, _, _ = rq1._replay(sub, min(workers, 4), f"{name} verify-cp")
                rq1._PRUNE, rq1._CRISP, rq1._CALPHA, rq1._FLOOR = False, True, alpha, None
                cu_sub, _, _, cu_tr = rq1._replay(sub, min(workers, 4), f"{name} verify-cu")
                # Pruning only skips zero-leaf branches and truncates
                # less (subset of expansions), so pruned ⊇ unpruned:
                # every unpruned-1 must be pruned-1. The reverse may
                # differ exactly where unpruned truncated.
                lost = int(((cu_sub > 0) & ~(cr_sub > 0)).sum())
                gained_ok = int(((cr_sub > 0) & ~(cu_sub > 0) & ~cu_tr).sum())
                if lost or gained_ok:
                    raise SystemExit(f"crisp prune check failed for {name}: "
                                     f"lost={lost} gained_completed={gained_ok}")
                log.info("  %s verify: pruned superset holds (%d positives in slice)",
                         name, int((cr_sub > 0).sum()))
            else:
                log.info("  %s verify: prune check skipped (groups never prune)", name)
            # Determinism pin: same computation, different pool
            # sharding, must agree exactly.
            rq1._PRUNE, rq1._CRISP, rq1._CALPHA, rq1._FLOOR = True, False, 0.0, EXACT_FLOOR
            fz_sub, _, _, _ = rq1._replay(sub, min(workers, 4), f"{name} verify-fz")
            if not np.array_equal(fz_sub, fz_top[vs]):
                raise SystemExit(f"fuzzy rerun not deterministic for {name}")
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
    hold_base = evaluate.with_tiebreak(hold_base)  # once, not per motif/arm/order
    tr_rows, inst_rows = [], []
    for name in motifs_wanted:
        # Fuzzy ranks once (graded scores, few ties); the binary exact
        # arm ranks under each requested label-blind tie-break.
        arm_orders = {"fuzzy": ["hash"],
                      "crisp": orders}
        for arm, arm_order_list in arm_orders.items():
            for order in arm_order_list:
                if order == "hash":
                    hold = evaluate.rank_per_day(
                        hold_base.with_columns(
                            pl.Series("score", scored[name][arm]["top"])))
                elif order == "amount_desc":
                    hold = evaluate.rank_per_day(
                        hold_base.with_columns(
                            pl.Series("score", scored[name][arm]["top"])),
                        secondary_col="amount_usd")
                else:
                    hold = evaluate.rank_per_day(
                        hold_base.with_columns(
                            pl.Series("score", scored[name][arm]["top"])),
                        secondary_col="amount_usd", secondary_desc=False)
                alpha = alphas[name] if arm == "crisp" else None
                tr_rows.append(
                    evaluate.transfer_recall(hold, budget_list)
                    .with_columns(pl.lit(name).alias("motif"), pl.lit(arm).alias("arm"),
                                  pl.lit(order).alias("order"),
                                  pl.lit(alpha, dtype=pl.Float64).alias("alpha"),
                                  pl.lit("transfer").alias("granularity")))
                inst_rows.append(
                    evaluate.instance_recall(hold, memb, inst, budget_list)
                    .with_columns(pl.lit(name).alias("motif"), pl.lit(arm).alias("arm"),
                                  pl.lit(order).alias("order"),
                                  pl.lit(alpha, dtype=pl.Float64).alias("alpha")))
    pl.concat(tr_rows).write_csv(out / "rq2_recall_transfer.csv")
    pl.concat(inst_rows).write_csv(out / "rq2_recall_instance.csv")

    log.info("RQ2 table (holdout instance recall, completable set)")
    inst_all = pl.concat(inst_rows).filter(pl.col("set") == "completable")
    for row in inst_all.sort(["motif", "arm", "order", "budget", "group"]).iter_rows(named=True):
        log.info("  %-14s %-5s %-11s budget %5d %-8s recall %.3f (%d/%d)",
                 row["motif"], row["arm"], row["order"], row["budget"], row["group"],
                 row["recall"] or 0.0, row["n_hit"], row["n_total"])
    log.info("done")


if __name__ == "__main__":
    typer.run(main)
