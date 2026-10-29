#!/usr/bin/env python3
"""Witness-overlap recall over saved RQ2 scores (#15 #2).

RQ2 recall counts any high score on a truth transfer, whatever the
witness shows — a truth leg-1 anchor can score through an unrelated
background group. This replays the top-budget truth hits per
(motif, arm) with witness capture and reports, per pattern and
budget, plain recall vs recall requiring >=2/3 witness/instance
overlap (the `replay_bench.py` rule). Targeted (top-budget hits only,
scored once and reused across budgets), not a full replay.

Usage:
    uv run eval/witness_overlap.py --scores eval/results/rq2_holdout_scores.parquet
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval import evaluate, splits  # noqa: E402
from motifql.degrees import libs_from_doc  # noqa: E402
from motifql.graph import TemporalGraph  # noqa: E402
from motifql.groups import score_any  # noqa: E402
from motifql.matcher import EXACT_FLOOR  # noqa: E402
from motifql.parser import parse_file  # noqa: E402
from motifql.planner import compile_motif  # noqa: E402

log = logging.getLogger("witness_overlap")


def main(
    data: Annotated[Path, typer.Option(help="Prepped dataset directory.")] = Path("data/hi-small"),
    scores: Annotated[Path, typer.Option(help="Score parquet (transfer_id + {motif}__{arm}).")] = Path(
        "eval/results/rq2_holdout_scores.parquet"),
    motifs: Annotated[str, typer.Option(help="Comma-separated motif names.")] = (
        "round_trip,layering_chain,scatter_gather"),
    arms: Annotated[str, typer.Option(help="Comma-separated: fuzzy,crisp.")] = "fuzzy,crisp",
    crisp_alpha: Annotated[str, typer.Option(help="Crisp α-cut: `emit`, `support` or float.")] = "emit",
    budgets: Annotated[str, typer.Option(help="Alerts/day budgets, comma-separated.")] = "50,100,500,1000,5000",
    min_overlap: Annotated[float, typer.Option(help="Witness/instance overlap to count.")] = 2 / 3,
    out: Annotated[Path, typer.Option(help="Directory for the recall CSVs.")] = Path("eval/results"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    motifs_wanted = [m.strip() for m in motifs.split(",") if m.strip()]
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    if any(a not in ("fuzzy", "crisp") for a in arm_list):
        raise SystemExit(f"--arms {arms!r}: use fuzzy,crisp")
    budget_list = sorted(int(b) for b in budgets.split(","))
    out.mkdir(parents=True, exist_ok=True)

    log.info("[1/3] loading graph, scores and labels")
    graph = TemporalGraph.from_parquet(data / "transfers.parquet")
    wide = pl.read_parquet(scores)
    df = pl.read_parquet(data / "transfers.parquet")
    memb = pl.read_parquet(data / "pattern_members.parquet")
    inst = splits.assign_instance_splits(
        pl.read_parquet(data / "pattern_instances.parquet"), memb, df)
    labels_one = (memb.join(inst.select(["instance_id", "pattern_type"]), on="instance_id")
                  .group_by("transfer_id").agg(pl.col("pattern_type").first()))
    base = (wide.join(df.select("transfer_id", "day"), on="transfer_id", how="left")
            .join(labels_one, on="transfer_id", how="left"))
    split_of = dict(inst.select("instance_id", "split").iter_rows())
    pattern_of = dict(inst.select("instance_id", "pattern_type").iter_rows())
    patterns = sorted(set(pattern_of.values()))
    members_of: dict[int, frozenset] = {
        iid: frozenset(tids) for iid, tids in
        memb.group_by("instance_id").agg("transfer_id").iter_rows()}
    iids_of: dict[int, list[int]] = {}
    for iid, tid in memb.select("instance_id", "transfer_id").iter_rows():
        iids_of.setdefault(tid, []).append(iid)
    pos_of = {int(t): i for i, t in enumerate(graph.transfer_id.tolist())}

    doc = parse_file(Path("examples/amlworld.motif"))
    libs = libs_from_doc(doc)
    plans = {m: compile_motif(doc, m) for m in motifs_wanted}
    alphas = {m: evaluate.resolve_crisp_alpha(crisp_alpha, plans[m].emit)
              for m in motifs_wanted}

    top_budget = budget_list[-1]
    tr_rows, inst_rows, hits_rows_all = [], [], []
    for m in motifs_wanted:
        plan = plans[m]
        for arm in arm_list:
            ranked = evaluate.rank_per_day(
                base.select("transfer_id", "day", "pattern_type",
                            pl.col(f"{m}__{arm}").alias("score")))
            hit_tids = (ranked.filter((pl.col("rank") <= top_budget)
                                      & pl.col("pattern_type").is_not_null())
                        .get_column("transfer_id").to_list())
            log.info("[2/3] %s %s: scoring %d top-%d truth hits with witnesses",
                     m, arm, len(hit_tids), top_budget)
            # One row per (hit transfer, instance it belongs to) with raw
            # overlap counts: any rule is recomputable from the hits file.
            # Rules: instance-fraction and witness-fraction (|∩|/|I| and
            # |∩|/|W| >= threshold) plus their union. Instance-fraction
            # alone is unreachable when the witness is shorter than the
            # instance (a 5-hop chain vs an 11-transfer RANDOM), and
            # witness-fraction alone forgives background-enriched group
            # witnesses — report all three, not one.
            rank_of = dict(ranked.select("transfer_id", "rank").iter_rows())
            pat_of_hit = dict(ranked.select("transfer_id", "pattern_type").iter_rows())
            day_of = dict(ranked.select("transfer_id", "day").iter_rows())
            hit_rows: list[dict] = []
            for i, tid in enumerate(hit_tids):
                if i and i % 200 == 0:
                    log.info("  ... %d/%d", i, len(hit_tids))
                res = score_any(
                    graph, pos_of[tid], plan, libs, prune=True,
                    crisp=(arm == "crisp"),
                    crisp_alpha=(alphas[m] if arm == "crisp" else 0.0),
                    floor=(EXACT_FLOOR if arm == "fuzzy" else None))
                wtids = frozenset(int(graph.transfer_id[e]) for e in res.witness)
                for iid in iids_of.get(tid, []):
                    members = members_of[iid]
                    hit_rows.append({
                        "motif": m, "arm": arm, "transfer_id": tid,
                        "rank": rank_of[tid], "instance_id": iid,
                        "pattern": pattern_of[iid],
                        "n_inter": len(wtids & members),
                        "n_wit": len(wtids), "n_inst": len(members)})
            for row in hit_rows:
                row["day"] = str(day_of[row["transfer_id"]])
            hits_rows_all.extend(hit_rows)

            def passes(row: dict, rule: str) -> bool:
                inter, wit, inst = row["n_inter"], row["n_wit"], row["n_inst"]
                inst_ok = inst > 0 and inter / inst >= min_overlap
                wit_ok = wit > 0 and inter / wit >= min_overlap
                if rule == "instfrac":
                    return inst_ok
                if rule == "witfrac":
                    return wit_ok
                return inst_ok or wit_ok

            by_tid: dict[int, list[dict]] = {}
            for row in hit_rows:
                by_tid.setdefault(row["transfer_id"], []).append(row)
            rule_tids = {rule: {t for t, rows in by_tid.items()
                                if any(passes(r, rule) for r in rows)}
                         for rule in ("instfrac", "witfrac", "either")}
            for budget in budget_list:
                hits = [t for t in hit_tids if rank_of[t] <= budget]
                ov = {rule: {t for t in hits if t in rule_tids[rule]}
                      for rule in rule_tids}
                alpha = alphas[m] if arm == "crisp" else None
                groups_tr = ([("ALL_FLAGGED", None)] +
                             [(p, p) for p in patterns])
                for group, want_pat in groups_tr:
                    denom = [t for t in rank_of
                             if pat_of_hit[t] is not None
                             and (want_pat is None or pat_of_hit[t] == want_pat)]
                    plain = [t for t in hits
                             if want_pat is None or pat_of_hit[t] == want_pat]
                    tr_rows.append({
                        "motif": m, "arm": arm, "alpha": alpha,
                        "budget": budget, "group": group,
                        "n_total": len(denom),
                        "n_hit_plain": len(plain),
                        "n_hit_instfrac": sum(1 for t in plain if t in ov["instfrac"]),
                        "n_hit_witfrac": sum(1 for t in plain if t in ov["witfrac"]),
                        "n_hit_either": sum(1 for t in plain if t in ov["either"]),
                        "granularity": "transfer"})
                for group, want_split in (("ALL", None), ("completable", "holdout"),
                                          ("partial", "holdout_partial")):
                    iids = [iid for iid, s in split_of.items()
                            if want_split is None or s == want_split]
                    for pat in [None, *patterns]:
                        sub = [iid for iid in iids
                               if pat is None or pattern_of[iid] == pat]
                        plain = sum(1 for iid in sub
                                    if members_of[iid] & set(hits))
                        inst_rows.append({
                            "motif": m, "arm": arm, "alpha": alpha,
                            "budget": budget,
                            "group": group if pat is None else pat,
                            "set": group.lower(),
                            "n_total": len(sub),
                            "n_hit_plain": plain,
                            **{f"n_hit_{rule}": sum(
                                1 for iid in sub if members_of[iid] & ov[rule])
                               for rule in ("instfrac", "witfrac", "either")},
                        })
    pl.DataFrame(tr_rows, schema_overrides={"alpha": pl.Float64}).write_csv(
        out / "witness_overlap_transfer.csv")
    pl.DataFrame(inst_rows, schema_overrides={"alpha": pl.Float64}).write_csv(
        out / "witness_overlap_instance.csv")
    pl.DataFrame(hits_rows_all).write_csv(out / "witness_overlap_hits.csv")
    log.info("[3/3] overlap recall (completable instances, min overlap %.2f)", min_overlap)
    for row in sorted(inst_rows, key=lambda r: (r["motif"], r["arm"], r["budget"])):
        if row["group"] != "completable":
            continue
        log.info("  %-14s %-5s budget %5d plain %d/%d inst %d wit %d either %d/%d",
                 row["motif"], row["arm"], row["budget"], row["n_hit_plain"],
                 row["n_total"], row["n_hit_instfrac"], row["n_hit_witfrac"],
                 row["n_hit_either"], row["n_total"])
    log.info("  wrote witness_overlap_{transfer,instance,hits}.csv")
    log.info("done")


if __name__ == "__main__":
    typer.run(main)
