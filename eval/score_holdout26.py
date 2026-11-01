#!/usr/bin/env python3
"""Post-hoc holdout diagnostic for issue #26 (NOT a gate pass).

Scores explicitly named train-selected motifs on holdout members +
a holdout background sample, with union gain vs the current motifs.
Selection happened on train only (combined-Pareto winners); this
script runs once per motif and tunes nothing — pure generalization
evidence for the n=4-overfit question. The agent never sees holdout.

Usage:
    uv run eval/score_holdout26.py --motif-file eval/issue26/winner.motif
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import eval.rq1 as rq1  # noqa: E402
from eval import splits  # noqa: E402
from eval.discover import BG_N, MOTIFS, _batch_factory  # noqa: E402
from motifql.degrees import libs_from_doc  # noqa: E402
from motifql.graph import TemporalGraph  # noqa: E402
from motifql.models import discovery as D  # noqa: E402
from motifql.parser import parse_file  # noqa: E402
from motifql.planner import compile_motif  # noqa: E402

log = logging.getLogger("score_holdout26")


def main(
    motif_file: Annotated[Path, typer.Option(help="Motif text file to probe.")],
    data: Annotated[Path, typer.Option(help="Prepped dataset directory.")] = Path("data/hi-small"),
    pattern: Annotated[str, typer.Option(help="Holdout pattern to score.")] = "FAN-OUT",
    workers: Annotated[int, typer.Option(help="Worker processes (1 = serial).")] = 16,
    out: Annotated[Path, typer.Option(help="Output CSV path.")] = Path("eval/issue26/holdout_probe.csv"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    text = motif_file.read_text()
    compiled, errors = D.repair.try_compile(text)
    if compiled is None:
        raise SystemExit(f"does not compile: {errors}")

    df = pl.read_parquet(data / "transfers.parquet")
    memb = pl.read_parquet(data / "pattern_members.parquet")
    inst = splits.assign_instance_splits(
        pl.read_parquet(data / "pattern_instances.parquet"), memb, df)
    graph = TemporalGraph.from_transfers(df)
    tids = df.get_column("transfer_id").to_numpy()
    pos_of = {int(t): i for i, t in enumerate(tids.tolist())}
    rq1._GRAPH = graph

    ho_inst = inst.filter((pl.col("split") == "holdout")
                          & (pl.col("pattern_type") == pattern))
    ho_members: dict[int, list[int]] = {}
    hm = memb.join(ho_inst.select("instance_id"), on="instance_id", how="inner")
    for iid, tid in zip(hm["instance_id"].to_list(), hm["transfer_id"].to_list()):
        ho_members.setdefault(int(iid), []).append(pos_of[int(tid)])
    ho_bg = (df.filter(pl.col("day").is_between(splits.HOLDOUT_START, splits.HOLDOUT_END))
             .filter(pl.col("is_laundering") == 0)
             .select("transfer_id").to_series().to_list())
    ho_bg = sorted(np.random.default_rng(12).choice(
        ho_bg, size=min(BG_N, len(ho_bg)), replace=False).tolist())
    ho_bg_pos = [pos_of[t] for t in ho_bg]
    log.info("holdout %s: %d instances, bg sample %d", pattern, len(ho_members), len(ho_bg_pos))

    doc = parse_file(Path("examples/amlworld.motif"))
    libs = libs_from_doc(doc)
    plans = {m: compile_motif(doc, m) for m in MOTIFS}
    batch = _batch_factory(workers, "probe")
    want = sorted({p for ps in ho_members.values() for p in ps} | set(ho_bg_pos))
    prop_tops = batch(want, compiled.plan, compiled.libs)
    emit = compiled.plan.emit
    caught = D.caught_instances(prop_tops, emit, ho_members, list(ho_members))
    fire = sum(prop_tops[p] >= emit for p in ho_bg_pos) / len(ho_bg_pos)

    base_caught_union: set[int] = set()
    base_fire_union: set[int] = set()
    rows = []
    for m in MOTIFS:
        tops = batch(want, plans[m], libs)
        c = D.caught_instances(tops, plans[m].emit, ho_members, list(ho_members))
        f = sum(tops[p] >= plans[m].emit for p in ho_bg_pos) / len(ho_bg_pos)
        base_caught_union.update(c)
        base_fire_union.update(p for p in ho_bg_pos if tops[p] >= plans[m].emit)
        rows.append({"motif": m, "caught": sorted(c), "n": len(ho_members),
                     "bg_fire": round(f, 4)})
        log.info("  %-18s holdout: %d/%d bg fire %.4f", m, len(c), len(ho_members), f)
    new = D.newly_caught(caught, base_caught_union)
    marg = (sum(1 for p in ho_bg_pos
                if prop_tops[p] >= emit and p not in base_fire_union) / len(ho_bg_pos))
    log.info("  PROPOSAL %s: %d/%d (new %s) bg fire %.4f (marginal %.4f)",
             motif_file.name, len(caught), len(ho_members), new, fire, marg)
    log.info("  gain vs union of current motifs: %+d", len(new))
    out.parent.mkdir(parents=True, exist_ok=True)
    pl.from_dicts([{"motif": "PROPOSAL", "caught": str(sorted(caught)), "n": len(ho_members),
                    "bg_fire": round(fire, 4), "new_caught": str(sorted(new)),
                    "marginal_bg_fire": round(marg, 4)}]
                  + [dict(r, caught=str(r["caught"]), new_caught=None,
                          marginal_bg_fire=None) for r in rows]
                  ).write_csv(out)
    log.info("wrote %s", out)


if __name__ == "__main__":
    typer.run(main)
