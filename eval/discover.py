#!/usr/bin/env python3
"""Discovery driver: miss tables, agent runs, holdout scoring (Nov 1).

Builds a train-only DiscoveryCtx (member anchors + fixed background
sample + current-motif scores via the crisp screen), prints per-pattern
miss tables, optionally runs the text-protocol agent on one pattern's
missed cluster, and scores any gate-passing proposal on holdout
members + a holdout background sample.

The agent never sees holdout: its ctx holds train positions only
(pinned by tests/test_discovery.py), and holdout scoring happens in
this harness after the run.

Usage:
    uv run eval/discover.py --data data/hi-small --workers 16        # miss tables
    MUSE_KEY=... uv run eval/discover.py --run-agent --pattern SCATTER-GATHER
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
from motifql.degrees import libs_from_doc  # noqa: E402
from motifql.graph import TemporalGraph  # noqa: E402
from motifql.models import discovery as D  # noqa: E402
from motifql.models import spark  # noqa: E402
from motifql.parser import parse_file  # noqa: E402
from motifql.planner import PathPlan, compile_motif  # noqa: E402

log = logging.getLogger("discover")

MOTIFS = ["round_trip", "layering_chain", "fan_out", "fan_in",
          "scatter_gather", "gather_scatter", "structuring_fan_in",
          "bipartite_butterfly"]
PATTERN_OF = {"round_trip": "CYCLE", "layering_chain": "RANDOM",
              "fan_out": "FAN-OUT", "fan_in": "FAN-IN",
              "scatter_gather": "SCATTER-GATHER",
              "gather_scatter": "GATHER-SCATTER",
              "bipartite_butterfly": "BIPARTITE"}
BG_N = 20000

CLUSTER_NOTES = {
    "SCATTER-GATHER": ("Scatter-gather misses: instances where money flows "
                       "source -> intermediaries -> destination but the current "
                       "motif (2+ intermediaries, balanced sums, 7d window) "
                       "does not emit on any member. Past probes: some misses "
                       "are 2-cycles (A->B->A), one is malformed; find what is "
                       "actually catchable."),
    "FAN-OUT": ("Fan-out misses: instances where one account pays several "
                "but the current motif (count distinct receivers graded by "
                "`many`, 4d window) does not emit. Past probes: truth groups "
                "are small (2-6 members); background hubs dominate counts."),
    "FAN-IN": ("Fan-in misses: instances where several accounts pay one but "
               "the current motif does not emit. Same small-group note as "
               "fan-out."),
    "CYCLE": ("Cycle misses: instances where money leaves and returns but "
              "round_trip (2-5 hops, 7d, patient gaps, 20% amounts) does not "
              "emit. One known miss is time-disordered (unmatchable); check "
              "whether the rest are length, gap, or amount misses."),
    "RANDOM": ("Random-chain misses: open chains the current motif (5-8 "
               "hops, 10d) does not emit on. Known: instances with <5 "
               "members cannot match; find catchable structure if any."),
    "GATHER-SCATTER": ("Gather-scatter misses: collect-then-redistribute "
                       "instances the current motif (2+ collectors and 2+ "
                       "payouts, count-balance) does not emit on."),
}


def _batch_factory(workers: int, label: str):
    def batch(want, plan, libs):
        want_arr = np.asarray(sorted(set(want)), dtype=np.int64)
        rq1._PLAN = plan
        rq1._LIBS = libs
        if isinstance(plan, PathPlan):
            rq1._PRUNE, rq1._CRISP = True, True
            cr, _, _, cr_tr = rq1._replay(want_arr, workers, f"{label} crisp")
            sub = np.flatnonzero((cr > 0) | cr_tr)
            tops = np.zeros(len(want_arr))
            if len(sub):
                rq1._PRUNE, rq1._CRISP = False, False
                s, _, _, _ = rq1._replay(want_arr[sub], workers, f"{label} fuzzy")
                tops[sub] = s
            return dict(zip(want_arr.tolist(), tops.tolist()))
        rq1._PRUNE, rq1._CRISP = False, False
        tops, _, _, _ = rq1._replay(want_arr, workers, f"{label} fuzzy")
        return dict(zip(want_arr.tolist(), tops.tolist()))
    return batch


def main(
    data: Annotated[Path, typer.Option(help="Prepped dataset directory.")] = Path("data/hi-small"),
    run_agent: Annotated[bool, typer.Option(help="Run the agent loop (needs MUSE_KEY).")] = False,
    pattern: Annotated[str, typer.Option(help="Target pattern for the agent run.")] = "SCATTER-GATHER",
    max_steps: Annotated[int, typer.Option(help="Agent step budget.")] = D.MAX_STEPS,
    bg_n: Annotated[int, typer.Option(help="Background sample size.")] = BG_N,
    workers: Annotated[int, typer.Option(help="Worker processes (1 = serial).")] = 16,
    out: Annotated[Path, typer.Option(help="Directory for outputs.")] = Path("eval/results"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out.mkdir(parents=True, exist_ok=True)

    log.info("[1/4] loading transfers, ground truth, splits")
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
    bg_tids = sorted(np.random.default_rng(11).choice(
        train_ids, size=min(bg_n, len(train_ids)), replace=False).tolist())
    bg_pos = [pos_of[t] for t in bg_tids]

    train_inst = inst.filter(pl.col("split") == "train")
    member_pos: dict[int, list[int]] = {}
    member_tid: dict[int, list[int]] = {}
    pat_of: dict[int, str] = {}
    for row in train_inst.iter_rows(named=True):
        iid = int(row["instance_id"])
        pat_of[iid] = row["pattern_type"]
    mm = memb.join(train_inst.select("instance_id"), on="instance_id", how="inner")
    for iid, tid in zip(mm["instance_id"].to_list(), mm["transfer_id"].to_list()):
        iid, tid = int(iid), int(tid)
        member_pos.setdefault(iid, []).append(pos_of[tid])
        member_tid.setdefault(iid, []).append(tid)
    log.info("  train instances %d (%d with members), bg sample %d",
             len(pat_of), len(member_pos), len(bg_pos))

    log.info("[2/4] compiling %d motifs and scoring member+bg anchors", len(MOTIFS))
    doc = parse_file(Path("examples/amlworld.motif"))
    libs = libs_from_doc(doc)
    plans = {m: compile_motif(doc, m) for m in MOTIFS}  # no stats: E_COST bypass (see rq2)
    batch = _batch_factory(workers, "base")
    want = sorted({p for ps in member_pos.values() for p in ps} | set(bg_pos))
    log.info("  scoring %d anchors per motif", len(want))
    ctx = D.DiscoveryCtx(graph=graph, transfer_ids=tids, pos_of=pos_of,
                         member_pos=member_pos, member_tid=member_tid, bg_pos=bg_pos)
    for m in MOTIFS:
        tops = batch(want, plans[m], libs)
        ctx.base_scores[m] = tops
        ctx.base_emit[m] = plans[m].emit
        ctx.base_bg_fire[m] = sum(tops[p] >= plans[m].emit for p in bg_pos) / len(bg_pos)
        log.info("  %-18s emit %.2f bg fire %.4f", m, plans[m].emit, ctx.base_bg_fire[m])
    pl.DataFrame({"transfer_id": [int(tids[p]) for p in want],
                  **{m: [ctx.base_scores[m][p] for p in want] for m in MOTIFS}}
                ).write_parquet(out / "discover_base_scores.parquet")

    log.info("[3/4] miss tables per pattern (train)")
    miss_rows = []
    for pat in sorted(set(pat_of.values())):
        iids = [i for i, p in pat_of.items() if p == pat and i in member_pos]
        sub = D.DiscoveryCtx(graph=None, transfer_ids=tids, pos_of=pos_of,
                             member_pos={i: member_pos[i] for i in iids},
                             member_tid={i: member_tid[i] for i in iids},
                             bg_pos=bg_pos, base_scores=ctx.base_scores,
                             base_bg_fire=ctx.base_bg_fire, base_emit=ctx.base_emit)
        missed = D.list_missed(sub, MOTIFS)
        for iid in missed:
            closest = max(ctx.base_scores[m].get(p, 0.0)
                          for m in MOTIFS for p in member_pos[iid])
            miss_rows.append({"pattern": pat, "instance_id": iid,
                              "n_members": len(member_pos[iid]),
                              "closest_top": round(closest, 3)})
        log.info("  %-14s %d/%d missed: %s", pat, len(missed), len(iids), missed)
    pl.from_dicts(miss_rows).write_csv(out / "discover_misses.csv")

    if not run_agent:
        log.info("done (miss tables only; pass --run-agent to loop)")
        return

    log.info("[4/4] agent run on %s misses", pattern)
    cluster = [r["instance_id"] for r in miss_rows if r["pattern"] == pattern]
    if not cluster:
        log.info("  no missed cluster; nothing to run")
        return
    note = CLUSTER_NOTES.get(pattern, f"{pattern} misses.")
    desc = (f"{note} Cluster instance ids: {cluster}. "
            f"Current motifs' train bg fire rates: "
            + ", ".join(f"{m}={ctx.base_bg_fire[m]:.4f}" for m in MOTIFS) + ".")
    med = float(np.median([ctx.base_bg_fire[m] for m in MOTIFS]))
    allowed = max(med * D.GATE_BG_MULT, 2 / len(ctx.bg_pos))
    need = min(D.GATE_MIN_CATCHES, len(cluster))
    gate_desc = (f"PROPOSE only a motif catching at least {need} of the {len(cluster)} "
                 f"cluster instances with background fire at or below {allowed:.4f} "
                 f"({allowed * len(ctx.bg_pos):.0f}/{len(ctx.bg_pos)} on the sample; "
                 f"base median {med:.4f}). If no TRY passes both bars, reply DONE "
                 f"instead of proposing — a failing PROPOSE wastes the run.")
    try:
        res = D.run_agent(desc, cluster, ctx, MOTIFS, df, batch,
                          lambda messages: spark.complete(messages),
                          max_steps=max_steps, gate_desc=gate_desc)
    except spark.SparkError as exc:
        raise SystemExit(f"model call failed: {exc}") from exc
    slug = pattern.lower().replace("-", "_")
    (out / f"discover_{slug}_transcript.md").write_text(
        "# Discovery transcript: " + pattern + "\n\n" + "\n\n".join(res["transcript"])
        + "\n\n## Proposal\n\n"
        + (res["proposal"] or "(none)")
        + "\n\n## Gate\n\n" + (res["gate"].detail if res["gate"] else "(no pass)"))
    log.info("  transcript -> discover_%s_transcript.md", slug)
    if res["proposal"] is None:
        log.info("  no gate-passing proposal")
        return
    log.info("  GATE PASS: %s", res["gate"].detail)

    log.info("holdout scoring (harness, not agent)")
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
        ho_bg, size=min(bg_n, len(ho_bg)), replace=False).tolist())
    ho_bg_pos = [pos_of[t] for t in ho_bg]
    compiled, _ = D.repair.try_compile(res["proposal"])
    prop_tops = batch(sorted({p for ps in ho_members.values() for p in ps} | set(ho_bg_pos)),
                      compiled.plan, compiled.libs)
    emit = compiled.plan.emit
    caught = [i for i, ps in ho_members.items()
              if max(prop_tops[p] for p in ps) >= emit]
    fire = sum(prop_tops[p] >= emit for p in ho_bg_pos) / len(ho_bg_pos)
    log.info("  proposal holdout: %d/%d instances, bg fire %.4f",
             len(caught), len(ho_members), fire)
    base_rows = []
    for m in MOTIFS:
        tops = batch(sorted({p for ps in ho_members.values() for p in ps} | set(ho_bg_pos)),
                     plans[m], libs)
        c = [i for i, ps in ho_members.items()
             if max(tops[p] for p in ps) >= plans[m].emit]
        f = sum(tops[p] >= plans[m].emit for p in ho_bg_pos) / len(ho_bg_pos)
        base_rows.append({"motif": m, "caught": len(c), "n": len(ho_members),
                          "bg_fire": round(f, 4)})
        log.info("  %-18s holdout: %d/%d bg fire %.4f", m, len(c), len(ho_members), f)
    pl.from_dicts([{"motif": "PROPOSAL", "caught": len(caught), "n": len(ho_members),
                    "bg_fire": round(fire, 4)}] + base_rows
                  ).write_csv(out / f"discover_{slug}_holdout.csv")
    log.info("done")


if __name__ == "__main__":
    typer.run(main)
