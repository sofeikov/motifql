#!/usr/bin/env python3
"""Issue #24 harness: learned fuzzy calibration (fit on train, report once).

Black-box arm: random search over one motif's membership-function
parameters (see `src/motifql/calibrate.py` for the space), scored by
train-only transfer hits at operational alert budgets via exact-floor
replay (`score_batch` branch-and-bound, the RQ2 fuzzy arm). Motif
structure is frozen — only numeric params move.

Two modes (the protocol in `docs/2026-11-08-issue24-prereg.md`):

* `fit`: sample `tries` vectors (seeded), replay each on the train eval
  set (all target-pattern member anchors + a seeded background sample),
  record the full trace, then confirm the winner and the hand-tuned
  baseline on ALL train anchors with unscaled budgets.
* `report`: replay frozen params (`--params fitted.json`) once on a
  sealed split (holdout or LI-Small) — no search, no peeking.

Usage:
    uv run eval/calibrate.py --mode fit --data data/hi-small --workers 16
    uv run eval/calibrate.py --mode report --data data/hi-small \\
        --params eval/results/calibrate_round_trip/fitted.json
    uv run eval/calibrate.py --mode report --data data/li-small \\
        --params eval/results/calibrate_round_trip/fitted.json --split holdout
"""

from __future__ import annotations

import json
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
from motifql import calibrate as cal  # noqa: E402
from motifql.degrees import libs_from_doc  # noqa: E402
from motifql.graph import TemporalGraph  # noqa: E402
from motifql.matcher import EXACT_FLOOR  # noqa: E402
from motifql.parser import parse_file  # noqa: E402
from motifql.planner import compile_motif  # noqa: E402

log = logging.getLogger("calibrate")

MOTIF_FILE = Path(__file__).resolve().parent.parent / "examples/amlworld.motif"
BG_SEED_TAG = 0
PARAM_SEED_TAG = 1


def build_eval_frame(df: pl.DataFrame, members: pl.DataFrame,
                     inst: pl.DataFrame, anchor_ids: np.ndarray) -> pl.DataFrame:
    """Label frame for replayed anchors, in replay (time) order.

    `anchor_ids` are transfer_ids in the same order as the score array.
    """
    labels_one = (members.join(inst.select(["instance_id", "pattern_type"]),
                               on="instance_id")
                  .group_by("transfer_id").agg(pl.col("pattern_type").first()))
    order = pl.DataFrame({"transfer_id": anchor_ids.tolist()})
    return (order.join(df.select("transfer_id", "day", "is_laundering"),
                       on="transfer_id", how="left", maintain_order="left")
            .join(labels_one, on="transfer_id", how="left", maintain_order="left"))


def sample_eval_positions(train_pos: np.ndarray, transfer_ids: np.ndarray,
                          truth_ids: set[int], bg_n: int,
                          seed: int) -> np.ndarray:
    """All truth anchors plus a seeded background sample, in time order.

    Every target-pattern member anchor is always in (the objective needs
    them all); background anchors are a seeded uniform sample. Separated
    RNG stream from parameter sampling so the eval set does not shift
    when the search budget changes.
    """
    rng = np.random.default_rng([seed, BG_SEED_TAG])
    is_truth = np.array([t in truth_ids for t in transfer_ids[train_pos]])
    truth_pos = train_pos[is_truth]
    bg_pos = train_pos[~is_truth]
    if len(bg_pos) > bg_n:
        bg_pos = rng.choice(bg_pos, size=bg_n, replace=False)
    return np.sort(np.concatenate([truth_pos, bg_pos]))


def sampled_objective(frame: pl.DataFrame, target: str, budgets: list[int],
                      full_per_day: dict) -> dict:
    """Target transfer hits on the sampled eval set at scaled budgets.

    `frame` carries score + labels for the eval anchors; `full_per_day`
    maps each day to its FULL anchor count. Each budget is scaled by the
    day's sample rate (k_d = budget * n_eval_d / n_full_d), so the
    objective approximates full-train recall@budget. Returns hits per
    budget plus the summed objective.
    """
    ranked = evaluate.rank_per_day(frame)
    n_eval = ranked.group_by("day").len().sort("day")
    rate = {r["day"]: len(ranked.filter(pl.col("day") == r["day"]))
            / full_per_day[r["day"]] for r in n_eval.iter_rows(named=True)}
    hits: dict[int, int] = {}
    for b in budgets:
        n_hit = 0
        for day, r in rate.items():
            k = min(int(round(b * r)), int((ranked["day"] == day).sum()))
            k = max(k, 1)
            top = ranked.filter((pl.col("day") == day) & (pl.col("rank") <= k))
            n_hit += top.filter(pl.col("pattern_type") == target).height
        hits[b] = n_hit
    return {"hits": hits, "objective": float(sum(hits.values()))}


def pick_winner(rows: list[dict]) -> dict:
    """Best trace row: highest objective, ties toward wider (looser) fits."""
    return max(rows, key=lambda r: (r["objective"], r["width"]))


def _replay_params(graph: TemporalGraph, positions: np.ndarray, doc,
                   motif: str, params: dict, workers: int,
                   label: str) -> tuple[np.ndarray, np.ndarray]:
    """Exact-floor top_scores for one param vector (the RQ2 fuzzy arm)."""
    fitted = cal.apply_params(doc, motif, params)
    libs = libs_from_doc(fitted)
    rq1._GRAPH, rq1._PLAN, rq1._LIBS = graph, compile_motif(fitted, motif), libs
    rq1._PRUNE, rq1._CRISP, rq1._CALPHA, rq1._FLOOR = True, False, 0.0, EXACT_FLOOR
    top, found, _, _ = rq1._replay(positions, workers, label)
    return top, found


def run_fit(data: Path, motif: str, tries: int, seed: int, budgets: list[int],
            bg_n: int, workers: int, out: Path,
            max_anchors: int = 0) -> dict:
    """Train-only random search + full-train confirm. Returns the summary."""
    target = cal.TARGET_PATTERN[motif]
    out.mkdir(parents=True, exist_ok=True)
    (out / "scores").mkdir(exist_ok=True)

    log.info("[1/5] loading transfers and building graph (queries look back)")
    df = pl.read_parquet(data / "transfers.parquet")
    graph = TemporalGraph.from_transfers(df)
    days = df.get_column("day").to_numpy()
    tids = df.get_column("transfer_id").to_numpy()
    train_pos = np.flatnonzero(
        (days >= np.datetime64(str(splits.TRAIN_START)))
        & (days <= np.datetime64(str(splits.TRAIN_END))))
    if max_anchors > 0:
        train_pos = train_pos[:max_anchors]
    n_tr_days = (splits.TRAIN_END - splits.TRAIN_START).days + 1
    log.info("  graph edges: %d, train anchors: %d", graph.n_edges, len(train_pos))

    memb = pl.read_parquet(data / "pattern_members.parquet")
    inst = splits.assign_instance_splits(
        pl.read_parquet(data / "pattern_instances.parquet"), memb, df)
    train_truth = (memb.join(inst.filter(pl.col("split") == "train")
                             .select("instance_id", "pattern_type"),
                             on="instance_id")
                   .filter(pl.col("pattern_type") == target)
                   .get_column("transfer_id").unique().to_list())
    truth_ids = set(train_truth)
    log.info("  train %s member anchors: %d", target, len(truth_ids))

    base_doc = parse_file(MOTIF_FILE)
    run_meta = {"motif": motif, "target": target, "seed": seed, "tries": tries,
                "budgets": budgets, "bg_n": bg_n,
                "space_version": cal.SPACE_VERSION,
                "train_days": [str(splits.TRAIN_START), str(splits.TRAIN_END)]}
    (out / "run.json").write_text(json.dumps(run_meta, indent=2))

    # Resume: trace rows + cached scores keyed by try index. A run.json
    # mismatch means the space or protocol moved — start over, loudly.
    param_rng = np.random.default_rng([seed, PARAM_SEED_TAG])
    planned = [cal.sample_params(motif, param_rng) for _ in range(tries)]
    trace_path = out / "trace.csv"
    cached: dict[int, dict] = {}
    if trace_path.exists():
        old = pl.read_csv(trace_path)
        for row in old.iter_rows(named=True):
            i = int(row["try"])
            sp = out / "scores" / f"try_{i:03d}.parquet"
            if row["is_baseline"] or (0 <= i < tries and sp.exists()):
                cached[i if not row["is_baseline"] else -1] = row
        log.info("  resume: %d/%d tries cached", len([k for k in cached if k >= 0]),
                 tries)

    log.info("[2/5] sampling train eval set (all truth + %d background)", bg_n)
    eval_pos = sample_eval_positions(train_pos, tids, truth_ids, bg_n, seed)
    eval_ids = tids[eval_pos]
    full_per_day = (df.filter(pl.col("day").is_between(splits.TRAIN_START, splits.TRAIN_END))
                    .group_by("day").len().sort("day"))
    full_counts = {r["day"]: r["len"] for r in full_per_day.iter_rows(named=True)}
    if max_anchors > 0:  # smoke runs: scale against the capped frame
        capped = df.filter(pl.col("day").is_between(splits.TRAIN_START, splits.TRAIN_END)) \
            .head(len(train_pos)).group_by("day").len().sort("day")
        full_counts = {r["day"]: r["len"] for r in capped.iter_rows(named=True)}
    eval_base = build_eval_frame(df, memb, inst, eval_ids)
    log.info("  eval anchors: %d (%d truth)", len(eval_pos),
             eval_base.filter(pl.col("pattern_type") == target).height)

    def _score_eval(params: dict, try_idx: int, label: str
                    ) -> tuple[np.ndarray, dict]:
        sp = out / "scores" / f"try_{try_idx:03d}.parquet"
        if try_idx in cached and sp.exists():
            top = pl.read_parquet(sp).get_column("top_score").to_numpy()
        else:
            top, _ = _replay_params(graph, eval_pos, base_doc, motif, params,
                                    workers, label)
            pl.DataFrame({"transfer_id": eval_ids, "top_score": top}).write_parquet(sp)
        frame = eval_base.with_columns(pl.Series("score", top))
        obj = sampled_objective(frame, target, budgets, full_counts)
        return top, obj

    log.info("[3/5] replaying baseline + %d tries (exact floor)", tries)
    rows: list[dict] = []
    base_params = cal.default_params(motif)
    base_top, base_obj = _score_eval(base_params, -1, f"{motif} baseline")
    base_fires = int((base_top >= float(base_params["emit"])).sum())
    rows.append({"try": -1, "is_baseline": True, **base_params,
                 "objective": base_obj["objective"],
                 **{f"hits_b{b}": base_obj["hits"][b] for b in budgets},
                 "fires_eval": base_fires, "width": cal.width_score(motif, base_params)})
    log.info("  baseline: obj=%.0f %s fires_eval=%d", base_obj["objective"],
             " ".join(f"b{b}={base_obj['hits'][b]}" for b in budgets), base_fires)
    for i, params in enumerate(planned):
        if i in cached and (out / "scores" / f"try_{i:03d}.parquet").exists():
            row = cached[i]
            rows.append({"try": i, "is_baseline": False,
                         **{k: params[k] for k in base_params},
                         "objective": float(row["objective"]),
                         **{f"hits_b{b}": int(row[f"hits_b{b}"]) for b in budgets},
                         "fires_eval": int(row["fires_eval"]),
                         "width": cal.width_score(motif, params)})
            continue
        top, obj = _score_eval(params, i, f"{motif} try {i + 1}/{tries}")
        fires = int((top >= float(params["emit"])).sum())
        rows.append({"try": i, "is_baseline": False, **params,
                     "objective": obj["objective"],
                     **{f"hits_b{b}": obj["hits"][b] for b in budgets},
                     "fires_eval": fires,
                     "width": cal.width_score(motif, params)})
        if (i + 1) % 5 == 0 or i + 1 == tries:
            pl.from_dicts(rows).write_csv(trace_path)
        log.info("  try %d: obj=%.0f %s fires_eval=%d", i, obj["objective"],
                 " ".join(f"b{b}={obj['hits'][b]}" for b in budgets), fires)
    pl.from_dicts(rows).write_csv(trace_path)

    winner = pick_winner([r for r in rows if not r["is_baseline"]])
    w_params = {k: winner[k] for k in base_params}
    log.info("[4/5] winner try %d: obj=%.0f (baseline %.0f)", winner["try"],
             winner["objective"], rows[0]["objective"])
    (out / "fitted.json").write_text(json.dumps(
        {"motif": motif, "params": w_params, "try": winner["try"],
         "train_objective_sampled": winner["objective"],
         "baseline_objective_sampled": rows[0]["objective"],
         "seed": seed, "space_version": cal.SPACE_VERSION}, indent=2))
    (out / f"fitted_{motif}.motif").write_text(
        cal.render_fitted_file(motif, w_params,
                               meta=f"seed {seed} try {winner['try']}"))

    log.info("[5/5] confirming winner + baseline on ALL train anchors")
    train_ids = tids[train_pos]
    train_base = evaluate.with_tiebreak(build_eval_frame(df, memb, inst, train_ids))
    confirm_rows, summary = [], []
    for key, params in (("hand_tuned", base_params), ("fitted", w_params)):
        top, _ = _replay_params(graph, train_pos, base_doc, motif, params,
                                workers, f"{motif} confirm {key}")
        hold = evaluate.rank_per_day(train_base.with_columns(pl.Series("score", top)))
        confirm_rows.append(
            evaluate.transfer_recall(hold, budgets)
            .with_columns(pl.lit(key).alias("key"),
                          pl.lit("transfer").alias("granularity")))
        # Emit sweep without replays: exact top_scores make found(top>=t)
        # exact too. Fires/day is a rate over the full train split.
        sweep = {t: float((top >= t).sum()) / n_tr_days
                 for t in (0.3, 0.4, 0.5, 0.6, 0.7)}
        summary.append({"key": key, "anchors": len(top),
                        "fires_per_day_at_emit":
                            float((top >= float(params["emit"])).sum()) / n_tr_days,
                        **{f"fires_per_day_at_{t:g}": v for t, v in sweep.items()}})
        log.info("  %s: fires/day at emit %.2f = %.1f", key, float(params["emit"]),
                 summary[-1]["fires_per_day_at_emit"])
    pl.concat(confirm_rows).write_csv(out / "confirm_train_transfer.csv")
    pl.from_dicts(summary).write_csv(out / "confirm_train_summary.csv")
    log.info("  wrote %s", out)
    return {"winner_try": winner["try"], "params": w_params,
            "baseline_obj": rows[0]["objective"],
            "winner_obj": winner["objective"]}


def run_report(data: Path, params_path: Path, split: str, budgets: list[int],
               workers: int, out: Path) -> dict:
    """Replay frozen params once on a sealed split (no search)."""
    fit = json.loads(params_path.read_text())
    motif, params = fit["motif"], fit["params"]
    cal.validate_params(motif, params)
    target = cal.TARGET_PATTERN[motif]
    out.mkdir(parents=True, exist_ok=True)
    if split == "holdout":
        s0, s1 = splits.HOLDOUT_START, splits.HOLDOUT_END
    elif split == "train":
        s0, s1 = splits.TRAIN_START, splits.TRAIN_END
    else:
        raise ValueError(f"split {split!r}: use train or holdout")

    log.info("[1/3] loading %s and building graph", data)
    df = pl.read_parquet(data / "transfers.parquet")
    graph = TemporalGraph.from_transfers(df)
    days = df.get_column("day").to_numpy()
    pos = np.flatnonzero((days >= np.datetime64(str(s0)))
                         & (days <= np.datetime64(str(s1))))
    n_days = (s1 - s0).days + 1
    log.info("  graph edges: %d, %s anchors: %d", graph.n_edges, split, len(pos))

    log.info("[2/3] replaying fitted + hand-tuned (exact floor, once)")
    base_doc = parse_file(MOTIF_FILE)
    memb = pl.read_parquet(data / "pattern_members.parquet")
    inst = splits.assign_instance_splits(
        pl.read_parquet(data / "pattern_instances.parquet"), memb, df)
    base = evaluate.with_tiebreak(
        build_eval_frame(df, memb, inst, df.get_column("transfer_id").to_numpy()[pos]))
    tr_rows, inst_rows, summary = [], [], []
    for key, p in (("hand_tuned", cal.default_params(motif)), ("fitted", params)):
        top, _ = _replay_params(graph, pos, base_doc, motif, p, workers,
                                f"{motif} {split} {key}")
        hold = evaluate.rank_per_day(base.with_columns(pl.Series("score", top)))
        tr_rows.append(evaluate.transfer_recall(hold, budgets)
                       .with_columns(pl.lit(key).alias("key"),
                                     pl.lit("transfer").alias("granularity")))
        inst_rows.append(evaluate.instance_recall(hold, memb, inst, budgets)
                         .with_columns(pl.lit(key).alias("key")))
        summary.append({"key": key, "split": split, "anchors": len(top),
                        "fires": int((top >= float(p["emit"])).sum()),
                        "fires_per_day":
                            float((top >= float(p["emit"])).sum()) / n_days})
        tgt = next(r for r in tr_rows[-1].iter_rows(named=True)
                   if r["group"] == target and r["budget"] == budgets[0])
        log.info("  %s: %s hits@%d=%d fires/day=%.1f", key, target, budgets[0],
                 tgt["n_hit"], summary[-1]["fires_per_day"])
    pl.concat(tr_rows).write_csv(out / f"report_{split}_transfer.csv")
    pl.concat(inst_rows).write_csv(out / f"report_{split}_instance.csv")
    pl.from_dicts(summary).write_csv(out / f"report_{split}_summary.csv")

    log.info("[3/3] %s transfer verdict (%s)", split, target)
    verdict = (pl.concat(tr_rows).filter(pl.col("group") == target)
               .sort(["key", "budget"]))
    for row in verdict.iter_rows(named=True):
        log.info("  %-10s budget %5d hits %d/%d", row["key"], row["budget"],
                 row["n_hit"], row["n_total"])
    log.info("done")
    return {"motif": motif, "split": split, "out": str(out)}


def main(
    mode: Annotated[str, typer.Option(help="fit (train search) or report (frozen replay).")] = "fit",
    data: Annotated[Path, typer.Option(help="Prepped dataset directory.")] = Path("data/hi-small"),
    motif: Annotated[str, typer.Option(help="Motif to fit (fit mode).")] = "round_trip",
    tries: Annotated[int, typer.Option(help="Random-search vectors (fit mode).")] = 30,
    seed: Annotated[int, typer.Option(help="Seed for sampling + search.")] = 24,
    budgets: Annotated[str, typer.Option(help="Alerts/day budgets, comma-separated.")] = "50,100",
    bg_n: Annotated[int, typer.Option(help="Background sample in the train eval set.")] = 20000,
    workers: Annotated[int, typer.Option(help="Worker processes (1 = serial).")] = 16,
    out: Annotated[Path, typer.Option(help="Output directory.")] = Path("eval/results/calibrate_round_trip"),
    params: Annotated[Path | None, typer.Option(help="fitted.json (report mode).")] = None,
    split: Annotated[str, typer.Option(help="Split to report (report mode).")] = "holdout",
    max_anchors: Annotated[int, typer.Option(help="Cap train anchors (0 = all).")] = 0,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    budget_list = [int(b) for b in budgets.split(",")]
    if mode == "fit":
        if motif not in cal.DEFAULTS:
            raise SystemExit(f"--motif {motif!r}: choose from {sorted(cal.DEFAULTS)}")
        run_fit(data, motif, tries, seed, budget_list, bg_n, workers, out,
                max_anchors)
    elif mode == "report":
        if params is None:
            raise SystemExit("--params fitted.json is required in report mode")
        run_report(data, params, split, budget_list, workers, out)
    else:
        raise SystemExit(f"--mode {mode!r}: use fit or report")


if __name__ == "__main__":
    typer.run(main)
