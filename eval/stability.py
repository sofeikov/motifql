#!/usr/bin/env python3
"""Issue-25 rank-stability metrics: overlap@k, score drops, evasion cost.

Reads per-level holdout score parquets written by `eval/perturb.py --budget
--save-scores` (one file per ladder rung, written by `level_name`) and
compares every rung against the clean rung (level 0 must be (0, x1)):

* rank stability: day-wise top-k set overlap between clean and noisy
  rankings, macro-averaged over holdout days — per score column, per k.
* score-drop distribution: per-anchor noisy-minus-clean top_score,
  truth vs background (mean/p50/p90 of the drop and its absolute
  value), on transfer_id-inner-joined anchors.
* evasion cost: for truth anchors in the clean top-k, the first ladder
  rung that pushes them out of the top-k — a cumulative eviction curve
  per rung, plus per-anchor costs. Anchors that drift out of the
  holdout set under gap scaling are censored (counted, excluded).

The pure functions (`overlap_at_k`, `mean_overlap_at_k`,
`score_drop_stats`, `evasion_costs`) run on synthetic inputs in
`tests/test_stability.py`; `main` is the thin data-dir CLI.

Usage:
    uv run eval/perturb.py --data data/hi-small --budget \\
        --budget-levels 0:1,0.05:1,0.10:1,0.20:1 \\
        --save-scores eval/results/perturb_scores_amount --workers 16
    uv run eval/stability.py --data data/hi-small \\
        --scores eval/results/perturb_scores_amount \\
        --levels 0:1,0.05:1,0.10:1,0.20:1 --out eval/results/stability_amount
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
from eval import evaluate  # noqa: E402

log = logging.getLogger("stability")


def level_name(amount_pct: float, gap_factor: float) -> str:
    """Parquet filename for one ladder rung (writer and reader share it)."""
    return f"level_p{amount_pct:g}_g{gap_factor:g}.parquet"


def overlap_at_k(clean_ranked: list[int], noisy_ranked: list[int], k: int) -> float:
    """Top-k set overlap between two rank orders (transfer_id lists)."""
    if k <= 0:
        raise ValueError(f"k={k}: use a positive budget")
    return len(set(clean_ranked[:k]) & set(noisy_ranked[:k])) / k


def mean_overlap_at_k(clean_days: dict, noisy_days: dict, k: int
                      ) -> tuple[float, dict, int]:
    """Macro-averaged day-wise overlap@k over shared days.

    Returns (mean overlap, per-day overlaps, n_days). Days missing on
    either side are skipped — the count travels with the mean so a
    shrunken denominator is visible, not silent.
    """
    per_day = {d: overlap_at_k(clean_days[d], noisy_days[d], k)
               for d in clean_days if d in noisy_days}
    mean = float(np.mean(list(per_day.values()))) if per_day else float("nan")
    return mean, per_day, len(per_day)


def _quantiles(vals: np.ndarray) -> dict[str, float | None]:
    if len(vals) == 0:
        return {"n": 0, "mean": None, "p50": None, "p90": None}
    return {"n": len(vals), "mean": float(vals.mean()),
            "p50": float(np.quantile(vals, 0.5)),
            "p90": float(np.quantile(vals, 0.9))}


def score_drop_stats(clean: dict[int, float], noisy: dict[int, float],
                     truth: set[int]) -> dict:
    """Per-anchor score-drop distributions, truth vs background.

    Drops are noisy-minus-clean on the transfer_id intersection; the
    match counts expose drift between rungs. Each group reports
    quantiles of the signed drop and of its absolute value.
    """
    shared = [t for t in clean if t in noisy]
    out: dict = {"n_clean": len(clean), "n_noisy": len(noisy),
                 "n_matched": len(shared)}
    for group, tids in (("truth", [t for t in shared if t in truth]),
                        ("background", [t for t in shared if t not in truth])):
        d = np.array([noisy[t] - clean[t] for t in tids])
        out[group] = {"drop": _quantiles(d), "abs": _quantiles(np.abs(d))}
    return out


def evasion_costs(level_ranks: list[dict[int, int]], k: int
                  ) -> tuple[dict[int, tuple[int | None, str]], dict]:
    """First ladder rung that evicts each clean-top-k anchor from top-k.

    `level_ranks[0]` is the clean rung. Population: anchors ranked <= k
    there. Cost is the first rung j >= 1 with rank > k (`evicted`);
    anchors never evicted are `survived`; anchors missing from a rung
    (drifted out of holdout under gap scaling) are `censored` and
    excluded from the curve. Returns per-anchor (cost, status) plus a
    summary with the cumulative eviction curve per rung.
    """
    if not level_ranks:
        raise ValueError("evasion_costs needs at least the clean rung")
    pop = [t for t, r in level_ranks[0].items() if r <= k]
    per_anchor: dict[int, tuple[int | None, str]] = {}
    for t in pop:
        for j in range(1, len(level_ranks)):
            if t not in level_ranks[j]:
                per_anchor[t] = (None, "censored")
                break
            if level_ranks[j][t] > k:
                per_anchor[t] = (j, "evicted")
                break
        else:
            per_anchor[t] = (None, "survived")
    n_cens = sum(1 for _, s in per_anchor.values() if s == "censored")
    denom = len(pop) - n_cens
    curve = []
    for j in range(1, len(level_ranks)):
        n_cum = sum(1 for c, s in per_anchor.values()
                    if s == "evicted" and c is not None and c <= j)
        curve.append({"rung": j, "n_evicted_cum": n_cum,
                      "frac_evicted_cum": n_cum / denom if denom else None})
    summary = {"n_pop": len(pop), "n_censored": n_cens,
               "n_evicted": sum(1 for _, s in per_anchor.values() if s == "evicted"),
               "n_survived": sum(1 for _, s in per_anchor.values() if s == "survived"),
               "curve": curve}
    return per_anchor, summary


def _ranked_by_day(scored: pl.DataFrame) -> dict:
    ranked = evaluate.rank_per_day(scored)
    return {d: tids for d, tids in
            ranked.sort("day", "rank").group_by("day", maintain_order=True)
            .agg(pl.col("transfer_id")).iter_rows()}


def main(
    data: Annotated[Path, typer.Option(help="Prepped dataset directory.")] = Path("data/hi-small"),
    scores: Annotated[Path, typer.Option(help="Per-rung score parquets from perturb --save-scores.")] = Path("eval/results/perturb_scores"),
    levels: Annotated[str, typer.Option(help="Ladder rungs as pct:factor,... (first must be 0:1).")] = "0:1,0.05:1,0.10:1,0.20:1",
    keys: Annotated[str, typer.Option(help="Score columns (`motif__arm`); empty = all.")] = "",
    budgets: Annotated[str, typer.Option(help="Top-k budgets, comma-separated.")] = "50,100,500",
    out: Annotated[Path, typer.Option(help="Directory for the CSVs.")] = Path("eval/results/stability"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rungs = [tuple(float(x) for x in lv.split(":")) for lv in levels.split(",")]
    if rungs[0] != (0.0, 1.0):
        raise SystemExit(f"--levels {levels!r}: first rung must be clean (0:1)")
    ks = [int(b) for b in budgets.split(",")]
    wanted = [c.strip() for c in keys.split(",") if c.strip()]
    out.mkdir(parents=True, exist_ok=True)

    log.info("[1/4] loading rung scores + timeless labels")
    frames = {}
    for pct, gf in rungs:
        path = scores / level_name(pct, gf)
        if not path.exists():
            raise SystemExit(f"missing rung file {path} (perturb --save-scores {scores})")
        frames[(pct, gf)] = pl.read_parquet(path)
    cols = [c for c in frames[rungs[0]].columns if c not in ("transfer_id", "day")]
    if wanted:
        missing = [c for c in wanted if c not in cols]
        if missing:
            raise SystemExit(f"keys {missing} not in rung files (have {cols})")
        cols = wanted
    transfers = pl.read_parquet(data / "transfers.parquet").select(
        "transfer_id", "is_laundering")
    truth = set(transfers.filter(pl.col("is_laundering") == 1)
                .get_column("transfer_id").to_list())
    memb = pl.read_parquet(data / "pattern_members.parquet")
    inst = pl.read_parquet(data / "pattern_instances.parquet")
    pat_of = dict(memb.join(inst.select("instance_id", "pattern_type"),
                            on="instance_id", how="left")
                  .group_by("transfer_id").agg(pl.col("pattern_type").first()).iter_rows())

    log.info("[2/4] ranking every rung (%d rungs x %d keys)", len(rungs), len(cols))
    ranked_days: dict = {}
    score_of: dict = {}
    rank_of: dict = {}
    for rung in rungs:
        base = evaluate.with_tiebreak(frames[rung])
        for key in cols:
            ranked = evaluate.rank_per_day(
                base.with_columns(pl.Series("score", frames[rung][key])))
            ranked_days[(rung, key)] = {
                d: tids for d, tids in ranked.sort("day", "rank")
                .group_by("day", maintain_order=True)
                .agg(pl.col("transfer_id")).iter_rows()}
            score_of[(rung, key)] = dict(
                zip(ranked["transfer_id"].to_list(), ranked["score"].to_list()))
            rank_of[(rung, key)] = dict(
                zip(ranked["transfer_id"].to_list(), ranked["rank"].to_list()))

    log.info("[3/4] overlap, drops, eviction curves")
    clean = rungs[0]
    overlap_rows, drop_rows, curve_rows, anchor_rows = [], [], [], []
    for key in cols:
        for k in ks:
            lvl_ranks = [rank_of[(r, key)] for r in rungs]
            per_anchor, summary = evasion_costs(lvl_ranks, k)
            groups: dict[str, list[int]] = {"ALL": list(per_anchor)}
            for t in per_anchor:
                groups.setdefault(pat_of.get(t, "UNATTRIBUTED"), []).append(t)
            for gname, gtids in groups.items():
                sub = {t: per_anchor[t] for t in gtids}
                n_cens = sum(1 for _, s in sub.values() if s == "censored")
                denom = len(sub) - n_cens
                for c in summary["curve"]:
                    rung = rungs[c["rung"]]
                    n_cum = sum(1 for t in gtids if per_anchor[t][1] == "evicted"
                                and per_anchor[t][0] is not None
                                and per_anchor[t][0] <= c["rung"])
                    curve_rows.append({
                        "key": key, "k": k, "group": gname,
                        "rung": c["rung"], "amount_pct": rung[0],
                        "gap_factor": rung[1], "n_pop": len(sub),
                        "n_censored": n_cens,
                        "n_evicted_cum": n_cum,
                        "frac_evicted_cum": n_cum / denom if denom else None})
            for t, (cost, status) in per_anchor.items():
                anchor_rows.append({"key": key, "k": k, "transfer_id": t,
                                    "pattern": pat_of.get(t, "UNATTRIBUTED"),
                                    "cost_rung": cost, "status": status})
        for rung in rungs[1:]:
            for k in ks:
                mean, _, n_days = mean_overlap_at_k(
                    ranked_days[(clean, key)], ranked_days[(rung, key)], k)
                overlap_rows.append({"key": key, "k": k, "amount_pct": rung[0],
                                     "gap_factor": rung[1], "overlap": mean,
                                     "n_days": n_days})
            stats = score_drop_stats(score_of[(clean, key)],
                                     score_of[(rung, key)], truth)
            for group in ("truth", "background"):
                row = {"key": key, "amount_pct": rung[0], "gap_factor": rung[1],
                       "group": group, "n_clean": stats["n_clean"],
                       "n_noisy": stats["n_noisy"],
                       "n_matched": stats["n_matched"]}
                for dist in ("drop", "abs"):
                    for stat in ("n", "mean", "p50", "p90"):
                        row[f"{dist}_{stat}"] = stats[group][dist][stat]
                drop_rows.append(row)

    log.info("[4/4] writing CSVs")
    pl.from_dicts(overlap_rows).write_csv(out / "stability_overlap.csv")
    pl.from_dicts(drop_rows).write_csv(out / "stability_drops.csv")
    pl.from_dicts(curve_rows).write_csv(out / "evasion_cost_curve.csv")
    pl.from_dicts(anchor_rows).write_parquet(out / "evasion_cost_anchors.parquet")
    log.info("  wrote stability_{overlap,drops}.csv evasion_cost_{curve.csv,anchors.parquet}")
    log.info("done")


if __name__ == "__main__":
    typer.run(main)
