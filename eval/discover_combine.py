#!/usr/bin/env python3
"""Combine issue #26 arms: verify arm-(a) TRYs, merge Pareto frontiers.

1. Parses arm-(a) TRY motifs from the transcript with the pinned
   D.parse_action, re-scores each on the full 20k bg (independent
   check of the transcript numbers), and writes arm_a_try_exact.csv.
2. Merges per-arm exact rows into frontier_all.csv with a combined
   Pareto flag, plus pareto.svg (recall vs bg fires/day, log-x).
3. Prints the gate analysis: per-arm best by recall level and
   #TRYs to first gate pass (grid/transcript order).

Usage:
    uv run eval/discover_combine.py --data /path/to/hi-small --workers 16
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from eval import discover_grid as G  # noqa: E402
from motifql.models import discovery as D  # noqa: E402

log = logging.getLogger("discover_combine")

CLUSTER = [123, 138, 223, 224]
ROOT = Path("eval/issue26")
_TOOL_RECALL = re.compile(r"cluster recall (\d+)/\d+")


def transcript_trys(transcript: str) -> list[dict]:
    """TRY motifs in transcript order via the pinned action parser."""
    sections = re.split(r"--- step (\d+) model ---\n", transcript)
    out = []
    for i in range(1, len(sections), 2):
        step, body = int(sections[i]), sections[i + 1]
        model_text = body.split("--- step", 1)[0]
        kind, motif_text, _ = D.parse_action(model_text)
        if kind == "TRY" and motif_text:
            out.append({"step": step, "text": motif_text})
    return out


def transcript_tool_numbers(transcript: str) -> list[tuple[int, int]]:
    """(recall, bg_hits) per COMPILED tool reply, in order."""
    recalls = [int(m.group(1)) for m in _TOOL_RECALL.finditer(transcript)]
    # Each COMPILED reply prints one recall line plus absolute and
    # marginal bg lines; anchor at line start to skip the marginal one.
    abs_bgs = [int(m.group(1)) for m in
               re.finditer(r"\nbackground fire (\d+)/\d+", transcript)]
    assert len(recalls) == len(abs_bgs), (len(recalls), len(abs_bgs))
    return list(zip(recalls, abs_bgs))


def main(
    data: Annotated[Path, typer.Option(help="Prepped dataset directory.")] = Path("data/hi-small"),
    workers: Annotated[int, typer.Option(help="Worker processes (1 = serial).")] = 16,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    transcript = (ROOT / "arm_a" / "discover_fan_out_transcript.md").read_text()
    trys = transcript_trys(transcript)
    claimed = transcript_tool_numbers(transcript)
    log.info("arm_a: %d TRY motifs, %d COMPILED replies", len(trys), len(claimed))
    assert len(trys) == len(claimed), "parser/grid drift vs transcript"

    ctx, info = G.build_ctx(data, workers)
    batch, train_per_day = info["batch"], info["train_per_day"]
    need, allowed, med = G.gate_bars(ctx, CLUSTER)
    log.info("gate need=%d allowed=%.4f (base median %.4f)", need, allowed, med)

    rows = []
    for k, t in enumerate(trys):
        r = G.score_candidate(ctx, t["text"], batch, CLUSTER, ctx.bg_pos)
        assert r is not None
        cr, cb = claimed[k]
        status = "OK" if (r["recall"], r["bg_hits"]) == (cr, cb) else "MISMATCH"
        log.info("  step %d: re-score %d/4 @%d vs transcript %d/4 @%d [%s]",
                 t["step"], r["recall"], r["bg_hits"], cr, cb, status)
        assert status == "OK", f"step {t['step']} does not reproduce"
        r.update({"id": f"a_step{t['step']}", "text": t["text"],
                  "fires_day": r["bg_rate"] * train_per_day,
                  "pass_absolute": r["recall"] >= need and r["bg_rate"] <= allowed + 1e-9,
                  "pass_marginal": r["recall"] >= need and r["marg_rate"] <= allowed + 1e-9})
        rows.append(r)
    arm_a = pl.from_dicts([{k: (str(v) if k in ("caught", "text") else v)
                            for k, v in sorted(r.items())} for r in rows])
    arm_a.write_csv(ROOT / "arm_a" / "arm_a_try_exact.csv")

    per_arm = {"llm": arm_a,
               "enum": pl.read_csv(ROOT / "arm_b" / "frontier_exact.csv"),
               "hybrid": pl.read_csv(ROOT / "arm_c" / "frontier_exact.csv")}
    for name, df in per_arm.items():
        df.write_csv(ROOT / f"frontier_{name}.csv")
    all_rows = []
    for name, df in per_arm.items():
        for r in df.iter_rows(named=True):
            r["arm"] = name
            all_rows.append(r)
    for r in G.pareto(all_rows):
        r["combined_pareto"] = True
    for r in all_rows:
        r.setdefault("combined_pareto", False)
    pl.from_dicts(all_rows).write_csv(ROOT / "frontier_all.csv")
    G.write_scatter({n: ROOT / f"frontier_{n}.csv" for n in per_arm},
                    ROOT / "pareto.svg", allowed, train_per_day)

    log.info("per-arm best bg_hits by recall (exact, full bg):")
    for name, df in per_arm.items():
        best = (df.group_by("recall").agg(pl.col("bg_hits").min())
                .sort("recall").iter_rows(named=True))
        log.info("  %s: %s", name,
                 ", ".join(f"{b['recall']}/4@{b['bg_hits']}" for b in best))
        for mode in ("pass_absolute", "pass_marginal"):
            n = df.filter(pl.col(mode)).height
            log.info("  %s %s passes: %d", name, mode, n)
    log.info("combined Pareto (distinct recall@bg):")
    seen = set()
    for r in sorted(all_rows, key=lambda r: (r["recall"], r["bg_hits"])):
        key = (r["recall"], r["bg_hits"])
        if r["combined_pareto"] and key not in seen:
            seen.add(key)
            log.info("  %d/4 @%d (%s %s)", r["recall"], r["bg_hits"], r["arm"], r["id"])
    log.info("done")


if __name__ == "__main__":
    typer.run(main)
