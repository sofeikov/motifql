#!/usr/bin/env python3
"""Precompute the demo evasion slider curve into ui/canned.json (#22).

Scores the canned witness (anchor 2012559) over a noise x gap grid via
`compile_service.score_evasion` — the same code the live sliders call —
so the canned curve and the live path agree by construction. Level
(0, x1) reproduces the pinned demo contract (top 0.718...).

Usage:
    uv run scripts/compute_evasion_curve.py --data data/hi-small
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from motifql.compile_service import EVASION_SEED, score_evasion  # noqa: E402
from motifql.engine import Engine  # noqa: E402
from motifql.server import CANNED_PATH, DEMO_ANCHOR_TID  # noqa: E402

NOISE_PCT = [0, 5, 10, 15, 20, 25, 30]
GAPS = [1.0, 1.25, 1.5, 1.75, 2.0]


def main(
    data: Annotated[str, typer.Option(help="Data dir or transfers parquet.")] = "data/hi-small",
    motifs: Annotated[str, typer.Option(help="Motif file.")] = "examples/amlworld.motif",
    seed: Annotated[int, typer.Option(help="Perturbation seed.")] = EVASION_SEED,
    canned: Annotated[str, typer.Option(help="Canned JSON to update.")] = str(CANNED_PATH),
) -> None:
    canned_path = Path(canned)
    obj = json.loads(canned_path.read_text())
    engine = Engine.from_motif_file(motifs, transfers=data)
    print(f"graph: {engine.graph.n_edges} edges; anchor {DEMO_ANCHOR_TID}")
    points: list[dict] = []
    baseline: dict | None = None
    for pct in NOISE_PCT:
        row = []
        for gap in GAPS:
            _, ws = score_evasion(
                obj["motif"], engine, anchor_tids=[DEMO_ANCHOR_TID],
                amount_noise_pct=pct / 100.0, gap_scale=gap, seed=seed)
            w = ws[0]
            if pct == 0 and gap == 1.0:
                baseline = {"fuzzy_top": w.baseline_fuzzy,
                            "crisp_top": w.baseline_crisp}
            points.append({"noise_pct": pct, "gap": gap,
                           "fuzzy_top": w.fuzzy_top, "crisp_top": w.crisp_top,
                           "found": w.found, "explanation": w.explanation})
            row.append(f"g{gap}: {w.fuzzy_top:.3f}/{w.crisp_top:.0f}")
        print(f"noise {pct:>2}%: " + "  ".join(row))
    assert baseline is not None
    zero = points[0]
    assert zero["fuzzy_top"] == baseline["fuzzy_top"], "level 0 must reproduce baseline"
    print(f"baseline: fuzzy {baseline['fuzzy_top']:.6f} "
          f"crisp {baseline['crisp_top']:.0f}")
    obj["evasion_curve"] = {"seed": seed, "anchor_tid": DEMO_ANCHOR_TID,
                            "baseline": baseline,
                            "grid": {"noise_pct": NOISE_PCT, "gap": GAPS},
                            "points": points}
    canned_path.write_text(json.dumps(obj, indent=2) + "\n")
    print(f"wrote {canned_path} ({len(points)} points)")


if __name__ == "__main__":
    typer.run(main)
