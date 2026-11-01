#!/usr/bin/env python3
"""Hybrid arm for issue #26: LLM structures + gridded numeric params.

Structures are the top-2 arm-(a) TRYs by (recall, quietness) —
selection rule pre-registered, structures copied verbatim from
eval/issue26/arm_a/discover_fan_out_transcript.md:

  S1 (step 7): small-count fan + most-members-big-amount, 3/4 @3.6%
  S2 (step 9): shared-receiver butterfly + tight span, 1/4 @0.455%

Numeric grids: S1 windows x amount-floor x count-shape x emit
(3x3x3x2 = 54), S2 span x tightness x emit (3x3x2 = 18).
Same ctx/seeds/scoring/gate as the other arms (discover_grid.py).

Usage:
    uv run eval/discover_hybrid.py --data /path/to/hi-small --workers 16
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from eval import discover_grid as G  # noqa: E402

log = logging.getLogger("discover_hybrid")

CLUSTER = [123, 138, 223, 224]  # FAN-OUT train misses (discover_misses.csv)

# S1: arm-a step 7 verbatim at (4d, 6000/10000, (1,2,4,6), 0.5).
S1_WINDOWS = ["2d", "4d", "7d"]
S1_AMT_LO = [3000, 6000, 10000]  # ramp(lo, lo+4000 USD)
S1_COUNTS = [(1, 2, 3, 4), (1, 2, 4, 6), (1, 3, 6, 9)]  # trapezoid(a,b,c,d)
S1_EMITS = [0.5, 0.3]

# S2: arm-a step 9 verbatim at (7d, 72h, 0.5).
S2_SPANS = ["2d", "7d", "10d"]
S2_FALLS = ["48h", "72h", "120h"]  # trapezoid(0h,0h,24h,F)
S2_EMITS = [0.5, 0.6]


def s1_text(window: str, lo: int, count: tuple[int, int, int, int],
            emit: float) -> str:
    a, b, c, d = count
    return (f"term small_n = trapezoid({a}, {b}, {c}, {d})\n"
            f"term big = ramp({lo} USD, {lo + 4000} USD)\n"
            "motif cand {\n"
            "  match (s:account) -[e:transfer]-> (r:account)\n"
            f"  group by s over r window {window}\n"
            "  having count(distinct r) >= 2\n"
            "  soft count(distinct r) is small_n and most r: e.amount is big\n"
            "  score product\n"
            f"  emit when score >= {emit}\n"
            "}\n")


def s2_text(span: str, fall: str, emit: float) -> str:
    return ("term tight = trapezoid(0h, 0h, 24h, FALL)\n"
            "motif cand {\n"
            "  match (x1:account) -[a:transfer]-> (y1:account),"
            " (x1:account) -[b:transfer]-> (y2:account),"
            " (x2:account) -[c:transfer]-> (y1:account),"
            " (x2:account) -[d:transfer]-> (y2:account)\n"
            f"  where x1 != x2 and y1 != y2 and span(all) <= {span}\n"
            "  soft span(all) is tight\n"
            "  score min\n"
            f"  emit when score >= {emit}\n"
            "}\n".replace("FALL", fall))


def build_templates() -> list[dict]:
    out = []
    for w in S1_WINDOWS:
        for lo in S1_AMT_LO:
            for count in S1_COUNTS:
                for emit in S1_EMITS:
                    out.append({"id": f"h_s1_{w}_{lo}_{count[2]}-{count[3]}_t{emit}",
                                "text": s1_text(w, lo, count, emit)})
    for s in S2_SPANS:
        for f in S2_FALLS:
            for emit in S2_EMITS:
                out.append({"id": f"h_s2_{s}_{f}_t{emit}",
                            "text": s2_text(s, f, emit)})
    return out


def main(
    data: Annotated[Path, typer.Option(help="Prepped dataset directory.")] = Path("data/hi-small"),
    workers: Annotated[int, typer.Option(help="Worker processes (1 = serial).")] = 16,
    out: Annotated[Path, typer.Option(help="Directory for grid outputs.")] = Path("eval/issue26/arm_c"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    templates = build_templates()
    assert len(templates) == 72, len(templates)
    # Pin: the verbatim-LLM corners must be in the grid.
    assert any(t["id"] == "h_s1_4d_6000_4-6_t0.5" for t in templates)
    assert any(t["id"] == "h_s2_7d_72h_t0.5" for t in templates)
    ctx, info = G.build_ctx(data, workers)
    G.run_grid(ctx, info, CLUSTER, templates, out, "hybrid")
    log.info("done")


if __name__ == "__main__":
    typer.run(main)
