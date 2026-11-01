#!/usr/bin/env python3
"""Enumerative arm for issue #26: template grid, no model cost.

Pre-registered grid over the fan-out family on the FAN-OUT train
miss cluster [123, 138, 223, 224]: windows x having x count-term x
amount-filter x emit = 4x3x4x4x2 = 384 TRYs. Same ctx seeds, same
exact-floor scoring, same gate as the LLM arm — only the proposer
differs. Two-stage protocol (see eval/discover_grid.py).

Usage:
    uv run eval/discover_enum.py --data /path/to/hi-small --workers 16
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

log = logging.getLogger("discover_enum")

CLUSTER = [123, 138, 223, 224]  # FAN-OUT train misses (discover_misses.csv)

WINDOWS = ["1d", "2d", "4d", "7d"]
HAVINGS = [1, 2, 3]
COUNT_TERMS = [(1, 3), (3, 8), (2, 5), (1, 2)]  # ramp(a, b) incl. several/many
AMT_FILTERS = [None, ("min", 5000, 10000), ("min", 10000, 20000),
               ("avg", 5000, 10000)]  # (agg, ramp lo/hi USD)
EMITS = [0.5, 0.3]


def template(window: str, having: int, count_ab: tuple[int, int],
             amt, emit: float) -> str:
    a, b = count_ab
    lines = [f"term c = ramp({a}, {b})"]
    soft = "count(distinct r) is c"
    if amt is not None:
        agg, lo, hi = amt
        lines.append(f"term amt = ramp({lo} USD, {hi} USD)")
        soft += f" and {agg}(e.amount) is amt"
    lines.append("motif cand {")
    lines.append("  match (s:account) -[e:transfer]-> (r:account)")
    lines.append(f"  group by s over r window {window}")
    lines.append(f"  having count(distinct r) >= {having}")
    lines.append(f"  soft {soft}")
    lines.append("  score product")
    lines.append(f"  emit when score >= {emit}")
    lines.append("}")
    return "\n".join(lines) + "\n"


def build_templates() -> list[dict]:
    out, i = [], 0
    for w in WINDOWS:
        for h in HAVINGS:
            for ab in COUNT_TERMS:
                for amt in AMT_FILTERS:
                    for emit in EMITS:
                        out.append({"id": f"e{i:03d}_w{w}_h{h}_c{ab[0]}-{ab[1]}"
                                          f"_a{amt[0] if amt else 'none'}"
                                          f"{amt[1] if amt else ''}_t{emit}",
                                    "text": template(w, h, ab, amt, emit)})
                        i += 1
    return out


def main(
    data: Annotated[Path, typer.Option(help="Prepped dataset directory.")] = Path("data/hi-small"),
    workers: Annotated[int, typer.Option(help="Worker processes (1 = serial).")] = 16,
    out: Annotated[Path, typer.Option(help="Directory for grid outputs.")] = Path("eval/issue26/arm_b"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    templates = build_templates()
    assert len(templates) == 384, len(templates)
    ctx, info = G.build_ctx(data, workers)
    G.run_grid(ctx, info, CLUSTER, templates, out, "enum")
    log.info("done")


if __name__ == "__main__":
    typer.run(main)
