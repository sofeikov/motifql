"""Tests for crisp (exact) scoring mode (Nov 1, RQ2).

Rule under test: with crisp=True every soft leaf degree (term use,
`~=` tolerance) is thresholded at the support boundary (degree > 0 ->
1.0, else 0.0); combination (and/or/not/quantifiers over the leaves)
is unchanged. So on motifs without most/few, crisp score ==
(fuzzy top_score > 0), and pruning stays sound (pruned crisp ==
unpruned crisp — the bound must use crisp degrees, since a branch
with fuzzy 0.5 < emit can still be crisp 1.0).
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from motifql.degrees import crisp_degree, libs_from_doc  # noqa: E402
from motifql.graph import TemporalGraph  # noqa: E402
from motifql.groups import score_any, score_butterfly, score_group  # noqa: E402
from motifql.matcher import score  # noqa: E402
from motifql.parser import parse_text  # noqa: E402
from motifql.planner import compile_motif  # noqa: E402

BASE = datetime(2022, 9, 3)
HEADER = "term soon = trapezoid(0h, 0h, 6h, 48h)\n"


def _frame(rows: list[tuple[int, str, str, str, str, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "transfer_id": list(range(len(rows))),
            "timestamp": [BASE + timedelta(minutes=r[0]) for r in rows],
            "from_bank": [r[1] for r in rows],
            "from_account": [r[2] for r in rows],
            "to_bank": [r[3] for r in rows],
            "to_account": [r[4] for r in rows],
            "amount_usd": [r[5] for r in rows],
        }
    )


def _round_trip():
    doc = parse_text(
        HEADER
        + "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
        + " where e is time_ordered and span(e) <= 7d"
        + " soft each e: e.gap is soon"
        + " and each e: e.amount ~= prev.amount within 20%"
        + " score min emit when score >= 0.6 }"
    )
    return doc, compile_motif(doc, "t"), libs_from_doc(doc)


def test_crisp_degree_table() -> None:
    assert crisp_degree(0.0, True) == 0.0
    assert crisp_degree(0.001, True) == 1.0
    assert crisp_degree(0.5, True) == 1.0
    assert crisp_degree(1.0, True) == 1.0
    assert crisp_degree(0.5, False) == 0.5  # flag off: identity


def test_path_crisp_is_support_boundary() -> None:
    _, plan, libs = _round_trip()
    # Partial witness: 27h gap reads soon as 0.5, amounts match.
    g = TemporalGraph.from_transfers(_frame([
        (0, "001", "B", "002", "A", 100.0),
        (27 * 60, "002", "A", "001", "B", 100.0),
    ]))
    fuzzy = score(g, 1, plan, libs, prune=False)
    assert not fuzzy.found and fuzzy.top_score == pytest.approx(0.5)
    exact = score(g, 1, plan, libs, crisp=True)
    assert exact.found and exact.score == pytest.approx(1.0)
    # Zero witness: 40% amount gap is outside the tolerance support.
    g0 = TemporalGraph.from_transfers(_frame([
        (0, "001", "B", "002", "A", 100.0),
        (60, "002", "A", "001", "B", 140.0),
    ]))
    assert score(g0, 1, plan, libs, prune=False).top_score == 0.0
    zero = score(g0, 1, plan, libs, crisp=True)
    assert not zero.found and zero.top_score == 0.0


def test_path_crisp_pruning_sound() -> None:
    # The partial witness above has fuzzy 0.5 < emit: a pruning bound
    # using fuzzy degrees would cut it, but crisp must keep it, so
    # pruned crisp has to equal unpruned crisp exactly.
    _, plan, libs = _round_trip()
    g = TemporalGraph.from_transfers(_frame([
        (0, "001", "B", "002", "A", 100.0),
        (60, "003", "C", "002", "A", 10000.0),  # decoy: crisp-0 amount
        (27 * 60, "002", "A", "001", "B", 100.0),
    ]))
    pruned = score(g, 2, plan, libs, prune=True, crisp=True)
    plain = score(g, 2, plan, libs, prune=False, crisp=True)
    assert pruned.found and pruned.score == pytest.approx(1.0)
    assert (plain.found, plain.score, plain.witness, plain.top_score) == (
        pruned.found, pruned.score, pruned.witness, pruned.top_score)


def test_group_fan_crisp_via_score_any() -> None:
    doc = parse_text(
        (Path("examples/amlworld.motif")).read_text())
    plan, libs = compile_motif(doc, "fan_out"), libs_from_doc(doc)
    # 5 distinct receivers: many(5) = 0.4 fuzzy, 1.0 crisp.
    g = TemporalGraph.from_transfers(_frame(
        [(i * 30, "001", "S", "002", f"R{i}", 100.0) for i in range(5)]))
    fuzzy = score_any(g, 4, plan, libs)
    assert not fuzzy.found and fuzzy.top_score == pytest.approx(0.4)
    exact = score_any(g, 4, plan, libs, crisp=True)
    assert exact.found and exact.score == pytest.approx(1.0)
    assert exact.top_score == pytest.approx(1.0)
    # 2 receivers: below the ramp support entirely — crisp stays 0.
    g2 = TemporalGraph.from_transfers(_frame([
        (0, "001", "S", "002", "R0", 100.0),
        (30, "001", "S", "002", "R1", 100.0),
    ]))
    assert score_group(g2, 1, plan, libs).top_score == 0.0
    assert score_group(g2, 1, plan, libs, crisp=True).top_score == 0.0


def test_most_still_grades_the_crisp_mean() -> None:
    # most/few/@name combine per-member leaves through a ramp: the
    # leaves crisp, the ramp still grades (its exact analog is
    # genuinely ambiguous — a ramp has two breakpoints — so the
    # combination is deliberately unchanged).
    doc = parse_text(
        "motif t { match (s:account) -[e:transfer]-> (r:account)"
        " group by s over r window 1d"
        " having count(distinct r) >= 2"
        " soft most r: e.amount ~= 150 USD within 100 USD"
        " score min emit when score >= 0.5 }")
    plan, libs = compile_motif(doc, "t"), libs_from_doc(doc)
    g = TemporalGraph.from_transfers(_frame([
        (0, "001", "S", "002", "R0", 100.0),   # ~=: 0.5
        (30, "001", "S", "002", "R1", 150.0),  # ~=: 1.0
        (60, "001", "S", "002", "R2", 300.0),  # ~=: 0.0
    ]))
    fuzzy = score_group(g, 2, plan, libs)
    assert fuzzy.top_score == pytest.approx(0.0)  # ramp(mean 0.5) = 0
    exact = score_group(g, 2, plan, libs, crisp=True)
    assert exact.top_score == pytest.approx((2 / 3 - 0.5) / 0.3)  # ramp(mean 2/3)


def test_butterfly_crisp() -> None:
    doc = parse_text(
        "term patient = trapezoid(0h, 0h, 24h, 120h)\n"
        "motif bf { match (x1:account) -[e1:transfer]-> (y1:account),"
        " (x1) -[e2:transfer]-> (y2:account),"
        " (x2:account) -[e3:transfer]-> (y1),"
        " (x2) -[e4:transfer]-> (y2)"
        " where x1 != x2 and y1 != y2 and span(all) <= 5d"
        " soft span(all) is patient"
        " score min emit when score >= 0.6 }")
    plan, libs = compile_motif(doc, "bf"), libs_from_doc(doc)
    g = TemporalGraph.from_transfers(_frame([
        (0, "001", "X1", "002", "Y1", 50.0),
        (60, "001", "X1", "002", "Y2", 50.0),
        (120, "001", "X2", "002", "Y1", 50.0),
        (4320, "001", "X2", "002", "Y2", 50.0),  # 72h span: patient = 0.5
    ]))
    fuzzy = score_butterfly(g, 3, plan, libs)
    assert not fuzzy.found and fuzzy.top_score == pytest.approx(0.5)
    exact = score_butterfly(g, 3, plan, libs, crisp=True)
    assert exact.found and exact.score == pytest.approx(1.0)
