"""Tests for group + butterfly executors and their planner (Oct 30).

Oracle strategy: every executor is checked against an independent
brute-force grouping written with plain Python loops over the transfer
frame (not the CSR store), so a shared-store bug cannot hide. Planner
pins: all 8 reference motifs compile, and non-v0 shapes still E_PLAN.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from motifql.degrees import libs_from_doc  # noqa: E402
from motifql.graph import TemporalGraph  # noqa: E402
from motifql.groups import score_any, score_butterfly, score_group  # noqa: E402
from motifql.parser import parse_text  # noqa: E402
from motifql.planner import (  # noqa: E402
    ButterflyPlan, GroupPlan, PlanError, compile_motif,
)

BASE = datetime(2022, 9, 3)
HEADER = "term soon = trapezoid(0h, 0h, 6h, 48h)\nterm several = ramp(2, 5)\nterm many = ramp(3, 8)\n"


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


def _doc():
    return parse_text(HEADER + (Path("examples/amlworld.motif")).read_text())


# -- planner ---------------------------------------------------------------

def test_all_reference_motifs_compile() -> None:
    doc = _doc()
    kinds = {}
    for name in ("round_trip", "layering_chain", "fan_out", "fan_in",
                 "scatter_gather", "gather_scatter", "structuring_fan_in",
                 "bipartite_butterfly"):
        plan = compile_motif(doc, name)
        kinds[name] = type(plan).__name__
    assert kinds == {
        "round_trip": "PathPlan", "layering_chain": "PathPlan",
        "fan_out": "GroupPlan", "fan_in": "GroupPlan",
        "scatter_gather": "GroupPlan", "gather_scatter": "GroupPlan",
        "structuring_fan_in": "GroupPlan",
        "bipartite_butterfly": "ButterflyPlan",
    }


def test_group_roles_and_windows() -> None:
    doc = _doc()
    fan = compile_motif(doc, "fan_out")
    assert isinstance(fan, GroupPlan)
    assert fan.kind == "fan" and fan.role("key") == "s" and fan.window_s == 4 * 86400.0
    scat = compile_motif(doc, "scatter_gather")
    assert scat.kind == "scatter" and scat.window_s == 48 * 3600.0
    assert (scat.role("s"), scat.role("m"), scat.role("d")) == ("s", "m", "d")
    gath = compile_motif(doc, "gather_scatter")
    assert gath.kind == "gather" and gath.role("c") == "c"
    bf = compile_motif(doc, "bipartite_butterfly")
    assert isinstance(bf, ButterflyPlan)
    assert bf.window_s == 2 * 86400.0 and len(bf.edge_vars) == 4


def test_non_v0_shapes_still_eplan() -> None:
    # Fan with a where clause: the walk carries no per-edge hard filters.
    with pytest.raises(PlanError, match="E_PLAN"):
        compile_motif(parse_text(HEADER + "motif t { match (a:account) -[e:transfer]-> (b:account)"
                                 " group by a over b window 1d"
                                 " where e.amount > 10 USD"
                                 " having count(distinct b) >= 2 }"), "t")
    # Three single-edge patterns sharing nothing: not gather, not butterfly.
    with pytest.raises(PlanError, match="E_PLAN"):
        compile_motif(parse_text(HEADER + "motif t { match (a:account) -[e1:transfer]-> (b:account),"
                                 " (c:account) -[e2:transfer]-> (d:account),"
                                 " (e:account) -[e3:transfer]-> (f:account)"
                                 " where span(all) <= 1d }"), "t")
    # prev has no meaning without a path quantifier.
    with pytest.raises(PlanError, match="E_PLAN"):
        compile_motif(parse_text(HEADER + "motif t { match (a:account) -[e:transfer]-> (b:account)"
                                 " group by a over b window 1d"
                                 " having count(distinct b) >= 2"
                                 " soft each b: e.amount ~= prev.amount within 10% }"), "t")


# -- fan oracle ------------------------------------------------------------

def test_fan_out_matches_brute_force() -> None:
    rows = [(i * 30, "001", "S", "002", f"R{i}", 100.0 + i) for i in range(6)]
    frame = _frame(rows)
    g = TemporalGraph.from_transfers(frame)
    doc, plan, libs = _doc(), compile_motif(_doc(), "fan_out"), libs_from_doc(_doc())
    anchor = 5
    res = score_group(g, anchor, plan, libs)
    # Brute force: every transfer from S within 1d before the anchor.
    anchor_t = frame["timestamp"][anchor]
    window = [r for r in frame.iter_rows(named=True)
              if r["from_account"] == "S" and r["timestamp"] <= anchor_t
              and (anchor_t - r["timestamp"]).total_seconds() <= 86400]
    assert res.truncated is False
    assert sorted(int(g.transfer_id[e]) for e in res.witness) == sorted(r["transfer_id"] for r in window)
    assert len(res.witness) == 6
    # Determinism: score twice, same answer.
    again = score_group(g, anchor, plan, libs)
    assert (again.found, again.score, again.witness) == (res.found, res.score, res.witness)


def test_fan_out_fires_with_many_members() -> None:
    rows = [(i * 10, "001", "S", "002", f"R{i}", 100.0) for i in range(9)]
    g = TemporalGraph.from_transfers(_frame(rows))
    doc = _doc()
    res = score_group(g, 8, compile_motif(doc, "fan_out"), libs_from_doc(doc))
    assert res.found and res.score == pytest.approx(1.0)
    assert len(res.witness) == 9


def test_fan_in_having_gate() -> None:
    # Only the anchor pays T: count(distinct x) == 1 < 2 -> miss.
    g = TemporalGraph.from_transfers(_frame([(0, "001", "X", "002", "T", 50.0)]))
    doc = _doc()
    res = score_group(g, 0, compile_motif(doc, "fan_in"), libs_from_doc(doc))
    assert not res.found and res.top_score == 0.0


# -- scatter / gather ------------------------------------------------------

def test_scatter_needs_two_members() -> None:
    # s -> m1, s -> m2, m1 -> d, m2 -> d(anchor): 2 members.
    # several = ramp(1, 3) since the Oct 30 recalibration, so 2 members
    # score 0.5 and fire exactly at the 0.5 emit; 1 member still misses.
    g = TemporalGraph.from_transfers(_frame([
        (0, "001", "S", "002", "M1", 100.0), (10, "001", "S", "002", "M2", 100.0),
        (20, "002", "M1", "003", "D", 100.0), (30, "002", "M2", "003", "D", 100.0),
    ]))
    doc = _doc()
    res = score_group(g, 3, compile_motif(doc, "scatter_gather"), libs_from_doc(doc))
    assert res.found and res.top_score == pytest.approx(0.5)
    assert sorted(int(g.transfer_id[e]) for e in res.witness) == [0, 1, 2, 3]


def test_scatter_fires_with_three_members() -> None:
    rows = [(0, "001", "S", "002", "M1", 100.0), (5, "001", "S", "002", "M2", 100.0),
            (10, "001", "S", "002", "M3", 100.0),
            (20, "002", "M1", "003", "D", 100.0), (25, "002", "M2", "003", "D", 100.0),
            (30, "002", "M3", "003", "D", 100.0)]
    g = TemporalGraph.from_transfers(_frame(rows))
    doc = _doc()
    res = score_group(g, 5, compile_motif(doc, "scatter_gather"), libs_from_doc(doc))
    # several(3) = 1.0 under ramp(1, 3), amounts and sums exact -> fires.
    assert res.found and res.top_score == pytest.approx(1.0)


def test_gather_witness_matches_brute_force() -> None:
    g = TemporalGraph.from_transfers(_frame([
        (0, "001", "X1", "002", "C", 100.0), (10, "001", "X2", "002", "C", 100.0),
        (20, "002", "C", "003", "Y1", 100.0), (30, "002", "C", "003", "Y2", 100.0),
    ]))
    doc = _doc()
    res = score_group(g, 3, compile_motif(doc, "gather_scatter"), libs_from_doc(doc))
    assert sorted(int(g.transfer_id[e]) for e in res.witness) == [0, 1, 2, 3]
    assert res.truncated is False


def test_scatter_pairs_older_leg1_when_newest_is_stale() -> None:
    # H6: per mid, the most recent leg1 strictly before leg2 pairs up.
    # M1's t=55 leg1 postdates its t=50 leg2, but the t=10 leg1 is still
    # a valid pair — previously the mid was dropped and the group missed.
    g = TemporalGraph.from_transfers(_frame([
        (10, "001", "S", "002", "M1", 100.0), (20, "001", "S", "002", "M2", 100.0),
        (50, "002", "M1", "003", "D", 100.0), (55, "001", "S", "002", "M1", 100.0),
        (60, "002", "M2", "003", "D", 100.0),
    ]))
    doc = _doc()
    res = score_group(g, 4, compile_motif(doc, "scatter_gather"), libs_from_doc(doc))
    assert res.found and res.top_score == pytest.approx(0.5)
    assert sorted(int(g.transfer_id[e]) for e in res.witness) == [0, 1, 2, 4]


def test_gather_drops_payout_before_any_collection() -> None:
    # H7: `p.time > g.time` is enforced — a payout preceding every
    # collection cannot count as redistribution.
    g = TemporalGraph.from_transfers(_frame([
        (5, "002", "C", "003", "Y0", 100.0),
        (10, "001", "X1", "002", "C", 100.0), (20, "001", "X2", "002", "C", 100.0),
        (30, "001", "X3", "002", "C", 100.0),
        (40, "002", "C", "003", "Y1", 100.0), (45, "002", "C", "003", "Y2", 100.0),
        (50, "002", "C", "003", "Y3", 100.0),
    ]))
    doc = _doc()
    res = score_group(g, 6, compile_motif(doc, "gather_scatter"), libs_from_doc(doc))
    assert res.found and res.top_score == pytest.approx(1.0)
    assert sorted(int(g.transfer_id[e]) for e in res.witness) == [1, 2, 3, 4, 5, 6]


# -- butterfly -------------------------------------------------------------

def test_butterfly_finds_core_and_ignores_decoy() -> None:
    g = TemporalGraph.from_transfers(_frame([
        (0, "001", "X1", "002", "Y1", 50.0), (5, "001", "X1", "002", "ZZ", 999.0),
        (10, "001", "X1", "002", "Y2", 50.0),
        (20, "001", "X2", "002", "Y1", 50.0), (30, "001", "X2", "002", "Y2", 50.0),
    ]))
    doc = _doc()
    plan = compile_motif(doc, "bipartite_butterfly")
    res = score_butterfly(g, 4, plan, libs_from_doc(doc))
    assert res.found and res.score == pytest.approx(1.0)
    assert sorted(int(g.transfer_id[e]) for e in res.witness) == [0, 2, 3, 4]
    # Anchor that cannot complete a core: the decoy edge.
    miss = score_butterfly(g, 1, plan, libs_from_doc(doc))
    assert not miss.found


def _butterfly_doc(soft: str | None) -> object:
    text = (
        "term soon = trapezoid(0h, 0h, 6h, 48h)\n"
        "term patient = trapezoid(0h, 0h, 24h, 120h)\n"
        "motif bf { match (x1:account) -[e1:transfer]-> (y1:account),"
        " (x1) -[e2:transfer]-> (y2:account),"
        " (x2:account) -[e3:transfer]-> (y1),"
        " (x2) -[e4:transfer]-> (y2)"
        " where x1 != x2 and y1 != y2 and span(all) <= 5d"
    )
    if soft is not None:
        text += f" soft {soft}"
    return parse_text(text + " score min emit when score >= 0.6 }")


def _butterfly_frame() -> object:
    # 2x2 core with a 72h span: soon(72h) = 0.0, patient(72h) = 0.5.
    return _frame([
        (0, "001", "X1", "002", "Y1", 50.0),
        (60, "001", "X1", "002", "Y2", 50.0),
        (120, "001", "X2", "002", "Y1", 50.0),
        (4320, "001", "X2", "002", "Y2", 50.0),
    ])


def test_butterfly_rejects_non_span_soft() -> None:
    # H8: the executor only grades span(all); anything else must E_PLAN
    # instead of being silently scored as `soon`.
    with pytest.raises(PlanError, match="E_PLAN"):
        compile_motif(_butterfly_doc("e1.amount ~= e2.amount within 10%"), "bf")


def test_butterfly_grades_named_span_term() -> None:
    # H8: the core is graded by the term the motif names, not hardcoded
    # `soon` (which would score this 72h span 0.0).
    doc = _butterfly_doc("span(all) is patient")
    g = TemporalGraph.from_transfers(_butterfly_frame())
    res = score_butterfly(g, 3, compile_motif(doc, "bf"), libs_from_doc(doc))
    assert res.top_score == pytest.approx(0.5)
    assert not res.found  # below the 0.6 emit


def test_butterfly_without_soft_scores_one() -> None:
    # H8: with no soft clause the score is 1 (LANGUAGE §4.6) — the old
    # code still graded by a global `soon` term when one was defined.
    doc = _butterfly_doc(None)
    g = TemporalGraph.from_transfers(_butterfly_frame())
    res = score_butterfly(g, 3, compile_motif(doc, "bf"), libs_from_doc(doc))
    assert res.found and res.score == pytest.approx(1.0)


def test_score_any_dispatches() -> None:
    g = TemporalGraph.from_transfers(_frame([
        (0, "001", "X1", "002", "Y1", 50.0), (10, "001", "X1", "002", "Y2", 50.0),
        (20, "001", "X2", "002", "Y1", 50.0), (30, "001", "X2", "002", "Y2", 50.0),
    ]))
    doc = _doc()
    libs = libs_from_doc(doc)
    for name, pos in (("fan_out", 1), ("bipartite_butterfly", 3)):
        plan = compile_motif(doc, name)
        assert score_any(g, pos, plan, libs).witness == \
            (score_group if name == "fan_out" else score_butterfly)(g, pos, plan, libs).witness
