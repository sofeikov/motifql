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
    assert scat.kind == "scatter" and scat.window_s == 7 * 86400.0
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


def test_group_member_account_fields_eplan() -> None:
    # F4: `r.in_degree` reads an account field the executors never fill,
    # so every member is skipped and the quantifier returns its empty-set
    # constant (`each` -> 1, `most` -> 0) regardless of the data. Reject
    # loudly like path motifs do instead of scoring a silent constant.
    with pytest.raises(PlanError, match="E_PLAN"):
        compile_motif(parse_text(HEADER + "motif t { match (s:account) -[e:transfer]-> (r:account)"
                                 " group by s over r window 1d"
                                 " having count(distinct r) >= 2"
                                 " soft most r: r.in_degree <= 5 }"), "t")


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


def test_group_self_loop_member_is_included() -> None:
    # Policy pin (Nov 1): unlike path witnesses (C1 excludes self-loops),
    # group members include self-loop edges — planted truth itself
    # contains one (GATHER-SCATTER 0048309->0048309), so excluding them
    # would drop real members.
    rows = [(0, "001", "S", "002", "R1", 100.0),
            (30, "001", "S", "001", "S", 200.0),
            (60, "001", "S", "002", "R2", 300.0)]
    g = TemporalGraph.from_transfers(_frame(rows))
    doc = _doc()
    res = score_group(g, 2, compile_motif(doc, "fan_out"), libs_from_doc(doc))
    assert sorted(int(g.transfer_id[e]) for e in res.witness) == [0, 1, 2]


def test_fan_out_fires_with_many_members() -> None:
    rows = [(i * 10, "001", "S", "002", f"R{i}", 100.0) for i in range(9)]
    g = TemporalGraph.from_transfers(_frame(rows))
    doc = _doc()
    res = score_group(g, 8, compile_motif(doc, "fan_out"), libs_from_doc(doc))
    assert res.found and res.score == pytest.approx(1.0)
    assert len(res.witness) == 9


def test_group_soft_arithmetic_evaluates() -> None:
    # M2: arithmetic newly passes checks in group clauses, so the
    # executors evaluate it instead of aborting (previously a loud
    # E_UNIT, never silent).
    doc = parse_text(
        "motif t { match (a:account) -[e:transfer]-> (b:account)"
        " group by a over b window 1d"
        " having count(distinct b) >= 2"
        " soft sum(e.amount) / count(distinct b) >= 50 USD"
        " score min emit when score >= 0.5 }")
    plan, libs = compile_motif(doc, "t"), libs_from_doc(doc)
    rich = TemporalGraph.from_transfers(_frame([
        (0, "001", "S", "002", "R1", 100.0),
        (10, "001", "S", "002", "R2", 100.0),
    ]))
    assert score_group(rich, 1, plan, libs).found
    poor = TemporalGraph.from_transfers(_frame([
        (0, "001", "S", "002", "R1", 10.0),
        (10, "001", "S", "002", "R2", 10.0),
    ]))
    assert not score_group(poor, 1, plan, libs).found


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


def test_scatter_aggregates_count_every_valid_leg() -> None:
    # E6: aggregates range over every edge joining a valid pair, not just
    # one pair per member. M1 is paid 50 + 50 and forwards 100: the sums
    # balance exactly (dropping the second 50 would score the sum term
    # 0.375 and miss emit).
    g = TemporalGraph.from_transfers(_frame([
        (10, "001", "S", "002", "M1", 50.0), (15, "001", "S", "002", "M1", 50.0),
        (20, "001", "S", "002", "M2", 100.0), (25, "001", "S", "002", "M3", 100.0),
        (30, "001", "S", "002", "M4", 100.0),
        (40, "002", "M1", "003", "D", 100.0), (45, "002", "M2", "003", "D", 100.0),
        (50, "002", "M3", "003", "D", 100.0), (55, "002", "M4", "003", "D", 100.0),
    ]))
    doc = _doc()
    res = score_group(g, 8, compile_motif(doc, "scatter_gather"), libs_from_doc(doc))
    assert res.found and res.top_score == pytest.approx(1.0)
    assert sorted(int(g.transfer_id[e]) for e in res.witness) == list(range(9))


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


def test_scatter_self_loop_anchor_never_completes() -> None:
    # F2: the main loop skips m == d members, so a self-loop anchor
    # (m0 == d) must not complete through the fallback either. Anchor 3
    # is A2 -> A2; it scores 0.5 via the fallback without the fix.
    g = TemporalGraph.from_transfers(_frame([
        (0, "001", "S", "002", "A1", 100.0), (1, "001", "S", "002", "A2", 100.0),
        (10, "002", "A1", "002", "A2", 100.0), (20, "002", "A2", "002", "A2", 100.0),
    ]))
    doc = _doc()
    res = score_group(g, 3, compile_motif(doc, "scatter_gather"), libs_from_doc(doc))
    assert not res.found and res.top_score == 0.0


def test_scatter_leg2_gap_scores_forwarding_delay() -> None:
    # F5: `leg2.gap` is the member's forward delay (leg2 - leg1). Both
    # mids forward after 10m, so `most m: leg2.gap is soon` is 1.0 —
    # without the reading every member is skipped and `most` gives 0.
    doc = parse_text(
        "term soon = trapezoid(0h, 0h, 6h, 48h)\n"
        "motif t { match (s:account) -[leg1:transfer]-> (m:account)"
        " -[leg2:transfer]-> (d:account)"
        " group by s, d over m window 2d"
        " where leg2.time > leg1.time and s != d"
        " having count(distinct m) >= 2"
        " soft most m: leg2.gap is soon"
        " score min emit when score >= 0.5 }")
    g = TemporalGraph.from_transfers(_frame([
        (0, "001", "S", "002", "M1", 100.0), (10, "002", "M1", "003", "D", 100.0),
        (20, "001", "S", "002", "M2", 100.0), (30, "002", "M2", "003", "D", 100.0),
    ]))
    res = score_group(g, 3, compile_motif(doc, "t"), libs_from_doc(doc))
    assert res.found and res.top_score == pytest.approx(1.0)


def test_gather_count_balance_gives_partial_credit_off_by_one() -> None:
    # M6 + Nov 1 tuning: `within 4` scores a diff-1 imbalance 0.75 —
    # linear tolerance is 0 at exactly the width, so narrower widths
    # zeroed pooled truth (diff-2 scores 0.5, the emit boundary).
    g = TemporalGraph.from_transfers(_frame([
        (10, "001", "X1", "002", "C", 100.0), (20, "001", "X2", "002", "C", 100.0),
        (30, "001", "X3", "002", "C", 100.0),
        (40, "002", "C", "003", "Y1", 100.0), (50, "002", "C", "003", "Y2", 100.0),
    ]))
    doc = _doc()
    res = score_group(g, 4, compile_motif(doc, "gather_scatter"), libs_from_doc(doc))
    assert not res.found
    assert res.top_score == pytest.approx(0.375)


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


def test_gather_without_collections_is_not_found() -> None:
    # F6: with no incoming edge there is no (g, p) binding, so nothing
    # should be found — even when `having` only gates the payout side.
    # A0 -> A1 @0, anchor A0 -> A2 @60, nothing ever pays A0.
    doc = parse_text(
        "term several = ramp(1, 3)\n"
        "motif t { match (x:account) -[g:transfer]-> (c:account),"
        " (c) -[p:transfer]-> (y:account)"
        " group by c over x, y window 4d"
        " where p.time > g.time"
        " having count(distinct y) >= 2"
        " soft count(distinct y) is several"
        " score min emit when score >= 0.5 }"
    )
    g = TemporalGraph.from_transfers(_frame([
        (0, "001", "A0", "002", "A1", 100.0),
        (60, "001", "A0", "002", "A2", 100.0),
    ]))
    res = score_group(g, 1, compile_motif(doc, "t"), libs_from_doc(doc))
    assert not res.found and res.top_score == 0.0


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


def _scatter_doc(having: str, soft: str | None) -> object:
    text = (
        "term several = ramp(1, 3)\n"
        "motif t { match (s:account) -[leg1:transfer]-> (m:account)"
        " -[leg2:transfer]-> (d:account)"
        " group by s, d over m window 2d"
        " where leg2.time > leg1.time and s != d"
        f" having {having}"
    )
    if soft is not None:
        text += f" soft {soft}"
    return parse_text(text + " score min emit when score >= 0.5 }")


def test_scatter_single_member_group_fires() -> None:
    # M5: `having count(distinct m) >= 1` accepts 1-member groups — the
    # old `< 2` shortcut skipped them (and their expansion accounting).
    g = TemporalGraph.from_transfers(_frame([
        (10, "001", "S", "002", "M", 100.0),
        (20, "002", "M", "003", "D", 100.0),
    ]))
    doc = _scatter_doc("count(distinct m) >= 1", None)
    res = score_group(g, 1, compile_motif(doc, "t"), libs_from_doc(doc))
    assert res.found and res.score == pytest.approx(1.0)
    assert sorted(int(g.transfer_id[e]) for e in res.witness) == [0, 1]


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


def test_butterfly_skips_truncated_partner_without_discarding_rest(monkeypatch) -> None:
    # M4: cross entries are built in id order but scored in time order,
    # so under truncation a missing key must `continue`, not `break` and
    # discard later partners that do have entries. B (early, truncated
    # out) precedes A (late, complete core) in scoring order.
    monkeypatch.setattr("motifql.groups.DEFAULT_MAX_EXPANSIONS", 4)
    g = TemporalGraph.from_transfers(_frame([
        (0, "001", "A", "002", "ZZ", 50.0),
        (60, "001", "B", "002", "Q", 50.0),
        (120, "001", "A", "002", "Q", 50.0),
        (180, "001", "A", "002", "R", 50.0),
        (240, "001", "P", "002", "R", 50.0),
        (300, "001", "P", "002", "Q", 50.0),
    ]))
    doc = _butterfly_doc("span(all) is soon")
    res = score_butterfly(g, 5, compile_motif(doc, "bf"), libs_from_doc(doc))
    assert res.truncated
    assert res.found and res.score == pytest.approx(1.0)


def test_butterfly_without_soft_scores_one() -> None:
    # H8: with no soft clause the score is 1 (LANGUAGE §4.6) — the old
    # code still graded by a global `soon` term when one was defined.
    doc = _butterfly_doc(None)
    g = TemporalGraph.from_transfers(_butterfly_frame())
    res = score_butterfly(g, 3, compile_motif(doc, "bf"), libs_from_doc(doc))
    assert res.found and res.score == pytest.approx(1.0)


def test_butterfly_tries_older_parallel_edges() -> None:
    # F7: only the newest edge per pair was tried, so `span(all) is not
    # soon` scored 0 here — the older X2 -> Y1 leg (span 71.7h, not-soon
    # 1.0) was never attempted. Every in-window edge per pair counts now.
    g = TemporalGraph.from_transfers(_frame([
        (0, "001", "X2", "002", "Y1", 50.0),
        (4000, "001", "X1", "002", "Y1", 50.0),
        (4100, "001", "X1", "002", "Y2", 50.0),
        (4200, "001", "X2", "002", "Y1", 50.0),
        (4300, "001", "X2", "002", "Y2", 50.0),
    ]))
    doc = _butterfly_doc("span(all) is not soon")
    res = score_butterfly(g, 4, compile_motif(doc, "bf"), libs_from_doc(doc))
    assert res.found and res.top_score == pytest.approx(1.0)
    assert sorted(int(g.transfer_id[e]) for e in res.witness) == [0, 1, 2, 4]


def test_member_skip_propagates_through_operators() -> None:
    # E5 mirror: an all-skipped member body is skipped (None), not 1.0/0.0.
    from motifql.groups import _Ctx, _eval_member
    from motifql.parser import And, Approx, IsTerm, Or, Pct, Ref
    libs = libs_from_doc(parse_text(HEADER))
    body = And((IsTerm(Ref(("e", "gap")), False, "soon", ()),
                Approx(Ref(("e", "amount")), Ref(("e", "amount")), Pct(0.1))))
    assert _eval_member(body, {}, _Ctx(), libs, "min") is None
    assert _eval_member(Or(body.items), {}, _Ctx(), libs, "min") is None


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


def test_all_but_forgives_one_bad_member() -> None:
    # Group mirror of the path test: one under-threshold member zeroes
    # `each` but `all but 1` still fires at 1.0.
    rows = [(0, "001", "S", "002", "R1", 100.0),
            (10, "001", "S", "002", "R2", 100.0),
            (20, "001", "S", "002", "R3", 100.0),
            (30, "001", "S", "002", "R4", 10.0)]
    g = TemporalGraph.from_transfers(_frame(rows))

    def _plan(quant: str):
        doc = parse_text(
            "motif t { match (a:account) -[e:transfer]-> (b:account)"
            " group by a over b window 1d"
            " having count(distinct b) >= 2"
            f" soft {quant} b: e.amount >= 50 USD"
            " score min emit when score >= 0.5 }")
        return compile_motif(doc, "t"), libs_from_doc(doc)

    plan, libs = _plan("each")
    assert score_group(g, 3, plan, libs).top_score == pytest.approx(0.0)
    plan, libs = _plan("all but 1")
    res = score_group(g, 3, plan, libs)
    assert res.found and res.top_score == pytest.approx(1.0)


def test_same_minute_ties_are_file_order_free() -> None:
    # F1: same-minute edges are simultaneous — swapping two tied rows in
    # the file must not change the score. The member predicate is scored
    # over every tied edge and the best reading wins (E6's rule).
    doc = parse_text(
        "motif t { match (s:account) -[leg1:transfer]-> (m:account)"
        " -[leg2:transfer]-> (d:account)"
        " group by s, d over m window 2d"
        " where leg2.time > leg1.time and s != d"
        " having count(distinct m) >= 2"
        " soft most m: leg2.amount ~= leg1.amount within 15%"
        " score min emit when score >= 0.5 }")
    plan, libs = compile_motif(doc, "t"), libs_from_doc(doc)
    base = [(0, "001", "S", "002", "M1", 1000.0), (1, "001", "S", "002", "M0", 1000.0),
            (10, "002", "M1", "003", "D", 100.0), (10, "002", "M1", "003", "D", 1000.0),
            (20, "002", "M0", "003", "D", 1000.0)]
    swapped = [base[0], base[1], base[3], base[2], base[4]]
    a = score_group(TemporalGraph.from_transfers(_frame(base)), 4, plan, libs)
    b = score_group(TemporalGraph.from_transfers(_frame(swapped)), 4, plan, libs)
    assert a.top_score == b.top_score == pytest.approx(1.0)
    assert a.found and b.found
    # Same rule on the fan side: R2's tied 50/100 readings both count.
    fdoc = parse_text(
        "motif t { match (s:account) -[e:transfer]-> (r:account)"
        " group by s over r window 1d"
        " having count(distinct r) >= 2"
        " soft most r: e.amount ~= 100 USD within 10 USD"
        " score min emit when score >= 0.5 }")
    fplan, flibs = compile_motif(fdoc, "t"), libs_from_doc(fdoc)
    fbase = [(0, "001", "S", "002", "R1", 100.0),
             (10, "001", "S", "002", "R2", 50.0), (10, "001", "S", "002", "R2", 100.0),
             (20, "001", "S", "002", "R3", 100.0)]
    fswapped = [fbase[0], fbase[2], fbase[1], fbase[3]]
    fa = score_group(TemporalGraph.from_transfers(_frame(fbase)), 3, fplan, flibs)
    fb = score_group(TemporalGraph.from_transfers(_frame(fswapped)), 3, fplan, flibs)
    assert fa.top_score == fb.top_score == pytest.approx(1.0)
