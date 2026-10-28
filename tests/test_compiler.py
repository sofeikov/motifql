"""Tests for the MotifQL parser and static checks (Oct 29, compiler steps 1-2).

Positive pins: the reference file parses to the expected AST and passes
every check. Negative pins: one crafted motif per error code, each failing
only its own check.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from motifql import checks  # noqa: E402
from motifql.parser import parse_text  # noqa: E402
from motifql.planner import PlanError, compile_motif, estimate_cost  # noqa: E402

HEADER = "term soon = trapezoid(0h, 0h, 6h, 48h)\n"


def _codes(source: str) -> list[str]:
    doc = parse_text(HEADER + source)
    return [e.code for e in checks.check_document(doc)]


def test_reference_file_parses_and_passes() -> None:
    doc = parse_text((Path("examples/amlworld.motif")).read_text())
    assert len(doc.motifs) == 8
    assert checks.check_document(doc) == []
    rt = next(m for m in doc.motifs if m.name == "round_trip")
    assert rt.line == 29  # shifts when shared vocabulary above gains lines
    (pat,) = rt.patterns
    assert [n.var for n in pat.nodes] == ["a", "a"]
    (edge,) = pat.edges
    assert (edge.var, edge.direction, edge.min_hops, edge.max_hops) == ("e", "out", 2, 5)
    assert rt.score == "min" and rt.emit == 0.6


def test_group_windows_parse() -> None:
    doc = parse_text((Path("examples/amlworld.motif")).read_text())
    by_name = {m.name: m for m in doc.motifs}
    assert by_name["fan_out"].group.window_s == 4 * 86400.0
    assert by_name["scatter_gather"].group.window_s == 48 * 3600.0
    assert by_name["scatter_gather"].group.keys == ("s", "d")


def test_time_bound_required() -> None:
    assert "E_TIME" in _codes(
        "motif t { match (a:account) -[e:transfer*2..5]-> (a) }")
    assert "E_TIME" not in _codes(
        "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
        " where span(e) <= 7d }")


def test_unit_mismatch_rejected() -> None:
    assert "E_UNIT" in _codes(
        "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
        " where span(e) <= 7d and e.amount == e.time }")
    assert "E_UNIT" in _codes(
        "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
        " where span(e) <= 7d"
        " soft e.amount ~= e.time within 10% }")


def test_tolerance_must_fit() -> None:
    bad = _codes(
        "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
        " where span(e) <= 7d"
        " soft e.amount ~= e.amount within 30m }")
    assert "E_UNIT" in bad
    good = _codes(
        "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
        " where span(e) <= 7d"
        " soft e.amount ~= e.amount within 10% }")
    assert good == []


def test_fuzzy_stays_out_of_where() -> None:
    assert "E_CRISP" in _codes(
        "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
        " where span(e) <= 7d and e.amount ~= e.amount within 10% }")
    # at least N is crisp and belongs in where.
    assert _codes(
        "motif t { match (a:account) -[e:transfer]-> (b:account)"
        " group by a over b window 1d"
        " having count(distinct b) >= 2"
        " where span(all) <= 1d }") == []


def test_scoping_errors() -> None:
    assert "E_SCOPE" in _codes(  # unbound variable
        "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
        " where span(e) <= 7d and zzz == a }")
    assert "E_SCOPE" in _codes(  # having without group
        "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
        " where span(e) <= 7d"
        " having count(distinct a) >= 2 }")
    assert "E_SCOPE" in _codes(  # aggregate outside a group
        "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
        " where span(e) <= 7d and count(distinct a) >= 1 }")
    assert "E_SCOPE" in _codes(  # unknown term
        "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
        " where span(e) <= 7d"
        " soft each e: e.gap is nosuchterm }")


def test_unquantified_step_term_rejected() -> None:
    # A bare `e.amount is high` over a path grades no step: E_PLAN at
    # compile time, not a crash (or silent value) at score time. The term
    # is money-domain so the H4 unit check passes and this still pins the
    # unquantified-IsTerm rejection (not E_CHECK).
    with pytest.raises(PlanError, match="E_PLAN"):
        compile_motif(parse_text(HEADER
            + "term high = ramp(900 USD, 1000 USD)\n"
            + "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
            + " where e is time_ordered and span(e) <= 7d"
            + " soft e.amount is high }"), "t")


def test_time_ordered_required_on_path_motifs() -> None:
    # The walk enforces time order structurally, so a motif that does not
    # say so would mean more than it says.
    with pytest.raises(PlanError, match="E_PLAN"):
        compile_motif(parse_text(HEADER
            + "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
            + " where span(e) <= 7d }"), "t")


def test_strict_span_bound_rejected() -> None:
    # The walk applies the span bound inclusively; `<` would promise a
    # strictness the executors cannot express.
    with pytest.raises(PlanError, match="E_PLAN"):
        compile_motif(parse_text(HEADER
            + "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
            + " where e is time_ordered and span(e) < 7d }"), "t")
    with pytest.raises(PlanError, match="E_PLAN"):
        compile_motif(parse_text(HEADER
            + "motif t { match (x1:account) -[e1:transfer]-> (y1:account),"
            + " (x1) -[e2:transfer]-> (y2:account),"
            + " (x2:account) -[e3:transfer]-> (y1),"
            + " (x2) -[e4:transfer]-> (y2)"
            + " where x1 != x2 and y1 != y2 and span(all) < 1d }"), "t")


def test_prev_outside_path_quantifier_rejected() -> None:
    assert "E_SCOPE" in _codes(
        "motif t { match (a:account) -[e:transfer]-> (b:account)"
        " where span(all) <= 1d and prev.amount == e.amount }")


def _doc():
    return parse_text(HEADER + (Path("examples/amlworld.motif")).read_text())


def test_round_trip_plan() -> None:
    plan = compile_motif(_doc(), "round_trip")
    assert plan.closed and (plan.min_hops, plan.max_hops) == (2, 5)
    assert plan.window_s == 7 * 86400
    assert [g.label for g in plan.prunable] == [
        "each e: gap is patient",
        "each e: amount ~= prev.amount within 20%",
    ]
    assert plan.deferred is None
    assert plan.score_tnorm == "min" and plan.emit == 0.6
    assert plan.est_expansions == -1.0  # no stats: unwired


def test_layering_chain_plan_is_open() -> None:
    plan = compile_motif(_doc(), "layering_chain")
    assert not plan.closed and (plan.min_hops, plan.max_hops) == (3, 6)
    assert len(plan.prunable) == 2


def test_group_and_butterfly_now_plan() -> None:
    # Oct 30: group motifs and the 2x2 core compile; other shapes still E_PLAN.
    from motifql.planner import ButterflyPlan, GroupPlan
    assert isinstance(compile_motif(_doc(), "fan_out"), GroupPlan)
    assert isinstance(compile_motif(_doc(), "bipartite_butterfly"), ButterflyPlan)
    with pytest.raises(PlanError, match="E_PLAN"):
        compile_motif(parse_text(HEADER + "motif t { match (a:account) -[e:transfer*2..5]-> (a),"
                                 " (b:account) -[f:transfer]-> (c:account)"
                                 " where span(all) <= 1d }"), "t")


def test_where_only_carries_what_the_walk_does() -> None:
    base = "motif t { match (a:account) -[e:transfer*2..5]-> (a) "
    with pytest.raises(PlanError, match="E_PLAN"):
        compile_motif(parse_text(HEADER + base
            + "where e is time_ordered and span(e) <= 7d and a.bank != a.bank }"), "t")
    with pytest.raises(PlanError, match="E_PLAN"):
        compile_motif(parse_text(HEADER + base
            + "where e is time_ordered and span(e) <= 7d"
            + " and each e: e.amount >= 1000000 USD }"), "t")


def test_in_edges_rejected() -> None:
    with pytest.raises(PlanError, match="E_PLAN"):
        compile_motif(parse_text(HEADER
            + "motif t { match (a:account) <-[e:transfer*2..5]- (a)"
            + " where e is time_ordered and span(e) <= 7d }"), "t")


def test_soft_field_must_be_carried() -> None:
    with pytest.raises(PlanError, match="E_PLAN"):
        compile_motif(parse_text(HEADER
            + "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
            + " where e is time_ordered and span(e) <= 7d"
            + " soft each e: e.amount_paid ~= prev.amount_paid within 5% }"), "t")


def test_gap_consistency_stays_deferred() -> None:
    # H1: only amount-vs-prev.amount is prunable. Gap/time consistency
    # must not be miscompiled into AmountConsCond (wrong scores plus an
    # unsound pruning bound); it stays deferred, where per-step eval
    # handles the true fields.
    plan = compile_motif(parse_text(HEADER
        + "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
        + " where e is time_ordered and span(e) <= 7d"
        + " soft each e: e.gap ~= prev.gap within 20% }"), "t")
    assert plan.prunable == ()
    assert plan.deferred is not None


def test_multiple_span_bounds_take_minimum() -> None:
    # H2: every conjunct must hold, so the tightest bound wins, in
    # either textual order.
    for where in ("span(e) <= 7d and span(e) <= 1d",
                  "span(e) <= 1d and span(e) <= 7d"):
        plan = compile_motif(parse_text(HEADER
            + "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
            + f" where e is time_ordered and {where} }}"), "t")
        assert plan.window_s == 86400


def test_duplicate_clauses_rejected() -> None:
    # H3: clauses after `match` may appear at most once each (LANGUAGE
    # §4); a repeat must fail loudly instead of silently last-winning.
    from lark.exceptions import VisitError
    with pytest.raises(VisitError, match="duplicate `soft`"):
        parse_text(HEADER
            + "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
            + " where span(e) <= 7d"
            + " soft each e: e.gap is soon"
            + " soft each e: e.gap is soon }")
    with pytest.raises(VisitError, match="duplicate `where`"):
        parse_text(HEADER
            + "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
            + " where span(e) <= 7d"
            + " where span(e) <= 1d }")


def test_term_domain_mismatch_rejected() -> None:
    # H4: LANGUAGE.md §5 item 2 — a duration term on money is E_UNIT.
    assert "E_UNIT" in _codes(
        "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
        " where span(e) <= 7d"
        " soft each e: e.amount is soon }")
    assert "E_UNIT" in _codes(
        "term several = ramp(1, 3)\n"
        "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
        " where span(e) <= 7d"
        " soft each e: e.amount is several }")
    # Matching domains still pass: duration term on gap, money term on
    # amount (parameterised, so the domain comes from the call-site arg).
    assert _codes(
        "term high = ramp(900 USD, 1000 USD)\n"
        "term just_below(limit) = trapezoid(0.85 * limit, 0.95 * limit, 0.999 * limit, limit)\n"
        "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
        " where span(e) <= 7d"
        " soft each e: e.gap is soon"
        " and each e: e.amount is high"
        " and each e: e.amount is just_below(10000 USD) }") == []


def test_group_legs_must_be_single_hop() -> None:
    # H5: scatter/gather legs accept `*m..n` in the grammar but the plan
    # has no hop fields and the executor does single-edge lookups, so a
    # multi-hop leg must E_PLAN like fan and butterfly already do.
    with pytest.raises(PlanError, match="E_PLAN"):
        compile_motif(parse_text(HEADER
            + "motif t { match (s:account) -[leg1:transfer*1..3]-> (m:account)"
            + " -[leg2:transfer]-> (d:account)"
            + " group by s, d over m window 2d"
            + " where leg2.time > leg1.time and s != d"
            + " having count(distinct m) >= 2 }"), "t")
    with pytest.raises(PlanError, match="E_PLAN"):
        compile_motif(parse_text(HEADER
            + "motif t { match (x:account) -[g:transfer]-> (c:account),"
            + " (c) -[p:transfer*1..2]-> (y:account)"
            + " group by c over x, y window 4d"
            + " where p.time > g.time"
            + " having count(distinct x) >= 2 and count(distinct y) >= 2 }"), "t")


def test_unquantified_step_comparison_rejected() -> None:
    # M1: an unquantified step-field read has no step to read from — the
    # evaluator aborts — so it is E_PLAN for comparisons and ~= alike,
    # not just for `is <term>`.
    for soft in ("e.amount >= 100 USD", "e.amount ~= e.amount within 10%"):
        with pytest.raises(PlanError, match="E_PLAN"):
            compile_motif(parse_text(HEADER
                + "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
                + " where e is time_ordered and span(e) <= 7d"
                + f" soft {soft} }}"), "t")


def test_single_hop_paths_rejected() -> None:
    # M3: the walk only enumerates witnesses of 2+ hops, so `*1..n`
    # must E_PLAN instead of silently under-firing.
    with pytest.raises(PlanError, match="E_PLAN"):
        compile_motif(parse_text(HEADER
            + "motif t { match (a:account) -[e:transfer*1..5]-> (a)"
            + " where e is time_ordered and span(e) <= 7d }"), "t")


def test_arithmetic_call_site_arg_accepted() -> None:
    # M2: arithmetic is its own AST node — `0.5 * 10000 USD` is money
    # (scaling), not a bool-vs-money comparison error, and the call-site
    # value binds into the term breakpoints.
    from motifql.degrees import libs_from_doc, resolve_term_args
    text = (
        "term just_below(limit) = trapezoid(0.85 * limit, 0.95 * limit, 0.999 * limit, limit)\n"
        "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
        " where e is time_ordered and span(e) <= 7d"
        " soft each e: e.amount is just_below(0.5 * 10000 USD) }")
    assert _codes(text) == []
    doc = parse_text(HEADER + text)
    plan = compile_motif(doc, "t")
    assert plan.deferred is not None
    terms, _ = libs_from_doc(doc)
    use_args = doc.motifs[0].soft.body.args
    assert resolve_term_args("just_below", use_args, terms) == (4250.0, 4750.0, 4995.0, 5000.0)


def test_mixed_unit_arithmetic_rejected() -> None:
    # M2: money mixed with a duration in arithmetic is a genuine E_UNIT
    # (caught where the old node-conflation could only misfire).
    assert "E_UNIT" in _codes(
        "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
        " where span(e) <= 7d"
        " soft each e: e.amount + e.gap is soon }")


def test_cost_budget_rejects_huge_hops() -> None:
    dense = np.zeros(30)
    dense[20] = 100.0  # mean in-degree 10: *2..5 estimates ~1e5, passes
    plan = compile_motif(_doc(), "round_trip", stats=dense)
    assert plan.est_expansions < 10_000_000
    huge = np.zeros(200)
    huge[100] = 100.0  # mean in-degree 50: *2..5 estimates ~3e8
    with pytest.raises(PlanError) as exc:
        compile_motif(_doc(), "round_trip", stats=huge)
    assert "E_COST" in str(exc.value)
