"""Tests for crisp (exact) scoring mode (Nov 1, RQ2).

Rule under test: with crisp=True every soft leaf degree (term use,
`~=` tolerance) is thresholded at the support boundary (degree > 0 ->
1.0, else 0.0); combination (and/or/not/quantifiers over the leaves)
is unchanged. So on motifs without most/few, crisp score ==
(fuzzy top_score > 0), and pruning stays sound (pruned crisp ==
unpruned crisp — the bound must use crisp degrees, since a branch
with fuzzy 0.5 < emit can still be crisp 1.0).

Issue #15 finding A: the support boundary is the loosest exact rule
the motif allows, so `crisp_alpha` generalises the leaves to α-cuts
(degree >= α -> 1.0). For `min` motifs the α-cut score equals
(fuzzy top_score >= α), which is why the RQ2 exact arm must run at a
calibrated α (the motif's emit), not at the support.
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


def _neg_pair():
    leaf = parse_text(
        HEADER
        + "motif t { match (a:account) -[e:transfer*2..2]-> (a)"
        + " where e is time_ordered and span(e) <= 7d"
        + " soft each e: e.gap is not soon"
        + " score min emit when score >= 0.5 }")
    expr = parse_text(
        HEADER
        + "motif t { match (a:account) -[e:transfer*2..2]-> (a)"
        + " where e is time_ordered and span(e) <= 7d"
        + " soft not each e: e.gap is soon"
        + " score min emit when score >= 0.5 }")
    return leaf, expr


def test_crisp_negation_is_strong_and_consistent() -> None:
    # Review #14.6: `is not T` ≡ `not (is T)` under crisp, with
    # strong negation (crisp 1 iff the positive degree is exactly 0).
    from motifql.degrees import libs_from_doc as _libs
    leaf_doc, expr_doc = _neg_pair()
    lp, ep = compile_motif(leaf_doc, "t"), compile_motif(expr_doc, "t")
    assert ep.deferred is not None  # expression-Not goes through eval_soft
    libs = _libs(leaf_doc)
    cases = [
        (27 * 60, 0.0),   # soon 0.5 -> strong-not 0
        (50 * 60, 1.0),   # soon 0.0 -> strong-not 1
        (3 * 60, 0.0),    # soon 1.0 -> strong-not 0
    ]
    for gap_min, want in cases:
        g = TemporalGraph.from_transfers(_frame([
            (0, "001", "B", "002", "A", 100.0),
            (gap_min, "002", "A", "001", "B", 100.0),
        ]))
        a = score(g, 1, lp, libs, prune=False, crisp=True)
        b = score(g, 1, ep, libs, prune=False, crisp=True)
        assert a.top_score == pytest.approx(want)
        assert b.top_score == pytest.approx(want)
        fz_a = score(g, 1, lp, libs, prune=False)
        fz_b = score(g, 1, ep, libs, prune=False)
        assert fz_a.top_score == pytest.approx(fz_b.top_score)


def test_crisp_deferred_some() -> None:
    # Pins crisping inside the deferred path evaluator (eval_soft):
    # `some` cannot prune (a partial max is a lower bound), so it is
    # always deferred.
    doc = parse_text(
        HEADER
        + "motif t { match (a:account) -[e:transfer*2..2]-> (a)"
        + " where e is time_ordered and span(e) <= 7d"
        + " soft some e: e.gap is soon"
        + " score min emit when score >= 0.6 }")
    plan, libs = compile_motif(doc, "t"), libs_from_doc(doc)
    assert plan.deferred is not None
    g = TemporalGraph.from_transfers(_frame([
        (0, "001", "B", "002", "A", 100.0),
        (27 * 60, "002", "A", "001", "B", 100.0),
    ]))
    fuzzy = score(g, 1, plan, libs, prune=False)
    assert fuzzy.top_score == pytest.approx(0.5)
    exact = score(g, 1, plan, libs, prune=False, crisp=True)
    assert exact.top_score == pytest.approx(1.0)


def test_crisp_deferred_span_leaves() -> None:
    # Pins crisping of span-level deferred leaves (degrees._eval_term
    # and the eval_soft Approx branch): 27h span, soon reads 0.5.
    for soft, want in (("span(e) is soon", 0.5),
                       ("span(e) ~= 24h within 12h", 0.75)):
        doc = parse_text(
            HEADER
            + "motif t { match (a:account) -[e:transfer*2..2]-> (a)"
            + " where e is time_ordered and span(e) <= 7d"
            + f" soft {soft}"
            + " score min emit when score >= 0.6 }")
        plan, libs = compile_motif(doc, "t"), libs_from_doc(doc)
        assert plan.deferred is not None
        g = TemporalGraph.from_transfers(_frame([
            (0, "001", "B", "002", "A", 100.0),
            (27 * 60, "002", "A", "001", "B", 100.0),
        ]))
        fuzzy = score(g, 1, plan, libs, prune=False)
        assert fuzzy.top_score == pytest.approx(want)
        exact = score(g, 1, plan, libs, prune=False, crisp=True)
        assert exact.top_score == pytest.approx(1.0)


def test_crisp_scatter_sum_balance() -> None:
    # Pins crisping of group-mirror Approx leaves (sums balance 0.35).
    doc = parse_text(
        "term several = ramp(1, 3)\n"
        "motif t { match (s:account) -[leg1:transfer]-> (m:account)"
        " -[leg2:transfer]-> (d:account)"
        " group by s, d over m window 2d"
        " where leg2.time > leg1.time and s != d"
        " having count(distinct m) >= 2"
        " soft count(distinct m) is several"
        " and sum(leg2.amount) ~= sum(leg1.amount) within 20%"
        " score min emit when score >= 0.5 }")
    plan, libs = compile_motif(doc, "t"), libs_from_doc(doc)
    g = TemporalGraph.from_transfers(_frame([
        (10, "001", "S", "002", "M1", 100.0),
        (20, "002", "M1", "003", "D", 100.0),
        (12, "001", "S", "002", "M2", 100.0),
        (25, "002", "M2", "003", "D", 130.0),
    ]))
    fuzzy = score_group(g, 3, plan, libs)
    assert fuzzy.top_score == pytest.approx(1 - 30 / 46, abs=1e-9)
    exact = score_group(g, 3, plan, libs, crisp=True)
    assert exact.top_score == pytest.approx(1.0)


def test_crisp_safety_predicates() -> None:
    from motifql.degrees import crisp_binary_safe as bsafe
    from motifql.degrees import crisp_differential_safe as dsafe
    from motifql.parser import Comparison, Not, Num, Quantified, Ref
    leaf = Comparison(">=", Ref(("e", "amount")), Num(1.0))
    assert bsafe(leaf) and dsafe(leaf)
    assert bsafe(Not(leaf)) and not dsafe(Not(leaf))
    for q in ("each", "all", "some", "at_least:2", "all_but:1"):
        n = Quantified(q, "e", leaf)
        assert bsafe(n) and dsafe(n), q
    for q in ("most", "few", "named:mostly"):
        n = Quantified(q, "e", leaf)
        assert not bsafe(n) and not dsafe(n), q
    assert not bsafe(object()) and not dsafe(object())
    # Plan level: reference motifs without most/few/not are in scope.
    doc = parse_text(Path("examples/amlworld.motif").read_text())
    for name in ("round_trip", "layering_chain", "fan_out", "fan_in",
                 "scatter_gather", "gather_scatter", "bipartite_butterfly"):
        plan = compile_motif(doc, name)
        soft = getattr(plan, "soft", None)
        if soft is None and hasattr(plan, "deferred"):
            parts = [a for grp in plan.prunable for a in grp.atoms]
            if plan.deferred is not None:
                parts.append(plan.deferred)
            assert all(dsafe(p) for p in parts), name
        else:
            assert soft is not None and dsafe(soft), name
    structuring = compile_motif(doc, "structuring_fan_in")
    assert not bsafe(structuring.soft) and not dsafe(structuring.soft)


def test_crisp_degree_alpha_table() -> None:
    assert crisp_degree(0.49, True, 0.5) == 0.0
    assert crisp_degree(0.5, True, 0.5) == 1.0  # α-cut is >=, support is >
    assert crisp_degree(0.6, True, 0.5) == 1.0
    assert crisp_degree(0.001, True, 0.0) == 1.0  # α=0 keeps the support
    assert crisp_degree(0.4, False, 0.5) == 0.4  # flag off: identity
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            crisp_degree(0.5, True, bad)


def test_path_alpha_cut_is_fuzzy_threshold() -> None:
    # min motif: per-leaf α-cuts ANDed == (min of leaves >= α), so the
    # α-cut top_score must equal (fuzzy top_score >= α) on every anchor.
    _, plan, libs = _round_trip()
    rows = [
        (0, "001", "B", "002", "A", 100.0),
        (27 * 60, "002", "A", "001", "B", 100.0),  # gap soon 0.5
        (57 * 60, "001", "B", "002", "A", 100.0),  # gap soon ~0.43
    ]
    g = TemporalGraph.from_transfers(_frame(rows))
    for pos in (1, 2):
        fuzzy = score(g, pos, plan, libs, prune=False).top_score
        for prune in (False, True):
            cut = score(g, pos, plan, libs, prune=prune,
                        crisp=True, crisp_alpha=0.5).top_score
            assert cut == pytest.approx(1.0 if fuzzy >= 0.5 else 0.0)
    # Pruned and unpruned α-cuts agree exactly (soundness at α > 0).
    g2 = TemporalGraph.from_transfers(_frame([
        (0, "001", "B", "002", "A", 100.0),
        (60, "003", "C", "002", "A", 10000.0),  # decoy: α-cut-0 amount
        (27 * 60, "002", "A", "001", "B", 100.0),
    ]))
    pruned = score(g2, 2, plan, libs, prune=True, crisp=True, crisp_alpha=0.5)
    plain = score(g2, 2, plan, libs, prune=False, crisp=True, crisp_alpha=0.5)
    assert (plain.found, plain.score, plain.witness, plain.top_score) == (
        pruned.found, pruned.score, pruned.witness, pruned.top_score)


def test_product_alpha_cut_differs_from_posthoc() -> None:
    # Why the α-cut lives in the engine, not in post-processing: under
    # `product`, per-leaf cuts (1 AND 1) differ from (fuzzy >= α).
    # several(2) = 0.5, sum-approx = 0.6 -> fuzzy 0.3 < 0.5, cut 1.0.
    doc = parse_text(
        "term several = ramp(1, 3)\n"
        "motif t { match (s:account) -[e:transfer]-> (r:account)"
        " group by s over r window 1d"
        " having count(distinct r) >= 2"
        " soft count(distinct r) is several"
        " and sum(e.amount) ~= 400 USD within 50%"
        " score product emit when score >= 0.5 }")
    plan, libs = compile_motif(doc, "t"), libs_from_doc(doc)
    g = TemporalGraph.from_transfers(_frame([
        (0, "001", "S", "002", "R0", 150.0),
        (30, "001", "S", "002", "R1", 170.0),
    ]))
    fuzzy = score_group(g, 1, plan, libs)
    assert fuzzy.top_score == pytest.approx(0.3)
    cut = score_group(g, 1, plan, libs, crisp=True, crisp_alpha=0.5)
    assert cut.top_score == pytest.approx(1.0)
    assert (fuzzy.top_score >= 0.5) != (cut.top_score > 0)


def test_alpha_cut_negation() -> None:
    # Strong negation at α: `is not T` ≡ `not (is T)` and reads 1 iff
    # the positive degree is below α.
    from motifql.degrees import libs_from_doc as _libs
    leaf_doc, expr_doc = _neg_pair()
    lp, ep = compile_motif(leaf_doc, "t"), compile_motif(expr_doc, "t")
    libs = _libs(leaf_doc)
    for gap_min, want in ((27 * 60, 0.0), (30 * 60, 1.0), (3 * 60, 0.0)):
        g = TemporalGraph.from_transfers(_frame([
            (0, "001", "B", "002", "A", 100.0),
            (gap_min, "002", "A", "001", "B", 100.0),
        ]))
        for plan in (lp, ep):
            got = score(g, 1, plan, libs, prune=False,
                        crisp=True, crisp_alpha=0.5).top_score
            assert got == pytest.approx(want)


def test_alpha_differential_predicate() -> None:
    from motifql.degrees import crisp_differential_safe as dsafe
    from motifql.parser import Comparison, Not, Num, Quantified, Ref
    leaf = Comparison(">=", Ref(("e", "amount")), Num(1.0))
    assert dsafe(leaf, alpha=0.5, tnorm="min")
    assert not dsafe(leaf, alpha=0.5, tnorm="product")
    assert not dsafe(Not(leaf), alpha=0.5, tnorm="min")
    assert not dsafe(Quantified("most", "e", leaf), alpha=0.5, tnorm="min")
    assert dsafe(Quantified("each", "e", leaf), alpha=0.5, tnorm="min")
    assert dsafe(leaf)  # α=0 default keeps the old support rule
