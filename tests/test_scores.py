"""Tests for fuzzy scoring with pruning (Oct 29, general path matcher).

Soundness contract under test: pruning may skip work but never changes the
emit-filtered answer — pruned and unpruned runs agree on found/witness/score
(only expansions and sub-emit top_score may differ, and then only down).
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from motifql import degrees as D  # noqa: E402
from motifql.degrees import libs_from_doc  # noqa: E402
from motifql.graph import TemporalGraph  # noqa: E402
from motifql.matcher import EXACT_FLOOR, score  # noqa: E402
from motifql.parser import Pct, parse_text  # noqa: E402
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


def test_term_shapes() -> None:
    assert D.term_degree("trapezoid", (0.0, 0.0, 6.0, 48.0), 3.0) == 1.0
    assert D.term_degree("trapezoid", (0.0, 0.0, 6.0, 48.0), 48.0) == 0.0
    assert D.term_degree("trapezoid", (0.0, 0.0, 6.0, 48.0), 27.0) == pytest.approx(0.5)
    assert D.term_degree("ramp", (2.0, 5.0), 2.0) == 0.0
    assert D.term_degree("ramp", (2.0, 5.0), 5.0) == 1.0
    # F3: degenerate shoulders are closed — at x == a == b the value is 1,
    # matching "0 below a, rises linearly to 1 at b" (LANGUAGE.md 3.1).
    assert D.term_degree("trapezoid", (2.0, 2.0, 4.0, 8.0), 2.0) == 1.0
    assert D.term_degree("trapezoid", (2.0, 2.0, 4.0, 8.0), 1.99) == 0.0
    assert D.term_degree("trapezoid", (0.0, 0.0, 6.0, 48.0), 0.0) == 1.0
    assert D.term_degree("trapezoid", (0.0, 2.0, 6.0, 6.0), 6.0) == 1.0
    assert D.term_degree("triangle", (1.0, 1.0, 2.0), 1.0) == 1.0
    # tolerance is relative to the larger side: 1 - 10/(0.2*110) = 12/22.
    assert D.tol_degree(Pct(0.2), 100.0, 110.0) == pytest.approx(12 / 22)


def test_perfect_witness_scores_one() -> None:
    g = TemporalGraph.from_transfers(
        _frame(
            [
                (0, "001", "A", "002", "B", 100.0),
                (10, "002", "B", "003", "C", 100.0),
                (20, "003", "C", "001", "A", 100.0),
            ]
        )
    )
    _, plan, libs = _round_trip()
    res = score(g, 2, plan, libs)
    assert res.found and res.score == pytest.approx(1.0)
    assert [int(g.transfer_id[e]) for e in res.witness] == [0, 1, 2]
    assert [label for label, _, _ in res.degrees] == [
        "each e: gap is soon",
        "each e: amount ~= prev.amount within 20%",
    ]


def test_per_step_arithmetic_evaluates() -> None:
    # M2: arithmetic over step fields evaluates per step in deferred
    # predicates (previously a loud E_UNIT, never silent).
    doc = parse_text(
        HEADER
        + "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
        + " where e is time_ordered and span(e) <= 7d"
        + " soft each e: e.amount * 2 >= 150 USD"
        + " score min emit when score >= 0.6 }"
    )
    plan = compile_motif(doc, "t")
    assert plan.prunable == () and plan.deferred is not None
    libs = libs_from_doc(doc)
    hot = TemporalGraph.from_transfers(_frame([
        (0, "001", "A", "002", "B", 100.0),
        (10, "002", "B", "001", "A", 100.0),
    ]))
    assert score(hot, 1, plan, libs).found
    cold = TemporalGraph.from_transfers(_frame([
        (0, "001", "A", "002", "B", 50.0),
        (10, "002", "B", "001", "A", 50.0),
    ]))
    assert not score(cold, 1, plan, libs).found


def test_compound_body_skips_first_step() -> None:
    # E5: a quantifier body whose atoms all skip on the first step is
    # itself skipped — previously `some` over an `and` always scored 1.0.
    doc = parse_text(
        HEADER
        + "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
        + " where e is time_ordered and span(e) <= 7d"
        + " soft some e: (e.gap is soon and e.amount ~= prev.amount within 10%)"
        + " score min emit when score >= 0.6 }"
    )
    plan = compile_motif(doc, "t")
    libs = libs_from_doc(doc)
    g = TemporalGraph.from_transfers(_frame([
        (0, "001", "A", "002", "B", 100.0),
        (10, "002", "B", "001", "A", 10000.0),
    ]))
    res = score(g, 1, plan, libs)
    assert not res.found and res.top_score == 0.0


def test_skip_propagates_through_operators() -> None:
    # E5: all-skipped `and`/`or` evaluate to None (skip), not 1.0/0.0,
    # so `not` and the quantifier propagate the skip.
    from motifql.parser import And, Approx, IsTerm, Not, Or, Ref
    gap = IsTerm(Ref(("e", "gap")), False, "soon", ())
    amt = Approx(Ref(("e", "amount")), Ref(("prev", "amount")), Pct(0.1))
    libs = libs_from_doc(parse_text(HEADER))
    vals = {"time": 1.0, "amount": 100.0, "gap": None}  # first step
    assert D._eval_step(And((gap, amt)), vals, None, libs, "min") is None
    assert D._eval_step(Or((gap, amt)), vals, None, libs, "min") is None
    assert D._eval_step(Not(And((gap, amt))), vals, None, libs, "min") is None


def test_prune_agrees_with_unpruned_on_random_graphs() -> None:
    rng = np.random.default_rng(9)
    _, plan, libs = _round_trip()
    for trial in range(15):
        accts = [f"N{i}" for i in range(6)]
        rows = [
            (
                int(rng.integers(0, 200)), "001", str(rng.choice(accts)),
                "001", str(rng.choice(accts)), float(rng.uniform(50, 150)),
            )
            for _ in range(14)
        ]
        g = TemporalGraph.from_transfers(_frame(rows))
        for anchor in range(g.n_edges):
            pruned = score(g, anchor, plan, libs, prune=True)
            raw = score(g, anchor, plan, libs, prune=False)
            assert (pruned.found, pruned.witness, pruned.score) == (
                raw.found, raw.witness, raw.score)
            assert pruned.top_score <= raw.top_score + 1e-9
            assert pruned.expansions <= raw.expansions
            if raw.top_score >= plan.emit:
                assert pruned.top_score == pytest.approx(raw.top_score)


def _product_round_trip():
    # One `each` per line, the reference-file convention: `and` binds
    # looser than the quantifier body, so a single `each e: A and B` would
    # mean `(each e: A) and B` and leave `prev` unbound.
    doc = parse_text(
        HEADER
        + "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
        + " where e is time_ordered and span(e) <= 7d"
        + " soft each e: e.gap is soon and each e: e.amount ~= prev.amount within 20%"
        + " score product emit when score >= 0.1 }")
    return doc, compile_motif(doc, "t"), libs_from_doc(doc)


def test_product_combines_atoms_per_step() -> None:
    # 27h gap -> soon = (48-27)/42 = 0.5; 100 -> 110 within 20% -> 1-10/22.
    # min t-norm would say 0.5; product says 0.5 * 6/11 = 3/11.
    g = TemporalGraph.from_transfers(
        _frame(
            [
                (0, "001", "A", "002", "B", 100.0),
                (27 * 60, "002", "B", "001", "A", 110.0),
            ]
        )
    )
    _, plan, libs = _product_round_trip()
    assert plan.score_tnorm == "product"
    res = score(g, 1, plan, libs)
    assert res.found and res.score == pytest.approx(0.5 * (1 - 10 / 22))


def test_some_omits_the_undefined_first_step() -> None:
    # All defined comparisons fail: `some` must score 0, not 1 from the
    # skipped first step.
    doc = parse_text(
        HEADER
        + "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
        + " where e is time_ordered and span(e) <= 7d"
        + " soft some e: e.amount ~= prev.amount within 0%"
        + " score min emit when score >= 0.1 }")
    plan = compile_motif(doc, "t")
    libs = libs_from_doc(doc)
    g = TemporalGraph.from_transfers(
        _frame(
            [
                (0, "001", "A", "002", "B", 100.0),
                (10, "002", "B", "003", "C", 200.0),
                (20, "003", "C", "001", "A", 300.0),
            ]
        )
    )
    res = score(g, 2, plan, libs)
    assert res.top_score == 0.0


def test_prev_gap_reads_the_previous_step_gap() -> None:
    # Review repro: 3-step cycle with gaps 1h then 9h, `each e: e.gap is
    # soon and e.gap < prev.gap`. Step 3 reads prev.gap = 1h against its
    # own 9h, so the comparison is 0 and the witness scores 0 — the step
    # used to be skipped outright (prev.gap was always undefined),
    # leaving the vacuous 1.0.
    doc = parse_text(
        HEADER
        + "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
        + " where e is time_ordered and span(e) <= 7d"
        + " soft each e: (e.gap is soon and e.gap < prev.gap)"
        + " score min emit when score >= 0.1 }")
    plan = compile_motif(doc, "t")
    libs = libs_from_doc(doc)
    g = TemporalGraph.from_transfers(
        _frame(
            [
                (0, "001", "A", "002", "B", 100.0),
                (60, "002", "B", "003", "C", 100.0),
                (600, "003", "C", "001", "A", 100.0),
            ]
        )
    )
    res = score(g, 2, plan, libs)
    assert res.top_score == 0.0
    # Reversed gaps (9h then 1h): step 3 reads prev.gap = 9h, 1h < 9h
    # holds, and `soon` gives 0.93 on the 9h gap, 1.0 on the 1h gap.
    g2 = TemporalGraph.from_transfers(
        _frame(
            [
                (0, "001", "A", "002", "B", 100.0),
                (540, "002", "B", "003", "C", 100.0),
                (600, "003", "C", "001", "A", 100.0),
            ]
        )
    )
    res2 = score(g2, 2, plan, libs)
    assert res2.found and res2.score == pytest.approx((172800 - 32400) / 151200)


def test_unquantified_span_term_scores() -> None:
    # `soft span(e) is soon` grades the whole witness, no quantifier
    # needed. This used to crash the evaluator (missing tnorm argument).
    doc = parse_text(
        HEADER
        + "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
        + " where e is time_ordered and span(e) <= 7d"
        + " soft span(e) is soon"
        + " score min emit when score >= 0.1 }")
    plan = compile_motif(doc, "t")
    libs = libs_from_doc(doc)
    g = TemporalGraph.from_transfers(
        _frame(
            [
                (0, "001", "A", "002", "B", 100.0),
                (20, "002", "B", "001", "A", 100.0),
            ]
        )
    )
    res = score(g, 1, plan, libs)
    assert res.found and res.score == pytest.approx(1.0)


def test_named_quantifier_evaluates() -> None:
    doc = parse_text(
        HEADER
        + "quantifier mostly = ramp(0.6, 0.9)\n"
        + "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
        + " where e is time_ordered and span(e) <= 7d"
        + " soft @mostly e: e.gap is soon"
        + " score min emit when score >= 0.1 }")
    plan = compile_motif(doc, "t")
    libs = libs_from_doc(doc)
    g = TemporalGraph.from_transfers(
        _frame(
            [
                (0, "001", "A", "002", "B", 100.0),
                (60, "002", "B", "003", "C", 100.0),
                (120, "003", "C", "001", "A", 100.0),
            ]
        )
    )
    res = score(g, 2, plan, libs)
    assert 0.0 <= res.top_score <= 1.0


def test_parameterized_term_instantiates() -> None:
    # `within` would collide with the `~=` tolerance keyword in the lexer,
    # so the term is named `inside`: term names shadowing keywords do not
    # parse (known wart, not part of the v0 surface).
    doc = parse_text(
        "term inside(x) = trapezoid(0h, 0h, x, 2*x)\n"
        + "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
        + " where e is time_ordered and span(e) <= 7d"
        + " soft each e: e.gap is inside(6h)"
        + " score min emit when score >= 0.1 }")
    plan = compile_motif(doc, "t")
    libs = libs_from_doc(doc)
    g = TemporalGraph.from_transfers(
        _frame(
            [
                (0, "001", "A", "002", "B", 100.0),
                (3 * 60, "002", "B", "003", "C", 100.0),
                (13 * 60, "003", "C", "001", "A", 100.0),
            ]
        )
    )
    # Gaps 3h (full) and 10h ((12-10)/6 = 1/3): min is 1/3.
    res = score(g, 2, plan, libs)
    assert res.found and res.score == pytest.approx(1 / 3)


def test_pruning_saves_expansions_on_hub() -> None:
    # 400 stale senders into H, each with its own feeder Y_i, plus one live
    # chain Y -> X0 -> H -> T anchored at H -> T. Every stale branch dies on
    # a zero gap degree at the first extension; unpruned search descends
    # into all 400 feeders first.
    rows = [(0, "009", "Y", "001", "X0", 100.0), (10, "001", "X0", "002", "H", 100.0)]
    for i in range(400):
        rows.append((5, "009", f"Y{i}", "001", f"S{i}", 100.0))
        rows.append((20, "001", f"S{i}", "002", "H", 100.0))
    rows += [(10000, "002", "H", "003", "T", 100.0)]
    g = TemporalGraph.from_transfers(_frame(rows))
    _, plan, libs = _round_trip()
    anchor = g.n_edges - 1
    pruned = score(g, anchor, plan, libs, prune=True)
    raw = score(g, anchor, plan, libs, prune=False)
    assert pruned.expansions < raw.expansions * 0.6
    assert (pruned.found, pruned.score) == (raw.found, raw.score)


HI_SMALL = Path("data/hi-small/transfers.parquet")


@pytest.mark.skipif(not HI_SMALL.exists(), reason="HI-Small parquet not present")
def test_real_cycle_top_score_matches_unpruned() -> None:
    # Instance 102's true path gaps (~39h, ~31h) now score ~0.72 under
    # `patient` (recalibrated Oct 30 pass 1; was ~0.21 under `soon`), so it
    # fires at the 0.6 emit. Pruning must not change the emit-filtered
    # answer either way.
    g = TemporalGraph.from_parquet(HI_SMALL)
    doc = parse_text((Path("examples/amlworld.motif")).read_text())
    plan = compile_motif(doc, "round_trip")
    libs = libs_from_doc(doc)
    memb = pl.read_parquet("data/hi-small/pattern_members.parquet")
    last_tid = (
        memb.filter(pl.col("instance_id") == 102).get_column("transfer_id").max())
    order = np.argsort(g.transfer_id)
    anchor = int(order[int(np.searchsorted(g.transfer_id[order], last_tid))])
    pruned = score(g, anchor, plan, libs, prune=True)
    raw = score(g, anchor, plan, libs, prune=False)
    assert pruned.found and raw.found
    assert (pruned.witness, pruned.score) == (raw.witness, raw.score)
    assert 0.65 < raw.top_score < 0.80
    assert pruned.expansions < raw.expansions


def test_all_but_forgives_one_bad_step() -> None:
    # all but K drops the K worst bindings: one 5x amount jump zeroes
    # `each` but `all but 1` still fires. Completion-only on paths, so
    # pruned and unpruned must agree exactly.
    rows = [(0, "001", "A", "001", "B", 100.0),
            (10, "001", "B", "001", "C", 100.0),
            (20, "001", "C", "001", "D", 100.0),
            (30, "001", "D", "001", "A", 500.0)]
    g = TemporalGraph.from_transfers(_frame(rows))

    def _plan(quant: str):
        doc = parse_text(
            HEADER
            + "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
            + " where e is time_ordered and span(e) <= 7d"
            + f" soft {quant} e: e.amount ~= prev.amount within 20%"
            + " score min emit when score >= 0.5 }")
        return doc, compile_motif(doc, "t"), libs_from_doc(doc)

    _, plan, libs = _plan("each")
    strict = score(g, 3, plan, libs, prune=False)
    assert not strict.found and strict.top_score == pytest.approx(0.0)
    _, plan, libs = _plan("all but 1")
    pruned = score(g, 3, plan, libs, prune=True)
    raw = score(g, 3, plan, libs, prune=False)
    assert pruned.found and raw.found
    assert (pruned.witness, pruned.score) == (raw.witness, raw.score)
    assert raw.top_score == pytest.approx(1.0)


def test_exact_floor_matches_unpruned() -> None:
    # Review #14.1: branch-and-bound from ~0 returns the exact best
    # witness without unpruned search (which risks truncation). This
    # 0.5 witness is zeroed by default pruning at emit 0.6.
    g = TemporalGraph.from_transfers(
        _frame(
            [
                (0, "001", "B", "002", "A", 100.0),
                (27 * 60, "002", "A", "001", "B", 100.0),
            ]
        )
    )
    _, plan, libs = _round_trip()
    unpruned = score(g, 1, plan, libs, prune=False)
    exact = score(g, 1, plan, libs, prune=True, floor=EXACT_FLOOR)
    default = score(g, 1, plan, libs, prune=True)
    assert unpruned.top_score == pytest.approx(0.5)
    assert exact.top_score == pytest.approx(0.5)
    assert exact.witness == unpruned.witness
    assert default.top_score == 0.0
    assert not exact.found  # found still uses the emit threshold


def test_open_chain_searches_only_to_min_hops() -> None:
    # G (#15): open chains with no deferred soft are decided at min hops.
    # Every longer witness's min_hops suffix is itself an accepted witness
    # scoring at least as high (min over a subset), and accept-before-
    # descend records it first — so deeper search only adds expansions
    # and truncation, never a better score or a different witness.
    import dataclasses

    doc = parse_text(
        HEADER
        + "motif t { match (a:account) -[e:transfer*3..6]-> (z:account)"
        + " where e is time_ordered and span(e) <= 7d and a != z"
        + " soft each e: e.gap is soon"
        + " and each e: e.amount ~= prev.amount within 20%"
        + " score min emit when score >= 0.5 }"
    )
    plan, libs = compile_motif(doc, "t"), libs_from_doc(doc)
    assert plan.deferred is None and not plan.closed
    assert (plan.min_hops, plan.max_hops) == (3, 3)
    rows = [(i * 60, "001", f"A{i}", "001", f"A{i + 1}", 100.0) for i in range(6)]
    g = TemporalGraph.from_transfers(_frame(rows))
    wide = dataclasses.replace(plan, max_hops=6)
    for prune in (False, True):
        narrow = score(g, 5, plan, libs, prune=prune)
        assert narrow.found and narrow.score == pytest.approx(1.0)
        full = score(g, 5, wide, libs, prune=prune)
        assert (full.found, full.score, full.witness, full.top_score) == (
            narrow.found, narrow.score, narrow.witness, narrow.top_score)
        assert full.expansions > narrow.expansions
