"""Property tests for fuzzy degree evaluation (`src/motifql/degrees.py`).

Example tests in `tests/test_scores.py` pin exact values; these pin the
*laws*: every degree lands in [0, 1], tolerances are symmetric with full
marks for equality, and the crisp-safety predicates mean what they say
(review #14.6, issue #15 finding A).
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from hypothesis import given, settings, strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from motifql import degrees as D  # noqa: E402
from motifql.degrees import libs_from_doc  # noqa: E402
from motifql.graph import TemporalGraph  # noqa: E402
from motifql.parser import (  # noqa: E402
    And, Approx, Comparison, Dur, IsTerm, Money, Not, Num, Or, Pct,
    Quantified, Ref, parse_text,
)

BASE = datetime(2022, 9, 3)
HEADER = "term soon = trapezoid(0h, 0h, 6h, 48h)\n"

UNIT = st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False)
MAG = st.floats(0.0, 1e6, allow_nan=False, allow_infinity=False)
XVAL = st.floats(-1e6, 2e6, allow_nan=False, allow_infinity=False)


def _ordered(n: int):
    return st.lists(MAG, min_size=n, max_size=n).map(lambda xs: tuple(sorted(xs)))


# Non-negative tolerances only: a negative width divides by a negative
# number, and `tol_degree` can exceed 1.0 there (e.g. Num(-5) scores 3.0).
# The motif surface has no negative-tolerance literal, so this pins the
# reachable contract, not the arithmetic edge.
TOLS = st.one_of(
    st.builds(Pct, st.floats(0.0, 2.0, allow_nan=False, allow_infinity=False)),
    st.builds(lambda v: Money(v, "USD"), MAG),
    st.builds(Dur, MAG),
    st.builds(Num, MAG),
)


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


@given(_ordered(4), XVAL)
def test_trapezoid_bounded_with_shoulder_regions(args, x) -> None:
    a, b, c, d = args
    deg = D.term_degree("trapezoid", args, x)
    assert 0.0 <= deg <= 1.0
    if x < a or x > d:
        assert deg == 0.0
    if b <= x <= c:
        assert deg == 1.0


@given(_ordered(3), XVAL)
def test_triangle_is_trapezoid_with_flat_peak(args, x) -> None:
    a, b, c = args
    assert D.term_degree("triangle", args, x) == D.term_degree(
        "trapezoid", (a, b, b, c), x)


@given(_ordered(2), XVAL, XVAL)
def test_ramp_monotone_and_down_is_complement(args, x, y) -> None:
    dx = D.term_degree("ramp", args, x)
    assert 0.0 <= dx <= 1.0
    assert D.term_degree("ramp_down", args, x) == 1.0 - dx
    if x <= y:
        assert dx <= D.term_degree("ramp", args, y)


@given(TOLS, XVAL, XVAL)
def test_tol_degree_bounded_symmetric_full_marks_for_equality(
        tol, x, y) -> None:
    deg = D.tol_degree(tol, x, y)
    assert 0.0 <= deg <= 1.0
    assert D.tol_degree(tol, y, x) == deg
    if x == y:
        assert deg == 1.0


@given(st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
       XVAL.filter(lambda v: v != 0.0))
def test_pct_tol_zero_against_zero(f, x) -> None:
    # |x - 0| >= f * max(|x|, 0) whenever f <= 1: outside the support.
    assert D.tol_degree(Pct(f), x, 0.0) == 0.0


@given(MAG.filter(lambda w: w > 0.0))
def test_abs_tol_zero_at_twice_width(w) -> None:
    for tol in (Money(w, "USD"), Dur(w), Num(w)):
        assert D.tol_degree(tol, 0.0, 2.0 * w) == 0.0


@given(XVAL, XVAL)
def test_zero_width_is_equality(tol_x, y) -> None:
    assert D.tol_degree(Num(0.0), tol_x, y) == (1.0 if tol_x == y else 0.0)


@given(UNIT, st.booleans(), UNIT)
def test_crisp_degree_laws(d, crisp, alpha) -> None:
    got = D.crisp_degree(d, crisp, alpha)
    if not crisp:
        assert got == d
    else:
        assert got in (0.0, 1.0)
        if alpha <= 0.0:
            assert (got == 1.0) == (d > 0.0)
        else:
            assert (got == 1.0) == (d >= alpha)


@given(UNIT, st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False)
       .filter(lambda a: not 0.0 <= a <= 1.0))
def test_crisp_degree_rejects_alpha_outside_unit(d, alpha) -> None:
    with pytest.raises(ValueError):
        D.crisp_degree(d, True, alpha)


@given(st.lists(UNIT, max_size=20), st.sampled_from(["min", "product"]))
def test_combine_bounded_empty_is_one(parts, tnorm) -> None:
    got = D._combine(parts, tnorm)
    assert 0.0 <= got <= 1.0
    if not parts:
        assert got == 1.0
    if tnorm == "min" and parts:
        assert got == min(parts)


@given(st.lists(UNIT, min_size=1, max_size=20))
def test_product_below_min(parts) -> None:
    assert D._combine(parts, "product") <= min(parts) + 1e-12


# -- crisp-safety predicates, evaluated over a fixed witness -------------
# Gaps 1h (soon = 1.0) then 27h (soon = 0.5); amount steps 100 -> 110
# (within 20%: interior) then 110 -> 140 (outside: 0.0). Every leaf kind
# takes 0, interior and 1 across the generated expressions.

_EVAL_GRAPH = TemporalGraph.from_transfers(_frame([
    (0, "001", "A", "002", "B", 100.0),
    (60, "002", "B", "003", "C", 110.0),
    (1680, "003", "C", "001", "A", 140.0),
]))
_EVAL_LIBS = libs_from_doc(parse_text(HEADER))
_EVAL_STEPS = [0, 1, 2]

_LEAVES = st.sampled_from([
    IsTerm(Ref(("e", "gap")), False, "soon", ()),
    IsTerm(Ref(("e", "gap")), True, "soon", ()),
    Approx(Ref(("e", "amount")), Ref(("prev", "amount")), Pct(0.2)),
    Comparison(">=", Ref(("e", "amount")), Money(105.0, "USD")),
])
_BODIES = st.recursive(
    _LEAVES,
    lambda c: st.one_of(
        st.builds(lambda a, b: And((a, b)), c, c),
        st.builds(lambda a, b: Or((a, b)), c, c),
        st.builds(Not, c),
    ),
    max_leaves=6,
)
_TOPS = st.builds(
    lambda q, b: Quantified(q, "e", b),
    st.sampled_from(["each", "all", "some", "at_least:1", "at_least:2",
                     "all_but:1", "most", "few"]),
    _BODIES,
)
_EXPRS = st.one_of(
    _TOPS,
    st.builds(Not, _TOPS),
    st.builds(lambda a, b: And((a, b)), _TOPS, _TOPS),
)


@given(_EXPRS)
@settings(max_examples=150, deadline=None)
def test_crisp_binary_safe_implies_binary_score(expr) -> None:
    crisp = D.eval_soft(expr, _EVAL_STEPS, _EVAL_GRAPH, _EVAL_LIBS, "min",
                        True, 0.0)
    if D.crisp_binary_safe(expr):
        assert crisp in (0.0, 1.0)


@given(_EXPRS)
@settings(max_examples=150, deadline=None)
def test_crisp_support_differential_when_safe(expr) -> None:
    fuzzy = D.eval_soft(expr, _EVAL_STEPS, _EVAL_GRAPH, _EVAL_LIBS, "min")
    crisp = D.eval_soft(expr, _EVAL_STEPS, _EVAL_GRAPH, _EVAL_LIBS, "min",
                        True, 0.0)
    if D.crisp_differential_safe(expr, 0.0, "min"):
        assert (crisp > 0.0) == (fuzzy > 0.0)


@given(_EXPRS, st.sampled_from([0.3, 0.6]))
@settings(max_examples=150, deadline=None)
def test_crisp_alpha_differential_when_safe(expr, alpha) -> None:
    fuzzy = D.eval_soft(expr, _EVAL_STEPS, _EVAL_GRAPH, _EVAL_LIBS, "min")
    crisp = D.eval_soft(expr, _EVAL_STEPS, _EVAL_GRAPH, _EVAL_LIBS, "min",
                        True, alpha)
    if D.crisp_differential_safe(expr, alpha, "min"):
        assert (crisp == 1.0) == (fuzzy >= alpha)
