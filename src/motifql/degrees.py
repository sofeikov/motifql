"""Fuzzy degree evaluation for MotifQL soft conditions (Oct 29).

Two call sites share the atom evaluators. The anchored search calls
`step_degrees` bottom-up as it extends a path: the gap and amount
tolerances of the edge *above* the new candidate become defined at that
moment (the deepest edge is the path's first step, so its gap and
prev-amount stay undefined and are skipped, per the agreed semantics).
The completed-witness interpreter `eval_soft` evaluates a whole deferred
expression over oldest-first edge positions, anchor included.

Quantifier meanings follow LANGUAGE.md section 3.2: each/all = minimum,
some = maximum, at least N = N-th largest, most/few = ramp over the mean,
named = user quantifier applied to the mean.
"""

from __future__ import annotations

import logging

from motifql.parser import (
    And, Approx, Call, Dur, IsTerm, Money, Not, Num, Or, Pct,
    Quantified, Ref,
)
from motifql.planner import AmountConsCond, GapTermCond

log = logging.getLogger("degrees")


def libs_from_doc(doc) -> tuple[dict, dict]:
    """Term and quantifier shapes from a parsed document for evaluators."""
    terms = {t.name: (t.shape, t.args) for t in doc.terms}
    quants = {q.name: (q.shape, q.args) for q in doc.quantifiers}
    return terms, quants


def term_degree(shape: str, args: tuple[float, ...], x: float) -> float:
    """Membership degree of x in a term shape (args already in x's unit)."""
    if shape == "trapezoid":
        a, b, c, d = args
        if x <= a or x >= d:
            return 0.0
        if x <= b:
            return 1.0 if b == a else (x - a) / (b - a)
        if x <= c:
            return 1.0
        return 1.0 if d == c else (d - x) / (d - c)
    if shape == "triangle":
        a, b, c = args
        return term_degree("trapezoid", (a, b, b, c), x)
    if shape == "ramp":
        lo, hi = args
        if x <= lo:
            return 0.0
        if x >= hi:
            return 1.0
        return (x - lo) / (hi - lo)
    if shape == "ramp_down":
        lo, hi = args
        return 1.0 - term_degree("ramp", (lo, hi), x)
    raise ValueError(f"unknown term shape {shape}")


def tol_degree(tol: object, x: float, y: float) -> float:
    """Degree of `x ~= y within tol` (LANGUAGE.md section 4.6)."""
    if isinstance(tol, Pct):
        denom = tol.fraction * max(abs(x), abs(y))
        if denom == 0:
            return 1.0 if x == y else 0.0
        return max(0.0, 1.0 - abs(x - y) / denom)
    width = tol.seconds if isinstance(tol, Dur) else tol.amount if isinstance(tol, Money) else tol.value
    if width == 0:
        return 1.0 if x == y else 0.0
    return max(0.0, 1.0 - abs(x - y) / width)


def step_degrees(atom, gap_s: float | None, amount: float,
                 prev_amount: float | None, terms: dict) -> float | None:
    """Degree of one prunable atom for a non-first step; None to skip."""
    if isinstance(atom, GapTermCond):
        if gap_s is None:
            return None
        shape, raw_args = terms[atom.term]
        args = tuple(_lit(a) for a in raw_args)
        deg = term_degree(shape, args, gap_s)
        return 1.0 - deg if atom.negated else deg
    if isinstance(atom, AmountConsCond):
        if prev_amount is None:
            return None
        deg = tol_degree(atom.tol, amount, prev_amount)
        return 1.0 - deg if atom.negated else deg
    raise ValueError(f"not a prunable atom: {atom!r}")


def _lit(node: object) -> float:
    if isinstance(node, Dur):
        return node.seconds
    if isinstance(node, Pct):
        return node.fraction
    if isinstance(node, Money):
        return node.amount
    if isinstance(node, Num):
        return node.value
    raise ValueError(f"non-literal term arg: {node!r}")


def eval_soft(expr: object, steps: list[int], graph, libs, tnorm: str) -> float:
    """Evaluate a deferred soft expression over completed witness steps."""
    from motifql.parser import Comparison as Cmp

    if isinstance(expr, And):
        return _combine([eval_soft(i, steps, graph, libs, tnorm) for i in expr.items], tnorm)
    if isinstance(expr, Or):
        return max([eval_soft(i, steps, graph, libs, tnorm) for i in expr.items] or [0.0])
    if isinstance(expr, Not):
        return 1.0 - eval_soft(expr.body, steps, graph, libs, tnorm)
    if isinstance(expr, Quantified):
        return _eval_quant(expr, steps, graph, libs, tnorm)
    if isinstance(expr, Approx):
        x = _eval_arith(expr.left, steps, graph)
        y = _eval_arith(expr.right, steps, graph)
        return tol_degree(expr.tol, x, y)
    if isinstance(expr, IsTerm):
        return _eval_term(expr, steps, graph, libs)
    if isinstance(expr, Cmp):
        x = _eval_arith(expr.left, steps, graph)
        y = _eval_arith(expr.right, steps, graph)
        return _compare(expr.op, x, y)
    raise ValueError(f"cannot evaluate in soft: {expr!r}")


def _combine(degrees: list[float], tnorm: str) -> float:
    if tnorm == "product":
        out = 1.0
        for d in degrees:
            out *= d
        return out
    return min(degrees) if degrees else 1.0


def _eval_quant(node: Quantified, steps: list[int], graph, libs, tnorm: str) -> float:
    terms, quants = libs
    per_step: list[float] = []
    times = [int(graph.times[e]) for e in steps]
    amounts = [float(graph.amount[e]) for e in steps]
    for i in range(len(steps)):
        gap = (times[i] - times[i - 1]) if i > 0 else None
        prev_amount = amounts[i - 1] if i > 0 else None
        per_step.append(_eval_in_step(node.body, gap, times[i], amounts[i], prev_amount, graph, libs, tnorm))
    q = node.quant
    if q in ("each", "all"):
        return min(per_step)
    if q == "some":
        return max(per_step)
    if q.startswith("at_least:"):
        n = int(q.split(":")[1])
        return sorted(per_step, reverse=True)[n - 1] if len(per_step) >= n else 0.0
    mean = sum(per_step) / len(per_step)
    if q == "most":
        return term_degree("ramp", (0.5, 0.8), mean)
    if q == "few":
        return term_degree("ramp_down", (0.1, 0.3), mean)
    if q.startswith("named:"):
        shape, args = quants[q.split(":")[1]]
        return term_degree(shape, tuple(args), mean)
    raise ValueError(f"unknown quantifier {q}")


def _eval_in_step(node: object, gap: float | None, t: int, amount: float,
                  prev_amount: float | None, graph, libs, tnorm: str) -> float:
    """Evaluate a quantifier body for one step (skips undefined gap/prev)."""
    terms, _ = libs
    if isinstance(node, And):
        ds = [_eval_in_step(i, gap, t, amount, prev_amount, graph, libs, tnorm)
              for i in node.items]
        return _combine(ds, tnorm)
    if isinstance(node, Or):
        return max([_eval_in_step(i, gap, t, amount, prev_amount, graph, libs, tnorm)
                    for i in node.items])
    if isinstance(node, Not):
        return 1.0 - _eval_in_step(node.body, gap, t, amount, prev_amount, graph, libs, tnorm)
    if isinstance(node, IsTerm) and node.term in terms and isinstance(node.base, Ref):
        parts = node.base.parts
        if len(parts) == 2 and parts[1] == "gap":
            if gap is None:
                return 1.0  # skipped on the first step: must not lower the score
            shape, raw_args = terms[node.term]
            deg = term_degree(shape, tuple(_lit(a) for a in raw_args), float(gap))
            return 1.0 - deg if node.negated else deg
    if isinstance(node, Approx):
        y = prev_amount
        if y is None:
            return 1.0  # prev undefined on the first step: skip
        return tol_degree(node.tol, amount, y)
    raise ValueError(f"cannot evaluate per step: {node!r}")


def _eval_term(node: IsTerm, steps: list[int], graph, libs, tnorm: str) -> float:
    terms, _ = libs
    if node.term == "time_ordered":
        times = [int(graph.times[e]) for e in steps]
        return 1.0 if all(b > a for a, b in zip(times, times[1:])) else 0.0
    if isinstance(node.base, Call) and node.base.name == "span":
        times = [int(graph.times[e]) for e in steps]
        shape, raw_args = terms[node.term]
        deg = term_degree(shape, tuple(_lit(a) for a in raw_args), float(times[-1] - times[0]))
        return 1.0 - deg if node.negated else deg
    raise ValueError(f"cannot evaluate term use: {node!r}")


def _eval_arith(node: object, steps: list[int], graph) -> float:
    if isinstance(node, Num):
        return node.value
    if isinstance(node, Dur):
        return node.seconds
    if isinstance(node, Money):
        return node.amount
    if isinstance(node, Ref):
        if node.parts[0] == "prev":
            raise ValueError("prev needs step context")
        raise ValueError(f"cannot evaluate ref {node!r} without step context")
    if isinstance(node, Call) and node.name == "span":
        times = [int(graph.times[e]) for e in steps]
        return float(times[-1] - times[0])
    raise ValueError(f"cannot evaluate arithmetic: {node!r}")


def _compare(op: str, x: float, y: float) -> float:
    return 1.0 if {
        "==": x == y, "!=": x != y, "<": x < y,
        "<=": x <= y, ">": x > y, ">=": x >= y,
    }[op] else 0.0
