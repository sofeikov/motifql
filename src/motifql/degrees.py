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
    AmountConsCond, And, Approx, Arith, Call, Comparison, Dur, GapTermCond, IsTerm,
    Money, Not, Num, Or, Pct, Quantified, Ref,
)

log = logging.getLogger("degrees")


def libs_from_doc(doc) -> tuple[dict, dict]:
    """Term and quantifier definitions from a parsed document."""
    terms = {t.name: t for t in doc.terms}
    quants = {q.name: q for q in doc.quantifiers}
    return terms, quants


# Fields the graph store carries per edge (plus derived gap). Anything else
# in a soft predicate is an E_PLAN at plan time, never a silent zero.
STORED_FIELDS = ("time", "amount", "gap")


def resolve_term_args(term_name: str, use_args: tuple, term_defs: dict) -> tuple[float, ...]:
    """Concrete float args for a term use: bind call-site values to params."""
    tdef = term_defs[term_name]
    if len(use_args) != len(tdef.params):
        raise ValueError(f"term `{term_name}` takes {len(tdef.params)} arguments")
    scope = {formal: _eval_term_arg(actual, {}) for formal, actual in zip(tdef.params, use_args)}
    return tuple(_eval_term_arg(a, scope) for a in tdef.args)


def _eval_term_arg(node: object, scope: dict[str, float]) -> float:
    if isinstance(node, Num):
        return node.value
    if isinstance(node, Dur):
        return node.seconds
    if isinstance(node, Money):
        return node.amount
    if isinstance(node, Pct):
        return node.fraction
    if isinstance(node, Ref) and len(node.parts) == 1 and node.parts[0] in scope:
        return scope[node.parts[0]]
    if isinstance(node, Arith):
        x, y = _eval_term_arg(node.left, scope), _eval_term_arg(node.right, scope)
        return _apply_arith(node.op, x, y)
    raise ValueError(f"non-literal term argument: {node!r}")


def _apply_arith(op: str, x: float, y: float) -> float:
    if op == "/" and y == 0:
        raise ValueError("division by zero in arithmetic")
    return {"+": x + y, "-": x - y, "*": x * y, "/": x / y}[op]


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
                 prev_amount: float | None, term_defs: dict) -> float | None:
    """Degree of one prunable atom for a non-first step; None to skip."""
    if isinstance(atom, GapTermCond):
        if gap_s is None:
            return None
        tdef = term_defs[atom.term]
        deg = term_degree(tdef.shape, resolve_term_args(atom.term, atom.term_args, term_defs), gap_s)
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
    if isinstance(expr, Comparison):
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
    term_defs, quants = libs
    times = [int(graph.times[e]) for e in steps]
    amounts = [float(graph.amount[e]) for e in steps]
    per_step: list[float] = []
    for i in range(len(steps)):
        vals = {"time": float(times[i]), "amount": amounts[i],
                "gap": float(times[i] - times[i - 1]) if i > 0 else None}
        # The previous gap is the gap *of the previous step*, which is
        # defined only from the third witness step on (the first step has
        # no gap at all). Reads of a genuinely undefined prev.gap skip
        # the step, per the agreed prev semantics.
        pvals = {"time": float(times[i - 1]), "amount": amounts[i - 1],
                 "gap": float(times[i - 1] - times[i - 2]) if i > 1 else None} if i > 0 else None
        d = _eval_step(node.body, vals, pvals, libs, tnorm)
        if d is None:
            continue  # undefined prev/gap on this step: omit the binding
        per_step.append(d)
    if not per_step:
        # No defined bindings: universal quants hold vacuously, existential
        # ones fail. Witnesses always have >= 2 steps, so this is a backstop.
        return 1.0 if (node.quant in ("each", "all")
                       or node.quant.startswith("all_but:")) else 0.0
    q = node.quant
    if q in ("each", "all"):
        return min(per_step)
    if q.startswith("all_but:"):
        k = int(q.split(":")[1])
        return sorted(per_step)[k] if len(per_step) > k else 1.0
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
        qdef = quants[q.split(":")[1]]
        return term_degree(qdef.shape, tuple(_lit(a) for a in qdef.args), mean)
    raise ValueError(f"unknown quantifier {q}")


def _eval_step(node: object, vals: dict, pvals: dict | None,
               libs, tnorm: str) -> float | None:
    """Degree of one quantifier-body node for one step; None means skip.

    Skips propagate up: And/Or omit skipped children, and an And/Or whose
    children are all skipped is itself skipped (None); Not propagates.
    The plan guarantees every field read here is one the store carries.
    """
    term_defs, _ = libs
    if isinstance(node, And):
        ds = [_eval_step(i, vals, pvals, libs, tnorm) for i in node.items]
        ds = [d for d in ds if d is not None]
        return _combine(ds, tnorm) if ds else None
    if isinstance(node, Or):
        ds = [_eval_step(i, vals, pvals, libs, tnorm) for i in node.items]
        ds = [d for d in ds if d is not None]
        return max(ds) if ds else None
    if isinstance(node, Not):
        d = _eval_step(node.body, vals, pvals, libs, tnorm)
        return None if d is None else 1.0 - d
    if isinstance(node, IsTerm) and node.term in term_defs:
        v = _resolve_ref(node.base, vals, pvals)
        if v is None:
            return None
        tdef = term_defs[node.term]
        deg = term_degree(tdef.shape, resolve_term_args(node.term, node.args, term_defs), v)
        return 1.0 - deg if node.negated else deg
    if isinstance(node, Approx):
        x = _resolve_arith(node.left, vals, pvals)
        y = _resolve_arith(node.right, vals, pvals)
        if x is None or y is None:
            return None
        return tol_degree(node.tol, x, y)
    if isinstance(node, Comparison):
        x = _resolve_arith(node.left, vals, pvals)
        y = _resolve_arith(node.right, vals, pvals)
        if x is None or y is None:
            return None
        return _compare(node.op, x, y)
    raise ValueError(f"cannot evaluate per step: {node!r}")


def _resolve_ref(ref: Ref, vals: dict, pvals: dict | None) -> float | None:
    if len(ref.parts) == 1:
        raise ValueError(f"bare variable `{ref.parts[0]}` has no degree")
    if ref.parts[0] == "prev":
        if pvals is None:
            return None
        return pvals.get(ref.parts[1])
    return vals.get(ref.parts[1])


def _resolve_arith(node: object, vals: dict, pvals: dict | None) -> float | None:
    if isinstance(node, (Num, Dur, Money, Pct)):
        return _lit(node)
    if isinstance(node, Ref):
        return _resolve_ref(node, vals, pvals)
    if isinstance(node, Arith):
        x = _resolve_arith(node.left, vals, pvals)
        y = _resolve_arith(node.right, vals, pvals)
        if x is None or y is None or (node.op == "/" and y == 0):
            return None
        return _apply_arith(node.op, x, y)
    raise ValueError(f"cannot evaluate arithmetic per step: {node!r}")


def _eval_term(node: IsTerm, steps: list[int], graph, libs) -> float:
    term_defs, _ = libs
    if node.term == "time_ordered":
        times = [int(graph.times[e]) for e in steps]
        return 1.0 if all(b > a for a, b in zip(times, times[1:])) else 0.0
    if isinstance(node.base, Call) and node.base.name == "span":
        times = [int(graph.times[e]) for e in steps]
        tdef = term_defs[node.term]
        deg = term_degree(tdef.shape, resolve_term_args(node.term, node.args, term_defs),
                          float(times[-1] - times[0]))
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
    if isinstance(node, Arith):
        return _apply_arith(node.op, _eval_arith(node.left, steps, graph),
                            _eval_arith(node.right, steps, graph))
    raise ValueError(f"cannot evaluate arithmetic: {node!r}")


def _compare(op: str, x: float, y: float) -> float:
    return 1.0 if {
        "==": x == y, "!=": x != y, "<": x < y,
        "<=": x <= y, ">": x > y, ">=": x >= y,
    }[op] else 0.0
