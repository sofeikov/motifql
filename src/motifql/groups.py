"""Group and butterfly executors: anchored scoring for Oct 30 plans.

Every entry point scores one anchor transfer (by edge position) against a
:class:`GroupPlan` or :class:`ButterflyPlan`, returning a :class:`GroupResult`
with the same found/score/top_score convention as the path matcher's
``ScoredResult``: `score` is the best group at or above emit (0.0 otherwise),
`top_score` the best group regardless (for ranking), `truncated` when a hub
cap cut the lookup short.

Anchoring (LANGUAGE.md section 4.4): only bindings whose edges fall within
[D before anchor] count, the anchor must be in the group, and the anchor is
the latest edge. Consequences the executors rely on:

* fan_out/fan_in: the group is the key's windowed adjacency plus the anchor.
* scatter_gather: only anchors serving as leg2 (m -> d) can complete, since
  leg2 must follow leg1 and be <= anchor. Anchors with no incoming senders
  score 0.
* gather_scatter: only anchors serving as p (c -> y) can complete, for the
  same reason. Every anchor is read as p with the sender as the hub.
* bipartite_butterfly: the anchor is one of e1..e4 (whichever position it
  fills); the other three legs are looked up within the window.

Hub control mirrors the path matcher: per-list caps (most recent first) and
a per-anchor expansion cap. Hitting a cap marks `truncated` and stops that
lookup. Unlike path motifs, a truncated group score is not a lower bound:
dropping members can raise non-monotone aggregates (means, most/few), so
truncated results may overstate. Groups have no score pruning, so
untruncated top_score is exact.

Usage:
    from motifql.parser import parse_file
    from motifql.planner import compile_motif
    from motifql.groups import score_group
    plan = compile_motif(parse_file("examples/amlworld.motif"), "fan_out")
    res = score_group(graph, anchor_pos, plan, libs)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field

import numpy as np

from motifql.degrees import crisp_degree, libs_from_doc, resolve_term_args, term_degree, tol_degree  # noqa: F401
from motifql.parser import (
    AllArgs, And, Approx, Arith, Call, Comparison, DistinctArg, Dur, IsTerm, Money, Not, Num,
    Or, Pct, Quantified, Ref,
)

log = logging.getLogger("groups")

# Hub caps: per adjacency-list cut (most recent first) and per-anchor budget.
# The fan executor touches one list; scatter touches up to three levels, so
# its sender fan-out gets its own tighter cap below.
DEFAULT_MAX_FANOUT = 1_000
DEFAULT_MAX_EXPANSIONS = 50_000
SCATTER_MAX_SENDERS = 100
BUTTERFLY_MAX_PARTNERS = 50
# Cap on same-timestamp readings tried per member (F1): real ties are a
# handful; beyond this the keyed-hash order below is cut and truncated set.
MAX_TIED_READINGS = 64

_TIE_SEED = "motifql-member-tie-v1"


def _hkey(edge_pos: int) -> int:
    """Label-blind order key for simultaneous edges.

    Graph position is file order, which correlates with the label
    (laundering rows cluster), so it must never pick between edges at
    the same timestamp. A keyed hash keeps the order deterministic
    while carrying no label signal (same idea as rank_per_day's
    tie-break in eval/evaluate.py).
    """
    return int(hashlib.md5(f"{_TIE_SEED}:{edge_pos}".encode()).hexdigest()[:16], 16)


def _tied_latest(graph, edges: list[int]) -> tuple[list[int], bool]:
    """Edges at the latest timestamp: same-minute rows are simultaneous
    (the trailing-features precedent), so every tied edge is a candidate
    reading. Keyed-hash order, capped; the flag reports a cap cut."""
    if not edges:
        return [], False
    tmax = max(int(graph.times[e]) for e in edges)
    tied = sorted((e for e in edges if int(graph.times[e]) == tmax), key=_hkey)
    if len(tied) > MAX_TIED_READINGS:
        return tied[:MAX_TIED_READINGS], True
    return tied, False


@dataclass
class GroupResult:
    """Outcome of scoring one anchor against a group/butterfly plan."""

    found: bool
    witness: tuple[int, ...]  # edge positions, oldest first, anchor included
    score: float  # best group at or above emit, else 0.0
    degrees: tuple  # tuple[(predicate label, degree, None)]
    truncated: bool
    expansions: int
    top_score: float = 0.0  # best group even below emit (ranking)


# -- shared aggregate / soft evaluation ------------------------------------

class _Ctx:
    """Group facts the having/soft evaluators read.

    `counts`: member var -> distinct members. `edge_amounts`/`edge_times`:
    edge var -> oldest-first values. `member_vals`: member var -> member key
    -> list of {(edge var, field): value} readings, one per tied edge at
    the latest timestamp (F1: simultaneous edges all count).
    `span`: max-min time over all group edges.
    """

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.edge_amounts: dict[str, list[float]] = {}
        self.edge_times: dict[str, list[float]] = {}
        self.member_vals: dict[str, dict[int, list[dict[tuple[str, str], float]]]] = {}
        self.span: float = 0.0


def _lit(node: object) -> float:
    if isinstance(node, Dur):
        return node.seconds
    if isinstance(node, Pct):
        return node.fraction
    if isinstance(node, Money):
        return node.amount
    if isinstance(node, Num):
        return node.value
    raise ValueError(f"non-literal aggregate bound: {node!r}")


def _arith(node: object, ctx: _Ctx) -> float:
    """Value of a group-level aggregate or literal (no member context)."""
    if isinstance(node, (Num, Dur, Money, Pct)):
        return _lit(node)
    if isinstance(node, Arith):
        x, y = _arith(node.left, ctx), _arith(node.right, ctx)
        if node.op == "/" and y == 0:
            raise ValueError("division by zero in arithmetic")
        return {"+": x + y, "-": x - y, "*": x * y, "/": x / y}[node.op]
    if isinstance(node, Call):
        vals = _edge_values(node, ctx)
        if node.name == "count":
            return float(len(vals))
        if not vals:
            return 0.0
        if node.name == "sum":
            return float(sum(vals))
        if node.name == "avg":
            return float(sum(vals) / len(vals))
        if node.name == "min":
            return float(min(vals))
        if node.name == "max":
            return float(max(vals))
        if node.name == "span":
            return ctx.span
    raise ValueError(f"cannot evaluate group arithmetic: {node!r}")


def _edge_values(call: Call, ctx: _Ctx) -> list[float]:
    """Raw values behind one aggregate call."""
    if call.name == "span":
        return [ctx.span]
    if call.name == "count":
        (arg,) = call.args
        if isinstance(arg, DistinctArg):
            return [1.0] * ctx.counts.get(arg.ref.parts[0], 0)
        if isinstance(arg, AllArgs):
            n = 0
            for v in ctx.edge_amounts.values():
                n += len(v)
            return [1.0] * n
        raise ValueError(f"count over {arg!r}: use count(distinct m).")
    (arg,) = call.args
    if not (isinstance(arg, Ref) and len(arg.parts) == 2):
        raise ValueError(f"{call.name} over {arg!r}: use <edge>.amount.")
    edge_var, field = arg.parts
    src = ctx.edge_amounts if field == "amount" else ctx.edge_times
    return list(src.get(edge_var, []))


def _eval_having(expr: object, ctx: _Ctx) -> bool:
    if isinstance(expr, And):
        return all(_eval_having(i, ctx) for i in expr.items)
    if isinstance(expr, Or):
        return any(_eval_having(i, ctx) for i in expr.items)
    if isinstance(expr, Not):
        return not _eval_having(expr.body, ctx)
    if isinstance(expr, Comparison):
        x = _arith(expr.left, ctx)
        y = _arith(expr.right, ctx)
        return bool({
            "==": x == y, "!=": x != y, "<": x < y,
            "<=": x <= y, ">": x > y, ">=": x >= y,
        }[expr.op])
    raise ValueError(f"cannot evaluate having: {expr!r}")


def _combine(parts: list[float], tnorm: str) -> float:
    if tnorm == "product":
        out = 1.0
        for d in parts:
            out *= d
        return out
    return min(parts) if parts else 1.0


def _eval_soft(expr: object, ctx: _Ctx, libs, tnorm: str,
               crisp: bool = False, crisp_alpha: float = 0.0) -> tuple[float, list[tuple]]:
    """Group soft score plus per-conjunct explanation degrees."""
    if isinstance(expr, And):
        parts: list[float] = []
        degs: list[tuple] = []
        for item in expr.items:
            s, d = _eval_soft(item, ctx, libs, tnorm, crisp, crisp_alpha)
            parts.append(s)
            degs.extend(d)
        return _combine(parts, tnorm), degs
    if isinstance(expr, Or):
        best, best_degs = 0.0, []
        for item in expr.items:
            s, d = _eval_soft(item, ctx, libs, tnorm, crisp, crisp_alpha)
            if s >= best:
                best, best_degs = s, d
        return best, best_degs
    if isinstance(expr, Not):
        s, _ = _eval_soft(expr.body, ctx, libs, tnorm, crisp, crisp_alpha)
        return 1.0 - s, [(f"not ({_label(expr.body)})", 1.0 - s, None)]
    if isinstance(expr, Quantified):
        return _eval_quant(expr, ctx, libs, tnorm, crisp, crisp_alpha)
    if isinstance(expr, Approx):
        x = _arith(expr.left, ctx)
        y = _arith(expr.right, ctx)
        d = crisp_degree(tol_degree(expr.tol, x, y), crisp, crisp_alpha)
        return d, [(_label(expr), d, None)]
    if isinstance(expr, IsTerm):
        d = _eval_term(expr, ctx, libs, crisp, crisp_alpha)
        return d, [(_label(expr), d, None)]
    if isinstance(expr, Comparison):
        x = _arith(expr.left, ctx)
        y = _arith(expr.right, ctx)
        d = 1.0 if {
            "==": x == y, "!=": x != y, "<": x < y,
            "<=": x <= y, ">": x > y, ">=": x >= y,
        }[expr.op] else 0.0
        return d, [(_label(expr), d, None)]
    raise ValueError(f"cannot evaluate group soft: {expr!r}")


def _eval_term(node: IsTerm, ctx: _Ctx, libs, crisp: bool = False,
               crisp_alpha: float = 0.0) -> float:
    term_defs, _ = libs
    if node.term == "time_ordered":
        return 1.0
    base = node.base
    if isinstance(base, Call) and base.name == "span":
        span_args = resolve_term_args(node.term, node.args, term_defs)
        deg = term_degree(term_defs[node.term].shape, span_args, ctx.span)
        return (1.0 - crisp_degree(deg, crisp, crisp_alpha) if node.negated
                else crisp_degree(deg, crisp, crisp_alpha))
    if isinstance(base, Call):
        vals = _edge_values(base, ctx)
        x = float(vals[0]) if vals else 0.0
        if base.name == "count":
            x = float(len(vals))
        elif base.name == "sum":
            x = float(sum(vals))
        elif base.name == "avg":
            x = float(sum(vals) / len(vals)) if vals else 0.0
        elif base.name == "min":
            x = float(min(vals)) if vals else 0.0
        elif base.name == "max":
            x = float(max(vals)) if vals else 0.0
        span_args = resolve_term_args(node.term, node.args, term_defs)
        deg = term_degree(term_defs[node.term].shape, span_args, x)
        return (1.0 - crisp_degree(deg, crisp, crisp_alpha) if node.negated
                else crisp_degree(deg, crisp, crisp_alpha))
    raise ValueError(f"cannot evaluate group term use: {node!r}")


def _eval_quant(node: Quantified, ctx: _Ctx, libs, tnorm: str,
                crisp: bool = False, crisp_alpha: float = 0.0) -> tuple[float, list[tuple]]:
    term_defs, quants = libs
    per_member = ctx.member_vals.get(node.var, {})
    hits: list[float] = []
    for _, readings in per_member.items():
        # F1: the body is scored over every tied reading and the best
        # wins — picking one by file order leaked the label. A member
        # whose readings all skip is skipped, as before.
        ds = []
        for vals in readings:
            d = _eval_member(node.body, vals, ctx, libs, tnorm, crisp, crisp_alpha)
            if d is not None:
                ds.append(d)
        if not ds:
            continue
        hits.append(max(ds))
    if not hits:
        vacuous = node.quant in ("each", "all") or node.quant.startswith("all_but:")
        deg = 1.0 if vacuous else 0.0
        return deg, [(f"{_quant_label(node.quant)} {node.var}: {_label(node.body)}", deg, None)]
    q = node.quant
    if q in ("each", "all"):
        deg = min(hits)
    elif q.startswith("all_but:"):
        k = int(q.split(":")[1])
        deg = sorted(hits)[k] if len(hits) > k else 1.0
    elif q == "some":
        deg = max(hits)
    elif q.startswith("at_least:"):
        n = int(q.split(":")[1])
        deg = sorted(hits, reverse=True)[n - 1] if len(hits) >= n else 0.0
    else:
        mean = sum(hits) / len(hits)
        if q == "most":
            deg = term_degree("ramp", (0.5, 0.8), mean)
        elif q == "few":
            deg = term_degree("ramp_down", (0.1, 0.3), mean)
        elif q.startswith("named:"):
            qdef = quants[q.split(":")[1]]
            deg = term_degree(qdef.shape, tuple(_lit(a) for a in qdef.args), mean)
        else:
            raise ValueError(f"unknown quantifier {q}")
    return deg, [(f"{_quant_label(q)} {node.var}: {_label(node.body)}", deg, None)]


def _eval_member(node: object, vals: dict, ctx: _Ctx, libs, tnorm: str,
                 crisp: bool = False, crisp_alpha: float = 0.0) -> float | None:
    """Degree of one quantifier body for one member; None skips the member."""
    term_defs, _ = libs
    if isinstance(node, And):
        ds = [_eval_member(i, vals, ctx, libs, tnorm, crisp, crisp_alpha) for i in node.items]
        ds = [d for d in ds if d is not None]
        return _combine(ds, tnorm) if ds else None
    if isinstance(node, Or):
        ds = [_eval_member(i, vals, ctx, libs, tnorm, crisp, crisp_alpha) for i in node.items]
        ds = [d for d in ds if d is not None]
        return max(ds) if ds else None
    if isinstance(node, Not):
        d = _eval_member(node.body, vals, ctx, libs, tnorm, crisp, crisp_alpha)
        return None if d is None else 1.0 - d
    if isinstance(node, Approx):
        x = _member_arith(node.left, vals, ctx)
        y = _member_arith(node.right, vals, ctx)
        if x is None or y is None:
            return None
        return crisp_degree(tol_degree(node.tol, x, y), crisp, crisp_alpha)
    if isinstance(node, IsTerm) and node.term in term_defs:
        v = _member_arith(node.base, vals, ctx)
        if v is None:
            return None
        tdef = term_defs[node.term]
        deg = term_degree(tdef.shape, resolve_term_args(node.term, node.args, term_defs), v)
        return (1.0 - crisp_degree(deg, crisp, crisp_alpha) if node.negated
                else crisp_degree(deg, crisp, crisp_alpha))
    if isinstance(node, Comparison):
        x = _member_arith(node.left, vals, ctx)
        y = _member_arith(node.right, vals, ctx)
        if x is None or y is None:
            return None
        return 1.0 if {
            "==": x == y, "!=": x != y, "<": x < y,
            "<=": x <= y, ">": x > y, ">=": x >= y,
        }[node.op] else 0.0
    raise ValueError(f"cannot evaluate per member: {node!r}")


def _member_arith(node: object, vals: dict, ctx: _Ctx) -> float | None:
    if isinstance(node, (Num, Dur, Money, Pct)):
        return _lit(node)
    if isinstance(node, Ref):
        if len(node.parts) == 2 and (node.parts[0], node.parts[1]) in vals:
            return vals[(node.parts[0], node.parts[1])]
        return None
    if isinstance(node, Arith):
        x = _member_arith(node.left, vals, ctx)
        y = _member_arith(node.right, vals, ctx)
        if x is None or y is None or (node.op == "/" and y == 0):
            return None
        return {"+": x + y, "-": x - y, "*": x * y, "/": x / y}[node.op]
    if isinstance(node, Call) and node.name in ("avg", "sum", "min", "max"):
        try:
            return _arith(node, ctx)
        except ValueError:
            return None
    raise ValueError(f"cannot evaluate member arithmetic: {node!r}")


def _quant_label(q: str) -> str:
    if q.startswith("named:"):
        return "@" + q.split(":")[1]
    if q.startswith("at_least:"):
        return "at least " + q.split(":")[1]
    if q.startswith("all_but:"):
        return "all but " + q.split(":")[1]
    return q


def _fmt_tol(tol: object) -> str:
    if isinstance(tol, Pct):
        return f"{tol.fraction * 100:g}%"
    if isinstance(tol, Money):
        return f"{tol.amount:g} {tol.currency}"
    if isinstance(tol, Dur):
        return f"{tol.seconds:g}s"
    if isinstance(tol, Num):
        return f"{tol.value:g}"
    return "?"


def _label(node: object) -> str:
    if isinstance(node, IsTerm):
        base = node.base
        if isinstance(base, Call):
            return f"{base.name}(...) is {'not ' if node.negated else ''}{node.term}"
        if isinstance(base, Ref):
            return f"{'.'.join(base.parts)} is {'not ' if node.negated else ''}{node.term}"
        return f"is {node.term}"
    if isinstance(node, Approx):
        return f"{_short(node.left)} ~= {_short(node.right)} within {_fmt_tol(node.tol)}"
    if isinstance(node, Comparison):
        return f"{_short(node.left)} {node.op} {_short(node.right)}"
    if isinstance(node, Quantified):
        return f"{_quant_label(node.quant)} {node.var}: {_label(node.body)}"
    if isinstance(node, And):
        return " and ".join(_label(i) for i in node.items)
    if isinstance(node, Or):
        return " or ".join(_label(i) for i in node.items)
    if isinstance(node, Not):
        return f"not ({_label(node.body)})"
    return type(node).__name__


def _short(node: object) -> str:
    if isinstance(node, Ref):
        return ".".join(node.parts)
    if isinstance(node, Call):
        args = ", ".join(_short(a) for a in node.args)
        return f"{node.name}({args})"
    if isinstance(node, (Num, Dur, Money, Pct)):
        return _fmt_tol(node)
    return type(node).__name__


def _finish(name: str, witness: list[int], graph, s: float, degs: list[tuple],
            truncated: bool, expansions: int, emit: float) -> GroupResult:
    ordered = sorted(witness, key=lambda e: (int(graph.times[e]), int(e)))
    found = s >= emit
    return GroupResult(found, tuple(ordered), s if found else 0.0,
                       tuple(degs), truncated, expansions, s)


# -- fan executor --------------------------------------------------------

def _score_fan(graph, anchor_pos: int, plan, libs, crisp: bool = False,
               crisp_alpha: float = 0.0) -> GroupResult:
    roles = dict(plan.roles)
    key_is_sender = roles["dir"] == "out-key-sender"
    anchor_t = int(graph.times[anchor_pos])
    key_node = int(graph.frm[anchor_pos]) if key_is_sender else int(graph.to[anchor_pos])
    window_s = plan.window_s
    if key_is_sender:
        edges, _ = graph.outgoing(key_node, anchor_t, window_s)
        side = graph.to
    else:
        edges, _ = graph.incoming(key_node, anchor_t, window_s)
        side = graph.frm
    truncated = False
    expansions = len(edges) + 1
    if len(edges) > DEFAULT_MAX_FANOUT:
        edges = edges[-DEFAULT_MAX_FANOUT:]
        truncated = True
    all_edges = [int(e) for e in edges.tolist()] + [int(anchor_pos)]
    if expansions > DEFAULT_MAX_EXPANSIONS:
        truncated = True
    # Tied edges at the latest timestamp per member (F1): the anchor is
    # the latest by time, so it is always among its member's readings.
    by_member: dict[int, list[int]] = {}
    for e in all_edges:
        by_member.setdefault(int(side[e]), []).append(int(e))
    rep: dict[int, list[int]] = {}
    for node, es in by_member.items():
        tied, cut = _tied_latest(graph, es)
        if cut:
            truncated = True
        rep[node] = tied
    edge_var = roles["edge"]
    member_var = roles["member"]
    ctx = _Ctx()
    ctx.counts[member_var] = len(rep)
    amounts = [float(graph.amount[e]) for e in all_edges]
    times = [float(graph.times[e]) for e in all_edges]
    ctx.edge_amounts[edge_var] = amounts
    ctx.edge_times[edge_var] = times
    ctx.span = max(times) - min(times) if times else 0.0
    ctx.member_vals[member_var] = {
        node: [{(edge_var, "amount"): float(graph.amount[pos]),
                (edge_var, "time"): float(graph.times[pos])} for pos in tied]
        for node, tied in rep.items()
    }
    if plan.having is not None and not _eval_having(plan.having, ctx):
        return GroupResult(False, (), 0.0, (), truncated, expansions, 0.0)
    if plan.soft is None:
        return _finish(plan.name, all_edges, graph, 1.0, [], truncated, expansions, plan.emit)
    s, degs = _eval_soft(plan.soft, ctx, libs, plan.score_tnorm, crisp, crisp_alpha)
    return _finish(plan.name, all_edges, graph, s, degs, truncated, expansions, plan.emit)


# -- scatter executor ------------------------------------------------------

def _score_scatter(graph, anchor_pos: int, plan, libs, crisp: bool = False,
                   crisp_alpha: float = 0.0) -> GroupResult:
    roles = dict(plan.roles)
    window_s = plan.window_s
    anchor_t = int(graph.times[anchor_pos])
    m0 = int(graph.frm[anchor_pos])
    d_node = int(graph.to[anchor_pos])
    if m0 == d_node:
        # F2: the anchor is read as leg2 (m -> d) and the main loop skips
        # m == d members, so a self-loop anchor can never complete —
        # without this it scored through the fallback below.
        return GroupResult(False, (), 0.0, (), False, 1, 0.0)
    s_var, m_var = roles["s"], roles["m"]
    leg1_var, leg2_var = roles["leg1"], roles["leg2"]
    in_m0, _ = graph.incoming(m0, anchor_t, window_s)
    in_d, _ = graph.incoming(d_node, anchor_t, window_s)
    truncated = False
    expansions = len(in_m0) + len(in_d) + 1
    if len(in_m0) > DEFAULT_MAX_FANOUT:
        in_m0 = in_m0[-DEFAULT_MAX_FANOUT:]
        truncated = True
    if len(in_d) > DEFAULT_MAX_FANOUT:
        in_d = in_d[-DEFAULT_MAX_FANOUT:]
        truncated = True
    # Most recent leg1 edge per sender into m0.
    leg1_by_sender: dict[int, int] = {}
    for e in sorted(in_m0.tolist(), key=lambda p: (int(graph.times[p]), int(p))):
        leg1_by_sender[int(graph.frm[int(e)])] = int(e)
    senders = sorted(leg1_by_sender,
                     key=lambda s: (int(graph.times[leg1_by_sender[s]]), s))[-SCATTER_MAX_SENDERS:]
    if len(leg1_by_sender) > len(senders):
        truncated = True
    # Leg2 edges per intermediary into d, oldest first.
    leg2s_by_mid: dict[int, list[int]] = {}
    for e in sorted(in_d.tolist(), key=lambda p: (int(graph.times[p]), int(p))):
        leg2s_by_mid.setdefault(int(graph.frm[int(e)]), []).append(int(e))
    best: tuple[float, list[tuple], list[int]] | None = None
    for s in senders:
        if s == d_node:
            continue  # s != d guard
        out_s, _ = graph.outgoing(s, anchor_t, window_s)
        expansions += len(out_s)
        if len(out_s) > DEFAULT_MAX_FANOUT:
            out_s = out_s[-DEFAULT_MAX_FANOUT:]
            truncated = True
        mids: dict[int, list[int]] = {}
        for e in sorted(out_s.tolist(), key=lambda p: (int(graph.times[p]), int(p))):
            mids.setdefault(int(graph.to[int(e)]), []).append(int(e))
        members: dict[int, tuple[list[int], list[int]]] = {}
        for mid, leg1s in mids.items():
            # F8 (#15): the source is never its own intermediary. A
            # self-loop s→s as leg1 plus the direct s→d payment as
            # leg2 would otherwise count as a member; truth has no
            # self-loops, but 11.6% of background transfers are.
            if mid == d_node or mid == s:
                continue
            if mid == m0:
                leg2s = leg2s_by_mid.get(mid, []) + [int(anchor_pos)]
            elif mid in leg2s_by_mid:
                leg2s = leg2s_by_mid[mid]
            else:
                continue
            # Aggregates range over every edge joining a valid pair: a leg1
            # before the latest leg2, a leg2 after the earliest leg1. The
            # mid qualifies iff the two sides overlap in that order.
            latest_l2 = int(graph.times[leg2s[-1]])
            earliest_l1 = int(graph.times[leg1s[0]])
            if not earliest_l1 < latest_l2:
                continue
            valid_l1 = [e for e in leg1s if int(graph.times[e]) < latest_l2]
            valid_l2 = [e for e in leg2s if int(graph.times[e]) > earliest_l1]
            members[mid] = (valid_l1, valid_l2)
        if m0 not in members and m0 != s and s in leg1_by_sender:
            # The anchor's own leg1 with no separate leg2 record: anchor is leg2.
            # (Only reachable when caps cut m0's out-edges; normally m0 pairs above.)
            # F8 (#15): m0 == s would re-add the source as its own mid.
            leg1 = leg1_by_sender[s]
            if int(graph.times[int(anchor_pos)]) > int(graph.times[leg1]):
                members[m0] = ([int(leg1)], [int(anchor_pos)])
        ctx = _Ctx()
        ctx.counts[m_var] = len(members)
        l1 = [float(graph.amount[e]) for l1s, _ in members.values() for e in l1s]
        l2 = [float(graph.amount[e]) for _, l2s in members.values() for e in l2s]
        ctx.edge_amounts[leg1_var] = l1
        ctx.edge_times[leg1_var] = [float(graph.times[e]) for l1s, _ in members.values() for e in l1s]
        ctx.edge_amounts[leg2_var] = l2
        ctx.edge_times[leg2_var] = [float(graph.times[e]) for _, l2s in members.values() for e in l2s]
        all_t = ctx.edge_times[leg1_var] + ctx.edge_times[leg2_var]
        ctx.span = max(all_t) - min(all_t) if all_t else 0.0
        # F1: one reading per tied leg1 x tied leg2 combo at the latest
        # timestamps — the quantifier scores each and the best wins.
        readings: dict[int, list[dict]] = {}
        for mid, (l1s, l2s) in members.items():
            tied_l1, cut1 = _tied_latest(graph, l1s)
            tied_l2, cut2 = _tied_latest(graph, l2s)
            if cut1 or cut2:
                truncated = True
            combos = [(a, b) for a in tied_l1 for b in tied_l2]
            if len(combos) > MAX_TIED_READINGS:
                truncated = True
                combos = combos[:MAX_TIED_READINGS]
            # F5: each combo is one (leg1, leg2) pair, so its gap is
            # exact (leg2 - leg1, always positive here: every tied leg1
            # predates the latest leg2). `leg1.gap` stays undefined.
            readings[mid] = [
                {(leg1_var, "amount"): float(graph.amount[a]),
                 (leg1_var, "time"): float(graph.times[a]),
                 (leg2_var, "amount"): float(graph.amount[b]),
                 (leg2_var, "time"): float(graph.times[b]),
                 (leg2_var, "gap"): float(graph.times[b]) - float(graph.times[a])}
                for a, b in combos
            ]
        ctx.member_vals[m_var] = readings
        if plan.having is not None and not _eval_having(plan.having, ctx):
            continue
        if plan.soft is None:
            s_score, degs = 1.0, []
        else:
            s_score, degs = _eval_soft(plan.soft, ctx, libs, plan.score_tnorm,
                                       crisp, crisp_alpha)
        witness = sorted({e for l1s, l2s in members.values() for e in (*l1s, *l2s)},
                         key=lambda p: (int(graph.times[p]), int(p)))
        if best is None or s_score > best[0]:
            best = (s_score, degs, witness)
        if expansions > DEFAULT_MAX_EXPANSIONS:
            truncated = True
            break
    if best is None:
        return GroupResult(False, (), 0.0, (), truncated, expansions, 0.0)
    s_score, degs, witness = best
    return _finish(plan.name, witness, graph, s_score, degs, truncated, expansions, plan.emit)


# -- gather executor -------------------------------------------------------

def _score_gather(graph, anchor_pos: int, plan, libs, crisp: bool = False,
                   crisp_alpha: float = 0.0) -> GroupResult:
    roles = dict(plan.roles)
    window_s = plan.window_s
    anchor_t = int(graph.times[anchor_pos])
    c_node = int(graph.frm[anchor_pos])  # anchor read as p: c -> y
    g_var, p_var = roles["g"], roles["p"]
    x_var, y_var = roles["x"], roles["y"]
    g_edges, _ = graph.incoming(c_node, anchor_t, window_s)
    p_edges, _ = graph.outgoing(c_node, anchor_t, window_s)
    truncated = False
    if len(g_edges) > DEFAULT_MAX_FANOUT:
        g_edges = g_edges[-DEFAULT_MAX_FANOUT:]
        truncated = True
    if len(p_edges) > DEFAULT_MAX_FANOUT:
        p_edges = p_edges[-DEFAULT_MAX_FANOUT:]
        truncated = True
    g_list = [int(e) for e in g_edges.tolist()]
    p_list = [int(e) for e in p_edges.tolist()] + [int(anchor_pos)]
    if not g_list:
        # F6: no collection means no (g, p) binding, so nothing is found
        # even when `having` only gates the payout side.
        return GroupResult(False, (), 0.0, (), truncated,
                            len(g_list) + len(p_list), 0.0)
    # Enforce `p.time > g.time`: a payout counts only if some
    # collection precedes it (join semantics over the guard). The
    # anchor is itself a p edge later than every g edge, so it always
    # qualifies; every g edge trivially has a later p edge.
    earliest_g = min(int(graph.times[e]) for e in g_list)
    p_list = [e for e in p_list if int(graph.times[e]) > earliest_g]
    expansions = len(g_list) + len(p_list)
    x_nodes = sorted({int(graph.frm[e]) for e in g_list})
    y_nodes = sorted({int(graph.to[e]) for e in p_list})
    ctx = _Ctx()
    ctx.counts[x_var] = len(x_nodes)
    ctx.counts[y_var] = len(y_nodes)
    ctx.edge_amounts[g_var] = [float(graph.amount[e]) for e in g_list]
    ctx.edge_times[g_var] = [float(graph.times[e]) for e in g_list]
    ctx.edge_amounts[p_var] = [float(graph.amount[e]) for e in p_list]
    ctx.edge_times[p_var] = [float(graph.times[e]) for e in p_list]
    all_t = ctx.edge_times[g_var] + ctx.edge_times[p_var]
    ctx.span = max(all_t) - min(all_t) if all_t else 0.0
    # Representative readings per member (F1): tied edges at the latest
    # timestamp each side, one reading each.
    gx: dict[int, list[int]] = {}
    by_sender: dict[int, list[int]] = {}
    for e in g_list:
        by_sender.setdefault(int(graph.frm[e]), []).append(e)
    for n, es in by_sender.items():
        tied, cut = _tied_latest(graph, es)
        if cut:
            truncated = True
        gx[n] = tied
    py: dict[int, list[int]] = {}
    by_receiver: dict[int, list[int]] = {}
    for e in p_list:
        by_receiver.setdefault(int(graph.to[e]), []).append(e)
    for n, es in by_receiver.items():
        tied, cut = _tied_latest(graph, es)
        if cut:
            truncated = True
        py[n] = tied
    ctx.member_vals[x_var] = {
        n: [{(g_var, "amount"): float(graph.amount[pos]),
             (g_var, "time"): float(graph.times[pos])} for pos in tied]
        for n, tied in gx.items()
    }
    ctx.member_vals[y_var] = {
        n: [{(p_var, "amount"): float(graph.amount[pos]),
             (p_var, "time"): float(graph.times[pos])} for pos in tied]
        for n, tied in py.items()
    }
    if plan.having is not None and not _eval_having(plan.having, ctx):
        return GroupResult(False, (), 0.0, (), truncated, expansions, 0.0)
    if plan.soft is None:
        return _finish(plan.name, g_list + p_list, graph, 1.0, [], truncated,
                       expansions, plan.emit)
    s, degs = _eval_soft(plan.soft, ctx, libs, plan.score_tnorm, crisp, crisp_alpha)
    return _finish(plan.name, g_list + p_list, graph, s, degs, truncated,
                   expansions, plan.emit)


# -- butterfly executor ----------------------------------------------------

def _score_butterfly(graph, anchor_pos: int, plan, libs, crisp: bool = False,
                     crisp_alpha: float = 0.0) -> GroupResult:
    window_s = plan.window_s
    anchor_t = int(graph.times[anchor_pos])
    a_node = int(graph.frm[anchor_pos])
    b_node = int(graph.to[anchor_pos])
    out_a, _ = graph.outgoing(a_node, anchor_t, window_s)
    in_b, _ = graph.incoming(b_node, anchor_t, window_s)
    truncated = False
    if len(out_a) > BUTTERFLY_MAX_PARTNERS:
        out_a = out_a[-BUTTERFLY_MAX_PARTNERS:]
        truncated = True
    if len(in_b) > BUTTERFLY_MAX_PARTNERS:
        in_b = in_b[-BUTTERFLY_MAX_PARTNERS:]
        truncated = True
    # Per-partner edge LISTS (F7): every parallel edge in the window is a
    # candidate leg. The old code kept only the newest edge per pair, so
    # `span(all) is not soon` (or any term rising with span) never saw the
    # longer-span bindings and scored too low. Keeping every edge also
    # fixes F1's butterfly leg: same-minute ties are all tried instead of
    # picked by file order.
    y2_edges: dict[int, list[int]] = {}
    for e in out_a.tolist():
        e = int(e)
        if int(graph.to[e]) != b_node:
            y2_edges.setdefault(int(graph.to[e]), []).append(e)
    x2_edges: dict[int, list[int]] = {}
    for e in in_b.tolist():
        e = int(e)
        if int(graph.frm[e]) != a_node:
            x2_edges.setdefault(int(graph.frm[e]), []).append(e)
    expansions = len(out_a) + len(in_b) + 1
    x2_nodes = sorted(x2_edges)
    # Per-x2 outgoing map (capped): which y2 does x2 pay within the window?
    cross: dict[int, dict[int, list[int]]] = {}
    for x2 in x2_nodes:
        out_x, _ = graph.outgoing(x2, anchor_t, window_s)
        expansions += len(out_x)
        if len(out_x) > DEFAULT_MAX_FANOUT:
            out_x = out_x[-DEFAULT_MAX_FANOUT:]
            truncated = True
        m: dict[int, list[int]] = {}
        for e in out_x.tolist():
            m.setdefault(int(graph.to[int(e)]), []).append(int(e))
        cross[x2] = m
        if expansions > DEFAULT_MAX_EXPANSIONS:
            truncated = True
            break
    term_defs, _ = libs
    if plan.soft is None:
        grade = None
    else:
        # Validated by the planner: one `span(all) is <term>` condition.
        soft = plan.soft
        grade = (term_defs[soft.term],
                 resolve_term_args(soft.term, soft.args, term_defs),
                 soft.negated)

    def _grade(span_s: float) -> float:
        if grade is None:
            return 1.0
        tdef, targs, negated = grade
        d = term_degree(tdef.shape, targs, span_s)
        return (1.0 - crisp_degree(d, crisp, crisp_alpha) if negated
                else crisp_degree(d, crisp, crisp_alpha))

    def _realize(lst: list[int], t: int) -> int:
        # One leg with the binding's min time t: prefer an edge exactly at
        # t (keyed-hash order, label-blind), else the earliest edge after
        # it — t never exceeds the list's max, so one always exists.
        at = [e for e in lst if int(graph.times[e]) == t]
        if at:
            return min(at, key=_hkey)
        return min((e for e in lst if int(graph.times[e]) > t),
                   key=lambda e: (int(graph.times[e]), _hkey(e)))

    best: tuple[float, list[int]] | None = None
    earliest = anchor_t - window_s
    for x2, x2_to_b_list in x2_edges.items():
        if x2 not in cross:
            # Truncated out of the cross map (built in id order, scored in
            # time order): skip it, not everything after it.
            continue
        for y2, a_to_y2_list in y2_edges.items():
            x2_to_y2_list = cross[x2].get(y2)
            if not x2_to_y2_list:
                continue
            # The span only sees the oldest leg, so per pair the optimum
            # is one of the achievable min times: a time t present in the
            # three lists with every list holding an edge at or after t
            # (i.e. t at most the smallest of the three maxima). Exact for
            # any term shape — no newest/oldest extremes heuristic.
            lists = (x2_to_b_list, a_to_y2_list, x2_to_y2_list)
            maxima = [max(int(graph.times[e]) for e in lst) for lst in lists]
            cap_t = min(maxima)
            cands = sorted({int(graph.times[e]) for lst in lists for e in lst
                            if earliest <= int(graph.times[e]) <= cap_t})
            # Counted but not budget-breaked: the maps above are already
            # cap-limited, and scoring must proceed over what they hold
            # (M4: truncated maps still score their surviving partners).
            expansions += len(cands)
            for t in cands:
                d = _grade(float(anchor_t - t))
                if best is not None and d <= best[0]:
                    continue
                witness = [_realize(lst, t) for lst in lists] + [int(anchor_pos)]
                best = (d, witness)
    if best is None:
        return GroupResult(False, (), 0.0, (), truncated, expansions, 0.0)
    deg, witness = best
    label = _label(plan.soft) if plan.soft is not None else "score"
    degs = [(label, deg, None)]
    return _finish(plan.name, witness, graph, deg, degs, truncated, expansions, plan.emit)


# -- dispatch --------------------------------------------------------------

def score_any(graph, anchor_pos: int, plan, libs, prune: bool = True,
              crisp: bool = False, crisp_alpha: float = 0.0,
              floor: float | None = None):
    """Score one anchor against any Oct 30 plan type.

    Ranking callers want exact top_scores: pass `floor=EXACT_FLOOR`
    (branch-and-bound from ~0, exact mod 1e-12) instead of
    `prune=False`, which additionally risks truncation. Under default
    pruning a path motif's top_score can only be 0 or >= emit, so
    every below-emit anchor ties at 0 instead of ranking by its best
    witness. With crisp=True every soft leaf is α-cut (support at
    α=0, `d >= α` above; RQ2 exact arm); path pruning stays sound
    because crisp degrees still bound their completions. `floor` is
    path-only (groups never prune) and `found` still uses the emit
    threshold.
    """
    from motifql import planner as _planner

    if isinstance(plan, _planner.PathPlan):
        from motifql.matcher import score as _path_score
        return _path_score(graph, anchor_pos, plan, libs, prune=prune, crisp=crisp,
                           crisp_alpha=crisp_alpha, floor=floor)
    if isinstance(plan, _planner.GroupPlan):
        return score_group(graph, anchor_pos, plan, libs, crisp=crisp,
                           crisp_alpha=crisp_alpha)
    if isinstance(plan, _planner.ButterflyPlan):
        return score_butterfly(graph, anchor_pos, plan, libs, crisp=crisp,
                               crisp_alpha=crisp_alpha)
    raise TypeError(f"unknown plan type: {type(plan).__name__}")


def score_group(graph, anchor_pos: int, plan, libs, crisp: bool = False,
                crisp_alpha: float = 0.0) -> GroupResult:
    """Score one anchor against a GroupPlan by kind."""
    if plan.kind == "fan":
        return _score_fan(graph, anchor_pos, plan, libs, crisp, crisp_alpha)
    if plan.kind == "scatter":
        return _score_scatter(graph, anchor_pos, plan, libs, crisp, crisp_alpha)
    if plan.kind == "gather":
        return _score_gather(graph, anchor_pos, plan, libs, crisp, crisp_alpha)
    raise ValueError(f"unknown group kind: {plan.kind}")


def score_butterfly(graph, anchor_pos: int, plan, libs, crisp: bool = False,
                    crisp_alpha: float = 0.0) -> GroupResult:
    """Score one anchor against a ButterflyPlan."""
    return _score_butterfly(graph, anchor_pos, plan, libs, crisp, crisp_alpha)
