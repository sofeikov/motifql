"""MotifQL planner: AST to executable plans (Oct 29 path compiler, Oct 30 groups).

For a single-pattern path motif the plan fixes the anchored backward
search: hop bounds, the anchor-relative window, and the soft conditions
split into prunable vs completion-only. Prunable conditions are
final-on-bind per-step predicates under `each`/`all` (gap terms and
amount-vs-prev tolerances); their running degree is an upper bound on the
final score, so the matcher cuts a branch the moment it drops below the
emit threshold (LANGUAGE.md section 6). Everything else stays deferred and
is evaluated once per completed witness.

Group motifs (`group by ... over ...`) become a :class:`GroupPlan`: key and
member roles plus window and thresholds; the executor (``groups.py``) does
one bounded anchor-touched lookup and evaluates `having`/`soft` once per
touched group. The 2x2 `bipartite_butterfly` becomes a :class:`ButterflyPlan`:
a bounded 4-edge join from the anchor. Anything else multi-pattern still
raises a clean E_PLAN.

Usage:
    from motifql.parser import parse_file
    from motifql.planner import compile_motif
    plan = compile_motif(parse_file("examples/amlworld.motif"), "round_trip")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from motifql import checks
from motifql.degrees import STORED_FIELDS, resolve_term_args
from motifql.parser import (
    AllArgs, AmountConsCond, And, Approx, Arith, Call, Comparison, DistinctArg, Document, Dur,
    GapTermCond, IsTerm, Money, Motif, Num, Pct, Quantified, Ref,
)

log = logging.getLogger("planner")

# No static cost gate (#15 #5): mean-degree expansion estimates were wrong
# by orders of magnitude (layering measured ~1e3 against a ~1e8 estimate)
# and refused feasible motifs. Cost is bounded at runtime by the per-step
# fan-out and per-anchor expansion caps, with truncation flagged per
# result; the evaluation reports truncation rates instead.


class PlanError(Exception):
    """Motif is valid but not executable by this planner (with what to do)."""

    def __init__(self, code: str, motif: str, problem: str, fix: str) -> None:
        super().__init__(f"[{code}] {motif}: {problem} Fix: {fix}")
        self.code = code
        self.motif = motif
        self.problem = problem
        self.fix = fix


# -- plan types ----------------------------------------------------------

@dataclass(frozen=True)
class PrunableGroup:
    """One quantified conjunct: label for explanations plus its step atoms."""

    label: str
    atoms: tuple  # tuple[GapTermCond | AmountConsCond, ...]


@dataclass(frozen=True)
class PathPlan:
    """Executable plan for one single-pattern path motif."""

    name: str
    edge_var: str
    closed: bool  # anchor receiver closes the walk (cycle)
    min_hops: int
    max_hops: int
    window_s: float
    prunable: tuple[PrunableGroup, ...]
    deferred: object | None  # soft AST evaluated on completion (None = all pruned)
    score_tnorm: str
    emit: float


@dataclass(frozen=True)
class GroupPlan:
    """Executable plan for one group motif (Oct 30).

    `kind` is one of "fan" (single-edge fan_out/fan_in/structuring),
    "scatter" (s -> m -> d chain) or "gather" (x -> c, c -> y pair).
    `roles` maps role names to variable names, e.g. fan:
    {"key": "s", "member": "r", "edge": "e", "dir": "out"};
    scatter: {"s":..,"m":..,"d":..,"leg1":..,"leg2":..};
    gather: {"c":..,"x":..,"y":..,"g":..,"p":..}.
    `having`/`soft`/`where` carry the ASTs; the executor in groups.py
    evaluates the v0-covered shapes and the planner rejects the rest.
    """

    name: str
    kind: str
    roles: tuple  # tuple[(role, var), ...]
    keys: tuple[str, ...]
    members: tuple[str, ...]
    window_s: float
    having: object | None
    soft: object | None
    where: object | None
    score_tnorm: str
    emit: float

    def role(self, name: str) -> str:
        """Variable name for a role (KeyError when the kind lacks it)."""
        return dict(self.roles)[name]


@dataclass(frozen=True)
class ButterflyPlan:
    """Executable plan for the 2x2 bipartite core (Oct 30).

    Four single-edge patterns covering every (xi, yj) combo; the anchor is
    whichever edge arrives, the other three are looked up within the window.
    """

    name: str
    edge_vars: tuple[str, ...]  # (e1, e2, e3, e4)
    senders: tuple[str, str]  # (x1, x2)
    receivers: tuple[str, str]  # (y1, y2)
    window_s: float
    soft: object | None
    score_tnorm: str
    emit: float


# -- soft classification ---------------------------------------------------

def _split_conjuncts(expr: object) -> list:
    if isinstance(expr, And):
        out: list = []
        for item in expr.items:
            out.extend(_split_conjuncts(item))
        return out
    return [expr]


def _classify_soft(motif: Motif, edge_var: str) -> tuple[tuple[PrunableGroup, ...], object | None]:
    """Split soft into prunable per-step groups and a deferred remainder."""
    if motif.soft is None:
        return (), None
    prunable: list[PrunableGroup] = []
    deferred: list = []
    for conjunct in _split_conjuncts(motif.soft):
        atoms = _as_prunable_atoms(conjunct, edge_var)
        if atoms is not None:
            prunable.append(PrunableGroup(_group_label(conjunct), tuple(atoms)))
        else:
            deferred.append(conjunct)
    rest = None
    if deferred:
        rest = deferred[0] if len(deferred) == 1 else And(tuple(deferred))
    return tuple(prunable), rest


def _group_label(conjunct: Quantified) -> str:
    """Human label for one prunable group, e.g. `each e: e.gap is soon`."""
    descs: list[str] = []
    items = conjunct.body.items if isinstance(conjunct.body, And) else [conjunct.body]
    for item in items:
        if isinstance(item, IsTerm):
            base = item.base
            field = ".".join(base.parts[1:]) if isinstance(base, Ref) else "?"
            descs.append(f"{field} is {'not ' if item.negated else ''}{item.term}")
        elif isinstance(item, Approx):
            descs.append(f"amount ~= prev.amount within {_fmt_tol(item.tol)}")
    return f"{conjunct.quant} {conjunct.var}: {' and '.join(descs)}"


def _fmt_tol(tol: object) -> str:
    if isinstance(tol, Pct):
        return f"{tol.fraction * 100:g}%"
    if isinstance(tol, Money):
        return f"{tol.amount:g} {tol.currency}"
    if isinstance(tol, Dur):
        s = tol.seconds
        for unit, mult in (("w", 604800.0), ("d", 86400.0), ("h", 3600.0), ("m", 60.0), ("s", 1.0)):
            if s >= mult and s % mult == 0:
                return f"{s / mult:g}{unit}"
        return f"{s:g}s"
    if isinstance(tol, Num):
        return f"{tol.value:g}"
    return "?"


def _as_prunable_atoms(conjunct: object, edge_var: str) -> list | None:
    """Decompose `each/all e: <per-step preds>`; None when not prunable."""
    if not (isinstance(conjunct, Quantified)
            and conjunct.quant in ("each", "all")
            and conjunct.var == edge_var):
        return None
    body = conjunct.body
    items = body.items if isinstance(body, And) else [body]
    atoms: list = []
    for item in items:
        atom = _as_step_atom(item, edge_var)
        if atom is None:
            return None
        atoms.append(atom)
    return atoms


def _as_step_atom(item: object, edge_var: str):
    """One per-step predicate, or None when it needs the whole witness."""
    if isinstance(item, IsTerm):
        base = item.base
        if isinstance(base, Ref) and base.parts == (edge_var, "gap"):
            return GapTermCond(edge_var, item.term, item.args, item.negated)
        return None
    if isinstance(item, Approx):
        if (isinstance(item.left, Ref) and item.left.parts == (edge_var, "amount")
                and isinstance(item.right, Ref) and item.right.parts == ("prev", "amount")):
            return AmountConsCond(edge_var, item.tol)
        # Gap/time consistency (e.g. e.gap ~= prev.gap) stays deferred:
        # AmountConsCond scores amounts only, and its pruning bound would
        # not upper-bound the true gap/time score.
        return None
    return None


# -- entry point -----------------------------------------------------------

def compile_motif(doc: Document, name: str):
    """Check, then plan one motif. Raises CheckError-list or PlanError."""
    motif = next((m for m in doc.motifs if m.name == name), None)
    if motif is None:
        raise PlanError("E_PLAN", name, f"no motif `{name}` in file.",
                        "check the name against the motifs defined.")
    errors = checks.check_document(doc)
    by_motif = [e for e in errors if e.motif == name]
    if by_motif:
        raise PlanError("E_CHECK", name,
                        "; ".join(e.message() for e in by_motif[:3]),
                        "fix the static errors first.")
    if motif.group is not None:
        return _compile_group(doc, motif)
    if len(motif.patterns) != 1:
        return _compile_butterfly(doc, motif)
    (pat,) = motif.patterns
    if len(pat.edges) != 1 or pat.edges[0].max_hops is None:
        raise PlanError("E_PLAN", name, "the single edge must be variable-length `*m..n`.",
                        "write the edge as `-[e:transfer*m..n]->`.")
    edge = pat.edges[0]
    if edge.min_hops is not None and edge.min_hops < 2:
        raise PlanError("E_PLAN", name, "single-hop path witnesses (`*1..n`) are not carried.",
                        "write `*2..n`: every witness needs the anchor plus at least one more edge.")
    if edge.direction == "in":
        raise PlanError("E_PLAN", name, "in-edges (`<-[e]-`) are not carried by the backward walk yet.",
                        "reverse the arrow so edges point away from the anchor, or wait for Oct 30.")
    window_s = _window_of(motif)
    closed = pat.nodes[0].var == pat.nodes[-1].var
    end_vars = (pat.nodes[0].var, pat.nodes[-1].var)
    account_vars = {n.var for n in pat.nodes}
    _require_time_ordered(motif, edge.var)
    _validate_where(motif, edge.var, closed, end_vars, account_vars)
    _validate_soft(motif, edge.var, doc)
    prunable, deferred = _classify_soft(motif, edge.var)
    max_hops = edge.max_hops
    if not closed and deferred is None and max_hops > edge.min_hops:
        # G (#15): open chains with no deferred soft are decided at min
        # hops. Every longer witness's min_hops suffix is itself an
        # accepted witness scoring at least as high (a min over fewer
        # steps), and accept-before-descend records it first — so
        # deeper search only adds expansions and truncation, never a
        # better score or a different witness. Plans carry the
        # effective bound; scores and witnesses are unchanged.
        log.info("%s: open chain without deferred soft, searching *%d..%d as *%d..%d",
                 motif.name, edge.min_hops, max_hops, edge.min_hops, edge.min_hops)
        max_hops = edge.min_hops
    return PathPlan(motif.name, edge.var, closed, edge.min_hops, max_hops,
                    window_s, prunable, deferred, motif.score, motif.emit)


def _require_time_ordered(motif: Motif, edge_var: str) -> None:
    """Path motifs run strictly time-ordered, so the text must say so.

    The backward walk enforces ordering structurally, which means a motif
    without the clause would silently mean more than it says. Requiring it
    keeps the text honest for the repair loop.
    """
    for conjunct in _split_conjuncts(motif.where):
        if (isinstance(conjunct, IsTerm) and conjunct.term == "time_ordered"
                and isinstance(conjunct.base, Ref)
                and conjunct.base.parts == (edge_var,) and not conjunct.args):
            return
    raise PlanError("E_PLAN", motif.name,
                    f"no `{edge_var} is time_ordered` in `where`.",
                    f"state `{edge_var} is time_ordered`: the walk enforces it anyway.")


def _validate_where(motif: Motif, edge_var: str, closed: bool,
                    end_vars: tuple[str, str], account_vars: set[str]) -> None:
    """Reject `where` conjuncts the walk does not carry out.

    Carried forms: `e is time_ordered`, the span bound, and end-node
    identity that agrees with the walk (`a != z` on open chains, where the
    walk structurally guarantees distinct ends). Anything else — amount
    filters, bank checks, contradictions like `a == z` on an open chain —
    compiles to a plan that ignores it, so it is an E_PLAN, not a warning.
    """
    if motif.where is None:
        return
    for conjunct in _split_conjuncts(motif.where):
        if isinstance(conjunct, IsTerm) and conjunct.term == "time_ordered":
            if (isinstance(conjunct.base, Ref) and conjunct.base.parts == (edge_var,)
                    and not conjunct.args):
                continue
        elif isinstance(conjunct, Comparison) and conjunct.op in ("<=", "<"):
            if (isinstance(conjunct.left, Call) and conjunct.left.name == "span"
                    and isinstance(conjunct.right, (Dur, Num))):
                if conjunct.op == "<":
                    raise PlanError("E_PLAN", motif.name,
                                    f"`span(...) < {_describe(conjunct.right)}` is strict, but the walk applies the bound inclusively.",
                                    "use `<=`: a span exactly on the bound counts as inside.")
                continue
        elif isinstance(conjunct, Comparison) and conjunct.op in ("==", "!="):
            left, right = conjunct.left, conjunct.right
            if (isinstance(left, Ref) and isinstance(right, Ref)
                    and len(left.parts) == 1 and len(right.parts) == 1
                    and left.parts[0] in account_vars and right.parts[0] in account_vars):
                ends = {end_vars[0], end_vars[1]}
                if closed:
                    raise PlanError("E_PLAN", motif.name,
                                    f"`{left.parts[0]} {conjunct.op} {right.parts[0]}` is not checked on closed walks.",
                                    "the walk closes by construction; drop the condition.")
                if conjunct.op == "!=" and {left.parts[0], right.parts[0]} == ends:
                    continue  # structurally guaranteed: the walk never revisits
                raise PlanError("E_PLAN", motif.name,
                                f"`{left.parts[0]} {conjunct.op} {right.parts[0]}` contradicts the open chain.",
                                "the walk guarantees distinct ends; drop the condition.")
        raise PlanError("E_PLAN", motif.name,
                        f"hard predicate `{_describe(conjunct)}` is not carried out by the walk.",
                        "keep only `e is time_ordered`, the span bound and end-node identity;"
                        " per-step hard filters land after Oct 29.")


def _describe(node: object) -> str:
    if isinstance(node, IsTerm):
        return f"{'.'.join(node.base.parts) if isinstance(node.base, Ref) else '?'} is {node.term}"
    if isinstance(node, Comparison):
        return f"{_describe(node.left)} {node.op} {_describe(node.right)}"
    if isinstance(node, Arith):
        return f"{_describe(node.left)} {node.op} {_describe(node.right)}"
    if isinstance(node, Ref):
        return ".".join(node.parts)
    if isinstance(node, Call):
        return f"{node.name}(...)"
    return type(node).__name__


def _validate_soft(motif: Motif, edge_var: str, doc: Document) -> None:
    """Reject soft predicates the evaluator cannot score.

    Every field read must be one the store carries (time/amount/gap);
    every term use must resolve to concrete numbers now (call-site args are
    literals in v0). Anything else is an E_PLAN at compile time, never a
    silent zero or a replay abort.
    """
    if motif.soft is None:
        return
    term_defs = {t.name: t for t in doc.terms}
    # span() takes the edge var or `all` as a set argument, not a degree:
    # its argument is exempt from the bare-reference rule below.
    span_args = set()
    for node in checks.walk(motif.soft):
        if isinstance(node, Call) and node.name == "span":
            span_args.update(id(a) for a in node.args)
    # Term grades over a path edge are per-step: without a quantifier over
    # the path variable there is no step to grade, so they are E_PLAN
    # (a bare `e.amount is high` has no defined meaning, and previously
    # crashed the evaluator instead).
    quantified = set()
    for node in checks.walk(motif.soft):
        if isinstance(node, Quantified) and node.var == edge_var:
            quantified.update(id(n) for n in checks.walk(node.body))
    for node in checks.walk(motif.soft):
        if id(node) in span_args:
            continue
        if isinstance(node, Ref) and len(node.parts) > 1:
            head, rest = node.parts[0], node.parts[1:]
            if len(rest) > 1 or head not in (edge_var, "prev"):
                raise PlanError("E_PLAN", motif.name,
                                f"reference `{'.'.join(node.parts)}` is not a step field.",
                                "use time, amount or gap on the path edge or prev.")
            if rest[0] not in STORED_FIELDS:
                raise PlanError("E_PLAN", motif.name,
                                f"field `{rest[0]}` is not carried by the graph store yet.",
                                "use time, amount or gap; other fields land with store growth.")
            if id(node) not in quantified and id(node) not in span_args:
                raise PlanError("E_PLAN", motif.name,
                                f"`{'.'.join(node.parts)}` reads one step but sits outside any quantifier.",
                                f"write `each {edge_var}: ...` around it, or grade a witness aggregate.")
        elif isinstance(node, Ref) and len(node.parts) == 1:
            raise PlanError("E_PLAN", motif.name,
                            f"bare `{node.parts[0]}` has no degree in soft.",
                            "compare or grade a time, amount or gap field instead.")
        elif isinstance(node, IsTerm) and node.term not in ("time_ordered",):
            if (isinstance(node.base, Ref) and len(node.base.parts) == 2
                    and node.base.parts[0] == edge_var
                    and id(node) not in quantified):
                raise PlanError("E_PLAN", motif.name,
                                f"`{'.'.join(node.base.parts)} is {node.term}` grades one step but sits outside any quantifier.",
                                f"write `each {edge_var}: ...` (or some/most) around it, or grade an aggregate.")
            try:
                resolve_term_args(node.term, node.args, term_defs)
            except ValueError as exc:
                raise PlanError("E_PLAN", motif.name, str(exc),
                                "pass literal call-site arguments for now.") from exc


def _window_of(motif: Motif) -> float:
    bounds: list[float] = []
    for node in checks.walk(motif.where):
        if isinstance(node, Comparison) and node.op in ("<=", "<"):
            if isinstance(node.left, Call) and node.left.name == "span":
                right = node.right
                if isinstance(right, Dur):
                    bounds.append(right.seconds)
                if isinstance(right, Num):
                    bounds.append(right.value)
    if bounds:
        # Every conjunct must hold, so the tightest bound wins.
        return min(bounds)
    raise PlanError("E_PLAN", motif.name, "no span bound found after checks passed.",
                    "this is a compiler bug: report it.")


# -- group + butterfly compilation (Oct 30) --------------------------------

def _compile_group(doc: Document, motif: Motif):
    """Plan one `group by ... over ...` motif; E_PLAN outside v0 coverage."""
    group = motif.group
    assert group is not None
    if group.window_s is None:
        raise PlanError("E_PLAN", motif.name, "group needs a `window D`.",
                        "write `group by k over m window 1d`.")
    if motif.score not in ("min", "product"):
        raise PlanError("E_PLAN", motif.name, f"unknown score `{motif.score}`.",
                        "use `score min` or `score product`.")
    kind, roles = _group_shape(motif)
    _validate_group_where(motif, kind, roles)
    _validate_group_having(motif, kind, roles)
    _validate_group_soft(motif, kind, roles, doc)
    return GroupPlan(motif.name, kind, tuple(roles.items()), motif.group.keys,
                     motif.group.members, group.window_s, motif.having,
                     motif.soft, motif.where, motif.score, motif.emit)


def _group_shape(motif: Motif) -> tuple[str, dict[str, str]]:
    """Infer the group kind and role vars from pattern + group clause."""
    pats = motif.patterns
    keys = motif.group.keys
    members = motif.group.members
    if len(pats) == 1 and len(pats[0].edges) == 1:
        (pat,) = pats
        edge = pat.edges[0]
        if edge.max_hops is not None:
            raise PlanError("E_PLAN", motif.name,
                            "group fan edges must be single-hop.",
                            "write `-[e:transfer]->` without `*m..n`.")
        if edge.direction == "in":
            raise PlanError("E_PLAN", motif.name,
                            "in-edges (`<-[e]-`) are not carried yet.",
                            "reverse the arrow so it points away from the anchor side.")
        s_var, r_var = pat.nodes[0].var, pat.nodes[1].var
        if set(keys) == {s_var} and tuple(members) == (r_var,):
            return "fan", {"key": s_var, "member": r_var, "edge": edge.var,
                           "dir": "out-key-sender"}
        if set(keys) == {r_var} and tuple(members) == (s_var,):
            return "fan", {"key": r_var, "member": s_var, "edge": edge.var,
                           "dir": "out-key-receiver"}
        raise PlanError("E_PLAN", motif.name,
                        f"single-edge group must key on one end and range over the other, got keys={keys} members={members}.",
                        "use `group by s over r` or `group by r over s` for `(s) -[e]-> (r)`.")
    if len(pats) == 1 and len(pats[0].edges) == 2:
        (pat,) = pats
        if {n.var for n in pat.nodes} != set(keys) | set(members):
            raise PlanError("E_PLAN", motif.name, "scatter roles do not cover the pattern.",
                            "use `group by s, d over m` for `(s) -[leg1]-> (m) -[leg2]-> (d)`.")
        if set(keys) == {pat.nodes[0].var, pat.nodes[2].var} and tuple(members) == (pat.nodes[1].var,):
            e1, e2 = pat.edges
            if e1.max_hops is not None or e2.max_hops is not None:
                raise PlanError("E_PLAN", motif.name,
                                "group scatter legs must be single-hop.",
                                "write `(s) -[leg1]-> (m) -[leg2]-> (d)` without `*m..n`.")
            if e1.direction != "out" or e2.direction != "out":
                raise PlanError("E_PLAN", motif.name, "scatter legs must point forward.",
                                "write `(s) -[leg1]-> (m) -[leg2]-> (d)`.")
            s_var, m_var, d_var = (n.var for n in pat.nodes)
            return "scatter", {"s": s_var, "m": m_var, "d": d_var,
                               "leg1": e1.var, "leg2": e2.var}
        raise PlanError("E_PLAN", motif.name, "two-leg groups must key on the ends and range over the middle.",
                        "use `group by s, d over m` for the chain pattern.")
    if len(pats) == 2 and all(len(p.edges) == 1 for p in pats):
        shared = {n.var for n in pats[0].nodes} & {n.var for n in pats[1].nodes}
        if len(shared) != 1:
            raise PlanError("E_PLAN", motif.name, "gather patterns must share exactly one account.",
                            "write `(x) -[g]-> (c), (c) -[p]-> (y)` sharing `c`.")
        (c_var,) = shared
        if tuple(keys) != (c_var,) or set(members) != (
                ({n.var for n in pats[0].nodes} | {n.var for n in pats[1].nodes}) - {c_var}):
            raise PlanError("E_PLAN", motif.name, "gather must `group by c over x, y` around the shared account.",
                            "key on the shared account, range over the two ends.")
        # The pattern whose receiver is c is the gather leg.
        try:
            gpat = next(p for p in pats if p.nodes[1].var == c_var)
            ppat = next(p for p in pats if p.nodes[0].var == c_var)
        except StopIteration:
            raise PlanError("E_PLAN", motif.name, "gather legs must meet at the shared account.",
                            "write `(x) -[g]-> (c), (c) -[p]-> (y)`.")
        if gpat is ppat:
            raise PlanError("E_PLAN", motif.name, "gather legs must meet at the shared account.",
                            "write `(x) -[g]-> (c), (c) -[p]-> (y)`.")
        if gpat.edges[0].direction != "out" or ppat.edges[0].direction != "out":
            raise PlanError("E_PLAN", motif.name, "gather legs must point forward.",
                            "write `(x) -[g]-> (c), (c) -[p]-> (y)`.")
        if gpat.edges[0].max_hops is not None or ppat.edges[0].max_hops is not None:
            raise PlanError("E_PLAN", motif.name,
                            "group gather legs must be single-hop.",
                            "write `(x) -[g]-> (c), (c) -[p]-> (y)` without `*m..n`.")
        x_var = gpat.nodes[0].var
        y_var = ppat.nodes[1].var
        return "gather", {"c": c_var, "x": x_var, "y": y_var,
                          "g": gpat.edges[0].var, "p": ppat.edges[0].var}
    raise PlanError("E_PLAN", motif.name,
                    f"{len(pats)} patterns with {[len(p.edges) for p in pats]} edges: not a v0 group shape.",
                    "use a single-edge fan, an s->m->d chain, or an x->c/c->y pair.")


def _validate_group_where(motif: Motif, kind: str, roles: dict[str, str]) -> None:
    """Only the structural where-forms the executor carries."""
    if motif.where is None:
        if kind == "fan":
            return
        raise PlanError("E_PLAN", motif.name, f"{kind} needs its ordering/identity guard in `where`.",
                        "keep `leg2.time > leg1.time and s != d` (scatter) or `p.time > g.time` (gather).")
    if kind == "fan":
        raise PlanError("E_PLAN", motif.name,
                        f"hard predicate `{_describe(motif.where)}` is not carried for fan groups.",
                        "drop `where` on single-edge groups; `having` already gates them.")
    ok_order = ok_ident = False
    if kind == "scatter":
        want_gt = (roles["leg2"], "time", roles["leg1"], "time")
        want_neq = {roles["s"], roles["d"]}
    elif kind == "gather":
        want_gt = (roles["p"], "time", roles["g"], "time")
        want_neq = set()
    else:
        raise PlanError("E_PLAN", motif.name, f"unknown group kind `{kind}`.",
                        "use fan, scatter or gather shapes.")
    for conjunct in _split_conjuncts(motif.where):
        if (isinstance(conjunct, Comparison) and conjunct.op == ">"
                and isinstance(conjunct.left, Ref) and isinstance(conjunct.right, Ref)
                and tuple(conjunct.left.parts) == (want_gt[0], want_gt[1])
                and tuple(conjunct.right.parts) == (want_gt[2], want_gt[3])):
            ok_order = True
            continue
        if (kind == "scatter" and isinstance(conjunct, Comparison) and conjunct.op == "!="
                and isinstance(conjunct.left, Ref) and isinstance(conjunct.right, Ref)
                and len(conjunct.left.parts) == 1 and len(conjunct.right.parts) == 1
                and {conjunct.left.parts[0], conjunct.right.parts[0]} == want_neq):
            ok_ident = True
            continue
        raise PlanError("E_PLAN", motif.name,
                        f"hard predicate `{_describe(conjunct)}` is not carried for {kind} groups.",
                        "keep only the time ordering and end-node identity guards.")
    if not ok_order or (kind == "scatter" and not ok_ident):
        raise PlanError("E_PLAN", motif.name, f"{kind} `where` must carry the ordering guard.",
                        "keep `leg2.time > leg1.time` (scatter) or `p.time > g.time` (gather).")


def _validate_group_having(motif: Motif, kind: str, roles: dict[str, str]) -> None:
    """Having must be count-distinct thresholds over member vars."""
    if motif.having is None:
        raise PlanError("E_PLAN", motif.name, "group motifs need `having` count gates.",
                        "add e.g. `having count(distinct r) >= 2`.")
    members = set(motif.group.members)
    for conjunct in _split_conjuncts(motif.having):
        if not (isinstance(conjunct, Comparison) and conjunct.op in (">=", ">")
                and isinstance(conjunct.left, Call) and conjunct.left.name == "count"
                and len(conjunct.left.args) == 1
                and isinstance(conjunct.left.args[0], DistinctArg)
                and isinstance(conjunct.left.args[0].ref, Ref)
                and len(conjunct.left.args[0].ref.parts) == 1
                and conjunct.left.args[0].ref.parts[0] in members
                and isinstance(conjunct.right, Num)):
            raise PlanError("E_PLAN", motif.name,
                            f"`{_describe(conjunct)}` is not a member-count gate.",
                            "use `count(distinct <member>) >= N` conjuncts in `having`.")
    # All conjuncts passed the shape above (the raise fires on mismatch).


def _validate_group_soft(motif: Motif, kind: str, roles: dict[str, str], doc: Document) -> None:
    """Soft must use only the v0-covered group forms (aggregates + members)."""
    if motif.soft is None:
        return
    edge_vars = {r for k, r in roles.items() if k in ("edge", "leg1", "leg2", "g", "p")}
    member_vars = set(motif.group.members)
    term_defs = {t.name: t for t in doc.terms}
    span_args = set()
    for node in checks.walk(motif.soft):
        if isinstance(node, Call) and node.name == "span":
            span_args.update(id(a) for a in node.args)
    for node in checks.walk(motif.soft):
        if id(node) in span_args:
            continue
        if isinstance(node, Ref):
            if len(node.parts) == 1:
                if node.parts[0] in member_vars or node.parts[0] in edge_vars:
                    continue
                raise PlanError("E_PLAN", motif.name,
                                f"bare `{node.parts[0]}` has no degree in soft.",
                                "grade a field (`e.amount`) or an aggregate (`count(...)`).")
            head, rest = node.parts[0], node.parts[1:]
            if head == "prev":
                raise PlanError("E_PLAN", motif.name,
                                "`prev` needs a path quantifier, which groups have none of.",
                                "compare the group's edge amounts/times directly.")
            if head in edge_vars and len(rest) == 1 and rest[0] in STORED_FIELDS:
                continue
            if head in ("avg", "sum", "min", "max"):
                # `avg(e.amount)`-style call args surface as Refs under Call.
                continue
            if head in member_vars:
                # F4: member readings only carry edge fields, so a
                # member-headed ref never resolves — every member is
                # skipped and the quantifier returns a silent constant.
                raise PlanError("E_PLAN", motif.name,
                                f"reference `{'.'.join(node.parts)}` reads a member account field, "
                                "which group readings do not carry.",
                                "grade edge fields (`e.amount`, `leg2.time`) or aggregates.")
            raise PlanError("E_PLAN", motif.name,
                            f"reference `{'.'.join(node.parts)}` is not a group field.",
                            "use the group edge/member amount, time or gap fields.")
        elif isinstance(node, Quantified):
            if node.var not in member_vars:
                raise PlanError("E_PLAN", motif.name,
                                f"quantifier over `{node.var}`, which is not a group member.",
                                f"quantify over one of {sorted(member_vars)}.")
        elif isinstance(node, Call):
            if node.name in ("count", "sum", "avg", "min", "max", "span"):
                continue
            raise PlanError("E_PLAN", motif.name,
                            f"function `{node.name}` is not a group aggregate.",
                            "use count, sum, avg, min, max or span.")
        elif isinstance(node, IsTerm) and node.term not in ("time_ordered",):
            try:
                resolve_term_args(node.term, node.args, term_defs)
            except ValueError as exc:
                raise PlanError("E_PLAN", motif.name, str(exc),
                                "pass literal call-site arguments for now.") from exc


def _compile_butterfly(doc: Document, motif: Motif):
    """Plan the 2x2 bipartite core; anything else multi-pattern is E_PLAN."""
    if len(motif.patterns) != 4 or any(len(p.edges) != 1 or len(p.nodes) != 2 for p in motif.patterns):
        raise PlanError("E_PLAN", motif.name,
                        f"{len(motif.patterns)} match patterns: only the 2x2 butterfly is covered.",
                        "write four single-edge patterns `(x1)->(y1), (x1)->(y2), (x2)->(y1), (x2)->(y2)`.")
    if motif.group is not None:
        raise PlanError("E_PLAN", motif.name, "the butterfly takes no `group` clause.",
                        "drop the group clause; the 2x2 core is already bounded.")
    senders: dict[str, list[str]] = {}
    receivers: dict[str, list[str]] = {}
    edge_of: dict[tuple[str, str], str] = {}
    for pat in motif.patterns:
        (edge,) = pat.edges
        if edge.max_hops is not None:
            raise PlanError("E_PLAN", motif.name, "butterfly edges must be single-hop.",
                            "write `-[e:transfer]->` without `*m..n`.")
        if edge.direction == "in":
            raise PlanError("E_PLAN", motif.name, "butterfly in-edges are not carried.",
                            "point every edge forward `(x) -[e]-> (y)`.")
        s_var, r_var = pat.nodes[0].var, pat.nodes[1].var
        senders.setdefault(s_var, []).append(r_var)
        receivers.setdefault(r_var, []).append(s_var)
        if (s_var, r_var) in edge_of:
            raise PlanError("E_PLAN", motif.name,
                            f"duplicate butterfly leg ({s_var})->({r_var}).",
                            "cover each (xi, yj) combo exactly once.")
        edge_of[(s_var, r_var)] = edge.var
    if set(senders) != set(receivers) and len(senders) == 2 and len(receivers) == 2:
        pass
    xs = sorted(senders)
    ys = sorted(receivers)
    if (len(xs) != 2 or len(ys) != 2 or any(sorted(v) != ys for v in senders.values())
            or any(sorted(v) != xs for v in receivers.values())):
        raise PlanError("E_PLAN", motif.name, "butterfly legs must form a complete 2x2 core.",
                        "write `(x1)->(y1), (x1)->(y2), (x2)->(y1), (x2)->(y2)`.")
    x1, x2 = xs
    y1, y2 = ys
    try:
        e1, e2, e3, e4 = (edge_of[(x1, y1)], edge_of[(x1, y2)],
                          edge_of[(x2, y1)], edge_of[(x2, y2)])
    except KeyError:
        raise PlanError("E_PLAN", motif.name, "butterfly legs must form a complete 2x2 core.",
                        "write `(x1)->(y1), (x1)->(y2), (x2)->(y1), (x2)->(y2)`.")
    _validate_butterfly_where(motif, x1, x2, y1, y2)
    _validate_butterfly_soft(motif, doc)
    window_s = _window_of(motif)
    return ButterflyPlan(motif.name, (e1, e2, e3, e4), (x1, x2), (y1, y2),
                         window_s, motif.soft, motif.score, motif.emit)


def _validate_butterfly_where(motif: Motif, x1: str, x2: str, y1: str, y2: str) -> None:
    if motif.where is None:
        raise PlanError("E_PLAN", motif.name, "the butterfly needs its guards in `where`.",
                        "keep `x1 != x2 and y1 != y2 and span(all) <= 1d`.")
    need_neq = [{x1, x2}, {y1, y2}]
    seen_neq: list[set] = []
    seen_span = False
    for conjunct in _split_conjuncts(motif.where):
        if (isinstance(conjunct, Comparison) and conjunct.op == "!="
                and isinstance(conjunct.left, Ref) and isinstance(conjunct.right, Ref)
                and len(conjunct.left.parts) == 1 and len(conjunct.right.parts) == 1):
            pair = {conjunct.left.parts[0], conjunct.right.parts[0]}
            if pair in need_neq:
                seen_neq.append(pair)
                continue
        elif (isinstance(conjunct, Comparison) and conjunct.op in ("<=", "<")
                and isinstance(conjunct.left, Call) and conjunct.left.name == "span"
                and isinstance(conjunct.right, (Dur, Num))):
            if conjunct.op == "<":
                raise PlanError("E_PLAN", motif.name,
                                "`span(all) < ...` is strict, but the join applies the bound inclusively.",
                                "use `<=`: a span exactly on the bound counts as inside.")
            seen_span = True
            continue
        raise PlanError("E_PLAN", motif.name,
                        f"hard predicate `{_describe(conjunct)}` is not carried for the butterfly.",
                        "keep only `x1 != x2`, `y1 != y2` and the `span(all)` bound.")
    if sorted(map(sorted, seen_neq)) != sorted(map(sorted, need_neq)) or not seen_span:
        raise PlanError("E_PLAN", motif.name, "the butterfly `where` must guard identities and span.",
                        "keep `x1 != x2 and y1 != y2 and span(all) <= 1d`.")


def _validate_butterfly_soft(motif: Motif, doc: Document) -> None:
    if motif.soft is None:
        return
    term_defs = {t.name: t for t in doc.terms}
    soft = motif.soft
    if (isinstance(soft, IsTerm) and isinstance(soft.base, Call)
            and soft.base.name == "span" and len(soft.base.args) == 1
            and isinstance(soft.base.args[0], AllArgs)
            and soft.term in term_defs and soft.term != "time_ordered"):
        try:
            resolve_term_args(soft.term, soft.args, term_defs)
        except ValueError as exc:
            raise PlanError("E_PLAN", motif.name, str(exc),
                            "pass literal call-site arguments for now.") from exc
        return
    raise PlanError("E_PLAN", motif.name,
                    f"soft `{_describe(soft)}` is not carried for the butterfly.",
                    "score the core with one `span(all) is <term>` condition.")
