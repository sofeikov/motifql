"""MotifQL planner: AST to executable path plans (Oct 29, compiler step 4).

For a single-pattern path motif the plan fixes the anchored backward
search: hop bounds, the anchor-relative window, and the soft conditions
split into prunable vs completion-only. Prunable conditions are
final-on-bind per-step predicates under `each`/`all` (gap terms and
amount-vs-prev tolerances); their running degree is an upper bound on the
final score, so the matcher cuts a branch the moment it drops below the
emit threshold (LANGUAGE.md section 6). Everything else stays deferred and
is evaluated once per completed witness.

Group motifs and multi-pattern motifs are rejected with a clean error:
their execution paths land Oct 30.

Usage:
    from motifql.parser import parse_file
    from motifql.planner import compile_motif
    plan = compile_motif(parse_file("examples/amlworld.motif"), "round_trip")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from motifql import checks
from motifql.parser import (
    And, Approx, Call, Comparison, Document, Dur, IsTerm, Money, Motif, Num,
    Pct, Quantified, Ref,
)

log = logging.getLogger("planner")

# Cost budget: estimated expansions per anchor above this refuse the motif.
# Calibrated so the reference motifs pass on HI-Small (round_trip ~1e5,
# layering_chain ~1e6); a hypothetical *2..8 motif estimates ~1e8 and is
# rejected with the bound to tighten.
DEFAULT_COST_BUDGET = 10_000_000


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
class GapTermCond:
    """Per-step `e.gap is <term>` (skipped, not failed, on the first step)."""

    var: str
    term: str
    term_args: tuple
    negated: bool


@dataclass(frozen=True)
class AmountConsCond:
    """Per-step `e.amount ~= prev.amount within <tol>` (skipped on step one)."""

    var: str
    tol: object  # Pct | Money | Dur | Num
    negated: bool = False


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
    est_expansions: float


# -- soft classification ---------------------------------------------------

def _split_conjuncts(expr: object) -> list:
    if isinstance(expr, And):
        return list(expr.items)
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
        if (isinstance(item.left, Ref) and item.left.parts[:1] == (edge_var,)
                and isinstance(item.right, Ref) and item.right.parts[0] == "prev"):
            return AmountConsCond(edge_var, item.tol)
        return None
    return None


# -- entry point -----------------------------------------------------------

def compile_motif(doc: Document, name: str, stats=None) -> PathPlan:
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
        raise PlanError("E_PLAN", name, "group motifs execute Oct 30.",
                        "use a single-pattern path motif for now.")
    if len(motif.patterns) != 1:
        raise PlanError("E_PLAN", name,
                        f"{len(motif.patterns)} match patterns: only single-pattern path motifs execute yet.",
                        "use one `(a) -[e*..]-> (b)` pattern; multi-pattern matching lands Oct 30.")
    (pat,) = motif.patterns
    if len(pat.edges) != 1 or pat.edges[0].max_hops is None:
        raise PlanError("E_PLAN", name, "the single edge must be variable-length `*m..n`.",
                        "write the edge as `-[e:transfer*m..n]->`.")
    edge = pat.edges[0]
    window_s = _window_of(motif)
    prunable, deferred = _classify_soft(motif, edge.var)
    closed = pat.nodes[0].var == pat.nodes[-1].var
    est = estimate_cost(edge.max_hops, stats) if stats is not None else -1.0
    if stats is not None and est > DEFAULT_COST_BUDGET:
        raise PlanError("E_COST", name,
                        f"estimated {est:,.0f} expansions/anchor over budget {DEFAULT_COST_BUDGET:,}.",
                        f"tighten `*{edge.min_hops}..{edge.max_hops}` or the window.")
    return PathPlan(motif.name, edge.var, closed, edge.min_hops, edge.max_hops,
                    window_s, prunable, deferred, motif.score, motif.emit, est)


def _window_of(motif: Motif) -> float:
    for node in checks.walk(motif.where):
        if isinstance(node, Comparison) and node.op in ("<=", "<"):
            if isinstance(node.left, Call) and node.left.name == "span":
                right = node.right
                if isinstance(right, Dur):
                    return right.seconds
                if isinstance(right, Num):
                    return right.value
    raise PlanError("E_PLAN", motif.name, "no span bound found after checks passed.",
                    "this is a compiler bug: report it.")


def estimate_cost(max_hops: int, stats, fanout: int = 1_000) -> float:
    """Estimated expansions per anchor: sum of mean in-degree powers.

    `stats` is the `degree_histogram()` array (index = in+out degree,
    value = node count). Uses mean in-degree ≈ mean total / 2, capped per
    level by the fanout cap and overall by the window edge count when the
    caller supplies it separately.
    """
    import numpy as np

    hist = np.asarray(stats, dtype=np.float64)
    total = hist.sum()
    if total == 0:
        return 0.0
    mean_deg = (np.arange(len(hist)) * hist).sum() / total / 2.0
    branch = min(mean_deg, fanout)
    return float(sum(branch ** i for i in range(1, max_hops + 1)))
