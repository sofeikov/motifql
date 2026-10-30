"""Exact anchored path matchers (Oct 28: crisp search, no fuzzy scores).

Both v0 path motifs are anchored backward depth-first searches from the
anchor transfer's sender over strictly earlier transfers:

* ``ROUND_TRIP`` (cycle): close on the anchor's receiver within 2-5 hops
  and a 7-day span, simple path (no account revisited before closing).
* ``LAYERING_CHAIN`` (open chain): end anywhere within 3-6 hops and a
  2-day span, provided the chain's start differs from the anchor's
  receiver; fully simple path.

Hub control follows LANGUAGE.md section 6.5: ``max_fanout`` caps how many
of a node's incoming edges one search step considers (most recent first),
``max_expansions`` caps total node expansions per anchor. Hitting either
marks the result ``truncated`` and the search stops -- the witness, when
one is found despite the caps, is still returned. Exact search returns the
first (most-recent) witness; ranking witnesses by degree is Oct 29 work,
when ``soft`` conditions arrive with the compiler.

Execution: ``match``, ``find_all`` and ``score`` run the walk in a numba
kernel (``motifql._pathkernel``) whenever it models the plan, and fall
back to the pure-Python walk below otherwise (deferred soft, graphs whose
``csr_ordered`` flag is False, ``find_all`` caps the kernel cannot
mirror). The two are bit-identical (tests/test_pathkernel_parity.py,
eval/parity_check_path.py); ``MOTIFQL_PURE_PYTHON=1`` or
``PURE_PYTHON = True`` forces the reference. ``score_path_batch`` scores
many anchors in one kernel call (optionally ``parallel=True``, which
starts numba threads: never in a process that forks afterwards).
Nothing is memoised between calls (see the fast-path section below).

Usage:
    from motifql.graph import TemporalGraph
    from motifql.matcher import ROUND_TRIP, match
    graph = TemporalGraph.from_parquet("data/hi-small/transfers.parquet")
    res = match(graph, anchor_pos=123, motif=ROUND_TRIP)
    if res.found: print(res.edges, res.truncated)
    batch = score_path_batch(graph, positions, plan, libs, floor=EXACT_FLOOR)
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from motifql.degrees import eval_soft, resolve_term_args, step_degrees
from motifql.graph import SECS_PER_DAY, TemporalGraph
from motifql.parser import AmountConsCond, Dur, GapTermCond, Money, Num, Pct

log = logging.getLogger("matcher")

# Search budget defaults: generous enough that ordinary accounts never hit
# them, tight enough that the bank-070 senders terminate.
DEFAULT_MAX_FANOUT = 1_000
DEFAULT_MAX_EXPANSIONS = 50_000

#: Force the pure-Python reference walk everywhere (parity testing). Set
#: ``MOTIFQL_PURE_PYTHON=1`` before import, or assign this at runtime.
#: The numba kernel (``motifql._pathkernel``) is a compute-only port: it
#: returns bit-identical results and caches no scores between calls.
PURE_PYTHON = os.environ.get("MOTIFQL_PURE_PYTHON", "") not in ("", "0")


@dataclass(frozen=True)
class PathMotif:
    """Crisp parameters of one anchored path search."""

    name: str
    min_hops: int
    max_hops: int
    window_s: int
    closed: bool  # True: must close on the anchor receiver (cycle).
    max_fanout: int = DEFAULT_MAX_FANOUT
    max_expansions: int = DEFAULT_MAX_EXPANSIONS


ROUND_TRIP = PathMotif(
    name="round_trip", min_hops=2, max_hops=5,
    window_s=7 * SECS_PER_DAY, closed=True,
)
LAYERING_CHAIN = PathMotif(
    name="layering_chain", min_hops=3, max_hops=6,
    window_s=2 * SECS_PER_DAY, closed=False,
)


@dataclass
class MatchResult:
    """Outcome of scoring one anchor against one motif."""

    found: bool
    edges: tuple[int, ...]  # edge positions, oldest first; () when not found
    truncated: bool  # a hub cap cut the search short (score is a lower bound)
    expansions: int  # node expansions spent (benchmark accounting)


@dataclass
class _Frame:
    """One DFS level: candidate incoming edges of a node, most recent first."""

    node: int
    edges: npt.NDArray[np.uint32] = field(repr=False)
    next_idx: int = 0


def _capped(edges: npt.NDArray[np.uint32], times: npt.NDArray[np.int64], cap: int
            ) -> tuple[npt.NDArray[np.uint32], bool]:
    """Most-recent-first candidates, capped; flag when the cap bites."""
    order = np.argsort(times, kind="stable")[::-1]
    ranked = edges[order]
    if len(ranked) > cap:
        return ranked[:cap], True
    return ranked, False


def match(graph: TemporalGraph, anchor_pos: int, motif: PathMotif) -> MatchResult:
    """Score one anchor transfer (by edge position) against one path motif."""
    witnesses, truncated, expansions = _search(graph, anchor_pos, motif)
    if witnesses:
        return MatchResult(True, witnesses[0], truncated, expansions)
    return MatchResult(False, (), truncated, expansions)


def find_all(
    graph: TemporalGraph, anchor_pos: int, motif: PathMotif, cap: int = 1_000
) -> tuple[list[tuple[int, ...]], bool, int]:
    """Enumerate witnesses for one anchor, oldest-first plus anchor each.

    Stops after ``cap`` witnesses (flags truncated): witness counts explode
    on hub anchors, and recall accounting only needs the best overlap.
    Shares the exact search core with :func:`match`.
    """
    return _search(graph, anchor_pos, motif, collect_all=True, cap=cap)


def _search(
    graph: TemporalGraph,
    anchor_pos: int,
    motif: PathMotif,
    collect_all: bool = False,
    cap: int = 1,
) -> tuple[list[tuple[int, ...]], bool, int]:
    """Exact anchored search behind :func:`match` and :func:`find_all`."""
    if (not PURE_PYTHON and isinstance(motif.max_fanout, (int, np.integer))
            and _fast_ok(graph, anchor_pos, motif.max_fanout)
            and float(motif.window_s) == float(motif.window_s)):
        enum_cap = _enum_cap(cap, collect_all, motif.max_expansions)
        if enum_cap is not None:
            return _search_fast(graph, anchor_pos, motif, collect_all, cap, enum_cap)
    return _search_py(graph, anchor_pos, motif, collect_all, cap)


#: Largest witness buffer the fast search allocates; a larger effective
#: cap runs the pure walk instead.
_ENUM_ROWS_MAX = 1 << 16


def _enum_cap(cap, collect_all: bool, max_expansions) -> int | None:
    """Kernel witness cap that stops exactly where the pure walk's
    ``len(witnesses) >= cap`` does, or None to run the pure walk (a cap or
    budget the kernel cannot mirror, or a buffer past ``_ENUM_ROWS_MAX``).
    """
    if (isinstance(max_expansions, (bool, np.bool_))
            or not isinstance(max_expansions, (int, np.integer))):
        return None  # the kernel takes an int64 budget
    if not collect_all:
        return 1  # match(): stop at the first witness
    # Every accept follows its own expansion, so a walk never accepts more
    # than max(max_expansions, 0) witnesses: one more is a count it never
    # reaches, i.e. "do not stop on the witness count".
    unreachable = max(int(max_expansions), 0) + 1
    if isinstance(cap, (bool, np.bool_, int, np.integer)):
        k = int(cap)
    elif isinstance(cap, (float, np.floating)):
        c = float(cap)
        if c != c or c == math.inf:
            k = unreachable  # len >= nan / inf never holds
        elif c == -math.inf:
            k = 1
        else:
            k = math.ceil(c)  # len >= 2.5 first holds at 3
    else:
        return None
    k = min(max(k, 1), unreachable)  # cap <= 1: the first witness stops it
    return k if k <= _ENUM_ROWS_MAX else None


def _search_py(
    graph: TemporalGraph,
    anchor_pos: int,
    motif: PathMotif,
    collect_all: bool = False,
    cap: int = 1,
) -> tuple[list[tuple[int, ...]], bool, int]:
    """Pure-Python reference for :func:`_search`."""
    witnesses: list[tuple[int, ...]] = []

    def on_extend(parent_pos: int, cand_pos: int) -> bool:
        return True

    def on_accept(witness: tuple[int, ...]) -> bool:
        witnesses.append(witness)
        return (not collect_all) or len(witnesses) >= cap

    truncated, expansions = _walk(
        graph, anchor_pos, motif.min_hops, motif.max_hops, motif.window_s,
        motif.closed, motif.max_fanout, motif.max_expansions,
        on_extend, on_accept)
    capped = collect_all and len(witnesses) >= cap
    return witnesses, truncated or capped, expansions


def _walk(
    graph: TemporalGraph,
    anchor_pos: int,
    min_hops: int,
    max_hops: int,
    window_s: float,
    closed: bool,
    max_fanout: int,
    max_expansions: int,
    on_extend,
    on_accept,
    on_push=None,
    on_pop=None,
) -> tuple[bool, int]:
    """Shared anchored backward DFS behind exact and scored search.

    ``on_extend(parent_pos, cand_pos)`` runs when a candidate survives the
    structural checks: False prunes the branch (sound score pruning sets no
    truncation flag). ``on_push(parent_pos, cand_pos)`` runs right before a
    surviving candidate is pushed; ``on_pop()`` when that level is
    abandoned, so hook state stays aligned with the walk. ``on_accept``
    runs per completed witness, oldest-first plus the anchor; True stops
    the search. Returns (truncated, expansions).
    """
    anchor_t = int(graph.times[anchor_pos])
    sender = int(graph.frm[anchor_pos])
    receiver = int(graph.to[anchor_pos])
    if sender == receiver:
        # Self-loop anchor: the edge before the anchor already ends where
        # the anchor starts, so every longer walk repeats the account and
        # no simple-path witness with min_hops >= 2 exists, open or closed.
        return False, 0
    window_start = anchor_t - window_s
    expansions = 0
    truncated = False

    first_edges, first_times = graph.incoming(sender, anchor_t, window_s)
    if len(first_edges) == 0:
        return False, 0
    first, hit = _capped(first_edges, first_times, max_fanout)
    truncated = hit

    # path holds the back-edges taken below the anchor (path[-1] is deepest).
    # visited holds every node on the current walk, always including the
    # anchor receiver: a walk through the receiver mid-path (X -> R -> S
    # under anchor S -> R) repeats R once the anchor closes it, so it is
    # not a simple path. The prev != receiver accept check below stays as
    # a backstop for open motifs.
    stack: list[_Frame] = [_Frame(sender, first)]
    path: list[int] = []
    visited: set[int] = {sender, receiver}
    stopped = False

    def _pop_level() -> None:
        stack.pop()
        if stack:
            # The popped level was reached via path[-1]: unvisit it.
            visited.discard(int(graph.frm[path.pop()]))
            if on_pop is not None:
                on_pop()

    while stack and not stopped:
        frame = stack[-1]
        if frame.next_idx >= len(frame.edges):
            _pop_level()
            continue
        if expansions >= max_expansions:
            # Over-flagging is impossible here: untried edges remain, since
            # the frame on top (or one below it) still has work to do.
            truncated = True
            break
        edge = int(frame.edges[frame.next_idx])
        frame.next_idx += 1
        edge_t = int(graph.times[edge])
        if edge_t < window_start:
            # Most-recent-first: every later candidate is older still.
            # A window bound is not a hub cap, so no truncation flag.
            _pop_level()
            continue
        expansions += 1
        prev = int(graph.frm[edge])
        hops = len(path) + 2  # candidate edge plus the anchor
        if hops > max_hops:
            continue
        if anchor_t - edge_t > window_s:
            continue  # paths only age as they extend: safe to prune
        # The edge above the candidate becomes fully defined the moment we
        # take a step below it, so its per-step atoms are evaluated here —
        # including for the accept below, whose parent would otherwise never
        # be scored (a 2-hop witness has no other extension point).
        parent_pos = path[-1] if path else anchor_pos
        if not on_extend(parent_pos, edge):
            continue  # score-pruned: sound, so no truncation flag
        if closed and prev == receiver:
            if hops >= min_hops:
                stopped = on_accept(tuple(reversed(path + [edge])) + (anchor_pos,))
            continue  # closes too early: below min_hops, keep looking
        if prev in visited:
            continue  # simple path (the closing step above is the exception)
        if not closed and hops >= min_hops and prev != receiver:
            stopped = on_accept(tuple(reversed(path + [edge])) + (anchor_pos,))
        if stopped:
            break  # accepted and done: skip the now-pointless lookup below,
            # whose fanout cap could spuriously flag truncation.
        if hops >= max_hops:
            continue
        # Window relative to the anchor, not the edge: edges older than the
        # anchor's window start can never complete, so they must not consume
        # fanout budget or count as truncation when cut.
        nxt_edges, nxt_times = graph.incoming(prev, edge_t, edge_t - window_start)
        if len(nxt_edges) == 0:
            continue
        nxt, hit = _capped(nxt_edges, nxt_times, max_fanout)
        truncated = truncated or hit
        if on_push is not None:
            on_push(parent_pos, edge)
        visited.add(prev)
        path.append(edge)
        stack.append(_Frame(prev, nxt))

    return truncated, expansions


@dataclass
class ScoredResult:
    """Best fuzzy witness for one anchor against one path plan."""

    found: bool
    witness: tuple[int, ...]  # edge positions, oldest first, anchor included
    score: float  # best witness at or above emit, else 0.0
    degrees: tuple  # tuple[(predicate label, degree, weakest 1-based step | None)]
    truncated: bool
    expansions: int
    top_score: float = 0.0  # best completed witness even below emit (ranking)


#: Floor for exact-max search: pruning below ~0 cuts only branches
#: that complete at ~0, so pruned search returns the exact top_score
#: (mod 1e-12) while skipping zero branches. Ranking harnesses use
#: this instead of unpruned search, which additionally truncates.
EXACT_FLOOR = 1e-12


def score(graph: TemporalGraph, anchor_pos: int, plan, libs,
          prune: bool = True, crisp: bool = False, crisp_alpha: float = 0.0,
          floor: float | None = None) -> ScoredResult:
    """Score one anchor against a path plan, returning the best witness.

    Prunable group degrees accumulate bottom-up as the walk extends; their
    running combination is an upper bound on any completion, so a branch is
    cut once it drops below the best score so far (or the starting floor).
    With crisp=True every leaf is α-cut (support at α=0, `d >= α` above;
    RQ2 exact arm); crisp degrees still bound their completions, so
    pruning stays sound. `floor` overrides the starting bound (`emit` by
    default): pass EXACT_FLOOR for the exact best witness regardless of
    emit.
    """
    if not PURE_PYTHON:
        compiled = _compiled_plan(plan, libs, crisp, crisp_alpha)
        if compiled is not None and _fast_ok(graph, anchor_pos, DEFAULT_MAX_FANOUT):
            return _score_fast(graph, anchor_pos, plan, libs, compiled, prune,
                               crisp, crisp_alpha, floor)
    return _score_py(graph, anchor_pos, plan, libs, prune, crisp, crisp_alpha, floor)


def _score_py(graph: TemporalGraph, anchor_pos: int, plan, libs,
              prune: bool = True, crisp: bool = False, crisp_alpha: float = 0.0,
              floor: float | None = None) -> ScoredResult:
    """Pure-Python reference for :func:`score` (and its fallback)."""
    terms, _ = libs
    tnorm = plan.score_tnorm
    emit = plan.emit
    # run_stack parallels the walk: per pushed level, the per-group minimum
    # atom degree of the step above, evaluated against the taken candidate.
    # Stale entries are impossible: pushes and pops pair up exactly, and the
    # accept below recomputes every step fresh from the witness.
    run_stack: list[list[float]] = []
    best: tuple | None = None  # (witness, score, degrees)
    best_score = -1.0
    # Best completed witness even below emit, for ranking. Pruning is
    # sound for the floor-filtered query (a pruned branch can never
    # reach `floor`), so pruned top_score may understate the unpruned
    # max — but never above the starting floor: any witness scoring
    # above it completes under pruning too.
    top_score = 0.0
    floor = emit if floor is None else floor

    def _parent_degrees(parent_pos: int, cand_pos: int) -> list[float]:
        # Atoms of one step combine with the motif t-norm; group degrees
        # combine across steps with min for each/all (LANGUAGE.md section 3).
        t_p = int(graph.times[parent_pos])
        t_c = int(graph.times[cand_pos])
        a_p = float(graph.amount[parent_pos])
        a_c = float(graph.amount[cand_pos])
        out: list[float] = []
        for group in plan.prunable:
            ds = [step_degrees(a, t_p - t_c, a_p, a_c, terms, crisp, crisp_alpha)
                  for a in group.atoms]
            ds = [d for d in ds if d is not None]
            out.append(_combine(ds, tnorm) if ds else 1.0)
        return out

    def _bound(extra: list[float] | None = None) -> float:
        parts = []
        for gi in range(len(plan.prunable)):
            vals = [level[gi] for level in run_stack]
            if extra is not None:
                vals.append(extra[gi])
            parts.append(min(vals) if vals else 1.0)
        return _combine(parts, tnorm)

    def on_extend(parent_pos: int, cand_pos: int) -> bool:
        if not prune or not plan.prunable:
            return True
        return _bound(_parent_degrees(parent_pos, cand_pos)) >= floor

    def on_push(parent_pos: int, cand_pos: int) -> None:
        run_stack.append(_parent_degrees(parent_pos, cand_pos))

    def on_pop() -> None:
        run_stack.pop()

    def on_accept(witness: tuple[int, ...]) -> bool:
        nonlocal best, best_score, floor, top_score
        parts, degs = _witness_parts(graph, witness, plan, libs, crisp, crisp_alpha)
        s = _combine(parts, tnorm)
        top_score = max(top_score, s)
        if s >= emit and (best is None or s > best_score):
            best_score = s
            best = (witness, s, tuple(degs))
            floor = s
        return False

    truncated, expansions = _walk(
        graph, anchor_pos, plan.min_hops, plan.max_hops, plan.window_s,
        plan.closed, DEFAULT_MAX_FANOUT, DEFAULT_MAX_EXPANSIONS,
        on_extend, on_accept, on_push, on_pop)
    if best is None:
        return ScoredResult(False, (), 0.0, (), truncated, expansions, top_score)
    witness, s, degs = best
    return ScoredResult(True, witness, s, degs, truncated, expansions, top_score)


def _witness_parts(graph: TemporalGraph, witness: tuple[int, ...], plan, libs,
                   crisp: bool, crisp_alpha: float) -> tuple[list[float], list[tuple]]:
    """Per-group degrees of one completed witness, plus explanation rows.

    Shared by the pure-Python accept hook and the fast path (which calls it
    once, for the winning witness), so explanations are byte-identical.
    """
    terms, _ = libs
    tnorm = plan.score_tnorm
    times = [int(graph.times[e]) for e in witness]
    amounts = [float(graph.amount[e]) for e in witness]
    parts: list[float] = []
    degs: list[tuple] = []
    for gi, group in enumerate(plan.prunable):
        hits: list[tuple[int, float]] = []
        for i in range(1, len(witness)):
            ds = [step_degrees(a, times[i] - times[i - 1], amounts[i],
                               amounts[i - 1], terms, crisp, crisp_alpha)
                  for a in group.atoms]
            ds = [d for d in ds if d is not None]
            if ds:
                hits.append((i, _combine(ds, tnorm)))
        if hits:
            deg = min(d for _, d in hits)
            weakest = min(hits, key=lambda t: t[1])[0] + 1
        else:
            deg, weakest = 1.0, None
        parts.append(deg)
        degs.append((group.label, deg, weakest))
    if plan.deferred is not None:
        ddef = eval_soft(plan.deferred, list(witness), graph, libs, tnorm,
                         crisp, crisp_alpha)
        parts.append(ddef)
        degs.append(("soft (deferred)", ddef, None))
    return parts, degs


def _combine(parts: list[float], tnorm: str) -> float:
    if tnorm == "product":
        out = 1.0
        for d in parts:
            out *= d
        return out
    return min(parts) if parts else 1.0


# -- numba fast path ---------------------------------------------------------
#
# A compute-only port of _walk/_score_py/_search_py (motifql._pathkernel).
# Nothing is cached across calls. A plan's atom parameters are compiled on
# every score() / score_path_batch() call from the term definitions as
# they are then (about 5 us per call on the amlworld path plans). The
# kernel reads the graph's own arrays at each call; the only graph-side
# precondition is the CSR ordering flag (graph.py), re-checked whenever
# the structure arrays are rebound. Anything the kernel does not model
# falls back to the pure-Python code.

_SHAPES = {"trapezoid": (0, 4), "triangle": (0, 3), "ramp": (2, 2), "ramp_down": (3, 2)}


@dataclass(frozen=True)
class _CompiledPlan:
    """Flat numeric form of a PathPlan (kernel input; see _pathkernel.ATOM_*)."""

    grp_ptr: npt.NDArray[np.int64]  # group g owns atom rows grp_ptr[g]:grp_ptr[g+1]
    atoms: npt.NDArray[np.float64]  # (n_atoms, ATOM_COLS)
    tnorm_prod: bool
    emit: float
    min_hops: int
    max_hops: int
    window_s: float
    closed: bool


def _compile_path_plan(plan, terms: dict) -> _CompiledPlan | None:
    """Compile a PathPlan's prunable atoms; None when the kernel can't run it.

    None covers deferred soft (scored per witness by the AST interpreter)
    and anything step_degrees would reject or compute differently (unknown
    atom/shape/tolerance, bad term arity): the pure walk then runs, raising
    exactly when and where it always did.
    """
    if plan.deferred is not None or float(plan.window_s) != float(plan.window_s):
        return None  # deferred soft, or a NaN window (searchsorted orders NaN last)
    grp_ptr = [0]
    rows: list[tuple[float, ...]] = []  # _pathkernel.ATOM_* column order
    try:
        for group in plan.prunable:
            for atom in group.atoms:
                if type(atom) is GapTermCond:
                    tdef = terms[atom.term]
                    shape = _SHAPES.get(tdef.shape)
                    vals = resolve_term_args(atom.term, atom.term_args, terms)
                    if shape is None or len(vals) != shape[1]:
                        return None
                    vals = tuple(float(v) for v in vals)
                    if tdef.shape == "triangle":
                        vals = (vals[0], vals[1], vals[1], vals[2])
                    vals = vals + (0.0,) * (4 - len(vals))
                    rows.append((0.0, float(shape[0]), *vals,
                                 float(bool(atom.negated)), 0.0, 0.0))
                elif type(atom) is AmountConsCond:
                    tol = atom.tol
                    if type(tol) is Pct:
                        pct, w = 1.0, tol.fraction
                    elif type(tol) is Dur:
                        pct, w = 0.0, tol.seconds
                    elif type(tol) is Money:
                        pct, w = 0.0, tol.amount
                    elif type(tol) is Num:
                        pct, w = 0.0, tol.value
                    else:
                        return None
                    rows.append((1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                 float(bool(atom.negated)), pct, float(w)))
                else:
                    return None
            grp_ptr.append(len(rows))
    except (KeyError, ValueError, TypeError, AttributeError):
        return None
    return _CompiledPlan(
        grp_ptr=np.asarray(grp_ptr, dtype=np.int64),
        atoms=np.asarray(rows, dtype=np.float64).reshape(len(rows), 9),
        tnorm_prod=plan.score_tnorm == "product",
        emit=float(plan.emit),
        min_hops=int(plan.min_hops),
        max_hops=int(plan.max_hops),
        window_s=float(plan.window_s),
        closed=bool(plan.closed),
    )


def _compiled_plan(plan, libs, crisp: bool, crisp_alpha: float) -> _CompiledPlan | None:
    """Compiled plan for the kernel, or None to use the pure-Python walk.

    Compiled afresh on each call from the plan and the term definitions as
    they are now; nothing is memoised, so an edited term library is always
    seen.
    """
    if crisp and not 0.0 <= crisp_alpha <= 1.0:
        return None  # crisp_degree raises (if a step is ever scored)
    return _compile_path_plan(plan, libs[0])


def _graph_ok(graph: TemporalGraph) -> bool:
    """True when the kernel's array assumptions hold for this graph.

    ``csr_ordered`` (time-sorted segments, so ``_capped``'s stable argsort
    is the identity) is checked when the graph is built and again after
    its structure arrays are rebound. The rest are attribute reads.
    """
    return (graph.csr_ordered and graph.in_times.dtype == np.int64
            and graph.times.dtype == np.int64 and graph.in_ptr.dtype == np.int64
            and graph._in_lo.dtype == np.int64 and graph.in_edges.dtype.kind in "iu"
            and graph.frm.dtype.kind in "iu" and graph.to.dtype.kind in "iu"
            and graph._in_lo.flags.writeable)


def _fast_ok(graph: TemporalGraph, anchor_pos, max_fanout: int) -> bool:
    """Kernel preconditions; anything else (odd anchors, negative caps) runs pure."""
    if isinstance(anchor_pos, (bool, np.bool_)) or not isinstance(anchor_pos, (int, np.integer)):
        return False  # numpy indexes with bools as masks; floats raise
    return 0 <= anchor_pos < graph.n_edges and max_fanout >= 0 and _graph_ok(graph)


def _kernel_args(graph: TemporalGraph) -> tuple:
    """Graph arrays for the kernel, read as they are at this call."""
    return (graph.times, graph.frm, graph.to, graph.amount, graph.in_ptr,
            graph.in_edges, graph.in_times)


_DUMMY_ENUM = np.empty((1, 1), dtype=np.int64)
_DUMMY_LEN = np.empty(1, dtype=np.int64)
_NO_GROUPS = np.zeros(1, dtype=np.int64)  # grp_ptr of a plan with no groups
_NO_ATOMS = np.empty((0, 9), dtype=np.float64)


def _score_fast(graph: TemporalGraph, anchor_pos: int, plan, libs, cp: _CompiledPlan,
                prune: bool, crisp: bool, crisp_alpha: float,
                floor: float | None) -> ScoredResult:
    from motifql import _pathkernel as K

    best = np.empty(max(cp.max_hops, 2), dtype=np.int64)
    floor0 = cp.emit if floor is None else float(floor)
    found, blen, bscore, top, truncated, expansions, _ = K.walk(
        *_kernel_args(graph), graph._in_lo, True, float(graph.watermark),
        int(anchor_pos), cp.min_hops, cp.max_hops, cp.window_s, cp.closed,
        DEFAULT_MAX_FANOUT, DEFAULT_MAX_EXPANSIONS, K.MODE_SCORE, bool(prune),
        cp.grp_ptr, cp.atoms, cp.tnorm_prod, cp.emit, floor0, bool(crisp),
        float(crisp_alpha), 1, _DUMMY_ENUM, _DUMMY_LEN, best)
    if not found:
        return ScoredResult(False, (), 0.0, (), bool(truncated), int(expansions),
                            float(top))
    witness = tuple(int(e) for e in best[:blen - 1]) + (anchor_pos,)
    _, degs = _witness_parts(graph, witness, plan, libs, crisp, crisp_alpha)
    return ScoredResult(True, witness, float(bscore), tuple(degs), bool(truncated),
                        int(expansions), float(top))


def _search_fast(graph: TemporalGraph, anchor_pos: int, motif: PathMotif,
                 collect_all: bool, cap, enum_cap: int
                 ) -> tuple[list[tuple[int, ...]], bool, int]:
    from motifql import _pathkernel as K

    rows = enum_cap  # >= 1, and the kernel never stores more than enum_cap
    width = max(int(motif.max_hops), 2)
    enum_out = np.empty((rows, width), dtype=np.int64)
    enum_len = np.empty(rows, dtype=np.int64)
    _, _, _, _, truncated, expansions, n_enum = K.walk(
        *_kernel_args(graph), graph._in_lo, True, float(graph.watermark),
        int(anchor_pos), int(motif.min_hops), int(motif.max_hops),
        float(motif.window_s), bool(motif.closed), int(motif.max_fanout),
        int(motif.max_expansions), K.MODE_ENUM, False, _NO_GROUPS, _NO_ATOMS,
        False, 0.0, 0.0, False, 0.0, enum_cap, enum_out, enum_len,
        np.empty(width, dtype=np.int64))
    witnesses = [tuple(int(e) for e in enum_out[r, :enum_len[r] - 1]) + (anchor_pos,)
                 for r in range(n_enum)]
    capped = collect_all and len(witnesses) >= cap
    return witnesses, bool(truncated) or capped, int(expansions)


class PathBatch(NamedTuple):
    """Per-anchor outputs of :func:`score_path_batch` (row r = positions[r]).

    ``witness`` rows hold the best witness's edge positions oldest first,
    anchor last, padded with -1 (``witness_len`` gives the length; 0 when
    not found). ``score`` is 0.0 when not found, as in ScoredResult.
    """

    top_score: npt.NDArray[np.float64]
    found: npt.NDArray[np.bool_]
    expansions: npt.NDArray[np.int64]
    truncated: npt.NDArray[np.bool_]
    score: npt.NDArray[np.float64]
    witness: npt.NDArray[np.int64]
    witness_len: npt.NDArray[np.int64]


def score_path_batch(graph: TemporalGraph, positions, plan, libs, prune: bool = True,
                     crisp: bool = False, crisp_alpha: float = 0.0,
                     floor: float | None = None, parallel: bool = False) -> PathBatch:
    """Score many anchors against one PathPlan with no per-anchor Python.

    Row r equals ``score(graph, positions[r], plan, libs, ...)`` field for
    field (degrees excepted: call :func:`score` for the few anchors you
    explain). Anchors run in the given order, so the lazy eviction floors
    in ``graph._in_lo`` end up exactly as after the serial loop. Plans or
    graphs the kernel does not model (deferred soft, ...) run the
    pure-Python walk per anchor.

    ``parallel=True`` uses a numba ``prange`` over anchors. Threads must
    not race on the lazy floors, so the kernel then reads a read-only
    copy (``max(_in_lo, in_ptr)``) and writes nothing. That is identical
    to serial only while the watermark sits at or below the earliest edge
    time (the default, ``watermark = times[0]``): every rebase floor is
    then the segment start, and the serial writes are unobservable
    (``in_degree``, ``degree_histogram`` and ``_rebase`` all take the max
    with the segment start). Once ``evict_before`` moves the watermark
    past that, ``parallel`` is ignored and the serial kernel runs.
    """
    from motifql import _pathkernel as K

    pos = np.ascontiguousarray(np.asarray(positions, dtype=np.int64))
    n = len(pos)
    width = max(int(plan.max_hops), 2)
    out = PathBatch(
        top_score=np.zeros(n, dtype=np.float64), found=np.zeros(n, dtype=np.bool_),
        expansions=np.zeros(n, dtype=np.int64), truncated=np.zeros(n, dtype=np.bool_),
        score=np.zeros(n, dtype=np.float64),
        witness=np.full((n, width), -1, dtype=np.int64),
        witness_len=np.zeros(n, dtype=np.int64))
    cp = None if PURE_PYTHON else _compiled_plan(plan, libs, crisp, crisp_alpha)
    in_range = bool(((pos >= 0) & (pos < graph.n_edges)).all()) if n else True
    if cp is None or not in_range or not _graph_ok(graph):
        for r, a in enumerate(pos.tolist()):
            res = score(graph, a, plan, libs, prune, crisp, crisp_alpha, floor)
            out.top_score[r] = res.top_score
            out.found[r] = res.found
            out.expansions[r] = res.expansions
            out.truncated[r] = res.truncated
            out.score[r] = res.score
            out.witness_len[r] = len(res.witness)
            out.witness[r, :len(res.witness)] = res.witness
        return out
    floor0 = cp.emit if floor is None else float(floor)
    common = (pos, cp.min_hops, cp.max_hops, cp.window_s, cp.closed,
              DEFAULT_MAX_FANOUT, DEFAULT_MAX_EXPANSIONS, bool(prune),
              cp.grp_ptr, cp.atoms, cp.tnorm_prod, cp.emit, floor0, bool(crisp),
              float(crisp_alpha), out.found, out.score, out.top_score,
              out.truncated, out.expansions, out.witness, out.witness_len)
    watermark = float(graph.watermark)
    if parallel and n > 1 and watermark <= float(graph.times.min()):
        lo_eff = np.maximum(graph._in_lo, graph.in_ptr[:-1])
        K.score_batch_parallel(*_kernel_args(graph), lo_eff, watermark, *common)
    else:
        K.score_batch(*_kernel_args(graph), graph._in_lo, watermark, *common)
    return out
