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

Usage:
    from motifql.graph import TemporalGraph
    from motifql.matcher import ROUND_TRIP, match
    graph = TemporalGraph.from_parquet("data/hi-small/transfers.parquet")
    res = match(graph, anchor_pos=123, motif=ROUND_TRIP)
    if res.found: print(res.edges, res.truncated)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from motifql.degrees import eval_soft, step_degrees
from motifql.graph import SECS_PER_DAY, TemporalGraph

log = logging.getLogger("matcher")

# Search budget defaults: generous enough that ordinary accounts never hit
# them, tight enough that the bank-070 senders terminate.
DEFAULT_MAX_FANOUT = 1_000
DEFAULT_MAX_EXPANSIONS = 50_000


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


def _combine(parts: list[float], tnorm: str) -> float:
    if tnorm == "product":
        out = 1.0
        for d in parts:
            out *= d
        return out
    return min(parts) if parts else 1.0
