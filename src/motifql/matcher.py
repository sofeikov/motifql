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
    edges: np.ndarray = field(repr=False)
    next_idx: int = 0


def _capped(edges: np.ndarray, times: np.ndarray, cap: int
            ) -> tuple[np.ndarray, bool]:
    """Most-recent-first candidates, capped; flag when the cap bites."""
    order = np.argsort(times, kind="stable")[::-1]
    ranked = edges[order]
    if len(ranked) > cap:
        return ranked[:cap], True
    return ranked, False


def match(graph: TemporalGraph, anchor_pos: int, motif: PathMotif) -> MatchResult:
    """Score one anchor transfer (by edge position) against one path motif."""
    anchor_t = int(graph.times[anchor_pos])
    sender = int(graph.frm[anchor_pos])
    receiver = int(graph.to[anchor_pos])
    window_start = anchor_t - motif.window_s
    expansions = 0
    truncated = False

    first_edges, first_times = graph.incoming(sender, anchor_t, motif.window_s)
    if len(first_edges) == 0:
        return MatchResult(False, (), False, 0)
    first, hit = _capped(first_edges, first_times, motif.max_fanout)
    truncated = hit

    # path holds the back-edges taken below the anchor (path[-1] is deepest);
    # visited holds every node on the current walk, plus the anchor receiver
    # for closed motifs so the walk cannot pass through it before closing.
    stack: list[_Frame] = [_Frame(sender, first)]
    path: list[int] = []
    visited: set[int] = {sender, receiver} if motif.closed else {sender}

    while stack:
        frame = stack[-1]
        if frame.next_idx >= len(frame.edges):
            stack.pop()
            if stack:
                # The popped level was reached via path[-1]: unvisit it.
                visited.discard(int(graph.frm[path.pop()]))
            continue
        if expansions >= motif.max_expansions:
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
            stack.pop()
            if stack:
                visited.discard(int(graph.frm[path.pop()]))
            continue
        expansions += 1
        prev = int(graph.frm[edge])
        hops = len(path) + 2  # candidate edge plus the anchor
        if hops > motif.max_hops:
            continue
        if anchor_t - edge_t > motif.window_s:
            continue  # paths only age as they extend: safe to prune
        if motif.closed and prev == receiver:
            if hops >= motif.min_hops:
                return MatchResult(True, tuple(reversed(path + [edge]))
                                        + (anchor_pos,), truncated, expansions)
            continue  # closes too early: below min_hops, keep looking
        if prev in visited:
            continue  # simple path (the closing step above is the exception)
        if not motif.closed and hops >= motif.min_hops and prev != receiver:
            return MatchResult(True, tuple(reversed(path + [edge]))
                                    + (anchor_pos,), truncated, expansions)
        if hops >= motif.max_hops:
            continue
        nxt_edges, nxt_times = graph.incoming(prev, edge_t, motif.window_s)
        if len(nxt_edges) == 0:
            continue
        nxt, hit = _capped(nxt_edges, nxt_times, motif.max_fanout)
        truncated = truncated or hit
        visited.add(prev)
        path.append(edge)
        stack.append(_Frame(prev, nxt))

    return MatchResult(False, (), truncated, expansions)
