"""Tests for the exact anchored matchers (`src/motifql/matcher.py`).

Synthetic frames pin the crisp semantics (close/min-hops/window/strict
time/a!=z/caps); one real-data smoke test anchors on the last member of
train cycle instance 102 and requires a witness ending at that anchor.
"""

from __future__ import annotations

import dataclasses
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from motifql.graph import TemporalGraph  # noqa: E402
from motifql.matcher import LAYERING_CHAIN, ROUND_TRIP, match  # noqa: E402

BASE = datetime(2022, 9, 3)


def _frame(rows: list[tuple[int, str, str, str, str]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "transfer_id": list(range(len(rows))),
            "timestamp": [BASE + timedelta(minutes=r[0]) for r in rows],
            "from_bank": [r[1] for r in rows],
            "from_account": [r[2] for r in rows],
            "to_bank": [r[3] for r in rows],
            "to_account": [r[4] for r in rows],
            "amount_usd": [100.0] * len(rows),
        }
    )


def _cycle3() -> pl.DataFrame:
    # A -10m-> B -10m-> C -10m-> A plus an unrelated edge.
    return _frame(
        [
            (0, "001", "A", "002", "B"),
            (10, "002", "B", "003", "C"),
            (20, "003", "C", "001", "A"),
            (30, "009", "Z", "008", "Y"),
        ]
    )


def test_cycle_found_witness_oldest_first() -> None:
    g = TemporalGraph.from_transfers(_cycle3())
    res = match(g, anchor_pos=2, motif=ROUND_TRIP)
    assert res.found and not res.truncated
    assert [int(g.transfer_id[e]) for e in res.edges] == [0, 1, 2]


def test_cycle_min_hops_rejects_early_close() -> None:
    g = TemporalGraph.from_transfers(
        _frame([(0, "001", "A", "002", "B"), (10, "002", "B", "001", "A")])
    )
    assert match(g, 1, ROUND_TRIP).found  # 2-hop close is legal
    strict = dataclasses.replace(ROUND_TRIP, min_hops=3)
    assert not match(g, 1, strict).found


def test_cycle_window_rejects_long_span() -> None:
    g = TemporalGraph.from_transfers(_cycle3())  # span is 20 minutes
    narrow = dataclasses.replace(ROUND_TRIP, window_s=5 * 60)
    assert not match(g, 2, narrow).found


def test_cycle_same_minute_hop_rejected() -> None:
    g = TemporalGraph.from_transfers(
        _frame([(0, "001", "A", "002", "B"), (0, "002", "B", "001", "A")])
    )
    assert not match(g, 1, ROUND_TRIP).found


def test_cycle_must_close() -> None:
    # Open chain anchored at its tip is not a cycle.
    g = TemporalGraph.from_transfers(
        _frame(
            [
                (0, "001", "A", "002", "B"),
                (10, "002", "B", "003", "C"),
                (20, "003", "C", "004", "D"),
            ]
        )
    )
    assert not match(g, 2, ROUND_TRIP).found
    assert match(g, 2, LAYERING_CHAIN).found


def test_chain_rejects_start_equal_to_receiver() -> None:
    # B -> C -> A anchored at C -> A: the only 3-hop walk starts at the
    # anchor receiver, violating a != z.
    g = TemporalGraph.from_transfers(
        _frame(
            [
                (0, "001", "A", "002", "B"),
                (10, "002", "B", "003", "C"),
                (20, "003", "C", "001", "A"),
            ]
        )
    )
    assert not match(g, 2, LAYERING_CHAIN).found


def test_chain_cannot_pass_through_receiver() -> None:
    # N0 -> N1 -> N2 -> N1 anchored at N2 -> N1: the only 3-hop walk passes
    # through the anchor receiver mid-path, repeating it once the anchor
    # closes the walk. Not a simple path, so no witness.
    g = TemporalGraph.from_transfers(
        _frame(
            [
                (0, "001", "N0", "002", "N1"),
                (10, "002", "N1", "003", "N2"),
                (20, "003", "N2", "002", "N1"),
            ]
        )
    )
    assert not match(g, 2, LAYERING_CHAIN).found


def test_self_loop_anchor_never_fires() -> None:
    # C1: a self-loop anchor cannot end a simple-path witness, open or
    # closed — the edge before the anchor already ends where it starts.
    # The two self-loops alone would previously "close" a round trip.
    g = TemporalGraph.from_transfers(
        _frame(
            [
                (0, "001", "S", "001", "S"),
                (10, "001", "S", "001", "S"),
            ]
        )
    )
    assert not match(g, 1, ROUND_TRIP).found
    assert not match(g, 1, LAYERING_CHAIN).found


def test_self_loop_anchor_misses_beside_a_real_cycle() -> None:
    # Guard against over-blocking: the cycle still fires at its own
    # anchor while the self-loop anchor beside it stays silent.
    g = TemporalGraph.from_transfers(
        _frame(
            [
                (0, "001", "A", "002", "B"),
                (10, "002", "B", "003", "C"),
                (20, "003", "C", "001", "A"),
                (30, "001", "S", "001", "S"),
            ]
        )
    )
    assert match(g, 2, ROUND_TRIP).found
    assert not match(g, 3, ROUND_TRIP).found
    assert not match(g, 3, LAYERING_CHAIN).found


def _brute_force(graph: TemporalGraph, anchor_pos: int, motif) -> bool:
    """Exhaustive simple-path enumeration without caps: oracle for match.

    Same anchored backward search, but no fanout/expansion caps and no
    early exit, over the anchor-relative window throughout.
    """
    anchor_t = int(graph.times[anchor_pos])
    sender = int(graph.frm[anchor_pos])
    receiver = int(graph.to[anchor_pos])
    if sender == receiver:
        return False  # mirrors _walk: no simple-path witness ends here
    window_start = anchor_t - motif.window_s

    def dfs(node: int, t_bound: int, depth: int, visited: set[int]) -> bool:
        edges, _ = graph.incoming(node, t_bound, t_bound - window_start)
        for edge in edges.tolist():
            edge = int(edge)
            prev = int(graph.frm[edge])
            hops = depth + 2  # this edge plus the anchor
            if hops > motif.max_hops:
                continue
            if motif.closed and prev == receiver:
                if hops >= motif.min_hops:
                    return True
                continue
            if prev in visited:
                continue
            if not motif.closed and hops >= motif.min_hops and prev != receiver:
                return True
            if hops >= motif.max_hops:
                continue
            if dfs(prev, int(graph.times[edge]), depth + 1, visited | {prev}):
                return True
        return False

    return dfs(sender, anchor_t, 0, {sender, receiver})


def test_match_agrees_with_brute_force_oracle() -> None:
    rng = np.random.default_rng(42)
    for trial in range(30):
        accts = [f"N{i}" for i in range(6)]
        rows = [
            (
                int(rng.integers(0, 200)),
                "001", str(rng.choice(accts)),
                "001", str(rng.choice(accts)),
            )
            for _ in range(14)
        ]
        g = TemporalGraph.from_transfers(_frame(rows))
        motifs = [ROUND_TRIP, LAYERING_CHAIN]
        if trial % 3 == 0:
            # Shrink the windows so the span bound binds on some anchors.
            motifs = [dataclasses.replace(m, window_s=60 * 60) for m in motifs]
        for anchor in range(g.n_edges):
            for motif in motifs:
                res = match(g, anchor, motif)
                if not res.truncated:
                    assert res.found == _brute_force(g, anchor, motif), (
                        trial, anchor, motif.name,
                    )


def test_chain_min_hops() -> None:
    g = TemporalGraph.from_transfers(
        _frame([(0, "001", "A", "002", "B"), (10, "002", "B", "003", "C")])
    )
    assert not match(g, 1, LAYERING_CHAIN).found  # only 2 hops, needs 3


def test_fanout_cap_flags_truncation() -> None:
    # Cycle Y -5m-> Q -10m-> P -20m-> Y with five noisier, more recent edges
    # into P burying the true back-edge Q -> P.
    rows = [(5, "001", "Y", "002", "Q"), (10, "002", "Q", "003", "P")]
    rows += [(15, "001", f"S{i}", "003", "P") for i in range(5)]
    rows += [(20, "003", "P", "001", "Y")]
    g = TemporalGraph.from_transfers(_frame(rows))
    tiny = dataclasses.replace(ROUND_TRIP, max_fanout=2)
    res = match(g, 7, tiny)
    assert (res.found, res.truncated) == (False, True)
    wide = dataclasses.replace(ROUND_TRIP, max_fanout=100)
    res = match(g, 7, wide)
    assert res.found and not res.truncated
    assert [int(g.transfer_id[e]) for e in res.edges] == [0, 1, 7]


def test_expansion_budget_exhausts() -> None:
    g = TemporalGraph.from_transfers(_cycle3())
    res = match(g, 2, dataclasses.replace(ROUND_TRIP, max_expansions=0))
    assert (res.found, res.truncated, res.expansions) == (False, True, 0)


def test_match_is_deterministic() -> None:
    g = TemporalGraph.from_transfers(_cycle3())
    assert match(g, 2, ROUND_TRIP).edges == match(g, 2, ROUND_TRIP).edges


HI_SMALL = Path("data/hi-small/transfers.parquet")


@pytest.mark.skipif(not HI_SMALL.exists(), reason="HI-Small parquet not present")
def test_train_cycle_instance_fires_on_last_member() -> None:
    g = TemporalGraph.from_parquet(HI_SMALL)
    tr = pl.read_parquet(HI_SMALL, columns=["transfer_id", "timestamp"])
    memb = pl.read_parquet("data/hi-small/pattern_members.parquet")
    tids = (
        memb.filter(pl.col("instance_id") == 102)
        .join(tr, on="transfer_id")
        .sort("timestamp")
        .get_column("transfer_id")
        .to_list()
    )
    anchor = int(np.where(g.transfer_id == tids[-1])[0][0])
    res = match(g, anchor, ROUND_TRIP)
    assert res.found and not res.truncated
    assert int(g.transfer_id[res.edges[-1]]) == tids[-1]  # anchor completes it
