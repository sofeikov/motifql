"""Tests for the temporal graph store (`src/motifql/graph.py`).

Strategy mirrors `tests/test_baseline_features.py`: a tiny hand-built frame
pins exact slice semantics, a randomised frame checks every windowed query
against a brute-force Polars filter (the oracle), and a real-data smoke test
-- skipped when `data/hi-small` is absent -- proves the hub scale loads.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from motifql.graph import TemporalGraph, SECS_PER_DAY  # noqa: E402

BASE = datetime(2022, 9, 3)


def _frame(rows: list[tuple[int, str, str, str, str, float]]) -> pl.DataFrame:
    """Rows of (minute_offset, from_bank, from_acct, to_bank, to_acct, usd)."""
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


def _chain_frame() -> pl.DataFrame:
    # A -10m-> B -20m-> C -30m-> A (cycle, strictly increasing times).
    return _frame(
        [
            (0, "001", "A", "002", "B", 100.0),
            (10, "002", "B", "003", "C", 95.0),
            (20, "003", "C", "001", "A", 90.0),
            (21, "003", "C", "001", "A", 91.0),  # same-pair repeat, later
            (30, "001", "A", "004", "D", 50.0),  # unrelated anchor candidate
        ]
    )


def _epoch(dt: datetime) -> int:
    # Explicit UTC arithmetic: naive timestamps are UTC by prep convention,
    # and `datetime.timestamp()` would silently use the machine zone.
    return int((dt - datetime(1970, 1, 1)).total_seconds())


def test_node_key_roundtrip() -> None:
    g = TemporalGraph.from_transfers(_chain_frame())
    assert g.node_id("001", "A") >= 0
    assert g.node_key(g.node_id("002", "B")) == "002/B"
    assert g.node_id("999", "ZZZ") == -1  # unknown account, never seen


def test_incoming_slice_exact() -> None:
    g = TemporalGraph.from_transfers(_chain_frame())
    a = g.node_id("001", "A")
    edges, times = g.incoming(a, before_t=_epoch(BASE + timedelta(minutes=25)),
                              window_s=7 * SECS_PER_DAY)
    # Both t=20 and t=21 land strictly before the anchor.
    assert [int(g.transfer_id[e]) for e in edges] == [2, 3]
    # An anchor exactly at t=21 excludes the same-minute edge (strict order).
    edges, _ = g.incoming(a, before_t=_epoch(BASE + timedelta(minutes=21)),
                           window_s=7 * SECS_PER_DAY)
    assert [int(g.transfer_id[e]) for e in edges] == [2]


def test_same_minute_excluded_strict_time_order() -> None:
    # Two hops in the same minute cannot be ordered: both land strictly
    # before a later anchor but neither may serve as the other's predecessor.
    g = TemporalGraph.from_transfers(
        _frame(
            [
                (0, "001", "A", "002", "B", 100.0),
                (0, "002", "B", "003", "C", 100.0),
            ]
        )
    )
    b = g.node_id("002", "B")
    edges, _ = g.incoming(b, before_t=_epoch(BASE), window_s=3600)
    assert len(edges) == 0  # same-minute edge is not strictly before


def test_window_bounds() -> None:
    g = TemporalGraph.from_transfers(_chain_frame())
    a = g.node_id("001", "A")
    # Window of 5 minutes ending at t=25 covers [t=20, t=25): both edges.
    edges, _ = g.incoming(a, before_t=_epoch(BASE + timedelta(minutes=25)),
                          window_s=5 * 60)
    assert [int(g.transfer_id[e]) for e in edges] == [2, 3]
    # A 4-minute window covers [t=21, t=25): only the later edge.
    edges, _ = g.incoming(a, before_t=_epoch(BASE + timedelta(minutes=25)),
                          window_s=4 * 60)
    assert [int(g.transfer_id[e]) for e in edges] == [3]
    # Outgoing from A at t=30 sees the t=0... no: window [25,30) is empty.
    a_out, _ = g.outgoing(a, before_t=_epoch(BASE + timedelta(minutes=30)),
                          window_s=5 * 60)
    assert len(a_out) == 0


def test_random_frame_matches_polars_oracle() -> None:
    rng = np.random.default_rng(7)
    banks = [f"{i:05d}" for i in range(3)]
    accts = [f"X{i}" for i in range(6)]
    rows = []
    for i in range(500):
        rows.append((
            int(rng.integers(0, 3 * 24 * 60)),
            str(rng.choice(banks)), str(rng.choice(accts)),
            str(rng.choice(banks)), str(rng.choice(accts)),
            float(rng.uniform(1, 1000)),
        ))
    df = _frame(rows)
    g = TemporalGraph.from_transfers(df)
    ts = df.get_column("timestamp").to_list()
    fkeys = (df.get_column("from_bank") + "/" + df.get_column("from_account")).to_list()
    tkeys = (df.get_column("to_bank") + "/" + df.get_column("to_account")).to_list()

    for trial in range(200):
        node = int(rng.integers(0, g.n_nodes))
        key = g.node_key(node)
        before = BASE + timedelta(minutes=int(rng.integers(0, 3 * 24 * 60)))
        window = int(rng.integers(60, 86400))
        lo_t, hi_t = before - timedelta(seconds=window), before
        if rng.random() < 0.5:
            edges, _ = g.incoming(node, _epoch(hi_t), window)
            want = {i for i, (t, k) in enumerate(zip(ts, tkeys))
                    if k == key and lo_t <= t < hi_t}
        else:
            edges, _ = g.outgoing(node, _epoch(hi_t), window)
            want = {i for i, (t, k) in enumerate(zip(ts, fkeys))
                    if k == key and lo_t <= t < hi_t}
        assert {int(g.transfer_id[e]) for e in edges} == want, f"trial {trial}"


def test_evict_parity_with_windowed_replay() -> None:
    rng = np.random.default_rng(11)
    rows = [
        (m, "001", f"S{m % 5}", "002", f"R{m % 7}", 10.0)
        for m in range(0, 4 * 24 * 60, 37)
    ]
    df = _frame(rows)
    full = TemporalGraph.from_transfers(df)
    live = TemporalGraph.from_transfers(df)
    assert len(rows) > 100

    # Replay in time order with eviction; compare every query to no-evict.
    order = np.argsort(full.times, kind="stable")
    for pos in order[::7]:
        t = int(full.times[pos])
        live.evict_before(t - 2 * SECS_PER_DAY)
        node = int(full.to[pos] if pos % 2 else full.frm[pos])
        got_e, got_t = live.incoming(node, t, 2 * SECS_PER_DAY)
        want_e, want_t = full.incoming(node, t, 2 * SECS_PER_DAY)
        assert [int(full.transfer_id[e]) for e in got_e] == \
               [int(full.transfer_id[e]) for e in want_e]
        assert list(got_t) == list(want_t)


HI_SMALL = Path("data/hi-small/transfers.parquet")


@pytest.mark.skipif(not HI_SMALL.exists(), reason="HI-Small parquet not present")
def test_hismall_loads_and_hub_present() -> None:
    g = TemporalGraph.from_parquet(HI_SMALL)
    stats = g.stats()
    assert stats["n_edges"] == 5_078_345
    # The bank-070 hub from the Oct 27 report sits on the sender side
    # (449,859 rows from bank 070, only 2,892 into it), and one account
    # dominates it: 070/100428660 alone sends 168,672 transfers, the exact
    # "168K hub" figure from the Oct 27 report. The fanout cap exists for it.
    import polars as pl

    raw = pl.read_parquet(HI_SMALL, columns=["from_bank"])
    assert (raw.get_column("from_bank") == "070").sum() > 400_000
    assert stats["max_out_degree"] > 10_000
    # Per-node segments stay time-ordered at full scale (sampled check).
    rng = np.random.default_rng(3)
    for node in rng.integers(0, g.n_nodes, size=2000):
        seg = g.in_times[g.in_ptr[node]:g.in_ptr[node + 1]]
        assert np.all(np.diff(seg.astype(np.int64)) >= 0)
        seg = g.out_times[g.out_ptr[node]:g.out_ptr[node + 1]]
        assert np.all(np.diff(seg.astype(np.int64)) >= 0)
