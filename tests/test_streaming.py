"""Streaming ingest tests (`TemporalGraph.append`, overflow, rebuild).

Strategy: a hand-built frame pins exact merge order, then randomised
prefix+append graphs must answer every windowed query and every motif
score identically to the batch-built graph over the same rows.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from motifql.degrees import libs_from_doc  # noqa: E402
from motifql.graph import TemporalGraph, SECS_PER_DAY  # noqa: E402
from motifql.groups import score_any  # noqa: E402
from motifql.parser import parse_file  # noqa: E402
from motifql.planner import compile_motif  # noqa: E402

BASE = datetime(2022, 9, 3)
MOTIFS = Path(__file__).resolve().parent.parent / "examples" / "amlworld.motif"


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


def _epoch(dt: datetime) -> int:
    return int((dt - datetime(1970, 1, 1)).total_seconds())


def _append_row(g: TemporalGraph, row: dict) -> int:
    return g.append(
        _epoch(row["timestamp"]),
        row["from_bank"], row["from_account"],
        row["to_bank"], row["to_account"],
        row["amount_usd"], row["transfer_id"],
    )


def _assert_same_answers(full: TemporalGraph, live: TemporalGraph,
                         windows: list[int]) -> None:
    assert live.n_edges == full.n_edges
    assert live.nodes == full.nodes
    for attr in ("times", "frm", "to", "amount", "transfer_id"):
        assert np.array_equal(getattr(live, attr), getattr(full, attr)), attr
    t_end = int(full.times[-1]) + 60
    for node in range(full.n_nodes):
        for before_t in (int(full.times[0]), int(full.times[-1]), t_end):
            for w in windows:
                le, lt = live.incoming(node, before_t, w)
                fe, ft = full.incoming(node, before_t, w)
                assert np.array_equal(le, fe), (node, before_t, w, "in")
                assert np.array_equal(lt, ft), (node, before_t, w, "in-t")
                le, lt = live.outgoing(node, before_t, w)
                fe, ft = full.outgoing(node, before_t, w)
                assert np.array_equal(le, fe), (node, before_t, w, "out")
                assert np.array_equal(lt, ft), (node, before_t, w, "out-t")
        assert live.in_degree(node) == full.in_degree(node)
        assert live.out_degree(node) == full.out_degree(node)
    assert np.array_equal(live.degree_histogram(), full.degree_histogram())


def _random_rows(rng: np.random.Generator, n: int,
                 accts: list[str]) -> list[tuple[int, str, str, str, str, float]]:
    rows = []
    t = 0
    for _ in range(n):
        t += int(rng.integers(1, 120))
        a, b = rng.choice(accts, size=2, replace=False)
        rows.append((t, "001", a, "001", b, float(rng.integers(10, 5000))))
    return rows


def test_append_merges_oldest_first() -> None:
    g = TemporalGraph.from_transfers(_frame([
        (0, "001", "A", "001", "B", 100.0),
        (10, "001", "C", "001", "B", 200.0),
    ]))
    t1 = _epoch(BASE + timedelta(minutes=20))
    t2 = _epoch(BASE + timedelta(minutes=30))
    p1 = g.append(t1, "001", "D", "001", "B", 300.0, 2)
    p2 = g.append(t2, "001", "A", "001", "B", 400.0, 3)
    assert (p1, p2) == (2, 3)
    assert g.n_edges == 4
    b = g.node_id("001", "B")
    edges, times = g.incoming(b, t2 + 60, 7 * SECS_PER_DAY)
    assert edges.tolist() == [0, 1, 2, 3]
    assert times.tolist() == sorted(times.tolist())
    assert int(g.amount[3]) == 400
    assert g.node_key(g.node_id("001", "D")) == "001/D"
    # The appended edge is itself a readable anchor.
    assert int(g.times[2]) == t1
    assert g.node_key(int(g.frm[2])) == "001/D"


def test_append_parity_with_batch_build() -> None:
    rng = np.random.default_rng(7)
    accts = [f"A{i:02d}" for i in range(12)]
    df = _frame(_random_rows(rng, 300, accts))
    full = TemporalGraph.from_transfers(df)
    live = TemporalGraph.from_transfers(df.head(150))
    for row in df.slice(150).iter_rows(named=True):
        _append_row(live, row)
    _assert_same_answers(full, live, [3600, 3 * SECS_PER_DAY, 30 * SECS_PER_DAY])


def test_append_new_accounts() -> None:
    g = TemporalGraph.from_transfers(_frame([
        (0, "001", "A", "001", "B", 100.0),
    ]))
    t = _epoch(BASE + timedelta(minutes=5))
    g.append(t, "009", "NEW", "001", "B", 50.0, 1)
    n = g.node_id("009", "NEW")
    assert n >= 0
    edges, _ = g.outgoing(n, t + 60, SECS_PER_DAY)
    assert edges.tolist() == [1]
    assert g.in_degree(n) == 0
    assert g.out_degree(n) == 1


def test_rebuild_preserves_queries_and_node_ids() -> None:
    rng = np.random.default_rng(11)
    accts = [f"A{i:02d}" for i in range(10)]
    df = _frame(_random_rows(rng, 120, accts))
    full = TemporalGraph.from_transfers(df)
    live = TemporalGraph.from_transfers(df.head(40))
    live.rebuild_threshold = 25
    before = {k: live.node_id("001", k) for k in accts}
    for row in df.slice(40).iter_rows(named=True):
        _append_row(live, row)
    # 80 appends at threshold 25: auto-rebuilds at 25/50/75, then 5
    # fresh overflow edges. Parity must hold both before and after the
    # explicit rebuild that drains them.
    assert live._appended == 5
    assert live._csr_nodes > 0 and len(live._ov_in) > 0
    _assert_same_answers(full, live, [3600, 7 * SECS_PER_DAY])
    live.rebuild()
    assert live._appended == 0
    assert live._ov_in == {} and live._ov_out == {}
    after = {k: live.node_id("001", k) for k in accts}
    assert before == after  # node ids stable across rebuilds
    _assert_same_answers(full, live, [3600, 7 * SECS_PER_DAY])


def test_empty_then_append() -> None:
    g = TemporalGraph.empty()
    assert g.n_edges == 0
    assert g.stats()["n_nodes"] == 0
    rows = [
        (0, "001", "A", "001", "B", 100.0),
        (5, "001", "B", "001", "A", 99.0),
    ]
    for i, (m, fb, fa, tb, ta, usd) in enumerate(rows):
        g.append(_epoch(BASE + timedelta(minutes=m)), fb, fa, tb, ta, usd, i)
    want = TemporalGraph.from_transfers(_frame(rows))
    _assert_same_answers(want, g, [3600, 7 * SECS_PER_DAY])


def test_eviction_filters_overflow() -> None:
    g = TemporalGraph.from_transfers(_frame([
        (0, "001", "A", "001", "B", 100.0),
    ]))
    t1 = _epoch(BASE + timedelta(minutes=60))
    t2 = _epoch(BASE + timedelta(minutes=120))
    g.append(t1, "001", "A", "001", "B", 200.0, 1)
    g.append(t2, "001", "A", "001", "B", 300.0, 2)
    g.evict_before(t2)
    b = g.node_id("001", "B")
    edges, _ = g.incoming(b, t2 + 60, 7 * SECS_PER_DAY)
    assert edges.tolist() == [2]  # positions 0,1 expired from both stores


def test_out_of_order_append_still_merges() -> None:
    g = TemporalGraph.from_transfers(_frame([
        (0, "001", "A", "001", "B", 100.0),
    ]))
    t_hi = _epoch(BASE + timedelta(minutes=100))
    t_mid = _epoch(BASE + timedelta(minutes=50))
    g.append(t_hi, "001", "C", "001", "B", 200.0, 1)
    g.append(t_mid, "001", "D", "001", "B", 300.0, 2)
    b = g.node_id("001", "B")
    edges, times = g.incoming(b, t_hi + 60, 7 * SECS_PER_DAY)
    assert edges.tolist() == [0, 2, 1]
    assert times.tolist() == sorted(times.tolist())


def test_score_parity_live_vs_batch() -> None:
    # The anchor path crosses the CSR/overflow boundary: the planted cycle
    # sits in the appended suffix while its witnesses reach back into CSR.
    rng = np.random.default_rng(23)
    accts = [f"A{i:02d}" for i in range(15)]
    rows = _random_rows(rng, 200, accts)
    last_t = rows[-1][0]
    rows += [
        (last_t + 10, "001", "X1", "001", "X2", 1000.0),
        (last_t + 20, "001", "X2", "001", "X3", 990.0),
        (last_t + 30, "001", "X3", "001", "X1", 980.0),
    ]
    df = _frame(rows)
    full = TemporalGraph.from_transfers(df)
    live = TemporalGraph.from_transfers(df.head(200))
    for row in df.slice(200).iter_rows(named=True):
        _append_row(live, row)
    doc = parse_file(MOTIFS)
    libs = libs_from_doc(doc)
    for name in ("round_trip", "fan_out"):
        plan = compile_motif(doc, name)
        fired = 0
        for anchor in range(full.n_edges):
            r_full = score_any(full, anchor, plan, libs)
            r_live = score_any(live, anchor, plan, libs)
            assert (r_live.found, r_live.truncated) == (r_full.found, r_full.truncated)
            assert abs(r_live.top_score - r_full.top_score) < 1e-12
            assert abs(r_live.score - r_full.score) < 1e-12
            assert r_live.witness == r_full.witness
            assert r_live.degrees == r_full.degrees
            fired += r_full.found
        assert fired > 0, name  # the planted cycle must fire round_trip
