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
import motifql.groups as _G  # noqa: E402
import motifql.groups_fast as _GF  # noqa: E402
import motifql.matcher as _M  # noqa: E402

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


def test_append_position_valid_across_auto_rebuild() -> None:
    # The auto-rebuild runs before the edge is written: even when it
    # compacts expired edges away, the returned position is the new
    # edge's (pre-fix it pointed past n_edges, or at another transfer).
    rng = np.random.default_rng(43)
    accts = [f"A{i:02d}" for i in range(6)]
    df = _frame(_random_rows(rng, 10, accts))
    g = TemporalGraph.from_transfers(df)
    g.rebuild_threshold = 1
    t1 = int(g.times[-1]) + 60
    g.append(t1, "001", "X", "001", "Y", 1.0, 100)
    g.evict_before(int(g.times[5]))  # expire the first 5 edges
    p = g.append(t1 + 60, "001", "X", "001", "Y", 2.0, 101)
    assert g._appended == 1  # the threshold fired first and reset
    assert p == g.n_edges - 1
    assert int(g.transfer_id[p]) == 101
    assert g.n_edges == 5 + 2  # 5 survivors + 2 appends (100 compacted in)


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


def _python_results(graph, plan, libs, anchors):
    """score_any with the compiled kernels forced off (restores flags)."""
    old_m, old_g = _M.PURE_PYTHON, _G.PURE_PYTHON
    _M.PURE_PYTHON = _G.PURE_PYTHON = True
    try:
        return [score_any(graph, a, plan, libs) for a in anchors]
    finally:
        _M.PURE_PYTHON, _G.PURE_PYTHON = old_m, old_g


def _assert_same_results(fast, python, label: str) -> None:
    assert len(fast) == len(python)
    for i, (r_fast, r_py) in enumerate(zip(fast, python)):
        assert (r_fast.found, r_fast.truncated) == (r_py.found, r_py.truncated), (label, i)
        assert r_fast.score == r_py.score, (label, i)
        assert r_fast.top_score == r_py.top_score, (label, i)
        assert r_fast.witness == r_py.witness, (label, i)
        assert r_fast.degrees == r_py.degrees, (label, i)
        assert r_fast.expansions == r_py.expansions, (label, i)


def test_fast_python_parity_with_overflow() -> None:
    # The compiled kernels read CSR segments only, so any pending overflow
    # must close the fast path (silently missing edges is wrong, not slow).
    # The gate assertions run while the overflow touches existing nodes
    # only: the CSR check alone would pass there, so only the guard can
    # close the gate.
    rng = np.random.default_rng(31)
    accts = [f"A{i:02d}" for i in range(10)]
    df = _frame(_random_rows(rng, 150, accts))
    live = TemporalGraph.from_transfers(df.head(100))
    for row in df.slice(100).iter_rows(named=True):
        _append_row(live, row)
    doc = parse_file(MOTIFS)
    libs = libs_from_doc(doc)
    assert live._appended == 50 and live._csr_nodes == len(live.nodes)
    assert live.csr_ordered is False
    assert _M._graph_ok(live) is False
    assert _GF.csr_time_ordered(live) is False
    anchors = list(range(live.n_edges))
    for name in ("round_trip", "fan_out", "scatter_gather"):
        plan = compile_motif(doc, name)
        fast = [score_any(live, a, plan, libs) for a in anchors]
        _assert_same_results(fast, _python_results(live, plan, libs, anchors),
                             f"{name} pre-rebuild")
    # New-node edge plus an out-of-order edge between existing nodes, so
    # older anchors' windows include an appended edge too; agreement must
    # hold before and after the rebuild drains them.
    t_new = int(live.times[-1]) + 3600
    t_old = int(live.times[50])
    live.append(t_new, "001", "ZZ1", "001", "ZZ2", 500.0, 1000)
    live.append(t_old, "001", accts[0], "001", accts[1], 700.0, 1001)
    anchors = list(range(live.n_edges))
    for name in ("round_trip", "fan_out", "scatter_gather"):
        plan = compile_motif(doc, name)
        fast = [score_any(live, a, plan, libs) for a in anchors]
        _assert_same_results(fast, _python_results(live, plan, libs, anchors),
                             f"{name} mixed-overflow")
    live.rebuild()
    # Rebuild sorts survivors into time order, so even the out-of-order
    # edge lands in place and the fast path re-engages; results agree.
    assert live.csr_ordered is True
    assert _M._graph_ok(live) is True
    assert _GF.csr_time_ordered(live) is True
    anchors = list(range(live.n_edges))  # positions renumber on rebuild
    for name in ("round_trip", "fan_out", "scatter_gather"):
        plan = compile_motif(doc, name)
        fast = [score_any(live, a, plan, libs) for a in anchors]
        _assert_same_results(fast, _python_results(live, plan, libs, anchors),
                             f"{name} post-rebuild")


def test_rebuild_compacts_expired_edges() -> None:
    # Evicted edges are freed at rebuild (transfer_ids stable, positions
    # renumbered); survivors answer exactly like a fresh graph built
    # from the unexpired rows. Node ids stay put, so compare by key.
    rng = np.random.default_rng(41)
    accts = [f"A{i:02d}" for i in range(10)]
    df = _frame(_random_rows(rng, 100, accts))
    g = TemporalGraph.from_transfers(df)
    cutoff = int(g.times[60])
    g.evict_before(cutoff)
    g.rebuild()
    assert g.n_edges == 40
    assert g.times.tolist() == sorted(g.times.tolist())
    fresh = TemporalGraph.from_transfers(
        df.filter(pl.col("timestamp") >= df["timestamp"][60]))
    assert fresh.n_edges == 40
    assert sorted(g.transfer_id.tolist()) == sorted(fresh.transfer_id.tolist())
    for key in fresh.nodes:
        bank, acct = key.split("/", 1)
        a, b = g.node_id(bank, acct), fresh.node_id(bank, acct)
        for before_t in (cutoff, int(g.times[-1]) + 60):
            for w in (3600, 30 * SECS_PER_DAY):
                for fn in ("incoming", "outgoing"):
                    ge, _ = getattr(g, fn)(a, before_t, w)
                    fe, _ = getattr(fresh, fn)(b, before_t, w)
                    assert ({int(g.transfer_id[e]) for e in ge}
                            == {int(fresh.transfer_id[e]) for e in fe})
    gpos = {int(t): i for i, t in enumerate(g.transfer_id.tolist())}
    fpos = {int(t): i for i, t in enumerate(fresh.transfer_id.tolist())}
    doc = parse_file(MOTIFS)
    libs = libs_from_doc(doc)
    for name in ("round_trip", "fan_out"):
        plan = compile_motif(doc, name)
        for tid in gpos:
            r = score_any(g, gpos[tid], plan, libs)
            s = score_any(fresh, fpos[tid], plan, libs)
            assert (r.found, r.score, r.top_score) == (s.found, s.score, s.top_score)
            assert ([int(g.transfer_id[e]) for e in r.witness]
                    == [int(fresh.transfer_id[e]) for e in s.witness])
    # Evicting everything rebuilds to an empty graph that still appends.
    g.evict_before(int(g.times[-1]) + 1)
    g.rebuild()
    assert g.n_edges == 0
    assert g.incoming(0, 2**40, SECS_PER_DAY)[0].tolist() == []
    t = _epoch(BASE + timedelta(minutes=10_000))
    assert g.append(t, "001", "A", "001", "B", 5.0, 999) == 0


def test_fast_path_reengages_after_in_order_rebuild() -> None:
    # In-order appends keep every invariant; rebuild() reopens the fast
    # path, and its outputs (expansions included) match the reference.
    rng = np.random.default_rng(33)
    accts = [f"A{i:02d}" for i in range(10)]
    df = _frame(_random_rows(rng, 150, accts))
    live = TemporalGraph.from_transfers(df.head(100))
    for row in df.slice(100).iter_rows(named=True):
        _append_row(live, row)
    t_new = int(live.times[-1]) + 3600
    live.append(t_new, "001", "ZZ1", "001", "ZZ2", 500.0, 1000)
    assert live.csr_ordered is False  # overflow pending, even fully ordered
    live.rebuild()
    assert live.csr_ordered is True
    assert _M._graph_ok(live) is True
    assert _GF.csr_time_ordered(live) is True
    anchors = list(range(live.n_edges))
    doc = parse_file(MOTIFS)
    libs = libs_from_doc(doc)
    for name in ("round_trip", "fan_out", "scatter_gather"):
        plan = compile_motif(doc, name)
        fast = [score_any(live, a, plan, libs) for a in anchors]
        _assert_same_results(fast, _python_results(live, plan, libs, anchors),
                             f"{name} rebuilt")


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
