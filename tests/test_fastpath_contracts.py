"""Contracts around the compiled fast paths (no hidden caches, edge cases).

* Nothing is memoised across calls: editing a term definition in the
  libs dict, or a graph's amounts in place, between calls changes fast
  and pure results alike; rebinding a structure array re-runs the CSR
  ordering check.
* ``find_all`` with non-int, infinite, NaN or absurdly large caps behaves
  exactly like the pure walk (the kernel mirrors the cap or falls back).
* The scatter count gate clamps literals past int64 (found=False, as in
  the reference) instead of raising OverflowError.
* ``groups.score_batch`` equals a ``score_any`` loop for every plan type,
  including the lazy eviction floors it leaves behind.
"""

from __future__ import annotations

import dataclasses
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from motifql import groups, groups_fast  # noqa: E402
from motifql import matcher as M  # noqa: E402
from motifql.degrees import libs_from_doc  # noqa: E402
from motifql.graph import TemporalGraph  # noqa: E402
from motifql.parser import (  # noqa: E402
    And, Call, Comparison, DistinctArg, Dur, Num, Ref, TermDef, parse_text,
)
from motifql.planner import compile_motif  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BASE = datetime(2022, 9, 3)

PATH_MOTIFS = """
motif rt_fast {
  match (a:account) -[e:transfer*2..4]-> (a)
  where e is time_ordered and span(e) <= 3d
  soft  each e: e.gap is soon and each e: e.amount ~= prev.amount within 20%
  score min
  emit when score >= 0.3
}
motif rt_deferred {
  match (a:account) -[e:transfer*2..4]-> (a)
  where e is time_ordered and span(e) <= 3d
  soft  each e: e.gap is soon and some e: e.amount ~= prev.amount within 10%
  score min
  emit when score >= 0.3
}
"""


@pytest.fixture(scope="module")
def doc():
    return parse_text((ROOT / "examples/amlworld.motif").read_text() + PATH_MOTIFS)


def _frame(seed: int, n_acc: int = 8, n: int = 260, minutes: int = 4 * 1440) -> pl.DataFrame:
    """Dense random transfers on a half-hour grid (ties, self-loops, cycles)."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        t = int(rng.integers(0, minutes // 30)) * 30
        a, b = int(rng.integers(0, n_acc)), int(rng.integers(0, n_acc))
        amt = float(rng.choice([100.0, 105.0, 95.0, 1000.0, 9990.0])) + float(rng.integers(0, 9))
        rows.append((t, a, b, amt))
    rows.sort(key=lambda r: r[0])
    return pl.DataFrame({
        "transfer_id": list(range(len(rows))),
        "timestamp": [BASE + timedelta(minutes=r[0]) for r in rows],
        "from_bank": ["001"] * len(rows),
        "from_account": [f"N{r[1]}" for r in rows],
        "to_bank": ["001"] * len(rows),
        "to_account": [f"N{r[2]}" for r in rows],
        "amount_usd": [r[3] for r in rows],
    })


def _canon(r) -> tuple:
    """Type- and bit-exact form of a ScoredResult / GroupResult."""
    def c(x):
        if isinstance(x, float):
            return ("f", x.hex())
        if isinstance(x, (tuple, list)):
            return tuple(c(i) for i in x)
        return (type(x).__name__, x)
    return c((r.found, r.witness, r.score, r.top_score, r.degrees, r.truncated,
              r.expansions))


def _both(fn):
    """fn() under the fast path and under the pure reference."""
    old = (M.PURE_PYTHON, groups.PURE_PYTHON)
    try:
        M.PURE_PYTHON = groups.PURE_PYTHON = False
        fast = fn()
        M.PURE_PYTHON = groups.PURE_PYTHON = True
        pure = fn()
    finally:
        M.PURE_PYTHON, groups.PURE_PYTHON = old
    return fast, pure


# -- no hidden caches -----------------------------------------------------------

def test_term_edit_between_calls_is_seen_by_path_fast_path(doc) -> None:
    """The reviewer's scenario: a term definition replaced in the libs dict
    between calls must reach the kernel (no plan memo keyed on identity)."""
    g = TemporalGraph.from_transfers(_frame(1))
    plan = compile_motif(doc, "rt_fast")
    terms, quants = libs_from_doc(doc)
    libs = (dict(terms), quants)  # one dict object, edited in place below
    anchors = list(range(g.n_edges))

    def run():
        single = [_canon(M.score(g, a, plan, libs, floor=M.EXACT_FLOOR)) for a in anchors]
        b = M.score_path_batch(g, anchors, plan, libs, floor=M.EXACT_FLOOR)
        sb = groups.score_batch(g, anchors, plan, libs, floor=M.EXACT_FLOOR)
        return single, b.top_score.tolist(), [t.hex() for t in sb[0].tolist()]

    before_fast, before_pure = _both(run)
    assert before_fast == before_pure
    assert M._compiled_plan(plan, libs, False, 0.0) is not None  # the kernel ran
    libs[0]["soon"] = TermDef("soon", (), "trapezoid",
                              (Dur(0.0), Dur(0.0), Dur(600.0), Dur(3600.0)))
    after_fast, after_pure = _both(run)
    assert after_fast == after_pure
    assert after_fast[0] != before_fast[0], "the edit should change some scores"
    # And back: the original definition gives the original results again.
    libs[0]["soon"] = terms["soon"]
    again_fast, _ = _both(run)
    assert again_fast == before_fast


def test_term_edit_between_calls_is_seen_by_group_fast_path(doc) -> None:
    g = TemporalGraph.from_transfers(_frame(2))
    terms, quants = libs_from_doc(doc)
    libs = (dict(terms), quants)
    anchors = list(range(g.n_edges))
    for name, term, new in (
            ("fan_out", "many", TermDef("many", (), "ramp", (Num(1.0), Num(2.0)))),
            ("bipartite_butterfly", "soon",
             TermDef("soon", (), "trapezoid", (Dur(0.0), Dur(0.0), Dur(60.0), Dur(7200.0))))):
        plan = compile_motif(doc, name)

        def run():
            return [_canon(groups.score_any(g, a, plan, libs)) for a in anchors]

        before_fast, before_pure = _both(run)
        assert before_fast == before_pure
        libs[0][term] = new
        after_fast, after_pure = _both(run)
        libs[0][term] = terms[term]
        assert after_fast == after_pure, name
        assert after_fast != before_fast, f"{name}: the edit should change some scores"


def test_graph_arrays_stay_writable() -> None:
    g = TemporalGraph.from_transfers(_frame(3))
    assert g.csr_ordered is True
    for name in ("times", "frm", "to", "amount", "in_times", "in_edges",
                 "out_times", "out_edges"):
        assert getattr(g, name).flags.writeable, name
    assert not hasattr(g, "in_frm") and not hasattr(g, "in_amount")  # no derived copies


@pytest.mark.parametrize("name", ["rt_fast", "scatter_gather"])
def test_in_place_amount_edit_is_seen(doc, name) -> None:
    plan = compile_motif(doc, name)
    libs = libs_from_doc(doc)
    g = TemporalGraph.from_transfers(_frame(4))
    anchors = list(range(g.n_edges))
    run = lambda: [_canon(groups.score_any(g, a, plan, libs)) for a in anchors]
    before_fast, before_pure = _both(run)
    g.amount[:] = g.amount * np.linspace(0.5, 1.5, g.n_edges, dtype=np.float32)
    assert M._graph_ok(g) and groups_fast.csr_time_ordered(g)  # fast path still on
    after_fast, after_pure = _both(run)
    assert before_fast == before_pure and after_fast == after_pure
    assert after_fast != before_fast, f"{name}: the edit should change some scores"


def test_rebinding_structure_rechecks_order() -> None:
    g = TemporalGraph.from_transfers(_frame(5))
    good = g.in_edges
    g.in_edges = good[::-1].copy()
    assert not g.csr_ordered
    assert not M._graph_ok(g) and not groups_fast.csr_time_ordered(g)
    g.in_edges = good
    assert g.csr_ordered and M._graph_ok(g)


def test_rebuilt_graph_with_new_amounts_agrees(doc) -> None:
    """A graph built from changed data scores the changed data."""
    frame = _frame(4)
    plan = compile_motif(doc, "rt_fast")
    libs = libs_from_doc(doc)
    g1 = TemporalGraph.from_transfers(frame)
    g2 = TemporalGraph.from_transfers(
        frame.with_columns(pl.col("amount_usd") * pl.Series(np.linspace(0.8, 1.2, frame.height))))
    anchors = list(range(g1.n_edges))
    r1 = _both(lambda: [_canon(M.score(g1, a, plan, libs, floor=M.EXACT_FLOOR)) for a in anchors])
    r2 = _both(lambda: [_canon(M.score(g2, a, plan, libs, floor=M.EXACT_FLOOR)) for a in anchors])
    assert r1[0] == r1[1] and r2[0] == r2[1]
    assert r1[0] != r2[0]


def test_unordered_graph_disables_both_fast_paths() -> None:
    g = TemporalGraph.from_transfers(_frame(5))
    bad = TemporalGraph(
        times=g.times, frm=g.frm, to=g.to, amount=g.amount, transfer_id=g.transfer_id,
        nodes=g.nodes, out_ptr=g.out_ptr, out_edges=g.out_edges, out_times=g.out_times,
        in_ptr=g.in_ptr, in_edges=g.in_edges[::-1].copy(), in_times=g.in_times,
        retention_s=g.retention_s)
    assert not bad.csr_ordered
    assert not M._graph_ok(bad) and not groups_fast.csr_time_ordered(bad)


# -- find_all caps ----------------------------------------------------------------

CAPS = [0, 1, 2, 2.5, 3.7, -1, -2.5, math.inf, -math.inf, math.nan, 10**30, 1e300,
        np.int64(3), np.float64(2.5), True]


@pytest.mark.parametrize("cap", CAPS, ids=repr)
def test_find_all_odd_caps_match_pure(cap) -> None:
    g_f = TemporalGraph.from_transfers(_frame(6, n_acc=5, n=300))
    g_p = TemporalGraph.from_transfers(_frame(6, n_acc=5, n=300))
    g_x = TemporalGraph.from_transfers(_frame(6, n_acc=5, n=300))  # uncapped census
    motifs = [M.ROUND_TRIP, dataclasses.replace(M.ROUND_TRIP, max_expansions=40),
              dataclasses.replace(M.LAYERING_CHAIN, min_hops=2, max_hops=4)]
    many = False
    for motif in motifs:
        for a in range(0, g_f.n_edges, 4):
            fast, pure = None, None
            old = M.PURE_PYTHON
            try:
                M.PURE_PYTHON = False
                fast = M.find_all(g_f, a, motif, cap=cap)
                M.PURE_PYTHON = True
                pure = M.find_all(g_p, a, motif, cap=cap)
            finally:
                M.PURE_PYTHON = old
            assert fast == pure, (motif.name, a, cap)
            many |= len(M._search_py(g_x, a, motif, True, math.inf)[0]) > 3
    assert many, "no anchor had more witnesses than the small caps"
    assert np.array_equal(g_f._in_lo, g_p._in_lo)


def test_enum_cap_mirrors_the_pure_stop_rule() -> None:
    ec = M._enum_cap
    assert ec(2.5, True, 100) == 3 and ec(3, True, 100) == 3 and ec(0, True, 100) == 1
    assert ec(-math.inf, True, 100) == 1
    assert ec(math.inf, True, 100) == 101 and ec(math.nan, True, 100) == 101
    assert ec(10**30, True, 100) == 101  # clamped: never reached, never allocated
    assert ec(math.inf, True, 10**9) is None  # buffer too large: pure walk
    assert ec(5, True, math.inf) is None  # non-int budget: pure walk
    assert ec("5", True, 100) is None
    assert ec(5, False, 100) == 1


def test_find_all_huge_budget_and_infinite_cap_fall_back() -> None:
    g = TemporalGraph.from_transfers(_frame(7, n_acc=5, n=200))
    for motif, cap in ((dataclasses.replace(M.ROUND_TRIP, max_expansions=10**12), math.inf),
                       (dataclasses.replace(M.ROUND_TRIP, max_expansions=math.inf), 10**30)):
        for a in range(0, g.n_edges, 9):
            fast, pure = _both(lambda: M.find_all(g, a, motif, cap=cap))
            assert fast == pure


# -- scatter count gate -----------------------------------------------------------

def _count_having(op: str, lit) -> Comparison:
    return Comparison(op, Call("count", (DistinctArg(Ref(("m",))),)), lit)


@pytest.mark.parametrize("lit", [Num(2.0**63), Num(2.0**64 + 7), Num(1e300), Num(10**400),
                                 Num(float(2**62)), Num(math.inf), Num(math.nan)], ids=repr)
@pytest.mark.parametrize("op", [">=", ">"])
def test_count_gate_clamps_huge_literals(doc, lit, op) -> None:
    having = _count_having(op, lit)
    assert groups_fast._count_gate(having, "m") == groups_fast._NEVER
    assert groups_fast._count_gate(And((having,)), "m") == groups_fast._NEVER
    plan = dataclasses.replace(compile_motif(doc, "scatter_gather"), having=having)
    libs = libs_from_doc(doc)
    g = TemporalGraph.from_transfers(_frame(8, n_acc=6, n=300))
    fast, pure = _both(lambda: [_canon(groups.score_any(g, a, plan, libs))
                                for a in range(g.n_edges)])
    assert fast == pure
    assert not any(k[0][1] for k in fast)  # found=False everywhere


def test_count_gate_negative_huge_literal_is_no_gate() -> None:
    assert groups_fast._count_gate(_count_having(">=", Num(-(10**400))), "m") == 0
    assert groups_fast._count_gate(_count_having(">", Num(-1e300)), "m") == 0


# -- score_batch ------------------------------------------------------------------

def _loop(graph, positions, plan, libs, **kw):
    rs = [groups.score_any(graph, p, plan, libs, **kw) for p in positions]
    return (np.array([r.top_score for r in rs], dtype=np.float64),
            np.array([r.found for r in rs], dtype=bool),
            np.array([r.expansions for r in rs], dtype=np.float64),
            np.array([r.truncated for r in rs], dtype=bool))


BATCH_MODES = [dict(), dict(prune=False), dict(floor=M.EXACT_FLOOR),
               dict(crisp=True, crisp_alpha=0.5), dict(crisp=True, crisp_alpha=1.5)]


@pytest.mark.parametrize("name", ["rt_fast", "rt_deferred", "round_trip", "layering_chain",
                                  "fan_out", "fan_in", "scatter_gather", "gather_scatter",
                                  "structuring_fan_in", "bipartite_butterfly"])
@pytest.mark.parametrize("evict_h", [None, 30])
def test_score_batch_equals_score_any_loop(doc, name, evict_h) -> None:
    plan = compile_motif(doc, name)
    libs = libs_from_doc(doc)
    frame = _frame(9, n_acc=7, n=240)
    for kw in BATCH_MODES:
        g_b, g_l = TemporalGraph.from_transfers(frame), TemporalGraph.from_transfers(frame)
        positions = np.arange(g_b.n_edges, dtype=np.int64)[::-1][::2].copy()
        if evict_h is not None:
            t = int(g_b.times[g_b.n_edges // 2]) - evict_h * 3600
            g_b.evict_before(t)
            g_l.evict_before(t)
        try:
            want = _loop(g_l, positions.tolist(), plan, libs, **kw)
        except ValueError:
            with pytest.raises(ValueError):
                groups.score_batch(g_b, positions, plan, libs, **kw)
            continue
        got = groups.score_batch(g_b, positions, plan, libs, **kw)
        for x, y, dt in zip(got, want, (np.float64, np.bool_, np.float64, np.bool_)):
            assert x.dtype == dt and x.shape == y.shape
            assert x.tobytes() == y.tobytes(), (name, kw)
        assert np.array_equal(g_b._in_lo, g_l._in_lo), (name, kw)
        assert np.array_equal(g_b._out_lo, g_l._out_lo), (name, kw)


def test_score_batch_empty_and_unknown_plans(doc) -> None:
    g = TemporalGraph.from_transfers(_frame(10))
    libs = libs_from_doc(doc)
    for name in ("rt_fast", "fan_out", "bipartite_butterfly"):
        out = groups.score_batch(g, np.empty(0, dtype=np.int64), compile_motif(doc, name), libs)
        assert [a.shape for a in out] == [(0,)] * 4
    assert [a.shape for a in groups.score_batch(g, [], object(), libs)] == [(0,)] * 4
    with pytest.raises(TypeError):
        groups.score_batch(g, [0], object(), libs)


# -- final-review regressions -------------------------------------------------------

def _rebuild(g, **over) -> TemporalGraph:
    kw = dict(times=g.times, frm=g.frm, to=g.to, amount=g.amount, transfer_id=g.transfer_id,
              nodes=g.nodes, out_ptr=g.out_ptr, out_edges=g.out_edges, out_times=g.out_times,
              in_ptr=g.in_ptr, in_edges=g.in_edges, in_times=g.in_times,
              retention_s=g.retention_s)
    kw.update(over)
    return TemporalGraph(**kw)


def test_pickled_graph_keeps_order_check() -> None:
    import copy
    import pickle
    g = TemporalGraph.from_transfers(_frame(7))
    for clone in (pickle.loads(pickle.dumps(g)), copy.deepcopy(g)):
        assert clone.csr_ordered and M._graph_ok(clone)
        assert clone.amount.flags.writeable
        clone.in_edges = clone.in_edges[::-1].copy()
        assert not clone.csr_ordered


def test_out_of_range_node_ids_disable_fast_paths() -> None:
    g = TemporalGraph.from_transfers(_frame(8))
    frm = g.frm.copy()
    frm[::7] = -1
    assert not _rebuild(g, frm=frm).csr_ordered


def test_csr_segment_membership_is_checked() -> None:
    g = TemporalGraph.from_transfers(_frame(9))
    to = g.to.copy()
    to[::5] = (to[::5] + 1) % g.n_nodes  # CSR left as built for the old `to`
    assert not _rebuild(g, to=to).csr_ordered


def test_tie_seed_is_read_per_call(doc, monkeypatch) -> None:
    plan = compile_motif(doc, "bipartite_butterfly")
    libs = libs_from_doc(doc)
    g = TemporalGraph.from_transfers(_frame(10))
    run = lambda: [_canon(groups.score_any(g, a, plan, libs)) for a in range(g.n_edges)]
    run()  # first fast call loads groups_fast with the default seed
    monkeypatch.setattr(groups, "_TIE_SEED", "motifql-member-tie-test")
    assert groups_fast._hkey(7) == groups._hkey(7)
    fast, pure = _both(run)
    assert fast == pure


@pytest.mark.parametrize("name", ["fan_out", "scatter_gather", "gather_scatter",
                                  "bipartite_butterfly", "rt_fast"])
def test_fractional_watermark_matches_pure(doc, name) -> None:
    plan = compile_motif(doc, name)
    libs = libs_from_doc(doc)
    frame = _frame(11)

    def run():
        g = TemporalGraph.from_transfers(frame)
        out = []
        for a in range(g.n_edges):
            g.evict_before(int(g.times[a]) - 6 * 3600 + 0.5)
            out.append(_canon(groups.score_any(g, a, plan, libs)))
        return out, g._in_lo.tolist(), g._out_lo.tolist()
    fast, pure = _both(run)
    assert fast == pure


@pytest.mark.parametrize("anchor", [5.0, np.float64(5.0), True])
def test_non_integer_anchor_errors_like_reference(doc, anchor) -> None:
    g = TemporalGraph.from_transfers(_frame(12))
    libs = libs_from_doc(doc)
    for name in ("fan_out", "scatter_gather", "gather_scatter", "bipartite_butterfly"):
        plan = compile_motif(doc, name)

        def run():
            try:
                return ("ok", _canon(groups.score_any(g, anchor, plan, libs)))
            except Exception as exc:  # noqa: BLE001
                return ("raise", type(exc).__name__)
        fast, pure = _both(run)
        assert fast == pure


def test_warm_up_leaves_real_graph_untouched(doc) -> None:
    g = TemporalGraph.from_transfers(_frame(13))
    g.evict_before(int(g.times[40]))
    before = (g._in_lo.copy(), g._out_lo.copy(), g.watermark)
    for name in ("rt_fast", "fan_out", "scatter_gather", "bipartite_butterfly"):
        groups.warm_up(compile_motif(doc, name), libs_from_doc(doc))
    assert np.array_equal(g._in_lo, before[0]) and np.array_equal(g._out_lo, before[1])
    assert g.watermark == before[2]
