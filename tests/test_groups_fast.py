"""Parity tests: compiled group executors vs the pure-Python reference.

`motifql.groups_fast` must reproduce `motifql.groups`' reference
executors bit for bit — found, witness, score, top_score, degrees,
truncated, expansions, and the graph's lazy eviction floors. Each test
scores the same anchors on two graphs built from the same frame (one
per path, so floor side effects stay separate) and compares type-strict,
float.hex-exact canonical forms. Synthetic graphs are small and dense:
few accounts, coarse timestamps (same-minute ties), self-loops and a hub
account, so every branch (ties, fallbacks, caps) is exercised.
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
from motifql import groups, groups_fast  # noqa: E402
from motifql.degrees import libs_from_doc  # noqa: E402
from motifql.graph import TemporalGraph  # noqa: E402
from motifql.matcher import EXACT_FLOOR  # noqa: E402
from motifql.parser import And, Call, Comparison, DistinctArg, Money, Num, Ref, parse_text  # noqa: E402
from motifql.planner import compile_motif  # noqa: E402

BASE = datetime(2022, 9, 3)
ROOT = Path(__file__).resolve().parent.parent

EXTRA = """
term soon2 = trapezoid(0h, 0h, 6h, 48h)
term big = ramp(50 USD, 900 USD)

motif fan_quant {
  match (s:account) -[e:transfer]-> (r:account)
  group by s over r window 2d
  having count(distinct r) >= 2
  soft  most r: e.amount ~= avg(e.amount) within 30%
    and sum(e.amount) / count(distinct r) >= 50 USD
  score product
  emit when score >= 0.3
}

motif fan_in_each {
  match (x:account) -[e:transfer]-> (t:account)
  group by t over x window 3d
  having count(distinct x) >= 1
  soft  each x: e.amount is big
    or some x: e.amount >= 900 USD
  score min
  emit when score >= 0.5
}

motif fan_mean_named {
  match (s:account) -[e:transfer]-> (r:account)
  group by s over r window 2d
  having count(distinct r) >= 2 and count(distinct r) > 1
  soft  @mostly r: e.amount ~= max(e.amount) within 50%
    and not (min(e.amount) >= 500 USD)
  score min
  emit when score >= 0.4
}

motif scatter_quant {
  match (s:account) -[leg1:transfer]-> (m:account) -[leg2:transfer]-> (d:account)
  group by s, d over m window 3d
  where leg2.time > leg1.time
    and s != d
  having count(distinct m) >= 1
  soft  most m: leg2.amount ~= leg1.amount within 15%
    and each m: leg2.gap is soon2
  score min
  emit when score >= 0.5
}

motif scatter_sum_gate {
  match (s:account) -[leg1:transfer]-> (m:account) -[leg2:transfer]-> (d:account)
  group by s, d over m window 7d
  where leg2.time > leg1.time
    and s != d
  having count(distinct m) >= 1
  soft  avg(leg2.amount) ~= avg(leg1.amount) within 40%
  score product
  emit when score >= 0.5
}

motif scatter_frac_gate {
  match (s:account) -[leg1:transfer]-> (m:account) -[leg2:transfer]-> (d:account)
  group by s, d over m window 2d
  where leg2.time > leg1.time
    and s != d
  having count(distinct m) > 1.5
     and count(distinct m) >= 1
  soft  count(distinct m) is several
    and at least 2 m: leg1.amount >= 100 USD
  score product
  emit when score >= 0.25
}

motif gather_quant {
  match (x:account) -[g:transfer]-> (c:account),
        (c) -[p:transfer]-> (y:account)
  group by c over x, y window 3d
  where p.time > g.time
  having count(distinct x) >= 1
  soft  sum(p.amount) ~= sum(g.amount) within 20%
    and most x: g.amount ~= avg(g.amount) within 50%
    and some y: p.amount >= 100 USD
  score product
  emit when score >= 0.2
}

motif bf_patient {
  match (x1:account) -[e1:transfer]-> (y1:account),
        (x1) -[e2:transfer]-> (y2:account),
        (x2:account) -[e3:transfer]-> (y1),
        (x2) -[e4:transfer]-> (y2)
  where x1 != x2
    and y1 != y2
    and span(all) <= 5d
  soft  span(all) is not soon2
  score min
  emit when score >= 0.6
}

motif bf_bare {
  match (x1:account) -[e1:transfer]-> (y1:account),
        (x1) -[e2:transfer]-> (y2:account),
        (x2:account) -[e3:transfer]-> (y1),
        (x2) -[e4:transfer]-> (y2)
  where x1 != x2
    and y1 != y2
    and span(all) <= 3d
  score min
  emit when score >= 0.6
}

motif fan_span {
  match (s:account) -[e:transfer]-> (r:account)
  group by s over r window 2d
  having count(distinct r) >= 2
  soft  span(e) is not soon2 or min(e.time) < max(e.time)
  score min
  emit when score >= 0.5
}

motif scatter_span {
  match (s:account) -[leg1:transfer]-> (m:account) -[leg2:transfer]-> (d:account)
  group by s, d over m window 3d
  where leg2.time > leg1.time
    and s != d
  having count(distinct m) >= 2
  soft  span(leg1) is soon2 and avg(leg2.time) > avg(leg1.time)
  score product
  emit when score >= 0.5
}

motif gather_span {
  match (x:account) -[g:transfer]-> (c:account),
        (c) -[p:transfer]-> (y:account)
  group by c over x, y window 3d
  where p.time > g.time
  having count(distinct x) >= 1
  soft  span(g) is not soon2 or min(p.time) > max(g.time)
  score min
  emit when score >= 0.5
}
"""

AML = ("fan_out", "fan_in", "structuring_fan_in", "scatter_gather",
       "gather_scatter", "bipartite_butterfly")
CUSTOM = ("fan_quant", "fan_in_each", "fan_bare", "fan_mean_named", "scatter_quant",
          "scatter_sum_gate", "scatter_frac_gate", "scatter_bare", "scatter_sum_having",
          "gather_quant", "gather_bare", "gather_nosoft", "fan_span", "scatter_span",
          "gather_span", "bf_patient", "bf_bare")

# The planner only emits `count(distinct m) >= N` having conjuncts; the
# executors (and the fast path's count gate) also take other shapes, so
# these derived plans pin parity where no count gate can be proven.
_SUM_GT0 = Comparison(">", Call("sum", (Ref(("leg1", "amount")),)), Money(0.0, "USD"))
DERIVED = {
    "fan_bare": ("fan_out", dict(having=None, soft=None)),
    "scatter_bare": ("scatter_gather", dict(having=None, soft=None)),
    "scatter_sum_having": ("scatter_sum_gate", dict(having=And((_SUM_GT0, Comparison(
        ">=", Call("count", (DistinctArg(Ref(("m",))),)), Num(2.0)))))),
    "gather_bare": ("gather_scatter", dict(having=None)),
    "gather_nosoft": ("gather_scatter", dict(soft=None)),
}


def _plan(doc, name):
    if name in DERIVED:
        base, changes = DERIVED[name]
        return dataclasses.replace(compile_motif(doc, base), **changes)
    return compile_motif(doc, name)


@pytest.fixture(scope="module")
def doc():
    return parse_text((ROOT / "examples/amlworld.motif").read_text() + EXTRA)


def _frame(rows) -> pl.DataFrame:
    return pl.DataFrame({
        "transfer_id": list(range(len(rows))),
        "timestamp": [BASE + timedelta(minutes=int(r[0])) for r in rows],
        "from_bank": ["001"] * len(rows),
        "from_account": [r[1] for r in rows],
        "to_bank": ["001"] * len(rows),
        "to_account": [r[2] for r in rows],
        "amount_usd": [float(r[3]) for r in rows],
    })


def _random_rows(seed: int, n: int = 260, n_acc: int = 9, step_min: int = 90,
                 span_days: int = 9, hub_share: float = 0.3) -> list[tuple]:
    """Dense random transfers: coarse times (ties), self-loops, one hub."""
    rng = np.random.default_rng(seed)
    accs = [f"A{i}" for i in range(n_acc)]
    amounts = [10.0, 50.0, 99.0, 100.0, 105.5, 1000.0, 9600.0, 9990.0]
    slots = span_days * 1440 // step_min
    rows = []
    for _ in range(n):
        t = int(rng.integers(0, slots)) * step_min
        f = "HUB" if rng.random() < hub_share / 2 else accs[rng.integers(n_acc)]
        to = "HUB" if rng.random() < hub_share / 2 else accs[rng.integers(n_acc)]
        rows.append((t, f, to, amounts[rng.integers(len(amounts))]))
    rows.sort(key=lambda r: r[0])
    return rows


def canon(x):
    if isinstance(x, bool):
        return ("bool", x)
    if isinstance(x, float):
        return (type(x).__name__, x.hex())
    if isinstance(x, int):
        return (type(x).__name__, int(x))
    if isinstance(x, (tuple, list)):
        return (type(x).__name__, tuple(canon(i) for i in x))
    if x is None or isinstance(x, str):
        return x
    return (type(x).__name__, repr(x))


def _key(r) -> tuple:
    return canon((r.found, r.witness, r.score, r.top_score, r.degrees, r.truncated,
                  r.expansions))


def _modes(plan):
    return [(False, 0.0), (True, 0.0), (True, float(plan.emit))]


def _run(graph, plan, libs, anchors, pure: bool, crisp, alpha, evict_lag=None):
    old = groups.PURE_PYTHON
    groups.PURE_PYTHON = pure
    try:
        out = []
        for a in anchors:
            if evict_lag is not None:
                graph.evict_before(int(graph.times[a]) - evict_lag)
            r = groups.score_any(graph, a, plan, libs, prune=True, crisp=crisp,
                                 crisp_alpha=alpha, floor=None)
            out.append(_key(r))
        return out
    finally:
        groups.PURE_PYTHON = old


def _assert_parity(frame, doc, names, evict_lag=None, require_trunc=False):
    libs = libs_from_doc(doc)
    any_trunc = False
    for name in names:
        plan = _plan(doc, name)
        for crisp, alpha in _modes(plan):
            g_ref = TemporalGraph.from_transfers(frame)
            g_fast = TemporalGraph.from_transfers(frame)
            anchors = list(range(g_ref.n_edges))
            ref = _run(g_ref, plan, libs, anchors, True, crisp, alpha, evict_lag)
            fast = _run(g_fast, plan, libs, anchors, False, crisp, alpha, evict_lag)
            for a, x, y in zip(anchors, ref, fast):
                assert x == y, (name, crisp, alpha, a, x, y)
            assert np.array_equal(g_ref._in_lo, g_fast._in_lo), name
            assert np.array_equal(g_ref._out_lo, g_fast._out_lo), name
            any_trunc |= any(k[1][5] == ("bool", True) for k in ref)
    if require_trunc:
        assert any_trunc, "caps never bound: the truncation branches went untested"


# -- invariant the kernels rely on -----------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_csr_segments_are_time_position_ordered(seed) -> None:
    rows = _random_rows(seed)
    shuffled = list(rows)
    np.random.default_rng(seed).shuffle(shuffled)  # from_transfers re-sorts
    for frame in (_frame(rows), _frame(shuffled)):
        g = TemporalGraph.from_transfers(frame)
        assert np.all(np.diff(g.times) >= 0)
        for ptr, edges in ((g.out_ptr, g.out_edges), (g.in_ptr, g.in_edges)):
            for node in range(g.n_nodes):
                seg = [int(e) for e in edges[ptr[node]:ptr[node + 1]]]
                assert seg == sorted(seg, key=lambda p: (int(g.times[p]), p))
                assert seg == sorted(seg)  # position order == (time, position) order
        assert groups_fast.csr_time_ordered(g)


def test_unordered_graph_falls_back_to_reference(doc) -> None:
    g = TemporalGraph.from_transfers(_frame(_random_rows(3, n=60)))
    bad = TemporalGraph(
        times=g.times[::-1].copy(), frm=g.frm, to=g.to, amount=g.amount,
        transfer_id=g.transfer_id, nodes=g.nodes, out_ptr=g.out_ptr,
        out_edges=g.out_edges, out_times=g.out_times, in_ptr=g.in_ptr,
        in_edges=g.in_edges, in_times=g.in_times, retention_s=g.retention_s)
    assert not groups_fast.csr_time_ordered(bad)
    plan = compile_motif(doc, "fan_out")
    assert groups_fast.score_group(bad, 5, plan, libs_from_doc(doc)) is None


def test_hkey_matches_reference() -> None:
    for p in list(range(2000)) + [2**32 - 1, 5_078_344]:
        assert groups_fast._hkey(p) == groups._hkey(p)
    for p in (np.uint32(7), np.int64(123456), np.int32(42)):
        assert groups_fast._hkey(p) == groups._hkey(p)


def test_tied_latest_matches_reference() -> None:
    g = TemporalGraph.from_transfers(_frame(_random_rows(4, n=200, step_min=600)))
    rng = np.random.default_rng(0)
    for _ in range(300):
        k = int(rng.integers(0, 12))
        edges = sorted(int(e) for e in rng.choice(g.n_edges, k, replace=False))
        assert groups_fast._tied_latest(g.times, edges) == groups._tied_latest(g, edges)


# -- executor parity --------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_parity_reference_motifs(doc, seed) -> None:
    _assert_parity(_frame(_random_rows(seed)), doc, AML)


@pytest.mark.parametrize("seed", [10, 11])
def test_parity_custom_motifs(doc, seed) -> None:
    _assert_parity(_frame(_random_rows(seed, n=180)), doc, CUSTOM)


def test_parity_heavy_ties_and_self_loops(doc) -> None:
    # Two time slots only: almost every member has tied latest edges.
    rows = _random_rows(5, n=220, n_acc=5, step_min=1440, span_days=2, hub_share=0.5)
    rows += [(2880, "A1", "A1", 50.0), (2880, "HUB", "HUB", 50.0)]
    _assert_parity(_frame(rows), doc, AML + ("scatter_quant", "gather_quant", "fan_quant"))


def test_parity_tie_cap_truncates(doc, monkeypatch) -> None:
    monkeypatch.setattr(groups, "MAX_TIED_READINGS", 2)
    rows = _random_rows(6, n=200, n_acc=4, step_min=1440, span_days=3, hub_share=0.6)
    _assert_parity(_frame(rows), doc,
                   ("fan_out", "structuring_fan_in", "scatter_quant", "gather_quant",
                    "scatter_gather"), require_trunc=True)


def test_parity_hub_caps_truncate(doc, monkeypatch) -> None:
    monkeypatch.setattr(groups, "DEFAULT_MAX_FANOUT", 4)
    monkeypatch.setattr(groups, "DEFAULT_MAX_EXPANSIONS", 9)
    monkeypatch.setattr(groups, "SCATTER_MAX_SENDERS", 2)
    monkeypatch.setattr(groups, "SCATTER_MAX_LIST", 3)
    monkeypatch.setattr(groups, "SCATTER_MAX_EXPANSIONS", 8)
    monkeypatch.setattr(groups, "BUTTERFLY_MAX_PARTNERS", 2)
    rows = _random_rows(7, n=240, n_acc=6, hub_share=0.6)
    _assert_parity(_frame(rows), doc, AML + ("scatter_quant", "scatter_bare", "gather_quant",
                                             "bf_patient"), require_trunc=True)


@pytest.mark.parametrize("cap", [0, 1])
def test_parity_degenerate_caps(doc, monkeypatch, cap) -> None:
    # Python slicing semantics of x[-0:] (keeps everything) must carry over.
    monkeypatch.setattr(groups, "SCATTER_MAX_SENDERS", cap)
    monkeypatch.setattr(groups, "SCATTER_MAX_LIST", cap)
    monkeypatch.setattr(groups, "DEFAULT_MAX_FANOUT", cap)
    monkeypatch.setattr(groups, "BUTTERFLY_MAX_PARTNERS", cap)
    _assert_parity(_frame(_random_rows(8, n=150)), doc, AML)


@pytest.mark.parametrize("strategy", [1, 2])
def test_parity_scatter_strategies(doc, monkeypatch, strategy) -> None:
    monkeypatch.setattr(groups_fast, "SCATTER_STRATEGY", strategy)
    for seed in (20, 21):
        _assert_parity(_frame(_random_rows(seed, n=180, hub_share=0.5)), doc,
                       ("scatter_gather", "scatter_quant", "scatter_sum_gate",
                        "scatter_frac_gate", "scatter_bare"))


def test_parity_scatter_strategies_under_caps(doc, monkeypatch) -> None:
    monkeypatch.setattr(groups, "SCATTER_MAX_LIST", 2)
    for strategy in (1, 2):
        monkeypatch.setattr(groups_fast, "SCATTER_STRATEGY", strategy)
        _assert_parity(_frame(_random_rows(22, hub_share=0.6)), doc,
                       ("scatter_gather", "scatter_quant"), require_trunc=True)


@pytest.mark.parametrize("lag_h", [6, 30])
def test_parity_lazy_eviction(doc, lag_h) -> None:
    # The watermark cuts into every window (lags shorter than 2d-7d).
    _assert_parity(_frame(_random_rows(9)), doc, AML + ("scatter_quant", "gather_quant"),
                   evict_lag=lag_h * 3600)


def test_parity_empty_windows(doc) -> None:
    # Every anchor's lookups are empty: lone edges days apart.
    rows = [(i * 3 * 1440, f"S{i}", f"R{i}", 100.0) for i in range(6)]
    rows += [(20 * 1440, "S9", "S9", 5.0)]
    _assert_parity(_frame(rows), doc, AML + CUSTOM)


def test_parity_all_modes_on_score_any(doc) -> None:
    # prune / floor are path-only knobs, but sweep them anyway.
    libs = libs_from_doc(doc)
    frame = _frame(_random_rows(12))
    g_ref, g_fast = TemporalGraph.from_transfers(frame), TemporalGraph.from_transfers(frame)
    for name in AML:
        plan = compile_motif(doc, name)
        for prune in (True, False):
            for floor in (None, EXACT_FLOOR):
                for crisp, alpha in _modes(plan):
                    for a in range(0, g_ref.n_edges, 3):
                        groups.PURE_PYTHON = True
                        x = groups.score_any(g_ref, a, plan, libs, prune=prune, crisp=crisp,
                                             crisp_alpha=alpha, floor=floor)
                        groups.PURE_PYTHON = False
                        y = groups.score_any(g_fast, a, plan, libs, prune=prune, crisp=crisp,
                                             crisp_alpha=alpha, floor=floor)
                        assert _key(x) == _key(y), (name, prune, floor, crisp, a)


def test_batch_matches_single(doc) -> None:
    libs = libs_from_doc(doc)
    g = TemporalGraph.from_transfers(_frame(_random_rows(13)))
    for name in AML:
        plan = compile_motif(doc, name)
        pos = np.arange(g.n_edges)
        top, found, exp, trunc = groups.score_group_batch(g, pos, plan, libs)
        for a in pos.tolist():
            r = groups.score_any(g, a, plan, libs)
            assert (top[a], found[a], exp[a], trunc[a]) == (r.top_score, r.found,
                                                             r.expansions, r.truncated)


def test_count_gate_shapes(doc) -> None:
    gate = groups_fast._count_gate
    assert gate(compile_motif(doc, "scatter_gather").having, "m") == 2
    assert gate(compile_motif(doc, "scatter_frac_gate").having, "m") == 2  # > 1.5
    assert gate(compile_motif(doc, "scatter_quant").having, "m") == 1
    assert gate(_plan(doc, "scatter_sum_having").having, "m") == 0  # not count-led
    assert gate(_SUM_GT0, "m") == 0
    assert gate(compile_motif(doc, "scatter_gather").having, "s") == 0  # other var
