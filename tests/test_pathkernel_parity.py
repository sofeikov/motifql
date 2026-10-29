"""Parity: numba path kernel vs the pure-Python reference walk.

The fast path (``motifql._pathkernel`` behind ``matcher.score``,
``match``, ``find_all`` and ``score_path_batch``) must return results
bit-identical to the pure-Python walk (``matcher.PURE_PYTHON = True``) in
every field — found, witness, score, top_score, degrees, truncated,
expansions — across prune / crisp / floor modes, and leave the lazy
eviction floors (``graph._in_lo``) in the same state.

Synthetic graphs cover same-timestamp ties, self-loops, hubs past the
fanout and expansion caps, empty windows and lazy eviction.
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
from motifql import matcher as M  # noqa: E402
from motifql.degrees import libs_from_doc  # noqa: E402
from motifql.graph import TemporalGraph  # noqa: E402
from motifql.parser import AmountConsCond, Dur, Money, Num, Pct, parse_text  # noqa: E402
from motifql.planner import PrunableGroup, compile_motif  # noqa: E402

BASE = datetime(2022, 9, 3)
TERMS = """term soon = trapezoid(0h, 0h, 6h, 48h)
term tri = triangle(0h, 2h, 30h)
term slow = ramp(1h, 20h)
term fast = ramp_down(30m, 10h)
term band(lo, hi) = trapezoid(lo, lo + 1h, hi, hi + 2h)
"""
SOFTS = {
    "gap_amt_min": ("each e: e.gap is soon and each e: e.amount ~= prev.amount within 20%", "min"),
    "gap_amt_prod": ("each e: e.gap is soon and each e: e.amount ~= prev.amount within 20%",
                     "product"),
    "two_atoms_prod": ("each e: (e.gap is tri and e.amount ~= prev.amount within 30%)",
                       "product"),
    "two_atoms_min": ("each e: (e.gap is tri and e.amount ~= prev.amount within 30%)", "min"),
    "negated_money": ("each e: e.gap is not fast and all e: e.amount ~= prev.amount within 50 USD",
                      "min"),
    "ramps": ("each e: e.gap is slow and each e: e.gap is fast", "product"),
    "param_term": ("each e: e.gap is band(1h, 12h)", "min"),
    "none": (None, "min"),
    "deferred": ("each e: e.gap is soon and some e: e.amount ~= prev.amount within 10%", "min"),
}


def _motif(soft: str | None, tnorm: str, closed: bool, span: str = "3d",
           hops: str = "2..5", emit: float = 0.4):
    if closed:
        pat, where = f"(a:account) -[e:transfer*{hops}]-> (a)", ""
    else:
        pat, where = f"(a:account) -[e:transfer*{hops}]-> (z:account)", " and a != z"
    text = (TERMS + f"motif t {{ match {pat} where e is time_ordered and span(e) <= {span}"
            + where + (f" soft {soft}" if soft else "")
            + f" score {tnorm} emit when score >= {emit} }}")
    doc = parse_text(text)
    return compile_motif(doc, "t"), libs_from_doc(doc)


def _frame(rows) -> pl.DataFrame:
    return pl.DataFrame({
        "transfer_id": list(range(len(rows))),
        "timestamp": [BASE + timedelta(minutes=int(r[0])) for r in rows],
        "from_bank": ["001"] * len(rows),
        "from_account": [f"N{r[1]}" for r in rows],
        "to_bank": ["001"] * len(rows),
        "to_account": [f"N{r[2]}" for r in rows],
        "amount_usd": [float(r[3]) for r in rows],
    })


def _random_rows(seed: int, n_nodes: int, n_edges: int, minutes: int,
                 hub: int | None = None, hub_in: int = 0):
    """Random transfers; coarse times force same-timestamp ties.

    Amounts come from a few round values plus noise so amount tolerances
    see exact ties, near-misses and zeros. Self-loops occur naturally
    (frm == to draws) and are also injected.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_edges):
        t = int(rng.integers(0, minutes // 30)) * 30  # half-hour grid: many ties
        a = int(rng.integers(0, n_nodes))
        b = int(rng.integers(0, n_nodes))
        amt = float(rng.choice([0.0, 100.0, 100.0, 110.0, 95.0, 1000.0]))
        if rng.random() < 0.5:
            amt += float(rng.integers(0, 30))
        rows.append((t, a, b, amt))
    for _ in range(n_edges // 20):  # explicit self-loops
        t = int(rng.integers(0, minutes))
        a = int(rng.integers(0, n_nodes))
        rows.append((t, a, a, 100.0))
    if hub is not None:
        for _ in range(hub_in):  # a hub with a huge in-degree in one window
            t = int(rng.integers(minutes // 2, minutes - 60))
            rows.append((t, int(rng.integers(0, n_nodes)), hub, 100.0))
        for i in range(12):  # late hub senders: their first lookup is capped
            rows.append((minutes - 1 - i, hub, int(rng.integers(1, n_nodes)), 100.0))
    rows.sort(key=lambda r: r[0])
    return rows


@pytest.fixture(scope="module")
def graphs():
    """(name, builder) pairs: builders give a fresh graph (lazy floors reset)."""
    specs = {
        "dense_ties": dict(seed=1, n_nodes=10, n_edges=200, minutes=4 * 24 * 60),
        "sparse": dict(seed=2, n_nodes=40, n_edges=300, minutes=6 * 24 * 60),
        "hub": dict(seed=3, n_nodes=25, n_edges=200, minutes=3 * 24 * 60, hub=0,
                    hub_in=1200),
    }
    frames = {k: _frame(_random_rows(**v)) for k, v in specs.items()}
    return {k: (lambda f=f: TemporalGraph.from_transfers(f)) for k, f in frames.items()}


MODES = [(prune, crisp, alpha, floor)
         for prune in (True, False)
         for crisp, alpha in ((False, 0.0), (True, 0.0), (True, "emit"))
         for floor in (None, M.EXACT_FLOOR)]


def _run(graph, plan, libs, anchors, mode, pure: bool):
    prune, crisp, alpha, floor = mode
    alpha = plan.emit if alpha == "emit" else alpha
    old = M.PURE_PYTHON
    M.PURE_PYTHON = pure
    try:
        return [M.score(graph, a, plan, libs, prune, crisp, alpha, floor) for a in anchors]
    finally:
        M.PURE_PYTHON = old


def _assert_same(fast, pure, ctx: str) -> None:
    assert len(fast) == len(pure)
    for i, (f, p) in enumerate(zip(fast, pure)):
        assert f == p, f"{ctx} anchor #{i}: fast {f} != pure {p}"
        # dataclass == compares floats by value; insist on the same bits too.
        assert repr(f.score) == repr(p.score) and repr(f.top_score) == repr(p.top_score)
        assert type(f.expansions) is type(p.expansions) and type(f.truncated) is bool


def _compare(build, plan, libs, anchors, mode, ctx, evict: int | None = None) -> None:
    g_fast, g_pure = build(), build()
    if evict is not None:
        g_fast.evict_before(evict)
        g_pure.evict_before(evict)
    fast = _run(g_fast, plan, libs, anchors, mode, pure=False)
    pure = _run(g_pure, plan, libs, anchors, mode, pure=True)
    _assert_same(fast, pure, ctx)
    assert np.array_equal(g_fast._in_lo, g_pure._in_lo), f"{ctx}: lazy floors differ"


@pytest.mark.parametrize("closed", [True, False])
@pytest.mark.parametrize("soft", sorted(SOFTS))
def test_score_parity_all_modes(graphs, soft, closed) -> None:
    text, tnorm = SOFTS[soft]
    plan, libs = _motif(text, tnorm, closed)
    for gname, build in graphs.items():
        n = build().n_edges
        step = 60 if soft == "deferred" else 5  # deferred: both sides run pure
        anchors = list(range(0, n, step)) + [n - 1]
        for mode in MODES:
            _compare(build, plan, libs, anchors, mode, f"{gname}/{soft}/{closed}/{mode}")


def test_fast_path_is_taken_for_supported_plans(graphs) -> None:
    plan, libs = _motif(*SOFTS["gap_amt_min"], closed=True)
    assert M._compiled_plan(plan, libs, False, 0.0) is not None
    assert M._graph_ok(graphs["dense_ties"]())
    deferred, dlibs = _motif(*SOFTS["deferred"], closed=True)
    assert M._compiled_plan(deferred, dlibs, False, 0.0) is None  # pure fallback


def test_open_chain_ranges_and_hand_built_tolerances(graphs) -> None:
    """Shapes the planner never emits but the kernel models: open chains
    searched past min_hops, and Num/Dur/Money/zero-width tolerances."""
    plan, libs = _motif(*SOFTS["gap_amt_min"], closed=False, hops="2..4")
    variants = [dataclasses.replace(plan, max_hops=4),
                dataclasses.replace(plan, min_hops=3, max_hops=5, window_s=1.5 * 86400 + 0.25)]
    for tol in (Num(25.0), Dur(30.0), Money(50.0, "USD"), Pct(0.0), Num(0.0)):
        groups = (PrunableGroup("amt", (AmountConsCond("e", tol),)),
                  PrunableGroup("neg", (AmountConsCond("e", Pct(0.1), negated=True),)))
        variants.append(dataclasses.replace(plan, max_hops=3, prunable=groups,
                                            score_tnorm="product", emit=0.2))
    for i, p in enumerate(variants):
        assert M._compiled_plan(p, libs, False, 0.0) is not None
        for gname in ("dense_ties", "sparse"):
            build = graphs[gname]
            anchors = list(range(0, build().n_edges, 2))
            for mode in MODES[::2] + MODES[1::5]:
                _compare(build, p, libs, anchors, mode, f"variant{i}/{gname}/{mode}")


def test_truncation_by_fanout_and_expansion_caps(graphs, monkeypatch) -> None:
    """Hub anchors hit the real fanout cap; shrunken caps hit both often."""
    build = graphs["hub"]
    g = build()
    hub_out = np.flatnonzero(g.frm == g.node_index["001/N0"])
    hub_in = np.flatnonzero(g.to == g.node_index["001/N0"])
    anchors = sorted(set(hub_out.tolist()[-40:] + hub_in.tolist()[-40:]))
    for closed in (True, False):
        plan, libs = _motif(*SOFTS["gap_amt_prod"], closed=closed)
        for mode in MODES:
            _compare(build, plan, libs, anchors, mode, f"hub/{closed}/{mode}")
        res = _run(build(), plan, libs, hub_out.tolist()[-40:], (False, False, 0.0, None), False)
        assert any(r.truncated for r in res), "hub never hit the fanout cap"
    monkeypatch.setattr(M, "DEFAULT_MAX_FANOUT", 4)
    monkeypatch.setattr(M, "DEFAULT_MAX_EXPANSIONS", 60)
    seen_trunc = False
    for gname in ("dense_ties", "hub"):
        b = graphs[gname]
        anchors = list(range(0, b().n_edges, 7))
        for closed in (True, False):
            plan, libs = _motif(*SOFTS["two_atoms_min"], closed=closed)
            for mode in MODES:
                _compare(b, plan, libs, anchors, mode, f"caps/{gname}/{closed}/{mode}")
            res = _run(b(), plan, libs, anchors, (False, False, 0.0, None), False)
            seen_trunc |= any(r.truncated and r.expansions == 60 for r in res)
    assert seen_trunc, "expansion cap never bit"


def test_empty_windows_and_self_loops(graphs) -> None:
    build = graphs["sparse"]
    g = build()
    loops = np.flatnonzero(g.frm == g.to).tolist()
    assert loops
    plan, libs = _motif(*SOFTS["gap_amt_min"], closed=True, span="1m")  # tiny window
    anchors = list(range(g.n_edges)) + loops
    for mode in MODES:
        _compare(build, plan, libs, anchors, mode, f"tiny-window/{mode}")
    res = _run(build(), plan, libs, loops, MODES[0], pure=False)
    assert all(r.expansions == 0 and not r.found for r in res)


def test_lazy_eviction_parity(graphs) -> None:
    for gname, build in graphs.items():
        g = build()
        mid = int(g.times[len(g.times) // 2])
        anchors = list(range(len(g.times) // 3, g.n_edges, 2))
        for closed in (True, False):
            plan, libs = _motif(*SOFTS["gap_amt_min"], closed=closed)
            for evict in (mid - 86400, mid, int(g.times[-1]) + 1):
                for mode in (MODES[0], MODES[7], MODES[-1]):
                    _compare(build, plan, libs, anchors, mode,
                             f"evict/{gname}/{closed}/{evict}/{mode}", evict=evict)


def _batch_rows(batch, r):
    return (float(batch.top_score[r]), bool(batch.found[r]), int(batch.expansions[r]),
            bool(batch.truncated[r]), float(batch.score[r]),
            tuple(batch.witness[r, :batch.witness_len[r]].tolist()))


@pytest.mark.parametrize("parallel", [False, True])
def test_batch_parity(graphs, parallel) -> None:
    for gname, build in graphs.items():
        n = build().n_edges
        for soft in ("gap_amt_min", "two_atoms_prod", "none", "deferred"):
            # deferred: per-anchor pure fallback on both sides, so sample thinly
            anchors = np.arange(0, n, 41 if soft == "deferred" else 4)
            for closed in (True, False):
                plan, libs = _motif(*SOFTS[soft], closed=closed)
                for mode in MODES if soft != "deferred" else MODES[::5]:
                    prune, crisp, alpha, floor = mode
                    alpha = plan.emit if alpha == "emit" else alpha
                    g_b, g_p = build(), build()
                    b = M.score_path_batch(g_b, anchors, plan, libs, prune, crisp, alpha,
                                           floor, parallel=parallel)
                    pure = _run(g_p, plan, libs, anchors.tolist(), mode, pure=True)
                    for r, p in enumerate(pure):
                        want = (p.top_score, p.found, p.expansions, p.truncated, p.score,
                                p.witness)
                        assert _batch_rows(b, r) == want, f"{gname}/{soft}/{closed}/{mode}/{r}"
                    if not parallel:
                        assert np.array_equal(g_b._in_lo, g_p._in_lo)


def test_parallel_batch_falls_back_after_eviction(graphs) -> None:
    build = graphs["dense_ties"]
    plan, libs = _motif(*SOFTS["gap_amt_min"], closed=True)
    g_b, g_p = build(), build()
    t = int(g_b.times[len(g_b.times) // 2])
    g_b.evict_before(t)
    g_p.evict_before(t)
    anchors = np.arange(len(g_b.times) // 2, g_b.n_edges)
    b = M.score_path_batch(g_b, anchors, plan, libs, True, parallel=True)
    pure = _run(g_p, plan, libs, anchors.tolist(), MODES[0], pure=True)
    for r, p in enumerate(pure):
        assert _batch_rows(b, r) == (p.top_score, p.found, p.expansions, p.truncated,
                                     p.score, p.witness)
    assert np.array_equal(g_b._in_lo, g_p._in_lo)  # serial kernel ran: same floors


@pytest.mark.parametrize("cap", [0, 1, 3, 1000])
def test_match_and_find_all_parity(graphs, cap) -> None:
    motifs = [M.ROUND_TRIP, M.LAYERING_CHAIN,
              dataclasses.replace(M.ROUND_TRIP, max_fanout=3, max_expansions=40),
              dataclasses.replace(M.LAYERING_CHAIN, min_hops=2, max_hops=4, max_fanout=2),
              dataclasses.replace(M.ROUND_TRIP, max_expansions=0)]
    for gname, build in graphs.items():
        g_f, g_p = build(), build()
        for motif in motifs:
            for a in range(0, g_f.n_edges, 3):
                old = M.PURE_PYTHON
                try:
                    M.PURE_PYTHON = False
                    mf, ff = M.match(g_f, a, motif), M.find_all(g_f, a, motif, cap=cap)
                    M.PURE_PYTHON = True
                    mp, fp = M.match(g_p, a, motif), M.find_all(g_p, a, motif, cap=cap)
                finally:
                    M.PURE_PYTHON = old
                assert mf == mp, f"match {gname}/{motif.name}/{a}"
                assert ff == fp, f"find_all {gname}/{motif.name}/{a}/{cap}"
        assert np.array_equal(g_f._in_lo, g_p._in_lo)


def test_numpy_int_anchor_keeps_its_type(graphs) -> None:
    """The witness ends with the caller's anchor object, as in the pure walk."""
    build = graphs["dense_ties"]
    plan, libs = _motif(*SOFTS["none"], closed=True)
    g = build()
    for a in np.arange(g.n_edges, dtype=np.int64)[::7]:
        f = _run(g, plan, libs, [a], MODES[0], pure=False)[0]
        p = _run(g, plan, libs, [a], MODES[0], pure=True)[0]
        assert f == p
        if f.found:
            assert type(f.witness[-1]) is type(p.witness[-1])
