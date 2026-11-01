"""Lipschitz constants behind the issue-25 evasion-robustness bound.

Each test pins one constant the bound in
`docs/2026-11-09-issue25-robustness.md` cites: the leaf slopes, the
composition lemmas, and the adversarial amount-cost `p* = f/(2-f)`.
Where a constant is exact (not just an upper bound), a tightness case
proves the bound is not vacuous. The discontinuities (degenerate
trapezoid shoulders, `%` tolerance at the origin) are pinned as the
documented domain boundaries, not bugs.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from motifql import degrees as D  # noqa: E402
from motifql.degrees import libs_from_doc  # noqa: E402
from motifql.graph import TemporalGraph  # noqa: E402
from motifql.matcher import score  # noqa: E402
from motifql.parser import Num, Pct, parse_text  # noqa: E402
from motifql.planner import compile_motif  # noqa: E402

BASE = datetime(2022, 9, 3)
PATIENT = "term patient = trapezoid(0h, 0h, 24h, 120h)\n"
# Falling slope of `patient`: 1/(120h - 24h), per second.
PATIENT_L = 1.0 / (96 * 3600)


def _trap_l(a: float, b: float, c: float, d: float) -> float:
    """Lipschitz constant of a non-degenerate trapezoid on its slopes."""
    return max(1.0 / (b - a), 1.0 / (d - c))


def test_trapezoid_lipschitz() -> None:
    rng = np.random.default_rng(25)
    shapes = [
        # (args, sample_lo, sample_hi, L): degenerate shoulders are only
        # sampled on their continuous side (gaps/spans are >= 0).
        ((0.0, 0.0, 86400.0, 432000.0), 0.0, 600000.0, PATIENT_L),  # patient
        ((0.0, 0.0, 21600.0, 172800.0), 0.0, 250000.0, 1.0 / 151200),  # soon
        ((8500.0, 9500.0, 9990.0, 10000.0), 8000.0, 10500.0, 1.0 / 10),  # just_below
        ((2.0, 4.0, 6.0, 10.0), -5.0, 15.0, _trap_l(2.0, 4.0, 6.0, 10.0)),
    ]
    for args, lo, hi, lips in shapes:
        xs = rng.uniform(lo, hi, 500)
        ys = rng.uniform(lo, hi, 500)
        for x, y in zip(xs, ys):
            assert abs(D.term_degree("trapezoid", args, x)
                       - D.term_degree("trapezoid", args, y)) <= lips * abs(x - y) + 1e-9


def test_trapezoid_tight_on_slopes() -> None:
    # `patient` falls from 1.0 at 24h to 0.0 at 120h: the slope attains L.
    assert D.term_degree("trapezoid", (0.0, 0.0, 86400.0, 432000.0), 86400.0) == 1.0
    assert D.term_degree("trapezoid", (0.0, 0.0, 86400.0, 432000.0), 432000.0) == 0.0
    assert 1.0 / (432000.0 - 86400.0) == PATIENT_L


def test_degenerate_shoulder_jumps() -> None:
    # A zero-width shoulder is a step, not a slope: trapezoid(2,2,4,8)
    # jumps 0 -> 1 at x = 2. The bound therefore restricts degenerate
    # terms to their continuous side (gap/span/count inputs are >= 0,
    # and `patient`-style shoulders sit at 0).
    assert D.term_degree("trapezoid", (2.0, 2.0, 4.0, 8.0), 1.99) == 0.0
    assert D.term_degree("trapezoid", (2.0, 2.0, 4.0, 8.0), 2.0) == 1.0


def test_ramp_triangle_lipschitz_and_tight() -> None:
    rng = np.random.default_rng(26)
    for shape, args, lips in [
        ("ramp", (1.0, 3.0), 0.5),          # several
        ("ramp", (3.0, 8.0), 0.2),          # many
        ("ramp", (0.5, 0.8), 1.0 / 0.3),    # most
        ("ramp_down", (0.1, 0.3), 5.0),     # few
        ("triangle", (0.0, 5.0, 15.0), 0.2),
    ]:
        lo, hi = (args[0] - 2.0, args[-1] + 2.0)
        for x, y in zip(rng.uniform(lo, hi, 300), rng.uniform(lo, hi, 300)):
            assert abs(D.term_degree(shape, args, x)
                       - D.term_degree(shape, args, y)) <= lips * abs(x - y) + 1e-9
        # Tight: ramps attain their slope end to end; the triangle
        # attains its rising slope from foot to peak (both feet are 0).
        if shape == "triangle":
            a, b, c = args
            assert D.term_degree(shape, args, b) - D.term_degree(shape, args, a) == 1.0
            assert 1.0 / (b - a) == pytest.approx(lips)
        else:
            assert abs(D.term_degree(shape, args, args[-1])
                       - D.term_degree(shape, args, args[0])) == pytest.approx(
                           lips * (args[-1] - args[0]))


def test_tol_absolute_lipschitz() -> None:
    # |f(x1,y1) - f(x2,y2)| <= (|dx| + |dy|) / width.
    rng = np.random.default_rng(27)
    for width in (500.0, 3600.0):
        tol = Num(width)
        for _ in range(500):
            x1, y1 = rng.uniform(-1000, 10000, 2)
            x2, y2 = rng.uniform(-1000, 10000, 2)
            assert abs(D.tol_degree(tol, x1, y1) - D.tol_degree(tol, x2, y2)) <= (
                abs(x1 - x2) + abs(y1 - y2)) / width + 1e-9


def test_tol_pct_log_lipschitz() -> None:
    # On amounts >= m > 0: |Df| <= (|Dlog x| + |Dlog y|) / f. The ratio
    # form is 1-Lipschitz in log-space, so a relative perturbation costs
    # what it should instead of scaling with the amount level.
    rng = np.random.default_rng(28)
    for f in (0.2, 0.25):
        tol = Pct(f)
        lx = rng.uniform(0.0, math.log(1e6), 1000)
        for lx1, ly1, lx2, ly2 in zip(lx[::4], lx[1::4], lx[2::4], lx[3::4]):
            x1, y1, x2, y2 = (math.exp(v) for v in (lx1, ly1, lx2, ly2))
            assert abs(D.tol_degree(tol, x1, y1) - D.tol_degree(tol, x2, y2)) <= (
                abs(lx1 - lx2) + abs(ly1 - ly2)) / f + 1e-9


def test_tol_pct_tight_for_anticorrelated_noise() -> None:
    # Worst case: the two sides move against each other, x = A(1+p),
    # y = A(1-p). At p = 0.05, f = 0.2 the drop attains 95% of the
    # log-bound, so the constant has no slack to hide in.
    f, p, big_a = 0.2, 0.05, 1000.0
    got = D.tol_degree(Pct(f), big_a, big_a) - D.tol_degree(
        Pct(f), big_a * (1 + p), big_a * (1 - p))
    log_bound = (math.log(1 + p) - math.log(1 - p)) / f
    assert got / log_bound > 0.9


def test_tol_pct_origin_is_singular() -> None:
    # (0, 0) scores 1 while (eps, 0) scores 0 for any eps > 0 when
    # f < 1: the `%` form is discontinuous at the origin, which is why
    # the bound needs amounts >= m > 0 (true of money transfers).
    assert D.tol_degree(Pct(0.2), 0.0, 0.0) == 1.0
    assert D.tol_degree(Pct(0.2), 1e-9, 0.0) == 0.0


def test_order_statistics_are_sup_lipschitz() -> None:
    # each/all (min), some (max), at least N (N-th largest) and
    # all but K ((K+1)-th smallest, the engine's sorted()[k] form) move
    # at most as far as the furthest-moving binding.
    rng = np.random.default_rng(29)
    for _ in range(500):
        n = int(rng.integers(2, 9))
        v = rng.uniform(0, 1, n)
        w = rng.uniform(0, 1, n)
        sup = float(np.abs(v - w).max())
        assert abs(float(v.min()) - float(w.min())) <= sup + 1e-9
        assert abs(float(v.max()) - float(w.max())) <= sup + 1e-9
        for rank in range(n):
            assert abs(sorted(v, reverse=True)[rank]
                       - sorted(w, reverse=True)[rank]) <= sup + 1e-9
            assert abs(sorted(v)[rank] - sorted(w)[rank]) <= sup + 1e-9


def test_product_and_mean_ramps_compose() -> None:
    # product over [0, 1]^n: |DP| <= sum |Dd_i| (telescoping, each
    # partial product <= 1). most/few/@name ramp over the mean:
    # |Dr| <= sum |Dd_i| / (n * width).
    rng = np.random.default_rng(30)
    for _ in range(300):
        n = int(rng.integers(1, 7))
        v = rng.uniform(0, 1, n)
        w = rng.uniform(0, 1, n)
        ell1 = float(np.abs(v - w).sum())
        assert abs(float(np.prod(v)) - float(np.prod(w))) <= ell1 + 1e-9
        for width in (0.3, 0.2):  # most, few
            lo = 0.5 if width == 0.3 else 0.1
            shape = "ramp" if width == 0.3 else "ramp_down"
            assert abs(D.term_degree(shape, (lo, lo + width), float(v.mean()))
                       - D.term_degree(shape, (lo, lo + width), float(w.mean()))
                       ) <= ell1 / (n * width) + 1e-9
    # product is tight to first order: (1,1,1) vs (1-eps)^3 drops ~3 eps.
    eps = 1e-4
    assert abs((1.0 - (1.0 - eps) ** 3) - 3 * eps) < 3 * eps * eps + 1e-9


def test_adversarial_amount_cost() -> None:
    # An evader anti-correlating two sides at +/-p zeroes a `%` leaf
    # exactly at p* = f/(2-f): 11.1% for within 20% (round_trip,
    # scatter_gather sums), 14.3% for within 25% (layering_chain).
    # Correlated noise (both sides +p) never hurts: the leaf reads the
    # ratio, which is why random noise is gentler than evasion.
    for f, p_star in ((0.2, 0.2 / 1.8), (0.25, 0.25 / 1.75)):
        tol = Pct(f)
        big_a = 1000.0
        assert D.tol_degree(tol, big_a * (1 + p_star),
                            big_a * (1 - p_star)) == pytest.approx(0.0, abs=1e-9)
        assert D.tol_degree(tol, big_a * (1 + 0.99 * p_star),
                            big_a * (1 - 0.99 * p_star)) > 0.0
        assert D.tol_degree(tol, big_a * 1.5, big_a * 1.5) == 1.0


def _frame(rows: list[tuple[int, str, str, str, str, float]]) -> pl.DataFrame:
    return pl.DataFrame({
        "transfer_id": list(range(len(rows))),
        "timestamp": [BASE + timedelta(minutes=r[0]) for r in rows],
        "from_bank": [r[1] for r in rows],
        "from_account": [r[2] for r in rows],
        "to_bank": [r[3] for r in rows],
        "to_account": [r[4] for r in rows],
        "amount_usd": [r[5] for r in rows],
    })


def test_amount_only_end_to_end_bound() -> None:
    # Amount-only noise leaves every hard cut untouched (the RQ2 motifs'
    # where/having clauses are time- and topology-only), so the witness
    # set is fixed and the per-anchor top_score inherits the per-leaf
    # log-bound. Worst-case alternation (1+p),(1-p),(1+p) attains the
    # predicted drop exactly: the bound is tight end to end, not just
    # per leaf.
    p, f, big_a = 0.02, 0.2, 100.0
    doc = parse_text(
        PATIENT
        + "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
        + " where e is time_ordered and span(e) <= 7d"
        + " soft each e: e.gap is patient"
        + " and each e: e.amount ~= prev.amount within 20%"
        + " score min emit when score >= 0.5 }")
    plan, libs = compile_motif(doc, "t"), libs_from_doc(doc)
    kw = dict(prune=False)
    clean = TemporalGraph.from_transfers(_frame([
        (0, "001", "A", "002", "B", big_a),
        (10, "002", "B", "003", "C", big_a),
        (20, "003", "C", "001", "A", big_a),
    ]))
    noisy = TemporalGraph.from_transfers(_frame([
        (0, "001", "A", "002", "B", big_a * (1 + p)),
        (10, "002", "B", "003", "C", big_a * (1 - p)),
        (20, "003", "C", "001", "A", big_a * (1 + p)),
    ]))
    s0 = score(clean, 2, plan, libs, **kw).top_score
    s1 = score(noisy, 2, plan, libs, **kw).top_score
    assert s0 == 1.0
    lam = abs(math.log(1 - p))  # max(|log(1+p)|, |log(1-p)|)
    assert s0 - s1 <= 2 * lam / f + 1e-9
    # Exact prediction: both amount leaves read h = 2p/(1+p); gap
    # leaves stay 1.0 and the single witness takes the min.
    assert s0 - s1 == pytest.approx(2 * p / ((1 + p) * f))
