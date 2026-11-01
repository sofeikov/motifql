"""Property tests for scored path search (`src/motifql/matcher.py`).

Soundness contract (pinned by example in `tests/test_scores.py`):
pruning may skip work but never changes the emit-filtered answer. These
run the same contract over generated transfer lists; when a mismatch
appears, Hypothesis shrinks it to a minimal failing case instead of a
hand-rolled 14-row blob.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from motifql.degrees import libs_from_doc  # noqa: E402
from motifql.graph import TemporalGraph  # noqa: E402
from motifql.matcher import EXACT_FLOOR, score  # noqa: E402
from motifql.parser import parse_text  # noqa: E402
from motifql.planner import compile_motif  # noqa: E402

BASE = datetime(2022, 9, 3)
HEADER = "term soon = trapezoid(0h, 0h, 6h, 48h)\n"
_ACCOUNTS = ["N0", "N1", "N2", "N3", "N4"]


def _frame(rows: list[tuple[int, str, str, str, str, float]]) -> pl.DataFrame:
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


def _round_trip():
    doc = parse_text(
        HEADER
        + "motif t { match (a:account) -[e:transfer*2..5]-> (a)"
        + " where e is time_ordered and span(e) <= 7d"
        + " soft each e: e.gap is soon"
        + " and each e: e.amount ~= prev.amount within 20%"
        + " score min emit when score >= 0.6 }"
    )
    return doc, compile_motif(doc, "t"), libs_from_doc(doc)


_DOC, _PLAN, _LIBS = _round_trip()


@st.composite
def _transfer_lists(draw):
    n = draw(st.integers(2, 10))
    times = sorted(draw(st.lists(st.integers(0, 300),
                                 min_size=n, max_size=n)))
    rows = []
    for t in times:
        rows.append((
            t, "001", draw(st.sampled_from(_ACCOUNTS)),
            "001", draw(st.sampled_from(_ACCOUNTS)),
            draw(st.floats(10.0, 1000.0,
                           allow_nan=False, allow_infinity=False)),
        ))
    return rows


def _check_witness_valid(g: TemporalGraph, witness: tuple[int, ...],
                         anchor: int) -> None:
    assert witness[-1] == anchor
    assert _PLAN.min_hops <= len(witness) <= _PLAN.max_hops
    ts = [int(g.times[e]) for e in witness]
    assert all(b > a for a, b in zip(ts, ts[1:]))
    assert ts[-1] - ts[0] <= _PLAN.window_s
    # Closed cycle: starts where the anchor lands, otherwise simple.
    seq = [int(g.frm[e]) for e in witness] + [int(g.to[witness[-1]])]
    assert seq[0] == seq[-1] == int(g.to[anchor])
    assert len(set(seq)) == len(seq) - 1


@given(_transfer_lists())
@settings(max_examples=60, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_prune_matches_unpruned(rows) -> None:
    g = TemporalGraph.from_transfers(_frame(rows))
    for anchor in range(g.n_edges):
        pruned = score(g, anchor, _PLAN, _LIBS, prune=True)
        raw = score(g, anchor, _PLAN, _LIBS, prune=False)
        assert (pruned.found, pruned.witness, pruned.score) == (
            raw.found, raw.witness, raw.score)
        assert pruned.top_score <= raw.top_score + 1e-9
        assert pruned.expansions <= raw.expansions
        if raw.top_score >= _PLAN.emit:
            assert pruned.top_score == pytest.approx(raw.top_score)
        if pruned.found:
            _check_witness_valid(g, pruned.witness, anchor)


@given(_transfer_lists())
@settings(max_examples=60, deadline=None,
          suppress_health_check=[HealthCheck.too_slow])
def test_exact_floor_matches_unpruned(rows) -> None:
    g = TemporalGraph.from_transfers(_frame(rows))
    for anchor in range(g.n_edges):
        exact = score(g, anchor, _PLAN, _LIBS, prune=True, floor=EXACT_FLOOR)
        raw = score(g, anchor, _PLAN, _LIBS, prune=False)
        assert (exact.found, exact.witness, exact.score) == (
            raw.found, raw.witness, raw.score)
        assert exact.top_score == pytest.approx(raw.top_score, abs=1e-9)
