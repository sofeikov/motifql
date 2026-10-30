"""Live engine tests (`src/motifql/engine.py`).

Pins the transfer contract (required fields, timestamp forms, transfer-id
assignment), witness serialization, and live-vs-replay parity: scoring
rows one by one must equal the batch replay over the same frame.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from motifql.engine import Engine  # noqa: E402

BASE = datetime(2022, 9, 3)
MOTIFS = Path(__file__).resolve().parent.parent / "examples" / "amlworld.motif"


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


def _cycle_frame() -> pl.DataFrame:
    return _frame([
        (0, "001", "A", "001", "B", 1000.0),
        (60, "001", "B", "001", "C", 990.0),
        (120, "001", "C", "001", "A", 980.0),
    ])


def _row_dict(row: dict) -> dict:
    return {
        "timestamp": row["timestamp"],
        "from_bank": row["from_bank"], "from_account": row["from_account"],
        "to_bank": row["to_bank"], "to_account": row["to_account"],
        "amount_usd": row["amount_usd"], "transfer_id": row["transfer_id"],
    }


def test_score_contract_and_witness() -> None:
    eng = Engine.from_motif_file(MOTIFS)
    assert eng.motif_names[:2] == ["round_trip", "layering_chain"]
    df = _cycle_frame()
    out = None
    for row in df.iter_rows(named=True):
        out = eng.score(_row_dict(row))
    assert out is not None
    assert [r.motif for r in out] == eng.motif_names
    rt = next(r for r in out if r.motif == "round_trip")
    assert rt.found and rt.score >= 0.5
    assert [h.transfer_id for h in rt.witness] == [0, 1, 2]
    assert rt.witness[0].from_account == "A"
    assert rt.witness[-1].to_account == "A"
    assert all(set(e) == {"predicate", "degree", "step"} for e in rt.explanation)
    assert all(0.0 <= e["degree"] <= 1.0 for e in rt.explanation)


def test_live_equals_replay() -> None:
    rng = np.random.default_rng(5)
    rows: list[tuple[int, str, str, str, str, float]] = []
    t = 0
    for _ in range(120):
        t += int(rng.integers(5, 180))
        a, b = rng.choice([f"A{i}" for i in range(10)], size=2, replace=False)
        rows.append((t, "001", a, "001", b, float(rng.integers(10, 3000))))
    rows += [(t + 10, "001", "X1", "001", "X2", 1000.0),
             (t + 20, "001", "X2", "001", "X3", 990.0),
             (t + 30, "001", "X3", "001", "X1", 980.0)]
    df = _frame(rows)
    eng = Engine.from_motif_file(MOTIFS)
    live = [[r.top_score for r in eng.score(_row_dict(row))]
            for row in df.iter_rows(named=True)]
    rep = eng.replay(df)
    assert rep["transfer_id"].to_list() == list(range(df.height))
    for j, name in enumerate(eng.motif_names):
        assert np.allclose(rep[name].to_numpy(),
                           np.array([row[j] for row in live])), name
    assert float(rep["round_trip"].max()) >= 0.5  # planted cycle fires


def test_timestamp_forms_and_tid_assignment() -> None:
    eng = Engine.from_motif_file(MOTIFS)
    base = {"from_bank": "001", "from_account": "A",
            "to_bank": "001", "to_account": "B", "amount_usd": 10.0}
    eng.score({**base, "timestamp": 1662566400})
    eng.score({**base, "timestamp": "2022-09-07T00:00:00"})
    eng.score({**base, "timestamp": datetime(2022, 9, 7, 1)})
    assert eng.graph.transfer_id.tolist() == [0, 1, 2]
    assert eng.graph.times.tolist() == [1662566400, 1662508800, 1662512400]
    # Explicit ids are honored and the counter continues past them.
    eng.score({**base, "timestamp": 1662566400, "transfer_id": 100})
    eng.score({**base, "timestamp": 1662566400})
    assert eng.graph.transfer_id.tolist()[-2:] == [100, 101]


def test_live_top_score_not_emit_gated() -> None:
    # Rules cut on the injected top_score, so path motifs must report
    # sub-emit scores live (default pruning gates them at emit).
    eng = Engine.from_motif_file(MOTIFS)
    hops = [(1000, 1000.0), (2000, 850.0), (3000, 850.0)]
    accts = [("A", "B"), ("B", "C"), ("C", "A")]
    out = None
    for (t, amt), (a, b) in zip(hops, accts):
        out = eng.score({"timestamp": t, "from_bank": "001",
                         "from_account": a, "to_bank": "001",
                         "to_account": b, "amount_usd": amt})
    rt = next(r for r in out if r.motif == "round_trip")
    assert rt.found is False
    assert 0.0 < rt.top_score < 0.5
    assert abs(rt.top_score - 0.25) < 1e-9
    assert rt.witness == []  # below-emit paths keep no witness


def test_aware_timestamps_normalize_to_utc() -> None:
    from datetime import timezone

    eng = Engine.from_motif_file(MOTIFS)
    base = {"from_bank": "001", "from_account": "A",
            "to_bank": "001", "to_account": "B", "amount_usd": 10.0}
    eng.score({**base, "timestamp": "2022-09-07T10:00:00+02:00"})
    eng.score({**base, "timestamp": datetime(2022, 9, 7, 10, 0,
                                             tzinfo=timezone.utc)})
    # 10:00+02:00 is 08:00Z; the aware datetime is 10:00Z.
    assert eng.graph.times.tolist() == [1662537600, 1662544800]


def test_retention_applies_live() -> None:
    eng = Engine.from_motif_file(MOTIFS)
    # Covers the longest motif window (layering_chain 10d) plus a margin.
    assert eng.graph.retention_s == 11 * 86400
    base = {"from_bank": "001", "from_account": "A",
            "to_bank": "001", "to_account": "B", "amount_usd": 10.0}
    t0 = 1662508800  # Sep 7 00:00Z
    eng.score({**base, "timestamp": t0})
    assert eng.graph.watermark == t0 - 11 * 86400
    eng.score({**base, "timestamp": t0 + 86400})
    assert eng.graph.watermark == t0 + 86400 - 11 * 86400
    # Backfilled older rows never move the watermark backwards.
    eng.score({**base, "timestamp": t0})
    assert eng.graph.watermark == t0 + 86400 - 11 * 86400


def test_bad_transfers_rejected() -> None:
    eng = Engine.from_motif_file(MOTIFS)
    with pytest.raises(ValueError, match="required fields"):
        eng.score({"timestamp": 1})
    with pytest.raises(ValueError, match="bad timestamp"):
        eng.score({"timestamp": True, "from_bank": "a", "from_account": "b",
                   "to_bank": "c", "to_account": "d", "amount_usd": 1.0})
    with pytest.raises(ValueError, match="bad amount"):
        eng.score({"timestamp": 1, "from_bank": "a", "from_account": "b",
                   "to_bank": "c", "to_account": "d", "amount_usd": "lots"})
