"""Compile-service tests (`src/motifql/compile_service.py` + route).

The model transport is stubbed everywhere: success with witnesses on a
tiny planted cycle, pinned anchors, uncompilable output, the canned
fallback, and the error codes the route maps.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from motifql.compile_service import (  # noqa: E402
    Canned, CompileError, compile_english, load_canned)
from motifql.engine import Engine  # noqa: E402
from motifql.models.spark import SparkError  # noqa: E402
from motifql.server import UI_DIR, create_app  # noqa: E402

BASE = datetime(2022, 9, 3)
CAND = """term patient = trapezoid(0h, 0h, 24h, 120h)

motif cand {
  match (a:account) -[e:transfer*2..5]-> (a:account)
  where e is time_ordered and span(e) <= 7d
  soft each e: e.gap is patient and each e: e.amount ~= prev.amount within 20%
  score min
  emit when score >= 0.5
}
"""


def _engine() -> Engine:
    rows = [(0, "A", "B", 1000.0), (60, "B", "C", 990.0),
            (120, "C", "A", 980.0), (200, "A", "D", 10.0)]
    df = pl.DataFrame(
        {
            "transfer_id": list(range(len(rows))),
            "timestamp": [BASE + timedelta(minutes=r[0]) for r in rows],
            "from_bank": ["001"] * len(rows),
            "from_account": [r[1] for r in rows],
            "to_bank": ["001"] * len(rows),
            "to_account": [r[2] for r in rows],
            "amount_usd": [r[3] for r in rows],
        }
    )
    eng = Engine.from_motif_file(
        Path(__file__).resolve().parent.parent / "examples" / "amlworld.motif")
    for row in df.iter_rows(named=True):
        eng.score({"timestamp": row["timestamp"],
                   "from_bank": row["from_bank"],
                   "from_account": row["from_account"],
                   "to_bank": row["to_bank"], "to_account": row["to_account"],
                   "amount_usd": row["amount_usd"],
                   "transfer_id": row["transfer_id"]})
    return eng


def _ok(messages):
    return CAND, {}


def _cached(messages):
    return CAND, {"cached": True}


def _garbage(messages):
    return "motif cand { this is not motifql", {}


def _down(messages):
    raise SparkError("refused")


def test_success_with_sample_witnesses() -> None:
    res = compile_english("some cycle please", _engine(), complete=_ok)
    assert res.compiled is True and res.source == "model"
    assert "motif cand" in res.motif and res.attempts == 1
    assert len(res.witnesses) >= 1
    w = res.witnesses[0]
    assert w.top_score >= 0.5
    assert [h.transfer_id for h in w.witness] == [0, 1, 2]


def test_cached_source_label() -> None:
    res = compile_english("some cycle please", _engine(), complete=_cached)
    assert res.compiled is True and res.source == "cache"


def test_pinned_anchors() -> None:
    res = compile_english("cycle", _engine(), complete=_ok, anchor_tids=[2])
    assert len(res.witnesses) == 1
    assert res.witnesses[0].witness[-1].transfer_id == 2
    with pytest.raises(CompileError) as exc:
        compile_english("cycle", _engine(), complete=_ok, anchor_tids=[999])
    assert exc.value.status_code == 422


def test_uncompilable_returns_errors() -> None:
    res = compile_english("cycle", _engine(), complete=_garbage,
                          max_attempts=2)
    assert res.compiled is False and res.attempts == 2
    assert res.errors and res.witnesses == []


def test_canned_fallback_and_missing_key() -> None:
    canned = Canned(english="the scripted sentence", motif=CAND)
    res = compile_english("the scripted sentence", _engine(), complete=_down,
                          canned=canned)
    assert (res.compiled, res.source) == (True, "canned")
    assert res.witnesses[0].top_score >= 0.5
    with pytest.raises(CompileError) as exc:
        compile_english("anything else", _engine(), complete=_down,
                        canned=canned)
    assert exc.value.status_code == 502
    with pytest.raises(CompileError):
        compile_english("   ", _engine(), complete=_ok)


def test_committed_canned_pair_compiles_and_fires() -> None:
    canned = load_canned(UI_DIR / "canned.json")
    assert canned is not None and canned.english
    res = compile_english(canned.english, _engine(), complete=_down,
                          canned=canned)
    assert (res.compiled, res.source) == (True, "canned")
    assert res.witnesses[0].top_score >= 0.5


def test_pos_of() -> None:
    eng = _engine()
    g = eng.graph
    assert [g.pos_of(t) for t in (0, 1, 2, 3)] == [0, 1, 2, 3]
    assert g.pos_of(999) == -1
    # Counter-assigned appends keep the identity; explicit ones may break it.
    g2 = _engine().graph
    g2.append(10**9, "001", "X", "001", "Y", 1.0, 4)
    assert g2.pos_of(4) == 4
    g2.append(10**9 + 1, "001", "X", "001", "Y", 1.0, 100)
    assert g2.pos_of(100) == 5 and g2.pos_of(4) == 4


def test_route_and_index() -> None:
    app = create_app(_engine(), complete_fn=_ok,
                     canned_path=UI_DIR / "canned.json")
    client = TestClient(app)
    r = client.post("/compile", json={"english": "a cycle"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["compiled"] is True and body["witnesses"]
    assert body["witnesses"][0]["witness"][0]["transfer_id"] == 0
    r = client.post("/compile", json={"english": "a cycle",
                                      "anchor_tids": [999]})
    assert r.status_code == 422
    app2 = create_app(_engine(), complete_fn=_down,
                      canned_path="/nonexistent/canned.json")
    r = TestClient(app2).post("/compile", json={"english": "a cycle"})
    assert r.status_code == 502
    r = client.get("/demo-config")
    assert r.json()["anchor_tids"] == [2012559]
    r = client.get("/")
    assert r.status_code == 200 and "motif" in r.text.lower()
