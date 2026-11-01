"""Evasion-slider tests (`POST /perturb` + canned curve).

Pins the demo contract without data: level (0, x1) reproduces the
live score exactly on a tiny planted cycle, the exact arm uses
alpha = emit, below-emit explanations still describe the displayed
witness (fixed-witness degrees), and the committed canned curve pins
the scripted anchor (2012559, top 0.718...).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from motifql.compile_service import (  # noqa: E402
    EVASION_SEED, CompileError, load_canned, score_evasion)
from motifql.engine import Engine  # noqa: E402
from motifql.models.repair import try_compile  # noqa: E402
from motifql.server import DEMO_ANCHOR_TID, UI_DIR, create_app  # noqa: E402

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


def test_level_zero_reproduces_live_score() -> None:
    eng = _engine()
    compiled, errs = try_compile(CAND)
    assert compiled is not None, errs
    live = eng.score_with(eng.graph.pos_of(2), compiled.plan,
                          compiled.libs)
    assert [h.transfer_id for h in live.witness] == [0, 1, 2]
    emit, ws = score_evasion(CAND, eng, anchor_tids=[2])
    assert emit == 0.5
    (w,) = ws
    assert w.anchor_tid == 2
    assert w.fuzzy_top == live.top_score == w.baseline_fuzzy
    assert [h.transfer_id for h in w.witness] == [0, 1, 2]
    assert [h.amount for h in w.witness] == [h.amount for h in live.witness]
    # Exact arm at alpha = emit: fuzzy >= emit ⟺ crisp 1 on min motifs.
    assert w.crisp_alpha == 0.5 == emit
    assert w.crisp_top == 1.0 == w.baseline_crisp
    assert w.found is True


def test_noise_degrades_fuzzy_and_cliffs_exact() -> None:
    eng = _engine()
    _, (base,) = score_evasion(CAND, eng, anchor_tids=[2])
    _, (noisy,) = score_evasion(CAND, eng, anchor_tids=[2],
                                amount_noise_pct=0.2, seed=EVASION_SEED)
    assert noisy.fuzzy_top < base.fuzzy_top
    assert noisy.crisp_top in (0.0, 1.0)
    assert noisy.crisp_top == 0.0  # pinned draw cliffs at ±20%
    # Below emit the bars still describe the displayed witness.
    assert len(noisy.explanation) == 2
    assert noisy.found is False
    assert [h.transfer_id for h in noisy.witness] == [0, 1, 2]
    # Deterministic: same seed, same scores.
    _, (again,) = score_evasion(CAND, eng, anchor_tids=[2],
                                amount_noise_pct=0.2, seed=EVASION_SEED)
    assert again.fuzzy_top == noisy.fuzzy_top


def test_gap_scaling_moves_timestamps_order_preserved() -> None:
    eng = _engine()
    _, (w,) = score_evasion(CAND, eng, anchor_tids=[2], gap_scale=2.0)
    ts = [h.timestamp for h in w.witness]
    assert ts == sorted(ts)  # order preserved by construction
    assert ts[0] == int(BASE.replace(tzinfo=timezone.utc).timestamp())  # t_min fixed
    assert ts[1] - ts[0] == 2 * 3600 and ts[2] - ts[0] == 2 * 7200
    # Minute gaps stay fully patient at x2: fuzzy untouched.
    assert w.fuzzy_top == w.baseline_fuzzy


def test_evasion_errors_are_422() -> None:
    eng = _engine()
    for kw in ({"anchor_tids": [999]}, {"anchor_tids": []},
               {"anchor_tids": [2], "amount_noise_pct": -0.1},
               {"anchor_tids": [2], "amount_noise_pct": 1.5},
               {"anchor_tids": [2], "gap_scale": 0.0}):
        try:
            score_evasion(CAND, eng, **kw)
        except CompileError as exc:
            assert exc.status_code == 422
        else:
            raise AssertionError(f"no raise for {kw}")
    try:
        score_evasion("motif cand { this is not motifql", eng,
                      anchor_tids=[2])
    except CompileError as exc:
        assert exc.status_code == 422
    else:
        raise AssertionError("bad motif accepted")


def _ok(messages):
    return CAND, {}


def test_perturb_route_and_demo_config() -> None:
    app = create_app(_engine(), complete_fn=_ok,
                     canned_path=UI_DIR / "canned.json")
    client = TestClient(app)
    r = client.post("/perturb", json={
        "motif_text": CAND, "anchor_tids": [2],
        "amount_noise_pct": 0.15, "gap_scale": 1.5, "seed": EVASION_SEED})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["emit"] == 0.5 and body["seed"] == EVASION_SEED
    (w,) = body["witnesses"]
    assert w["anchor_tid"] == 2 and w["crisp_alpha"] == 0.5
    assert w["fuzzy_top"] < w["baseline_fuzzy"]
    assert len(w["witness"]) == 3 and len(w["explanation"]) == 2
    r = client.post("/perturb", json={"motif_text": CAND,
                                      "anchor_tids": [999]})
    assert r.status_code == 422
    r = client.post("/perturb", json={"motif_text": "nope",
                                      "anchor_tids": [2]})
    assert r.status_code == 422
    cfg = client.get("/demo-config").json()
    assert cfg["anchor_tids"] == [DEMO_ANCHOR_TID]
    assert cfg["evasion_curve"]["anchor_tid"] == DEMO_ANCHOR_TID


def test_canned_curve_pins_demo_contract() -> None:
    canned = load_canned(UI_DIR / "canned.json")
    assert canned is not None and canned.evasion is not None
    curve = canned.evasion
    assert curve["seed"] == EVASION_SEED
    assert curve["anchor_tid"] == DEMO_ANCHOR_TID
    assert curve["baseline"] == {"fuzzy_top": 0.7182153782559514,
                                 "crisp_top": 1.0}
    by_level = {(p["noise_pct"], p["gap"]): p for p in curve["points"]}
    zero = by_level[(0, 1.0)]
    assert zero["fuzzy_top"] == 0.7182153782559514
    assert zero["crisp_top"] == 1.0
    # The visceral shape: fuzzy sinks gradually, exact cliffs.
    assert by_level[(30, 1.0)]["fuzzy_top"] < 0.05
    assert by_level[(30, 1.0)]["crisp_top"] == 0.0
    assert by_level[(0, 2.0)]["fuzzy_top"] < zero["fuzzy_top"]
    assert len(curve["points"]) == 35
