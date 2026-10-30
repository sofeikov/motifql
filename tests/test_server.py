"""Service tests (`src/motifql/server.py` + `proxy.py`).

Pins the /score shape, the /evaluate inject-and-forward contract against
a stub upstream (payload forwarded with motifs added, nothing else
touched), and the upstream-down 502 that still reports the scores.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from motifql.engine import Engine  # noqa: E402
from motifql.server import create_app  # noqa: E402

MOTIFS = Path(__file__).resolve().parent.parent / "examples" / "amlworld.motif"

CYCLE = [
    {"timestamp": 1000, "from_bank": "001", "from_account": "A",
     "to_bank": "001", "to_account": "B", "amount_usd": 1000.0},
    {"timestamp": 2000, "from_bank": "001", "from_account": "B",
     "to_bank": "001", "to_account": "C", "amount_usd": 990.0},
    {"timestamp": 3000, "from_bank": "001", "from_account": "C",
     "to_bank": "001", "to_account": "A", "amount_usd": 980.0},
]


def _client(handler) -> tuple[TestClient, Engine]:
    engine = Engine.from_motif_file(MOTIFS)
    transport = httpx.MockTransport(handler)
    app = create_app(engine, "http://upstream/api/v2/evaluate",
                     httpx.Client(transport=transport))
    return TestClient(app), engine


def test_healthz() -> None:
    client, engine = _client(lambda request: httpx.Response(200, json={}))
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "motifs": engine.motif_names, "edges": 0}


def test_score_cycle() -> None:
    client, _ = _client(lambda request: httpx.Response(200, json={}))
    out = None
    for hop in CYCLE:
        r = client.post("/score", json=hop)
        assert r.status_code == 200, r.text
        out = r.json()
    by_name = {x["motif"]: x for x in out["results"]}
    assert by_name["round_trip"]["found"] is True
    assert by_name["round_trip"]["score"] >= 0.5
    assert [h["transfer_id"] for h in by_name["round_trip"]["witness"]] == [0, 1, 2]
    r = client.get("/healthz")
    assert r.json()["edges"] == 3


def test_score_rejects_bad_transfer() -> None:
    client, _ = _client(lambda request: httpx.Response(200, json={}))
    r = client.post("/score", json={"timestamp": 1})
    assert r.status_code == 422


def test_evaluate_injects_and_forwards() -> None:
    seen: dict = {}

    def stub(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"decision": "HOLD"})

    client, _ = _client(stub)
    for hop in CYCLE[:2]:
        client.post("/score", json=hop)
    payload = {"event_id": "e1", "event_data": dict(CYCLE[2])}
    r = client.post("/evaluate", json=payload)
    assert r.status_code == 200
    assert r.json() == {"decision": "HOLD"}
    assert seen["url"] == "http://upstream/api/v2/evaluate"
    forwarded = seen["body"]
    assert forwarded["event_id"] == "e1"  # untouched envelope
    assert {k: v for k, v in forwarded["event_data"].items()
            if k != "motifs"} == CYCLE[2]  # transfer untouched
    assert forwarded["event_data"]["motifs"]["round_trip"] >= 0.5


def test_evaluate_upstream_down_reports_scores() -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    client, _ = _client(down)
    for hop in CYCLE[:2]:
        client.post("/score", json=hop)
    r = client.post("/evaluate", json={"event_data": dict(CYCLE[2])})
    assert r.status_code == 502
    body = r.json()
    assert body["motifs"]["round_trip"] >= 0.5
    assert "ConnectError" in body["upstream_error"]


def test_evaluate_rejects_bad_payload() -> None:
    client, _ = _client(lambda request: httpx.Response(200, json={}))
    assert client.post("/evaluate", json={}).status_code == 422
    assert client.post("/evaluate", json={"event_data": {"a": 1}}).status_code == 422
