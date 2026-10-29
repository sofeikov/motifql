"""HTTP service: live scoring plus the ezrules proxy (DESIGN.md 6.6).

* ``POST /score`` — one transfer in, per-motif results out.
* ``POST /evaluate`` — ezrules-shaped payload in; the transfer inside
  ``event_data`` is scored, ``event_data.motifs`` injected, and the
  payload forwarded to the upstream (stock ezrules ``/api/v2/evaluate``).
* ``GET /healthz`` — motifs loaded and edges held.

Run: ``uv run uvicorn motifql.server:app --port 8000`` (after setting the
``MOTIFQL_MOTIFS`` / ``MOTIFQL_DATA`` / ``MOTIFQL_UPSTREAM`` env vars), or
``uv run python -m motifql.server --motifs examples/amlworld.motif``.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from motifql.engine import Engine, MotifResult, WitnessHop  # noqa: E402
from motifql.proxy import ProxyError, UpstreamError, inject_and_forward  # noqa: E402

DEFAULT_UPSTREAM = "http://localhost:8080/api/v2/evaluate"


class TransferIn(BaseModel):
    timestamp: int | float | str | datetime
    from_bank: str
    from_account: str
    to_bank: str
    to_account: str
    amount_usd: float
    transfer_id: int | None = None


class HopOut(BaseModel):
    transfer_id: int
    timestamp: int
    from_bank: str
    from_account: str
    to_bank: str
    to_account: str
    amount: float


class ResultOut(BaseModel):
    motif: str
    found: bool
    score: float
    top_score: float
    witness: list[HopOut] = Field(default_factory=list)
    explanation: list[dict] = Field(default_factory=list)
    truncated: bool = False
    expansions: int = 0


def _hop_out(h: WitnessHop) -> HopOut:
    return HopOut(transfer_id=h.transfer_id, timestamp=h.timestamp,
                  from_bank=h.from_bank, from_account=h.from_account,
                  to_bank=h.to_bank, to_account=h.to_account, amount=h.amount)


def _result_out(r: MotifResult) -> ResultOut:
    return ResultOut(motif=r.motif, found=r.found, score=r.score,
                     top_score=r.top_score,
                     witness=[_hop_out(h) for h in r.witness],
                     explanation=r.explanation, truncated=r.truncated,
                     expansions=r.expansions)


def create_app(engine: Engine, upstream_url: str = DEFAULT_UPSTREAM,
               client: httpx.Client | None = None) -> FastAPI:
    """Build the app over a ready engine (tests inject a stub client)."""
    app = FastAPI(title="MotifQL live scoring + ezrules proxy")
    app.state.engine = engine
    app.state.upstream_url = upstream_url
    app.state.client = client or httpx.Client(timeout=30.0)

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True, "motifs": engine.motif_names,
                "edges": engine.graph.n_edges}

    @app.post("/score")
    def score(transfer: TransferIn) -> dict:
        try:
            results = engine.score(transfer.model_dump())
        except ValueError as e:
            return JSONResponse({"detail": str(e)}, status_code=422)
        return {"results": [r.model_dump() for r in
                            (_result_out(x) for x in results)]}

    @app.post("/evaluate")
    def evaluate(payload: dict) -> JSONResponse:
        try:
            upstream = inject_and_forward(payload, engine, upstream_url,
                                          app.state.client)
        except ProxyError as e:
            return JSONResponse({"detail": str(e)}, status_code=422)
        except UpstreamError as e:
            return JSONResponse({"motifs": e.motifs,
                                 "upstream_error": e.detail}, status_code=502)
        try:
            body = upstream.json()
        except ValueError:
            body = {"upstream_body": upstream.text}
        return JSONResponse(body, status_code=upstream.status_code)

    return app


def _default_app() -> FastAPI:
    motifs = os.environ.get("MOTIFQL_MOTIFS", "examples/amlworld.motif")
    data = os.environ.get("MOTIFQL_DATA")
    upstream = os.environ.get("MOTIFQL_UPSTREAM", DEFAULT_UPSTREAM)
    engine = Engine.from_motif_file(motifs, transfers=data or None)
    return create_app(engine, upstream)


# Import-time app for `uvicorn motifql.server:app`.
app = _default_app()


def main(
    motifs: Annotated[str, typer.Option(help="Motif file.")] = "examples/amlworld.motif",
    data: Annotated[str | None, typer.Option(help="Warmup data dir or transfers parquet.")] = None,
    upstream: Annotated[str, typer.Option(help="Upstream /evaluate URL.")] = DEFAULT_UPSTREAM,
    host: Annotated[str, typer.Option(help="Bind host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Bind port.")] = 8000,
) -> None:
    """Serve live scoring plus the proxy (blocking)."""
    import uvicorn

    engine = Engine.from_motif_file(motifs, transfers=data or None)
    uvicorn.run(create_app(engine, upstream), host=host, port=port)


if __name__ == "__main__":
    typer.run(main)
