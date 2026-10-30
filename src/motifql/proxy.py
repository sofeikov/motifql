"""ezrules proxy core (DESIGN.md section 6.6).

The proxy scores the transfer inside an ezrules ``/evaluate`` payload,
adds ``event_data.motifs = {name: top_score}`` and forwards the payload
to a stock ezrules ``/api/v2/evaluate`` — no ezrules changes needed.
Rules read the scores as nested fields
(``if $motifs.round_trip > 0.8: return !HOLD``); ``top_score`` (not the
emit-gated ``score``) is injected so rules can pick their own cut, even
below the motif's emit threshold.

This module is plain functions over ``httpx`` (no FastAPI), so the
forwarding contract is testable without HTTP.
"""

from __future__ import annotations

import httpx

from motifql.engine import REQUIRED_FIELDS, Engine


class ProxyError(Exception):
    """Bad proxy payload (the caller's fault; maps to 422)."""


class UpstreamError(Exception):
    """The upstream was unreachable or errored; carries the motif scores."""

    def __init__(self, motifs: dict[str, float], detail: str) -> None:
        super().__init__(detail)
        self.motifs = motifs
        self.detail = detail


def inject_and_forward(payload: dict, engine: Engine, upstream_url: str,
                       client: httpx.Client) -> httpx.Response:
    """Score ``payload["event_data"]``, inject ``motifs``, forward it.

    Returns the upstream response verbatim. Raises ``ProxyError`` on a
    malformed payload or unscorable transfer, ``UpstreamError`` when the
    forward fails (the scores are on the exception, so the caller can
    still report what the engine saw).
    """
    if not isinstance(payload, dict) or not isinstance(
            payload.get("event_data"), dict):
        raise ProxyError("payload must be an object with an "
                         "'event_data' object")
    event = payload["event_data"]
    missing = [f for f in REQUIRED_FIELDS if f not in event]
    if missing:
        raise ProxyError(f"event_data misses required fields: {missing}")
    try:
        results = engine.score({k: event[k] for k in (*REQUIRED_FIELDS, "transfer_id")
                                if k in event})
    except ValueError as e:
        raise ProxyError(str(e))
    motifs = {r.motif: r.top_score for r in results}
    event["motifs"] = motifs
    try:
        return client.post(upstream_url, json=payload)
    except httpx.HTTPError as e:
        raise UpstreamError(motifs, f"{type(e).__name__}: {e}")
