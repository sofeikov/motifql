"""Live scoring engine (DESIGN.md section 6.6).

``Engine`` compiles a ``.motif`` file once, then scores each transfer as it
arrives: the transfer is appended to the temporal graph and becomes the
anchor for every motif. ``replay`` runs the same plans over a frame in one
batch pass; both paths share ``groups.score_any``, so live and batch agree
by construction.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from motifql.degrees import libs_from_doc
from motifql.graph import SECS_PER_DAY, TemporalGraph
from motifql.groups import score_any
from motifql.matcher import EXACT_FLOOR
from motifql.parser import parse_file, parse_text

REQUIRED_FIELDS = ("timestamp", "from_bank", "from_account",
                   "to_bank", "to_account", "amount_usd")


@dataclass
class WitnessHop:
    transfer_id: int
    timestamp: int
    from_bank: str
    from_account: str
    to_bank: str
    to_account: str
    amount: float


@dataclass
class MotifResult:
    motif: str
    found: bool
    score: float
    top_score: float
    witness: list[WitnessHop] = field(default_factory=list)
    explanation: list[dict] = field(default_factory=list)
    truncated: bool = False
    expansions: int = 0


def _epoch(ts: int | float | str | datetime) -> int:
    """Epoch seconds; naive datetimes are UTC by prep convention."""
    if isinstance(ts, bool):
        raise ValueError(f"bad timestamp: {ts!r}")
    if isinstance(ts, (int, float)):
        return int(ts)
    if isinstance(ts, datetime):
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc)
        return int((ts.replace(tzinfo=None) - datetime(1970, 1, 1)).total_seconds())
    if isinstance(ts, str):
        return _epoch(datetime.fromisoformat(ts.replace("Z", "+00:00")))
    raise ValueError(f"bad timestamp: {ts!r}")


class Engine:
    """Compiled motifs over a streaming temporal graph (thread-safe)."""

    def __init__(self, motif_text: str,
                 transfers: pl.DataFrame | Path | str | None = None,
                 retention_days: int = 8) -> None:
        from motifql.planner import compile_motif

        doc = parse_text(motif_text)
        libs = libs_from_doc(doc)
        self._libs = libs
        self._plans = [(m.name, compile_motif(doc, m.name)) for m in doc.motifs]
        if isinstance(transfers, (Path, str)):
            transfers = pl.read_parquet(Path(transfers) / "transfers.parquet"
                                        if Path(transfers).is_dir()
                                        else transfers)
        # Retention must cover the longest motif window (plus a margin),
        # or advancing the watermark would eat witnesses.
        need_days = max([retention_days]
                        + [math.ceil(p.window_s / SECS_PER_DAY) + 1
                           for _, p in self._plans])
        self.graph = (TemporalGraph.from_transfers(
            transfers, retention_days=need_days)
            if transfers is not None
            else TemporalGraph.empty(retention_days=need_days))
        tids = self.graph.transfer_id
        self._next_tid = int(tids.max()) + 1 if len(tids) else 0
        self._lock = threading.Lock()

    @classmethod
    def from_motif_file(cls, path: str | Path,
                        transfers: pl.DataFrame | Path | str | None = None,
                        retention_days: int = 8) -> Engine:
        doc_path = Path(path)
        return cls(doc_path.read_text(), transfers, retention_days)

    @property
    def motif_names(self) -> list[str]:
        return [name for name, _ in self._plans]

    def score(self, transfer: dict) -> list[MotifResult]:
        """Append one transfer and score it against every motif.

        ``transfer`` holds the prep-schema fields (``timestamp`` as epoch
        seconds, ISO-8601 or datetime; ``transfer_id`` optional). Returns
        one result per motif in file order. Runs the pure-Python
        reference while the append sits in the overflow (the compiled
        kernels read CSR segments only); ``rebuild()`` reopens them.
        """
        missing = [f for f in REQUIRED_FIELDS if f not in transfer]
        if missing:
            raise ValueError(f"transfer misses required fields: {missing}")
        t = _epoch(transfer["timestamp"])
        try:
            amount = float(transfer["amount_usd"])
        except (TypeError, ValueError):
            raise ValueError(f"bad amount_usd: {transfer['amount_usd']!r}")
        tid = transfer.get("transfer_id")
        with self._lock:
            if tid is None:
                tid = self._next_tid
                self._next_tid += 1
            else:
                tid = int(tid)
                self._next_tid = max(self._next_tid, tid + 1)
            pos = self.graph.append(
                t, str(transfer["from_bank"]), str(transfer["from_account"]),
                str(transfer["to_bank"]), str(transfer["to_account"]),
                amount, tid)
            # Retention applies as transfers arrive; backfilled older rows
            # never move the watermark backwards. Rebuilds compact the
            # evicted edges, so long-running servers hold ~retention only.
            self.graph.evict_before(t - self.graph.retention_s)
            return [self._score_pos(pos, name, plan) for name, plan in self._plans]

    def score_with(self, pos: int, plan, libs, name: str = "cand") -> MotifResult:
        """Score one position against an ad-hoc plan (compile-then-show).

        For candidates that are not among the engine's own motifs; the
        result shape matches `score` exactly.
        """
        with self._lock:
            return self._score_pos(pos, name, plan, libs)

    def _score_pos(self, pos: int, name: str, plan, libs=None) -> MotifResult:
        # Exact floor: default pruning gates path top_scores at emit,
        # which would hide sub-emit scores from downstream rules.
        res = score_any(self.graph, pos, plan,
                        self._libs if libs is None else libs,
                        floor=EXACT_FLOOR)
        return MotifResult(
            motif=name, found=res.found, score=res.score,
            top_score=res.top_score,
            witness=[self._hop(e) for e in res.witness],
            explanation=[{"predicate": label, "degree": d, "step": s}
                         for label, d, s in res.degrees],
            truncated=res.truncated, expansions=res.expansions)

    def _hop(self, pos: int) -> WitnessHop:
        g = self.graph
        fb, fa = g.node_key(int(g.frm[pos])).split("/", 1)
        tb, ta = g.node_key(int(g.to[pos])).split("/", 1)
        return WitnessHop(
            transfer_id=int(g.transfer_id[pos]), timestamp=int(g.times[pos]),
            from_bank=fb, from_account=fa, to_bank=tb, to_account=ta,
            amount=float(g.amount[pos]))

    def replay(self, df: pl.DataFrame) -> pl.DataFrame:
        """Score every row of ``df`` in batch; returns per-anchor top_scores.

        Builds a fresh graph from the frame (queries only look back, so
        later rows are invisible to earlier anchors — the rq1 protocol).
        """
        graph = TemporalGraph.from_transfers(df)
        cols: dict[str, list] = {"transfer_id": graph.transfer_id.tolist()}
        for name, plan in self._plans:
            cols[name] = [score_any(graph, pos, plan, self._libs,
                                    floor=EXACT_FLOOR).top_score
                          for pos in range(graph.n_edges)]
        return pl.DataFrame(cols)


def load_motif_file(path: str | Path) -> str:
    """Read a ``.motif`` file (kept beside the engine for CLI/server use)."""
    return Path(path).read_text()


# Re-exported for callers that compile without an engine (tests, rq3).
__all__ = ["Engine", "MotifResult", "WitnessHop", "REQUIRED_FIELDS",
           "load_motif_file", "parse_file"]
