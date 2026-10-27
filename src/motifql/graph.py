"""Temporal graph store for the MotifQL engine (DESIGN.md section 6.2).

The store holds every transfer in flat time-sorted arrays plus per-account
incoming/outgoing adjacency in CSR form (one contiguous segment per account,
each segment time-ordered). A matcher query such as "incoming transfers of
account X in [T-D, T)" is then two binary searches plus a backwards walk --
no hash joins and no adjacency matrices at query time.

Layout
------
* Global edge arrays, all in timestamp order: ``times`` (epoch seconds),
  ``frm`` / ``to`` (int32 node ids), ``amount`` (USD, float32),
  ``transfer_id`` (original prep row number, for witnesses and labels).
* Accounts are interned to int32 ids from the ``"bank/account"`` composite
  key (banks and accounts stay strings upstream -- leading zeros matter --
  and the composite keeps per-bank account blocks distinct). ``nodes[id]``
  maps back for witness output.
* ``out_ptr`` / ``in_ptr`` delimit each node's segment in ``out_edges`` /
  ``in_edges`` (edge positions, uint32). ``out_times`` / ``in_times``
  duplicate the timestamps in adjacency order so every window lookup is a
  plain ``numpy.searchsorted`` on a contiguous view (one extra int64 per
  edge per direction; revisited only if HI-Medium forces it).

Retention
---------
Queries are always window-bounded, so retention is a watermark plus lazy
per-node floor offsets: ``evict_before(t)`` only moves the watermark
(O(1)); each query rebases the touched node's floor with one binary
search (O(log degree)). Memory of evicted edges is not freed until a
rebuild -- documented, not silent.

Time semantics
--------------
Edges strictly before the anchor count (``searchsorted(..., side="left")``),
matching the ``time_ordered`` crisp predicate: with minute-resolution
timestamps, same-minute hops cannot be ordered and are excluded. This is
deliberate and measured -- if exact recall tanks on same-minute instance
hops, that is evidence for the fuzzy layer (Oct 29), not a bug to patch
with ``<=``.

Usage:
    graph = TemporalGraph.from_parquet("data/hi-small/transfers.parquet")
    edges, times = graph.incoming(node, before_t=epoch_s, window_s=7*86400)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

log = logging.getLogger("graph")

SECS_PER_DAY = 86_400
# Longest v0 window (round_trip 7d) plus a margin; CLI default derives from this.
DEFAULT_RETENTION_DAYS = 8


class TemporalGraph:
    """CSR-backed temporal multigraph built from time-sorted transfers."""

    def __init__(
        self,
        *,
        times: npt.NDArray[np.int64],
        frm: npt.NDArray[np.int32],
        to: npt.NDArray[np.int32],
        amount: npt.NDArray[np.float32],
        transfer_id: npt.NDArray[np.uint32],
        nodes: list[str],
        out_ptr: npt.NDArray[np.int64],
        out_edges: npt.NDArray[np.uint32],
        out_times: npt.NDArray[np.int64],
        in_ptr: npt.NDArray[np.int64],
        in_edges: npt.NDArray[np.uint32],
        in_times: npt.NDArray[np.int64],
        retention_s: int,
    ) -> None:
        self.times = times
        self.frm = frm
        self.to = to
        self.amount = amount
        self.transfer_id = transfer_id
        self.nodes = nodes
        self.node_index = {key: i for i, key in enumerate(nodes)}
        self.out_ptr = out_ptr
        self.out_edges = out_edges
        self.out_times = out_times
        self.in_ptr = in_ptr
        self.in_edges = in_edges
        self.in_times = in_times
        self.retention_s = retention_s
        self.watermark = int(times[0]) if len(times) else 0
        # Lazy per-node eviction floors, rebased on first access past watermark.
        self._out_lo = np.zeros(len(nodes), dtype=np.int64)
        self._in_lo = np.zeros(len(nodes), dtype=np.int64)
        self._floors_clean = True

    # -- construction ----------------------------------------------------

    @classmethod
    def from_parquet(
        cls, path: str | Path, retention_days: int = DEFAULT_RETENTION_DAYS
    ) -> TemporalGraph:
        """Load a prepped ``transfers.parquet`` (must be time-sorted)."""
        df = pl.read_parquet(path)
        return cls.from_transfers(df, retention_days=retention_days)

    @classmethod
    def from_transfers(
        cls, df: pl.DataFrame, retention_days: int = DEFAULT_RETENTION_DAYS
    ) -> TemporalGraph:
        """Build the store from a transfers frame with prep-schema columns."""
        if df.height == 0:
            raise ValueError("cannot build a graph from zero transfers")
        mono = df.select(pl.col("timestamp").is_sorted().alias("ok")).item()
        if not mono:
            log.warning("transfers not time-sorted; sorting on build")
            df = df.sort("timestamp")

        keyed = df.with_columns(
            (pl.col("from_bank") + "/" + pl.col("from_account")).alias("_fk"),
            (pl.col("to_bank") + "/" + pl.col("to_account")).alias("_tk"),
        )
        node_frame = (
            pl.concat([keyed.select(pl.col("_fk").alias("key")),
                       keyed.select(pl.col("_tk").alias("key"))])
            .unique(maintain_order=True)
            .with_row_index("node")
        )
        nodes: list[str] = node_frame.get_column("key").to_list()
        n = len(nodes)
        # maintain_order="left": Polars joins give no row-order guarantee by
        # default, and the CSR build plus replay anchor positions both assume
        # the transfers frame stays in timestamp order. The sorted check
        # above stays as a backstop.
        keyed = (
            keyed.join(node_frame, left_on="_fk", right_on="key", how="left",
                       maintain_order="left")
            .rename({"node": "_fn"})
            .join(node_frame, left_on="_tk", right_on="key", how="left",
                  maintain_order="left")
            .rename({"node": "_tn"})
            .drop("_fk", "_tk")
        )

        times = keyed.get_column("timestamp").dt.epoch("s").to_numpy().astype(np.int64)
        frm = keyed.get_column("_fn").to_numpy().astype(np.int32)
        to = keyed.get_column("_tn").to_numpy().astype(np.int32)
        amount = keyed.get_column("amount_usd").to_numpy().astype(np.float32)
        transfer_id = keyed.get_column("transfer_id").to_numpy()
        n_edges = len(times)
        if n_edges >= 2**32:
            raise ValueError(f"edge count {n_edges} exceeds uint32 adjacency")

        out_ptr, out_edges, out_times = _csr(frm, times, n)
        in_ptr, in_edges, in_times = _csr(to, times, n)
        log.info("graph: %d nodes, %d edges, %d unique accounts",
                 n, n_edges, n)
        return cls(
            times=times, frm=frm, to=to, amount=amount,
            transfer_id=transfer_id, nodes=nodes,
            out_ptr=out_ptr, out_edges=out_edges, out_times=out_times,
            in_ptr=in_ptr, in_edges=in_edges, in_times=in_times,
            retention_s=retention_days * SECS_PER_DAY,
        )

    # -- node lookup -----------------------------------------------------

    def node_id(self, bank: str, account: str) -> int:
        """Interned id for ``bank/account``; -1 when never seen."""
        return self.node_index.get(f"{bank}/{account}", -1)

    def node_key(self, node: int) -> str:
        """Composite ``"bank/account"`` key for a node id (witness output)."""
        return self.nodes[node]

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.times)

    # -- windowed adjacency ----------------------------------------------

    def incoming(
        self, node: int, before_t: int, window_s: int
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.int64]]:
        """Edge positions and times into ``node`` over ``[before_t-window_s, before_t)``.

        Returns views, oldest first. Unknown nodes yield empty slices.
        """
        if node < 0:
            return _EMPTY_U32, _EMPTY_I64
        lo = self._rebase(self.in_ptr, self.in_times, self._in_lo, node)
        hi = self.in_ptr[node + 1]
        seg = self.in_times[lo:hi]
        i = lo + int(np.searchsorted(seg, before_t - window_s, side="left"))
        j = lo + int(np.searchsorted(seg, before_t, side="left"))
        return self.in_edges[i:j], self.in_times[i:j]

    def outgoing(
        self, node: int, before_t: int, window_s: int
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.int64]]:
        """Edge positions and times out of ``node`` over ``[before_t-window_s, before_t)``."""
        if node < 0:
            return _EMPTY_U32, _EMPTY_I64
        lo = self._rebase(self.out_ptr, self.out_times, self._out_lo, node)
        hi = self.out_ptr[node + 1]
        seg = self.out_times[lo:hi]
        i = lo + int(np.searchsorted(seg, before_t - window_s, side="left"))
        j = lo + int(np.searchsorted(seg, before_t, side="left"))
        return self.out_edges[i:j], self.out_times[i:j]

    # -- retention --------------------------------------------------------

    def evict_before(self, t: int) -> None:
        """Move the retention watermark (O(1); floors rebase lazily)."""
        if t > self.watermark:
            self.watermark = t
            self._floors_clean = False

    def _rebase(
        self, ptr: np.ndarray, times: np.ndarray, lo: np.ndarray, node: int
    ) -> int:
        """Advance one node's eviction floor past the watermark; return it."""
        base = int(ptr[node])
        # One searchsorted per call: O(log degree). Floors only move forward,
        # so the expired prefix is never re-reported, but it is re-searched.
        floor = base + int(
            np.searchsorted(times[base:ptr[node + 1]], self.watermark, side="left")
        )
        if floor > lo[node]:
            lo[node] = floor
        return int(lo[node])

    def _rebase_all(self) -> None:
        """Advance every node's floor past the watermark (estimation-time).

        Adjacency queries rebase lazily, so untouched nodes lag behind; the
        degree histogram must not report expired edges. Never called per
        anchor — only ahead of degree statistics for the cost estimator.
        """
        if self._floors_clean:
            return
        for node in range(self.n_nodes):
            self._rebase(self.in_ptr, self.in_times, self._in_lo, node)
            self._rebase(self.out_ptr, self.out_times, self._out_lo, node)
        self._floors_clean = True

    # -- degrees and cost statistics --------------------------------------

    def in_degree(self, node: int) -> int:
        return int(self.in_ptr[node + 1] - max(self.in_ptr[node], self._in_lo[node]))

    def out_degree(self, node: int) -> int:
        return int(self.out_ptr[node + 1] - max(self.out_ptr[node], self._out_lo[node]))

    def degree_histogram(self) -> npt.NDArray[np.int64]:
        """Live in+out degree histogram (feeds the Oct 29 cost estimator)."""
        self._rebase_all()
        in_deg = np.diff(self.in_ptr) - (self._in_lo - self.in_ptr[:-1]).clip(min=0)
        out_deg = np.diff(self.out_ptr) - (self._out_lo - self.out_ptr[:-1]).clip(min=0)
        return np.bincount((in_deg + out_deg).astype(np.int64))

    def stats(self) -> dict:
        in_deg = np.diff(self.in_ptr)
        out_deg = np.diff(self.out_ptr)
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "max_in_degree": int(in_deg.max(initial=0)),
            "max_out_degree": int(out_deg.max(initial=0)),
            "mean_degree": float((in_deg + out_deg).mean()),
            "watermark": self.watermark,
            "retention_s": self.retention_s,
        }


_EMPTY_U32 = np.empty(0, dtype=np.uint32)
_EMPTY_I64 = np.empty(0, dtype=np.int64)


def _csr(
    endpoint: npt.NDArray[np.int32], times: npt.NDArray[np.int64], n_nodes: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.uint32], npt.NDArray[np.int64]]:
    """CSR segments over ``endpoint``; stable argsort keeps time order."""
    order = np.argsort(endpoint, kind="stable").astype(np.uint32)
    counts = np.bincount(endpoint, minlength=n_nodes)
    ptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(counts, out=ptr[1:])
    return ptr, order, times[order.astype(np.int64)]
