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
  The arrays are over-allocated (geometric growth on append); the
  ``times``/``frm``/``to``/``amount``/``transfer_id`` attributes are
  read-only views of the first ``n_edges`` entries, so scalar and
  whole-array readers both see exactly the live edges.
* Accounts are interned to int32 ids from the ``"bank/account"`` composite
  key (banks and accounts stay strings upstream -- leading zeros matter --
  and the composite keeps per-bank account blocks distinct). ``nodes[id]``
  maps back for witness output.
* ``out_ptr`` / ``in_ptr`` delimit each node's segment in ``out_edges`` /
  ``in_edges`` (edge positions, uint32). ``out_times`` / ``in_times``
  duplicate the timestamps in adjacency order so every window lookup is a
  plain ``numpy.searchsorted`` on a contiguous view (one extra int64 per
  edge per direction; revisited only if HI-Medium forces it).
* ``csr_ordered`` says whether the CSR holds the invariant the compiled
  fast paths rely on: ``times`` non-decreasing, every CSR segment in
  strictly increasing position order (so position order is (time,
  position) order), each segment holding only its node's edges, node ids
  in range, and ``out_times`` / ``in_times`` equal to ``times`` at those
  positions. It is also False while appended transfers sit in the
  overflow (the kernels cannot see them). ``from_transfers`` always
  produces it; when it is False the fast paths run the pure-Python
  reference. It is checked at construction and again on first use after
  a backing or CSR array is rebound (``_STRUCTURE``).

The arrays are ordinary writable numpy arrays and the graph keeps no
other copy of them. ``amount`` can be edited in place: every reader,
compiled or not, reads it directly. ``times``,
the endpoints and the CSR are the structure (``out_times`` / ``in_times``
already duplicate ``times``), so changing them means rebuilding the CSR,
i.e. rebinding its arrays, which also re-runs the ordering check.

Streaming
---------
``append`` adds one transfer as edge position ``n_edges`` (O(1)
amortized): attributes go into the spare capacity, and the edge joins
per-node overflow lists that ``incoming``/``outgoing`` merge with the
frozen CSR segments. The CSR is rebuilt (amortized over
``rebuild_threshold`` appends) once the overflow grows past the
threshold, or on an explicit ``rebuild()`` call. Node ids and
transfer_ids are stable across rebuilds; positions are not. Appends
are expected in non-decreasing timestamp order (live streams satisfy
this); out-of-order appends still merge correctly but cost an
insertion sort step per edge, and the next rebuild sorts them into
their time order.

Retention
---------
Queries are always window-bounded, so retention is a watermark plus lazy
per-node floor offsets: ``evict_before(t)`` only moves the watermark
(O(1)); each query rebases the touched node's floor with one binary
search (O(log degree)). Overflow edges are filtered against the
watermark on read. Memory of evicted edges is not freed until a
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

import bisect
import logging
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

log = logging.getLogger("graph")

SECS_PER_DAY = 86_400
# Longest v0 window (layering_chain 10d) plus a margin.
DEFAULT_RETENTION_DAYS = 11


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
        rebuild_threshold: int = 100_000,
    ) -> None:
        n = len(times)
        self._times = _with_spare(np.asarray(times))
        self._frm = _with_spare(np.asarray(frm))
        self._to = _with_spare(np.asarray(to))
        self._amount = _with_spare(np.asarray(amount))
        self._transfer_id = _with_spare(np.asarray(transfer_id))
        self._n_edges = n
        self.nodes = nodes
        self.node_index = {key: i for i, key in enumerate(nodes)}
        self.out_ptr = out_ptr
        self.out_edges = out_edges
        self.out_times = out_times
        self.in_ptr = in_ptr
        self.in_edges = in_edges
        self.in_times = in_times
        # Checked now, in the building process, so forked workers inherit it.
        self._csr_ok = _csr_ordered(self)
        self.retention_s = retention_s
        self.rebuild_threshold = rebuild_threshold
        self.watermark = int(times[0]) if len(times) else 0
        # Lazy per-node eviction floors, rebased on first access past watermark.
        # Sized to the CSR-covered nodes only; appended nodes live in the
        # overflow until a rebuild extends the CSR over them.
        self._out_lo = np.zeros(len(nodes), dtype=np.int64)
        self._in_lo = np.zeros(len(nodes), dtype=np.int64)
        self._floors_clean = True
        # Nodes covered by the CSR segments (== len(nodes) at build/rebuild;
        # appends intern nodes past this until the next rebuild).
        self._csr_nodes = len(nodes)
        # Post-CSR edges per node: lists of (time, position), sorted.
        self._ov_in: dict[int, list[tuple[int, int]]] = {}
        self._ov_out: dict[int, list[tuple[int, int]]] = {}
        self._appended = 0

    # -- live edge views -------------------------------------------------
    # The attribute arrays are over-allocated for appends; readers see
    # exactly the live prefix. Scalar indexing costs one slice (a view,
    # no copy); whole-array readers (argsort, tolist) see no spare slots.

    @property
    def times(self) -> npt.NDArray[np.int64]:
        return self._times[:self._n_edges]

    @property
    def frm(self) -> npt.NDArray[np.int32]:
        return self._frm[:self._n_edges]

    @property
    def to(self) -> npt.NDArray[np.int32]:
        return self._to[:self._n_edges]

    @property
    def amount(self) -> npt.NDArray[np.float32]:
        return self._amount[:self._n_edges]

    @property
    def transfer_id(self) -> npt.NDArray:
        return self._transfer_id[:self._n_edges]

    def __setattr__(self, name: str, value) -> None:
        object.__setattr__(self, name, value)
        if name in _STRUCTURE:
            # New structure: re-run the ordering check on next use.
            object.__setattr__(self, "_csr_ok", None)

    @property
    def csr_ordered(self) -> bool:
        """The CSR invariant (module docstring) for the current arrays.

        False while any appended transfer is still in the overflow: the
        compiled kernels read CSR segments only, so running them would
        silently miss the newest edges (wrong, not just slower). The
        pure-Python reference runs instead until `rebuild()`.
        """
        if self._appended or self._csr_nodes < len(self.nodes):
            return False
        if self._csr_ok is None:
            self._csr_ok = _csr_ordered(self)
        return self._csr_ok

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

    @classmethod
    def empty(cls, retention_days: int = DEFAULT_RETENTION_DAYS,
              rebuild_threshold: int = 100_000) -> TemporalGraph:
        """An edge-free graph that only grows through ``append`` (live use)."""
        z64 = np.empty(0, dtype=np.int64)
        return cls(
            times=z64, frm=np.empty(0, dtype=np.int32),
            to=np.empty(0, dtype=np.int32),
            amount=np.empty(0, dtype=np.float32),
            transfer_id=np.empty(0, dtype=np.int64), nodes=[],
            out_ptr=np.zeros(1, dtype=np.int64),
            out_edges=np.empty(0, dtype=np.uint32),
            out_times=np.empty(0, dtype=np.int64),
            in_ptr=np.zeros(1, dtype=np.int64),
            in_edges=np.empty(0, dtype=np.uint32),
            in_times=np.empty(0, dtype=np.int64),
            retention_s=retention_days * SECS_PER_DAY,
            rebuild_threshold=rebuild_threshold,
        )

    def append(self, timestamp: int, from_bank: str, from_account: str,
               to_bank: str, to_account: str, amount_usd: float,
               transfer_id: int) -> int:
        """Add one transfer as edge position ``n_edges``; return the position.

        O(1) amortized: attributes go into spare capacity, adjacency into
        the per-node overflow. Rebuilds the CSR once ``rebuild_threshold``
        appends accumulate since the last rebuild; a rebuild renumbers
        positions (transfer_ids stay put), so use the returned position
        promptly.
        """
        # Rebuild first: it compacts and renumbers, so a position taken
        # before it would be stale on return (wrong anchor, or out of
        # range). The new edge always lands in fresh overflow.
        if self._appended >= self.rebuild_threshold:
            self.rebuild()
        if self._n_edges >= 2**32:
            raise ValueError("edge count exceeds uint32 adjacency")
        if self._n_edges == len(self._times):
            self._grow()
        f = self._intern(f"{from_bank}/{from_account}")
        t = self._intern(f"{to_bank}/{to_account}")
        p = self._n_edges
        self._times[p] = int(timestamp)
        self._frm[p] = f
        self._to[p] = t
        self._amount[p] = amount_usd
        self._transfer_id[p] = transfer_id
        self._n_edges += 1
        _ov_insert(self._ov_out, f, int(timestamp), p)
        _ov_insert(self._ov_in, t, int(timestamp), p)
        self._appended += 1
        return p

    def _intern(self, key: str) -> int:
        i = self.node_index.get(key)
        if i is None:
            i = len(self.nodes)
            self.nodes.append(key)
            self.node_index[key] = i
        return i

    def _grow(self) -> None:
        cap = max(len(self._times) * 3 // 2, len(self._times) + 1024)
        for attr in ("_times", "_frm", "_to", "_amount", "_transfer_id"):
            a = getattr(self, attr)
            grown = np.empty(cap, dtype=a.dtype)
            grown[:len(a)] = a
            setattr(self, attr, grown)

    def rebuild(self) -> None:
        """Compact, sort and re-index every live edge; clear the overflow.

        Drops edges older than the watermark (the retention promise:
        evicted edges are freed here), sorts survivors by (time,
        position) so every CSR segment is time-ordered even after
        out-of-order appends, and renumbers positions 0..m-1. Node ids
        and transfer_ids are stable; edge positions are not — nothing
        may hold a position across a rebuild (the engine converts to
        transfer_ids synchronously under its lock). Floors reset to
        zero/clean, which is exact: no expired edge survives.
        """
        n = self._n_edges
        t = self._times[:n]
        keep = np.flatnonzero(t >= self.watermark)
        # Stable: time-ordered input keeps its positions bit for bit.
        order = keep[np.argsort(t[keep], kind="stable")]
        m = len(order)
        for attr in ("_times", "_frm", "_to", "_amount", "_transfer_id"):
            a = getattr(self, attr)
            a[:m] = a[order]  # RHS is a fancy-index copy: safe in place
        self._n_edges = m
        times = self._times[:m]
        self.out_ptr, self.out_edges, self.out_times = _csr(
            self._frm[:m], times, len(self.nodes))
        self.in_ptr, self.in_edges, self.in_times = _csr(
            self._to[:m], times, len(self.nodes))
        self._out_lo = np.zeros(len(self.nodes), dtype=np.int64)
        self._in_lo = np.zeros(len(self.nodes), dtype=np.int64)
        self._floors_clean = True
        self._csr_nodes = len(self.nodes)
        self._ov_in = {}
        self._ov_out = {}
        self._appended = 0

    @property
    def n_edges(self) -> int:
        return self._n_edges

    # -- windowed adjacency ----------------------------------------------

    def incoming(
        self, node: int, before_t: int, window_s: int
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.int64]]:
        """Edge positions and times into ``node`` over ``[before_t-window_s, before_t)``.

        Returns views, oldest first (a fresh merged array while the node
        holds post-CSR overflow). Unknown nodes yield empty slices.
        """
        if node < 0:
            return _EMPTY_U32, _EMPTY_I64
        edges, times = _EMPTY_U32, _EMPTY_I64
        if node < self._csr_nodes:
            lo = self._rebase(self.in_ptr, self.in_times, self._in_lo, node)
            hi = self.in_ptr[node + 1]
            seg = self.in_times[lo:hi]
            i = lo + int(np.searchsorted(seg, before_t - window_s, side="left"))
            j = lo + int(np.searchsorted(seg, before_t, side="left"))
            edges, times = self.in_edges[i:j], self.in_times[i:j]
        ov = self._ov_in.get(node)
        if ov:
            o_pos, o_t = _ov_slice(ov, before_t - window_s, before_t, self.watermark)
            if len(o_pos):
                if not len(edges):
                    return o_pos, o_t
                return _merge_runs(edges, times, o_pos, o_t)
        return edges, times

    def outgoing(
        self, node: int, before_t: int, window_s: int
    ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.int64]]:
        """Edge positions and times out of ``node`` over ``[before_t-window_s, before_t)``."""
        if node < 0:
            return _EMPTY_U32, _EMPTY_I64
        edges, times = _EMPTY_U32, _EMPTY_I64
        if node < self._csr_nodes:
            lo = self._rebase(self.out_ptr, self.out_times, self._out_lo, node)
            hi = self.out_ptr[node + 1]
            seg = self.out_times[lo:hi]
            i = lo + int(np.searchsorted(seg, before_t - window_s, side="left"))
            j = lo + int(np.searchsorted(seg, before_t, side="left"))
            edges, times = self.out_edges[i:j], self.out_times[i:j]
        ov = self._ov_out.get(node)
        if ov:
            o_pos, o_t = _ov_slice(ov, before_t - window_s, before_t, self.watermark)
            if len(o_pos):
                if not len(edges):
                    return o_pos, o_t
                return _merge_runs(edges, times, o_pos, o_t)
        return edges, times

    # -- retention --------------------------------------------------------

    def evict_before(self, t: int) -> None:
        """Move the retention watermark (O(1); floors rebase lazily)."""
        if t > self.watermark:
            self.watermark = t
            self._floors_clean = False

    def _rebase(
        self, ptr: npt.NDArray[np.int64], times: npt.NDArray[np.int64],
        lo: npt.NDArray[np.int64], node: int
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
        for node in range(self._csr_nodes):
            self._rebase(self.in_ptr, self.in_times, self._in_lo, node)
            self._rebase(self.out_ptr, self.out_times, self._out_lo, node)
        self._floors_clean = True

    # -- degrees and cost statistics --------------------------------------

    def in_degree(self, node: int) -> int:
        base = 0
        if 0 <= node < self._csr_nodes:
            base = int(self.in_ptr[node + 1] - max(self.in_ptr[node], self._in_lo[node]))
        return base + _ov_live(self._ov_in.get(node), self.watermark)

    def out_degree(self, node: int) -> int:
        base = 0
        if 0 <= node < self._csr_nodes:
            base = int(self.out_ptr[node + 1] - max(self.out_ptr[node], self._out_lo[node]))
        return base + _ov_live(self._ov_out.get(node), self.watermark)

    def degree_histogram(self) -> npt.NDArray[np.int64]:
        """Live in+out degree histogram (feeds the Oct 29 cost estimator)."""
        self._rebase_all()
        n = len(self.nodes)
        in_deg = np.zeros(n, dtype=np.int64)
        out_deg = np.zeros(n, dtype=np.int64)
        in_deg[:self._csr_nodes] = (
            np.diff(self.in_ptr) - (self._in_lo - self.in_ptr[:-1]).clip(min=0))
        out_deg[:self._csr_nodes] = (
            np.diff(self.out_ptr) - (self._out_lo - self.out_ptr[:-1]).clip(min=0))
        for node, lst in self._ov_in.items():
            in_deg[node] += _ov_live(lst, self.watermark)
        for node, lst in self._ov_out.items():
            out_deg[node] += _ov_live(lst, self.watermark)
        return np.bincount(in_deg + out_deg)

    def stats(self) -> dict:
        # Raw CSR widths, as before (expired edges included; use
        # degree_histogram for watermark-aware degrees).
        n = len(self.nodes)
        in_deg = np.zeros(n, dtype=np.int64)
        out_deg = np.zeros(n, dtype=np.int64)
        if self._csr_nodes:
            in_deg[:self._csr_nodes] = np.diff(self.in_ptr)
            out_deg[:self._csr_nodes] = np.diff(self.out_ptr)
        for node, lst in self._ov_in.items():
            in_deg[node] += len(lst)
        for node, lst in self._ov_out.items():
            out_deg[node] += len(lst)
        return {
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "max_in_degree": int(in_deg.max(initial=0)),
            "max_out_degree": int(out_deg.max(initial=0)),
            "mean_degree": float((in_deg + out_deg).mean()) if n else 0.0,
            "watermark": self.watermark,
            "retention_s": self.retention_s,
        }


_EMPTY_U32 = np.empty(0, dtype=np.uint32)
_EMPTY_I64 = np.empty(0, dtype=np.int64)

# Backing arrays the CSR ordering check reads through; rebinding one
# re-runs it. `times`/`frm`/`to` are read-only properties over the
# underscore arrays (which `_grow` rebinds), so the properties
# themselves are not listed: they cannot be rebound.
_STRUCTURE = frozenset(("_times", "_frm", "_to", "_amount", "_transfer_id",
                        "nodes", "out_ptr", "out_edges", "out_times",
                        "in_ptr", "in_edges", "in_times"))


def _csr_ordered(graph: TemporalGraph) -> bool:
    """The (time, position) CSR invariant (module docstring), vectorised.

    Malformed CSR (bad pointers, out-of-range positions) also reads False:
    the pure-Python reference then runs and fails or not exactly as before.
    """
    times = np.asarray(graph.times)
    n_edges = len(times)
    if n_edges > 1 and not bool(np.all(times[1:] >= times[:-1])):
        return False
    for end in (graph.frm, graph.to):
        end = np.asarray(end)
        if len(end) != n_edges or (n_edges and (int(end.min()) < 0
                                                or int(end.max()) >= graph.n_nodes)):
            return False  # unknown node ids: the reference returns empty slices
    for ptr, edges, adj_t, endpoint in (
            (graph.out_ptr, graph.out_edges, graph.out_times, graph.frm),
            (graph.in_ptr, graph.in_edges, graph.in_times, graph.to)):
        ptr = np.asarray(ptr)
        e = np.asarray(edges).astype(np.int64)
        if (len(ptr) != graph.n_nodes + 1 or int(ptr[0]) != 0 or int(ptr[-1]) != len(e)
                or bool(np.any(np.diff(ptr) < 0)) or len(adj_t) != len(e)):
            return False
        if len(e) and (int(e.min()) < 0 or int(e.max()) >= n_edges):
            return False
        if not np.array_equal(np.asarray(adj_t), times[e]):
            return False
        # Segment membership: node v's segment holds only v's edges (the
        # scatter kernel reads leg1s from mids' in-segments, not senders').
        owner = np.repeat(np.arange(graph.n_nodes, dtype=np.int64), np.diff(ptr))
        if not np.array_equal(np.asarray(endpoint)[e].astype(np.int64), owner):
            return False
        if len(e) < 2:
            continue
        inside = np.ones(len(e) - 1, dtype=bool)
        starts = ptr[1:-1].astype(np.int64)
        starts = starts[(starts > 0) & (starts < len(e))]
        inside[starts - 1] = False  # pair (k-1, k) straddles a segment start
        if not bool(np.all(np.diff(e)[inside] > 0)):
            return False
    return True


def _with_spare(a: npt.NDArray, min_spare: int = 1024) -> npt.NDArray:
    """Copy ``a`` into an over-allocated array sized for appends."""
    cap = len(a) + max(min_spare, len(a) // 32)
    out = np.empty(cap, dtype=a.dtype)
    out[:len(a)] = a
    return out


def _ov_insert(ov: dict[int, list[tuple[int, int]]], node: int,
               t: int, pos: int) -> None:
    """Insert ``(t, pos)`` keeping the node's overflow list sorted."""
    lst = ov.setdefault(node, [])
    if not lst or (t, pos) >= lst[-1]:
        lst.append((t, pos))  # live streams append in order: O(1)
    else:
        bisect.insort(lst, (t, pos))


def _ov_slice(ov: list[tuple[int, int]], lo_t: int, hi_t: int,
              watermark: int
              ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.int64]]:
    """Overflow edges in ``[max(lo_t, watermark), hi_t)``, oldest first."""
    lo = max(lo_t, watermark)
    i = bisect.bisect_left(ov, (lo, -1))
    j = bisect.bisect_left(ov, (hi_t, -1))
    if j <= i:
        return _EMPTY_U32, _EMPTY_I64
    return (np.array([p for _, p in ov[i:j]], dtype=np.uint32),
            np.array([t for t, _ in ov[i:j]], dtype=np.int64))


def _ov_live(ov: list[tuple[int, int]] | None, watermark: int) -> int:
    """Unexpired overflow edges (degree accounting)."""
    if not ov:
        return 0
    return len(ov) - bisect.bisect_left(ov, (watermark, -1))


def _merge_runs(edges: npt.NDArray[np.uint32], times: npt.NDArray[np.int64],
                 o_pos: npt.NDArray[np.uint32], o_t: npt.NDArray[np.int64]
                 ) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.int64]]:
    """Merge a CSR window slice with an overflow slice, oldest first.

    Overflow positions always exceed CSR positions, so same-timestamp ties
    order CSR-first: every overflow edge inserts after the CSR edges at or
    below its time.
    """
    at = np.searchsorted(times, o_t, side="right")
    return (np.insert(edges, at, o_pos).astype(np.uint32),
            np.insert(times, at, o_t).astype(np.int64))


def _csr(
    endpoint: npt.NDArray[np.int32], times: npt.NDArray[np.int64], n_nodes: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.uint32], npt.NDArray[np.int64]]:
    """CSR segments over ``endpoint``; stable argsort keeps time order."""
    order = np.argsort(endpoint, kind="stable").astype(np.uint32)
    counts = np.bincount(endpoint, minlength=n_nodes)
    ptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(counts, out=ptr[1:])
    return ptr, order, times[order.astype(np.int64)]
