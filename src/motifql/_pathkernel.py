"""Numba kernel for the anchored backward path walk (compute-only speedup).

A line-for-line port of ``matcher._walk`` with the scoring hooks of
``matcher._score_py`` (MODE_SCORE) or the enumeration hooks of
``matcher._search_py`` (MODE_ENUM), over the graph's flat numpy arrays.
It only removes interpreter overhead and caches nothing across calls. The
pure-Python code in ``matcher`` stays the reference and the fallback;
``tests/test_pathkernel_parity.py`` and ``eval/parity_check_path.py`` check the
two agree bit for bit.

Bit-identity rules the port follows (each mirrors a Python construct):

* Candidate order: ``_capped`` is a stable argsort of the segment times,
  reversed, cut to ``max_fanout``. On a time-sorted CSR segment that
  argsort is the identity, so candidate ``k`` sits at slot ``j - 1 - k``
  (equal times: later positions first). The caller checks every
  in-segment is sorted before using the kernel.
* Window arithmetic is float64, exactly as Python mixes ints and floats
  (``anchor_t - window_s``, ``edge_t - (edge_t - window_start)``);
  timestamps are far below 2**53, so int-to-float conversion is exact.
* Python ``min`` keeps the first minimal item (replace only on a strict
  ``<``) and ``max(a, b)`` keeps ``a`` unless ``b > a``; products start at
  1.0 and multiply left to right. Every bound, per-step and per-witness
  combination iterates in the same order as the Python lists.
* ``graph.incoming`` rebases the touched node's lazy eviction floor; the
  kernel does the same write on ``in_lo`` (unless ``write_lo`` is False,
  used only by the guarded parallel batch, see ``matcher``).
* Per-step degrees are pure functions of (parent, candidate), so the
  kernel computes each candidate's once and reuses it for the bound, the
  accept and the push, where Python recomputes the identical value.

Plan atoms arrive as one float64 table, one row per atom (see ``ATOM_*``);
helpers take scalars only, since passing arrays into inlined helpers costs
a reference-count round trip per call.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

# Atom table columns, as compiled by matcher._compile_path_plan.
(ATOM_KIND, ATOM_SHAPE, ATOM_A0, ATOM_A1, ATOM_A2, ATOM_A3, ATOM_NEG, ATOM_PCT,
 ATOM_TOLW) = range(9)
ATOM_COLS = 9
KIND_GAP = 0
KIND_AMOUNT = 1
SHAPE_TRAPEZOID = 0  # triangle(a, b, c) compiles to trapezoid(a, b, b, c)
SHAPE_RAMP = 2
SHAPE_RAMP_DOWN = 3

MODE_SCORE = 0
MODE_ENUM = 1


@njit(cache=True)
def _ss_left(arr, lo, hi, x):
    """``lo + searchsorted(arr[lo:hi], x, side="left")`` (x compared as float64)."""
    while lo < hi:
        mid = (lo + hi) >> 1
        if arr[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _gallop_left(arr, lo, hi, x):
    """``_ss_left(arr, lo, hi, x)`` probing back from ``hi`` (short windows)."""
    step = 1
    while True:
        probe = hi - step
        if probe < lo:
            break
        if arr[probe] < x:
            lo = probe + 1
            break
        hi = probe
        step <<= 1
    return _ss_left(arr, lo, hi, x)


@njit(cache=True)
def _incoming(in_ptr, in_times, in_lo, write_lo, watermark, node, before_t, lo_bound):
    """Port of ``TemporalGraph.incoming`` (with ``_rebase``): (i, j) bounds.

    Same indices as the two-searchsorted original, found more cheaply: the
    rebase search is skipped when the segment head is already at or past
    the watermark (searchsorted would return the head), and ``i`` is
    searched back from ``j`` when ``lo_bound <= before_t`` (then
    ``i <= j``). Callers treat ``j - i <= 0`` as the empty slice.
    """
    base = in_ptr[node]
    hi = in_ptr[node + 1]
    if base == hi or not (in_times[base] < watermark):
        floor = base
    else:
        floor = _ss_left(in_times, base, hi, watermark)
    cur = in_lo[node]
    if floor > cur:
        if write_lo:
            in_lo[node] = floor
        cur = floor
    j = _ss_left(in_times, cur, hi, before_t)
    if lo_bound <= before_t:
        i = _gallop_left(in_times, cur, j, lo_bound)
    else:
        i = _ss_left(in_times, cur, hi, lo_bound)
    return i, j


@njit(cache=True, inline="always")
def _term_degree(shape, a0, a1, a2, a3, x):
    """Port of ``degrees.term_degree`` (triangle pre-expanded)."""
    if shape == SHAPE_TRAPEZOID:
        if x < a0 or x > a3:
            return 0.0
        if x < a1:
            return 1.0 if a1 == a0 else (x - a0) / (a1 - a0)
        if x <= a2:
            return 1.0
        return 1.0 if a3 == a2 else (a3 - x) / (a3 - a2)
    if x <= a0:
        r = 0.0
    elif x >= a1:
        r = 1.0
    else:
        r = (x - a0) / (a1 - a0)
    if shape == SHAPE_RAMP:
        return r
    return 1.0 - r  # SHAPE_RAMP_DOWN


@njit(cache=True, inline="always")
def _tol_degree(pct, w, x, y):
    """Port of ``degrees.tol_degree`` (pct: w is the fraction, else the width)."""
    if pct:
        ax = abs(x)
        ay = abs(y)
        denom = w * (ay if ay > ax else ax)  # fraction * max(abs(x), abs(y))
    else:
        denom = w
    if denom == 0:
        return 1.0 if x == y else 0.0
    v = 1.0 - abs(x - y) / denom
    return v if v > 0.0 else 0.0  # max(0.0, v)


@njit(cache=True, inline="always")
def _crisp_neg(d, neg, crisp, alpha):
    """``degrees.crisp_degree`` then strong negation (alpha validated upstream)."""
    if crisp:
        if alpha <= 0.0:
            d = 1.0 if d > 0.0 else 0.0
        else:
            d = 1.0 if d >= alpha else 0.0
    if neg:
        return 1.0 - d
    return d


@njit(cache=True)
def walk(times, frm, to, amount, in_ptr, in_edges, in_times, in_frm, in_amt,
         in_lo, write_lo, watermark, anchor_pos, min_hops, max_hops, window_s,
         closed, max_fanout, max_expansions, mode, prune, grp_ptr, atoms,
         tnorm_prod, emit, floor0, crisp, alpha, enum_cap, enum_out, enum_len,
         best_out):
    """Anchored backward DFS (``matcher._walk``) with score or enum hooks.

    Returns (found, best_len, best_score, top_score, truncated, expansions,
    n_enum). MODE_SCORE writes the best witness (oldest first, anchor last)
    to ``best_out``; MODE_ENUM writes witnesses to ``enum_out``/``enum_len``
    and stops once ``enum_cap`` are collected (``on_accept`` returning True).

    ``in_frm`` / ``in_amt`` are ``frm[in_edges]`` / ``amount[in_edges]``
    (as ``in_times`` is ``times[in_edges]``), so candidate fields come from
    the contiguous segment rather than scattered global rows.
    """
    ngroups = grp_ptr.shape[0] - 1
    anchor_t = times[anchor_pos]
    sender = np.int64(frm[anchor_pos])
    receiver = np.int64(to[anchor_pos])
    if sender == receiver:
        return False, 0, -1.0, 0.0, False, 0, 0
    anchor_tf = np.float64(anchor_t)
    anchor_a = np.float64(amount[anchor_pos])
    window_start = anchor_tf - window_s
    expansions = 0
    truncated = False

    i0, j0 = _incoming(in_ptr, in_times, in_lo, write_lo, watermark, sender,
                       anchor_tf, window_start)
    if j0 - i0 <= 0:
        return False, 0, -1.0, 0.0, False, 0, 0
    n0 = j0 - i0
    if n0 > max_fanout:
        n0 = max_fanout
        truncated = True

    depth_cap = max_hops + 2 if max_hops > 0 else 2
    gcap = ngroups if ngroups > 0 else 1
    lvl_j = np.empty(depth_cap, dtype=np.int64)
    lvl_n = np.empty(depth_cap, dtype=np.int64)
    lvl_next = np.empty(depth_cap, dtype=np.int64)
    path = np.empty(depth_cap, dtype=np.int64)  # path[q]: q-th back-edge taken
    path_node = np.empty(depth_cap, dtype=np.int64)  # frm[path[q]] (visited)
    path_t = np.empty(depth_cap, dtype=np.int64)
    path_a = np.empty(depth_cap, dtype=np.float64)
    run_deg = np.empty((depth_cap, gcap), dtype=np.float64)  # run_stack
    cand_deg = np.empty(gcap, dtype=np.float64)
    parts = np.empty(gcap, dtype=np.float64)
    use_groups = mode == MODE_SCORE and ngroups > 0
    do_prune = prune and use_groups

    lvl_j[0] = j0
    lvl_n[0] = n0
    lvl_next[0] = 0
    nlev = 1  # stack depth; the path holds nlev - 1 edges
    stopped = False
    found = False
    best_len = 0
    best_score = -1.0
    top = 0.0
    floor = floor0
    n_enum = 0

    while nlev > 0 and not stopped:
        t = nlev - 1
        if lvl_next[t] >= lvl_n[t]:
            nlev -= 1  # _pop_level: path[-1] and run_stack[-1] go with it
            continue
        if expansions >= max_expansions:
            truncated = True
            break
        k = lvl_next[t]
        lvl_next[t] = k + 1
        slot = lvl_j[t] - 1 - k
        edge_t = in_times[slot]
        edge_tf = np.float64(edge_t)
        if edge_tf < window_start:
            nlev -= 1
            continue
        expansions += 1
        prev = np.int64(in_frm[slot])
        plen = nlev - 1
        hops = plen + 2
        if hops > max_hops:
            continue
        if np.float64(anchor_t - edge_t) > window_s:
            continue
        # Structural outcome (pure, so it may be decided before on_extend):
        # a closing step accepts at >= min_hops; otherwise the candidate
        # must not revisit the walk (sender, receiver, path senders).
        is_close = closed and prev == receiver
        if is_close:
            ok_struct = hops >= min_hops
        else:
            ok_struct = not (prev == sender or prev == receiver)
            if ok_struct:
                for q in range(plen):
                    if path_node[q] == prev:
                        ok_struct = False
                        break
        if use_groups and (do_prune or ok_struct):
            # _parent_degrees(parent, candidate): one value per group.
            if plen > 0:
                par_t = path_t[plen - 1]
                par_a = path_a[plen - 1]
            else:
                par_t = anchor_t
                par_a = anchor_a
            gap = np.float64(par_t - edge_t)
            cand_a = np.float64(in_amt[slot])
            for g in range(ngroups):
                k0 = grp_ptr[g]
                k1 = grp_ptr[g + 1]
                acc = 1.0
                for ka in range(k0, k1):
                    if atoms[ka, ATOM_KIND] == KIND_GAP:
                        d = _term_degree(atoms[ka, ATOM_SHAPE], atoms[ka, ATOM_A0],
                                         atoms[ka, ATOM_A1], atoms[ka, ATOM_A2],
                                         atoms[ka, ATOM_A3], gap)
                    else:
                        d = _tol_degree(atoms[ka, ATOM_PCT] != 0.0,
                                        atoms[ka, ATOM_TOLW], par_a, cand_a)
                    d = _crisp_neg(d, atoms[ka, ATOM_NEG] != 0.0, crisp, alpha)
                    if tnorm_prod:
                        acc *= d
                    elif ka == k0 or d < acc:
                        acc = d
                cand_deg[g] = acc  # empty group: 1.0, as `if ds else 1.0`
        if do_prune:
            # _bound(extra): per group min over stack levels then candidate,
            # combined across groups with the t-norm.
            bound = 1.0
            for g in range(ngroups):
                m = cand_deg[g]
                if plen > 0:
                    m = run_deg[0, g]
                    for lv in range(1, plen):
                        v = run_deg[lv, g]
                        if v < m:
                            m = v
                    v = cand_deg[g]
                    if v < m:
                        m = v
                if tnorm_prod:
                    bound *= m
                elif g == 0 or m < bound:
                    bound = m
            if not (bound >= floor):
                continue  # score-pruned: no truncation flag
        if not ok_struct:
            continue
        accept = is_close or ((not closed) and hops >= min_hops and prev != receiver)
        if accept:
            if mode == MODE_SCORE:
                # Witness steps i = 1..hops-1 in order: step 1 is the
                # candidate pair, step m >= 2 is run_deg[plen - m + 1].
                for g in range(ngroups):
                    m = cand_deg[g]
                    for lv in range(plen - 1, -1, -1):
                        v = run_deg[lv, g]
                        if v < m:
                            m = v
                    parts[g] = m
                s = 1.0
                for g in range(ngroups):
                    if tnorm_prod:
                        s *= parts[g]
                    elif g == 0 or parts[g] < s:
                        s = parts[g]
                if s > top:
                    top = s
                if s >= emit and ((not found) or s > best_score):
                    found = True
                    best_score = s
                    floor = s
                    best_len = hops
                    best_out[0] = np.int64(in_edges[slot])
                    for q in range(plen):
                        best_out[1 + q] = path[plen - 1 - q]
                    best_out[hops - 1] = anchor_pos
            else:
                enum_out[n_enum, 0] = np.int64(in_edges[slot])
                for q in range(plen):
                    enum_out[n_enum, 1 + q] = path[plen - 1 - q]
                enum_out[n_enum, hops - 1] = anchor_pos
                enum_len[n_enum] = hops
                n_enum += 1
                if n_enum >= enum_cap:
                    stopped = True
        if is_close:
            continue  # a closing step is never extended
        if stopped:
            break
        if hops >= max_hops:
            continue
        i2, j2 = _incoming(in_ptr, in_times, in_lo, write_lo, watermark, prev,
                           edge_tf, edge_tf - (edge_tf - window_start))
        if j2 - i2 <= 0:
            continue
        n2 = j2 - i2
        if n2 > max_fanout:
            n2 = max_fanout
            truncated = True
        if use_groups:
            for g in range(ngroups):
                run_deg[plen, g] = cand_deg[g]
        path[plen] = np.int64(in_edges[slot])
        path_node[plen] = prev
        path_t[plen] = edge_t
        path_a[plen] = np.float64(in_amt[slot])
        lvl_j[nlev] = j2
        lvl_n[nlev] = n2
        lvl_next[nlev] = 0
        nlev += 1

    return found, best_len, best_score, top, truncated, expansions, n_enum


@njit(cache=True)
def score_batch(times, frm, to, amount, in_ptr, in_edges, in_times, in_frm,
                in_amt, in_lo, watermark, positions, min_hops, max_hops, window_s,
                closed, max_fanout, max_expansions, prune, grp_ptr, atoms,
                tnorm_prod, emit, floor0, crisp, alpha, out_found, out_score,
                out_top, out_trunc, out_exp, out_wit, out_wlen):
    """Serial batch: anchors in order, floors written exactly as one-by-one."""
    dummy_enum = np.empty((1, 1), dtype=np.int64)
    dummy_len = np.empty(1, dtype=np.int64)
    width = out_wit.shape[1]
    best = np.empty(max(width, 2), dtype=np.int64)
    for r in range(positions.shape[0]):
        f, bl, bs, top, tr, ex, _ = walk(
            times, frm, to, amount, in_ptr, in_edges, in_times, in_frm, in_amt,
            in_lo, True, watermark, positions[r], min_hops, max_hops, window_s,
            closed, max_fanout, max_expansions, MODE_SCORE, prune, grp_ptr, atoms,
            tnorm_prod, emit, floor0, crisp, alpha, 1, dummy_enum, dummy_len, best)
        out_found[r] = f
        out_score[r] = bs if f else 0.0
        out_top[r] = top
        out_trunc[r] = tr
        out_exp[r] = ex
        out_wlen[r] = bl if f else 0
        for q in range(width):
            out_wit[r, q] = best[q] if (f and q < bl) else -1


@njit(cache=True, parallel=True)
def score_batch_parallel(times, frm, to, amount, in_ptr, in_edges, in_times,
                         in_frm, in_amt, lo_eff, watermark, positions, min_hops,
                         max_hops, window_s, closed, max_fanout, max_expansions,
                         prune, grp_ptr, atoms, tnorm_prod, emit, floor0, crisp,
                         alpha, out_found, out_score, out_top, out_trunc, out_exp,
                         out_wit, out_wlen):
    """prange batch over a read-only floor array (``write_lo=False``).

    Only valid when ``lo_eff`` already holds every node's rebased floor and
    the caller guarantees the serial writes would be unobservable (see
    ``matcher.score_path_batch``).
    """
    width = out_wit.shape[1]
    for r in prange(positions.shape[0]):
        dummy_enum = np.empty((1, 1), dtype=np.int64)
        dummy_len = np.empty(1, dtype=np.int64)
        best = np.empty(max(width, 2), dtype=np.int64)
        f, bl, bs, top, tr, ex, _ = walk(
            times, frm, to, amount, in_ptr, in_edges, in_times, in_frm, in_amt,
            lo_eff, False, watermark, positions[r], min_hops, max_hops, window_s,
            closed, max_fanout, max_expansions, MODE_SCORE, prune, grp_ptr, atoms,
            tnorm_prod, emit, floor0, crisp, alpha, 1, dummy_enum, dummy_len, best)
        out_found[r] = f
        out_score[r] = bs if f else 0.0
        out_top[r] = top
        out_trunc[r] = tr
        out_exp[r] = ex
        out_wlen[r] = bl if f else 0
        for q in range(width):
            out_wit[r, q] = best[q] if (f and q < bl) else -1
