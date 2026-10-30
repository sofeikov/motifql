"""Compiled fast path for the group and butterfly executors.

The executors in :mod:`motifql.groups` stay the reference implementation.
This module reproduces their output bit for bit (found, witness, score,
top_score, degrees, truncated, expansions) with the member construction
moved into numba kernels:

* Adjacency windows are read straight from the CSR arrays (same
  watermark rebase and ``searchsorted`` bounds as ``TemporalGraph``,
  including the side effect on the lazy eviction floors).
* Every CSR segment is already in (time, position) order: positions are
  time-sorted and ``_csr`` uses a stable argsort. The reference re-sorts
  each list with a Python key lambda; the kernels rely on the order
  instead. ``TemporalGraph`` checks the invariant once at construction
  (``csr_ordered``, read by :func:`csr_time_ordered`) and the fast path
  falls back to the reference when it fails.
* Group facts that the having/soft ASTs never read are not built. The
  context is lazy: edge amounts, times, span and per-member readings are
  materialised on first access, with the same Python lists (same element
  order) the reference builds, so every float sum, mean and product the
  shared evaluators compute is unchanged. Hub-cap and tie-cap truncation
  flags are computed in the kernels for every member, read or not.
* For scatter, senders that fail a leading ``count(distinct m) >= k``
  having gate are skipped inside the kernel (the reference evaluates the
  gate first and ``all()`` short-circuits, so nothing else is observable).

The having/soft evaluators themselves are the reference functions from
``motifql.groups``; nothing here caches scores across calls.

Set ``MOTIFQL_PURE_PYTHON=1`` (or ``motifql.groups.PURE_PYTHON = True``)
to force the reference path.
"""

from __future__ import annotations

import hashlib
import math

import numba as nb
import numpy as np

from motifql import groups as _g
from motifql.parser import And, Call, Comparison, DistinctArg, Dur, Money, Num, Pct
from motifql.planner import ButterflyPlan, GroupPlan

_NEVER = 1 << 62  # a count gate no member set can reach
# Scatter leg1 lookup: 0 = per anchor, whichever of the two reads fewer
# edges; 1 = scan each sender's outgoing window; 2 = read the table
# mids' incoming windows. All three give identical results (tests pin it).
SCATTER_STRATEGY = 0


# -- graph invariant ---------------------------------------------------------

def csr_time_ordered(graph) -> bool:
    """True when every CSR segment is in (time, position) order and
    positions are time-sorted, so position order is (time, position)
    order everywhere. ``TemporalGraph`` checks this once at construction
    (``csr_ordered``); its arrays are read-only afterwards."""
    return bool(graph.csr_ordered)


# -- label-blind tie order (identical to groups._hkey) -----------------------

def _hkey(edge_pos) -> int:
    # Reads the seed per call, like the reference: nothing captured at import.
    h = hashlib.md5(f"{_g._TIE_SEED}:{edge_pos}".encode())
    return int.from_bytes(h.digest()[:8], "big")


def _tied_latest(times, edges: list[int]) -> tuple[list[int], bool]:
    """Same result as groups._tied_latest (keyed-hash order, capped)."""
    if not edges:
        return [], False
    tv = times[edges].tolist()
    tmax = max(tv)
    tied = [e for e, t in zip(edges, tv) if t == tmax]
    if len(tied) > 1:
        tied.sort(key=_hkey)
    if len(tied) > _g.MAX_TIED_READINGS:
        return tied[:_g.MAX_TIED_READINGS], True
    return tied, False


# -- numba kernels ------------------------------------------------------------

@nb.njit(cache=True)
def _tail_start(n, cap):
    """Start index of ``x[-cap:]`` for ``len(x) == n`` (Python slicing)."""
    start = -cap
    if start < 0:
        start += n
        if start < 0:
            start = 0
    elif start > n:
        start = n
    return start


@nb.njit(cache=True)
def _window(ptr, adj_t, lo_arr, node, watermark, before_t, lb):
    """TemporalGraph.incoming/outgoing bounds over [lb, before_t)."""
    base = ptr[node]
    hi = ptr[node + 1]
    floor = base + np.searchsorted(adj_t[base:hi], watermark)
    if floor > lo_arr[node]:
        lo_arr[node] = floor
    lo = lo_arr[node]
    seg = adj_t[lo:hi]
    i = lo + np.searchsorted(seg, lb)
    j = lo + np.searchsorted(seg, before_t)
    return i, j


@nb.njit(cache=True)
def _distinct_ties(keys, edges, times, max_tied):
    """Distinct keys, and whether any key's latest-time run exceeds max_tied.

    `edges` are in (time, position) order, so a stable sort by key keeps
    each key's edges time-ordered and its latest run is its tail.
    """
    n = keys.shape[0]
    if n == 0:
        return 0, False
    order = np.argsort(keys, kind="mergesort")
    distinct = 0
    cut = False
    idx = n - 1
    while idx >= 0:
        k = keys[order[idx]]
        tl = times[edges[order[idx]]]
        run = 0
        while idx >= 0 and keys[order[idx]] == k:
            if times[edges[order[idx]]] == tl:
                run += 1
            idx -= 1
        distinct += 1
        if run > max_tied:
            cut = True
    return distinct, cut


@nb.njit(cache=True)
def _fan_kernel(times, side, ptr, adj_edges, adj_t, lo_arr, watermark,
                anchor, anchor_t, lb, key_node, max_fanout, max_exp, max_tied):
    i, j = _window(ptr, adj_t, lo_arr, key_node, watermark, anchor_t, lb)
    n = j - i
    trunc = False
    expansions = n + 1
    if n > max_fanout:
        i += _tail_start(n, max_fanout)
        trunc = True
    if expansions > max_exp:
        trunc = True
    m = j - i
    all_e = np.empty(m + 1, np.int64)
    keys = np.empty(m + 1, np.int64)
    for k in range(m):
        e = np.int64(adj_edges[i + k])
        all_e[k] = e
        keys[k] = side[e]
    all_e[m] = anchor
    keys[m] = side[anchor]
    nmem, cut = _distinct_ties(keys, all_e, times, max_tied)
    if cut:
        trunc = True
    return all_e, nmem, trunc, expansions


@nb.njit(cache=True)
def _gather_kernel(times, frm, to, in_ptr, in_edges, in_t, in_lo,
                   out_ptr, out_edges, out_t, out_lo, watermark,
                   anchor, anchor_t, lb, c, max_fanout, max_tied):
    gi, gj = _window(in_ptr, in_t, in_lo, c, watermark, anchor_t, lb)
    pi, pj = _window(out_ptr, out_t, out_lo, c, watermark, anchor_t, lb)
    trunc = False
    if gj - gi > max_fanout:
        gi += _tail_start(gj - gi, max_fanout)
        trunc = True
    if pj - pi > max_fanout:
        pi += _tail_start(pj - pi, max_fanout)
        trunc = True
    ng = gj - gi
    if ng == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64), 0, 0, trunc, (pj - pi) + 1
    g = np.empty(ng, np.int64)
    gk = np.empty(ng, np.int64)
    for k in range(ng):
        e = np.int64(in_edges[gi + k])
        g[k] = e
        gk[k] = frm[e]
    earliest_g = times[g[0]]
    k0 = pi
    while k0 < pj and times[out_edges[k0]] <= earliest_g:
        k0 += 1
    npf = pj - k0
    p = np.empty(npf + 1, np.int64)
    pk = np.empty(npf + 1, np.int64)
    for k in range(npf):
        e = np.int64(out_edges[k0 + k])
        p[k] = e
        pk[k] = to[e]
    p[npf] = anchor
    pk[npf] = to[anchor]
    nx, cutx = _distinct_ties(gk, g, times, max_tied)
    ny, cuty = _distinct_ties(pk, p, times, max_tied)
    if cutx or cuty:
        trunc = True
    return g, p, nx, ny, trunc, ng + npf + 1




@nb.njit(cache=True)
def _window_ro(ptr, adj_t, lo_arr, node, watermark, before_t, lb):
    """`_window` without advancing the node's eviction floor."""
    base = ptr[node]
    hi = ptr[node + 1]
    lo = base + np.searchsorted(adj_t[base:hi], watermark)
    if lo_arr[node] > lo:
        lo = lo_arr[node]
    seg = adj_t[lo:hi]
    i = lo + np.searchsorted(seg, lb)
    j = lo + np.searchsorted(seg, before_t)
    return i, j


@nb.njit(cache=True)
def _scatter_setup(times, frm, in_ptr, in_edges, in_t, in_lo,
                   out_ptr, out_t, out_lo, watermark,
                   anchor, anchor_t, lb, m0, d, max_list, max_senders, strategy):
    i0, j0 = _window(in_ptr, in_t, in_lo, m0, watermark, anchor_t, lb)
    i1, j1 = _window(in_ptr, in_t, in_lo, d, watermark, anchor_t, lb)
    expansions = (j0 - i0) + (j1 - i1) + 1
    trunc = False
    if j0 - i0 > max_list:
        i0 += _tail_start(j0 - i0, max_list)
        trunc = True
    if j1 - i1 > max_list:
        i1 += _tail_start(j1 - i1, max_list)
        trunc = True
    # Latest leg1 per sender into m0 (last in (time, position) order).
    n0 = j0 - i0
    snd = np.empty(n0, np.int64)
    for k in range(n0):
        snd[k] = frm[in_edges[i0 + k]]
    order = np.argsort(snd, kind="mergesort")
    us = np.empty(n0, np.int64)
    ul = np.empty(n0, np.int64)
    nu = 0
    for idx in range(n0):
        k = order[idx]
        if idx == n0 - 1 or snd[order[idx + 1]] != snd[k]:
            us[nu] = snd[k]
            ul[nu] = in_edges[i0 + k]
            nu += 1
    # Senders by (leg1 time, sender id): us is id-ascending, stable by time.
    lt = np.empty(nu, np.int64)
    for k in range(nu):
        lt[k] = times[ul[k]]
    o2 = np.argsort(lt, kind="mergesort")
    st = _tail_start(nu, max_senders)
    if st > 0:
        trunc = True
    ns = nu - st
    senders = np.empty(ns, np.int64)
    leg1 = np.empty(ns, np.int64)
    for k in range(ns):
        senders[k] = us[o2[st + k]]
        leg1[k] = ul[o2[st + k]]
    # Leg2 table: in_d grouped by intermediary (each group oldest first),
    # the anchor appended to m0's group.
    n1 = j1 - i1
    cm = np.empty(n1 + 1, np.int64)
    ce = np.empty(n1 + 1, np.int64)
    for k in range(n1):
        e = np.int64(in_edges[i1 + k])
        cm[k] = frm[e]
        ce[k] = e
    cm[n1] = m0
    ce[n1] = anchor
    order = np.argsort(cm, kind="mergesort")
    l2e = np.empty(n1 + 1, np.int64)
    mids = np.empty(n1 + 1, np.int64)
    mid_ptr = np.empty(n1 + 2, np.int64)
    nm = 0
    for idx in range(n1 + 1):
        k = order[idx]
        l2e[idx] = ce[k]
        if idx == 0 or cm[k] != cm[order[idx - 1]]:
            mids[nm] = cm[k]
            mid_ptr[nm] = idx
            nm += 1
    mid_ptr[nm] = n1 + 1
    mids = mids[:nm].copy()
    mid_ptr = mid_ptr[:nm + 1].copy()
    l2_tied = np.empty(nm, np.int64)
    for s in range(nm):
        g1 = mid_ptr[s + 1]
        tl = times[l2e[g1 - 1]]
        cnt = 0
        q = g1 - 1
        while q >= mid_ptr[s] and times[l2e[q]] == tl:
            cnt += 1
            q -= 1
        l2_tied[s] = cnt
    m0_slot = np.searchsorted(mids, m0)
    # Leg1 lookup strategy. A sender's leg1 candidates are its window
    # edges into table mids; reading them from each mid's incoming window
    # (same time bounds, so the same edge set) beats scanning every
    # sender's whole outgoing window when hub senders dominate.
    use_in = strategy == 2
    if strategy == 0:
        cost_out = 0
        for k in range(ns):
            if senders[k] != d:
                i, j = _window_ro(out_ptr, out_t, out_lo, senders[k], watermark, anchor_t, lb)
                cost_out += j - i
        cost_in = 0
        for slot in range(nm):
            if mids[slot] != d:
                i, j = _window_ro(in_ptr, in_t, in_lo, mids[slot], watermark, anchor_t, lb)
                cost_in += j - i
        use_in = cost_in < cost_out
    pair_key = np.empty(0, np.int64)
    pair_slot = np.empty(0, np.int64)
    if use_in:
        tot = 0
        for slot in range(nm):
            if mids[slot] != d:
                i, j = _window_ro(in_ptr, in_t, in_lo, mids[slot], watermark, anchor_t, lb)
                tot += j - i
        keys = np.empty(tot, np.int64)
        slots = np.empty(tot, np.int64)
        npair = 0
        for slot in range(nm):
            mid = mids[slot]
            if mid == d:
                continue
            i, j = _window_ro(in_ptr, in_t, in_lo, mid, watermark, anchor_t, lb)
            for k in range(i, j):
                e = np.int64(in_edges[k])
                sd = np.int64(frm[e])
                if sd == mid:
                    continue  # the source is never its own intermediary
                keys[npair] = (sd << 32) | e  # (sender, position) order
                slots[npair] = slot
                npair += 1
        po = np.argsort(keys[:npair])
        pair_key = np.empty(npair, np.int64)
        pair_slot = np.empty(npair, np.int64)
        for k in range(npair):
            pair_key[k] = keys[po[k]]
            pair_slot[k] = slots[po[k]]
    return (expansions, trunc, senders, leg1, mids, mid_ptr, l2e, l2_tied, m0_slot,
            use_in, pair_key, pair_slot)


@nb.njit(cache=True)
def _scatter_next(times, to, out_ptr, out_edges, out_t, out_lo, watermark,
                  anchor, anchor_t, lb, m0, d, senders, leg1, start,
                  mids, mid_ptr, l2e, l2_tied, m0_slot, use_in, pair_key, pair_slot,
                  min_count, max_list, max_tied, expansions, trunc):
    """Advance the sender loop to the next sender whose member set passes
    the count gate; return its members (or ns when none is left)."""
    nmid = mids.shape[0]
    member_of = np.full(nmid, -1, np.int64)
    cand_slot = np.empty(nmid, np.int64)
    cnt = np.empty(nmid, np.int64)
    mem_slot = np.empty(nmid + 1, np.int64)
    mem_a0 = np.empty(nmid + 1, np.int64)
    mem_a1 = np.empty(nmid + 1, np.int64)
    mem_b0 = np.empty(nmid + 1, np.int64)
    mem_b1 = np.empty(nmid + 1, np.int64)
    ns = senders.shape[0]
    for si in range(start, ns):
        s = senders[si]
        if s == d:
            continue  # s != d guard
        i, j = _window(out_ptr, out_t, out_lo, s, watermark, anchor_t, lb)
        n = j - i
        expansions += n
        capped = False
        if n > max_list:
            i += _tail_start(n, max_list)
            trunc = True
            capped = True
        # Pass 1: the sender's window edges into table mids, oldest first,
        # tagged with candidate ids in first-occurrence order.
        if use_in:
            klo = np.searchsorted(pair_key, s << 32)
            khi = np.searchsorted(pair_key, (s + 1) << 32)
            nmax = khi - klo
        else:
            klo = 0
            khi = 0
            nmax = j - i
        rel_e = np.empty(nmax, np.int64)
        rel_c = np.empty(nmax, np.int64)
        nrel = 0
        ncand = 0
        if use_in:
            if i < j:
                cut_pos = np.int64(out_edges[i])  # out segment is position-sorted
                for k in range(klo, khi):
                    e = pair_key[k] & 0xFFFFFFFF
                    if capped and e < cut_pos:
                        continue
                    slot = pair_slot[k]
                    c = member_of[slot]
                    if c < 0:
                        c = ncand
                        member_of[slot] = c
                        cand_slot[c] = slot
                        cnt[c] = 0
                        ncand += 1
                    cnt[c] += 1
                    rel_e[nrel] = e
                    rel_c[nrel] = c
                    nrel += 1
        else:
            for k in range(i, j):
                e = np.int64(out_edges[k])
                mid = to[e]
                if mid == d or mid == s:
                    continue
                slot = np.searchsorted(mids, mid)
                if slot >= nmid or mids[slot] != mid:
                    continue
                c = member_of[slot]
                if c < 0:
                    c = ncand
                    member_of[slot] = c
                    cand_slot[c] = slot
                    cnt[c] = 0
                    ncand += 1
                cnt[c] += 1
                rel_e[nrel] = e
                rel_c[nrel] = c
                nrel += 1
        for c in range(ncand):
            member_of[cand_slot[c]] = -1
        # Pass 2: leg1 edges per candidate, contiguous, oldest first.
        off = np.empty(ncand + 1, np.int64)
        off[0] = 0
        for c in range(ncand):
            off[c + 1] = off[c] + cnt[c]
        buf = np.empty(off[ncand], np.int64)
        fill = off[:ncand].copy()
        for k in range(nrel):
            c = rel_c[k]
            buf[fill[c]] = rel_e[k]
            fill[c] += 1
        # Members: leg1/leg2 overlap in order; valid prefix / suffix.
        nmem = 0
        m0_in = False
        for c in range(ncand):
            slot = cand_slot[c]
            g0 = mid_ptr[slot]
            g1 = mid_ptr[slot + 1]
            latest_l2 = times[l2e[g1 - 1]]
            a0 = off[c]
            a1 = off[c + 1]
            earliest_l1 = times[buf[a0]]
            if not earliest_l1 < latest_l2:
                continue
            v1 = a0
            while v1 < a1 and times[buf[v1]] < latest_l2:
                v1 += 1
            v2 = g0
            while times[l2e[v2]] <= earliest_l1:
                v2 += 1
            tl = times[buf[v1 - 1]]
            n1 = 0
            q = v1 - 1
            while q >= a0 and times[buf[q]] == tl:
                n1 += 1
                q -= 1
            n2 = l2_tied[slot]
            if n1 > max_tied or n2 > max_tied:
                trunc = True
            if min(n1, max_tied) * min(n2, max_tied) > max_tied:
                trunc = True
            mem_slot[nmem] = slot
            mem_a0[nmem] = a0
            mem_a1[nmem] = v1
            mem_b0[nmem] = v2
            mem_b1[nmem] = g1
            nmem += 1
            if slot == m0_slot:
                m0_in = True
        fallback = False
        if not m0_in and m0 != s and anchor_t > times[leg1[si]]:
            fallback = True
        tot = nmem + (1 if fallback else 0)
        if tot < min_count:
            continue
        mm = np.empty(tot, np.int64)
        l1p = np.empty(tot + 1, np.int64)
        l2p = np.empty(tot + 1, np.int64)
        l1p[0] = 0
        l2p[0] = 0
        for r in range(nmem):
            mm[r] = mids[mem_slot[r]]
            l1p[r + 1] = l1p[r] + (mem_a1[r] - mem_a0[r])
            l2p[r + 1] = l2p[r] + (mem_b1[r] - mem_b0[r])
        if fallback:
            mm[nmem] = m0
            l1p[nmem + 1] = l1p[nmem] + 1
            l2p[nmem + 1] = l2p[nmem] + 1
        l1f = np.empty(l1p[tot], np.int64)
        l2f = np.empty(l2p[tot], np.int64)
        for r in range(nmem):
            p = l1p[r]
            for k in range(mem_a0[r], mem_a1[r]):
                l1f[p] = buf[k]
                p += 1
            p = l2p[r]
            for k in range(mem_b0[r], mem_b1[r]):
                l2f[p] = l2e[k]
                p += 1
        if fallback:
            l1f[l1p[nmem]] = leg1[si]
            l2f[l2p[nmem]] = anchor
        return si, expansions, trunc, mm, l1f, l1p, l2f, l2p
    e0 = np.empty(0, np.int64)
    return ns, expansions, trunc, e0, e0.copy(), e0.copy(), e0.copy(), e0.copy()


@nb.njit(cache=True)
def _first_occurrence_groups(keys, valid):
    """Group ids by first occurrence among valid elements.

    Returns (group key per id, group id per element (-1 invalid),
    element ptr per id, elements grouped (indices, original order)).
    """
    n = keys.shape[0]
    gid = np.full(n, -1, np.int64)
    gkey = np.empty(n, np.int64)
    ng = 0
    for k in range(n):
        if not valid[k]:
            continue
        found = -1
        for q in range(ng):  # n <= partner cap (small)
            if gkey[q] == keys[k]:
                found = q
                break
        if found < 0:
            found = ng
            gkey[ng] = keys[k]
            ng += 1
        gid[k] = found
    cnt = np.zeros(ng + 1, np.int64)
    for k in range(n):
        if gid[k] >= 0:
            cnt[gid[k] + 1] += 1
    for q in range(ng):
        cnt[q + 1] += cnt[q]
    fill = cnt[:ng].copy()
    elems = np.empty(cnt[ng], np.int64)
    for k in range(n):
        if gid[k] >= 0:
            elems[fill[gid[k]]] = k
            fill[gid[k]] += 1
    return gkey[:ng].copy(), gid, cnt, elems


@nb.njit(cache=True)
def _butterfly_kernel(times, frm, to, in_ptr, in_edges, in_t, in_lo,
                      out_ptr, out_edges, out_t, out_lo, watermark,
                      anchor_t, lb, a, b, max_partners, max_fanout, max_exp):
    ai, aj = _window(out_ptr, out_t, out_lo, a, watermark, anchor_t, lb)
    bi, bj = _window(in_ptr, in_t, in_lo, b, watermark, anchor_t, lb)
    trunc = False
    if aj - ai > max_partners:
        ai += _tail_start(aj - ai, max_partners)
        trunc = True
    if bj - bi > max_partners:
        bi += _tail_start(bj - bi, max_partners)
        trunc = True
    na = aj - ai
    nb_ = bj - bi
    ea = np.empty(na, np.int64)
    ka = np.empty(na, np.int64)
    va = np.empty(na, np.bool_)
    for k in range(na):
        e = np.int64(out_edges[ai + k])
        ea[k] = e
        ka[k] = to[e]
        va[k] = to[e] != b
    eb = np.empty(nb_, np.int64)
    kb = np.empty(nb_, np.int64)
    vb = np.empty(nb_, np.bool_)
    for k in range(nb_):
        e = np.int64(in_edges[bi + k])
        eb[k] = e
        kb[k] = frm[e]
        vb[k] = frm[e] != a
    y2_key, _y2_gid, y2_ptr, y2_el = _first_occurrence_groups(ka, va)
    x2_key, _x2_gid, x2_ptr, x2_el = _first_occurrence_groups(kb, vb)
    ny2 = y2_key.shape[0]
    nx2 = x2_key.shape[0]
    y2_edges = np.empty(y2_el.shape[0], np.int64)
    for k in range(y2_el.shape[0]):
        y2_edges[k] = ea[y2_el[k]]
    x2_edges = np.empty(x2_el.shape[0], np.int64)
    for k in range(x2_el.shape[0]):
        x2_edges[k] = eb[x2_el[k]]
    expansions = na + nb_ + 1
    # x2 cross-map build order: newest partner first (stable on ties).
    neg_max = np.empty(nx2, np.int64)
    for q in range(nx2):
        mx = times[x2_edges[x2_ptr[q]]]
        for k in range(x2_ptr[q], x2_ptr[q + 1]):
            if times[x2_edges[k]] > mx:
                mx = times[x2_edges[k]]
        neg_max[q] = -mx
    build = np.argsort(neg_max, kind="mergesort")
    in_cross = np.zeros(nx2, np.bool_)
    cx_i = np.zeros(nx2, np.int64)
    cx_j = np.zeros(nx2, np.int64)
    for r in range(nx2):
        q = build[r]
        i, j = _window(out_ptr, out_t, out_lo, x2_key[q], watermark, anchor_t, lb)
        expansions += j - i
        if j - i > max_fanout:
            i += _tail_start(j - i, max_fanout)
            trunc = True
        in_cross[q] = True
        cx_i[q] = i
        cx_j[q] = j
        if expansions > max_exp:
            trunc = True
            break
    # y2 lookup: sorted keys -> insertion id.
    y2_order = np.argsort(y2_key, kind="mergesort")
    y2_sorted = np.empty(ny2, np.int64)
    for k in range(ny2):
        y2_sorted[k] = y2_key[y2_order[k]]
    # Pairs in scoring order: x2 insertion order, then y2 insertion order.
    pair_x = np.empty(0, np.int64)
    pair_y = np.empty(0, np.int64)
    np_cap = 0
    npairs = 0
    xy_flat = np.empty(0, np.int64)
    xy_ptr = np.zeros(1, np.int64)
    nflat = 0
    for q in range(nx2):
        if not in_cross[q]:
            continue
        i = cx_i[q]
        j = cx_j[q]
        m = j - i
        if m == 0 or ny2 == 0:
            continue
        yid = np.empty(m, np.int64)
        ycnt = np.zeros(ny2 + 1, np.int64)
        for k in range(m):
            t = to[out_edges[i + k]]
            pos = np.searchsorted(y2_sorted, t)
            if pos < ny2 and y2_sorted[pos] == t:
                yid[k] = y2_order[pos]
                ycnt[yid[k] + 1] += 1
            else:
                yid[k] = -1
        for y in range(ny2):
            ycnt[y + 1] += ycnt[y]
        fill = ycnt[:ny2].copy()
        grouped = np.empty(ycnt[ny2], np.int64)
        for k in range(m):
            if yid[k] >= 0:
                grouped[fill[yid[k]]] = np.int64(out_edges[i + k])
                fill[yid[k]] += 1
        for y in range(ny2):
            cnt_y = ycnt[y + 1] - ycnt[y]
            if cnt_y == 0:
                continue
            if npairs == np_cap:
                np_cap = max(8, 2 * np_cap)
                nx = np.empty(np_cap, np.int64)
                ny = np.empty(np_cap, np.int64)
                nxp = np.empty(np_cap + 1, np.int64)
                nx[:npairs] = pair_x[:npairs]
                ny[:npairs] = pair_y[:npairs]
                nxp[:npairs + 1] = xy_ptr[:npairs + 1]
                pair_x = nx
                pair_y = ny
                xy_ptr = nxp
            if nflat + cnt_y > xy_flat.shape[0]:
                nf = np.empty(max(16, 2 * (nflat + cnt_y)), np.int64)
                nf[:nflat] = xy_flat[:nflat]
                xy_flat = nf
            for k in range(ycnt[y], ycnt[y + 1]):
                xy_flat[nflat] = grouped[k]
                nflat += 1
            pair_x[npairs] = q
            pair_y[npairs] = y
            npairs += 1
            xy_ptr[npairs] = nflat
    return (trunc, expansions, x2_ptr, x2_edges, y2_ptr, y2_edges,
            pair_x[:npairs].copy(), pair_y[:npairs].copy(),
            xy_ptr[:npairs + 1].copy(), xy_flat[:nflat].copy())


# -- lazy group context ------------------------------------------------------

class _LazyCtx:
    """`groups._Ctx` lookalike whose facts are built on first read.

    `edges` maps edge var -> int64 positions in the reference's list
    order; amounts/times/span derive from it exactly as the reference
    builds them. `members_fn` builds `member_vals` (reference code).
    """

    __slots__ = ("counts", "_graph", "_edges", "_members_fn",
                 "_amounts", "_times", "_span", "_mv")

    def __init__(self, graph, counts: dict, edges: dict, members_fn) -> None:
        self.counts = counts
        self._graph = graph
        self._edges = edges
        self._members_fn = members_fn
        self._amounts = None
        self._times = None
        self._span = None
        self._mv = None

    @property
    def edge_amounts(self) -> dict:
        if self._amounts is None:
            amt = self._graph.amount
            self._amounts = {v: amt[e].tolist() for v, e in self._edges.items()}
        return self._amounts

    @property
    def edge_times(self) -> dict:
        if self._times is None:
            tm = self._graph.times
            self._times = {v: tm[e].astype(np.float64).tolist()
                           for v, e in self._edges.items()}
        return self._times

    @property
    def span(self) -> float:
        if self._span is None:
            arrs = [e for e in self._edges.values() if len(e)]
            if not arrs:
                self._span = 0.0
            else:
                t = self._graph.times[np.concatenate(arrs)]
                self._span = float(t.max()) - float(t.min())
        return self._span

    @property
    def member_vals(self) -> dict:
        if self._mv is None:
            self._mv = self._members_fn()
        return self._mv


def _finish_sorted(witness: tuple, s: float, degs: list, truncated: bool,
                   expansions: int, emit: float) -> _g.GroupResult:
    """groups._finish for a witness already in (time, position) order."""
    found = s >= emit
    return _g.GroupResult(found, witness, s if found else 0.0,
                          tuple(degs), truncated, expansions, s)


def _readings_one(graph, var: str, groups: dict) -> dict:
    """member -> [{(var, amount|time): value}] over tied latest edges."""
    times, amount = graph.times, graph.amount
    out = {}
    for node, es in groups.items():
        tied, _ = _tied_latest(times, es)
        out[node] = [{(var, "amount"): float(amount[pos]),
                      (var, "time"): float(times[pos])} for pos in tied]
    return out


def _group_by(keys: list, edges: list) -> dict:
    by: dict = {}
    for k, e in zip(keys, edges):
        by.setdefault(k, []).append(e)
    return by


# -- per-plan constants ------------------------------------------------------

def _count_gate(having, var: str) -> int:
    """Smallest member count that can pass `having`, when its first
    evaluated conjunct is `count(distinct var) >= k` (or `> k`); 0 when
    no such gate is provable."""
    first = having
    while isinstance(first, And):
        if not first.items:
            return 0
        first = first.items[0]
    if not (isinstance(first, Comparison) and first.op in (">=", ">")):
        return 0
    left, right = first.left, first.right
    if not (isinstance(left, Call) and left.name == "count" and len(left.args) == 1
            and isinstance(left.args[0], DistinctArg)
            and left.args[0].ref.parts[0] == var):
        return 0
    if not isinstance(right, (Num, Dur, Money, Pct)):
        return 0
    try:
        lit = _g._lit(right)
    except (TypeError, ValueError):
        return 0
    try:
        y = float(lit)
    except (TypeError, ValueError):
        return 0
    except OverflowError:  # an int literal past float range: sign decides
        return _NEVER if lit > 0 else 0
    if math.isnan(y) or y == math.inf:
        return _NEVER
    if y == -math.inf:
        return 0
    k = math.ceil(y) if first.op == ">=" else math.floor(y) + 1
    # Clamp into int64 for the kernel: no member set reaches 2**62 either.
    return min(max(int(k), 0), _NEVER)


def _prep(plan) -> dict:
    """Per-call plan constants (roles dict, scatter count gate). Derived
    afresh each call: about a microsecond, and nothing to go stale."""
    p: dict = {}
    if isinstance(plan, GroupPlan):
        roles = dict(plan.roles)
        p["roles"] = roles
        if plan.kind == "scatter" and plan.having is not None:
            p["gate"] = _count_gate(plan.having, roles["m"])
        else:
            p["gate"] = 0
    return p


# -- executors ---------------------------------------------------------------

def _caps_ok() -> bool:
    return _g.MAX_TIED_READINGS >= 0


def _score_fan(graph, anchor_pos: int, plan, libs, crisp, crisp_alpha):
    roles = _prep(plan)["roles"]
    key_is_sender = roles["dir"] == "out-key-sender"
    anchor_t = int(graph.times[anchor_pos])
    lb = anchor_t - plan.window_s
    if key_is_sender:
        key_node = int(graph.frm[anchor_pos])
        args = (graph.out_ptr, graph.out_edges, graph.out_times, graph._out_lo)
        side = graph.to
    else:
        key_node = int(graph.to[anchor_pos])
        args = (graph.in_ptr, graph.in_edges, graph.in_times, graph._in_lo)
        side = graph.frm
    all_e, nmem, trunc, expansions = _fan_kernel(
        graph.times, side, *args, float(graph.watermark), anchor_pos, anchor_t,
        float(lb), key_node, _g.DEFAULT_MAX_FANOUT, _g.DEFAULT_MAX_EXPANSIONS,
        _g.MAX_TIED_READINGS)
    truncated = bool(trunc)
    expansions = int(expansions)
    edge_var, member_var = roles["edge"], roles["member"]

    def members():
        lst = all_e.tolist()
        return {member_var: _readings_one(graph, edge_var,
                                          _group_by(side[all_e].tolist(), lst))}

    ctx = _LazyCtx(graph, {member_var: int(nmem)}, {edge_var: all_e}, members)
    if plan.having is not None and not _g._eval_having(plan.having, ctx):
        return _g.GroupResult(False, (), 0.0, (), truncated, expansions, 0.0)
    if plan.soft is None:
        s, degs = 1.0, []
    else:
        s, degs = _g._eval_soft(plan.soft, ctx, libs, plan.score_tnorm, crisp, crisp_alpha)
    return _finish_sorted(tuple(all_e.tolist()), s, degs, truncated, expansions, plan.emit)


def _score_gather(graph, anchor_pos: int, plan, libs, crisp, crisp_alpha):
    roles = _prep(plan)["roles"]
    anchor_t = int(graph.times[anchor_pos])
    c_node = int(graph.frm[anchor_pos])
    g_e, p_e, nx, ny, trunc, expansions = _gather_kernel(
        graph.times, graph.frm, graph.to,
        graph.in_ptr, graph.in_edges, graph.in_times, graph._in_lo,
        graph.out_ptr, graph.out_edges, graph.out_times, graph._out_lo,
        float(graph.watermark), anchor_pos, anchor_t, float(anchor_t - plan.window_s),
        c_node, _g.DEFAULT_MAX_FANOUT, _g.MAX_TIED_READINGS)
    truncated = bool(trunc)
    expansions = int(expansions)
    if len(g_e) == 0:
        return _g.GroupResult(False, (), 0.0, (), truncated, expansions, 0.0)
    g_var, p_var = roles["g"], roles["p"]
    x_var, y_var = roles["x"], roles["y"]

    def members():
        gl, pl_ = g_e.tolist(), p_e.tolist()
        return {
            x_var: _readings_one(graph, g_var, _group_by(graph.frm[g_e].tolist(), gl)),
            y_var: _readings_one(graph, p_var, _group_by(graph.to[p_e].tolist(), pl_)),
        }

    ctx = _LazyCtx(graph, {x_var: int(nx), y_var: int(ny)},
                   {g_var: g_e, p_var: p_e}, members)
    if plan.having is not None and not _g._eval_having(plan.having, ctx):
        return _g.GroupResult(False, (), 0.0, (), truncated, expansions, 0.0)
    if plan.soft is None:
        s, degs = 1.0, []
    else:
        s, degs = _g._eval_soft(plan.soft, ctx, libs, plan.score_tnorm, crisp, crisp_alpha)
    witness = tuple(np.sort(np.concatenate((g_e, p_e)), kind="stable").tolist())
    return _finish_sorted(witness, s, degs, truncated, expansions, plan.emit)


def _scatter_members_fn(graph, m_var, leg1_var, leg2_var, mm, l1f, l1p, l2f, l2p):
    def build():
        times, amount = graph.times, graph.amount
        l1, l2 = l1f.tolist(), l2f.tolist()
        p1, p2 = l1p.tolist(), l2p.tolist()
        readings: dict = {}
        for r, mid in enumerate(mm.tolist()):
            tied_l1, _ = _tied_latest(times, l1[p1[r]:p1[r + 1]])
            tied_l2, _ = _tied_latest(times, l2[p2[r]:p2[r + 1]])
            combos = [(x, y) for x in tied_l1 for y in tied_l2]
            if len(combos) > _g.MAX_TIED_READINGS:
                combos = combos[:_g.MAX_TIED_READINGS]
            readings[mid] = [
                {(leg1_var, "amount"): float(amount[x]),
                 (leg1_var, "time"): float(times[x]),
                 (leg2_var, "amount"): float(amount[y]),
                 (leg2_var, "time"): float(times[y]),
                 (leg2_var, "gap"): float(times[y]) - float(times[x])}
                for x, y in combos
            ]
        return {m_var: readings}
    return build


def _score_scatter(graph, anchor_pos: int, plan, libs, crisp, crisp_alpha):
    prep = _prep(plan)
    roles = prep["roles"]
    anchor_t = int(graph.times[anchor_pos])
    m0 = int(graph.frm[anchor_pos])
    d_node = int(graph.to[anchor_pos])
    if m0 == d_node:
        return _g.GroupResult(False, (), 0.0, (), False, 1, 0.0)
    m_var = roles["m"]
    leg1_var, leg2_var = roles["leg1"], roles["leg2"]
    lb = float(anchor_t - plan.window_s)
    wm = float(graph.watermark)
    (expansions, trunc, senders, leg1, mids, mid_ptr, l2e, l2_tied,
     m0_slot, use_in, pair_key, pair_slot) = _scatter_setup(
        graph.times, graph.frm, graph.in_ptr, graph.in_edges, graph.in_times,
        graph._in_lo, graph.out_ptr, graph.out_times, graph._out_lo, wm,
        anchor_pos, anchor_t, lb, m0, d_node,
        _g.SCATTER_MAX_LIST, _g.SCATTER_MAX_SENDERS, SCATTER_STRATEGY)
    gate = prep["gate"]
    max_exp = _g.SCATTER_MAX_EXPANSIONS
    ns = len(senders)
    best = None
    start = 0
    while start < ns:
        si, expansions, trunc, mm, l1f, l1p, l2f, l2p = _scatter_next(
            graph.times, graph.to, graph.out_ptr, graph.out_edges, graph.out_times,
            graph._out_lo, wm, anchor_pos, anchor_t, lb, m0, d_node, senders, leg1,
            start, mids, mid_ptr, l2e, l2_tied, m0_slot, use_in, pair_key, pair_slot, gate,
            _g.SCATTER_MAX_LIST, _g.MAX_TIED_READINGS, expansions, trunc)
        if si >= ns:
            break
        start = si + 1
        ctx = _LazyCtx(graph, {m_var: len(mm)}, {leg1_var: l1f, leg2_var: l2f},
                       _scatter_members_fn(graph, m_var, leg1_var, leg2_var,
                                           mm, l1f, l1p, l2f, l2p))
        if plan.having is not None and not _g._eval_having(plan.having, ctx):
            continue
        if plan.soft is None:
            s_score, degs = 1.0, []
        else:
            s_score, degs = _g._eval_soft(plan.soft, ctx, libs, plan.score_tnorm,
                                          crisp, crisp_alpha)
        if best is None or s_score > best[0]:
            best = (s_score, degs, l1f, l2f)
        if expansions > max_exp:
            trunc = True
            break
    truncated = bool(trunc)
    expansions = int(expansions)
    if best is None:
        return _g.GroupResult(False, (), 0.0, (), truncated, expansions, 0.0)
    s_score, degs, l1f, l2f = best
    witness = tuple(np.unique(np.concatenate((l1f, l2f))).tolist())
    return _finish_sorted(witness, s_score, degs, truncated, expansions, plan.emit)


def _realize(times, lst: list[int], t: int) -> int:
    """groups' _realize: an edge exactly at t (hash order) else the
    earliest edge after t (time, then hash)."""
    tv = times[lst].tolist()
    at = [e for e, x in zip(lst, tv) if x == t]
    if at:
        return at[0] if len(at) == 1 else min(at, key=_hkey)
    tmin = min(x for x in tv if x > t)
    nxt = [e for e, x in zip(lst, tv) if x == tmin]
    return nxt[0] if len(nxt) == 1 else min(nxt, key=_hkey)


def _score_butterfly(graph, anchor_pos: int, plan, libs, crisp, crisp_alpha):
    window_s = plan.window_s
    anchor_t = int(graph.times[anchor_pos])
    a_node = int(graph.frm[anchor_pos])
    b_node = int(graph.to[anchor_pos])
    (trunc, expansions, x2_ptr, x2_edges, y2_ptr, y2_edges, pair_x, pair_y,
     xy_ptr, xy_flat) = _butterfly_kernel(
        graph.times, graph.frm, graph.to,
        graph.in_ptr, graph.in_edges, graph.in_times, graph._in_lo,
        graph.out_ptr, graph.out_edges, graph.out_times, graph._out_lo,
        float(graph.watermark), anchor_t, float(anchor_t - window_s), a_node, b_node,
        _g.BUTTERFLY_MAX_PARTNERS, _g.DEFAULT_MAX_FANOUT, _g.DEFAULT_MAX_EXPANSIONS)
    truncated = bool(trunc)
    expansions = int(expansions)
    times = graph.times
    term_defs, _ = libs
    if plan.soft is None:
        grade = None
    else:
        soft = plan.soft
        grade = (term_defs[soft.term].shape,
                 _g.resolve_term_args(soft.term, soft.args, term_defs),
                 soft.negated)
    if len(pair_x) == 0:
        return _g.GroupResult(False, (), 0.0, (), truncated, expansions, 0.0)
    term_degree, crisp_degree = _g.term_degree, _g.crisp_degree
    x2p, x2l = x2_ptr.tolist(), x2_edges.tolist()
    y2p, y2l = y2_ptr.tolist(), y2_edges.tolist()
    xyp, xyl = xy_ptr.tolist(), xy_flat.tolist()
    best = None
    earliest = anchor_t - window_s
    for r, (q, y) in enumerate(zip(pair_x.tolist(), pair_y.tolist())):
        lists = (x2l[x2p[q]:x2p[q + 1]], y2l[y2p[y]:y2p[y + 1]], xyl[xyp[r]:xyp[r + 1]])
        tvs = [times[lst].tolist() for lst in lists]
        cap_t = min(max(tv) for tv in tvs)
        cands = sorted({x for tv in tvs for x in tv if earliest <= x <= cap_t})
        expansions += len(cands)
        for t in cands:
            span_s = float(anchor_t - t)
            if grade is None:
                d = 1.0
            else:
                shape, targs, negated = grade
                dd = term_degree(shape, targs, span_s)
                d = (1.0 - crisp_degree(dd, crisp, crisp_alpha) if negated
                     else crisp_degree(dd, crisp, crisp_alpha))
            if best is not None and d <= best[0]:
                continue
            witness = [_realize(times, lst, t) for lst in lists] + [int(anchor_pos)]
            best = (d, witness)
    if best is None:
        return _g.GroupResult(False, (), 0.0, (), truncated, expansions, 0.0)
    deg, witness = best
    label = _g._label(plan.soft) if plan.soft is not None else "score"
    return _g._finish(plan.name, witness, graph, deg, [(label, deg, None)],
                      truncated, expansions, plan.emit)


# -- entry points ------------------------------------------------------------

def _int_anchor(anchor_pos) -> bool:
    """Integer, non-bool anchors only; the reference raises (or masks) on
    anything else, so those run there and fail exactly as before."""
    return isinstance(anchor_pos, (int, np.integer)) and not isinstance(anchor_pos, bool)


def score_group(graph, anchor_pos, plan, libs, crisp: bool = False,
                crisp_alpha: float = 0.0):
    """Fast GroupPlan scoring; None when the reference must run."""
    if not _caps_ok() or not csr_time_ordered(graph) or not _int_anchor(anchor_pos):
        return None
    anchor_pos = int(anchor_pos)
    if plan.kind == "fan":
        return _score_fan(graph, anchor_pos, plan, libs, crisp, crisp_alpha)
    if plan.kind == "scatter":
        return _score_scatter(graph, anchor_pos, plan, libs, crisp, crisp_alpha)
    if plan.kind == "gather":
        return _score_gather(graph, anchor_pos, plan, libs, crisp, crisp_alpha)
    return None


def score_butterfly(graph, anchor_pos, plan, libs, crisp: bool = False,
                    crisp_alpha: float = 0.0):
    """Fast ButterflyPlan scoring; None when the reference must run."""
    if not csr_time_ordered(graph) or not _int_anchor(anchor_pos):
        return None
    return _score_butterfly(graph, int(anchor_pos), plan, libs, crisp, crisp_alpha)


def score_group_batch(graph, positions, plan, libs, crisp: bool = False,
                      crisp_alpha: float = 0.0):
    """Score many anchors against one Group/Butterfly plan.

    Returns (top_score float64, found bool, expansions int64, truncated
    bool) arrays aligned with `positions`. Anchors are scored in the
    given order through the same entry point as `score_any`, so lazy
    eviction floors advance identically.
    """
    pos = np.asarray(positions, dtype=np.int64)
    top = np.zeros(len(pos), dtype=np.float64)
    found = np.zeros(len(pos), dtype=bool)
    exp = np.zeros(len(pos), dtype=np.int64)
    trunc = np.zeros(len(pos), dtype=bool)
    if isinstance(plan, ButterflyPlan):
        fn = _g.score_butterfly
    elif isinstance(plan, GroupPlan):
        fn = _g.score_group
    else:
        raise TypeError(f"not a group/butterfly plan: {type(plan).__name__}")
    for k, a in enumerate(pos.tolist()):
        r = fn(graph, a, plan, libs, crisp=crisp, crisp_alpha=crisp_alpha)
        top[k] = r.top_score
        found[k] = r.found
        exp[k] = r.expansions
        trunc[k] = r.truncated
    return top, found, exp, trunc
