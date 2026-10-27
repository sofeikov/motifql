# Day report: Oct 28 — temporal graph, exact matcher, replay benchmark

Branch: `oct28-graph-matcher`. Plan row: DESIGN.md §9, Oct 28.
Status: store + both matchers + benchmark done on HI-Small; Python-vs-Rust
decided (Python stays); pool added for eval speed.

## 1. What was built

`src/motifql/graph.py` — CSR temporal multigraph from `transfers.parquet`:
per-account incoming/outgoing segments, each time-ordered by construction
(stable argsort on the sender/receiver column), window queries as two
`searchsorted` calls plus a backwards walk. Accounts are `bank/account`
composites interned to int32 (strings stay upstream; leading zeros safe).
Retention is a watermark plus lazy per-node floors (O(1) evict, amortised O(1)
rebase; memory of evicted edges freed only on rebuild — documented).

`src/motifql/matcher.py` — anchored backward DFS, crisp only, shared by both
motifs: `ROUND_TRIP` (cycle, 2–5 hops, 7d, must close on the anchor receiver)
and `LAYERING_CHAIN` (open chain, 3–6 hops, 2d, start must differ from the
anchor receiver). `max_fanout` (1,000, most-recent-first) and `max_expansions`
(50,000) per anchor; hitting either stops the search and flags `truncated`.
First (most-recent) witness wins — ranking witnesses by degree is Oct 29.

`eval/replay_bench.py` — replays every train anchor in time order through
both motifs: wall time, throughput, p50/p99 per-anchor latency, truncation
rate, fire rate, and per-instance recall (anchor = last member). `--workers N`
shards anchors round-robin across forked processes sharing the graph
copy-on-write; `--workers 1` runs the identical code serially.

Tests: `tests/test_graph.py` (7) + `tests/test_matcher.py` (11) pass with
the 3 pre-existing baseline tests (21 total). Oracle style throughout:
windowed queries checked against brute-force Polars filters, evict parity,
fanout-cap behaviour, real-data smoke on train cycle instance 102.

## 2. Graph scale (warmup+train)

513,512 accounts, 3,731,672 edges, mean degree 19.7. The Oct 27 "168K hub"
turns out to be one account: `070/100428660` alone sends 168,672 transfers
(max in-degree anywhere is 1,084 — the hub is sender-side). The fanout cap
exists for exactly this account.

## 3. Benchmark (train split, 1,862,302 anchors, 16 workers)

| motif | wall | anchors/s | p50 | p99 | fires (/day) | truncated | mean expansions |
|---|---|---|---|---|---|---|---|
| round_trip | 575s | 3,238 | 610µs | 68ms | 41,310 (8,262) | 0.52% | 1,733 |
| layering_chain | 5.6s | 331,053 | 11µs | 826µs | 673,629 (134,726) | 0% | 6.9 |

Per-instance recall (anchor = last member): CYCLE 8/9, RANDOM 13/15.
Serial-vs-parallel parity on a 100K-anchor prefix is exact (fires, truncation,
expansions, recall identical); speedup there is 7.8× / 7× on 16 workers —
sublinear, see §5.

## 4. Reading the numbers

- **Exact matching fires far too much to be a detector.** 2.2% of anchors
  complete a 2–5 hop cycle, 36% complete a 3–6 hop chain — orders of magnitude
  above any alert budget. The Oct 27 aggregate baseline already showed
  windowed features catching ~nothing; this shows exact structure alone
  over-fires instead. Both directions point the same way: the value is in
  fuzzy ranking of witnesses, not crisp firing. Oct 29 work.
- **The three misses are all principled bound effects, not search bugs.**
  Missed CYCLE 128 has 7 members (above max_hops 5); missed RANDOM 165 is a
  single-member instance (a 3-hop motif cannot fire by construction);
  missed RANDOM 202 has 2 members (below min_hops 3).
- **Most-recent-first returns shortcuts.** Anchoring on instance 102's last
  member finds a 2-hop cycle through a non-member edge (1 expansion) instead
  of the true 3-hop laundering path. Correct per spec — but witness quality,
  not just witness existence, is what the fuzzy layer must rank.
- **The tail is all hubs.** round_trip p99 (68ms) is ~100× p50 (610µs):
  maxed-out anchors each burn up to 50K expansions. Truncation stays low
  (0.52%) because the caps only bite on the worst anchors.

## 5. Decision: Python stays, pool for eval, Numba before Rust

Live needs ~20 transfers/sec; even serial per-anchor p50 (120–610µs) clears
that by 100×. The p99 hub tail (68ms) is noted, not blocking: score-time
pruning on Oct 29 should cut it, and only if it doesn't, the DFS hot loop
moves to Numba first (the Oct 27 precedent), Rust only if Numba fails.
Nothing in the numbers forces compiled code today.

For eval speed, `--workers 16` replays the train split in ~10 min instead of
a projected ~1–2h serial (serial never finished round_trip in 24 min; the
7.8× prefix speedup is measured). Scaling is sublinear — 16
pointer-chasing processes saturate memory bandwidth (~170 anchors/s/worker
vs ~2,200 serial on prefix anchors) — so further gains must come from less
work per anchor (pruning), not more cores. The pool is bought and paid for;
no further parallelism work is planned.

## 6. Open items / next (Oct 29)

Compiler (AST, checks, planner) + general path matcher with degrees,
pruning and witnesses; grammar frozen. The 7-member cycle and the
single-member instance stay as boundary test cases for bound handling.
