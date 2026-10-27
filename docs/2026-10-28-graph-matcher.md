# Day report: Oct 28 — temporal graph, exact matcher, replay benchmark

Branch: `oct28-graph-matcher`. Plan row: DESIGN.md §9, Oct 28.
Status: store + both matchers + benchmark done on HI-Small; review fixes
applied (receiver-in-visited, anchor-relative window, overlap recall);
Python-vs-Rust decided (Python stays); pool added for eval speed.

## 1. What was built

`src/motifql/graph.py` — CSR temporal multigraph from `transfers.parquet`:
per-account incoming/outgoing segments, each time-ordered by construction
(stable argsort on the sender/receiver column; both Polars joins carry
`maintain_order="left"` since join output order is otherwise unspecified).
Window queries are two `searchsorted` calls plus a backwards walk. Accounts
are `bank/account` composites interned to int32 (strings stay upstream;
leading zeros safe). Retention is a watermark plus lazy per-node floors
(O(1) evict; O(log degree) rebase per lookup); the degree histogram rebases
all floors first so it never reports expired edges.

`src/motifql/matcher.py` — anchored backward DFS, crisp only, shared by both
motifs: `ROUND_TRIP` (cycle, 2–5 hops, 7d, must close on the anchor receiver)
and `LAYERING_CHAIN` (open chain, 3–6 hops, 2d, start must differ from the
anchor receiver). The receiver is always in the visited set: a walk through
it mid-path repeats it once the anchor closes the walk, so it is not a
simple path (caught by review; brute-force oracle agrees on 0/400+ anchors
after the fix, disagreed on 116 before). `max_fanout` (1,000,
most-recent-first) and `max_expansions` (50,000) per anchor; hitting either
stops the search and flags `truncated`. First (most-recent) witness wins;
`find_all` enumerates witnesses (bounded, for recall accounting and Oct 29
ranking). Deeper levels query the anchor-relative window
(`edge_t - window_start`), so out-of-window edges consume no fanout budget.

`eval/replay_bench.py` — replays every train anchor through both motifs:
wall time, throughput, p50/p99 latency, truncation, fire rate, and
overlap-based per-instance recall (anchor = last member; caught = best
witness overlap ≥ 2/3 members; chance level beside each; sub-`min_hops`
instances excluded and counted). `--workers N` shards anchors round-robin
across forked processes sharing the graph copy-on-write; `--workers 1`
runs the identical code serially.

Tests: `tests/test_graph.py` (7) + `tests/test_matcher.py` (13, incl. a
brute-force oracle over 30 random graphs × all anchors × both motifs and a
receiver-revisit regression) pass with the 3 pre-existing baseline tests
(23 total). Both new matcher tests fail on the pre-fix code.

## 2. Graph scale (warmup+train)

513,512 accounts, 3,731,672 edges, mean degree 19.7. The Oct 27 "168K hub"
is one account: `070/100428660` sends 168,672 transfers (max in-degree
anywhere is 1,084 — the hub is sender-side, and the backward search only
walks incoming edges, so its out-degree never enters a fanout list).

## 3. Benchmark (train split, 1,862,302 anchors, 16 workers)

| motif | wall | anchors/s | p50 | p99 | fires (/day) | truncated | mean expansions |
|---|---|---|---|---|---|---|---|
| round_trip | 570s | 3,268 | 620µs | 68ms | 41,310 (8,262) | 0.52% | 1,733 |
| layering_chain | 5.4s | 343,626 | 11µs | 807µs | 673,462 (134,692) | 0% | 6.9 |

Per-instance recall (anchor = last member, overlap ≥ 2/3):

| pattern | caught | excluded (< min_hops) | mean best | mean first | chance |
|---|---|---|---|---|---|
| CYCLE | 8/9 | 0 | 0.89 | 0.72 | ~0.2 |
| RANDOM | 2/4 | 11 | 0.73 | 0.43 | ~1.4 |

Serial-vs-parallel parity on a 100K-anchor prefix is exact (fires,
truncation, expansions, recall identical); speedup there is 7.8× / 7× on
16 workers.

## 4. Reading the numbers

- **Exact matching fires far too much to be a detector.** 2.2% of anchors
  complete a 2–5 hop cycle, 36% complete a 3–6 hop chain — orders of magnitude
  above any alert budget. The Oct 27 aggregate baseline showed windowed
  features catching ~nothing; this shows exact structure alone over-fires
  instead. Both directions point the same way: the value is in fuzzy ranking
  of witnesses, not crisp firing. Oct 29 work.
- **Overlap recall is the honest RQ1 leg.** Any-witness counting credited
  the motif for chains that merely pass through the anchor (36% of anchors
  fire, so chance alone catches ~5 of 15 RANDOM). With overlap ≥ 2/3: CYCLE
  8/9 against chance 0.2 is a real signal; RANDOM 2/4 against chance 1.4 is
  barely above chance, and 11 of 15 RANDOM train instances are too short to
  evaluate (single members like 165, pairs like 202). The chain leg needs
  the fuzzy layer before it says anything.
- **First witness ≠ best witness.** Mean first-witness overlap trails mean
  best by 0.17 (CYCLE) and 0.30 (RANDOM); instance 102's anchor completes
  via a 2-hop non-member shortcut while the true 3-hop path exists. Ranking
  witnesses — not just finding one — is what the degree layer must do.
- **The three CYCLE/RANDOM non-catches are all bound effects, not search
  bugs.** Missed CYCLE 128 has 7 members (above max_hops 5); missed RANDOM
  202 has 2 (below min_hops 3); single-member 165 is excluded by the same
  rule that excludes 10 more like it.
- **The p99 tail is branching, not hubs.** The 0.52% round_trip truncation
  comes from the expansion budget on deep searches over many
  moderate-degree accounts; the fanout cap bit zero times on sampled
  anchors. p99 (68ms) is ~100× p50 (620µs) because maxed-out anchors each
  burn up to 50K expansions. Score pruning is the lever, and no
  per-account special-casing is needed.

## 5. Decision: Python stays, pool for eval, Numba before Rust

Live needs ~8 transfers/sec at observed peak (Sep 9: 654,467/day); serial
mean per-anchor cost is ~447µs for round_trip (~2,200/sec), so single-core
Python clears the live bar by two orders of magnitude. The 68ms hub tail is
noted, not blocking: score-time pruning on Oct 29 should cut it, and only
if it doesn't, the DFS hot loop moves to Numba first (the Oct 27
precedent), Rust only if Numba fails. Nothing in the numbers forces
compiled code today.

For eval speed, `--workers 16` replays the train split in ~10 min vs a
projected ~1–2h serial (serial never finished round_trip in 24 min; the
7.8× prefix speedup is measured). Scaling is sublinear because the box
(i5-14600KF: 6 performance + 8 efficiency cores) tops out near ~8× on
this shape — no bandwidth story is needed. Further gains must come from
less work per anchor (pruning), not more cores. The pool is bought and
paid for; no further parallelism work is planned.

## 6. Streaming inserts (plan, for the proxy work)

The CSR is built once from a frame, which suits batch replay but not the
live ezrules proxy. Plan: an append-only tail buffer holding full edge rows,
queried alongside the CSR segments in `incoming`/`outgoing` and merged
into a rebuilt CSR on a size or time threshold. The matcher only reaches
edges through `incoming`/`outgoing` (plus raw-array reads for witness
output, which take the same dual path), so the change stays local. No
streaming code is built today.

## 7. Open items / next (Oct 29)

Compiler (AST, checks, planner) + general path matcher with degrees,
pruning and witnesses; grammar frozen. The 7-member cycle and the
single-member instances stay as boundary test cases for bound handling.
