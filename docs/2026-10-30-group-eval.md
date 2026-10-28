# Day report: Oct 30 — group motifs, butterfly, RQ1 harness

Branch: `oct30-group-eval`. Plan row: DESIGN.md §9, Oct 30.
Status: group executor + butterfly matcher + RQ1 harness done; all 8
reference motifs replay end-to-end (train + holdout). 58/58 tests pass.

## 1. What was built

`src/motifql/planner.py` — `compile_motif` now dispatches on shape:
`GroupPlan` (fan / scatter / gather, with key/member roles, window,
having/soft ASTs, cost bound) and `ButterflyPlan` (4 edge vars, sender and
receiver pairs, span window). Anything else multi-pattern is still a clean
E_PLAN. One deliberate tightening: `prev` in group soft is now E_PLAN —
it has no path-step meaning inside a group.

`src/motifql/groups.py` — one bounded anchor-touched lookup per kind:

* fan: the key's windowed adjacency plus the anchor; most-recent edge per
  member; `having` then `soft` once per group.
* scatter: anchor read as leg2 (m → d); senders into m, each sender's
  intermediaries re-checked against d, `leg2.time > leg1.time` enforced
  per member. Anchors serving only as leg1 score 0 by construction.
* gather: anchor read as p (c → y); incoming senders and outgoing
  receivers of c. Anchor-as-g groups have no valid (g, p) pair with both
  legs before the anchor, so reading every anchor as p is complete.
* butterfly: anchor is one of e1..e4; partner receivers of the sender and
  partner senders of the receiver (capped at 50 each), cross-edge existence
  per pair; score is `span(all) is soon`.

Group soft reuses `degrees.py` atoms (`term_degree`, `tol_degree`,
`resolve_term_args`) with its own aggregate layer (`count distinct`,
`sum`/`avg`/`min`/`max`, `span`) and per-member quantifier evaluation
(each/all/some/at-least/most/few/@name, same semantics as paths).
Hub caps mirror the path matcher (1,000 per list, 50K expansions);
hitting one marks `truncated`, never a silent cut. Groups have no score
pruning — aggregates are not monotone as the group grows — so `top_score`
is exact everywhere.

`eval/rq1.py` — replays all 8 motifs over train (Sep 3–7, 1,862,302
anchors) and holdout (Sep 8–10), ranks holdout per day by `top_score`
with transfer id as the deterministic tie-break, and reports transfer
and instance recall at 50/100/500/1,000/5,000 alerts/day through the
shared `eval/evaluate.py` protocol. Also writes `rq1_summary.csv`
(fires, nonzeros, expansions, truncation per motif/split) and per-anchor
score parquets for Nov 1 tuning (train-only fitting guardrail).

## 2. Full replay (16 workers)

| motif | train fires/day | holdout fires/day | nonzero/day (tr/ho) | mean exp (tr/ho) | trunc (tr/ho) |
|---|---|---|---|---|---|
| round_trip @0.6 | 3,065 | 3,820 | 3,065 / 3,820 | 41 / 41 | 0% / 0% |
| layering_chain @0.7 | 3.6 | 17.7 | 3.6 / 17.7 | 8 / 15 | 0% / 0% |
| fan_out @0.5 | 0 | 0 | 13.6 / 13.0 | 887 / 1,013 | 9.0% / 9.2% |
| fan_in @0.5 | 0 | 0 | 20.2 / 17.0 | 3.6 / 4.0 | 0% / 0% |
| scatter_gather @0.5 | 0 | 0 | 0 / 0 | 53 / 73 | 1.3% / 2.9% |
| gather_scatter @0.5 | 6.8 | 6.0 | 211 / 386 | 107 / 112 | 9.0% / 9.2% |
| structuring_fan_in @0.6 | 0 | 0 | 0 / 0.3 | 3.6 / 4.0 | 0% / 0% |
| bipartite_butterfly @0.6 | 595 | 687 | 663 / 774 | 287 / 429 | 12.8% / 13.4% |

Per-anchor cost is 9–1,000 µs (fan_out heaviest at ~283 µs); the whole
8-motif × 3.2M-anchor replay runs in minutes on 16 workers.

## 3. RQ1 reading (honest version)

At emit (no ranking artifact — an instance counts when any member scores
at or above threshold), holdout completable instances:

| motif | alerts (holdout) | own-pattern recall | other hits |
|---|---|---|---|
| round_trip @0.6 | 11,460 | CYCLE 1/3 | 1 each of BIPARTITE, FAN-OUT, GATHER-SCATTER, RANDOM, STACK |
| layering_chain @0.7 | 53 | RANDOM 0/6 | none |
| fan_out / fan_in / scatter_gather @0.5 | 0 | 0 everywhere | none |
| gather_scatter @0.5 | 18 | GATHER-SCATTER 0/2 | none |
| structuring_fan_in @0.6 | 0 | 0 everywhere | none |
| bipartite_butterfly @0.6 | 2,061 | BIPARTITE 0/9, FAN-IN 1/4 | CYCLE 1/12, STACK 1/8 (all-set) |

Two findings, both tuning input for Nov 1 rather than engine bugs:

1. The hand-written breakpoints are too strict as written.
`several(2) = 0` and `many(≤3) = 0` zero out every small group under
`min`/`product`, so scatter/fan/structuring score exactly 0 on ~all
anchors (scatter: 0 nonzeros on 3.2M anchors). round_trip fires at
3,800/day yet ranks its 1 caught CYCLE below thousands of background
nonzeros. The GFP-default term shapes do not match HI-Small timing —
same conclusion as the Oct 29 report, now extended to groups.
2. Fixed-budget recall is degenerate for strict motifs and is reported
with that warning: with fewer nonzeros than the budget, the top-K fills
with unscored transfers in transfer-id order, identically for every
all-zero motif (fan_in, fan_out, layering_chain, scatter_gather and
structuring agree exactly: 706/1,495 at 500/day). Only round_trip
(3,820 nonzeros/day) and butterfly (774/day) out-fire the 500/day
budget, so only their budget rows measure ranking. round_trip's row is
real and damning: ~0 transfer recall at 100–1,000/day — its nonzeros
are background.

Truncation concentrates where expected: fan_out and gather (9% — the
1,000-edge window cap on hub senders) and butterfly (13% — the 50-partner
cap). Path motifs truncate at 0%.

## 4. Recalibration pass (same branch, post-review)

Motivated by the investigation into why the hand-written motifs catch
nothing (probe scripts in `/tmp/probe*.py`, not committed): true spans
dwarf the windows (SCATTER 61h vs 6h, FAN 61–85h vs 24h), `soon` dies
before true gaps (13–70h), and avg-referenced similarity self-defeats in
background-enriched groups. Changes, all in `examples/amlworld.motif`:

* New term `patient = trapezoid(0h, 0h, 24h, 120h)`; `round_trip` gap
  term `soon` → `patient`.
* Windows: scatter 6h → 48h; fan/gather/structuring 1d → 4d;
  `layering_chain` span 2d → 4d; butterfly span 1d → 2d.
* `several` ramp(2, 5) → ramp(1, 3): truth groups hold 2–6 members.
* fan soft dropped `@mostly … avg(…)` (inst 86 measured count 1.0 but
  `@mostly` 0.0 — background drags the mean; mean-degrees stay 0 even at
  100% tolerance).
* gather sum-balance → count-balance
  (`count(distinct x) ~= count(distinct y) within 1`): sums are
  outlier-fragile (38M transfer in inst 99) and background sums swamp
  truth sums.

Second full replay (16 workers), train/day → holdout/day fires:

| motif | train | holdout | recall movement |
|---|---|---|---|
| round_trip | 3,065 → 6,730 | 3,820 → 8,122 | CYCLE 1/3 → 2/3 (102 fires at 0.72) |
| gather_scatter | 6.8 → 15,860 | 6 → 6,496 | still 0 completable (all background) |
| fan_out (count-only) | 0 → 106,167 | 0 → 98,858 | fires on 28% of traffic |
| fan_in (count-only) | 0 → 9,507 | 0 → 8,778 | fires, precision ~0 |
| scatter_gather | 0 → 1.4 | 0 → 0.3 | 7 train fires, signs of life |
| butterfly / layering / structuring | ~unchanged | ~unchanged | — |

Reading: widening unblocked firing everywhere, but count-only softs
cannot rank — `many` saturates at 8 while hub groups reach dozens, so
truth (13–17 members) ties background and transfer-id order decides.
Probe 6 measured why the similarity term can't return in v0 form:
background FAN-IN groups run p50 = 2, p99 = 12 senders (35% sit in
3–16, exactly the truth band), and mean-referenced coherence stays 0
even at 100% tolerance. The required fix is anchor-relative
(member-vs-anchor) or pairwise similarity — a language addition, not a
breakpoint. Round_trip has no threshold separating 8k/day background
from truth (CYCLE transfers score 0.6–0.7, background fills above and
below). Tests updated to the new pins (58/58 pass); instance-102 probe
test now pins the hit at ~0.72 with pruned/unpruned agreement.

## 5. Open items (Nov 1)

Discovery/tuning: fit term breakpoints and thresholds on train only,
report on holdout with background rates attached (protocol from Oct 29).
Per-anchor score parquets (`eval/results/rq1_{train,holdout}_scores.parquet`,
gitignored) are saved for exactly that. `stack` stays out (needs
set-valued variables, LANGUAGE.md §7).

## 6. Correction (Oct 31): critical + high bugfix pass

Rerun after the review-fix pass (1 critical + 8 high; issue #6): the
self-loop fix removes ~99% of `round_trip` background, the gather
`p.time > g.time` guard drops pre-collection payouts, and scatter
pairs older leg1s when the newest is stale. Corrected post-recalibration
replay, train/day → holdout/day fires:

| motif | train | holdout | recall movement vs §4 |
|---|---|---|---|
| round_trip @0.6 | 6,730 → 22.6 | 8,122 → 24.0 | CYCLE 2/3 holds (72 holdout alerts, was ~24K) |
| layering_chain @0.7 | 3.6 → 3.4 | 17.7 → 16.0 | RANDOM 0/6, unchanged |
| fan_out @0.5 | 106,167 | 98,858 | unchanged (fix no-op, confirmed) |
| fan_in @0.5 | 9,507 | 8,778 | unchanged (fix no-op, confirmed) |
| scatter_gather @0.5 | 1.4 | 0.3 | unchanged on this data (fix pinned by test) |
| gather_scatter @0.5 | 15,860 → 15,370 | 6,496 → 6,471 | GATHER 0/2 → 1/2 |
| structuring_fan_in @0.6 | 0 | 0 | unchanged |
| bipartite_butterfly @0.6 | 594.8 | 687.0 | unchanged (fix no-op, confirmed) |

At emit, holdout completable instances: `round_trip` catches CYCLE 2/3
plus 8 cross-pattern (STACK/BIPARTITE/FAN-IN×2/GATHER/FAN-OUT/SCATTER/
RANDOM 1 each); `gather_scatter` catches GATHER 1/2 plus BIPARTITE 1,
RANDOM 3, FAN-IN 2; `bipartite_butterfly` still BIPARTITE 0/9, FAN-IN
1/4; `fan_out` still FAN-OUT 0/2 (2 BIPARTITE); `fan_in`, `layering`,
`scatter`, `structuring` catch nothing. Net for Nov 1: path background
collapsed (thresholds can come down), gather gained a truth hit, and
the §3 degeneracy warning now covers `round_trip` too (24 nonzeros/day
cannot fill a 500/day budget). 75/75 tests pass.
