# Nov 1 — RQ2 (fuzzy vs exact) and the discovery agent

DESIGN.md §9 Nov 1 row: the discovery agent (§7.2) plus RQ2
fuzzy-against-exact and perturbation runs. Exit check: the RQ2 chart.

## 1. The exact arm: support-boundary crisp flag

"Every soft condition made exact" is implemented as a scoring flag,
not text rewrites: with `crisp=True` every soft leaf (term use, `~=`
tolerance) is thresholded at its support boundary (degree > 0 → 1,
else 0). Combination is unchanged, with two deliberate rules:

- Negation is strong everywhere: leaves negate after crisping, so
  `is not T` ≡ `not (is T)` under crisp (crisp 1 iff the positive
  degree is exactly 0 — the classical support reading).
- `most`/`few`/`@named` still grade the crisped mean through their
  ramp (a ramp has two breakpoints, so no single exact analog
  exists; this choice also favors the exact arm, making any fuzzy
  win conservative).

Threaded through `score_any`/`matcher.score`/all four group
executors/butterfly (`src/motifql/degrees.py:crisp_degree`).

Two validity scopes gate the harness checks, decided per motif
before any replay (`crisp_binary_safe`, `crisp_differential_safe`):
binarity needs no mean-grading quantifiers, and the differential
`crisp>0 ⟺ fuzzy>0` additionally needs no `not` (strong negation
gives crisp=1 ⟺ fuzzy=0 while fuzzy(not E)>0 ⟺ fuzzy(E)<1). All
seven RQ2 motifs are in scope for both; `structuring_fan_in`
(`most`) is out of both. Out-of-scope checks skip with a log line —
a custom `--motifs` run never aborts after a full replay.
(`tests/test_crisp.py` pins the rule, strong negation, deferred and
group-mirror coverage, pruning soundness, and the predicates.)

## 2. Exact-max search: branch-and-bound from ~0

The first cut replayed fuzzy unpruned — and unpruned `*5..8`
layering truncates ~97% of its positive anchors at the 50k cap,
SILENTLY ZEROING a third of the candidates (fuzzy>0 on 630 anchors
vs crisp>0 on 945 in the old tables). Review #14.1 caught it.
The fix is exact branch-and-bound: pruned search
from `EXACT_FLOOR = 1e-12` explores every branch that could complete
above ~0 and skips only zero branches, returning the exact best
witness (mod 1e-12) — faster than unpruned (layering fuzzy dropped
from 213s to seconds per 20k) with no truncation-driven losses.

Side finding, kept: pruning is not just faster but MORE COMPLETE
under truncation — unpruned RQ1-style replays of deep paths miss
witnesses the pruned run finds. The Oct 30 §8 numbers predate this
understanding, and RQ1 itself should migrate to exact-floor search
(follow-up; RQ1 defaults untouched in this PR).

## 3. Level-0: fuzzy wins at tight budgets (HI-Small holdout)

`eval/rq2.py`: holdout-only replay (1.35M anchors), shared
`evaluate.py` ranking at 50/100/500/1000/5000 alerts/day.
`eval/results/rq2_{summary,recall_transfer,recall_instance}.csv`,
`rq2_holdout_scores.parquet`. Fuzzy tops are exact (§2); crisp and
fuzzy nonzero sets agree to the anchor (281/281, 945/945, 258/258)
with zero differential mismatches on completed anchors.

Mechanism: crisp scores are binary, so the exact arm holds a lottery
among emitters (945 layering fires over 3 days = 315/day); fuzzy
ranks the same candidate set by degree. Transfer recall on target
patterns (fuzzy / crisp hits):

| motif → pattern | @50 | @100 | @500 | @5000 |
|---|---|---|---|---|
| round_trip → CYCLE (92) | 12 / 7 | 14 / 13 | 15 / 15 | 15 / 15 |
| layering → RANDOM (56) | 18 / 2 | 19 / 9 | 19 / 19 | 19 / 19 |
| scatter → SG (170) | 63 / 44 | 65 / 58 | 66 / 66 | 68 / 68 |

Instance recall (completable): round_trip→CYCLE 3/3 fuzzy vs 2/3
crisp at budget 50 (3/3 both above). Layering→RANDOM 0/6 both arms
— the known holdout vacuity (no n≥5 RANDOM instances on holdout;
the transfer win is on partial/crosser members). Scatter→SG 0/1
both arms (the one completable instance's members don't rank; SG
transfers are caught cross-pattern — see §6).

Robustness caveat (review #14.4): the small-margin cells sit inside
tie-break lottery noise — crisp ties fuzzy's 3/3 CYCLE in ~26% of
seeds. The big-margin cells (layering 18v2, scatter 63v44) are
robust. Convergence at high budgets is expected: identical
candidate sets, so the value is RANKING within the set at
operational budgets, not a larger positive set.

## 4. Threshold perturbation: different operating points, not a verdict

`eval/perturb.py` threshold mode: perturb ALL members of every
target-pattern instance touching a scored (train+holdout) day —
the first cut pooled train+holdout splits only, which left most
scored targets clean (6/92 CYCLE, 6/56 RANDOM, 2/170 SG; review
#14.2 — now 92/92, 56/56, 170/170). Noise is per-tid
(SeedSequence(seed, tid)), gaps scale per instance with order
preserved. Replay truth + 20k fixed background, threshold recall at
emit. 5 unit tests (`tests/test_perturb.py`): level-0 identity,
determinism, union-independence, order preservation, day
recomputation.

Transfer recall, level-0 → 20% noise (bg fires per 20k):

| motif | crisp recall | crisp bg | fuzzy recall | fuzzy bg |
|---|---|---|---|---|
| round_trip (282) | 0.121 → 0.099 | 6 | 0.103 → 0.028 | 1 |
| layering (188) | 0.404 → 0.255 | 12 | 0.351 → 0.021 | 0 |
| scatter (626) | 0.380 → 0.340 | 9 | 0.270 → 0.203 | 0 |

Crisp recalls more at every level — structurally ({fuzzy ≥ 0.5} ⊆
{crisp = 1} always) — but pays 6–12x the background fire. The two
arms sit at different operating points, so threshold comparison
cannot declare a winner; it only shows evasion degrading both
(fuzzy more steeply in threshold-recall, since sinking scores cross
0.5 while supports hold). This table's job is motivating §3/§5:
the question isn't "who emits more at a fixed threshold" but "who
ranks truth higher at a fixed alert count".

## 5. Budget recall under noise: the edge erodes, honestly

`eval/perturb.py --budget`: full holdout replay + ranking per level
at (0,×1), (10%,×1), (20%,×1), (0,×2), (20%,×2)
(`rq2_perturb_budget_{transfer,instance}.csv`). Level 0 reproduces §3
EXACTLY (0 mismatches both granularities) through the independent
perturbed-frame pipeline — a strong end-to-end check. Splits are
recomputed from the perturbed frame, so drifted instances
reclassify (CYCLE denominators 92→89, RANDOM 56→59, SG 170→207 at
×2 — drift in and out both happen; instance CYCLE goes 3→2 for the
same reason).

Transfer hits @50 by level [(0,×1), (10%,×1), (20%,×1), (0,×2),
(20%,×2)]:

| motif → pattern | fuzzy | crisp |
|---|---|---|
| round_trip → CYCLE | 12, 10, 6, 6, 6 | 7, 7, 4, 10, 9 |
| layering → RANDOM | 18, 6, 3, 18, 0 | 2, 2, 1, 3, 4 |
| scatter → SG | 63, 61, 59, 64, 58 | 44, 44, 44, 52, 50 |

Instance CYCLE @50: fuzzy 3/3, 2/3, 2/3, 2/2, 1/2 vs crisp 2/3,
2/3, 1/3, 2/2, 1/2. (The first draft's flat curves measured mostly
clean data — pool bug, review #14.2 — and its "3/3 every level"
was written from rows that had scrolled past. Both corrected.)

The honest shape: NO cliff for either arm, but erosion for fuzzy.
Amount noise sinks truth degrees into the background mass, so the
fuzzy ranking edge decays — gradually for scatter (63→58, sums
average noise out) and round_trip (12→6), steeply for layering
(18→0: long chains multiply per-hop noise). Crisp was never
concentrated, so it has nothing to lose: flat lottery throughout
(occasionally above fuzzy at extreme noise — 4v0 — where both are
at the floor anyway). Gap scaling alone barely matters (truth gaps
are small; ×2 stays in support). Fuzziness wins when calibrated
and degrades gracefully to the lottery floor — the value is in the
low-noise regime, and the report no longer pretends otherwise.

## 6. Discovery agent: loop works, fragments don't pay

`src/motifql/models/discovery.py` + `eval/discover.py`: train-only
ctx (member anchors + 20k bg + current-motif scores), SHOW/TRY/
PROPOSE/DONE text protocol (no tool API in the Spark client),
train gate (≥2 cluster catches, bg ≤ base median 0.095% = 19/20k
with a 2-fire floor), holdout scoring by the harness. 9
stub-transport tests pin parsing (uppercase, fence-aware, positional
id/block — the first live run exposed a line-start bug), gate
arithmetic incl. the floor, fail-closed tools, and the train-only
guard. Exact branch-and-bound scores every TRY (no screen, so no
monotonicity precondition — sound for `not`/`few` too).

Live runs (transcripts in `eval/results/discover_fan_{out,in}_
transcript.md`, gitignored): both FAN clusters explored sensibly —
4 SHOWs + 8 TRYs each. FAN-OUT: all 8 compiling; best recall 3/4
(twice, at 32–45% bg flood); quietest useful TRY 2/4 at 110/20k
(0.55% ≈ 2050/day at train volumes of 372k anchors/day); never
PROPOSEd — its prompt lacked the gate bars (added after), so it
optimized open-ended instead of stopping. FAN-IN (with gate bars):
8 TRYs, 2 compile failures (one E_UNIT self-repaired from feedback,
one E_PLAN); best recall 3/4 (three times, at 12–65% bg), quietest
1/4 at 1.45%; budget exhausted without PROPOSE or DONE. No gate()
call ever ran — "no pass" means the agent never proposed, not that
the gate rejected. Honest wrinkle: the gate holds the specialist
(≤0.095%) stricter than the generalists it complements — fan_out
itself fires 28.6% bg, gather 11.0%, fan_in 2.5%
(`discover_base_scores.parquet`, 20k train bg @emit — the same
file the 19/20k gate median recomputes from). Defensible (narrow
coverage must be cheap) but strict; a union-marginal gate would be
the fairer alternative (follow-up).

Why fragments are (mostly) unrecoverable — probed, not asserted
(`eval/probe_fragments.py`, committed): background senders hit ≥2
distinct receivers per 4d 0.42–0.47 of the time across seeds (the
flood the agent drowned in; the 37% in the first draft was one
draw). Truth multi-fragments ARE unusual vs their own
all-available history (123: 1→5, 138: 1→3, 105: 2→5 — note the
window is ≤6 days of data, not 30), but singletons (223/224/160:
win ≤ hist) carry no count-only signal, and 208 (6→4) runs against
the anomaly story. So: baseline-relative counts (the refits doc's
named lever) would address the multi-fragments (~half the misses);
singletons need amount-conditioned or cross-pattern logic (224
already scores 0.38 gather) — a harder problem, specified not
solved.

Catch matrix (train, @emit over train-completable instances;
recomputes from `discover_base_scores.parquet`): cross-pattern
catching is rampant — gather_scatter catches 6/17 BIPARTITE and
5/9 STACK, round_trip catches 6/9 STACK and the holdout SG
instance. SG has 0/7 missed —
but "covered" by generalists firing 2.5–28.6% bg, which is exactly
why a cheap specialist was wanted and not found. Structural misses
stand: 128 (disorder), layering-<5, BIPARTITE (no 2×2), STACK (no
v0 form).

## 7. System level: combine ranks, not raw scores

Per-motif RQ2 (§3) is the experiment's unit, but deployment combines
motifs. Naive max-of-raw-scores over 7 motifs (all but the
truth-less `structuring_fan_in` demo, whose `most` also breaks crisp
binarity) COLLAPSES: 0/1495 @50 both arms. Diagnosis: fan background
groups score 1.0 on counts and bury path truth at 0.5-0.9 — motif
scores are uncalibrated across motifs, for fuzzy and crisp alike.
This is a deployment finding, not an RQ2 finding.

The standard fix rescues it: min-rank union (per-motif
`rank_per_day`, alert by best rank). ALL_FLAGGED transfer recall
(`/tmp/unionrank.py`, `/tmp/unionrank3.py` over saved scores, n=1495):

| budget | 3-motif fuzzy/crisp | 7-motif fuzzy/crisp |
|---|---|---|
| 50 | 86 / 30 | 46 / 12 |
| 100 | 131 / 55 | 76 / 26 |
| 500 | 180 / 161 | 177 / 125 |
| 5000 | 191 / 191 | 220 / 214 |

System-level fuzzy wins 2.4–3.8x at budgets 50–100 under both
unions; @5000 is converged support (the ±6 residuals are
tie-lottery noise). Dilution is real but bounded: adding 4 noisier
motifs halves absolute hits (86→46) while preserving the
fuzzy/crisp ratio — rank combination contains what raw max cannot.
Calibration across motifs (rank combination at minimum) is now a
documented Nov-2+ requirement, not an open question.

## 8. Value verdict

Limited, real, and held to its bounds:

1. **Fuzzy ranking beats exact at operational budgets** — 3/3
   motifs on HI holdout (§3), 2.4–3.8x at system level via min-rank
   union (§7), replicated 3/3 on LI (§9). Mechanism understood:
   identical candidate sets, better order; exact lotteries among
   emitters. Small-margin cells sit in lottery noise (stated, §3).
2. **The edge erodes gracefully under evasion** (§5) — no cliff
   for either arm, but fuzzy decays as truth scores sink (gently
   for scatter/round_trip, steeply for long chains: 18→0), while
   crisp idles at its lottery floor. Fuzziness wins when calibrated;
   the value is in the low-noise regime.
3. **Discovery loop works; fan fragments don't pay** — agent
   explores and self-repairs competently (§6), but precision-walled:
   0.42–0.47 of background senders hit ≥2 distinct receivers/4d,
   and singletons carry no count-only signal (probed).
   Specified path: baseline-relative counts for multi-fragments
   (~half the misses); amount-conditioned or cross-pattern logic
   for the rest. Deliberately NOT built here: the win would cover
   a handful of train instances at full-extension cost; the
   probe-backed diagnosis is the deliverable.

Bounds: thin holdout (29 completable instances; layering holdout
vacuous at instance level — its transfer win is on partial members);
simulator-fitted supports (§4); calibration required for deployment
(§7); evasion past ±20% amounts erases the edge (§5).

## 9. LI-small replication: same direction, 3/3 motifs

`eval/rq2.py --data data/li-small` (1.83M holdout anchors; exact
fuzzy (§2), zero differential mismatches, verifies agree; `eval/
results/rq2-li-small/`). Transfer recall on target patterns
(fuzzy / crisp hits):

| motif → pattern | @50 | @100 | @500 | @5000 |
|---|---|---|---|---|
| round_trip → CYCLE (29) | 3 / 2 | 3 / 3 | 3 / 3 | 3 / 3 |
| layering → RANDOM (28) | 9 / 2 | 9 / 3 | 9 / 7 | 9 / 9 |
| scatter → SG (65) | 33 / 27 | 34 / 30 | 34 / 34 | 34 / 34 |

Instance level is too thin to discriminate (1-3 completable per
pattern, mostly 0/0 both arms). Combined with HI: 6/6
motif-dataset cells favor fuzzy at budget 50, all converging at
high budgets (LI numbers are unchanged from the pre-fix run — the
old truncated fuzzy happened to rank the same hits). The 3/2
round_trip cell sits in lottery noise (§3 caveat applies here
too). LI has longer patterns and 1.8% layering truncation vs HI's
0.9%, yet the edge survives — the ranking effect generalizes
across simulator variants.
