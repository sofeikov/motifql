# Nov 1 — RQ2 (fuzzy vs exact) and the discovery agent

DESIGN.md §9 Nov 1 row: the discovery agent (§7.2) plus RQ2
fuzzy-against-exact and perturbation runs. Exit check: the RQ2 chart.

## 1. The exact arm: support-boundary crisp flag

"Every soft condition made exact" is implemented as a scoring flag,
not text rewrites: with `crisp=True` every soft leaf (term use, `~=`
tolerance) is thresholded at its support boundary (degree > 0 → 1,
else 0). Combination (`and`/`or`/`not`/quantifiers) is unchanged —
for `most`/`few`/`@name` the ramp still grades the crisped mean (a
ramp has two breakpoints, so no single exact analog exists; this
choice also favors the exact arm, making any fuzzy win conservative).
Threaded through `score_any`/`matcher.score`/all four group
executors/butterfly (`src/motifql/degrees.py:crisp_degree`).

The differential `crisp_top == (fuzzy_top > 0)` holds anchor-wise
wherever both searches complete — asserted by `eval/rq2.py` on every
run (6 unit tests in `tests/test_crisp.py` pin the rule, pruning
soundness, and the `most` behavior).

## 2. Two-phase replay: the exact arm screens for the fuzzy arm

Unpruned `*5..8` layering costs ~10ms/anchor and truncates ~5% of
anchors (50k cap) — a full fuzzy holdout replay is ~1h and
truncation-ridden. The screen is exact: crisp completed + 0 ⟹ no
all-positive-leaf witness exists ⟹ fuzzy top is 0 without replaying.
So: crisp pruned over the full set, fuzzy unpruned only on
crisp-positive or crisp-truncated anchors (281/1.35M for round_trip,
~12k for layering). Pruned crisp is a SUPERSET of unpruned crisp
(pruning explores fewer branches, truncates less) — verified on a
20k sample per motif per run, plus a fuzzy-shortcut agreement check.

Side finding: pruning is not just faster but MORE COMPLETE under
truncation — full unpruned RQ1-style replays of deep paths miss
witnesses the pruned run finds (2 such anchors on the 20k smoke
sample). The Oct 30 §8 numbers predate this understanding.

## 3. Level-0: fuzzy wins at tight budgets (HI-Small holdout)

`eval/rq2.py`: holdout-only replay (1.35M anchors), shared
`evaluate.py` ranking at 50/100/500/1000/5000 alerts/day.
`eval/results/rq2_{summary,recall_transfer,recall_instance}.csv`,
`rq2_holdout_scores.parquet`.

Mechanism: crisp scores are binary, so the exact arm holds a lottery
among emitters (945 layering fires/day); fuzzy ranks the same
candidate set by degree. Transfer recall on target patterns:

| motif → pattern | budget 50 | budget 100 | budget 500+ |
|---|---|---|---|
| round_trip → CYCLE (92) | fuzzy 12, crisp 7 | 14 / 13 | 15 / 15 |
| layering → RANDOM (56) | fuzzy 19, crisp 2 | 19 / 9 | 19 / 19 |
| scatter → SG (170) | fuzzy 63, crisp 44 | 65 / 58 | 66 / 66 |

Instance recall: round_trip→CYCLE 3/3 fuzzy vs 2/3 crisp at budget
50 (3/3 both above). Layering→RANDOM 0/6 both arms — the known
holdout vacuity (no n≥5 RANDOM instances on holdout). Scatter→SG 0/1
both arms (the one completable instance's members don't rank; its
transfers are caught cross-pattern — see §6).

Convergence at high budgets is expected: same support modulo
truncation. The value is RANKING within the candidate set at
operational budgets, not a larger positive set.

## 4. Threshold perturbation: the wrong metric (kept as a negative)

`eval/perturb.py` threshold mode: perturb ONLY target-pattern member
transfers (amount ×(1+U(-p,p)), per-instance gap scaling
t' = t_min + f·(t−t_min), order-preserving by construction),
train+holdout pooled for n (no fitting → no leakage), replay truth +
20k fixed background, threshold recall at emit. 4 unit tests
(`tests/test_perturb.py`): level-0 identity, determinism, order
preservation, day recomputation.

Result: crisp ≥ fuzzy at EVERY level (e.g. layering transfer 0.271
flat vs fuzzy 0.25→0.08; round_trip instance 11/12→9/12 crisp vs
11/12→6/12 fuzzy). This is structural, not a finding about
robustness: {fuzzy ≥ 0.5} ⊆ {crisp = 1} always, so threshold recall
can NEVER show fuzzy winning — it measures emit propensity, and the
Nov 1 tuning widened supports (20-25%, patient gaps) until
perturbations ±20%/×2 barely exit them. The honest evasion test is
budget-recall under noise (§5), which reuses the §3 table as level 0.

## 5. Budget recall under noise: the edge persists

`eval/perturb.py --budget`: full holdout replay + ranking per level
at (0,×1), (10%,×1), (20%,×1), (0,×2), (20%,×2)
(`rq2_perturb_budget_{transfer,instance}.csv`). Level 0 reproduces §3
EXACTLY (0 mismatches both granularities) through the independent
perturbed-frame pipeline — a strong end-to-end check.

Transfer recall @50 on target patterns, by level
[(0,×1), (10%,×1), (20%,×1), (0,×2), (20%,×2)]:

| motif → pattern | fuzzy | crisp |
|---|---|---|
| round_trip → CYCLE | 12, 12, 13, 12, 14 | 7, 7, 7, 12, 11 |
| layering → RANDOM | 19, 19, 19, 25, 21 | 2, 2, 2, 4, 4 |
| scatter → SG | 63, 63, 63, 71, 70 | 44, 44, 44, 49, 49 |

Instance CYCLE @50: fuzzy 3/3 every level, crisp 2/3 every level.
Gap-×2 denominators grow (train members drift into holdout days);
rates stay flat. The predicted "cliff" never appears for EITHER arm
— supports absorb ±20%/×2 — but the fuzzy ranking edge persists at
every level. Mechanism: background is sparse (mostly exact 0), so
perturbed truth still outranks it; ranking needs relative order,
not absolute scores. Robustness = edge persistence, not
cliff-vs-graceful.

## 6. Discovery agent: loop works, fragments don't pay

`src/motifql/models/discovery.py` + `eval/discover.py`: train-only
ctx (member anchors + 20k bg + current-motif screens), SHOW/TRY/
PROPOSE/DONE text protocol (no tool API in the Spark client),
train gate (≥2 cluster catches, bg ≤ base median with a 2-fire
floor), holdout scoring by the harness. 6 stub-transport tests pin
parsing (incl. live-model shapes: glued keywords, RATIONALE traps),
gate arithmetic, and the train-only guard.

Live runs: both FAN clusters explored sensibly (4 SHOWs, 7 TRYs each,
all compiling; one E_UNIT self-repaired from feedback). Best FAN-OUT
TRY: 2/4 at 0.55% bg fire (110/20k ≈ 2750/day — gate needs
≤~0.03%). No PROPOSE passed; the first run also exposed a
line-start parser bug (fixed) and missing gate text in the prompt
(fixed: the agent now gets the exact bars).

Why fragments are (mostly) unrecoverable — probed, not asserted
(`/tmp/probe_frag2.py`): background senders hit ≥2 distinct
receivers in 4d 37% of the time (the flood the agent drowned in);
truth multi-fragments ARE anomalous vs own history (win4d 3-5 vs
hist30d 1-2) but singletons (223/224/160: win ≤ hist) are
unrecoverable by ANY count logic. Baseline-relative counts (the
refits doc's named lever) would address ~half the misses; the rest
need cross-pattern paths (224 already scores 0.38 gather).

Catch matrix (train, @emit): cross-pattern catching is rampant —
gather_scatter catches 6/17 BIPARTITE and 5/9 STACK, round_trip
catches 6/9 STACK and the holdout SG instance. SG has 0/7 missed
(the "2-cycle orphans" are covered by fans+gather). Structural
misses stand: 128 (disorder), layering-<5, BIPARTITE (no 2×2),
STACK (no v0 form).

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

| budget | 3-motif fuzzy | 3-motif crisp | 7-motif fuzzy | 7-motif crisp |
|---|---|---|---|---|
| 50 | 87 | 30 | 46 | 12 |
| 100 | 129 | 55 | 78 | 26 |
| 500 | 175 | 161 | 173 | 125 |
| 5000 | 186 | 191 | 215 | 214 |

System-level fuzzy wins ~3-4x at budgets 50-100 under both unions;
@5000 differences are tie-lottery noise (converged support).
Dilution is real but bounded: adding 4 noisier motifs halves
absolute hits (87→46) while preserving the fuzzy/crisp ratio —
rank combination contains what raw max cannot. Calibration across
motifs (rank combination at minimum) is now a documented Nov-2+
requirement, not an open question.

## 8. Value verdict

Limited, real, and held to its bounds:

1. **Fuzzy ranking beats exact at operational budgets** — 3/3
   motifs on HI holdout (§3), ~4x at system level via min-rank union
   (§7). Mechanism understood: same candidate set, better order;
   exact lotteries among emitters.
2. **The edge survives evasion noise** — flat budget curves through
   ±20% amounts and ×2 gaps (§5). Not the predicted cliff shape:
   supports absorb the noise, background stays sparse.
3. **Discovery loop works; fan fragments don't pay** — agent
   explores and self-repairs competently (§6), but precision-walled:
   37% of background senders hit ≥2 distinct receivers/4d, and
   singleton members are unrecoverable by any count logic (probed).
   Specified path: baseline-relative counts for multi-fragments
   (would address ~half the misses); cross-pattern paths for the
   rest. Deliberately NOT built here: the win would cover ~4 train
   instances at full-extension cost; the probe-backed diagnosis is
   the deliverable.

Bounds: thin holdout (29 completable instances; layering holdout
vacuous at instance level — its transfer win is on partial members);
simulator-fitted supports (threshold-perturbation lesson, §4);
calibration required for deployment (§7).

## 9. LI-small replication: same direction, 3/3 motifs

`eval/rq2.py --data data/li-small` (1.83M holdout anchors; differentials
clean, verifies agree). Transfer recall on target patterns:

| motif → pattern | @50 fuzzy/crisp | @100 | @500 |
|---|---|---|---|
| round_trip → CYCLE (29) | 3 / 2 | 3 / 3 | 3 / 3 |
| layering → RANDOM (28) | 9 / 2 | 9 / 3 | 9 / 7 |
| scatter → SG (65) | 33 / 27 | 34 / 30 | 34 / 34 |

Instance level is too thin to discriminate (1-3 completable per
pattern, mostly 0/0 both arms). Combined with HI: 6/6
motif-dataset cells favor fuzzy at budget 50, all converging at
high budgets. The ranking edge generalizes across simulator
variants (LI has longer patterns and 1.8% layering truncation vs
HI's 0.9%; the screen handled 34k/1.83M fuzzy replays).

Cost note: LI took ~75 min on 6 workers (1.83M anchors, longer
chains, 20k-anchor verify replays per motif). The screen replayed
34k/1.83M anchors for layering fuzzy (vs 12k/1.35M on HI) —
without it the unpruned deep-path replay dominates everything.
