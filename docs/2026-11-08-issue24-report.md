# Nov 8 — Issue #24 report: fits don't beat hand-tuned (negative result)

Registered fit + sealed reports for learned fuzzy calibration on
`round_trip` → CYCLE, per the [pre-registration](2026-11-08-issue24-prereg.md)
(as amended in §7) and the [synthetic validation](2026-11-08-issue24-synthetic-validation.md).
Verdict: **negative** — 30 seeded tries found nothing that beats
hand-tuned on train, and both sealed sets agree in sign. The README
claim is unchanged.

## 1. Registered fit (HI-Small train, seed 24, 30 tries, precision)

Constrained-precision objective on all 1.86M train anchors: fewest
background fires/day at the fitted emit subject to keeping every
baseline-caught CYCLE instance caught.

- **Baseline** (hand-tuned): 8/9 caught (128 structural at 0.0),
  146 bg fires (29.2/day). Emit sweep on default scores: 0.3→229,
  0.4→190, **0.5→146**, 0.6→108 (only 4 caught), 0.7→65 (3 caught) —
  the 0.5 bar is 146 fires.
- **Winner** (try 18, feasible): plateau 29.7h, support 126.9h,
  tol 20.7%, min, **emit 0.39** → 8/9 caught, 212 fires (42.4/day).
- Only 5/30 tries kept 8/9 (caught distribution: 1×1, 5×2, 2×3, 4×4,
  4×5, 7×6, 2×7, 5×8); all five feasible vectors fire more than
  baseline (212–479). Nothing dominates hand-tuned — it sits on the
  empirical Pareto frontier.

The winner is essentially the default *shape* with a lower emit
(0.39 vs 0.5); among feasible vectors fires track emit, not the
membership numbers. The frontier is sharp: tightening to
{12, 80, 12} drops 8/9→3/9, loosening doubles fires at fixed catches.

## 2. Sealed reports (once each, frozen params, no selection)

| Split | Key | CYCLE transfer hits@50/100/500 | CYCLE instances | Fires/day |
|---|---|---|---|---|
| HI holdout | hand-tuned | 12/14/15 (of 92) | 3/3 | 37.7 |
| HI holdout | fitted | 13/14/15 | 3/3 | 53.3 (+41%) |
| LI holdout | hand-tuned | 3/3/3 (of 29) | 1/1 | 29.0 |
| LI holdout | fitted | 3/3/3 | 1/1 | 42.3 (+46%) |

Full catches everywhere at every budget for both keys; the fitted
vector's +1 transfer hit on HI@50 costs +41% fires/day, and LI
(with identical hits) confirms the ordering. Transfer gap: none —
the train ordering (baseline ahead) is preserved on both sealed sets.

## 3. Verdict

Per prereg §7: transfer holds iff fitted beats the swept baseline on
holdout fires/day at full catches with LI agreeing in sign. It loses
on both. **Negative result**: uniform random search over this space
does not improve on the hand-tuned `round_trip`, which appears
at/near optimal for train precision at fixed recall.

Honest summary: the calibration machinery is validated on synthetic
data (slope, planted-parameter recovery, within-family transfer),
but no real dataset at hand has headroom left to demonstrate a win
on — the shipped defaults sit on the empirical frontier and transfer
across splits and datasets as-is. What this work delivers is not a
better number but the fitting loop itself: the defaults can now be
retuned on new data instead of hand-tweaked.

Honest limits of this negative:

- 30 uniform tries over 5 dimensions sample the neighborhood of the
  (good) default thinly. A local search around default (new
  amendment + new seed) could still shave fires — but the frontier
  shape says to expect little.
- Recall-side calibration was already dead (train at ceiling, §7
  amendment); this extends the negative to the precision side for
  `round_trip` only. `layering_chain`/`scatter_gather` spaces are
  implemented but unfitted — the same ceiling concern applies there
  (all motifs at/near train ceiling per Nov 1–2), so any follow-up
  should go straight to precision-side.
- 9/3/1 instances: tiny-n throughout; every count is ±1–2 of noise.
  The mechanism (frontier shape + emit dominance) carries more weight
  than any single count.

## 4. Artifacts

- `eval/results/calibrate_round_trip_prec/` (gitignored): run.json,
  full trace.csv (31 rows), fitted.json, fitted motif text,
  confirm_train_summary.csv, report_holdout_{transfer,instance}.csv.
- `eval/results/calibrate_round_trip_prec_li/`: LI sealed reports.
- Code fix during the run: the truth-margin tie-break ran over all
  instances, so 128's structural zero pinned every margin at −emit.
  Fixed to min-over-caught (`eval/calibrate.py`, covered by
  `tests/test_calibrate.py`). Selection unaffected — no fires ties
  among feasible vectors, winner try 18 stands.
