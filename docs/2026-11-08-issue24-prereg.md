# Nov 8 — Issue #24 pre-registration: learned fuzzy calibration

Pre-registered search space, budget, seed and protocol for fitting
motif membership-function parameters on train-only data. Committed
before the first fit; the fit itself must cite this doc and report
every deviation from it.

## 1. Research question

Can membership-function parameters be fit on train-only data to
maximize budget-recall at operational alert budgets (50–100/day), and
do the fitted functions transfer HI-Small → LI-Small?

## 2. Search space (SPACE_VERSION 1, `src/motifql/calibrate.py`)

Start with `round_trip` → CYCLE (cleanest signal: 8/9 train at
ceiling, mechanism understood). Structure frozen — match, where,
having, quantifier choice and term shapes never move; only numbers:

| Param | Meaning | Bounds | Hand-tuned |
|---|---|---|---|
| `gap_plateau_h` | `patient` trapezoid shoulder c (hours) | [6, 72] | 24 |
| `gap_support_h` | `patient` trapezoid support d (hours) | [max(plateau, 48), 168] | 120 |
| `tol_pct` | `~=` amount tolerance (percent) | [5, 50] | 20 |
| `score` | t-norm | {min, product} | min |
| `emit` | emit threshold (fires/day only; ranking uses top_score) | [0.3, 0.7] | 0.5 |

Fixed with rationale (deviations from the issue's ~10-param sketch):

- Trapezoid a/b stay 0: an instant hop is fully patient by definition;
  fitting onset would only express "fast hops are suspicious".
- Gap support capped at 168h = the hard `span(e) <= 7d` window: a
  shoulder beyond the window could never fire.
- No quantifier-ramp params: `round_trip` uses `each` (no ramp to fit).
  The ramp slots exist on the group-motif spaces instead
  (`scatter_gather`: `several` lo/hi + sum tolerance + score + emit).
- No absolute-money tolerance: amounts span orders of magnitude, so
  percent is the right shape; swapping node types would be structural.
- `layering_chain` and `scatter_gather` spaces are defined in code but
  frozen for follow-up runs — this registration covers `round_trip`
  only.

Shared terms are copied, not mutated: the fit introduces a private
`patient_fit_rt` term and repoints `round_trip` at it, so
`layering_chain`'s scores are bit-identical before and after.

## 3. Objective and protocol

- **Eval set (search):** all train CYCLE member anchors (last-member
  completeness is NOT required — transfer hits, like RQ2) plus a
  seeded 20,000-anchor background sample (`BG_N` precedent from
  `eval/discover.py` / `eval/perturb.py`).
- **Replay:** exact-floor branch-and-bound (`EXACT_FLOOR`, the RQ2
  fuzzy arm) — top_scores exact mod 1e-12, no unpruned truncation.
- **Objective:** train CYCLE transfer hits @50/day + hits @100/day on
  the sampled eval set with per-day sample-rate-scaled budgets.
- **Regularization:** exact objective ties break toward wider (looser)
  supports; the full trace (params → objective) is committed, not just
  the winner.
- **Confirm:** winner + hand-tuned baseline replayed on ALL train
  anchors with unscaled budgets (50/100/500) and exact fires/day.
- **Baselines:** hand-tuned reference, hand-tuned + emit sweep
  {0.3, 0.4, 0.5, 0.6, 0.7} (no replays — exact top_scores make
  found(top ≥ t) exact).

## 4. Budget and seed

- **30 tries**, **seed 24** (separate RNG streams for the background
  sample and the parameter draws, so the eval set does not shift with
  the budget). One seed, one run — no seed hacking; a second seed is a
  new pre-registration.
- Fit on train only (`eval/splits.py` boundaries). Report once on
  sealed holdout + LI-Small (`--mode report`, no search): per-motif
  transfer + instance recall at 50/100/500 with fires/day, fitted
  params, and the transfer gap (train-fit gain vs holdout/LI gain).

## 5. Verdict rules

- Transfer holds iff the fitted params beat hand-tuned on holdout
  CYCLE transfer hits @50–100/day at comparable fires/day AND the
  LI-Small report agrees in sign. Otherwise the result is negative.
- A negative result (fits don't transfer) is publishable and must be
  reported as such; the README claim ("you don't have to guess the
  threshold") is updated only if transfer holds.
- Deliverables: this doc, `eval/results/calibrate_round_trip/` (run
  config, full trace, fitted motif text, train confirm, holdout + LI
  reports), and a dated report section with the transfer verdict.

## 6. Status

- [ ] Fit run (seed 24, 30 tries) on HI-Small train
- [ ] Train confirm (winner + baseline, full train)
- [ ] Sealed holdout report (once)
- [ ] LI-Small report (once)
- [ ] Transfer verdict + dated report section
