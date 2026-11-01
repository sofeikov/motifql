# Nov 9 — Issue #25: evasion robustness as theorem + metric

RQ2's shape is "fuzzy erodes gracefully, exact idles at a lottery floor"
(`docs/2026-11-04-rq2-discovery.md` §5). This report formalizes the
mechanism (§1: a per-witness Lipschitz bound with an adversarial-cost
corollary), defines the rank-level metrics that verify it (§2), and
pre-registers the experiment (§3). The verdict template (§4) is filled
after the runs. No engine/semantics change: analysis + `eval/` code only.

## 1. The bound

Scope: the three RQ2 motifs (`round_trip`, `layering_chain`,
`scatter_gather`) under value-only perturbation of a fixed witness —
amounts jittered, timestamps scaled, witness/group membership held
fixed. Every constant below is pinned by a test in
`tests/test_lipschitz.py` (tightness cases included); the proof sketches
are short enough to check by hand.

### 1.1 Leaves

| Leaf | Bound | Domain / note |
|---|---|---|
| `trapezoid(a,b,c,d)`, strict | L = max(1/(b−a), 1/(d−c)) | piecewise linear, continuous |
| `trapezoid` degenerate (b=a or d=c) | same L **per side** | step discontinuity at the shoulder (`f(1.99)=0, f(2.0)=1` for `(2,2,4,8)`); RQ2 uses sit on one side (gaps/spans ≥ 0, shoulders at 0) |
| `triangle` | via `trapezoid` | same treatment |
| `ramp(a,b)` / `ramp_down(a,b)` | L = 1/(b−a) | `several` 1/2, `many` 1/5, `most`-ramp 1/0.3, `few` 5, `mostly` 1/0.3 |
| `x ~= y within w` (absolute) | \|Δ\| ≤ (\|Δx\|+\|Δy\|)/w | joint ℓ1; `max(0,·)` cap is 1-Lipschitz |
| `x ~= y within f` (%) | \|Δ\| ≤ (\|Δlog x\|+\|Δlog y\|)/f | amounts ≥ m > 0. Proof: h = \|x−y\|/max(x,y) depends only on the ratio r; \|dφ/d(log r)\| ≤ 1 on both branches (1/r above 1, r below), φ continuous at r = 1, cap 1-Lipschitz |
| comparisons / `time_ordered` in `soft` | — | discontinuous 0/1: **out of scope** (absent from the 3 motifs' soft clauses) |

Two singularities, both pinned as documented boundaries: the `%` form
is discontinuous at the origin (`f(0,0)=1` but `f(ε,0)=0`), hence the
`m > 0` floor (true of money transfers); degenerate trapezoid
shoulders are steps, hence the one-sided domain. The log form for `%`
is load-bearing: an absolute bound would scale with the amount level
and go vacuous on large transfers, while evasion noise is relative.

For `patient` on gaps: L = 1/(120h−24h) = 1/345600 per second
(1/96 per hour), falling slope only.

### 1.2 Composition (all on [0,1] degrees)

- `not`: isometry. `and`-min / `or`-max, `each`/`all` (min),
  `some` (max), `at least N` (N-th largest), `all but K`
  ((K+1)-th smallest): 1-Lipschitz in sup norm —
  |Δout| ≤ maxᵢ|Δinᵢ|. Order statistics inherit it because sorting
  is 1-Lipschitz (|x₍ₖ₎−y₍ₖ₎| ≤ max|xᵢ−yᵢ|).
- `and`-product: |Δout| ≤ Σᵢ|Δinᵢ| (telescoping; every partial
  product ≤ 1). Tight to first order.
- `most`/`few`/`@named`: ramp over the mean —
  |Δout| ≤ Σᵢ|Δinᵢ|/(n·width).
- Skipped bindings: the skip set is position-determined (`prev`/`gap`
  undefined by step index), hence fixed under value noise — except a
  division-by-zero in soft arithmetic, which can flip a binding
  defined/undefined (**out of scope**; unused by the 3 motifs).

### 1.3 Per-witness instantiation (fixed witness, value-only noise)

Write λ(p) = max(|log(1+p)|, |log(1−p)|) ≈ p for the log-cost of ±p
multiplicative noise. Amount leaves: Δ ≤ 2λ(p)/f (both sides move).
Gap leaves under ×g scaling: Δ ≤ (g−1)·x/345600s.

- `round_trip`: score = min over ≤ 8 leaves (first step skips).
  Δ ≤ max(Δgaps/345600, 2λ(p)/0.2).
- `layering_chain`: same with f = 0.25, over exactly 5 hops / 8
  leaves — the planner searches open chains without deferred soft at
  `min_hops` (G #15), so the maximum is the same 8 as `round_trip`,
  not 14. The min-fragility gradient survives as a typical-case
  claim: layering always pays all 8 leaves while round_trip witnesses
  are often shorter (2–5 hops). Note the mixed-rung attribution from
  the §5 table: layering holds 18 at (0,×2), falls to 3 at (20%,×1),
  and only the mixed (20%,×2) rung reads 0 — the §4 curves separate
  the axes properly.
- `scatter_gather`: score = C·S. C (`several` on the count) is fixed
  under value-only noise (integral counts, fixed membership); S is
  the `%` leaf on leg sums with f = 0.2. Memberwise ±p scaling keeps
  each sum within ±p (convex combination, amounts positive), so
  Δ ≤ C·2λ(p)/0.2 ≤ 2λ(p)/0.2 — **one** noisy leaf instead of up
  to 8. That is the structural reason scatter erodes gently (63→58)
  while layering collapses (18→3 amount-only, 0 mixed): min-fragility
  over typically more leaves, not per-leaf sensitivity.

Numerically (per-leaf worst case): p = 5% → 0.51 (f = 0.2), 0.41
(f = 0.25); p = 10% → 1.05, vacuous. The bound goes vacuous around
p ≈ f/2 — correctly: anti-correlated ±10% noise truly can zero a
20%-tolerance leaf (§1.4). Expected drop under iid random ±p is
≈ 2p/(3f) (E|u₁−u₂| = 2p/3), about a third of the bound. So the
honest reading: worst-case Lipschitz explains small-noise stability,
while the observed graceful erosion under random noise is
distributional — a strategic evader pays much less than the random
curves suggest. Both numbers are reported; they must not be conflated.

### 1.4 Adversarial evasion cost (corollary)

Anti-correlating the two sides of a `%` leaf at ±p gives
h = 2p/(1+p); the leaf zeroes exactly at **p\* = f/(2−f)**: 11.1%
for `within 20%` (`round_trip`, scatter sums), 14.3% for `within 25%`
(`layering_chain`). One dead amount leaf kills a min-witness, so a
strategic evader needs only ~11–14% hop-to-hop anti-correlated
jitter. Correlated moves (both sides +p) cost nothing — the leaf
reads the ratio — which is exactly why random noise is gentler than
evasion. Pinned in `test_adversarial_amount_cost`, including the
just-below-p\* positive case.

Gap axis: the cheap evasion is the hard span/window cut, not the
soft `patient` slope. Scaling a witness by g ejects it once g·span(W)
exceeds the window — cost g\* = window/span(W) per witness (7d for
round_trip, 10d for layering chains, 7d group window for scatter).
Pinned by `test_span_cut_ejects_scaled_witness` (PR #34 review probe:
a 5-hop 24h-gap cycle scores 0.81 at ×1.75 and 0 at ×1.76). This
corrects the draft's "gap scaling barely matters": the `patient`
leaves themselves barely move, but the window ejects long-span
witnesses discretely — round_trip budget recall already halves (12→6)
at (0,×2) in the §5 table. p\* is the floor a strategic evader pays; the
measured random-noise eviction cost (§2) will systematically exceed
it. The gap between the two is the price of uncoordinated noise.

### 1.5 Amount-axis end-to-end theorem

Under **amount-only** perturbation, the RQ2 motifs' per-anchor
`top_score` inherits the per-witness bound. Proof sketch: every hard
cut (`where`/`having`/match topology/windows) in the three motifs is
amount-free (time-ordered, span, `a != z`, `leg2.time > leg1.time`,
count thresholds over time windows) — so witness/group *sets* are
fixed, and top = max over a fixed set is 1-Lipschitz in the
per-witness scores — assuming neither run truncates. Amount noise can
change the witness set through the expansion cap (a branch scoring 0
and pruned below EXACT_FLOOR can go positive and exhaust the budget),
so the fixed-set step carries the no-truncation hypothesis. Pinned
end to end (`test_amount_only_end_to_end_bound`): worst-case
alternation attains the predicted drop exactly, so the bound is tight
through the whole evaluator, not just per leaf.

Boundaries (where the end-to-end claim stops): (a) gap/time noise can
eject witnesses (span/window cuts) — per-fixed-witness bound only,
`top_score` may cliff; (b) hub-cap truncation can differ discretely
between runs — budget mode logs truncation rates per rung × arm
(perturb.py), RQ2 arms run exact-floor, and RQ3 puts layering
truncation at ~0.9% of anchors (0% elsewhere), so the hypothesis is
checked rung by rung, not assumed blindly;
(c) group window/count jumps under time noise are steps (count leaves
move continuously through ramps *given* membership; membership itself
is discrete). Calibrated widths change every L above — recompute §1.1
for calibrated motifs rather than reusing these constants.

### 1.6 Exact-arm contrast (no bound exists)

`crisp_degree` is a step function: per-leaf Δ ∈ {0,1} for arbitrarily
small threshold crossings — no Lipschitz constant at any value. But
the ranking is not a fresh lottery per rung: the tie-break is a fixed
keyed hash of `transfer_id`, so the order inside the crisp=1 set is
frozen and crisp overlap@k(clean, noisy) measures emitter-set churn
only — expect near 1.0 at small noise, decaying as supports change.
(The "lottery" is across tie-break *choices*, the `--exact-order`
experiment — not across rungs.) Consequence for the verdict:
stability alone can favor crisp (frozen order, nothing to re-rank)
while its ranking carries no signal — pair overlap with recall and
cost, never read it alone. Crisp score drops stay trimodal {−1,0,+1}
against fuzzy's concentrated drops.

## 2. Metrics (`eval/stability.py`, tested in `tests/test_stability.py`)

All metrics run per score column (`motif__arm`), transfer_id-keyed so
gap rungs realign across drift. Rankings reuse `evaluate.rank_per_day`
(same tie-break as RQ2).

- **Rank stability**: day-wise top-k set overlap |top-k ∩ top-k|/k
  between clean and noisy rankings, macro-averaged over holdout days.
  Full Kendall tau was considered and rejected: 1.35M anchors make it
  background-shuffle-dominated and O(n log n) per rung × key, while
  analysts see the top-k — tau would measure what nobody reads.
- **Score-drop distribution**: per-anchor noisy-minus-clean top_score
  on the tid intersection (match counts reported); mean/p50/p90 of
  the signed drop and its absolute value, truth vs background.
- **Evasion cost**: population = truth anchors in the clean top-k
  (per day, per key); cost = first ladder rung ranked out of top-k.
  Reported as a cumulative eviction curve per rung (ALL + per-pattern
  groups) plus per-anchor costs. Anchors that drift out of holdout
  under gap scaling are censored — counted, excluded from fractions.
  Single-axis ladders only: mixed amount+gap rungs cannot order cost.

Recall, stability and cost can disagree (an arm can keep recall while
churning ranks, or rank stably while sinking below emit) — the verdict
reports all three, not a composite. In particular §1.6 predicts crisp
to win overlap@k at small noise by doing nothing — the joint verdict
is the verdict.

## 3. Pre-registration

**Ladders** (single-axis; rung 0 is clean in both):

- amount: (0,×1), (5%,×1), (10%,×1), (20%,×1) — 5% is below p\*≈11%,
  10% at the threshold, 20% above.
- gap: (0,×1), (0,×1.5), (0,×2) — ×1.5 from the threshold ladder, ×2
  established.

**Fixed choices**: motifs `round_trip,layering_chain,scatter_gather`;
arms fuzzy + crisp at α=emit; k ∈ {50,100,500}; seed 12345 (default);
splits recomputed per rung (established protocol); full-holdout replay
+ ranking per rung (existing `--budget` mode).

**Seed sensitivity**: rung (10%,×1) additionally at seeds 999 and 2026
(`--budget-levels 0:1,0.10:1`, overlap rows only). Full 3-seed ladders
are compute-prohibitive — stated as a boundary, not silently skipped.

**Validation gates**: rung 0 reproduces the RQ2 §3 table exactly
(fuzzy arm; the crisp baseline is re-taken from the new rung 0 — the
old `rq2_perturb_budget_*.csv` used the pre-#15 support-boundary arm,
so gating crisp against it would fail on the arm change, not the
data). Rungs (10%,×1), (20%,×1), (0,×2) reproduce the old CSVs' fuzzy
arm exactly (same seed, same levels — 0 mismatches expected,
independent pipeline). Truncation rates are logged per rung × arm
(perturb.py) to check the §1.5 hypothesis rung by rung.

**Commands** (run where AMLworld lives; `data/` is not in the repo):

```bash
# Amount ladder (~4 rung-waves: 3 motifs x 2 arms full-holdout replays each)
uv run eval/perturb.py --data data/hi-small --budget \
  --budget-levels 0:1,0.05:1,0.10:1,0.20:1 \
  --save-scores eval/results/perturb_scores_amount --workers 16
uv run eval/stability.py --data data/hi-small \
  --scores eval/results/perturb_scores_amount \
  --levels 0:1,0.05:1,0.10:1,0.20:1 --out eval/results/stability_amount

# Gap ladder (~3 rung-waves)
uv run eval/perturb.py --data data/hi-small --budget \
  --budget-levels 0:1,0:1.5,0:2 \
  --save-scores eval/results/perturb_scores_gap --workers 16
uv run eval/stability.py --data data/hi-small \
  --scores eval/results/perturb_scores_gap \
  --levels 0:1,0:1.5,0:2 --out eval/results/stability_gap

# Seed check (2 rung-waves, overlap rows only)
for s in 999 2026; do
  uv run eval/perturb.py --data data/hi-small --budget \
    --budget-levels 0:1,0.10:1 --seed $s \
    --save-scores eval/results/perturb_scores_seed$s --workers 16
  uv run eval/stability.py --data data/hi-small \
    --scores eval/results/perturb_scores_seed$s \
    --levels 0:1,0.10:1 --out eval/results/stability_seed$s
done
```

Outputs: `stability_{overlap,drops}.csv`,
`evasion_cost_{curve.csv,anchors.parquet}` per ladder. Total ≈ 9
rung-waves at ~100k anchors/s/core (RQ3 path-motif rates: 1.35M
anchors ≈ 15–25s per motif-arm, minutes per rung at 16 workers) —
about an hour in one sitting, not overnight.

## 4. Verdict (NOT RUN — fill after §3)

Predicted shapes from §1.4/§1.6: fuzzy overlap decays smoothly from
1.0 while crisp stays near 1.0 at small noise (frozen order — churn
only) and decays as emitter sets change; fuzzy drops concentrate near
0, crisp trimodal; the fuzzy eviction curve runs right of crisp
(costlier to push truth out), both at the floor by 20%; amount noise
dominates, gap scaling ejects discretely via span/window cuts (expect
round_trip/layering dips at ×1.5–2 from ejection, not soft decay);
layering's curves fall steepest (always 8 leaves), scatter's
shallowest.

| Axis | Fuzzy | Crisp | Regime where fuzzy wins |
|---|---|---|---|
| recall@budget | | | |
| overlap@k | | | |
| eviction cost | | | |

*Status: theory + metrics code merged; experiment awaits a checkout
with AMLworld data. Update this section with the curves, the
recall/stability/cost comparison, and the noise regime where
fuzziness wins — or correct the predictions above if the data
disagrees.*
