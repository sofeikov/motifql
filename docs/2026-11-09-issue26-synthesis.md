# Nov 9 — Issue #26: discovery as program-synthesis benchmark

Three proposers (LLM-guided, enumerative, hybrid) on one missed
cluster, one gate, train-only. Verdict: **the bottleneck is the
language, not the LLM and not the search budget** — all three arms
independently find the same difficulty ladder and all three stop
5.8× short of the gate. No proposal gated, so no candidate motif is
committed and the holdout probes below are post-hoc diagnostics, not
gate passes.

## 1. Pre-registration (fixed before the arm runs)

- **Cluster:** FAN-OUT train misses `[123, 138, 223, 224]`
  (`discover_misses.csv`; n = 4/2/1/1 members). FAN-OUT over STACK
  because fan templates are enumerable — STACK has no v0 form, so a
  template grid on it would be vacuous and the LLM-vs-search
  comparison unfair. Recomputed in-worktree before the runs; the
  miss table matches the main-repo file exactly.
- **Gate (primary: absolute):** need ≥ 2/4 catches, bg ≤ base median
  0.095% (19/20k; recomputed 0.00095, same as §6 of the Nov 4
  report). Marginal bg fire reported on every TRY; marginal-gate
  re-analysis pre-registered for all frontiers.
- **Arm (a) LLM-guided:** `eval/discover.py --run-agent`, max_steps=12
  (≈4 SHOW + 8 TRY, same as the Nov 4 runs), gate bars in prompt.
- **Arm (b) enumerative:** 384-TRY fan grid (windows × having ×
  count-term × amount-filter × emit = 4×3×4×4×2), exact-floor
  scoring, no model cost. Two-stage: coarse scan on a seeded 2k bg
  subsample, exact re-score of a generous per-recall net (≤3× the
  level best) on the full 20k sample. Only exact numbers reported.
- **Arm (c) hybrid:** top-2 arm-(a) TRYs by (recall, quietness) —
  selection rule pre-registered, structures copied verbatim — ×
  numeric grids (S1: 54, S2: 18 = 72 TRYs), same two-stage protocol.
- **Metrics:** Pareto frontier (train recall vs bg fires/day at train
  volume 372,460 anchors/day) per arm; #TRYs to first gate pass in
  grid/transcript order; holdout scoring of gated proposals only.

## 2. Instrument fix (shipped first)

- **Union gain** (`src/motifql/models/discovery.py`: `union_caught`,
  `newly_caught`, `gain_vs_union`; wired into `eval/discover.py`
  holdout scoring): newly-caught completable instances = proposal
  catches minus the union of current-motif catches. The holdout
  filter is split == "holdout", i.e. the completable set, so no new
  plumbing was needed. The old max-based gain (caught(proposal) −
  max caught(motif)) is **superseded**: it scores a complementary
  specialist ≤ 0 and an overlapping re-catch > 0 (docstrings +
  §10 of the Nov 5 report marked accordingly).
- **Union-marginal bg gate** (`gate(..., bg_mode=...)`,
  `--bg-mode` on the driver, threaded through `run_agent`): charges
  only proposal fires on background no current motif fires on, same
  allowance. Default stays `absolute`; unknown modes raise.
  `try_motif` reports the marginal line whenever motifs are passed.
- **Tests:** 4 new tests in `tests/test_discovery.py` (union gain in
  both bias directions, marginal counting, marginal-pass/
  absolute-fail specialist, bg_mode threading). 13/13 green.

## 3. Results

Per-arm best bg fires by recall (exact, full 20k bg; gate needs 2/4
at ≤ 19):

| arm | TRYs | 1/4 | 2/4 | 3/4 | 4/4 | gate pass |
|---|---|---|---|---|---|---|
| (a) LLM | 6 | 91 | 1433 | 729 | 6947 | 0/6, either mode |
| (b) enum | 384 + 168 exact | 16 | 110 | 8390 | 16543 | 0/384, either mode |
| (c) hybrid | 72 + 57 exact | 63 | 394 | 609 | — | 0/72, either mode |

Combined Pareto (distinct points): 0/4@0 (llm), 1/4@16 (enum),
2/4@110 (enum, 2049/day), 3/4@609 (hybrid, 11,338/day), 4/4@6947
(llm). Plot: `eval/issue26/pareto.svg`; data: `frontier_all.csv`.

What each arm did:

- **LLM (transcript `arm_a/discover_fan_out_transcript.md`):** 4 SHOWs,
  6 TRYs across four families (small-count fan, scatter copy, two
  amount-conditioned fans, butterfly), one malformed TRY (fence
  before keyword — the pinned parser correctly rejected it), then a
  correct DONE. All 6 TRYs re-scored independently by
  `eval/discover_combine.py` reproduce the transcript exactly.
  Cost: 12 model calls, ≈52k prompt / 11.4k completion tokens.
- **Enum:** 384/384 compiled. Best 2/4 (`e226`: 4d, having ≥ 2,
  several, min ≥ 5k, t0.5) is the Nov 4 run's best LLM TRY (step 5,
  also 2/4 @110) modulo term names — independent rediscovery of
  the same in-family optimum.
- **Hybrid:** S1 tuning improves the LLM verbatim 3/4@729 → 3/4@609
  (tighter count trapezoid); S2 tuning improves 1/4@91 → 1/4@63 but
  no butterfly param reaches recall 2 (all 18 S2 rows catch exactly
  one instance).

## 4. Verdict: blocked on language, with receipts

1. **Not the LLM.** It explored four motif families in 6 TRYs,
   invented amount-conditioned filters the template grid lacked
   (`most r: amount is big` reaches 3/4 at 729 — 11.5× quieter than
   enum's best 3/4 at 8390), found the quietest point in the study
   (butterfly 1/4 @91, marginal 19/20k = exactly the bg bar), and
   stopped correctly. Its failure is not for lack of trying shapes.
2. **Not the search budget.** 384 TRYs dominate recall 1–2
   (enum's 2/4@110 is 13× quieter than the LLM's 2/4@1433) and
   converge with the prior LLM run on the identical optimum —
   the in-family ceiling is found, and it still misses the gate 5.8×
   (110 vs 19 fires; marginal 101 vs 19).
3. **The language.** All three arms independently recover the same
   strict difficulty ladder — 138 → 123 → 224 → 223 (every 2/4 is
   {123,138}, every 3/4 is {123,138,224}) — and the cost explodes at
   the singletons: 2/4@110 → 3/4@609 → 4/4@6947. The missing
   primitives are the named ones from the Nov 1 tuning report §9:
   **baseline-relative counts** (123: 1→5 and 138: 1→3 vs own
   history per `probe_fragments.py`, but absolute counts drown in
   the 0.42–0.47 background flood) for the multi-fragments, and
   graded having / amount-conditioned logic for the singletons
   (223 catches only at flood: cheapest 4/4 is 6947/20k).
   The marginal gate doesn't change the outcome here — the winners
   fire where generalists don't (110 abs / 101 marg), so there is
   almost no overlap to forgive. The minimal unblocking primitive
   is per-anchor trailing-rate counts; singletons stay open.

## 5. Holdout (post-hoc diagnostic, no gate pass)

No proposal gated, so per the preregistration there is no holdout
scoring of a proposal. As a labeled post-hoc check on the n=4
overfit question, the two train-Pareto winners were scored once each
on holdout FAN-OUT (2 instances, both singletons — 311, 317) with
`eval/score_holdout26.py` (selection on train only, nothing tuned):

| train winner | train | holdout | holdout bg (marg) | gain |
|---|---|---|---|---|
| enum 2/4@110 | 2/4 | 0/2 | 0.0046 (0.0044) | +0 |
| hybrid 3/4@609 | 3/4 | 0/2 | 0.0199 (0.0182) | +0 |

Background rates roughly transfer (0.55%→0.46%, 3.0%→2.0%); recall
does not — the holdout fragments are structurally outside any fan
motif (consistent with the Nov 1 tuning §3 verdict). For reference,
round_trip cross-catches 1/2 holdout FAN-OUT at 0.0001 bg.

## 6. Files and reproduction

- Code: `src/motifql/models/discovery.py` (gain + gate),
  `eval/discover.py` (--bg-mode, union gain),
  `eval/discover_grid.py` (shared two-stage harness),
  `eval/discover_enum.py`, `eval/discover_hybrid.py`,
  `eval/discover_combine.py`, `eval/score_holdout26.py`.
- Data (all committed under `eval/issue26/`): arm transcript +
  miss table, per-arm coarse/exact CSVs, `frontier_all.csv`,
  `pareto.svg`, winner motifs + holdout probe CSVs.
- Reproduce (needs `MUSE_KEY` for arm (a) only):
  `uv run eval/discover.py --run-agent --pattern FAN-OUT
  --out eval/issue26/arm_a` →
  `uv run eval/discover_enum.py` →
  `uv run eval/discover_hybrid.py` →
  `uv run eval/discover_combine.py`.
