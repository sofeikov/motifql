# Build log (Meta AI tooling)

"Use of Meta AI tooling" is 25% of the hackathon score. This log records
what was built with it, per day. All dates 2026.

## Oct 26 — design, grammar, reference motifs

Author's session (tooling note to confirm): `docs/DESIGN.md`,
`docs/LANGUAGE.md`, `grammar/motifql.lark`, `examples/amlworld.motif`.

## Oct 27 — data prep, eval harness, baselines

Built with Muse Code (agentic coding CLI powered by Meta Muse Spark):

- `scripts/prep_amlworld.py`: AMLworld CSVs into time-sorted Parquet, USD
  conversion, pattern-instance ground truth.
- `eval/features.py`, `eval/splits.py`, `eval/evaluate.py`: trailing-window
  features, instance-to-split assignment, shared ranking/recall protocol.
- `eval/baseline.py` (unsupervised max-CDF) and `eval/supervised.py`
  (LightGBM with/without payment_format).
- `tests/test_baseline_features.py`: committed oracle test for the features.
- `docs/2026-10-27-data-prep.md`: this day's report, including the
  attribution-gap and shortcut-policy findings.
- PR review triage: the author's review threads plus a Claude Code review
  were addressed in follow-up commits on the same branch.

## Oct 28 — temporal graph, exact matcher, replay benchmark

Built with Muse Code (agentic coding CLI powered by Meta Muse Spark):

- `src/motifql/graph.py`: CSR temporal multigraph, windowed adjacency,
  lazy eviction floors.
- `src/motifql/matcher.py`: anchored backward-DFS exact matchers for
  `round_trip` and `layering_chain`, fanout/expansion caps, truncation flags.
- `eval/replay_bench.py`: full train-split replay with a fork-based worker
  pool sharing the graph copy-on-write.
- `tests/test_graph.py`, `tests/test_matcher.py`: committed oracle tests
  (windowed queries vs brute-force Polars, time-ordering, caps, eviction).
- `docs/2026-10-28-graph-matcher.md`: this day's report, including the
  Python-stays decision and the single-account 168K hub finding.
- PR review triage: the author's 13 threads plus three Codex findings were
  addressed in follow-up commits (visited-set fix, overlap recall, report
  corrections).
- Post-merge note (Oct 29 branch): replay pool results are realigned to
  anchor order by position key. This changes no Oct 28 numbers —
  percentiles and fire-count sums are order-free.

## Oct 29 — compiler, fuzzy matcher, scored replay

Built with Muse Code (agentic coding CLI powered by Meta Muse Spark):

- `src/motifql/parser.py`: Lark Earley grammar to typed AST (durations to
  seconds, percents to fractions); one parse-tree-only grammar restructure,
  language unchanged under the freeze.
- `src/motifql/checks.py`: the five LANGUAGE.md static checks with
  repair-ready error codes for the Oct 31 model loop.
- `src/motifql/planner.py`: path plans with prunable/deferred split,
  per-predicate labels, cost estimate and budget.
- `src/motifql/degrees.py` plus the shared DFS core in `matcher.py`:
  fuzzy degrees, sound score pruning, best witness with explanations.
- `eval/score_replay.py`: text-to-plan-to-scores end-to-end replay with
  pruning validation.
- `tests/test_compiler.py`, `tests/test_scores.py`: parser/check/plan pins
  plus pruning-soundness property tests.
- `docs/2026-10-29-compiler-matcher.md`: this day's report, including the
  strict-reference-motif finding that sets up the Nov 1 discovery agent.

## Oct 30 — group motifs, butterfly, RQ1 harness

Built with Muse Code (agentic coding CLI powered by Meta Muse Spark):

- `src/motifql/planner.py`: `GroupPlan` (fan/scatter/gather roles,
  having/soft/where validation) and `ButterflyPlan` (2x2 core) alongside
  `PathPlan`; `prev` in group soft is E_PLAN.
- `src/motifql/groups.py`: bounded anchor-touched executors per kind
  (fan adjacency, scatter leg2-anchored, gather p-anchored, butterfly
  4-edge join), aggregate soft layer reusing `degrees.py` atoms, hub caps
  with truncation flags; `score_any` dispatches any plan type.
- `eval/rq1.py`: all-motif train+holdout replay, per-day ranking by
  top_score, transfer/instance recall at fixed budgets via the shared
  `eval/evaluate.py` protocol, summary CSV plus per-anchor score parquets
  for Nov 1 tuning.
- `tests/test_groups.py`: executor oracles vs brute-force frame loops,
  firing/having/witness pins, butterfly decoy test; `test_compiler.py`
  updated for the new plan types.
- `docs/2026-10-30-group-eval.md`: this day's report, including the
  zero-fill degeneracy warning for fixed-budget recall of strict motifs.

## Oct 31 — correctness review fixes (critical + high)

Built with Muse Code (agentic coding CLI powered by Meta Muse Spark):

- Issue #6 triage: multi-agent review of the Oct 27-30 rush found 1
  critical + 8 high + 7 medium correctness bugs; this pass fixes the
  critical and highs.
- `src/motifql/matcher.py`: self-loop anchors return immediately (no
  simple-path witness ends there); test oracle fixed identically.
- `src/motifql/planner.py`: amount-only prunable classification, min of
  span bounds, single-hop scatter/gather legs, butterfly soft restricted
  to `span(all) is <term>`.
- `src/motifql/parser.py`: duplicate clauses rejected.
- `src/motifql/checks.py`: term-domain unit check (duration/money/number).
- `src/motifql/groups.py`: scatter older-leg1 pairing, gather
  `p.time > g.time` guard, butterfly named-term grading (soft-less = 1.0).
- Regression tests per fix (75/75 pass); full bench, scored-replay and
  RQ1 reruns with dated correction sections in the Oct 28-30 reports.
- Medium pass (same day, same issue): unquantified-comparison E_PLAN,
  distinct arithmetic AST node with unit inference, `*1..n` rejection,
  butterfly `continue`, scatter shortcut removal, gather `within 2`,
  truncation-claim corrections (84/84 pass; RQ1 rerun, gather row only).

## Nov 1 — eval-leak fixes (issue #6, E1–E6)

Built with Muse Code (agentic coding CLI powered by Meta Muse Spark),
from a second review thread with independent repros:

- `eval/evaluate.py`: fixed-budget ranking breaks score ties by a keyed
  hash of the transfer id (file order correlates with the label).
- `eval/features.py`: trailing windows count strictly earlier
  timestamps (same-minute rows are simultaneous); oracle updated with
  a shuffled order-independence check.
- `eval/supervised.py`: GBMs train unweighted (18/80 trees, valid AUC
  0.976/0.942, continuous scores — was 1 tree each).
- `eval/rq1.py` + `groups.score_any`: RQ1 ranks path motifs unpruned
  (pruning zeroes every sub-emit top_score).
- `src/motifql/degrees.py` + `groups.py`: all-skipped `and`/`or` bodies
  propagate the skip (was 1.0/0.0), both mirrors.
- `src/motifql/groups.py`: scatter aggregates range over every valid
  leg edge (was one pair per member).
- Baselines + RQ1 rerun; Oct 27 §10 and Oct 30 §8 corrections.

## Nov 1 (tuning) — calibration pass

Built with Muse Code (agentic coding CLI powered by Meta Muse Spark):

- `docs/2026-11-01-tuning.md`: train-fit tuning report. Adopted:
  round_trip emit 0.6→0.5 (8/9 train, 3/3 holdout CYCLE), scatter
  window 7d + dropped per-mid amount term (0/7→4/7 train, all
  catchable), gather balance within 4 (4/6→5/6 train).
- Rejected with evidence: fan several (precision death), layering
  threshold/*2 (truth below background), butterfly (proof: no 2×2 in
  any of 49 truth instances), structuring (demo, no truth).
- Full unpruned RQ1 rerun with tuned motifs; reference pins updated
  (91/91 pass).

## Nov 1 (cont.) — motif refits + `all but K`

Built with Muse Code (agentic coding CLI powered by Meta Muse Spark):

- `docs/2026-11-02-refits.md`: layering recovered to ceiling (*5..8
  reshape, 1/15→3/15, background 0/1600), scatter-fragment rejected
  (floods; misses are 2-cycle orphans), inst 128 diagnosed
  (time-disordered 7-cycle, round_trip ceiling 8/9).
- Language: shipped `all but K` (grammar, checks, both evaluators,
  LANGUAGE.md, 3 tests); killed graded-having (no new power),
  pairing/B2 (overlap + engine cost), weights (no customer).
- Self-loop group-inclusion pinned; no full replays (94/94 pass).

## Nov 2 — group-executor findings (issue #11, F1–F7)

Built with Muse Code (agentic coding CLI powered by Meta Muse Spark),
from a post-#6 review of main with independent repros:

- F1 (high): same-minute member readings picked by file order leaked the
  label back into scores. Same-timestamp edges are simultaneous now: the
  member body scores over every tied reading and the best wins, in all
  four executors (keyed-hash order + 64-reading cap as backstop).
- F2: scatter self-loop anchors no longer complete via the fallback
  (main loop skips m == d; the anchor is leg2).
- F3: `trapezoid`/`triangle` degenerate shoulders closed (1 at a == b).
- F4: member account refs (`r.in_degree`) E_PLAN instead of scoring a
  silent empty-set constant; LANGUAGE.md notes account fields uncarried.
- F5: scatter fills `leg2.gap` (exact per-pair forward delay); time
  minus time infers duration, so `leg2.time - leg1.time <= 1h` checks.
- F6: gather with no collections is not-found even when `having` only
  gates payouts.
- F7: butterfly tries every in-window edge per pair via achievable
  min-times (exact for any term shape; also fixes F1's butterfly leg).
- Verified: 7 focused tests (101/101 pass) plus a randomized
  butterfly-vs-brute-force differential (150 graphs x anchors x 3 term
  shapes, exact). Oct 30 §8 marked stale pending an RQ1 rerun (F1/F2
  move those numbers; no long batches run).

## Nov 3 — Oct 31 stage: Spark compiler + RQ4 first numbers

Built with Muse Code (agentic coding CLI powered by Meta Muse Spark),
overnight:

- `src/motifql/models/`: Spark client (Model API, prompt-hash disk
  cache, retries incl. flaky content-filter 400s) + §7.1 repair loop
  (error feedback, sanity fire-rate feedback, crashes-as-feedback).
- `eval/rq4_items.py`: 50-item English set with verified references
  (all compile; committed test pins that).
- `eval/rq4.py`: three-arm harness (repair / repair_nosanity /
  oneshot), train-sample witness Jaccard, resume + reference cache.
- `docs/2026-11-03-rq4.md`: first numbers. Compile-error feedback
  works (3/3 E_PLAN fixes, nosanity 47/47); sanity-rate feedback
  HURTS (repair Jaccard 0.641 vs 0.954 — rare-but-correct motifs get
  "loosen"ed into wrong ones); provider content filter flaky on 3
  benign items. 12 tests (112/112 pass).
