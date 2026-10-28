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
