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
