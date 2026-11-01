# Nov 8 — Issue #24 synthetic validation: the loop can find wins

Three pre-registered-style checks that the calibration scaffolding
(`src/motifql/calibrate.py`, `eval/calibrate.py`, PR #28) is capable
of finding wins at all, run on synthetic beds before touching real
data. Verdict: all three pass. Probes (re-runnable, not committed):
`/tmp/i24_slope.py`, `/tmp/i24bed.py`, `/tmp/i24_recover.py`,
`/tmp/i24_transfer.py`.

## Bed

Truth: 3-cycles with controlled gaps/wobble (train CYCLE instances,
all members inside the Sep 3–7 anchor window). Distractors (not
laundering): volatile-amount 3-cycles (ratios 1.25–1.6×) and slow-gap
2-cycles (60–100h) that compete in the ranking. Filler: sparse
lognormal background. Budgets scaled to bed size ([3,5,10]/day, not
the prereg [50,100]).

Two bed bugs were found and fixed during step 1, both synthetic-only:
planted closings initially fell outside the anchor window (targets
scored exactly 0), and zero-mass background turned ranking into a
lottery (fixed with in-window placements + near-miss distractors).

## Step 1 — slope: PASS

Single-param sweeps (others at hand-tuned default). Objective spans
13–22 with interior peaks and cliffs on every continuous param:

| Sweep | Range | Shape |
|---|---|---|
| tol 5→50% | 13 → 22 | narrow zeroes volatile truth |
| plateau 6→72h | 22 → 15 at 72 | wide admits slow distractors, buries truth |
| support 48→168h | 19 → 22 | narrow clips slow truth |
| score min/product | 22 = 22 | flat on this bed (data-dependent) |

## Step 2 — planted recovery: PASS

Truth slow + volatile (gaps 35–55h, 30% wobble), where the default
(tol 20%, support 120h) reads truth's amounts as ~0. Real `run_fit`,
12 tries, seed 24: baseline obj 4 → winner obj 15 (try 7), moving
exactly as planted (tol 20→33.9, support 120→73.3). The full trace
shows a clean gradient: every high-tol try beats baseline.

## Step 3 — synthetic transfer: PASS

Step-2 winner replayed on a fresh bed (new background seed, wobble
0.28, reversed gap order): fitted 15 vs baseline 4 on both beds,
margin +11 preserved, transfer gap zero within the synthetic family.

## Caveats

- The synthetic family is favorable by construction (planted optimum
  far from default, controlled distractors). Real HI-Small may be
  flatter — the real-data slope check and registered run are still
  required (next section).
- Sampling noise untested (all anchors fit in memory, bg sample = all).

## Real-data follow-up

- [x] Slope check on HI-Small train → objective flat at ceiling, prereg amended (§7)
- [x] Registered fit (seed 24, 30 tries) + train confirm
- [x] Sealed holdout + LI-Small reports, transfer verdict ([report](2026-11-08-issue24-report.md): negative)
