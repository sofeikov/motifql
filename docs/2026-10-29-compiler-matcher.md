# Day report: Oct 29 — compiler, fuzzy matcher, scored replay

Branch: `oct29-compiler-matcher`. Plan row: DESIGN.md §9, Oct 29.
Status: compiler (parse → check → plan) done for the v0 surface; fuzzy path
matcher with degrees, pruning and witnesses done; both reference path motifs
replay end-to-end from `.motif` text. 47/47 tests pass (post-review batch).

## 1. Compiler

`src/motifql/parser.py` — Lark Earley grammar to frozen dataclasses
(literals normalised: durations to seconds, percents to fractions, money to
amount+currency). Parses all 8 reference motifs with no ambiguity (rechecked
after the change below).

`src/motifql/checks.py` — the five LANGUAGE.md §5 checks with stable codes
(E_TIME, E_UNIT, E_CRISP, E_SCOPE, E_COST), each carrying motif, line,
problem and suggested fix for the Oct 31 repair loop. All 8 reference
motifs pass; one crafted failing case per code is pinned in tests.

`src/motifql/planner.py` — single-pattern path motifs become a `PathPlan`:
hop bounds, anchor-relative window, term bindings, emit threshold, and the
soft split into prunable per-step groups (gap terms, amount-vs-prev
tolerances under `each`/`all`) vs a deferred remainder (evaluated once per
completed witness). Cost estimate is mean-in-degree powers summed to
max_hops (round_trip ~1e5, layering_chain ~1e6 on HI-Small); the 10M budget
rejects a hypothetical `*2..8` motif (~1e8) while the references pass.
Group and multi-pattern motifs are rejected with a clean "lands Oct 30".

**Grammar freeze.** The v0 surface is frozen: no syntax changed. One
parse-tree-only restructure was needed — the group rule now carries
`key_list`/`member_list` sub-rules, because anonymous `("," NAME)*`
repetitions otherwise erase the keys/members boundary from the tree. The
accepted language is byte-identical. Known wart, unchanged by the freeze:
term names shadowing keywords (`within`, `count`, …) do not lex as NAMEs,
so they cannot be defined; the repair loop should avoid them.

## 2. Fuzzy matcher

`src/motifql/matcher.py` + `src/motifql/degrees.py` — one shared anchored
backward DFS serves exact `match`/`find_all` and scored `score` (the Oct 28
refactor, so the visited and window fixes cannot diverge). Degrees:
trapezoid/triangle/ramp shapes, `~=` tolerances, min/product t-norms,
Zadeh quantifiers. Each completed witness returns per-predicate degrees
with the weakest 1-based step, e.g. instance 142's firing witness reports
which gap and which hop held the score down.

Pruning: prunable group degrees accumulate bottom-up along the walk; their
running combination upper-bounds any completion, so a branch is cut once it
drops below the best score (or emit). Pruning sets no truncation flag —
cuts are sound, caps are not. Soundness is a pinned property, not a claim:
pruned and unpruned runs agree on found/witness/score everywhere (30 random
graphs × all anchors, plus a 2K-anchor HI-Small sample at 100%). Sub-emit
top scores may understate under pruning (documented on `ScoredResult`);
above emit they are exact.

## 3. Scored replay (train split, 1,862,302 anchors, 16 workers)

| motif | fires (/day) | mean expansions | truncated | wall |
|---|---|---|---|---|
| round_trip exact (Oct 28) | 41,310 (8,262) | 1,733 | 0.52% | 570s |
| round_trip fuzzy @0.6 | 15,326 (3,065) | 41.3 | 0% | 17s |
| layering_chain exact (Oct 28) | 673,462 (134,692) | 6.9 | 0% | 5s |
| layering_chain fuzzy @0.7 | 18 (3.6) | 7.8 | 0% | 5s |

Pruning saves 42×/8.4× expansions on a 2K uniform random sample (185×
on the instance-102 anchor: 30 vs 5,575); the full fuzzy replay is 33×
faster wall-clock than the exact one on `round_trip` because the search,
not the degree arithmetic, dominates. Truncation fell to zero: pruned
branches die on scores long before hitting expansion caps. Found-agreement
pruned-vs-unpruned is exact; top scores agree on 99.0–99.95% (differences
are sub-emit only, by construction).

Score distribution, unpruned: ~1–2% of anchors score above 0, under 1%
land in (0, 0.6) with a median non-zero score around 0.58 — so sub-emit
ranking is real but costs the full unpruned search. (An independent
review sample, seed 11 over 60k anchors, found 1.35% / 0.72% / 0.578:
same shape.) The pruned replay's empty (0, 0.6) band is a measurement
artifact of pruning against emit, not evidence that partial degrees do
not exist.

## 4. RQ1 reading (honest version)

At emit: CYCLE 2/9 (142 at 0.749, 159 at 0.826), RANDOM 0/4 evaluable (11
excluded below min_hops 3). At ~500/day the round_trip threshold is 0.931:
no train instance reaches it. The hand-written reference motifs are too
strict as written — instance 102's true path gaps (39h, 31h) score 0.21
under a `soon` that is fully true only to 6h, and its amounts are conserved
to ~6%, so amount degrees cannot save it. The GFP-default breakpoints do
not match HI-Small timing.

That is a tuning problem, not an engine problem, and it sets up Nov 1:
the discovery agent's job is term-breakpoint and threshold tuning.
Tuning protocol, agreed up front: fit breakpoints and thresholds on train
only, report recall on holdout — never the reverse — and every tuned motif
ships with its train background rate (alerts/day at the reported
threshold) next to its recall, so a tuned-up number cannot hide a blown-up
budget.

A note on RQ2 (exact vs fuzzy): the fair comparison is at equal alert
budgets. The baseline to beat is a 6-day/8%-tolerant exact `round_trip` —
the most permissive bound that keeps every train CYCLE reachable, tuned on
train to the same alerts/day as the fuzzy motif — not the current 7-day
exact row, which over-fires by allowing the slowest member of every
family (and not the current fuzzy row either, which under-fires on
untuned breakpoints). Neither published number is the comparison; the
tuned pair is Nov 1 work.
The engine contribution stands regardless: ranking exists (142/159
separate from background; 102's 0.21 beats 99% of anchors at 0), while
exact matching only counts.

## 5. Open items (Oct 30)

Group motifs (fan_out, fan_in, scatter_gather, gather_scatter,
structuring_fan_in), the bipartite_butterfly multi-pattern matcher, and
the evaluation harness (RQ1 tables at fixed alert budgets, holdout split).
Sub-emit ranking fidelity under pruning holds at 99.0–99.95% on the 2K
random sample; the residual is sub-emit by construction.

## 6. Correction (Oct 31): self-loop anchors

The §3 rows inherit the Oct 28 self-loop bug (see that report's §8):
bogus 2-self-loop witnesses also dominated the fuzzy fires. The exact
rows are corrected there (41,310 → 2,255; 673,462 → 660,233). Fuzzy
re-measured on the current (recalibrated) motifs, train, 16 workers:

| motif | fires (/day) | mean expansions | truncated | wall |
|---|---|---|---|---|
| round_trip fuzzy @0.6 | 113 (22.6) | 42.5 | 0% | 21s |
| layering_chain fuzzy @0.7 | 17 (3.4) | 24.2 | 0% | 13s |

Pruning still saves 39×/32× expansions on the 2K sample with exact
found-agreement and 0.9995 top-score agreement — the soundness story is
intact. Train CYCLE hits at emit: 102 (0.72), 142 (0.90), 159 (0.92).
The §4 "8k/day background" concern inverts for `round_trip`: background
is now 23/day and no threshold fills a 500/day budget (threshold 0.0).
Holdout/RQ1 readings are corrected in the Oct 30 report.
