# Day report: Oct 29 — compiler, fuzzy matcher, scored replay

Branch: `oct29-compiler-matcher`. Plan row: DESIGN.md §9, Oct 29.
Status: compiler (parse → check → plan) done for the v0 surface; fuzzy path
matcher with degrees, pruning and witnesses done; both reference path motifs
replay end-to-end from `.motif` text. 40/40 tests pass.

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
accepted language is byte-identical.

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

Pruning saves 4.9×/5.6× expansions on the head sample (185× on the
instance-102 anchor: 30 vs 5,575); the full fuzzy replay is 33× faster
wall-clock than the exact one on `round_trip` because the search, not the
degree arithmetic, dominates. Truncation fell to zero: pruned branches die
on scores long before hitting expansion caps.

Score distribution is bimodal: no anchor scores in (0, 0.6) — a witness
either clears emit or dies at zero (one gap past 48h zeroes the min). The
fuzzy layer as written behaves nearly crisply; partial degrees are rare.

## 4. RQ1 reading (honest version)

At emit: CYCLE 2/9 (142 at 0.749, 159 at 0.826), RANDOM 0/4 evaluable (11
excluded below min_hops 3). At ~500/day the round_trip threshold is 0.931:
no train instance reaches it. The hand-written reference motifs are too
strict as written — instance 102's true path gaps (39h, 31h) score 0.21
under a `soon` that is fully true only to 6h, and its amounts are conserved
to ~6%, so amount degrees cannot save it. The GFP-default breakpoints do
not match HI-Small timing.

That is a tuning problem, not an engine problem, and it sets up Nov 1:
the discovery agent's job is term-breakpoint and threshold tuning against
the training split, and any fair RQ2 exact-vs-fuzzy comparison needs tuned
fuzzy motifs. The engine contribution stands regardless: ranking exists
(142/159 separate from background; 102's 0.21 beats 99% of anchors at 0),
while exact matching only counts.

## 5. Open items (Oct 30)

Group motifs (fan_out, fan_in, scatter_gather, gather_scatter,
structuring_fan_in), the bipartite_butterfly multi-pattern matcher, and
the evaluation harness (RQ1 tables at fixed alert budgets, holdout split).
Sub-emit ranking fidelity under pruning still needs a full-split
pruned-vs-unpruned comparison; the 2K sample agrees at 99.95–100%.
