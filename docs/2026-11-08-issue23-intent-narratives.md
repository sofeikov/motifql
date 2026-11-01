# Loose-intent tier + witness narratives (Nov 8, issue #23)

Closes the two gaps in the issue: RQ4 measured "renders a complete
spec" on 50 fully-pinned items, and the demo showed degrees but no
investigator story (DESIGN.md §7's witness-narrative role was never
built). One deliverable covers both: a 10-item loose-intent tier plus
a generated narrative per witness, shown in the demo UI.

## 1. What was built

**Intent tier (eval).** `eval/rq4_intent_items.py`: 10 loose
descriptions with no hop counts, windows or thresholds (pinned by
test: no digits, no units), each with 1–2 acceptable reference motifs
(a disjunction reusing precise-tier references, not one gold) and a
tight 3-check rubric (shape / window / condition). Grades committed in
`eval/rq4_intent_grades.csv`.

`eval/rq4.py` grew `--tier precise|intent` (default precise) and
`--compile-only` (repair loop with no data: compile rate + saved
motifs, the rubric input where data is absent). Intent keeps a
separate runs CSV (`rq4_intent_runs.csv`) and separate ref-cache row
ids (`id#k`); Jaccard is max-over-refs. The precise tier is
byte-compatible: same 50 items, same CSV path and columns, same
ref-hash provenance, old rows resume. Full-run motif texts go to a
`<stem>_motifs.json` sidecar so the CSV schema never changes.

**Narratives (service).** `src/motifql/models/narrate.py` (the
`models/narrate.py` DESIGN.md §11 planned): builds a deterministic
extractive witness table (per-transfer amounts/dates/accounts,
gaps on paths, per-condition degrees, weakest step) and asks Muse
Spark for 3–5 plain sentences through the stock Spark client, so
prompts and replies share the compiler's disk cache. Fail closed:
no hops, missing fields, model errors, `CANNOT NARRATE` replies and
verification failures (an amount/date not in the witness) all yield
no narrative, never an invented one.

**Demo path.** `compile_service` narrates each witness (a
`narrate_complete` transport, `None` sketches none); the committed
`ui/canned.json` story covers the scripted anchor with no model call.
`/compile` ships `narrative` + `narrative_weakest_step` per witness;
the page renders the paragraph under the stage with the weakest step
badged, including in `?at=` stills. The canned narrative is genuine
Spark output over the pinned witness (tids 2012557–9, top 0.718),
verified verbatim before committing — recording still needs no key.

## 2. Intent tier results (full run, HI-Small train sample)

10 items × repair/oneshot, same 2,000-anchor sample protocol as the
precise tier (seed 7). 39K input / 19K completion tokens; oneshot
rows were fully cache-served from repair round 1s.

| arm | compiled /10 | rubric pass /10 | Jaccard (max-over-refs, n=9) |
|---|---|---|---|
| repair | 10 | 9 | 0.622 |
| oneshot | 10 | 9 | 0.570 |

(Jaccard n=9 excludes intent-structuring: both refs fire 0 on the
sample, the same uninformative oracle as the precise structuring
refs. Reported separately from the precise tier, never merged.)

Per-item (repair arm; oneshot differs only where noted):

| item | compiled | rubric | Jaccard | note |
|---|---|---|---|---|
| roundtrip | ✓ | pass | 1.00 | identical to a reference |
| quick-return | ✓ | pass | 0.71 | 7d span is the loose end of "quickly"; soon gaps carry it |
| layering | ✓ | pass | 1.00 | 2..5 open chain |
| fanout | ✓ | pass | 0.31 | sanity tightened round 2 (≥3/2d/0.6); oneshot 1.00 |
| fanin | ✓ | pass | 1.00 | |
| scatter | ✓ | pass | 1.00 | round 2 dropped an over-tight soon condition; oneshot 0.00 |
| gather | ✓ | pass | 0.21 | round 2 loosened ≥3 to ≥2 per side; oneshot 0.05 |
| structuring | ✓ | pass | n/a | fan-in + most just_below(10000); refs fire 0 |
| slow-drift | ✓ | pass | 0.00 | 4..7/30d: narrower than both refs yet defensible |
| backandforth | ✓ | **fail** | 0.37 | generic 2..5 cycle, missed the pair shape (max 5 > 4) |

Findings:

- **The rubric earns its keep.** slow-drift scores Jaccard 0.00
  against a disjunction that doesn't cover its reading, while the
  text passes all three checks — exactly the case a single-gold
  Jaccard mishandles. Conversely backandforth is the honest fail:
  the model wrote the generic cycle and missed "two accounts".
- **Sanity feedback, both directions.** scatter round 2 fixed a real
  over-tightening (0.00→1.00) and gather improved (0.05→0.21), but
  fanout shows the Finding-2 tighten pattern again (1.00→0.31,
  still rubric-passing). Small-n, but consistent with the Nov 3
  result that rate feedback is the risky half of the loop.
- **Loose English compiles.** 20/20 across arms at ≤3 attempts; the
  compiler + repair loop carry underdetermined input without new
  failure modes (one E_PLAN-class round-2 fix in scatter).

## 3. Narrative faithfulness spot-check (n=11)

Ten diverse witnesses (3 path + 7 group shapes: scripted + 2 CYCLE
truth cycles, 2 RANDOM truth chains, fan-out/in tops, gather and
butterfly tops, one SCATTER-GATHER truth) plus a 722-transfer fan as
a scale probe — all narrated live, all passing `verify_narrative`
(amounts to the cent, dates, durations and degrees echoing the
table, every account in the witness, weakest step named). A full
manual read then confirmed roles, order and counts: **11/11
faithful.**

The spot-check forced two fixes, both shipped and re-verified:

1. **Group-shape prompt.** The first prompt was path-centric ("in
   order", per-hop gaps); the model honestly declined all 5 group
   witnesses with `CANNOT NARRATE`. The table now marks
   path-vs-group (members without gap lines for groups) and the
   prompt narrates groups as shared-accounts-plus-members. 6/6
   groups narrate after the fix.
2. **DISTINCT counts.** One narrative wrote "14 different accounts"
   for 14 transfers with a repeated recipient (13 distinct) — a
   soft error no amount/date check can catch. Group tables now
   carry a `DISTINCT: 1 sender, 13 recipients across 14 transfers`
   line and the prompt takes counts from it; the re-narration says
   "14 transfers to 13 recipients". All group rows were re-narrated
   against the final table and re-verified.

Notes: one all-1.000 tie attributes the weakest predicate
differently than the structured field (both degrees 1.000 — no
unique weakest exists); predicate labels quote planner shorthand
(`count(...)`, `DistinctArg`), ugly but verbatim. Cost: the
722-member table cost a 32K-token prompt — capping large tables
(with the counts kept exact) is the follow-up.

## 4. Validation

- 21 new tests, all stubbed/cache-fed, no network: tier contract
  (10 items, underdetermined English, compiling refs, rubric shape,
  precise-tier legacy compat, max-Jaccard, sidecar roundtrip),
  narration (prompt figures, weakest extraction, 6 fail-closed
  paths, hallucination drops, disk caching, group tables),
  service + route (live attach, fail-closed transport, canned
  offline cover, committed canned narrative, route keys).
- Full suite green (343 passed, 3 skipped — the skips predate this
  change), `node --check` on the UI,
  plus a node probe of the narrative renderer (null→empty, HTML
  escaping, weakest badge) and a static check that every DOM id
  the script touches exists in the page.
- Live server check offline (`MUSE_KEY` empty, real data): the
  scripted compile returns source `canned`, top 0.718, the
  committed narrative and weakest hop 3.
- In-browser: headless-Firefox screenshots fail inside this
  sandbox (its software compositor can't map a framebuffer), so
  three states were shot with headless Chromium instead — the
  scripted final still, a mid-trace still, and a live fan-out
  compile: the narrative paragraph and weakest badge render as
  designed in all three, with no page errors. Shots are
  throwaway (`/tmp/shot_*.png`, kept out of the repo).

## 5. Claim scope

The intent tier is **rubric-graded, n=10**: compile rate 10/10 and
rubric pass 9/10 per arm, graded by the author against the committed
rubric (grades + judgment notes in `eval/rq4_intent_grades.csv`).
Jaccard is secondary here by design — the disjunction is thin (two
refs can't span the reading space) and one item's refs are
uninformative. Precise-tier numbers are unchanged and reported
separately. Narratives are extractive and verified on the
spot-check set (n=11); the UI shows none rather than an unverified
one.

## 6. Open items

- Disjunction coverage is thin by construction (slow-drift's 0.00);
  more refs or a looser overlap rule would strengthen the
  automatic half of the tier.
- Cap large group tables (counts exact, members sampled) before
  the demo meets another 722-member fan.
- Weakest-tie attribution (narrative vs structured field) could
  agree by passing the structured weakest into the prompt.
- One dev-box screenshot of the narrative paragraph pre-recording.
