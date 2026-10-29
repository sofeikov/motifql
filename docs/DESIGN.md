# MotifQL design

Status: draft, Oct 26 2026 (hackathon kickoff day). Owner: Konstantin Sofeikov.
Language reference: [LANGUAGE.md](LANGUAGE.md). Grammar:
[`grammar/motifql.lark`](../grammar/motifql.lark).

## 1. Summary

MotifQL is a **fuzzy, time-ordered money-flow motif engine**. Analysts or a
model describe a laundering pattern ("money that comes back to where it
started within a week, through 2 to 5 accounts, losing a little on each
hop"). Meta's Muse Spark compiles that into a MotifQL motif. A streaming
engine then scores every new transfer by how well it completes the motif,
returning a score in [0, 1], a witness subgraph and a per-condition
explanation.

It is built for the Meta Global AI Developer Hackathon (Oct 26 – Nov 4,
2026), **Indie Developer track**, under the "Push the Frontier" /
"Build for Your World" problem statements.

## 2. Provenance (what is pre-existing, what is new)

The hackathon requires a standalone repo containing only work done during the
event, and a video that opens by saying what was pre-existing.

| Pre-existing, not in this repo | Built during the hackathon, in this repo |
|---|---|
| [ezrules](https://github.com/sofeikov/ezrules), the author's open-source rules engine, used as an optional downstream consumer over its public HTTP API | the MotifQL language, compiler, streaming engine, evaluation harness, model integrations, the ezrules proxy and the demo UI |
| IBM AMLworld synthetic AML datasets (Altman et al., 2023), used for evaluation | data preparation scripts (the data itself is downloaded, not committed) |
| Meta's Muse models, called through Meta Model API and Baseten | prompts, repair loop, discovery agent, evaluation of the models' outputs |

No ezrules code is copied into this repo, and no ezrules changes are planned.
If a blocker forces one, it will be released in ezrules first and disclosed.

**Opening line for the video:** "ezrules, my open-source rules engine, and
IBM's AMLworld data existed before this week; everything you'll see, the
MotifQL language, the engine and the Muse Spark integration, I built this
week."

## 3. Problem

Transaction-monitoring rules look at **one transaction at a time**, plus
windowed aggregates. Laundering patterns are **shapes in a money-flow graph
over time**: cycles, scatter-gather, layered chains.

Three gaps follow from that:

1. **Graph features ignore direction and time.** For example, ezrules' graph
   feature is an undirected breadth-first search over event–entity links,
   recomputed with SQL for every event, with a hard node cap that raises an
   error. It can count reachable entities. It can't follow the money or
   detect a cycle.
2. **Exact patterns are brittle.** A rule for "a cycle where every hop moves
   exactly the same amount within 24h" is beaten by ±5% noise or a 25-hour
   delay. Real launderers add exactly that noise.
3. **Patterns are hard to write.** Graph query languages are powerful but
   few compliance analysts write Cypher, and fewer can tune it.

## 4. Goals and non-goals

**Goals (by Nov 4):**
- A MotifQL v0 parser and compiler with static checks (bounds, units, scope,
  cost) that produce error messages a model can act on.
- A streaming engine that scores each transfer against a set of motifs,
  with witnesses, explanations and pruning on the score.
- Reference motifs for 7 of the 8 AMLworld patterns (stack is out of scope
  for v0).
- English-to-MotifQL compilation with Muse Spark, with a repair loop and an
  "is this what you meant?" step that shows the user real witnesses.
- A motif-discovery agent that proposes motifs from laundering the current
  motifs miss.
- An evaluation on HI-Small and LI-Small answering the research questions in
  §8.
- A proxy that adds motif scores to events before they reach ezrules'
  `/evaluate`, so rules can use them with no ezrules changes.
- A demo UI and a 3-minute video.

**Non-goals:**
- Distributed or persistent storage. The graph lives in memory.
- Set-valued variables (the AMLworld stack pattern).
- A general graph database or full GQL/Cypher compatibility.
- Production security hardening, multi-tenancy or authentication beyond an
  API key on the demo service.

## 5. Prior art and positioning

| Work | What it does | How MotifQL differs |
|---|---|---|
| Fuzzy Cypher (Castelltort & Laurent, 2014); Cypherf and Fuzzy4S (Castelltort & Martin, 2018) | fuzzy predicates on static Neo4j graphs, compiled to exact Cypher plus post-scoring | MotifQL is temporal, flow-aware and streaming, prunes on the score during the search, and is written by a model |
| IBM Graph Feature Preprocessor (Blanuša et al., 2024) | real-time mining of fixed laundering subgraph patterns as features for gradient boosting | GFP's patterns are hard-coded and exact; MotifQL's are declarative, fuzzy, explainable and model-authored |
| Temporal motifs (Paranjape, Benson & Leskovec, WSDM 2017) | counting δ-temporal motifs in temporal networks | MotifQL scores individual events online rather than counting motifs offline, and adds amounts and fuzziness |
| ISO GQL (ISO/IEC 39075:2024) | standard graph query language with quantified path patterns | MotifQL borrows the path-pattern syntax and adds time ordering, flow and degrees |
| Multi-GNN (IBM, on AMLworld) | graph neural network baselines for AMLworld | a learned black box, where MotifQL produces readable, auditable detectors; Multi-GNN is a comparison point |

**The claim:** the combination is new. That's fuzzy plus time-ordered plus
money-flow semantics, evaluated incrementally on a stream with pruning on
the score, with motifs written and discovered by a language model. The README
will say this plainly.

## 6. Architecture

```
          AMLworld CSV ──► prep (DuckDB) ──► time-sorted Parquet
                                                   │
                                     replay (batch) │ live (HTTP /score)
                                                   ▼
 English ──► Muse Spark ──► MotifQL text ──► compiler ──► plans
     ▲          ▲   repair loop ◄── errors      │
     │          │                               ▼
     │     discovery agent ◄── misses ─── engine: temporal graph + matcher
     │                                          │
     └── "is this what you meant?" ◄── witnesses, scores, explanations
                                                │
                     ┌──────────────────────────┼─────────────────────┐
                     ▼                          ▼                     ▼
             evaluation harness          ezrules proxy             demo UI
        (recall per pattern, latency)  (adds motifs.* to events)
```

### 6.1 Data preparation
- Load `HI-Small_Trans.csv` and `LI-Small_Trans.csv` into DuckDB.
- Rename the two `Account` columns to `from_account` and `to_account`. Keep
  bank ids as strings, because leading zeros matter.
- **Convert to USD.** Derive exchange rates from rows where the paid and
  received currencies differ. Use the median implied rate per currency pair,
  and check the rates for consistency.
- Parse `*_Patterns.txt` into a table of pattern instances: id, pattern type,
  and its transactions. This is the ground truth per pattern.
- **Split by time.** Warm-up days fill the retention windows. Training days
  are what the agent may look at. Holdout days are never shown to any model.
  Exact day boundaries are set once the timestamp distribution has been
  inspected.
- **Check the licence on the Kaggle page before use.** The data is
  downloaded by script, never committed.

### 6.2 Temporal graph store
- For each account, keep incoming and outgoing transfers in time order, in
  per-node append-only arrays.
- **Retention:** drop transfers older than the longest window used by any
  active motif, plus a margin.
- **Live statistics** (degree distributions per node and globally) feed the
  cost estimator.
- **Start in Python.** If the day-2 benchmark misses its target, the hot loops
  move to Rust (PyO3) or Numba. The data structures are chosen so that move
  is mechanical.

### 6.3 Compiler
1. Parse with Lark into a typed AST.
2. Resolve names and units.
3. Run the static checks in LANGUAGE.md §5.
4. Classify every `soft` condition as **final on bind** (prunable) or
   **evaluate on completion**.
5. Plan. Path motifs become a backward search order from the anchor. Group
   motifs become an anchor lookup plus per-key aggregate state.
6. **Estimate cost** from live statistics, and reject motifs above the
   budget with the bound to tighten.

Every compiler message has a stable code, the location, the problem and a
suggested fix, because the repair loop (§7.1) feeds it straight back to the
model.

### 6.4 Matcher
- **Anchored, backward, depth-first search** for path motifs. Edges must be
  earlier than the anchor and inside the hard bounds. Simple-path checks use
  a small visited set.
- **Pruning on the score** (LANGUAGE.md §6). The running degree of the
  conditions that are final on bind is an upper bound, so a branch is cut when
  that bound drops below the threshold or the best witness so far.
- **Group motifs** keep running aggregates (count distinct, sum, average,
  min, max, span) per group key, updated as transfers arrive and expire.
- **One code path** serves live scoring and batch replay, so they agree by
  construction.

### 6.5 Hubs and truncation
Each anchor has an expansion cap, and each node has a per-step fan-out cap.
Hitting a cap **does not raise an error**. The search stops, the result is
marked `truncated: true` and, for path motifs, the score is treated as a
lower bound. Group-motif scores are not lower bounds under truncation:
dropping members can raise non-monotone aggregates, so a truncated group
result may overstate. The evaluation reports how often truncation happens
and on which motifs.

### 6.6 Interfaces
- **Python API:** `Engine(motifs).score(transfer) -> list[MotifResult]` and
  `Engine.replay(parquet) -> results`.
- **CLI:**
  - `motifql check file.motif` for static checks only
  - `motifql replay --data hi-small --motifs examples/amlworld.motif`
  - `motifql eval` for the research-question tables
- **HTTP:** `POST /score` (one transfer, returns results) and
  `POST /compile` (English or MotifQL in, motif plus checks and sample
  witnesses out).
- **ezrules proxy:** `POST /evaluate` with the ezrules payload. It calls the
  engine, adds `event_data.motifs = {name: score}` and forwards the request to
  a stock ezrules `/api/v2/evaluate`. Rules then read the score as a nested
  field, for example `if $motifs.round_trip > 0.8: return !HOLD`. ezrules'
  rule docs confirm `$` references accept dotted paths into nested JSON.

### 6.7 Compiled fast paths
The day-2 fallback in §6.2 happened: the hot loops run in Numba, and the
pure-Python code stays as the reference.
- **What is compiled.** `motifql._pathkernel` runs the anchored path walk
  behind `match`, `find_all` and `score`. `motifql.groups_fast` builds the
  members for fan, scatter, gather and butterfly. The having and soft
  evaluators stay in Python and are shared with the reference.
- **Batch API.** `groups.score_batch(graph, positions, plan, libs, ...)`
  returns `(top_score, found, expansions, truncated)` arrays. It equals a
  `score_any` loop over the same anchors in the same order, including the
  lazy eviction floors it moves. The eval harnesses (`rq1._replay`, which
  RQ2, perturb and discover reuse, and `score_replay`) call it once per
  pool shard. It is serial: `score_path_batch(parallel=True)` starts Numba
  threads, which are not fork-safe, so it is never used in a process that
  forks a pool later.
- **Reference switch.** `MOTIFQL_PURE_PYTHON=1`, or setting
  `matcher.PURE_PYTHON` / `groups.PURE_PYTHON`, forces the pure-Python
  code. Anything a kernel does not model also runs the reference: deferred
  soft, odd `find_all` caps, a graph whose CSR order check fails.
- **Parity.** The fast paths are bit-identical to the reference in every
  field. `tests/test_pathkernel_parity.py` and `tests/test_groups_fast.py`
  pin this on synthetic graphs. `eval/parity_check_path.py` and
  `eval/parity_check_groups.py` check sampled and hub anchors on HI-Small.
  Run them after any change to either side.
- **No hidden caches.** Nothing is memoised between calls. A path plan's
  atom table is compiled again on every `score` / batch call (about 5 µs),
  so an edited term library is always seen. The kernels' graph-side inputs
  (`in_frm`, `in_amount`, the `csr_ordered` flag) are `TemporalGraph`
  fields built with the CSR. Graph arrays are fixed after construction:
  they are read-only and cannot be rebound, so to change the data, build a
  new graph. Only the retention state (watermark, lazy floors) mutates.

## 7. Use of Meta models

| Role | Model | Why that model |
|---|---|---|
| English → MotifQL compiler | Muse Spark (Meta Model API) | strongest reasoning; output is checked by the compiler, so mistakes are caught |
| Motif discovery agent | Muse Spark | multi-step tool use over the evaluation harness |
| Witness narratives | Muse Spark, or Muse Glimmer if quality holds | short structured summaries; cheaper model if good enough |
| Distilled compiler (stretch) | Muse Glimmer, fine-tuned | runs on own or partner hardware; open-weight bonus |
| Open-weight runs (stretch) | Muse Spark Open on Baseten (`meta-models/muse-spark-open-1026`) | the same compiler prompts on open weights, for comparison and the bonus |

Exact Meta Model API model ids are to be confirmed from Meta's model list on
day 1.

### 7.1 Compile and repair loop
1. Send the model the English request, the grammar, LANGUAGE.md's summary
   tables, the data model, and three or four reference motifs as examples.
2. It returns MotifQL text.
3. The compiler checks it. On any error, the stable error messages go back to
   the model, up to N rounds.
4. **Check sanity on the training split.** Replay a sample. If the motif
   matches nothing, or more than a set share of traffic, that becomes feedback
   too ("matches 38% of transfers; tighten…").
5. **Ask the user "is this what you meant?"** Show three real witnesses with
   their degrees. The user accepts, or rephrases and loops.

The model never sees holdout data.

### 7.2 Discovery agent
**Tools:**
- `list_missed(pattern_type)`: laundering instances in the training split
  that no current motif emits on.
- `show_instance(id)`: its transfers, as a table and as a rendered graph
  image. The image tests whether Spark's vision helps; it is ablated in §8.
- `try_motif(text)`: compile, then replay on the training split. Returns
  recall per pattern, the alert rate and sample false positives.
- `propose(text, rationale)`.

**Loop:**
1. Pick a cluster of misses.
2. Form a hypothesis.
3. Write a motif and try it.
4. Tune the term breakpoints.
5. Stop when it beats the gate (a recall gain at a fixed alert budget) or runs
   out of attempts.

Proposals are scored on the holdout split by the harness, not by the agent.

### 7.3 Distillation (stretch)
1. Spark generates pairs of English descriptions and motifs.
2. **Each pair is verified:** it must compile, and its witnesses must
   overlap a reference.
3. Muse Glimmer is fine-tuned on the verified pairs.
4. Compare its compile and correctness rates with Spark (RQ4).

## 8. Evaluation

**Datasets:**
- HI-Small for development and the main results (5M transfers, 10 days, 1 in
  981 laundering).
- LI-Small for generalisation (1 in 1,942 laundering, longer patterns).
- HI-Medium for a scale test if time allows.

**Alert budget:** results are reported at a fixed number of alerts per day,
because analyst time is what limits AML teams. Exact precision and recall
curves go in an appendix.

| # | Question | Method | Baselines |
|---|---|---|---|
| RQ1 | Can MotifQL express AMLworld patterns that windowed aggregates can't? | recall per pattern at the alert budget, holdout split | windowed aggregates in the style of ezrules (count, count distinct, sum over 1h/24h/7d); hand-written motifs; Spark-written motifs |
| RQ2 | Does fuzziness help, especially under evasion? | the same motifs with every soft condition made exact, at equal alert budget; then perturb laundering instances (amount noise ±5/10/20%, extra delays) and plot recall against perturbation | exact versions of each motif |
| RQ3 | Is it fast enough to run live? | replay HI-Small: transfers per second, median and 99th-percentile latency per transfer, truncation rate, effect of pruning (on/off) | ezrules' SQL breadth-first graph feature for a comparable query |
| RQ4 | Can a model write correct motifs? | about 50 English descriptions with hand-written references: share that compile within N repair rounds; correctness as Jaccard overlap of witness sets with the reference on the training split | Spark with no repair loop; Spark Open; distilled Glimmer (stretch) |
| RQ5 | Do motif scores help a learned model? (optional) | motif scores as features for a gradient-boosted model; minority-class F1 | published Multi-GNN and GFP results on HI-Small |

**The headline chart is RQ2:** as laundering gets noisier, exact motifs fall
off a cliff and fuzzy ones degrade gracefully.

## 9. Plan

| Day | Deliverable | Exit check |
|---|---|---|
| Oct 26 (today) | this design, the v0 grammar, the reference motifs | grammar parses every example with no ambiguity ✅ |
| Oct 27 | data prep, pattern-instance table, time split; Muse Spark API smoke test; credits claimed | recall-per-pattern table for the aggregate baseline |
| Oct 28 | temporal graph store; exact cycle matcher; replay benchmark | **decide Python or Rust** for the hot path, from measured throughput |
| Oct 29 | compiler (AST, checks, planner); general path matcher with degrees, pruning and witnesses | all path motifs in `examples/` run on HI-Small; **grammar frozen** |
| Oct 30 | group motifs; evaluation harness | RQ1 table with hand-written motifs |
| Oct 31 | Spark compiler with repair loop; English test set | RQ4 first numbers. **This is a complete submission.** |
| Nov 1 | discovery agent; RQ2 fuzzy-against-exact and perturbation runs | RQ2 chart |
| Nov 2 | ezrules proxy; RQ3 benchmarks; stretch: Spark Open run, Glimmer distillation | RQ3 table |
| Nov 3 | demo UI (English box → motif → witness graph animated hop by hop); README with results | rehearsal recording |
| Nov 4 | final video, X post, submission before 9 PM ET | submitted |

## 10. Risks

| Risk | Mitigation |
|---|---|
| Python too slow for 5M transfers | measure Oct 28; move hot loops to Rust or Numba; fall back to fewer days |
| Language scope creep | freeze the grammar on Oct 29; new needs become built-in functions, not syntax |
| Motifs that run but mean the wrong thing | execution-based checks; the witness confirmation step; RQ4 measures it |
| Hubs blow up the search | caps plus truncation flags; report truncation rates |
| Amount conservation isn't real in AMLworld | it's stated as a hypothesis; RQ1 shows whether those conditions help |
| Meta API rate limits or cost | cache every model call by prompt hash; the evaluation replays from cache |
| AMLworld licence doesn't allow redistribution or use | never commit the data; check the licence on Oct 27 |

## 11. Proposed repo layout

```
motifql/
  grammar/motifql.lark        surface syntax (v0, this commit)
  examples/amlworld.motif     reference motifs (this commit)
  docs/DESIGN.md              this document
  docs/LANGUAGE.md            language reference
  src/motifql/
    ast.py  compiler.py  checks.py  planner.py
    graph.py  matcher.py  groups.py  engine.py
    models/ (spark.py, repair.py, discovery.py, narrate.py)
    proxy.py  server.py  cli.py
  data/ (gitignored)  scripts/prep_amlworld.py
  eval/ (rq1.py … rq5.py, results/)
  ui/
  tests/
```

## 12. Open questions
1. **UI stack:** a small FastAPI plus a single page with a graph library, or
   Streamlit for speed.
2. **ezrules in the video:** whether to show the ezrules proxy integration,
   or keep it as a README section so the video stays on what's new.
