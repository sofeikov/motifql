# MotifQL

**Catch the laundering shapes that transaction-level models miss, and see
the money trail behind every alert.**

MotifQL is a small language for money-flow patterns: cycles, layering
chains, scatter-gather. You describe the shape, Meta's Muse Spark writes it
as a motif, and the engine scores every transfer against it. Each alert
comes with its witness: the actual chain of transfers that matched, and how
well each condition held.

> Built for the Meta Global AI Developer Hackathon (Oct 26 – Nov 4, 2026).
> **Pre-existing:** [ezrules](https://github.com/sofeikov/ezrules), the
> author's open-source rules engine (used only through its HTTP API), and
> IBM's AMLworld synthetic datasets (used for evaluation). **Everything in
> this repository was built during the hackathon.**

## The problem

Transaction monitoring looks at **one transfer at a time**, plus windowed
totals. Laundering is a **shape over time**: money that leaves an account
and comes back a few days later, or moves down a chain of accounts losing a
little at each hop.

- **Aggregates can't see shapes.** Counts and sums per account have no idea
  that the money came back to where it started.
- **Exact rules are brittle, and hard to write.** "Every hop within 48h and
  within 10% of the last" is beaten by a 50-hour delay. And who picked 48h?
- **Graph query languages are not for analysts.** Few compliance teams
  write Cypher, and fewer can tune it.

## What a motif looks like

```
motif round_trip {
  match (a:account) -[e:transfer*2..5]-> (a)
  where e is time_ordered
    and span(e) <= 7d
  soft  each e: e.gap is patient
    and each e: e.amount ~= prev.amount within 20%
  score min
  emit when score >= 0.5
}
```

Money leaves `a` and comes back through 2 to 5 accounts within a week,
with each hop following the last reasonably soon and for roughly the same
amount. `where` is the hard shape. `soft` holds the fuzzy conditions: they
grade from 0 to 1 instead of passing or failing, so a 50-hour hop scores a
little lower instead of vanishing.

For every transfer, the engine returns:

- **a score** in [0, 1]: how well this transfer completes the motif
- **a witness**: the chain of transfers that matched
- **an explanation**: the degree of every condition on every hop

## Why fuzzy: you don't have to guess the threshold

To write an exact rule you have to pick the cut-off. On the HI-Small
holdout, the right cut matches the fuzzy motif; a cut that's a little off
loses most of the catches. (Exact rules here are the fuzzy motif with every
condition cut at a fixed level; target-pattern transfers caught at 50
alerts per day.)

| Motif | Exact, loose cut | Exact, best cut | Exact, tight cut | **Fuzzy** |
|---|---|---|---|---|
| round_trip → CYCLE | 8 | 12 | 2 | **12** |
| layering_chain → RANDOM | 9 | 17 | 3 | **18** |

You only find the best exact cut by sweeping thresholds against labels.
The fuzzy motif ranks instead of cutting, so it gets there without the
search, at every alert budget. Under evasion (±10–20% noise on amounts),
early runs show the best exact rule collapsing faster than the fuzzy
ranking; full curves are in progress.

## Results so far

All numbers are on AMLworld HI-Small, holdout days only, at **50 alerts per
day**: the analyst budget a real team works to. The comparison is a
LightGBM model trained on trailing-window aggregates plus each transfer's
own amount, payment format and self-loop flag: a strong transaction-level
baseline. (These numbers come from the issue #15 re-evaluation in
[PR #16](https://github.com/sofeikov/motifql/pull/16).)

Laundering transfers caught, by pattern:

| Pattern | MotifQL (3 motifs) | LightGBM |
|---|---|---|
| CYCLE (money comes back) | **24** | 1 |
| RANDOM (layering chains) | **13** | 0 |
| SCATTER-GATHER | 43 | 44 |
| FAN-IN / FAN-OUT / GATHER-SCATTER | 4 | 77 |

The two are complementary. Motifs catch the structural patterns that a
transaction-level model barely sees, while the model wins on high-volume
fan patterns, where counting is enough. Every motif alert can be explained
to an investigator; a model score can't.

## Written with Muse Spark

Muse Spark compiles English descriptions into MotifQL. A repair loop feeds
the compiler's errors back to the model, and every error has a stable code,
a location and a suggested fix, written for a model to act on.

- **47 of 47** attempts compiled on a 50-item test set of precise English
  descriptions (the model provider blocked the other 3), with **0.96** mean overlap between the generated motif's
  alerts and a hand-written reference.
- A **discovery agent** looks at laundering the current motifs miss and
  proposes new motifs, tested on training data only. It is experimental.
- The project itself was built with **Muse Code**, Meta's agentic coding
  CLI. [docs/BUILD_LOG.md](docs/BUILD_LOG.md) records what was built with
  it, day by day.

## How it works

```
 English ──► Muse Spark ──► MotifQL ──► compiler ──► plan
                 ▲  repair loop ◄── errors    │
                 │                            ▼
          discovery agent ◄── misses ── engine: temporal graph + matcher
                                              │
                         score · witness · explanation per transfer
```

- **Compiler:** a Lark grammar, a typed AST, static checks (bounds, units,
  scope) and a planner that splits conditions into ones that can prune the
  search and ones checked at the end.
- **Path matcher:** an anchored, backward, time-ordered search from each new
  transfer. It prunes branches whose best possible score can't win, so the
  best witness is exact.
- **Group executors:** fan, scatter-gather, gather-scatter and butterfly
  motifs, with aggregates (count distinct, sum, span) per group.
- **Hub safety:** per-node and per-anchor caps. Hitting one marks the result
  `truncated` instead of failing.

Language reference: [docs/LANGUAGE.md](docs/LANGUAGE.md). Design:
[docs/DESIGN.md](docs/DESIGN.md). Reference motifs for the AMLworld
patterns: [examples/amlworld.motif](examples/amlworld.motif).

## Honest about the evidence

We ran several adversarial review rounds on our own evaluation and fixed
what they found. Some limits remain, and we'd rather state them than have
you find them:

- **The holdout is small and was partly used during tuning.** HI-Small's
  holdout has 29 complete laundering instances, and several tuning
  decisions were made with holdout numbers in view. Treat the results as
  strong development-set evidence, not a sealed test.
- **The ground truth has many fragments.** Many AMLworld "instances" are
  one or two transfers between a few reused account pairs, so instance
  counts overstate how many multi-hop structures were caught. The tables
  above count transfers.
- **The English test set is precise.** Its descriptions state hop ranges,
  windows and thresholds. Compiling loose analyst intent is the next tier.

The review history is public:
[#6](https://github.com/sofeikov/motifql/issues/6),
[#11](https://github.com/sofeikov/motifql/issues/11) and
[#15](https://github.com/sofeikov/motifql/issues/15). The dated reports in
[docs/](docs/) record what was found and fixed.

## Status

| Done | In progress |
|---|---|
| Language, grammar, compiler and static checks | LI-Small replication on the current engine |
| Path matcher with exact best-witness search | Discovery agent proposals |
| Group motifs (fan, scatter, gather, butterfly) | |
| Evaluation harness, baselines, fuzzy vs exact | |
| Muse Spark compiler with repair loop | |
| Streaming ingest, live engine, ezrules proxy (stub-tested) | |
| RQ3 latency benchmarks (~15× live headroom over peak minute) | |
| Demo UI: English box → motif → animated witness | |

## Demo

Describe a pattern in English; Muse Spark writes the motif; the
witness animates hop by hop with per-hop amounts, gaps and condition
degrees. Needs `MUSE_KEY` for live compilations — the scripted demo
(a committed Spark reply tracing train CYCLE instance 102) works
without one:

```bash
PYTHONPATH=src uv run python -m motifql.server --demo --port 8000
# open http://127.0.0.1:8000/
```

Rehearsal script: [docs/REHEARSAL.md](docs/REHEARSAL.md).

## Reproduce

AMLworld data is not included. Download it from its source under its own
terms, then:

```bash
uv sync
uv run scripts/prep_amlworld.py --input ~/Downloads/moneyLaundering --dataset hi-small --out data
uv run eval/rq2.py --data data/hi-small --workers 16
uv run eval/supervised.py --data data/hi-small --out eval/results
uv run pytest
```

## Licence

[MIT](LICENSE).
