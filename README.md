# MotifQL

Fuzzy, time-ordered money-flow motifs for transaction monitoring, written in
plain English and compiled by Meta's Muse Spark.

> Built for the Meta Global AI Developer Hackathon (Oct 26 – Nov 4, 2026).
> **Pre-existing:** [ezrules](https://github.com/sofeikov/ezrules), the
> author's open-source rules engine (used only through its HTTP API), and
> IBM's AMLworld synthetic datasets (used for evaluation). **Everything in
> this repository was built during the hackathon.**

Status: design stage. Start with:

- [docs/DESIGN.md](docs/DESIGN.md): problem, architecture, use of Meta
  models, evaluation plan, schedule
- [docs/LANGUAGE.md](docs/LANGUAGE.md): MotifQL v0 language reference
- [grammar/motifql.lark](grammar/motifql.lark): v0 grammar (Lark)
- [examples/amlworld.motif](examples/amlworld.motif): reference motifs for
  the AMLworld laundering patterns

```
motif round_trip {
  match (a:account) -[e:transfer*2..5]-> (a)
  where e is time_ordered
    and span(e) <= 7d
  soft  each e: e.gap is soon
    and each e: e.amount ~= prev.amount within 20%
  score min
  emit when score >= 0.6
}
```

## Licence

[MIT](LICENSE). AMLworld data is not included; download it from its source
under its own terms.
