# MotifQL v0 language specification

Status: draft, Oct 26 2026. The surface syntax is fixed by
[`grammar/motifql.lark`](../grammar/motifql.lark); this document gives the
meaning. Worked examples for all eight AMLworld patterns are in
[`examples/amlworld.motif`](../examples/amlworld.motif).

MotifQL describes **time-ordered money-flow patterns** in a stream of
transfers between accounts and scores how well each new transfer completes
one. It is a graph pattern language (in the spirit of GQL path patterns) with
three additions:

1. **Time.** Edges carry timestamps, and paths can be required to run forwards
   in time.
2. **Flow.** Conditions can relate amounts along a path or across a group, so
   "the money that went in comes out again" is expressible.
3. **Fuzziness.** Conditions in the `soft` clause return a degree in [0, 1]
   instead of true/false, so a motif yields a score and a witness rather than
   a yes/no.

## 1. Data model

The engine sees a directed multigraph built from a stream of **transfers**.

| Kind | Field | Type | Notes |
|---|---|---|---|
| node `account` | `id` | string | AMLworld: bank id + account id, kept as strings |
| | `bank` | string | leading zeros are significant |
| | `first_seen` | time | first transfer involving the account |
| | `in_degree`, `out_degree` | int | over the engine's retention window |
| edge `transfer` | `time` | time | AMLworld `Timestamp` |
| | `amount` | money | normalised to USD (see design doc, data prep) |
| | `amount_paid`, `currency` | money, string | original values |
| | `payment_format` | string | ACH, Wire, Cheque, Credit Card, Cash, Reinvestment, Bitcoin |
| | `gap` | duration | derived: time since the previous step on the same path; undefined on the first step |
| | `index` | int | derived: 1-based position on the path |

The mapping from source columns to these fields is configuration, not part of
the language.

## 2. Lexical structure

- Comments run from `//` to end of line.
- **Names** are lowercase-first identifiers. Reserved words (`all`, `and`,
  `at`, `distinct`, `each`, `false`, `few`, `is`, `least`, `most`, `not`,
  `or`, `some`, `true`, `within`) can't be used as names.
- **Literals:**
  - numbers: `12`, `0.85`
  - durations: `30m`, `6h`, `7d`, `2w`, `500ms`
  - percentages: `10%`
  - money: `10000 USD` (ISO currency code in capitals)

## 3. Top-level items

```
term soon = trapezoid(0h, 0h, 6h, 48h)
term just_below(limit) = trapezoid(0.85 * limit, 0.95 * limit, 0.999 * limit, limit)
quantifier mostly = ramp(0.6, 0.9)
motif round_trip { ... }
```

### 3.1 Terms (membership functions)

A `term` names a fuzzy set over numbers, durations or money. Its body is a
call to a built-in shape:

| Shape | Degree |
|---|---|
| `trapezoid(a, b, c, d)` | 0 below a, rises linearly to 1 at b, 1 until c, falls to 0 at d |
| `triangle(a, b, c)` | `trapezoid(a, b, b, c)` |
| `ramp(a, b)` | 0 at or below a, 1 at or above b, linear in between |
| `ramp_down(a, b)` | `1 - ramp(a, b)` |

Terms can take parameters (`just_below(limit)`), and all arguments must share
one unit. `x is soon` evaluates the term on `x`; `x is not soon` is
`1 - degree`.

### 3.2 Quantifiers

Quantifiers range over the bindings of one variable (see §5). Built-ins:

| Quantifier | Crisp meaning (allowed in `where`/`having`) | Fuzzy meaning (allowed in `soft`) |
|---|---|---|
| `each` / `all` | every binding holds | minimum degree |
| `some` | at least one holds | maximum degree |
| `at least N` | at least N hold | N-th largest degree |
| `most` | — | `ramp(0.5, 0.8)` applied to the mean degree |
| `few` | — | `ramp_down(0.1, 0.3)` applied to the mean degree |
| `@name` | — | a user `quantifier` applied to the mean degree |

`most` and `few` are relative quantifiers in Zadeh's sense: they score the
share of bindings that satisfy the body.

## 4. Motifs

```
motif NAME {
  match   PATTERN, PATTERN, ...                  // required
  group   by VAR, ... over VAR, ... window D     // optional
  where   EXPR                                   // crisp, per binding
  having  EXPR                                   // crisp, per group (needs group)
  soft    EXPR                                   // fuzzy
  score   min | product                          // default: min
  emit    when score >= T                        // default: 0.5
}
```

Clauses after `match` can appear in any order, at most once each.

### 4.1 `match`: patterns

```
(a:account) -[e:transfer*2..5]-> (a)
(x) <-[e]- (y)
```

- **Nodes:** `(var)` or `(var:label)`. A variable repeated in a pattern, or
  across comma-separated patterns, refers to the same node, so
  `(a) -[...]-> (a)` is a cycle.
- **Edges:** `-[var]->` points out and `<-[var]-` points in.
- **Variable-length edges:** `*m..n` makes the edge variable a **path**
  variable. The grammar requires an upper bound, so there are no unbounded
  paths. Paths are simple: no account is visited twice except where the
  pattern closes on a repeated variable.
- **Labels:** in v0 the only node label is `account` and the only edge label
  is `transfer`. Labels exist so the language can grow, for example card,
  device or merchant nodes for fraud.

### 4.2 Anchoring (how a motif is evaluated)

A motif is always evaluated **against one new transfer, the anchor**. A match
counts only if:

1. the anchor is one of its edges, and
2. it is the **latest** edge in the match.

So the engine asks "does this transfer complete the pattern?" and only
searches backwards in time from it. The same transfer can't be scored twice for
one motif, live and batch replay give the same results, and the search space is
bounded by the hard limits.

### 4.3 `where`: hard per-binding constraints

`where` is evaluated on each full binding of the pattern variables and must be
**crisp**. Using `~=`, `is <term>` with a non-crisp term, `most`, `few` or a
named quantifier there is a static error.

Built-in crisp predicates and functions:

| Form | Meaning |
|---|---|
| `p is time_ordered` | each step of path `p` happens strictly after the previous one (required on path motifs: the walk enforces it regardless, so the text must say so) |
| `span(p)` | last time − first time over path `p` |
| `span(all)` | last time − first time over every edge in the binding (or group) |
| `x != y`, `x == y` on node variables | identity comparison |

### 4.4 `group`: collapsing bindings into one candidate

`group by K1, K2 over M1, M2 window D` gathers every binding that shares the
key variables `K…` into one candidate. The `over` variables are the
**members** that vary inside the group.

- **Window:** only bindings whose edges all fall within `D` before the anchor
  count, and the anchor must be in the group.
- **Aggregates are evaluated over the group:**
  - `count(distinct M)`
  - `sum(e.amount)`, `avg(e.amount)`, `min(…)`, `max(…)`, where `e` ranges over
    the group's edges
  - `span(all)`
- **Quantifiers over a member variable** (`most m: …`) iterate that member's
  bindings. A member's edges are the edges that bind it.

### 4.5 `having`: hard group constraints

The same as `where`, but evaluated once per group, after `where` has filtered
the bindings. It may use aggregates. It is a static error without `group`.

### 4.6 `soft`: fuzzy constraints and the score

`soft` combines graded conditions:

| Form | Degree |
|---|---|
| `x ~= y within 10%` | `max(0, 1 − |x − y| / (0.10 · max(|x|, |y|)))`: 1 when equal, 0 at 10% apart |
| `x ~= y within 500 USD` (or a duration) | `max(0, 1 − |x − y| / 500 USD)` |
| `x is term(args)` | the term's membership function at `x` |
| `x <= y` and other comparisons | 1 or 0 |
| `A and B` | the motif's t-norm (`min` or `product`) |
| `A or B` | `max` |
| `not A` | `1 − A` |
| `Q v: A` | quantifier `Q` over the bindings of `v` (§3.2) |

**`prev`** refers to the previous step of the path variable a quantifier
iterates. A predicate that mentions `prev` is skipped (it doesn't lower the
score) on the first step.

**The score of a candidate** is its `soft` expression with `and` evaluated by
the `score` t-norm. Without a `soft` clause the score is 1.

**The score of the anchor** for a motif is the best candidate score. That
candidate is the **witness**. A motif emits when the anchor score is at least
the `emit` threshold.

### 4.7 Output

For each anchor transfer and each motif that emits, the engine returns:

```json
{
  "motif": "round_trip",
  "score": 0.72,
  "witness": {"bindings": {"a": "0119/812A09CF0"}, "edges": ["...", "..."]},
  "degrees": [
    {"predicate": "each e: e.gap is soon", "degree": 0.91},
    {"predicate": "each e: e.amount ~= prev.amount within 20%", "degree": 0.72, "weakest_step": 3}
  ],
  "truncated": false
}
```

`degrees` is the explanation: which condition held the score down, and where.
`truncated` is true when a hub cap cut the search short. For path motifs
the score is then a lower bound (see design doc, §6.5); for group motifs
it is not — dropping members can raise non-monotone aggregates, so a
truncated group score may overstate.

## 5. Static checks

The compiler rejects a motif, with a message meant to be fed back to a model,
when:

1. **No time bound.** Every motif needs `span(...) <= D` in `where` or a
   `group … window D`.
2. **Unknown or mismatched units.** For example `e.amount ~= e.time within
   10%`, or a duration term applied to money.
3. **Fuzzy construct in a crisp clause.** See §4.3.
4. **Scoping errors.** These are unbound variables, `prev` outside a
   quantifier over a path variable, a quantifier over a non-member variable,
   `having` without `group`, or aggregates outside a group.
5. **Estimated cost above budget.** The planner computes a sanity bound on
   expansions per anchor from live degree statistics: mean in-degree powers
   summed to the max hops (measured ~1.7K mean expansions on HI-Small against
   a ~1e5 estimate for `round_trip`, so the bound is deliberately loose).
   It refuses motifs above the configured budget, saying which bound to
   tighten.

## 6. Why the fuzzy layer is cheap

With `min` or `product`, combining one more degree can only keep a score the
same or lower it, because every degree is at most 1. The matcher uses this in
two ways, depending on the kind of motif.

**Path motifs** (cycles, chains, butterflies). The matcher extends a partial
binding one edge at a time, backwards from the anchor. Many conditions are
**final once their variables are bound**: per-step conditions under
`each`/`all` (`each e: e.gap is soon`), and conditions on bound nodes and
edges. Their combined degree so far is an **upper bound** on the score of any
completion. The matcher drops a partial binding as soon as that bound falls
below the emit threshold, or below the best witness found so far when only
the best is needed. At the same threshold, a fuzzy motif then explores roughly
what the equivalent exact motif explores.

**Group motifs** (fan-in, fan-out, scatter-gather). Aggregates and the
relative quantifiers `most`, `few` and `@name` are **not** monotone as a group
grows: one more member can raise `count(distinct m) is many`. For these
motifs the engine does a short bounded lookup from the anchor to find the
group keys it touches. For scatter-gather, that means the senders into the
anchor's source account. It keeps each group's aggregates incrementally per
key and evaluates `having` and `soft` once per touched group. The cost is
bounded by the window and the hub caps rather than by pruning.

Conditions that are neither (`or`, `not`, `some` over an unfinished path) are
evaluated only once the binding is complete. In both cases the hard `where` /
`having` bounds keep the work finite, and the soft layer only ranks within
them.

## 7. Open problems for v1

- **Set-valued variables.** Needed for `stack` (two bipartite layers) and for
  general bipartite patterns beyond the 2×2 butterfly core.
- **Per-condition weights.** A weighted t-norm, so one weak condition doesn't
  dominate.
- **Relative-time terms.** For example "faster than this account usually
  moves money", using per-node baselines.
- **Learned membership functions.** Fit term breakpoints from labelled data
  instead of hand-setting them.
- **Other entity types.** Card, device and merchant nodes, to cover fraud
  patterns as well as laundering.
