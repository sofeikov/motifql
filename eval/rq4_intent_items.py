"""RQ4 loose-intent tier (Nov 8, issue #23): underdetermined English.

The precise tier (`rq4_items.py`) dictates hops, windows, thresholds and
breakpoints — it measures "renders a complete spec". These 10 items are
the plain-English pitch ("money that comes back after a while"): no hop
counts, no windows, no thresholds (pinned by test: no digits, no units).

Each item carries:

* ``english`` — the loose description the model compiles.
* ``references`` — 1-2 acceptable hand-written motifs (a DISJUNCTION,
  not one gold reading). Reused from the precise tier by id so the
  oracle stays single-sourced. Full-tier Jaccard is the max over refs.
* ``rubric`` — three tight checks for manual grading: is the shape
  right, is the window sane, is the expected condition present?
  A motif passes iff all three hold. Grades live in
  ``eval/rq4_intent_grades.csv``; the claim is scoped as
  "intent tier is rubric-graded, n=10".

Report compile rate + rubric pass rate SEPARATELY from the precise
tier; never merge the numbers (``eval/rq4.py --tier`` keeps separate
runs CSVs).
"""

from __future__ import annotations

from eval.rq4_items import ITEMS as _PRECISE

_REF = {it["id"]: it["reference"] for it in _PRECISE}

INTENT_ITEMS: list[dict] = [
    {
        "id": "intent-roundtrip",
        "english": ("Money that leaves an account and comes back after "
                    "passing through a few others."),
        "references": [_REF["cycle-basic"], _REF["cycle-long"]],
        "rubric": [
            {"id": "shape",
             "criterion": "Cycle returning to the start account "
                          "(`(a) -[e]-> (a)` or equivalent)."},
            {"id": "window",
             "criterion": "Span bound present, between 1d and 30d."},
            {"id": "amount",
             "criterion": "A hop-to-hop amount-conservation condition "
                          "(`~=` on amounts) in soft."},
        ],
    },
    {
        "id": "intent-quick-return",
        "english": "Money that comes back to where it started pretty quickly.",
        "references": [_REF["cycle-tight-window"], _REF["cycle-step-delay"]],
        "rubric": [
            {"id": "shape",
             "criterion": "Cycle returning to the start account."},
            {"id": "window",
             "criterion": "A short span bound (at most 7d) or a "
                          "short per-hop gap bound (at most 24h)."},
            {"id": "gap",
             "criterion": "A gap/timing condition in soft "
                          "(term on `e.gap` or a time-difference bound)."},
        ],
    },
    {
        "id": "intent-layering",
        "english": ("Money passed along a chain of accounts, each one "
                    "forwarding it to the next."),
        "references": [_REF["chain-basic"], _REF["chain-short"]],
        "rubric": [
            {"id": "shape",
             "criterion": "Open chain to a different account "
                          "(`a != z` or two distinct endpoints)."},
            {"id": "length",
             "criterion": "Hop range allows at least 3 hops "
                          "(max bound >= 3)."},
            {"id": "condition",
             "criterion": "An amount-conservation or gap condition "
                          "in soft."},
        ],
    },
    {
        "id": "intent-fanout",
        "english": ("One account paying a bunch of different accounts "
                    "around the same time."),
        "references": [_REF["fanout-basic"], _REF["fanout-big"]],
        "rubric": [
            {"id": "shape",
             "criterion": "Fan-out grouped by the sender "
                          "(`group by s over r`)."},
            {"id": "count",
             "criterion": "`having count(distinct r) >= K` with K >= 2."},
            {"id": "window",
             "criterion": "Group window present, at most 14d."},
        ],
    },
    {
        "id": "intent-fanin",
        "english": "Lots of different accounts all paying into the same account.",
        "references": [_REF["fanin-basic"], _REF["fanin-count-most"]],
        "rubric": [
            {"id": "shape",
             "criterion": "Fan-in grouped by the recipient "
                          "(`group by t over x`)."},
            {"id": "count",
             "criterion": "`having count(distinct x) >= K` with K >= 2."},
            {"id": "window",
             "criterion": "Group window present, at most 14d."},
        ],
    },
    {
        "id": "intent-scatter",
        "english": ("Money split out from one account through several "
                    "middlemen who each forward it to one destination."),
        "references": [_REF["scatter-basic"], _REF["scatter-count-only"]],
        "rubric": [
            {"id": "shape",
             "criterion": "Scatter `s -> m -> d` with `group by s, d over m`."},
            {"id": "mids",
             "criterion": "`having count(distinct m) >= K` with K >= 2."},
            {"id": "condition",
             "criterion": "A sums-conservation (`sum(leg2) ~ sum(leg1)`) "
                          "or count-term condition in soft."},
        ],
    },
    {
        "id": "intent-gather",
        "english": ("An account that collects money from several places and "
                    "then pays it out to several others."),
        "references": [_REF["gather-basic"], _REF["gather-sums"]],
        "rubric": [
            {"id": "shape",
             "criterion": "Gather with `group by c over x, y`."},
            {"id": "sides",
             "criterion": "Both `count(distinct x) >= 2` and "
                          "`count(distinct y) >= 2` in having."},
            {"id": "condition",
             "criterion": "A balance/sums/count condition in soft."},
        ],
    },
    {
        "id": "intent-structuring",
        "english": ("Several people each sending an amount just under the "
                    "reporting limit to the same account."),
        "references": [_REF["structuring-basic"], _REF["structuring-5k"]],
        "rubric": [
            {"id": "shape",
             "criterion": "Fan-in grouped by the recipient."},
            {"id": "amounts",
             "criterion": "Amounts graded near a round limit (a "
                          "`just_below`-style term or a range hugging "
                          "5k/10k USD)."},
            {"id": "quantifier",
             "criterion": "A `most`/`at least`/named quantifier over the "
                          "payers for the amount condition."},
        ],
    },
    {
        "id": "intent-slow-drift",
        "english": "Money drifting slowly from account to account over a long while.",
        "references": [_REF["chain-patient-loose"], _REF["chain-basic"]],
        "rubric": [
            {"id": "shape",
             "criterion": "Open chain to a different account."},
            {"id": "window",
             "criterion": "Span bound present, at least 3d (slow drift "
                          "needs room) and at most 60d."},
            {"id": "gap",
             "criterion": "A gap condition in soft (term on `e.gap`)."},
        ],
    },
    {
        "id": "intent-backandforth",
        "english": "Two accounts bouncing money back and forth between them.",
        "references": [_REF["cycle-2hop-only"], _REF["cycle-basic"]],
        "rubric": [
            {"id": "shape",
             "criterion": "Cycle returning to the start account."},
            {"id": "small",
             "criterion": "Small hop range (max bound at most 4) — a pair, "
                          "not a long tour."},
            {"id": "amount",
             "criterion": "A hop-to-hop amount-conservation condition "
                          "in soft."},
        ],
    },
]
