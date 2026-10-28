"""RQ4 English test set (Oct 31): descriptions with hand-written references.

Each item is English precise enough to pin the motif (hops, windows,
thresholds, terms named), plus a standalone reference (terms inlined,
motif named `ref`). Correctness in RQ4 is witness-set Jaccard against
the reference on a train sample — tight English keeps valid-but-different
motifs rare, but Jaccard stays the secondary metric behind compile rate.
"""

from __future__ import annotations

ITEMS: list[dict[str, str]] = [
    # -- cycles -------------------------------------------------------
    {"id": "cycle-basic",
     "english": ("Money leaves an account and comes back to it within 7 days, "
                 "through 2 to 5 transfers. Every hop moves nearly the same "
                 "amount (within 20% of the previous hop) with patient gaps "
                 "(fully true to 1 day, false by 5 days). All conditions must "
                 "hold; fire at 0.5."),
     "reference": (
         "term patient = trapezoid(0h, 0h, 24h, 120h)\n"
         "motif ref { match (a:account) -[e:transfer*2..5]-> (a)\n"
         " where e is time_ordered and span(e) <= 7d\n"
         " soft each e: e.gap is patient\n"
         " and each e: e.amount ~= prev.amount within 20%\n"
         " score min emit when score >= 0.5 }")},
    {"id": "cycle-tight-window",
     "english": ("A quick cycle: money returns to its start within 48 hours "
                 "over 2 to 4 transfers. Gaps are soon (fully true to 6 "
                 "hours, false by 48 hours) and amounts stay within 10% hop "
                 "to hop. All conditions must hold; fire at 0.6."),
     "reference": (
         "term soon = trapezoid(0h, 0h, 6h, 48h)\n"
         "motif ref { match (a:account) -[e:transfer*2..4]-> (a)\n"
         " where e is time_ordered and span(e) <= 48h\n"
         " soft each e: e.gap is soon\n"
         " and each e: e.amount ~= prev.amount within 10%\n"
         " score min emit when score >= 0.6 }")},
    {"id": "cycle-long",
     "english": ("A slow long cycle: 3 to 6 transfers over up to 14 days. "
                 "Patient gaps (true to 1 day, false by 5), amounts within "
                 "25%. Conditions multiply; fire at 0.5."),
     "reference": (
         "term patient = trapezoid(0h, 0h, 24h, 120h)\n"
         "motif ref { match (a:account) -[e:transfer*3..6]-> (a)\n"
         " where e is time_ordered and span(e) <= 14d\n"
         " soft each e: e.gap is patient\n"
         " and each e: e.amount ~= prev.amount within 25%\n"
         " score product emit when score >= 0.5 }")},
    {"id": "cycle-amount-only",
     "english": ("A value-conserving cycle: 2 to 5 transfers within 7 days "
                 "where each hop moves within 15% of the previous one. No "
                 "gap condition. Fire at 0.5."),
     "reference": (
         "motif ref { match (a:account) -[e:transfer*2..5]-> (a)\n"
         " where e is time_ordered and span(e) <= 7d\n"
         " soft each e: e.amount ~= prev.amount within 15%\n"
         " score min emit when score >= 0.5 }")},
    {"id": "cycle-gap-only",
     "english": ("A fast cycle: 2 to 5 transfers within 7 days with soon gaps "
                 "(true to 6 hours, false by 48). No amount condition. "
                 "Fire at 0.5."),
     "reference": (
         "term soon = trapezoid(0h, 0h, 6h, 48h)\n"
         "motif ref { match (a:account) -[e:transfer*2..5]-> (a)\n"
         " where e is time_ordered and span(e) <= 7d\n"
         " soft each e: e.gap is soon\n"
         " score min emit when score >= 0.5 }")},
    {"id": "cycle-all-but-1",
     "english": ("A forgiving cycle: 2 to 6 transfers within 7 days where all "
                 "but one hop conserve the amount within 20%. Gaps patient "
                 "(true to 1 day, false by 5 days) on every hop. Fire at 0.5."),
     "reference": (
         "term patient = trapezoid(0h, 0h, 24h, 120h)\n"
         "motif ref { match (a:account) -[e:transfer*2..6]-> (a)\n"
         " where e is time_ordered and span(e) <= 7d\n"
         " soft each e: e.gap is patient\n"
         " and all but 1 e: e.amount ~= prev.amount within 20%\n"
         " score min emit when score >= 0.5 }")},
    {"id": "cycle-2hop-only",
     "english": ("A there-and-back pair: exactly 2 transfers that start and "
                 "end at the same account within 7 days, the second moving "
                 "within 20% of the first. Fire at 0.5."),
     "reference": (
         "motif ref { match (a:account) -[e:transfer*2..2]-> (a)\n"
         " where e is time_ordered and span(e) <= 7d\n"
         " soft each e: e.amount ~= prev.amount within 20%\n"
         " score min emit when score >= 0.5 }")},
    {"id": "cycle-emit-high",
     "english": ("A strict cycle: 2 to 5 transfers within 7 days, patient "
                 "gaps (true to 1 day, false by 5), amounts within 20%. "
                 "Fire only at 0.9 or above."),
     "reference": (
         "term patient = trapezoid(0h, 0h, 24h, 120h)\n"
         "motif ref { match (a:account) -[e:transfer*2..5]-> (a)\n"
         " where e is time_ordered and span(e) <= 7d\n"
         " soft each e: e.gap is patient\n"
         " and each e: e.amount ~= prev.amount within 20%\n"
         " score min emit when score >= 0.9 }")},
    {"id": "cycle-gap-product",
     "english": ("A cycle scored by product: 2 to 5 transfers within 7 days, "
                 "patient gaps (true to 1 day, false by 5), amounts within "
                 "20%, conditions multiplied. Fire at 0.5."),
     "reference": (
         "term patient = trapezoid(0h, 0h, 24h, 120h)\n"
         "motif ref { match (a:account) -[e:transfer*2..5]-> (a)\n"
         " where e is time_ordered and span(e) <= 7d\n"
         " soft each e: e.gap is patient\n"
         " and each e: e.amount ~= prev.amount within 20%\n"
         " score product emit when score >= 0.5 }")},
    {"id": "cycle-step-delay",
     "english": ("A steadily-paced cycle: 2 to 5 transfers within 7 days "
                 "where every hop follows the previous one by at most 12 "
                 "hours. Amounts within 20%. Fire at 0.5."),
     "reference": (
         "motif ref { match (a:account) -[e:transfer*2..5]-> (a)\n"
         " where e is time_ordered and span(e) <= 7d\n"
         " soft each e: e.time - prev.time <= 12h\n"
         " and each e: e.amount ~= prev.amount within 20%\n"
         " score min emit when score >= 0.5 }")},
    # -- chains -------------------------------------------------------
    {"id": "chain-basic",
     "english": ("A layering chain: money flows through 5 to 8 transfers to a "
                 "different account within 10 days. Patient gaps (true to 1 "
                 "day, false by 5 days), amounts within 25%. Fire at 0.5."),
     "reference": (
         "term patient = trapezoid(0h, 0h, 24h, 120h)\n"
         "motif ref { match (a:account) -[e:transfer*5..8]-> (z:account)\n"
         " where e is time_ordered and span(e) <= 10d and a != z\n"
         " soft each e: e.gap is patient\n"
         " and each e: e.amount ~= prev.amount within 25%\n"
         " score min emit when score >= 0.5 }")},
    {"id": "chain-short",
     "english": ("A short chain: 2 to 4 transfers to a different account "
                 "within 3 days, soon gaps (true to 6 hours, false by 48), "
                 "amounts within 20%. Fire at 0.5."),
     "reference": (
         "term soon = trapezoid(0h, 0h, 6h, 48h)\n"
         "motif ref { match (a:account) -[e:transfer*2..4]-> (z:account)\n"
         " where e is time_ordered and span(e) <= 3d and a != z\n"
         " soft each e: e.gap is soon\n"
         " and each e: e.amount ~= prev.amount within 20%\n"
         " score min emit when score >= 0.5 }")},
    {"id": "chain-quick",
     "english": ("A quick pass-through: 3 to 5 transfers to a different "
                 "account within 48 hours, quickly gaps (fully true to 1 "
                 "hour, false by 12), amounts within 15%. Fire at 0.5."),
     "reference": (
         "term quickly = trapezoid(0h, 0h, 1h, 12h)\n"
         "motif ref { match (a:account) -[e:transfer*3..5]-> (z:account)\n"
         " where e is time_ordered and span(e) <= 48h and a != z\n"
         " soft each e: e.gap is quickly\n"
         " and each e: e.amount ~= prev.amount within 15%\n"
         " score min emit when score >= 0.5 }")},
    {"id": "chain-amount-tight",
     "english": ("A value-tight chain: 4 to 6 transfers to a different "
                 "account within 10 days with amounts within 5% hop to hop "
                 "and patient gaps (true to 1 day, false by 5). Fire at 0.5."),
     "reference": (
         "term patient = trapezoid(0h, 0h, 24h, 120h)\n"
         "motif ref { match (a:account) -[e:transfer*4..6]-> (z:account)\n"
         " where e is time_ordered and span(e) <= 10d and a != z\n"
         " soft each e: e.gap is patient\n"
         " and each e: e.amount ~= prev.amount within 5%\n"
         " score min emit when score >= 0.5 }")},
    {"id": "chain-amount-only",
     "english": ("An amount-only chain: 3 to 6 transfers to a different "
                 "account within 7 days, amounts within 10%, no gap "
                 "condition. Fire at 0.5."),
     "reference": (
         "motif ref { match (a:account) -[e:transfer*3..6]-> (z:account)\n"
         " where e is time_ordered and span(e) <= 7d and a != z\n"
         " soft each e: e.amount ~= prev.amount within 10%\n"
         " score min emit when score >= 0.5 }")},
    {"id": "chain-patient-loose",
     "english": ("A loose chain: 2 to 5 transfers to a different account "
                 "within 10 days with patient gaps (true to 1 day, false by "
                 "5 days) and no amount condition. Fire at 0.5."),
     "reference": (
         "term patient = trapezoid(0h, 0h, 24h, 120h)\n"
         "motif ref { match (a:account) -[e:transfer*2..5]-> (z:account)\n"
         " where e is time_ordered and span(e) <= 10d and a != z\n"
         " soft each e: e.gap is patient\n"
         " score min emit when score >= 0.5 }")},
    # -- fans ---------------------------------------------------------
    {"id": "fanout-basic",
     "english": ("Fan-out: one account pays at least 2 distinct accounts "
                 "within 4 days. More payees score higher (0 at 3 or fewer, "
                 "1 at 8 or more). Fire at 0.5."),
     "reference": (
         "term many = ramp(3, 8)\n"
         "motif ref { match (s:account) -[e:transfer]-> (r:account)\n"
         " group by s over r window 4d\n"
         " having count(distinct r) >= 2\n"
         " soft count(distinct r) is many\n"
         " score product emit when score >= 0.5 }")},
    {"id": "fanout-big",
     "english": ("Big fan-out: one account pays at least 5 distinct accounts "
                 "within 7 days. More payees score higher (0 at 3 or fewer, "
                 "1 at 8 or more). Fire at 0.5."),
     "reference": (
         "term many = ramp(3, 8)\n"
         "motif ref { match (s:account) -[e:transfer]-> (r:account)\n"
         " group by s over r window 7d\n"
         " having count(distinct r) >= 5\n"
         " soft count(distinct r) is many\n"
         " score product emit when score >= 0.5 }")},
    {"id": "fanout-12h",
     "english": ("Burst fan-out: one account pays at least 2 distinct "
                 "accounts within 12 hours. Several payees score higher (0 "
                 "at 1, 1 at 3 or more). Fire at 0.5."),
     "reference": (
         "term several = ramp(1, 3)\n"
         "motif ref { match (s:account) -[e:transfer]-> (r:account)\n"
         " group by s over r window 12h\n"
         " having count(distinct r) >= 2\n"
         " soft count(distinct r) is several\n"
         " score product emit when score >= 0.5 }")},
    {"id": "fanout-at-least-amount",
     "english": ("Funded fan-out: one account pays at least 3 distinct "
                 "accounts within 4 days, and at least 3 of the payments are "
                 "100 USD or more. Fire at 0.5."),
     "reference": (
         "motif ref { match (s:account) -[e:transfer]-> (r:account)\n"
         " group by s over r window 4d\n"
         " having count(distinct r) >= 3\n"
         " soft at least 3 r: e.amount >= 100 USD\n"
         " score min emit when score >= 0.5 }")},
    {"id": "fanout-lots",
     "english": ("Huge fan-out: one account pays at least 4 distinct "
                 "accounts within 4 days, scoring 0 at 5 payees and 1 at 10 "
                 "or more. Fire at 0.5."),
     "reference": (
         "term lots = ramp(5, 10)\n"
         "motif ref { match (s:account) -[e:transfer]-> (r:account)\n"
         " group by s over r window 4d\n"
         " having count(distinct r) >= 4\n"
         " soft count(distinct r) is lots\n"
         " score product emit when score >= 0.5 }")},
    {"id": "fanin-basic",
     "english": ("Fan-in: at least 2 distinct accounts pay one account "
                 "within 4 days. More payers score higher (0 at 3 or fewer, "
                 "1 at 8 or more). Fire at 0.5."),
     "reference": (
         "term many = ramp(3, 8)\n"
         "motif ref { match (x:account) -[e:transfer]-> (t:account)\n"
         " group by t over x window 4d\n"
         " having count(distinct x) >= 2\n"
         " soft count(distinct x) is many\n"
         " score product emit when score >= 0.5 }")},
    {"id": "fanin-most-small",
     "english": ("Small-payments fan-in: at least 3 distinct accounts pay "
                 "one account within 4 days, and most payments are 500 USD "
                 "or less. Fire at 0.5."),
     "reference": (
         "motif ref { match (x:account) -[e:transfer]-> (t:account)\n"
         " group by t over x window 4d\n"
         " having count(distinct x) >= 3\n"
         " soft most x: e.amount <= 500 USD\n"
         " score min emit when score >= 0.5 }")},
    {"id": "fanin-count-most",
     "english": ("Busy small fan-in: at least 3 distinct accounts pay one "
                 "account within 4 days; more payers score higher (0 at 3 or "
                 "fewer, 1 at 8 or more) and most payments are 1000 USD or "
                 "less. Both hold; fire at 0.5."),
     "reference": (
         "term many = ramp(3, 8)\n"
         "motif ref { match (x:account) -[e:transfer]-> (t:account)\n"
         " group by t over x window 4d\n"
         " having count(distinct x) >= 3\n"
         " soft count(distinct x) is many\n"
         " and most x: e.amount <= 1000 USD\n"
         " score min emit when score >= 0.5 }")},
    {"id": "fan-mostly-large",
     "english": ("Mostly-large fan-out: one account pays at least 2 distinct "
                 "accounts within 4 days, and a mostly share (ramp 0.6 to "
                 "0.9 of the mean) of the payments are 50 USD or more. "
                 "Fire at 0.6."),
     "reference": (
         "quantifier mostly = ramp(0.6, 0.9)\n"
         "motif ref { match (s:account) -[e:transfer]-> (r:account)\n"
         " group by s over r window 4d\n"
         " having count(distinct r) >= 2\n"
         " soft @mostly r: e.amount >= 50 USD\n"
         " score min emit when score >= 0.6 }")},
    {"id": "fan-few-large",
     "english": ("Few-large fan-out: one account pays at least 2 distinct "
                 "accounts within 4 days, and few of the payments reach "
                 "10000 USD. Fire at 0.5."),
     "reference": (
         "motif ref { match (s:account) -[e:transfer]-> (r:account)\n"
         " group by s over r window 4d\n"
         " having count(distinct r) >= 2\n"
         " soft few r: e.amount >= 10000 USD\n"
         " score min emit when score >= 0.5 }")},
    {"id": "structuring-basic",
     "english": ("Structuring: at least 3 distinct accounts pay one account "
                 "within 4 days; more payers score higher (0 at 3 or fewer, "
                 "1 at 8 or more) and most payments sit just below 10000 USD "
                 "(rising from 8500, full at 9500 to 9990, cut at 10000). "
                 "Fire at 0.6."),
     "reference": (
         "term many = ramp(3, 8)\n"
         "term just_below(limit) = trapezoid(0.85 * limit, 0.95 * limit, 0.999 * limit, limit)\n"
         "motif ref { match (x:account) -[e:transfer]-> (t:account)\n"
         " group by t over x window 4d\n"
         " having count(distinct x) >= 3\n"
         " soft count(distinct x) is many\n"
         " and most x: e.amount is just_below(10000 USD)\n"
         " score min emit when score >= 0.6 }")},
    {"id": "structuring-5k",
     "english": ("Small structuring: at least 3 distinct accounts pay one "
                 "account within 4 days, and most payments sit just below "
                 "5000 USD (rising from 4250, full at 4750 to 4995, cut at "
                 "5000). Fire at 0.6."),
     "reference": (
         "term just_below(limit) = trapezoid(0.85 * limit, 0.95 * limit, 0.999 * limit, limit)\n"
         "motif ref { match (x:account) -[e:transfer]-> (t:account)\n"
         " group by t over x window 4d\n"
         " having count(distinct x) >= 3\n"
         " soft most x: e.amount is just_below(5000 USD)\n"
         " score min emit when score >= 0.6 }")},
    # -- scatter ------------------------------------------------------
    {"id": "scatter-basic",
     "english": ("Scatter-gather: one source pays at least 2 intermediaries "
                 "who forward to one destination within 7 days. Several "
                 "intermediaries score higher (0 at 1, 1 at 3 or more) and "
                 "the forwarded sums match the received sums within 20%. "
                 "Fire at 0.5."),
     "reference": (
         "term several = ramp(1, 3)\n"
         "motif ref { match (s:account) -[leg1:transfer]-> (m:account) -[leg2:transfer]-> (d:account)\n"
         " group by s, d over m window 7d\n"
         " where leg2.time > leg1.time and s != d\n"
         " having count(distinct m) >= 2\n"
         " soft count(distinct m) is several\n"
         " and sum(leg2.amount) ~= sum(leg1.amount) within 20%\n"
         " score product emit when score >= 0.5 }")},
    {"id": "scatter-count-only",
     "english": ("Wide scatter: at least 3 intermediaries between one source "
                 "and one destination within 4 days. More intermediaries "
                 "score higher (0 at 3 or fewer, 1 at 8 or more). Fire "
                 "at 0.5."),
     "reference": (
         "term many = ramp(3, 8)\n"
         "motif ref { match (s:account) -[leg1:transfer]-> (m:account) -[leg2:transfer]-> (d:account)\n"
         " group by s, d over m window 4d\n"
         " where leg2.time > leg1.time and s != d\n"
         " having count(distinct m) >= 3\n"
         " soft count(distinct m) is many\n"
         " score product emit when score >= 0.5 }")},
    {"id": "scatter-gap",
     "english": ("Quick-forward scatter: at least 2 intermediaries between "
                 "one source and one destination within 2 days, where most "
                 "intermediaries forward soon after receiving (fully true to "
                 "6 hours, false by 48). Fire at 0.5."),
     "reference": (
         "term soon = trapezoid(0h, 0h, 6h, 48h)\n"
         "motif ref { match (s:account) -[leg1:transfer]-> (m:account) -[leg2:transfer]-> (d:account)\n"
         " group by s, d over m window 2d\n"
         " where leg2.time > leg1.time and s != d\n"
         " having count(distinct m) >= 2\n"
         " soft most m: leg2.gap is soon\n"
         " score min emit when score >= 0.5 }")},
    {"id": "scatter-sums-tight",
     "english": ("Tight scatter: at least 2 intermediaries between one "
                 "source and one destination within 7 days, with forwarded "
                 "sums matching received sums within 10%. Fire at 0.5."),
     "reference": (
         "motif ref { match (s:account) -[leg1:transfer]-> (m:account) -[leg2:transfer]-> (d:account)\n"
         " group by s, d over m window 7d\n"
         " where leg2.time > leg1.time and s != d\n"
         " having count(distinct m) >= 2\n"
         " soft sum(leg2.amount) ~= sum(leg1.amount) within 10%\n"
         " score min emit when score >= 0.5 }")},
    {"id": "scatter-6h-window",
     "english": ("Flash scatter: at least 2 intermediaries between one "
                 "source and one destination within 6 hours. Several "
                 "intermediaries score higher (0 at 1, 1 at 3 or more). "
                 "Fire at 0.5."),
     "reference": (
         "term several = ramp(1, 3)\n"
         "motif ref { match (s:account) -[leg1:transfer]-> (m:account) -[leg2:transfer]-> (d:account)\n"
         " group by s, d over m window 6h\n"
         " where leg2.time > leg1.time and s != d\n"
         " having count(distinct m) >= 2\n"
         " soft count(distinct m) is several\n"
         " score product emit when score >= 0.5 }")},
    {"id": "scatter-all-but-1",
     "english": ("Conserving scatter: at least 3 intermediaries between one "
                 "source and one destination within 7 days, where all but "
                 "one intermediary forwards within 20% of what it received. "
                 "Fire at 0.5."),
     "reference": (
         "motif ref { match (s:account) -[leg1:transfer]-> (m:account) -[leg2:transfer]-> (d:account)\n"
         " group by s, d over m window 7d\n"
         " where leg2.time > leg1.time and s != d\n"
         " having count(distinct m) >= 3\n"
         " soft all but 1 m: leg2.amount ~= leg1.amount within 20%\n"
         " score min emit when score >= 0.5 }")},
    {"id": "scatter-time-diff",
     "english": ("Same-day-forward scatter: at least 2 intermediaries "
                 "between one source and one destination within 7 days, "
                 "where most intermediaries forward within 6 hours of "
                 "receiving. Fire at 0.5."),
     "reference": (
         "motif ref { match (s:account) -[leg1:transfer]-> (m:account) -[leg2:transfer]-> (d:account)\n"
         " group by s, d over m window 7d\n"
         " where leg2.time > leg1.time and s != d\n"
         " having count(distinct m) >= 2\n"
         " soft most m: leg2.time - leg1.time <= 6h\n"
         " score min emit when score >= 0.5 }")},
    {"id": "scatter-single-member",
     "english": ("Single-hop relay: one source pays one intermediary who "
                 "forwards to one destination within 7 days, the forwarded "
                 "sum matching the received sum within 20%. Fire at 0.5."),
     "reference": (
         "motif ref { match (s:account) -[leg1:transfer]-> (m:account) -[leg2:transfer]-> (d:account)\n"
         " group by s, d over m window 7d\n"
         " where leg2.time > leg1.time and s != d\n"
         " having count(distinct m) >= 1\n"
         " soft sum(leg2.amount) ~= sum(leg1.amount) within 20%\n"
         " score min emit when score >= 0.5 }")},
    {"id": "scatter-full",
     "english": ("Full scatter: at least 2 intermediaries between one source "
                 "and one destination within 7 days; several intermediaries "
                 "score higher (0 at 1, 1 at 3+), sums match within 20%, and "
                 "most forward soon (true to 6 hours, false by 48). All "
                 "hold; fire at 0.5."),
     "reference": (
         "term soon = trapezoid(0h, 0h, 6h, 48h)\n"
         "term several = ramp(1, 3)\n"
         "motif ref { match (s:account) -[leg1:transfer]-> (m:account) -[leg2:transfer]-> (d:account)\n"
         " group by s, d over m window 7d\n"
         " where leg2.time > leg1.time and s != d\n"
         " having count(distinct m) >= 2\n"
         " soft count(distinct m) is several\n"
         " and sum(leg2.amount) ~= sum(leg1.amount) within 20%\n"
         " and most m: leg2.gap is soon\n"
         " score min emit when score >= 0.5 }")},
    # -- gather -------------------------------------------------------
    {"id": "gather-basic",
     "english": ("Gather-scatter: one account collects from at least 2 "
                 "distinct accounts and pays at least 2 distinct accounts "
                 "within 4 days. Several on each side score higher (0 at 1, "
                 "1 at 3 or more) and the two counts balance within 4. "
                 "Fire at 0.5."),
     "reference": (
         "term several = ramp(1, 3)\n"
         "motif ref { match (x:account) -[g:transfer]-> (c:account), (c) -[p:transfer]-> (y:account)\n"
         " group by c over x, y window 4d\n"
         " where p.time > g.time\n"
         " having count(distinct x) >= 2 and count(distinct y) >= 2\n"
         " soft count(distinct x) is several\n"
         " and count(distinct y) is several\n"
         " and count(distinct x) ~= count(distinct y) within 4\n"
         " score product emit when score >= 0.5 }")},
    {"id": "gather-count-only",
     "english": ("Collector: one account collects from at least 3 distinct "
                 "accounts within 4 days. More collectors score higher (0 at "
                 "3 or fewer, 1 at 8 or more). Fire at 0.5."),
     "reference": (
         "term many = ramp(3, 8)\n"
         "motif ref { match (x:account) -[g:transfer]-> (c:account), (c) -[p:transfer]-> (y:account)\n"
         " group by c over x, y window 4d\n"
         " where p.time > g.time\n"
         " having count(distinct x) >= 3\n"
         " soft count(distinct x) is many\n"
         " score product emit when score >= 0.5 }")},
    {"id": "gather-sums",
     "english": ("Redistributor: one account collects from at least 2 "
                 "distinct accounts and pays at least 2 distinct accounts "
                 "within 4 days, with paid-out sums matching collected sums "
                 "within 20%. Fire at 0.5."),
     "reference": (
         "motif ref { match (x:account) -[g:transfer]-> (c:account), (c) -[p:transfer]-> (y:account)\n"
         " group by c over x, y window 4d\n"
         " where p.time > g.time\n"
         " having count(distinct x) >= 2 and count(distinct y) >= 2\n"
         " soft sum(p.amount) ~= sum(g.amount) within 20%\n"
         " score min emit when score >= 0.5 }")},
    {"id": "gather-wide",
     "english": ("Slow gather-scatter: one account collects from at least 2 "
                 "distinct accounts and pays at least 2 distinct accounts "
                 "within 7 days. Several on each side score higher (0 at 1, "
                 "1 at 3 or more). Fire at 0.5."),
     "reference": (
         "term several = ramp(1, 3)\n"
         "motif ref { match (x:account) -[g:transfer]-> (c:account), (c) -[p:transfer]-> (y:account)\n"
         " group by c over x, y window 7d\n"
         " where p.time > g.time\n"
         " having count(distinct x) >= 2 and count(distinct y) >= 2\n"
         " soft count(distinct x) is several\n"
         " and count(distinct y) is several\n"
         " score product emit when score >= 0.5 }")},
    {"id": "gather-emit-low",
     "english": ("Loose gather-scatter: one account collects from at least 2 "
                 "distinct accounts and pays at least 2 distinct accounts "
                 "within 4 days, with the two counts balancing within 10. "
                 "Fire at 0.3."),
     "reference": (
         "motif ref { match (x:account) -[g:transfer]-> (c:account), (c) -[p:transfer]-> (y:account)\n"
         " group by c over x, y window 4d\n"
         " where p.time > g.time\n"
         " having count(distinct x) >= 2 and count(distinct y) >= 2\n"
         " soft count(distinct x) ~= count(distinct y) within 10\n"
         " score min emit when score >= 0.3 }")},
    {"id": "gather-at-least-payout",
     "english": ("Big-payout gather: one account collects from at least 2 "
                 "distinct accounts and pays at least 2 distinct accounts "
                 "within 4 days, with at least 2 payouts of 1000 USD or "
                 "more. Fire at 0.5."),
     "reference": (
         "motif ref { match (x:account) -[g:transfer]-> (c:account), (c) -[p:transfer]-> (y:account)\n"
         " group by c over x, y window 4d\n"
         " where p.time > g.time\n"
         " having count(distinct x) >= 2 and count(distinct y) >= 2\n"
         " soft at least 2 y: p.amount >= 1000 USD\n"
         " score min emit when score >= 0.5 }")},
    # -- butterfly ----------------------------------------------------
    {"id": "butterfly-soon",
     "english": ("Quick bipartite core: two distinct senders each pay the "
                 "same two distinct receivers, all within 2 days and soon "
                 "(fully true to 6 hours, false by 48). Fire at 0.6."),
     "reference": (
         "term soon = trapezoid(0h, 0h, 6h, 48h)\n"
         "motif ref { match (x1:account) -[e1:transfer]-> (y1:account), (x1) -[e2:transfer]-> (y2:account), (x2:account) -[e3:transfer]-> (y1), (x2) -[e4:transfer]-> (y2)\n"
         " where x1 != x2 and y1 != y2 and span(all) <= 2d\n"
         " soft span(all) is soon\n"
         " score min emit when score >= 0.6 }")},
    {"id": "butterfly-patient",
     "english": ("Slow bipartite core: two distinct senders each pay the "
                 "same two distinct receivers within 5 days, patiently "
                 "(true to 1 day, false by 5). Fire at 0.5."),
     "reference": (
         "term patient = trapezoid(0h, 0h, 24h, 120h)\n"
         "motif ref { match (x1:account) -[e1:transfer]-> (y1:account), (x1) -[e2:transfer]-> (y2:account), (x2:account) -[e3:transfer]-> (y1), (x2) -[e4:transfer]-> (y2)\n"
         " where x1 != x2 and y1 != y2 and span(all) <= 5d\n"
         " soft span(all) is patient\n"
         " score min emit when score >= 0.5 }")},
    {"id": "butterfly-quick",
     "english": ("Flash bipartite core: two distinct senders each pay the "
                 "same two distinct receivers within 12 hours, quickly "
                 "(fully true to 1 hour, false by 12). Fire at 0.6."),
     "reference": (
         "term quickly = trapezoid(0h, 0h, 1h, 12h)\n"
         "motif ref { match (x1:account) -[e1:transfer]-> (y1:account), (x1) -[e2:transfer]-> (y2:account), (x2:account) -[e3:transfer]-> (y1), (x2) -[e4:transfer]-> (y2)\n"
         " where x1 != x2 and y1 != y2 and span(all) <= 12h\n"
         " soft span(all) is quickly\n"
         " score min emit when score >= 0.6 }")},
    {"id": "butterfly-not-soon",
     "english": ("Delayed bipartite core: two distinct senders each pay the "
                 "same two distinct receivers within 5 days, but NOT soon — "
                 "the longer the span past 6 hours toward 48, the better. "
                 "Fire at 0.5."),
     "reference": (
         "term soon = trapezoid(0h, 0h, 6h, 48h)\n"
         "motif ref { match (x1:account) -[e1:transfer]-> (y1:account), (x1) -[e2:transfer]-> (y2:account), (x2:account) -[e3:transfer]-> (y1), (x2) -[e4:transfer]-> (y2)\n"
         " where x1 != x2 and y1 != y2 and span(all) <= 5d\n"
         " soft span(all) is not soon\n"
         " score min emit when score >= 0.5 }")},
    {"id": "fanout-most-big",
     "english": ("Big-payment fan-out: one account pays at least 2 distinct "
                 "accounts within 4 days, and most payments are big (0 below "
                 "1000 USD, rising to 1 at 5000). Fire at 0.5."),
     "reference": (
         "term big = trapezoid(1000 USD, 5000 USD, 1000000 USD, 1000001 USD)\n"
         "motif ref { match (s:account) -[e:transfer]-> (r:account)\n"
         " group by s over r window 4d\n"
         " having count(distinct r) >= 2\n"
         " soft most r: e.amount is big\n"
         " score min emit when score >= 0.5 }")},
    {"id": "gather-balance-tight",
     "english": ("Balanced gather-scatter: one account collects from at "
                 "least 2 distinct accounts and pays at least 2 distinct "
                 "accounts within 4 days, with the two counts balancing "
                 "within 2. Fire at 0.5."),
     "reference": (
         "motif ref { match (x:account) -[g:transfer]-> (c:account), (c) -[p:transfer]-> (y:account)\n"
         " group by c over x, y window 4d\n"
         " where p.time > g.time\n"
         " having count(distinct x) >= 2 and count(distinct y) >= 2\n"
         " soft count(distinct x) ~= count(distinct y) within 2\n"
         " score min emit when score >= 0.5 }")},
    {"id": "chain-all-but-1",
     "english": ("A forgiving chain: 3 to 6 transfers to a different account "
                 "within 7 days where all but one hop conserve the amount "
                 "within 20%, with patient gaps (true to 1 day, false by 5) "
                 "on every hop. Fire at 0.5."),
     "reference": (
         "term patient = trapezoid(0h, 0h, 24h, 120h)\n"
         "motif ref { match (a:account) -[e:transfer*3..6]-> (z:account)\n"
         " where e is time_ordered and span(e) <= 7d and a != z\n"
         " soft each e: e.gap is patient\n"
         " and all but 1 e: e.amount ~= prev.amount within 20%\n"
         " score min emit when score >= 0.5 }")},
]
