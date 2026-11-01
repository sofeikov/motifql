# Rehearsal script (Nov 3 demo UI → Nov 4 video)

Video is ~3 min. Judges don't know AML or fraud — so the first 45
seconds sell *why this matters* before a single line of MotifQL
appears. Rule of thumb: problem → why it's hard → demo → proof.

Required opening line (hackathon provenance, DESIGN.md §2):

> "ezrules, my open-source rules engine, and IBM's AMLworld data
> existed before this week; everything you'll see — the MotifQL
> language, the engine and the Muse Spark integration — I built
> this week."

## The 30-second version (if they watch nothing else)

> "Up to $2 trillion is laundered every year — 2 to 5% of global
> GDP. Banks spend $20 billion a year on software to catch it and
> still pay billions in fines. Why? Because laundering is a *team
> sport* — money hopping through a network of accounts — while
> today's software looks at *one transfer at a time*. MotifQL fixes
> that: analysts describe the shape in English, AI translates it
> into a query, and a fuzzy graph search engine finds the money
> trail across millions of transfers — showing its work on every
> alert."

## Full script (~180s)

### 0:00–0:20 — Hook: the cost of the problem (title cards, no demo)

**Screen:** fullscreen title cards over black — no app, no
browser chrome. Four cards, one number each in big type with a
one-line label underneath.
**Action:** hold each card ~4–5s, cutting on the narration
beats: $2T laundered (2–5% of GDP, UNODC) → $16.6B US fraud
2024, +33% (FBI IC3) → $4–6B/year in bank fines (Fenergo) →
~$20B monitoring market, growing past $50B (analyst range).
Numbers only on screen — no jargon yet.

> "Every year, criminals launder up to **$2 trillion** — 2 to 5%
> of everything the world produces. Drugs, scams, corruption —
> it all needs washing before it can be spent. [UNODC estimate]
>
> In the US alone, reported cyber-fraud losses hit **$16.6
> billion** in 2024, up 33% in one year. [FBI IC3]
>
> Banks are legally required to catch this. When they miss, they
> pay: **$4–6 billion a year** in fines. [Fenergo]
>
> And they spend **~$20 billion a year** on transaction-monitoring
> software — growing to $50 billion-plus by the early 2030s.
> [Fortune/Mordor] That's the market we're in."

### 0:20–0:50 — Why it's hard: crime is social, software is lonely

**Screen:** simple slides (still no app). First a single
A→B→C→A triangle diagram that builds edge by edge; then a
split frame — left: one transfer row ("$9,000 — looks fine"),
right: the full triangle ("looks criminal").
**Action:** build the triangle while saying "picture it"; cut
to the split frame on "each single transfer looks innocent"
and hold it through the "one transfer at a time" paragraph.
Slow down — this is the talk's one idea.

> "Here's the thing judges should take away: **money laundering
> and fraud are fundamentally social activities.** Nobody launders
> money alone. It takes a *network* — mules, shell accounts,
> chains of hops, money scattered out and gathered back.
>
> Picture it: dirty money enters account A, splits to B, C, D,
> hops through a few more, and quietly lands back where it
> started — clean. Each single transfer looks totally innocent.
> A $9,000 payment? Normal. The *shape across accounts over
> time* is the tell.
>
> But today's monitoring looks at **one transfer at a time**,
> plus per-account totals. Totals can't see that the money came
> back. And the old fix — exact rules like 'every hop within 48
> hours and within 10%' — breaks the moment a criminal waits 50
> hours. Which is exactly what they do."

### 0:50–1:10 — The solution in one breath (landing → workspace)

**Screen:** the app landing — page title, one centered chat
box pre-filled with *"Money leaves an account and comes back
to it within 7 days…"*, six "Try one" chips below, footer
with "5,078,345 transfers · 8 motifs". After Enter: the
workspace — left rail (English box, Motif panel, witness
tabs), center stage (graph + score line + caption +
transport), right rail (Why-this-score bars, narrative,
evasion sliders).
**Action:** manual take (`?autorun=0`): press Enter on
"describe the criminal shape" — the UI dives into the
workspace (~0.3s) and the trace starts playing by itself.
Say the translator/engine distinction over the dive and the
first seconds of motion. (Auto-run plays this alone: landing
reads ~1.1s, then the dive; `?landing=0` skips it.)

> "MotifQL: describe the criminal shape in plain English — and a
> search engine finds the money trail across millions of
> transfers.
>
> Two parts are important here. Muse Spark is the
> *translator*: it turns the sentence into a query. The *engine*
> is the product: it compiles that query, searches 5 million
> transfers as a time-ordered graph, fuzzy-scores every
> candidate, and returns the single best money trail — with
> proof. The model never touches the money. All finding happens
> in the engine."

### 1:10–1:25 — Beat: Spark writes the motif (~15s)

**Screen:** left-rail Motif panel holding the compiled
detector plus the source badge (*scripted · Muse Spark
output* — a committed, verified model reply, so the take
needs no network). Stage parked at frame 0 (separate take
with `?autoplay=0`) so nothing moves behind the explanation.
**Action:** cursor tours the three parts top to bottom, one
phrase each — `match` (the shape), `soft` (the fuzzy
conditions), `emit` (the threshold) — and lands on the badge
at the end. Badge meanings: *live* = just compiled, *cached*
= prompt-hash cache hit, *scripted* = committed fallback.

Narration:

> "One English sentence in, one detector out: the shape to
> match, the fuzzy conditions, the threshold. The analyst never
> touches query code — and Spark's job is done. Everything after
> this is the engine."

### 1:25–1:50 — Beat: the trace — the engine shows its work (~25s)

**Screen:** center stage — the triangle draws hop by hop on
train CYCLE instance 102 (amber active edge, amount + gap
labels, from → to caption below). Around it: score line
"0.718 FIRES · 51 expansions", *Why this score* bars (gap
0.841, amount 0.718), verdict lede + investigator paragraph
with the "Weakest step: hop 3" badge under the stage, and the
transport (step / play / scrub / clock / Loop).
**Action:** fresh load, hands off — let the 4.7s trace play
untouched, hold the end frame ("Trace complete — 3 hops,
span 2.9d") 2–3s. Gesture at the score meta on "search
work". Transport stepping or `?at=SECONDS` stills cover any
frozen frame; `?loop=1` (or the Loop tick) holds a living
frame instead.

Narration:

> "And here's the payoff — all engine, no LLM. Not a risk score
> from a black box, but the actual money trail the search found:
> hop by hop, amounts, delays, how strongly each step matches.
> The score line even shows the search work — dozens of graph
> expansions behind this one witness. And on top, a one-line
> verdict saying why it matches, plus the full investigator
> paragraph with the weakest link flagged."

### 1:50–2:25 — Beat: evasion (~35s)

**Screen:** start from the completed-triangle still
(`?at=99`): closed triangle, evasion panel with **Amount
noise** (0–30%) and **Gap scale** (×1–×2) sliders at rest,
fuzzy vs exact bars side by side, live caption underneath.
**Action:** slow drag of Amount noise 0→+15% — fuzzy bar
sinks 0.718→0.357 while exact (α=emit) has already cliffed
1→0; pause 2s on the caption "+15% noise: fuzzy 0.718→0.357,
exact 1→0" as the *Why this score* amount bar sinks with it.
Reset noise to 0, drag Gap scale to ×2.0 — fuzzy 0.718→0.431,
exact 1→0 again. Sliders re-score live in <500ms; the
scripted anchor falls back to a committed curve, so this beat
needs no key either. Cursor visible throughout.

Narration:

> "Now watch what happens when the criminal adapts — amounts
> wobbled by 15%. The old-style exact rule falls off a cliff:
> one to zero. Gone. Our fuzzy score degrades gracefully,
> 0.72 to 0.36 — still flagging it. Same when they slow down:
> double the delays, and the exact rule dies while we still
> catch it. This is the engine's fuzzy matcher at work — every
> condition grades 0 to 1 instead of passing or failing.
> Criminals evade thresholds for a living. Don't give them one."

Stills (trace end + slider state deep-link together):
`?at=99&noise=15&gap=1` (mid-attack),
`?at=99&noise=0&gap=2` (gap cliff),
`?at=99` (clean witness).

### 2:25–2:50 — Beat: proof (results card, ~25s)

**Screen:** one results slide (app gone): 2-row table —
CYCLE 24-vs-1, RANDOM 13-vs-0 at 50 alerts/day — plus a
2-line engine footer (exact best-trail search, ~15× stream
headroom; 47/47 English → working detectors).
**Action:** hard cut from the app; read the two lines; hold
the slide 3s. (Full table + honest limits live in README —
linked, never read on camera.)

> "On the standard IBM benchmark, at 50 alerts a day — a real
> analyst's workload — MotifQL catches the structural patterns a
> strong machine-learning baseline barely sees: 24 to 1 on money
> cycles, 13 to 0 on layering chains. And the model can't show
> its work. We can.
>
> The engine earns it: exact best-trail search with score
> pruning, live on the stream at ~15× headroom over peak
> traffic — and 47 out of 47 English descriptions compiled into
> working detectors. Analysts type sentences; the engine does
> the rest."

| Pattern | MotifQL (3 motifs) | LightGBM |
|---|---|---|
| CYCLE (money comes back) | **24** | 1 |
| RANDOM (layering chains) | **13** | 0 |

### 2:50–3:00 — Close (~10s)

**Screen:** end card — repo URL + "Built with Muse Spark +
Muse Code".
**Action:** hold 3s under the closing line.

> "Crime moves in networks. It's time our detectors did too.
> MotifQL — follow the money."

### Optional live-English beat (~20s, needs `MUSE_KEY`)

**Screen:** workspace sidebar English box with a fresh
sentence (e.g. the RQ4 fan-out item: *"Fan-out: one account
pays at least 2 distinct accounts within 4 days…"*); result
area clears on submit.
**Action:** only if time allows or as a separate clip — paste
the sentence, press **Compile + show**; the Motif panel fills
live (badge: *live*), sample witnesses render into tabs. If
the model is down, the page shows the honest error box — cut
the beat, the scripted path carries the story.

## Boot

```bash
PYTHONPATH=src uv run python -m motifql.server --demo --port 8000
# demo: 5078345 transfers, 8 motifs (...)
# demo: open http://127.0.0.1:8000/ and press Scripted demo
```

Warmup takes ~2s (full HI-Small graph build). Open
<http://127.0.0.1:8000/> — the landing reads ~1.1s, dives into
the workspace, and the witness plays for 4.7s.

Record at 1600×1000 or wider; the layout collapses below 900px.

## Sources for the numbers (don't say on camera, keep for Q&A)

- $800B–$2T / 2–5% of GDP laundered yearly: UNODC (via IMF
  consensus estimate, cited by FATF).
- $16.6B US reported cybercrime losses 2024 (+33% YoY), $13.7B
  of it fraud: FBI IC3 2024 Annual Report.
- $12.5B US consumer fraud losses 2024 (+25%): FTC Consumer
  Sentinel.
- $3.8B global AML fines 2025, $4.6B in 2024, $6.6B in 2023:
  Fenergo annual enforcement review.
- ~$20B transaction-monitoring market in 2025 (AML ~42% of it),
  → $48–62B by early 2030s, ~14–17% CAGR: Fortune Business
  Insights / Mordor Intelligence / MRFR (analyst estimates,
  ranges vary — say "around $20 billion").

## Fallbacks

- No `MUSE_KEY` / API down: all scripted beats unaffected
  (committed canned motif + pinned anchor + evasion curve).
  Optional live beat shows the error box.
- Slow box: warmup is one 5M-row build (~2s here); the trace
  itself is microseconds per anchor (fast path, no appends), and
  each slider tick re-scores one witness only (<500ms).
- `?autorun=0` holds on the landing (manual take: Enter dives
  in); `?landing=0` skips it; `?at=` stills skip it automatically.
  `?autoplay=0` renders frame 0 without playing; reduced-motion
  users get the final frame (and an instant transition). If a
  slider take misfires, cut to the `?at=99&noise=…` stills — same
  pixels, zero timing risk.

## Checklist before recording

- [ ] `pytest tests/test_compile.py tests/test_server.py tests/test_evasion.py` green
- [ ] Boot prints the demo card, landing shows, page dives in and auto-plays the triangle
- [ ] All six "Try one" chips fire (cycle 0.718 / fan-out 1.0 / chain 0.809 / scatter 0.998 / gather 0.75 / combo 0.698+0.75)

## Combo demo (chain → gather, anchor 418378)

Train RANDOM instance 10: a 5-hop layering chain (0.698) whose
tail feeds a 15-payment gather (0.75) — two shapes, one anchor,
two shared transfers.

**Screen:** landing with all six chips; after submit, ONE
unified graph (17 transfers, 9 accounts, 9.6s trace) — score
reads "2/2 composite", strip reads "2/2 shapes fire on
transfer #418378 · 2 shared transfers link the shapes",
shared edges draw thicker, caption calls "in both shapes".
Side panels stack both shapes: dual motif blocks, per-shape
degree bars, unified diagnosis lede ("the chain's tail feeds
the gather") with per-shape verdicts below, evasion shape
toggle (chain/gather).
**Action:** click the **Chain into gather** chip (box fills
with both queries), Enter; let the union trace play, hold the
end frame. Optionally toggle evasion to gather and drag one
slider to show per-shape attack.

Narration: "Same money, two shapes — one graph. The chain's
tail lands in the gather, and the two shared transfers prove
it's one criminal structure, not a coincidence."
(Needs `MUSE_KEY`: the gather query compiles live, like the
optional live-English beat.)
- [ ] Badge reads *scripted* or *cached* (never mid-take surprises)
- [ ] `?at=99` shows the completed trace (title-card still)
- [ ] `?at=99&noise=15&gap=1` shows fuzzy 0.357 / exact 0 (evasion still)
- [ ] Title cards rendered: $2T / $16.6B / $4-6B fines / $20B market
- [ ] Results slide rendered: 24-vs-1 / 13-vs-0
- [ ] Opening provenance line recorded first
