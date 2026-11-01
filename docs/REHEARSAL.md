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
> that: analysts describe the shape in English, AI writes the
> detector, and every alert shows its money trail."

## Full script (~180s)

### 0:00–0:20 — Hook: the cost of the problem (title cards, no demo)

Say this over simple fullscreen numbers. No jargon yet.

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

On screen: 4 cards, one number each: $2T laundered / $16.6B US
fraud / $4-6B fines per year / $20B monitoring market.

### 0:20–0:50 — Why it's hard: crime is social, software is lonely

This is the key insight. Slow down here.

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

On screen: animate a simple A→B→C→A triangle while talking. Then
show one transfer row highlighted ("looks fine") vs the full
shape ("looks criminal").

### 0:50–1:05 — The solution in one breath (cut to demo UI, idle)

> "MotifQL: describe the criminal shape in plain English, Muse
> Spark writes the detector, and the engine scores every transfer
> — returning not just a score, but the *money trail* behind it.
>
> Three things make it different: it's fuzzy, so criminals can't
> dodge it by waiting an extra hour. It runs live on the stream.
> And every alert explains itself."

Demo UI idle. The English box is pre-filled: *"Money leaves an
account and comes back to it within 7 days…"*. Press **Scripted
demo** (or let the auto-run play).

### 1:05–1:25 — Beat: Spark writes the motif (~20s)

The Motif panel fills with genuine Muse Spark output (badge:
*scripted · Muse Spark output* — a committed, verified model
reply, so the recording never depends on the network).

Narration:

> "I typed one English sentence. Spark wrote the detector: the
> shape to match, the fuzzy conditions, the threshold. The
> analyst never touches query code."

Point at the three parts: the shape (`match`), the fuzzy
conditions (`soft`), the threshold (`emit`). Badge meanings:
*live* = just compiled, *cached* = prompt-hash cache hit,
*scripted* = committed fallback.

### 1:25–1:50 — Beat: the trace (~25s)

The witness plays hop by hop on train CYCLE instance 102 (score
0.718, FIRES): each hop draws in amber with its amount and gap,
the caption narrates from → to, and the *Why this score* bars
show both conditions holding (gap 0.841, amount 0.718).

Narration:

> "And here's the payoff: not a risk score from a black box, but
> the actual money trail — hop by hop, amounts, delays, and how
> strongly each step matches. Plus a one-paragraph story for the
> investigator, with the weakest link flagged."

Under the stage, the investigator narrative retells the witness
in one paragraph with the weakest step badged (hop 3 — committed
for the scripted take, model-written for live compiles). End
frame: the closed triangle, "Trace complete — 3 hops, span 2.9d".
Scrub or step with the transport if the take needs a still;
`?at=SECONDS` deep-links any moment, narrative included (total
5.6s).

### 1:50–2:25 — Beat: evasion (~35s)

The "it actually works" minute, visceral: with the triangle on
screen, drag the **Amount noise** slider to +15%.

Narration:

> "Now watch what happens when the criminal adapts — amounts
> wobbled by 15%. The old-style exact rule falls off a cliff:
> one to zero. Gone. Our fuzzy score degrades gracefully,
> 0.72 to 0.36 — still flagging it. Same when they slow down:
> double the delays, and the exact rule dies while we still
> catch it. Criminals evade thresholds for a living. Don't give
> them one."

What happens on screen: the fuzzy bar sinks 0.718→0.357 while
the exact bar (α=emit) has already cliffed 1→0 at +10% — the
caption reads "+15% noise: fuzzy 0.718→0.357, exact 1→0" and the
*Why this score* amount bar sinks with it. Reset noise to 0, drag
**Gap scale** to ×2.0: fuzzy 0.718→0.431, exact 1→0 again. The
same draws as the report's perturbation curves (link under the
sliders), applied to this witness only. Sliders re-score live in
<500ms (debounced); the scripted anchor falls back to a committed
curve, so this beat needs no key either.

Stills (trace end + slider state deep-link together):
`?at=99&noise=15&gap=1` (mid-attack),
`?at=99&noise=0&gap=2` (gap cliff),
`?at=99` (clean witness).

### 2:25–2:50 — Beat: proof (results card, ~25s)

Cut away from the UI to one results slide. Keep it to two lines:

> "On the standard IBM benchmark, at 50 alerts a day — a real
> analyst's workload — MotifQL catches the structural patterns a
> strong machine-learning baseline barely sees: 24 to 1 on money
> cycles, 13 to 0 on layering chains. And the model can't show
> its work. We can.
>
> Plus: 47 out of 47 English descriptions compiled into working
> detectors. Analysts type sentences; the system does the rest."

| Pattern | MotifQL (3 motifs) | LightGBM |
|---|---|---|
| CYCLE (money comes back) | **24** | 1 |
| RANDOM (layering chains) | **13** | 0 |

(Full table + honest limits in README; don't read caveats on
camera, just link them.)

### 2:50–3:00 — Close (~10s)

> "Crime moves in networks. It's time our detectors did too.
> MotifQL — follow the money."

End card: repo URL, "Built with Muse Spark + Muse Code".

### Optional live-English beat (~20s, needs `MUSE_KEY`)

Only if time allows or as a separate clip. Type a new description
(e.g. the RQ4 fan-out item: *"Fan-out: one account pays at least
2 distinct accounts within 4 days…"*) and press **Compile +
show**. The repair loop runs (≤3 attempts) and the best sample
witnesses render. If the model is down, the page says so honestly
— cut this beat, the scripted path carries the story.

## Boot

```bash
PYTHONPATH=src uv run python -m motifql.server --demo --port 8000
# demo: 5078345 transfers, 8 motifs (...)
# demo: open http://127.0.0.1:8000/ and press Scripted demo
```

Warmup takes ~2s (full HI-Small graph build). Open
<http://127.0.0.1:8000/> — the scripted demo auto-runs on load
(~0.3s to first paint; the witness then plays for 5.6s).

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
- `?autorun=0` opens idle (no auto-run); `?autoplay=0` renders
  frame 0 without playing; reduced-motion users get the final
  frame. If a slider take misfires, cut to the `?at=99&noise=…`
  stills — same pixels, zero timing risk.

## Checklist before recording

- [ ] `pytest tests/test_compile.py tests/test_server.py tests/test_evasion.py` green
- [ ] Boot prints the demo card, page auto-plays the triangle
- [ ] Badge reads *scripted* or *cached* (never mid-take surprises)
- [ ] `?at=99` shows the completed trace (title-card still)
- [ ] `?at=99&noise=15&gap=1` shows fuzzy 0.357 / exact 0 (evasion still)
- [ ] Title cards rendered: $2T / $16.6B / $4-6B fines / $20B market
- [ ] Results slide rendered: 24-vs-1 / 13-vs-0
- [ ] Opening provenance line recorded first
