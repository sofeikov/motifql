# Rehearsal script (Nov 3 demo UI → Nov 4 video)

One-command boot, a deterministic scripted path for the recording, and
an optional live-English beat. Timings measured Oct 30 on the dev box.

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

## Beat 1 — the problem (page idle, ~10s)

The English box is pre-filled: *"Money leaves an account and comes
back to it within 7 days…"* Narration: transaction monitoring sees
one transfer at a time; laundering is a shape over time. Press
**Scripted demo** (or let the auto-run play).

## Beat 2 — Spark writes the motif (~15s)

The Motif panel fills with genuine Muse Spark output (badge:
*scripted · Muse Spark output* — a committed, verified model reply,
so the recording never depends on the network). Point at the three
parts: the shape (`match`), the fuzzy conditions (`soft`), the
threshold (`emit`). Badge meanings: *live* = just compiled,
*cached* = prompt-hash cache hit, *scripted* = committed fallback.

## Beat 3 — the trace (~20s)

The witness plays hop by hop on train CYCLE instance 102
(score 0.718, FIRES): each hop draws in amber with its amount and
gap, the caption narrates from → to, and the *Why this score* bars
show both conditions holding (gap 0.841, amount 0.718). End frame:
the closed triangle, "Trace complete — 3 hops, span 2.9d".
Scrub or step with the transport if the take needs a still;
`?at=SECONDS` deep-links any moment (total 5.6s).

## Beat 4 (optional) — live English (~20s, needs `MUSE_KEY`)

Type a new description (e.g. the RQ4 fan-out item: *"Fan-out: one
account pays at least 2 distinct accounts within 4 days…"*) and
press **Compile + show**. The repair loop runs (≤3 attempts) and
the best sample witnesses render. If the model is down, the page
says so honestly — cut this beat, the scripted path carries the
story.

## Fallbacks

- No `MUSE_KEY` / API down: Beats 1–3 unaffected (committed
  canned motif + pinned anchor). Beat 4 shows the error box.
- Slow box: warmup is one 5M-row build (~2s here); the trace
  itself is microseconds per anchor (fast path, no appends).
- `?autorun=0` opens idle (no auto-run); `?autoplay=0` renders
  frame 0 without playing; reduced-motion users get the final
  frame.

## Checklist before recording

- [ ] `pytest tests/test_compile.py tests/test_server.py` green
- [ ] Boot prints the demo card, page auto-plays the triangle
- [ ] Badge reads *scripted* or *cached* (never mid-take surprises)
- [ ] `?at=99` shows the completed trace (title-card still)
