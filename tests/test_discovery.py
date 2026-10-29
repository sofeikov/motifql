"""Tests for the discovery agent loop (Nov 1, DESIGN.md section 7.2).

All model transport is stubbed: scripted replies drive `run_agent`,
and a stub batch scorer returns fixed tops. Pins: action parsing,
miss listing, try/gate arithmetic, and the train-only guard (the loop
can only query positions the ctx holds — holdout never enters).
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from motifql.models import discovery as D  # noqa: E402

MOTIF = ("motif cand { match (a:account) -[e:transfer*2..3]-> (a)"
         " where e is time_ordered and span(e) <= 7d }")


def _ctx() -> D.DiscoveryCtx:
    return D.DiscoveryCtx(
        graph=None,
        transfer_ids=np.array([100, 101, 102, 103, 200, 201, 202, 203]),
        pos_of={100: 0, 101: 1, 102: 2, 103: 3, 200: 4, 201: 5, 202: 6, 203: 7},
        member_pos={1: [0, 1], 2: [2, 3]},
        member_tid={1: [100, 101], 2: [102, 103]},
        bg_pos=[4, 5, 6, 7],
        base_scores={"m0": {0: 0.1, 1: 0.1, 2: 0.9, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0}},
        base_bg_fire={"m0": 0.0},
        base_emit={"m0": 0.5},
    )


def test_parse_action() -> None:
    assert D.parse_action("SHOW 12") == ("SHOW", None, 12)
    assert D.parse_action("SHOW 223SHOW 223") == ("SHOW", None, 223)
    kind, motif, _ = D.parse_action("TRY\n```motifql\n" + MOTIF + "\n```")
    assert (kind, motif) == ("TRY", MOTIF)
    assert D.parse_action("TRY with no block") == ("TRY", None, None)
    kind, motif, _ = D.parse_action("PROPOSE\n```motifql\n" + MOTIF + "\n```\nRATIONALE: x")
    assert (kind, motif) == ("PROPOSE", MOTIF)
    assert D.parse_action("DONE")[0] == "DONE"
    assert D.parse_action("hmm, interesting") == (None, None, None)
    # Live-model shapes: keyword glued to prose, last keyword wins.
    assert D.parse_action("why count misses them.SHOW 123") == ("SHOW", None, 123)
    kind, motif, _ = D.parse_action("a tighter fan.TRY\n```motifql\n" + MOTIF + "\n```")
    assert (kind, motif) == ("TRY", MOTIF)
    kind, motif, _ = D.parse_action(
        "PROPOSE\n```motifql\n" + MOTIF + "\n```\nRATIONALE: this is done")
    assert (kind, motif) == ("PROPOSE", MOTIF)


def test_parse_action_ignores_lowercase_prose() -> None:
    # Review #14.9: lowercase prose must never trigger an action, and
    # a fenced motif beats trailing uppercase commentary.
    assert D.parse_action("my last try") == (None, None, None)
    assert D.parse_action("once done, more checks") == (None, None, None)
    assert D.parse_action("show 7 please") == (None, None, None)
    kind, motif, _ = D.parse_action(
        "PROPOSE\n```motifql\n" + MOTIF + "\n```\nRATIONALE: ALL DONE")
    assert (kind, motif) == ("PROPOSE", MOTIF)
    # Fence before the keyword is not its block.
    assert D.parse_action("```motifql\n" + MOTIF + "\n```\nTRY again") == ("TRY", None, None)
    # Keyword-looking text inside the fence is ignored.
    kind, motif, _ = D.parse_action(
        "TRY\n```motifql\n" + MOTIF + " // DONE\n```")
    assert (kind, motif) == ("TRY", MOTIF + " // DONE")


def test_list_missed() -> None:
    # Instance 1 peaks 0.1 (missed), instance 2 hits 0.9 (caught).
    assert D.list_missed(_ctx(), ["m0"]) == [1]


def test_try_motif_reports_recall_and_fire() -> None:
    ctx = _ctx()
    tops = {0: 0.9, 1: 0.1, 2: 0.8, 3: 0.0, 4: 0.0, 5: 0.6, 6: 0.0, 7: 0.0}
    out = D.try_motif(ctx, MOTIF, lambda want, plan, libs: {p: tops[p] for p in want}, [1, 2])
    assert "cluster recall 2/2" in out
    assert "background fire 1/4" in out
    bad = D.try_motif(ctx, "motif cand { nonsense", lambda w, p, l: {}, [1])
    assert bad.startswith("COMPILE FAILED")


def test_gate_pass_and_fail_modes() -> None:
    ctx = _ctx()
    hi = {0: 0.9, 1: 0.9, 2: 0.9, 3: 0.9, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0}
    lo = {p: 0.1 for p in hi}
    hot = {**hi, 4: 0.9, 5: 0.9, 6: 0.9}
    run = lambda tops: (lambda want, plan, libs: {p: tops[p] for p in want})  # noqa: E731
    g = D.gate(ctx, MOTIF, run(hi), [1, 2], ["m0"])
    assert g.passed and g.catches == [1, 2] and g.bg_fire == 0.0
    g = D.gate(ctx, MOTIF, run(lo), [1, 2], ["m0"])
    assert not g.passed and "need 2" in g.detail
    g = D.gate(ctx, MOTIF, run(hot), [1, 2], ["m0"])
    assert not g.passed and "bg fire" in g.detail
    g = D.gate(ctx, "motif cand { nonsense", run(hi), [1, 2], ["m0"])
    assert not g.passed and "does not compile" in g.detail
    # Floor pin: base median is 0, but 2/4 background fires (the floor)
    # still passes while 3/4 fails.
    warm = {**hi, 4: 0.9, 5: 0.9}
    g = D.gate(ctx, MOTIF, run(warm), [1, 2], ["m0"])
    assert g.passed


def test_run_agent_passes_proposal_and_stays_in_ctx() -> None:
    ctx = _ctx()
    df = pl.DataFrame({
        "transfer_id": [100, 101, 102, 103],
        "timestamp": [datetime(2022, 9, 3, 10, i) for i in range(4)],
        "from_account": ["A", "B", "C", "D"],
        "to_account": ["B", "A", "D", "C"],
        "amount_usd": [100.0, 100.0, 50.0, 50.0],
    })
    replies = ["SHOW 1",
               "TRY\n```motifql\n" + MOTIF + "\n```",
               "PROPOSE\n```motifql\n" + MOTIF + "\n```\nRATIONALE: catches both"]
    queried: set[int] = set()

    def batch(want, plan, libs):
        queried.update(want)
        return {p: (0.9 if p < 4 else 0.0) for p in want}

    out = D.run_agent("test cluster", [1, 2], ctx, ["m0"], df, batch,
                      lambda messages: (replies.pop(0), {}))
    assert out["proposal"] == MOTIF
    assert out["gate"] is not None and out["gate"].passed
    assert any("instance 1" in t for t in out["transcript"])
    assert any("cluster recall 2/2" in t for t in out["transcript"])
    assert any("GATE PASS" in t for t in out["transcript"])
    # Train-only guard: every scored position comes from the ctx.
    assert queried <= {0, 1, 2, 3, 4, 5, 6, 7}


def test_run_agent_done_stops() -> None:
    ctx = _ctx()
    out = D.run_agent("test", [1], ctx, ["m0"], pl.DataFrame(),
                      lambda w, p, l: {}, lambda messages: ("DONE", {}))
    assert out["proposal"] is None and len(out["transcript"]) == 2


def test_run_agent_fails_closed_on_unknown_instance() -> None:
    # Discriminating guard: a cluster id outside the ctx must become
    # feedback, never an escaped exception or an out-of-ctx query.
    ctx = _ctx()
    replies = ["SHOW 999", "TRY\n```motifql\n" + MOTIF + "\n```", "DONE"]
    queried: set[int] = set()

    def batch(want, plan, libs):
        queried.update(want)
        return {p: 0.0 for p in want}

    out = D.run_agent("test", [999], ctx, ["m0"], pl.DataFrame(), batch,
                      lambda messages: (replies.pop(0), {}), max_steps=3)
    assert any("unknown instance 999" in t for t in out["transcript"])
    assert any("tool error" in t for t in out["transcript"])
    assert out["proposal"] is None
    assert queried <= {0, 1, 2, 3, 4, 5, 6, 7}


def test_run_agent_nudges_final_turn() -> None:
    ctx = _ctx()
    out = D.run_agent("test", [1], ctx, ["m0"], pl.DataFrame(),
                      lambda w, p, l: {p: 0.0 for p in w},
                      lambda messages: ("SHOW 1", {}), max_steps=2)
    assert any("LAST STEP" in t for t in out["transcript"])
