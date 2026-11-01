"""Learned fuzzy calibration: spaces, doc rewriting and the fit harness.

Issue #24 fits membership-function numbers on train-only data. The
contract under test: sampling stays in bounds and is seeded,
apply_params moves only the fittable numbers (structure frozen, other
motifs bit-identical), wider supports score at least as high on a fixed
witness, the rendered fitted file parses and matches, and the harness
runs end to end on a synthetic dataset.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from eval.calibrate import (
    build_eval_frame,
    pick_winner,
    run_fit,
    sample_eval_positions,
    sampled_objective,
)
from motifql.calibrate import (
    DEFAULTS,
    TARGET_PATTERN,
    apply_params,
    default_params,
    render_fitted_file,
    sample_params,
    validate_params,
    width_score,
)
from motifql.degrees import libs_from_doc
from motifql.graph import TemporalGraph
from motifql.matcher import EXACT_FLOOR, score
from motifql.parser import parse_file, parse_text
from motifql.planner import compile_motif

BASE = datetime(2022, 9, 3)
REPO_MOTIFS = Path(__file__).resolve().parent.parent / "examples/amlworld.motif"


def _cycle_frame() -> pl.DataFrame:
    # One 3-hop cycle with 10h gaps and a 15% amount wobble step: narrow
    # tolerances read the wobble as ~0, wide ones grade it.
    rows = [
        (0, "001", "A", "002", "B", 100.0),
        (600, "002", "B", "003", "C", 115.0),
        (1200, "003", "C", "001", "A", 100.0),
    ]
    return pl.DataFrame({
        "transfer_id": list(range(len(rows))),
        "timestamp": [BASE + timedelta(minutes=r[0]) for r in rows],
        "from_bank": [r[1] for r in rows],
        "from_account": [r[2] for r in rows],
        "to_bank": [r[3] for r in rows],
        "to_account": [r[4] for r in rows],
        "amount_usd": [r[5] for r in rows],
    })


def _top(motif_doc, motif: str, params: dict) -> float:
    fitted = apply_params(motif_doc, motif, params)
    plan = compile_motif(fitted, motif)
    libs = libs_from_doc(fitted)
    graph = TemporalGraph.from_transfers(_cycle_frame())
    return score(graph, 2, plan, libs, floor=EXACT_FLOOR).top_score


# -- spaces ---------------------------------------------------------------

def test_each_space_has_target_and_defaults() -> None:
    for motif in ("round_trip", "layering_chain", "scatter_gather"):
        assert motif in TARGET_PATTERN
        validate_params(motif, default_params(motif))


def test_sampled_params_stay_in_bounds() -> None:
    rng = np.random.default_rng(24)
    for _ in range(50):
        for motif in DEFAULTS:
            params = sample_params(motif, rng)
            validate_params(motif, params)
            if motif in ("round_trip", "layering_chain"):
                assert params["gap_plateau_h"] <= params["gap_support_h"]


def test_sampling_is_seeded() -> None:
    a = sample_params("round_trip", np.random.default_rng(24))
    b = sample_params("round_trip", np.random.default_rng(24))
    assert a == b
    c = sample_params("round_trip", np.random.default_rng(25))
    assert c != a


@pytest.mark.parametrize("motif,params", [
    ("round_trip", {"gap_plateau_h": 100.0, "gap_support_h": 120.0,
                   "tol_pct": 20.0, "score": "min", "emit": 0.5}),  # plateau wide
    ("round_trip", {"gap_plateau_h": 24.0, "gap_support_h": 200.0,
                   "tol_pct": 20.0, "score": "min", "emit": 0.5}),  # past window
    ("round_trip", {"gap_plateau_h": 60.0, "gap_support_h": 50.0,
                   "tol_pct": 20.0, "score": "min", "emit": 0.5}),  # inverted
    ("round_trip", {"gap_plateau_h": 24.0, "gap_support_h": 120.0,
                   "tol_pct": 80.0, "score": "min", "emit": 0.5}),  # tol wide
    ("round_trip", {"gap_plateau_h": 24.0, "gap_support_h": 120.0,
                   "tol_pct": 20.0, "score": "avg", "emit": 0.5}),  # bad tnorm
    ("round_trip", {"gap_plateau_h": 24.0, "gap_support_h": 120.0,
                   "tol_pct": 20.0, "score": "min", "emit": 0.9}),  # emit wide
    ("scatter_gather", {"several_lo": 3.0, "several_hi": 3.0,
                        "sum_tol_pct": 20.0, "score": "product",
                        "emit": 0.5}),  # flat ramp
])
def test_validation_rejects_out_of_space(motif: str, params: dict) -> None:
    with pytest.raises(ValueError):
        validate_params(motif, params)


def test_validation_rejects_unknown_motif_and_keys() -> None:
    with pytest.raises(ValueError, match="no calibration space"):
        validate_params("fan_out", {})
    with pytest.raises(ValueError, match="wants params"):
        validate_params("round_trip", {"tol_pct": 20.0})


# -- doc rewriting ----------------------------------------------------------

def test_apply_params_freezes_structure() -> None:
    doc = parse_file(REPO_MOTIFS)
    params = {"gap_plateau_h": 30.0, "gap_support_h": 140.0,
              "tol_pct": 27.0, "score": "product", "emit": 0.6}
    fitted = apply_params(doc, "round_trip", params)
    before, after = compile_motif(doc, "round_trip"), compile_motif(fitted, "round_trip")
    assert (after.window_s, after.min_hops, after.max_hops, after.closed) == \
        (before.window_s, before.min_hops, before.max_hops, before.closed)
    assert len(after.prunable) == len(before.prunable)
    assert (after.deferred is None) == (before.deferred is None)
    assert after.score_tnorm == "product" and after.emit == 0.6
    # The private term exists and the original is untouched.
    libs = libs_from_doc(fitted)
    assert libs[0]["patient_fit_rt"].shape == "trapezoid"
    assert libs_from_doc(doc)[0]["patient"].args[2].seconds == 24 * 3600.0


def test_apply_params_leaves_other_motifs_identical() -> None:
    doc = parse_file(REPO_MOTIFS)
    fitted = apply_params(doc, "round_trip",
                          {"gap_plateau_h": 30.0, "gap_support_h": 140.0,
                           "tol_pct": 27.0, "score": "product", "emit": 0.6})
    for other in ("layering_chain", "scatter_gather", "fan_out"):
        assert compile_motif(fitted, other) == compile_motif(doc, other)


def test_apply_params_rejects_drifted_structure() -> None:
    doc = parse_file(REPO_MOTIFS)
    with pytest.raises(ValueError, match="no calibration space"):
        apply_params(doc, "nope", default_params("round_trip"))
    bare = parse_text("motif t { match (a:account) -[e:transfer*2..3]-> (a)"
                      " where e is time_ordered and span(e) <= 7d"
                      " score min emit when score >= 0.5 }")
    with pytest.raises(ValueError, match="no motif `round_trip` in document"):
        apply_params(bare, "round_trip", default_params("round_trip"))
    softless = parse_text("motif round_trip { match (a:account) -[e:transfer*2..3]-> (a)"
                          " where e is time_ordered and span(e) <= 7d"
                          " score min emit when score >= 0.5 }")
    with pytest.raises(ValueError, match="no soft clause"):
        apply_params(softless, "round_trip", default_params("round_trip"))


def test_wider_supports_score_at_least_as_high() -> None:
    doc = parse_file(REPO_MOTIFS)
    narrow = {"gap_plateau_h": 6.0, "gap_support_h": 48.0,
              "tol_pct": 5.0, "score": "min", "emit": 0.5}
    wide_tol = dict(narrow, tol_pct=40.0)
    wide_gap = {"gap_plateau_h": 72.0, "gap_support_h": 168.0,
                "tol_pct": 5.0, "score": "min", "emit": 0.5}
    s_narrow = _top(doc, "round_trip", narrow)
    assert s_narrow == pytest.approx(0.0)  # 15% wobble reads 0 at width 5
    assert _top(doc, "round_trip", wide_tol) > s_narrow
    # Wider gap shoulders never hurt on a fixed witness either.
    assert _top(doc, "round_trip", wide_gap) >= s_narrow


def test_rendered_default_matches_reference_scores() -> None:
    doc = parse_file(REPO_MOTIFS)
    text = render_fitted_file("round_trip", default_params("round_trip"),
                              meta="test")
    rendered = parse_text(text)
    plan_r = compile_motif(rendered, "round_trip")
    plan_o = compile_motif(doc, "round_trip")
    assert (plan_r.window_s, plan_r.min_hops, plan_r.max_hops,
            plan_r.score_tnorm, plan_r.emit) == \
        (plan_o.window_s, plan_o.min_hops, plan_o.max_hops,
         plan_o.score_tnorm, plan_o.emit)
    graph = TemporalGraph.from_transfers(_cycle_frame())
    s_r = score(graph, 2, plan_r, libs_from_doc(rendered),
                floor=EXACT_FLOOR).top_score
    s_o = score(graph, 2, plan_o, libs_from_doc(doc),
                floor=EXACT_FLOOR).top_score
    assert s_r == pytest.approx(s_o)


def test_rendered_fitted_file_parses_for_every_space() -> None:
    rng = np.random.default_rng(24)
    for motif in DEFAULTS:
        params = sample_params(motif, rng)
        plan = compile_motif(
            parse_text(render_fitted_file(motif, params)), motif)
        assert plan.emit == pytest.approx(float(params["emit"]))
        assert plan.score_tnorm == params["score"]


# -- objective + selection ----------------------------------------------------

def test_sampled_objective_scales_budgets_per_day() -> None:
    frame = pl.DataFrame({
        "transfer_id": [0, 1, 2, 3, 4, 5],
        "day": [date(2022, 9, 3)] * 3 + [date(2022, 9, 4)] * 3,
        "score": [0.9, 0.5, 0.4, 0.8, 0.7, 0.1],
        "is_laundering": [1, 0, 0, 0, 0, 1],
        "pattern_type": ["CYCLE", None, None, None, None, "CYCLE"],
    })
    # 3 eval anchors of 6 full per day: budget 4 -> k=2/day. Day 1 hits
    # (target ranks 1st), day 2 misses (target ranks 3rd).
    out = sampled_objective(frame, "CYCLE", [4], full_per_day={
        date(2022, 9, 3): 6, date(2022, 9, 4): 6})
    assert out["hits"] == {4: 1}
    assert out["objective"] == 1.0


def test_pick_winner_prefers_wider_supports_on_ties() -> None:
    rows = [
        {"try": 0, "objective": 5.0, "width": 0.5},
        {"try": 1, "objective": 5.0, "width": 0.9},
        {"try": 2, "objective": 4.0, "width": 1.5},
    ]
    assert pick_winner(rows)["try"] == 1
    assert width_score("round_trip", default_params("round_trip")) == \
        pytest.approx(120.0 / 168.0 + 20.0 / 50.0)


def test_eval_positions_keep_all_truth() -> None:
    train_pos = np.arange(100)
    tids = np.arange(100)
    got = sample_eval_positions(train_pos, tids, {5, 50}, bg_n=10, seed=24)
    assert {5, 50} <= set(tids[got])
    assert len(got) == 12
    assert (np.diff(got) > 0).all()  # time order kept
    again = sample_eval_positions(train_pos, tids, {5, 50}, bg_n=10, seed=24)
    assert np.array_equal(got, again)


def test_build_eval_frame_aligns_scores() -> None:
    df = pl.DataFrame({
        "transfer_id": [10, 11, 12], "day": [date(2022, 9, 3)] * 3,
        "is_laundering": [1, 0, 0]})
    memb = pl.DataFrame({"instance_id": [1], "transfer_id": [10]})
    inst = pl.DataFrame({"instance_id": [1], "pattern_type": ["CYCLE"]})
    frame = build_eval_frame(df, memb, inst, np.array([12, 10]))
    assert frame["transfer_id"].to_list() == [12, 10]
    assert frame["pattern_type"].to_list() == [None, "CYCLE"]


# -- end to end on synthetic data ----------------------------------------------

def _synthetic_dataset(root: Path) -> None:
    """Two planted 3-cycles (Sep 3, Sep 4) in random background."""
    rng = np.random.default_rng(7)
    rows: list[tuple] = []
    tid = 0

    def add(day: date, hour: int, minute: int, fa: str, ta: str, amt: float,
            laund: int) -> int:
        nonlocal tid
        rows.append((tid, datetime(day.year, day.month, day.day, hour, minute),
                     "001", fa, "001", ta, amt, day, laund))
        tid += 1
        return tid - 1

    cycle_tids = []
    for day, tag in ((date(2022, 9, 3), "A"), (date(2022, 9, 4), "B")):
        a, b, c = f"{tag}1", f"{tag}2", f"{tag}3"
        cycle_tids.append([add(day, 1, 0, a, b, 100.0, 1),
                           add(day, 2, 0, b, c, 102.0, 1),
                           add(day, 3, 0, c, a, 99.0, 1)])
    accts = [f"X{i}" for i in range(30)]
    for day in (date(2022, 9, 3), date(2022, 9, 4), date(2022, 9, 5)):
        for _ in range(60):
            f, t = rng.choice(accts, size=2, replace=False)
            add(day, int(rng.integers(0, 24)), int(rng.integers(0, 60)),
                str(f), str(t), float(rng.uniform(10.0, 50000.0)), 0)
    df = pl.DataFrame({
        "transfer_id": [r[0] for r in rows], "timestamp": [r[1] for r in rows],
        "from_bank": [r[2] for r in rows], "from_account": [r[3] for r in rows],
        "to_bank": [r[4] for r in rows], "to_account": [r[5] for r in rows],
        "amount_usd": [r[6] for r in rows], "day": [r[7] for r in rows],
        "is_laundering": [r[8] for r in rows],
    }).sort("timestamp")
    memb = pl.DataFrame({
        "instance_id": [1] * 3 + [2] * 3,
        "transfer_id": cycle_tids[0] + cycle_tids[1]})
    inst = pl.DataFrame({
        "instance_id": [1, 2], "pattern_type": ["CYCLE", "CYCLE"],
        "t_min": [datetime(2022, 9, 3, 1), datetime(2022, 9, 4, 1)],
        "t_max": [datetime(2022, 9, 3, 3), datetime(2022, 9, 4, 3)]})
    root.mkdir(parents=True, exist_ok=True)
    df.write_parquet(root / "transfers.parquet")
    memb.write_parquet(root / "pattern_members.parquet")
    inst.write_parquet(root / "pattern_instances.parquet")


def test_fit_runs_end_to_end_and_resumes(tmp_path: Path) -> None:
    data = tmp_path / "data"
    _synthetic_dataset(data)
    out = tmp_path / "cal"
    summary = run_fit(data, "round_trip", tries=2, seed=24, budgets=[50, 100],
                      bg_n=60, workers=1, out=out)
    trace = pl.read_csv(out / "trace.csv")
    assert trace.height == 3  # baseline + 2 tries
    assert set(trace.columns) >= {"try", "is_baseline", "objective",
                                  "hits_b50", "hits_b100", "fires_eval"}
    fit = json.loads((out / "fitted.json").read_text())
    validate_params("round_trip", fit["params"])
    assert fit["try"] == summary["winner_try"]
    assert (out / "confirm_train_transfer.csv").exists()
    assert (out / "confirm_train_summary.csv").exists()
    fitted_doc = parse_text((out / "fitted_round_trip.motif").read_text())
    assert compile_motif(fitted_doc, "round_trip").emit == \
        pytest.approx(float(fit["params"]["emit"]))
    # Resume: same winner without recomputing the search.
    again = run_fit(data, "round_trip", tries=2, seed=24, budgets=[50, 100],
                    bg_n=60, workers=1, out=out)
    assert again["winner_try"] == summary["winner_try"]
