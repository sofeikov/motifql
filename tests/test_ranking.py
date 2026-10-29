"""Per-day ranking: deterministic, input-order free, label-blind ties.

E1: transfer_id is raw file order, which correlates with the label, so
score ties must break by a keyed hash — never by transfer_id directly.
"""

from datetime import date

import polars as pl
import pytest

from eval.evaluate import rank_per_day, transfer_recall


def _frame():
    # Adversarial file order: every flagged row precedes every clean row.
    return pl.DataFrame({
        "transfer_id": list(range(100)),
        "day": [date(2022, 9, 8)] * 100,
        "score": [0.0] * 100,
        "is_laundering": [1] * 10 + [0] * 90,
        "pattern_type": [None] * 100,
    })


def test_tied_scores_do_not_follow_file_order():
    # All scores tie; a file-order tie-break would alert the 10 flagged
    # rows first (10/10 hits). The keyed hash mixes them (md5-pinned).
    ranked = rank_per_day(_frame())
    hits = (transfer_recall(ranked, [10])
            .filter(pl.col("group") == "ALL_FLAGGED")["n_hit"].item())
    assert hits == 2


def test_ranking_ignores_input_row_order():
    a = rank_per_day(_frame()).sort("transfer_id")["rank"].to_list()
    b = (rank_per_day(_frame().sample(fraction=1.0, shuffle=True, seed=1))
         .sort("transfer_id")["rank"].to_list())
    assert a == b


def test_score_order_still_dominates():
    df = _frame().with_columns(
        pl.when(pl.col("is_laundering") == 1).then(1.0).otherwise(0.0).alias("score"))
    ranked = rank_per_day(df)
    hits = (transfer_recall(ranked, [10])
            .filter(pl.col("group") == "ALL_FLAGGED")["n_hit"].item())
    assert hits == 10


def test_secondary_direction_orders_ties() -> None:
    # #15 A: the exact arm's tie-break sensitivity needs both directions.
    df = _frame().with_columns(
        pl.Series("score", [1.0] * 100),
        pl.Series("amount_usd", [float(i) for i in range(100)]))
    desc = rank_per_day(df, secondary_col="amount_usd").sort("transfer_id")
    asc = rank_per_day(df, secondary_col="amount_usd",
                       secondary_desc=False).sort("transfer_id")
    assert desc.filter(pl.col("transfer_id") == 99)["rank"].item() == 1
    assert asc.filter(pl.col("transfer_id") == 0)["rank"].item() == 1


def test_resolve_crisp_alpha() -> None:
    from eval.evaluate import resolve_crisp_alpha
    assert resolve_crisp_alpha("emit", 0.5) == 0.5
    assert resolve_crisp_alpha("support", 0.5) == 0.0
    assert resolve_crisp_alpha("0.25", 0.5) == 0.25
    with pytest.raises(ValueError):
        resolve_crisp_alpha("1.5", 0.5)
    with pytest.raises(ValueError):
        resolve_crisp_alpha("bogus", 0.5)
