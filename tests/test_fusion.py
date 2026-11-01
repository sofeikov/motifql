"""RQ5 fusion units: min-rank union, score coverage, shared GBM training."""

import sys
from datetime import date
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval import evaluate, union_rank  # noqa: E402
from eval.rq5 import check_coverage  # noqa: E402
from eval.supervised import GBM_PARAMS, train_gbm  # noqa: E402


def _base():
    # One day, 6 transfers. Motif A loves tid 0, motif B loves tid 5;
    # both truths would be missed by the other single at budget 1.
    return pl.DataFrame({
        "transfer_id": [0, 1, 2, 3, 4, 5],
        "day": [date(2022, 9, 8)] * 6,
        "is_laundering": [1, 0, 0, 0, 0, 1],
        "pattern_type": ["CYCLE", None, None, None, None, "RANDOM"],
        "a": [0.9, 0.1, 0.1, 0.1, 0.1, 0.0],
        "b": [0.0, 0.1, 0.1, 0.1, 0.1, 0.9],
    })


def test_min_rank_union_catches_what_either_single_catches():
    hold = union_rank.min_rank_union(_base(), ["a", "b"])
    hits = (evaluate.transfer_recall(hold, [2])
            .filter(pl.col("group") == "ALL_FLAGGED")["n_hit"].item())
    assert hits == 2
    # Each single alone catches only its own truth at budget 1.
    for col, want in (("a", 0), ("b", 5)):
        single = evaluate.rank_per_day(
            _base().with_columns(pl.col(col).alias("score")))
        top = single.filter(pl.col("rank") == 1)["transfer_id"].item()
        assert top == want


def test_min_rank_union_single_column_matches_direct_ranking():
    base = _base()
    fused = union_rank.min_rank_union(base, ["a"]).sort("transfer_id")
    direct = evaluate.rank_per_day(
        base.with_columns(pl.col("a").alias("score"))).sort("transfer_id")
    assert fused["rank"].to_list() == direct["rank"].to_list()


def test_check_coverage():
    check_coverage({1, 2, 3}, {1, 2, 3}, "ok")
    with pytest.raises(SystemExit):
        check_coverage({1, 2}, {1, 2, 3}, "missing")
    with pytest.raises(SystemExit):
        check_coverage({1, 2, 3, 4}, {1, 2, 3}, "extra")


def test_train_gbm_uses_shared_params_and_ranks():
    import numpy as np

    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    clf = train_gbm(X[:1500], y[:1500], X[1500:], y[1500:])
    for k, v in GBM_PARAMS.items():
        assert clf.get_params()[k] == v
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(y[1500:], clf.predict_proba(X[1500:])[:, 1])
    assert auc > 0.9
    assert clf.best_iteration_ >= 1
