"""Tests for supervised-baseline own-transfer features (#15 B)."""

import math
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.supervised import own_transfer_features  # noqa: E402


def test_own_transfer_features() -> None:
    df = pl.DataFrame({
        "transfer_id": [0, 1, 2],
        "from_key": ["a/1", "a/1", "a/2"],
        "to_key": ["a/1", "b/2", "c/3"],
        "amount_usd": [100.0, None, 0.0],
    })
    out = own_transfer_features(df).sort("transfer_id")
    assert out["own_log_amount"].to_list() == [
        math.log1p(100.0), 0.0, 0.0]  # null reads as 0.0
    assert out["own_is_self_loop"].to_list() == [1.0, 0.0, 0.0]
