#!/usr/bin/env python3
"""Prepare AMLworld CSVs into time-sorted Parquet plus ground-truth tables.

Implements DESIGN.md section 6.1 (data preparation), using Polars instead of
DuckDB — see docs/2026-10-27-data-prep.md for the rationale.

Inputs (downloaded, never committed):
    <input>/<PREFIX>_Trans.csv        raw transfers, e.g. HI-Small_Trans.csv
    <input>/<PREFIX>_Patterns.txt     laundering pattern instances

Outputs under <out>/<dataset>/ (gitignored):
    transfers.parquet         time-sorted, USD-normalised, typed transfers
    pattern_instances.parquet one row per laundering instance
    pattern_members.parquet   (instance_id, transfer_id) membership rows
    fx_rates.parquet          median implied rate per currency pair

Usage:
    uv run scripts/prep_amlworld.py --input ~/Downloads/moneyLaundering \
        --dataset hi-small --out data
"""

from __future__ import annotations

import argparse
import io
import re
from collections import deque
from pathlib import Path

import polars as pl

PREFIX = {"hi-small": "HI-Small", "li-small": "LI-Small",
          "hi-medium": "HI-Medium", "li-medium": "LI-Medium"}

# The raw CSV has two columns both called "Account" (sender, receiver).
COLS = ["timestamp", "from_bank", "from_account", "to_bank", "to_account",
        "amount_received", "receiving_currency", "amount_paid",
        "payment_currency", "payment_format", "is_laundering"]

STRING_COLS = ["from_bank", "from_account", "to_bank", "to_account"]

# Pattern header lines come in two shapes, with and without a ": ..." suffix:
#   BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out
#   BEGIN LAUNDERING ATTEMPT - STACK
HEADER_RE = re.compile(r"BEGIN LAUNDERING ATTEMPT - ([A-Z\- ]+?)(?::|$)")
TYPE_TO_MOTIF = {
    "CYCLE": "round_trip",
    "RANDOM": "layering_chain",
    "FAN-OUT": "fan_out",
    "FAN-IN": "fan_in",
    "SCATTER-GATHER": "scatter_gather",
    "GATHER-SCATTER": "gather_scatter",
    "BIPARTITE": "bipartite_butterfly",
    "STACK": "stack",  # out of scope for MotifQL v0
}

TS_FMT = "%Y/%m/%d %H:%M"


def load_transfers(path: Path) -> pl.DataFrame:
    df = (pl.read_csv(path, has_header=True, new_columns=COLS,
                      schema_overrides={c: pl.String for c in STRING_COLS})
            .with_row_index("transfer_id")
            .with_columns(pl.col("timestamp").alias("ts_raw"))
            .with_columns(pl.col("timestamp").str.strptime(pl.Datetime, TS_FMT))
            .with_columns(pl.col("timestamp").dt.date().alias("day")))
    assert all(df.schema[c] == pl.String for c in STRING_COLS), "bank/account id type drift"
    return df


def derive_fx(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Median implied rate (received/paid) per currency pair, plus per-day
    medians for pairs involving Bitcoin, whose rate drifts over time."""
    mix = df.filter(pl.col("payment_currency") != pl.col("receiving_currency"))
    rates = (mix.with_columns((pl.col("amount_received") / pl.col("amount_paid")).alias("rate"))
                .group_by(["payment_currency", "receiving_currency"])
                .agg(pl.len().alias("n"),
                     pl.col("rate").median().alias("median"),
                     (pl.col("rate").quantile(0.95) / pl.col("rate").quantile(0.05)).alias("p95_p5")))
    btc_daily = (mix.filter(pl.col("payment_currency").str.contains("Bitcoin") |
                            pl.col("receiving_currency").str.contains("Bitcoin"))
                    .with_columns((pl.col("amount_received") / pl.col("amount_paid")).alias("rate"))
                    .group_by(["payment_currency", "receiving_currency", "day"])
                    .agg(pl.col("rate").median().alias("daily_median")))
    return rates, btc_daily


def usd_rate_per_currency(rates: pl.DataFrame, min_n: int = 10) -> dict[str, float]:
    """Rate of 1 unit of each currency in USD, from direct pairs where they
    exist, else via the shortest chain of well-supported pairs (BFS)."""
    edges: dict[str, list[tuple[str, float]]] = {}
    for pay, recv, n, med in rates.select(
            ["payment_currency", "receiving_currency", "n", "median"]).iter_rows():
        if n >= min_n and med > 0:
            edges.setdefault(pay, []).append((recv, med))
            edges.setdefault(recv, []).append((pay, 1.0 / med))
    out = {"US Dollar": 1.0}
    queue = deque(["US Dollar"])
    while queue:
        cur = queue.popleft()
        for nxt, r in edges.get(cur, []):
            if nxt not in out:
                out[nxt] = out[cur] / r  # r converts nxt -> cur
                queue.append(nxt)
    return out


def add_usd(df: pl.DataFrame, rates: pl.DataFrame, btc_daily: pl.DataFrame) -> pl.DataFrame:
    to_usd = usd_rate_per_currency(rates)
    missing = set(df["payment_currency"].unique()) - set(to_usd)
    if missing:
        print(f"  WARNING: no USD path for {sorted(missing)}; amount_usd left null")
    df = df.with_columns(
        pl.col("payment_currency").replace_strict(to_usd, default=None).alias("usd_rate"))
    # Override with the daily median for rows touching Bitcoin.
    if len(btc_daily):
        df = df.join(btc_daily, on=["payment_currency", "receiving_currency", "day"], how="left")
        df = df.with_columns(
            pl.when(pl.col("daily_median").is_not_null() &
                    (pl.col("receiving_currency") == "US Dollar"))
            .then(pl.col("daily_median"))
            .when(pl.col("daily_median").is_not_null() &
                  (pl.col("payment_currency") == "US Dollar"))
            .then(1.0)  # paid in USD already; nothing to convert
            .otherwise(pl.col("usd_rate")).alias("usd_rate")).drop("daily_median")
    return df.with_columns((pl.col("amount_paid") * pl.col("usd_rate")).alias("amount_usd"))


def parse_patterns(path: Path, csv_header: str) -> tuple[pl.DataFrame, list[tuple[int, str, int]]]:
    """Returns (member-row frame aligned to COLS with instance_id, instance meta list)."""
    instances: list[tuple[int, str, int]] = []  # (id, type, n_txn_lines)
    txn_lines: list[str] = []
    line_instance: list[int] = []
    cur_type: str | None = None
    cur_count = 0

    def close_instance() -> None:
        nonlocal cur_count
        if cur_type is not None:
            instances.append((len(instances), cur_type, cur_count))
            cur_count = 0

    with open(path) as f:
        for line in f:
            if line.startswith("BEGIN"):
                close_instance()
                m = HEADER_RE.search(line)
                if not m:
                    raise ValueError(f"unparseable pattern header: {line.strip()}")
                cur_type = m.group(1).strip()
                if cur_type not in TYPE_TO_MOTIF:
                    raise ValueError(f"unknown pattern type: {cur_type}")
            elif line.strip() and not line.startswith("END"):
                # Transaction line. "END LAUNDERING ATTEMPT - ..." footers are skipped.
                txn_lines.append(line)
                line_instance.append(len(instances))
                cur_count += 1
    close_instance()

    members = (pl.read_csv(io.StringIO(csv_header + "".join(txn_lines)),
                           has_header=True, new_columns=COLS,
                           schema_overrides={c: pl.String for c in STRING_COLS})
                 .with_columns(pl.Series("instance_id", line_instance, dtype=pl.Int64)))
    return members, instances


JOIN_KEYS = ["ts_raw", *COLS[1:]]  # all original CSV fields as parsed


def _with_occ(df: pl.DataFrame, keys: list[str]) -> pl.DataFrame:
    return df.with_columns(pl.int_range(pl.len()).cum_count().over(keys).alias("_occ"))


def join_members(transfers: pl.DataFrame, members: pl.DataFrame) -> pl.DataFrame:
    """Match each pattern transaction line to one transfer row on all original
    fields. Duplicate rows are disambiguated by occurrence order."""
    left = _with_occ(members.rename({"timestamp": "ts_raw"}), JOIN_KEYS)
    right = _with_occ(transfers.select([*JOIN_KEYS, "transfer_id"]), JOIN_KEYS)
    joined = left.join(right, on=[*JOIN_KEYS, "_occ"], how="left")
    n_miss = joined.filter(pl.col("transfer_id").is_null()).select(pl.len()).item()
    print(f"  pattern lines: {len(members)}, matched: {len(members) - n_miss}, "
          f"unmatched: {n_miss}")
    if n_miss:
        print("  sample unmatched pattern lines:")
        print(joined.filter(pl.col("transfer_id").is_null()).head(5))
    return joined


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare one AMLworld dataset.")
    ap.add_argument("--input", required=True, type=Path, help="download directory")
    ap.add_argument("--dataset", default="hi-small", choices=sorted(PREFIX),
                    help="dataset id (default: hi-small)")
    ap.add_argument("--out", default=Path("data"), type=Path, help="output root")
    args = ap.parse_args()

    prefix = PREFIX[args.dataset]
    trans_path = args.input / f"{prefix}_Trans.csv"
    patt_path = args.input / f"{prefix}_Patterns.txt"
    outdir = args.out / args.dataset
    outdir.mkdir(parents=True, exist_ok=True)
    for p in (trans_path, patt_path):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

    print(f"[1/5] loading {trans_path.name}")
    df = load_transfers(trans_path)
    n_flag = df.filter(pl.col("is_laundering") == 1).select(pl.len()).item()
    print(f"  rows: {len(df)}, flagged laundering: {n_flag} "
          f"(1 in {len(df) / max(n_flag, 1):.0f})")

    print("[2/5] deriving USD rates")
    rates, btc_daily = derive_fx(df)
    bad = rates.filter((pl.col("n") >= 50) & (pl.col("p95_p5") > 1.2))
    print(f"  mixed-currency rows: {rates['n'].sum()}, pairs: {len(rates)}")
    print(f"  inconsistent pairs (n>=50, p95/p5>1.2): {len(bad)}")
    for row in bad.iter_rows(named=True):
        print(f"    {row['payment_currency']} -> {row['receiving_currency']}: "
              f"n={row['n']}, p95/p5={row['p95_p5']:.3f} (daily medians used)")
    df = add_usd(df, rates, btc_daily)
    n_null_usd = df.filter(pl.col("amount_usd").is_null()).select(pl.len()).item()
    print(f"  rows with no USD conversion: {n_null_usd}")

    print("[3/5] sorting and writing transfers.parquet")
    df = df.sort(["timestamp", "transfer_id"]).drop("ts_raw")
    df.write_parquet(outdir / "transfers.parquet")
    rates.write_parquet(outdir / "fx_rates.parquet")

    print("[4/5] parsing pattern instances")
    with open(trans_path) as f:
        csv_header = f.readline()
    members, instances = parse_patterns(patt_path, csv_header)
    print(f"  instances: {len(instances)}")
    df_rawkeys = df.with_columns(
        pl.col("timestamp").dt.strftime(TS_FMT).alias("ts_raw"))
    joined = join_members(df_rawkeys, members)
    n_covered = (joined.filter(pl.col("transfer_id").is_not_null())
                 .select(pl.col("transfer_id").n_unique()).item())
    print(f"  flagged rows covered by >=1 instance: {n_covered} of {n_flag} "
          f"({n_flag - n_covered} flagged but unattributed)")
    if n_covered < n_flag:
        covered_ids = joined.filter(pl.col("transfer_id").is_not_null()
                                    ).select("transfer_id").unique()
        uncovered = (df_rawkeys.filter(pl.col("is_laundering") == 1)
                       .join(covered_ids, on="transfer_id", how="anti"))
        dup_like = uncovered.join(
            members.rename({"timestamp": "ts_raw"}).select(JOIN_KEYS).unique(),
            on=JOIN_KEYS, how="semi").select(pl.len()).item()
        print(f"    of which exact duplicates of a pattern row: {dup_like}, "
              f"absent from {patt_path.name}: {len(uncovered) - dup_like}")

    inst_df = (pl.DataFrame({"instance_id": [i for i, _, _ in instances],
                             "pattern_type": [t for _, t, _ in instances]})
                 .with_columns(pl.col("pattern_type").replace_strict(TYPE_TO_MOTIF)
                               .alias("motif")))
    txns = joined.filter(pl.col("transfer_id").is_not_null()
                         ).select(["instance_id", "transfer_id"]).unique()
    spans = (txns.join(df.select(["transfer_id", "timestamp"]), on="transfer_id")
             .group_by("instance_id")
             .agg(pl.len().alias("n_txns"),
                  pl.col("timestamp").min().alias("t_min"),
                  pl.col("timestamp").max().alias("t_max")))
    inst_df = inst_df.join(spans, on="instance_id", how="left")
    inst_df.write_parquet(outdir / "pattern_instances.parquet")
    txns.write_parquet(outdir / "pattern_members.parquet")
    print("  instances per type:")
    print(inst_df.group_by(["pattern_type", "motif"]).len().sort("pattern_type"))

    print("[5/5] rows and flagged laundering per day")
    with pl.Config(tbl_rows=-1):
        print(df.group_by("day").agg(pl.len().alias("n"), pl.col("is_laundering").sum()
              .alias("laundering")).sort("day"))
    print(f"done -> {outdir}")


if __name__ == "__main__":
    main()
