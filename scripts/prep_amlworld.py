#!/usr/bin/env python3
"""Prepare AMLworld CSVs into time-sorted Parquet plus ground-truth tables.

Implements DESIGN.md section 6.1 (data preparation), using Polars instead of
DuckDB — see docs/2026-10-27-data-prep.md for the rationale.

Pipeline
--------
1. Load `<PREFIX>_Trans.csv`: rename the duplicated `Account` columns, keep
   bank/account ids as strings (leading zeros matter), parse timestamps.
2. Derive USD exchange rates from mixed-currency rows (median implied rate
   per pair) and convert.
3. Sort by time and write `transfers.parquet`.
4. Parse `<PREFIX>_Patterns.txt` into instance and membership tables, joining
   each pattern line to its transfer row on all original fields.

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

import io
import logging
import re
from collections import deque
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

log = logging.getLogger("prep")

PREFIX = {"hi-small": "HI-Small", "li-small": "LI-Small",
          "hi-medium": "HI-Medium", "li-medium": "LI-Medium"}

# The raw CSV has two columns both called "Account" (sender, receiver).
COLS = ["timestamp", "from_bank", "from_account", "to_bank", "to_account",
        "amount_received", "receiving_currency", "amount_paid",
        "payment_currency", "payment_format", "is_laundering"]

STRING_COLS = ["from_bank", "from_account", "to_bank", "to_account"]

# A currency pair needs this many mixed-currency rows before its consistency
# check (p95/p5 spread) is trusted; sparser pairs pass through unchecked.
FX_MIN_PAIR_ROWS = 50
# Pairs whose 95th/5th percentile implied-rate ratio exceeds this are logged
# for the analyst (informational only: conversion always uses medians, which
# are robust to the outliers that drive the spread).
FX_MAX_P95_P5 = 1.2
# An FX edge needs this many rows to join the USD conversion graph below.
FX_MIN_CHAIN_ROWS = 10

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
    """Raw CSV into a typed frame with stable row ids and a day column."""
    df = (pl.read_csv(path, has_header=True, new_columns=COLS,
                      schema_overrides={c: pl.String for c in STRING_COLS})
            .with_row_index("transfer_id")
            .with_columns(
                pl.col("timestamp").alias("ts_raw"),
                pl.col("timestamp").str.strptime(pl.Datetime, TS_FMT),
                # Day straight from the fixed-format string: one projection
                # over the raw columns instead of a second pass over the
                # parsed timestamp.
                pl.col("timestamp").str.slice(0, 10).str.to_date("%Y/%m/%d").alias("day"),
            ))
    assert all(df.schema[c] == pl.String for c in STRING_COLS), "bank/account id type drift"
    return df


def derive_fx(df: pl.DataFrame) -> pl.DataFrame:
    """Median implied rate (received/paid) per currency pair.

    Medians are robust to the outlier rows that widen some pairs' spreads
    (notably Bitcoin's); every pair, Bitcoin included, converts with one
    global median — the daily medians are flat, so per-day rates would add
    nothing. See the day report, section 3.
    """
    mix = df.filter(pl.col("payment_currency") != pl.col("receiving_currency"))
    return (mix.with_columns((pl.col("amount_received") / pl.col("amount_paid")).alias("rate"))
               .group_by(["payment_currency", "receiving_currency"])
               .agg(pl.len().alias("n"),
                    pl.col("rate").median().alias("median"),
                    (pl.col("rate").quantile(0.95) / pl.col("rate").quantile(0.05)).alias("p95_p5")))


def usd_rate_per_currency(rates: pl.DataFrame,
                          min_n: int = FX_MIN_CHAIN_ROWS) -> dict[str, float]:
    """USD value of 1 unit of each currency.

    From direct pairs where they exist, else via the shortest chain of
    well-supported pairs (BFS over pair medians). Only pairs with at least
    `min_n` rows become graph edges, so one-off rows can't skew a chain.
    """
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


def add_usd(df: pl.DataFrame, rates: pl.DataFrame) -> pl.DataFrame:
    """Convert `amount_paid` to USD, one rate per payment currency."""
    to_usd = usd_rate_per_currency(rates)
    missing = set(df["payment_currency"].unique()) - set(to_usd)
    if missing:
        log.warning("no USD path for %s; amount_usd left null", sorted(missing))
    return df.with_columns(
        pl.col("payment_currency").replace_strict(to_usd, default=None).alias("usd_rate"),
        (pl.col("amount_paid")
         * pl.col("payment_currency").replace_strict(to_usd, default=None)).alias("amount_usd"))


def parse_patterns(path: Path, csv_header: str) -> tuple[pl.DataFrame, list[tuple[int, str, int]]]:
    """Pattern instances from a `*_Patterns.txt` file.

    Instance ids are assigned sequentially (the file headers carry only the
    pattern type). Transaction lines are parsed with the same CSV options as
    the main file, so values compare exactly at join time. `END LAUNDERING
    ATTEMPT - ...` footers carry no data and are skipped.

    Returns the member-row frame (aligned to COLS, with `instance_id`) and
    the (id, type, transaction-line count) instance list.
    """
    instances: list[tuple[int, str, int]] = []
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
    """Number each row's occurrence within its duplicate-key group (1-based).

    Lets the join below match duplicate rows one-to-one in order instead of
    exploding them combinatorially.
    """
    return df.with_columns(pl.int_range(pl.len()).cum_count().over(keys).alias("_occ"))


def join_members(transfers: pl.DataFrame, members: pl.DataFrame) -> pl.DataFrame:
    """Match each pattern transaction line to one transfer row.

    Join key is all original CSV fields plus the within-duplicate occurrence
    number, so identical rows pair up in order rather than cross-joining.
    """
    left = _with_occ(members.rename({"timestamp": "ts_raw"}), JOIN_KEYS)
    right = _with_occ(transfers.select([*JOIN_KEYS, "transfer_id"]), JOIN_KEYS)
    joined = left.join(right, on=[*JOIN_KEYS, "_occ"], how="left")
    n_miss = joined.filter(pl.col("transfer_id").is_null()).select(pl.len()).item()
    log.info("  pattern lines: %d, matched: %d, unmatched: %d",
             len(members), len(members) - n_miss, n_miss)
    if n_miss:
        log.warning("sample unmatched pattern lines:\n%s",
                    joined.filter(pl.col("transfer_id").is_null()).head(5))
    return joined


def main(
    input: Annotated[Path, typer.Option(help="Download directory.")],
    dataset: Annotated[str, typer.Option(help="Dataset id.")] = "hi-small",
    out: Annotated[Path, typer.Option(help="Output root (gitignored).")] = Path("data"),
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if dataset not in PREFIX:
        raise typer.BadParameter(f"unknown dataset {dataset!r}; choose from {sorted(PREFIX)}")

    prefix = PREFIX[dataset]
    trans_path = input / f"{prefix}_Trans.csv"
    patt_path = input / f"{prefix}_Patterns.txt"
    outdir = out / dataset
    outdir.mkdir(parents=True, exist_ok=True)
    for p in (trans_path, patt_path):
        if not p.exists():
            raise SystemExit(f"missing input: {p}")

    log.info("[1/5] loading %s", trans_path.name)
    df = load_transfers(trans_path)
    n_flag = df.filter(pl.col("is_laundering") == 1).select(pl.len()).item()
    log.info("  rows: %d, flagged laundering: %d (1 in %.0f)",
             len(df), n_flag, len(df) / max(n_flag, 1))

    log.info("[2/5] deriving USD rates")
    rates = derive_fx(df)
    bad = rates.filter((pl.col("n") >= FX_MIN_PAIR_ROWS)
                       & (pl.col("p95_p5") > FX_MAX_P95_P5))
    log.info("  mixed-currency rows: %d, pairs: %d", rates["n"].sum(), len(rates))
    log.info("  wide-spread pairs (n>=%d, p95/p5>%.1f), medians still used: %d",
             FX_MIN_PAIR_ROWS, FX_MAX_P95_P5, len(bad))
    for row in bad.iter_rows(named=True):
        log.info("    %s -> %s: n=%d, p95/p5=%.3f",
                 row["payment_currency"], row["receiving_currency"],
                 row["n"], row["p95_p5"])
    df = add_usd(df, rates)
    n_null_usd = df.filter(pl.col("amount_usd").is_null()).select(pl.len()).item()
    log.info("  rows with no USD conversion: %d", n_null_usd)

    log.info("[3/5] sorting and writing transfers.parquet")
    df = df.sort(["timestamp", "transfer_id"]).drop("ts_raw")
    df.write_parquet(outdir / "transfers.parquet")
    rates.write_parquet(outdir / "fx_rates.parquet")

    log.info("[4/5] parsing pattern instances")
    with open(trans_path) as f:
        csv_header = f.readline()
    members, instances = parse_patterns(patt_path, csv_header)
    log.info("  instances: %d", len(instances))
    # Re-derive the raw timestamp text for the join: the parsed column was
    # dropped after sorting, and the join must compare raw text to raw text.
    df_rawkeys = df.with_columns(
        pl.col("timestamp").dt.strftime(TS_FMT).alias("ts_raw"))
    joined = join_members(df_rawkeys, members)
    n_covered = (joined.filter(pl.col("transfer_id").is_not_null())
                 .select(pl.col("transfer_id").n_unique()).item())
    log.info("  flagged rows covered by >=1 instance: %d of %d (%d flagged but unattributed)",
             n_covered, n_flag, n_flag - n_covered)
    if n_covered < n_flag:
        covered_ids = joined.filter(pl.col("transfer_id").is_not_null()
                                    ).select("transfer_id").unique()
        uncovered = (df_rawkeys.filter(pl.col("is_laundering") == 1)
                       .join(covered_ids, on="transfer_id", how="anti"))
        dup_like = uncovered.join(
            members.rename({"timestamp": "ts_raw"}).select(JOIN_KEYS).unique(),
            on=JOIN_KEYS, how="semi").select(pl.len()).item()
        log.info("    of which exact duplicates of a pattern row: %d, absent from %s: %d",
                 dup_like, patt_path.name, len(uncovered) - dup_like)

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
    log.info("  instances per type:\n%s",
             inst_df.group_by(["pattern_type", "motif"]).len().sort("pattern_type"))

    log.info("[5/5] rows and flagged laundering per day")
    with pl.Config(tbl_rows=-1):
        log.info("\n%s", df.group_by("day").agg(pl.len().alias("n"), pl.col("is_laundering").sum()
                 .alias("laundering")).sort("day"))
    log.info("done -> %s", outdir)


if __name__ == "__main__":
    typer.run(main)
