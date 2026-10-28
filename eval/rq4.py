"""RQ4: can a model write correct motifs? (Oct 31, DESIGN.md §8).

For each English item: run the repair loop (arm `repair`, up to N
attempts with sanity feedback) and a single-shot baseline (arm
`oneshot`), then score witness-set Jaccard against the hand-written
reference on a fixed train-anchor sample (seeded: every train
instance's last-member anchor plus uniform background fill).

Primary metric: share compiling within N attempts. Secondary: Jaccard
over fired witness edge-sets (None when the reference fires nothing on
the sample — uninformative, flagged for review). Reference witness sets
are cached by (item, reference hash, sample); model calls are cached by
prompt hash, so reruns are free. Resume by default; --fresh restarts.

Usage:
    uv run eval/rq4.py --check            # compile all references, no model/data
    uv run eval/rq4.py --limit 5          # pilot: 5 items x both arms
    uv run eval/rq4.py                    # full run (default 50 items)
"""

from __future__ import annotations

import hashlib
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval import splits  # noqa: E402
from eval.rq4_items import ITEMS  # noqa: E402
from motifql.degrees import libs_from_doc  # noqa: E402
from motifql.graph import TemporalGraph  # noqa: E402
from motifql.groups import score_any  # noqa: E402
from motifql.models import repair, spark  # noqa: E402
from motifql.parser import parse_text  # noqa: E402
from motifql.planner import compile_motif  # noqa: E402

log = logging.getLogger("rq4")

REF_CACHE = Path("eval/results/cache/rq4_refs.parquet")
RUNS_CSV = Path("eval/results/rq4_runs.csv")
SANITY_N = 400
SANITY_LO = 0.0
SANITY_HI = 0.20

app = typer.Typer(add_completion=False)


def _witnesses(graph, positions, plan, libs) -> set[frozenset[int]]:
    out: set[frozenset[int]] = set()
    for pos in positions.tolist():
        try:
            res = score_any(graph, int(pos), plan, libs, prune=True)
        except Exception:
            continue  # engine crash on a candidate: no witness, not a death
        if res.found:
            out.add(frozenset(int(graph.transfer_id[e]) for e in res.witness))
    return out


def _jaccard(ref: set, cand: set) -> float | None:
    if not ref:
        return None
    if not cand:
        return 0.0
    return len(ref & cand) / len(ref | cand)


def _load_or_build_refs(graph, sample: np.ndarray, seed: int) -> dict[str, set]:
    """Reference witness sets, cached by reference-text hash."""
    want = {it["id"]: hashlib.sha256(it["reference"].encode()).hexdigest()[:16]
            for it in ITEMS}
    cached: dict[str, set] = {}
    if REF_CACHE.exists():
        frame = pl.read_parquet(REF_CACHE)
        frame = frame.filter(pl.col("seed") == seed)
        for row in frame.iter_rows(named=True):
            if want.get(row["item_id"]) == row["ref_hash"]:
                cached[row["item_id"]] = {
                    frozenset(map(int, w)) for w in row["witnesses"]}
    missing = [it for it in ITEMS if it["id"] not in cached]
    if missing:
        log.info("scoring %d references over %d sampled anchors", len(missing), len(sample))
    rows = []
    for it in missing:
        doc = parse_text(it["reference"])
        plan = compile_motif(doc, "ref")
        libs = libs_from_doc(doc)
        wit = _witnesses(graph, sample, plan, libs)
        cached[it["id"]] = wit
        rows.append({"item_id": it["id"], "ref_hash": want[it["id"]], "seed": seed,
                     "witnesses": [sorted(w) for w in wit]})
        log.info("  ref %-24s fires %d witness sets", it["id"], len(wit))
    if rows:
        REF_CACHE.parent.mkdir(parents=True, exist_ok=True)
        new = pl.DataFrame(rows, schema={"item_id": pl.String, "ref_hash": pl.String,
                                         "seed": pl.Int64,
                                         "witnesses": pl.List(pl.List(pl.Int64))})
        if REF_CACHE.exists():
            old = pl.read_parquet(REF_CACHE).filter(
                pl.col("seed") != seed)
            new = pl.concat([old, new], how="vertical")
        new.write_parquet(REF_CACHE)
    return cached


def _sample_anchors(data: Path, graph, scoped: pl.DataFrame,
                    anchor_pos: np.ndarray, n: int, seed: int) -> np.ndarray:
    """Every train instance's last-member anchor plus background fill."""
    inst = splits.assign_instance_splits(
        pl.read_parquet(data / "pattern_instances.parquet"),
        pl.read_parquet(data / "pattern_members.parquet"),
        pl.read_parquet(data / "transfers.parquet"))
    memb = pl.read_parquet(data / "pattern_members.parquet")
    order = np.argsort(graph.transfer_id)
    tids = graph.transfer_id[order]
    anchor_set = set(anchor_pos.tolist())
    truth: set[int] = set()
    sub = inst.filter(pl.col("split") == "train")
    m = (memb.join(sub.select("instance_id"), on="instance_id", how="inner")
         .join(scoped.select("transfer_id", "timestamp"), on="transfer_id", how="inner"))
    for tids_in in m.group_by("instance_id").agg(
            pl.col("transfer_id").sort_by("timestamp")).get_column("transfer_id"):
        i = int(np.searchsorted(tids, tids_in[-1]))
        pos = int(order[i])
        if pos in anchor_set:
            truth.add(pos)
    rng = np.random.default_rng(seed)
    rest = np.array([p for p in anchor_pos if p not in truth])
    fill = rng.choice(rest, min(max(n - len(truth), 0), len(rest)), replace=False)
    sample = np.sort(np.array(sorted(truth) + fill.tolist()))
    log.info("sample: %d truth anchors + %d background = %d (seed %d)",
             len(truth), len(fill), len(sample), seed)
    return sample


def _complete(messages):
    text, usage = spark.complete(messages)
    return text, usage


def _run_item(item: dict, arm: str, max_attempts: int, graph,
              sample: np.ndarray, ref_wit: set) -> dict:
    sanity_sample = sample[:min(SANITY_N, len(sample))]

    def sanity(compiled) -> list[str]:
        fires = 0
        for pos in sanity_sample.tolist():
            try:
                res = score_any(graph, int(pos), compiled.plan, compiled.libs,
                                prune=True)
            except Exception as exc:
                return [f"scoring crashed: {type(exc).__name__}: {exc}"[:300]]
            fires += bool(res.found)
        rate = fires / max(len(sanity_sample), 1)
        if rate <= SANITY_LO:
            return [f"matches nothing on {len(sanity_sample)} sampled train "
                    f"anchors; loosen a condition (wider window, softer term, "
                    f"lower count gate)."]
        if rate > SANITY_HI:
            return [f"matches {rate:.0%} of sampled train traffic; tighten a "
                    f"condition (narrower window, stricter term, higher gate)."]
        return []

    # Arms: `repair` (compile + sanity feedback), `repair_nosanity`
    # (compile feedback only — isolates the sanity step's effect),
    # `oneshot` (single attempt, no feedback).
    attempts = 1 if arm == "oneshot" else max_attempts
    res = repair.repair_loop(
        item["english"], _complete, max_attempts=attempts,
        sanity=sanity if arm == "repair" else None)
    cand_wit: set = set()
    if res.compiled is not None:
        cand_wit = _witnesses(graph, sample, res.compiled.plan, res.compiled.libs)
    fresh = [u for u in res.usage if not u.get("cached")]
    tok_in = sum(u.get("prompt_tokens", 0) for u in fresh)
    tok_out = sum(u.get("completion_tokens", 0) for u in fresh)
    return {"item_id": item["id"], "arm": arm, "compiled": res.compiled is not None,
            "attempts": res.attempts, "jaccard": _jaccard(ref_wit, cand_wit),
            "ref_fires": len(ref_wit), "cand_fires": len(cand_wit),
            "prompt_tokens": tok_in, "completion_tokens": tok_out,
            "errors": " | ".join(res.errors)[:500]}


@app.command()
def main(
    data: Annotated[Path, typer.Option(help="Prepped dataset directory.")] = Path("data/hi-small"),
    sample_n: Annotated[int, typer.Option(help="Eval anchors (truth + background).")] = 2000,
    seed: Annotated[int, typer.Option(help="Sample seed.")] = 7,
    limit: Annotated[int, typer.Option(help="Items to run (0 = all).")] = 0,
    max_attempts: Annotated[int, typer.Option(help="Repair attempts per item.")] = 3,
    arms: Annotated[str, typer.Option(help="Comma-separated: repair,repair_nosanity,oneshot.")] = "repair,oneshot",
    check: Annotated[bool, typer.Option(help="Compile references only; no model/data.")] = False,
    fresh: Annotated[bool, typer.Option(help="Ignore previous runs/resume state.")] = False,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if check:
        for it in ITEMS:
            doc = parse_text(it["reference"])
            plan = compile_motif(doc, "ref")
            log.info("  %-24s %s", it["id"], type(plan).__name__)
        log.info("%d references compile", len(ITEMS))
        return
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    items = ITEMS[:limit] if limit > 0 else ITEMS
    done: set[tuple[str, str]] = set()
    if RUNS_CSV.exists() and not fresh:
        for row in pl.read_csv(RUNS_CSV).iter_rows(named=True):
            done.add((row["item_id"], row["arm"]))
    todo = [(it, a) for it in items for a in arm_list if (it["id"], a) not in done]
    log.info("%d items x %s: %d runs (%d cached)", len(items), arm_list,
             len(todo), len(items) * len(arm_list) - len(todo))
    if not todo:
        log.info("nothing to do")
        return
    tr0, tr1 = splits.TRAIN_START, splits.TRAIN_END
    df = pl.read_parquet(data / "transfers.parquet")
    scoped = df.filter(pl.col("day") <= tr1)
    graph = TemporalGraph.from_transfers(scoped)
    days = scoped.get_column("day").to_numpy()
    anchor_pos = np.flatnonzero(
        (days >= np.datetime64(str(tr0))) & (days <= np.datetime64(str(tr1))))
    log.info("graph edges: %d, train anchors: %d", graph.n_edges, len(anchor_pos))
    sample = _sample_anchors(data, graph, scoped, anchor_pos, sample_n, seed)
    ref_wits = _load_or_build_refs(graph, sample, seed)
    RUNS_CSV.parent.mkdir(parents=True, exist_ok=True)
    header_written = RUNS_CSV.exists() and not fresh
    if fresh and RUNS_CSV.exists():
        RUNS_CSV.unlink()
        header_written = False
    for i, (it, arm) in enumerate(todo):
        log.info("[%d/%d] %s (%s)", i + 1, len(todo), it["id"], arm)
        try:
            row = _run_item(it, arm, max_attempts, graph, sample, ref_wits[it["id"]])
        except Exception as exc:  # harness never dies on one item
            log.warning("  item failed: %r", exc)
            row = {"item_id": it["id"], "arm": arm, "compiled": False,
                   "attempts": 0, "jaccard": None, "ref_fires": len(ref_wits[it["id"]]),
                   "cand_fires": 0, "prompt_tokens": 0, "completion_tokens": 0,
                   "errors": f"harness: {exc}"[:500]}
        frame = pl.DataFrame([row])
        with open(RUNS_CSV, "a" if header_written else "w") as handle:
            frame.write_csv(handle, include_header=not header_written)
        header_written = True
        log.info("  compiled=%s attempts=%d jaccard=%s ref/cand=%d/%d tokens=%d/%d",
                 row["compiled"], row["attempts"],
                 "n/a" if row["jaccard"] is None else f"{row['jaccard']:.2f}",
                 row["ref_fires"], row["cand_fires"],
                 row["prompt_tokens"], row["completion_tokens"])
    log.info("done; results in %s", RUNS_CSV)


if __name__ == "__main__":
    app()
