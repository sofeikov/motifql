"""RQ4: can a model write correct motifs? (Oct 31, DESIGN.md §8).

For each English item: run the repair loop (arm `repair`, up to N
attempts with sanity feedback) and a single-shot baseline (arm
`oneshot`), then score witness-set Jaccard against the hand-written
reference on a fixed train-anchor sample (seeded: every train
instance's last-member anchor plus uniform background fill).

Sanity (#15 #7): the repair candidate scores a sanity sample of all
train truth anchors plus background disjoint from the eval fill. The
only feedback signals are a shape failure (no truth anchor fires)
and an over-broad motif (background rate above SANITY_HI); fixed
low-rate thresholds cannot work at 0.1% truth rates (0 background
fires is the expected outcome for a rare-pattern motif, and the old
"loosen" feedback degraded Jaccard 0.957 -> 0.629).

Primary metric: share compiling within N attempts. Secondary: Jaccard
over fired witness edge-sets (None when the reference fires nothing on
the sample — uninformative, flagged for review). Reference witness sets
are cached by (sample, reference text); model calls are cached by
prompt hash, so reruns are free. Resume matches (item, arm, attempts,
sample, seed, reference) by default; --fresh redoes the requested arms.

Usage:
    uv run eval/rq4.py --check            # compile all references, no model/data
    uv run eval/rq4.py --limit 5          # pilot: 5 items x both arms
    uv run eval/rq4.py                    # full run (default 50 items)
    uv run eval/rq4.py --tier intent --compile-only   # loose tier, no data:
                                          # compile rate only (Jaccard needs
                                          # the train sample); rubric-graded
                                          # by hand from the saved motifs
Tiers (issue #23): `precise` (default, 50 pinned items) and `intent`
(10 loose items, `rq4_intent_items.py`, each with 1-2 acceptable
references — full-tier Jaccard is the max over refs). Tiers keep
SEPARATE runs CSVs; never merge the numbers.
"""

from __future__ import annotations

import hashlib
import json
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
from eval.rq4_intent_items import INTENT_ITEMS  # noqa: E402
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
INTENT_RUNS_CSV = Path("eval/results/rq4_intent_runs.csv")
COMPILED_JSON = Path("eval/results/rq4_compiled.json")
INTENT_COMPILED_JSON = Path("eval/results/rq4_intent_compiled.json")


def tier_items(tier: str) -> list[dict]:
    """Normalised items: every item carries a `references` list.

    Precise items wrap their single reference (row ids, ref hashes and
    the runs CSV stay byte-compatible with earlier runs); intent items
    carry the disjunction as-is.
    """
    if tier == "intent":
        return INTENT_ITEMS
    if tier == "precise":
        return [{"id": it["id"], "english": it["english"],
                 "references": [it["reference"]]} for it in ITEMS]
    raise ValueError(f"unknown tier: {tier!r} (want precise|intent)")


def tier_ref_hash(references: list[str]) -> str:
    """Provenance hash: single refs hash the text (legacy); intent joins."""
    text = references[0] if len(references) == 1 else "\n---\n".join(references)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def tier_runs_csv(tier: str) -> Path:
    return INTENT_RUNS_CSV if tier == "intent" else RUNS_CSV


def tier_compiled_json(tier: str) -> Path:
    return INTENT_COMPILED_JSON if tier == "intent" else COMPILED_JSON
SANITY_BG_N = 1000
SANITY_BG_SEED = 7
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


def _engine_hash() -> str:
    """Hash of the engine sources: any scoring change rebuilds refs.

    Without this, cached reference witness sets silently survive engine
    fixes and candidates are compared against stale oracles (#15).
    """
    h = hashlib.sha256()
    src = Path(__file__).resolve().parent.parent / "src" / "motifql"
    for name in sorted(p.name for p in src.glob("*.py")):
        h.update((src / name).read_bytes())
    return h.hexdigest()[:16]


def _load_or_build_refs(graph, sample: np.ndarray, seed: int,
                      sample_n: int, data: Path,
                      items: list[dict]) -> dict[str, list[set]]:
    """Reference witness sets, cached by (sample, reference text, engine).

    The key carries a hash of the evaluated sample plus the run's seed,
    sample size, data dir and engine sources: reruns that change any of
    them rebuild instead of comparing candidates against stale oracles.
    Items carry a `references` list; single-ref rows keep the legacy
    `item_id` (old cache entries still hit), multi-ref rows are
    `item_id#k`. Returns item id -> per-reference witness sets in order.
    """
    units: list[tuple[str, str, str]] = []  # (row id, item id, ref text)
    for it in items:
        refs = it["references"]
        for k, ref in enumerate(refs):
            row_id = it["id"] if len(refs) == 1 else f"{it['id']}#{k}"
            units.append((row_id, it["id"], ref))
    sample_hash = hashlib.sha256(np.ascontiguousarray(sample).tobytes()).hexdigest()[:16]
    engine_hash = _engine_hash()
    want = {row_id: hashlib.sha256(ref.encode()).hexdigest()[:16]
            for row_id, _, ref in units}
    per_item: dict[str, dict[int, set]] = {it["id"]: {} for it in items}
    if REF_CACHE.exists():
        frame = pl.read_parquet(REF_CACHE)
        if {"sample_hash", "sample_n", "data", "engine_hash"} <= set(frame.columns):
            frame = frame.filter(
                (pl.col("sample_hash") == sample_hash)
                & (pl.col("seed") == seed)
                & (pl.col("sample_n") == sample_n)
                & (pl.col("engine_hash") == engine_hash)
                & (pl.col("data") == str(data)))
            for row in frame.iter_rows(named=True):
                if want.get(row["item_id"]) == row["ref_hash"]:
                    item_id, _, k = row["item_id"].partition("#")
                    per_item[item_id][int(k or 0)] = {
                        frozenset(map(int, w)) for w in row["witnesses"]}
    missing = [(row_id, item_id, ref) for row_id, item_id, ref in units
               for k in [int(row_id.partition("#")[2] or 0)]
               if k not in per_item[item_id]]
    if missing:
        log.info("scoring %d references over %d sampled anchors", len(missing), len(sample))
    rows = []
    for row_id, item_id, ref in missing:
        doc = parse_text(ref)
        plan = compile_motif(doc, "ref")
        libs = libs_from_doc(doc)
        wit = _witnesses(graph, sample, plan, libs)
        per_item[item_id][int(row_id.partition("#")[2] or 0)] = wit
        rows.append({"item_id": row_id, "ref_hash": want[row_id],
                     "seed": seed, "sample_n": sample_n, "data": str(data),
                     "sample_hash": sample_hash, "engine_hash": engine_hash,
                     "witnesses": [sorted(w) for w in wit]})
        log.info("  ref %-24s fires %d witness sets", row_id, len(wit))
    if rows:
        REF_CACHE.parent.mkdir(parents=True, exist_ok=True)
        new = pl.DataFrame(rows, schema={"item_id": pl.String, "ref_hash": pl.String,
                                         "seed": pl.Int64, "sample_n": pl.Int64,
                                         "data": pl.String, "sample_hash": pl.String,
                                         "engine_hash": pl.String,
                                         "witnesses": pl.List(pl.List(pl.Int64))})
        if REF_CACHE.exists():
            old = pl.read_parquet(REF_CACHE)
            new = pl.concat([old, new], how="diagonal").unique(
                subset=["sample_hash", "seed", "item_id"], keep="last",
                maintain_order=True).filter(
                    pl.col("sample_hash").is_not_null())
        new.write_parquet(REF_CACHE)
    return {item_id: [per_item[item_id][k]
                      for k in sorted(per_item[item_id])]
            for item_id in per_item}


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
    return sample, sorted(truth), fill.tolist()


def sanity_background_ids(anchor_pos: np.ndarray, eval_used: set[int],
                            n: int = SANITY_BG_N,
                            seed: int = SANITY_BG_SEED) -> list[int]:
    """Background positions for repair sanity, disjoint from eval.

    The old sanity slice came from the eval sample itself, so feedback
    was derived from eval data. Truth anchors are shared deliberately
    (they are the repair target, not a secret); only the background
    fill is disjoint.
    """
    pool = [int(p) for p in anchor_pos.tolist() if int(p) not in eval_used]
    rng = np.random.default_rng(seed)
    return rng.choice(pool, size=min(n, len(pool)), replace=False).tolist()


def sanity_feedback(truth_hit: int, n_truth: int, bg_hit: int, n_bg: int) -> str | None:
    """Repair feedback from a sanity scoring, if any is warranted.

    A zero background count is the expected outcome for a rare-pattern
    motif, so there is no low-rate rule — only a shape failure (no
    train-truth anchor fires) and an over-broad motif (background rate
    above SANITY_HI) speak.
    """
    if n_truth and truth_hit == 0:
        return (f"fires on 0/{n_truth} train truth anchors: the shape is wrong, "
                "not the thresholds — re-read the pattern before loosening anything")
    if n_bg and bg_hit / n_bg > SANITY_HI:
        return (f"background fire rate {bg_hit}/{n_bg} is over 20%: the motif is "
                "too broad — tighten the shape or thresholds")
    return None


def _complete(messages):
    text, usage = spark.complete(messages)
    return text, usage


def _max_jaccard(ref_wits: list[set], cand_wit: set) -> float | None:
    """Best overlap across the disjunction (None iff no ref fires)."""
    scores = [_jaccard(r, cand_wit) for r in ref_wits]
    scores = [s for s in scores if s is not None]
    return max(scores) if scores else None


def _run_item(item: dict, arm: str, max_attempts: int, graph,
              sample: np.ndarray, ref_wits: list[set], prov: dict,
              sanity_truth: list[int], sanity_bg: list[int]) -> dict:
    def sanity(compiled) -> list[str]:
        def fires(positions: list[int]) -> int:
            n = 0
            for pos in positions:
                try:
                    res = score_any(graph, int(pos), compiled.plan, compiled.libs,
                                    prune=True)
                except Exception as exc:
                    raise RuntimeError(
                        f"scoring crashed: {type(exc).__name__}: {exc}"[:300])
                n += bool(res.found)
            return n

        try:
            truth_hit = fires(sanity_truth)
            bg_hit = fires(sanity_bg)
        except RuntimeError as exc:
            return [str(exc)]
        fb = sanity_feedback(truth_hit, len(sanity_truth), bg_hit, len(sanity_bg))
        return [fb] if fb else []

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
            "attempts": res.attempts,
            "motif": res.text if res.compiled is not None else None,
            "jaccard": _max_jaccard(ref_wits, cand_wit),
            "ref_fires": max(len(r) for r in ref_wits), "cand_fires": len(cand_wit),
            "prompt_tokens": tok_in, "completion_tokens": tok_out,
            "errors": " | ".join(res.errors)[:500],
            "max_attempts": prov["max_attempts"], "sample_n": prov["sample_n"],
            "seed": prov["seed"], "ref_hash": prov["ref_hash"]}


@app.command()
def main(
    data: Annotated[Path, typer.Option(help="Prepped dataset directory.")] = Path("data/hi-small"),
    sample_n: Annotated[int, typer.Option(help="Eval anchors (truth + background).")] = 2000,
    seed: Annotated[int, typer.Option(help="Sample seed.")] = 7,
    limit: Annotated[int, typer.Option(help="Items to run (0 = all).")] = 0,
    max_attempts: Annotated[int, typer.Option(help="Repair attempts per item.")] = 3,
    arms: Annotated[str, typer.Option(help="Comma-separated: repair,repair_nosanity,oneshot.")] = "repair,oneshot",
    check: Annotated[bool, typer.Option(help="Compile references only; no model/data.")] = False,
    fresh: Annotated[bool, typer.Option(help="Redo these arms (drops their rows; other arms survive).")] = False,
    tier: Annotated[str, typer.Option(help="Item tier: precise (50) or intent (10 loose).")] = "precise",
    compile_only: Annotated[bool, typer.Option(help="Repair loop only, no data/graph: compile rate + saved motifs (intent rubric input).")] = False,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if tier not in ("precise", "intent"):
        raise typer.BadParameter(f"unknown tier: {tier!r} (want precise|intent)")
    if check:
        for name, tier_items_ in (("precise", tier_items("precise")),
                                  ("intent", tier_items("intent"))):
            n = 0
            for it in tier_items_:
                for ref in it["references"]:
                    doc = parse_text(ref)
                    plan = compile_motif(doc, "ref")
                    log.info("  %-24s %s", it["id"], type(plan).__name__)
                    n += 1
            log.info("%s tier: %d references compile", name, n)
        return
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    all_items = tier_items(tier)
    items = all_items[:limit] if limit > 0 else all_items
    if compile_only:
        _compile_only(items, arm_list, max_attempts, tier)
        return
    runs_csv = tier_runs_csv(tier)
    motifs_path = runs_csv.with_name(runs_csv.stem + "_motifs.json")
    ref_hashes = {it["id"]: tier_ref_hash(it["references"]) for it in all_items}
    if runs_csv.exists() and fresh:
        # Redo these arms only; other arms' rows survive.
        remaining = pl.read_csv(runs_csv).filter(~pl.col("arm").is_in(arm_list))
        if len(remaining):
            remaining.write_csv(runs_csv)
        else:
            runs_csv.unlink()
        _drop_motif_texts(motifs_path, arm_list)
    done: set[tuple] = set()
    if runs_csv.exists():
        frame = pl.read_csv(runs_csv)
        if {"max_attempts", "sample_n", "seed", "ref_hash"} <= set(frame.columns):
            for row in frame.iter_rows(named=True):
                done.add((row["item_id"], row["arm"], row["max_attempts"],
                          row["sample_n"], row["seed"], row["ref_hash"]))
        else:
            log.warning("runs CSV predates provenance columns; rerunning everything")
    def _key(it, a):
        att = 1 if a == "oneshot" else max_attempts
        return (it["id"], a, att, sample_n, seed, ref_hashes[it["id"]])

    todo = [(it, a) for it in items for a in arm_list
            if _key(it, a) not in done]
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
    sample, sanity_truth, eval_fill = _sample_anchors(
        data, graph, scoped, anchor_pos, sample_n, seed)
    sanity_bg = sanity_background_ids(
        anchor_pos, set(sanity_truth) | set(eval_fill))
    log.info("sanity: %d truth anchors + %d disjoint background",
             len(sanity_truth), len(sanity_bg))
    ref_wits = _load_or_build_refs(graph, sample, seed, sample_n, data, items)
    runs_csv.parent.mkdir(parents=True, exist_ok=True)
    header_written = runs_csv.exists()
    for i, (it, arm) in enumerate(todo):
        log.info("[%d/%d] %s (%s)", i + 1, len(todo), it["id"], arm)
        prov = {"max_attempts": _key(it, arm)[2],
                "sample_n": sample_n, "seed": seed,
                "ref_hash": ref_hashes[it["id"]]}
        try:
            row = _run_item(it, arm, max_attempts, graph, sample,
                            ref_wits[it["id"]], prov, sanity_truth, sanity_bg)
        except Exception as exc:  # harness never dies on one item
            log.warning("  item failed: %r", exc)
            row = {"item_id": it["id"], "arm": arm, "compiled": False,
                   "attempts": 0, "motif": None, "jaccard": None,
                   "ref_fires": max(len(r) for r in ref_wits[it["id"]]),
                   "cand_fires": 0, "prompt_tokens": 0, "completion_tokens": 0,
                   "errors": f"harness: {exc}"[:500], **prov}
        _save_motif_text(motifs_path, it["id"], arm, row["compiled"],
                         row.pop("motif"))
        frame = pl.DataFrame([row])
        with open(runs_csv, "a" if header_written else "w") as handle:
            frame.write_csv(handle, include_header=not header_written)
        header_written = True
        log.info("  compiled=%s attempts=%d jaccard=%s ref/cand=%d/%d tokens=%d/%d",
                 row["compiled"], row["attempts"],
                 "n/a" if row["jaccard"] is None else f"{row['jaccard']:.2f}",
                 row["ref_fires"], row["cand_fires"],
                 row["prompt_tokens"], row["completion_tokens"])
    log.info("done; results in %s", runs_csv)


def _motif_sidecar(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []
    return rows if isinstance(rows, list) else []


def _save_motif_text(path: Path, item_id: str, arm: str,
                     compiled: bool, motif: str | None) -> None:
    """Compiled motif texts beside the runs CSV (rubric-grading input).

    The runs CSV keeps its legacy columns (old rows still resume);
    texts live in `<stem>_motifs.json` keyed by (item, arm).
    """
    rows = [r for r in _motif_sidecar(path)
            if (r.get("item_id"), r.get("arm")) != (item_id, arm)]
    rows.append({"item_id": item_id, "arm": arm, "compiled": compiled,
                 "motif": motif})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2))


def _drop_motif_texts(path: Path, arm_list: list[str]) -> None:
    rows = [r for r in _motif_sidecar(path) if r.get("arm") not in arm_list]
    if rows:
        path.write_text(json.dumps(rows, indent=2))
    else:
        path.unlink(missing_ok=True)


def _compile_only(items: list[dict], arm_list: list[str],
                  max_attempts: int, tier: str) -> None:
    """Repair loop without data: compile rate plus saved motif texts.

    No sanity feedback (there is no graph), so `repair` and
    `repair_nosanity` coincide here. Motif texts are the intent tier's
    rubric-grading input; full-tier Jaccard needs a data rerun.
    """
    out = tier_compiled_json(tier)
    rows = []
    for i, it in enumerate(items):
        for arm in arm_list:
            log.info("[%d/%d] %s (%s)", i + 1, len(items), it["id"], arm)
            attempts = 1 if arm == "oneshot" else max_attempts
            try:
                res = repair.repair_loop(
                    it["english"], _complete, max_attempts=attempts)
                fresh = [u for u in res.usage if not u.get("cached")]
                row = {"item_id": it["id"], "arm": arm,
                       "compiled": res.compiled is not None,
                       "attempts": res.attempts,
                       "motif": res.text if res.compiled is not None else None,
                       "errors": " | ".join(res.errors)[:500],
                       "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in fresh),
                       "completion_tokens": sum(u.get("completion_tokens", 0)
                                                for u in fresh)}
            except Exception as exc:  # harness never dies on one item
                log.warning("  item failed: %r", exc)
                row = {"item_id": it["id"], "arm": arm, "compiled": False,
                       "attempts": 0, "motif": None,
                       "errors": f"harness: {exc}"[:500],
                       "prompt_tokens": 0, "completion_tokens": 0}
            rows.append(row)
            log.info("  compiled=%s attempts=%d", row["compiled"], row["attempts"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    n = sum(1 for r in rows if r["compiled"])
    log.info("%s tier compile-only: %d/%d compiled; motifs in %s",
             tier, n, len(rows), out)


if __name__ == "__main__":
    app()
