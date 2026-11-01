"""English-to-motif service behind POST /compile (DESIGN.md 6.6, 7.1).

Runs the §7.1 repair loop and scores the candidate on a fixed engine
sample, returning the motif plus its best witnesses — the "is this
what you meant?" step. The model transport is injected (`complete`),
so tests stub it; the committed canned pair (`ui/canned.json`) keeps
the scripted demo deterministic without a model key. Every response
says where its motif came from: `model`, `cache` or `canned`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from motifql.engine import Engine, MotifResult, WitnessHop
from motifql.graph import TemporalGraph
from motifql.groups import score_any
from motifql.matcher import EXACT_FLOOR
from motifql.models import narrate as narrate_mod
from motifql.models.repair import RepairResult, repair_loop, try_compile
from motifql.models.spark import SparkError, complete as spark_complete

#: Default seed for evasion noise draws (matches `eval/perturb.py`).
EVASION_SEED = 12345


class CompileError(Exception):
    """The English could not be turned into a motif (model or data)."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class Canned:
    english: str
    motif: str
    narrative: str | None = None  # committed story for the scripted anchor
    narrative_anchor_tid: int | None = None
    evasion: dict | None = None  # precomputed slider curve, if committed


@dataclass
class CompileResult:
    motif: str
    compiled: bool
    attempts: int
    errors: list[str]
    source: str  # "model" | "cache" | "canned"
    witnesses: list[MotifResult]
    narratives: list[str | None] | None = None  # parallel to witnesses
    narrative_weakest: list[int | None] | None = None  # 1-based hop or None


def load_canned(path: str | Path) -> Canned | None:
    """Read the committed scripted-demo pair; None when absent."""
    p = Path(path)
    if not p.exists():
        return None
    obj = json.loads(p.read_text())
    return Canned(english=obj["english"], motif=obj["motif"],
                  narrative=obj.get("narrative"),
                  narrative_anchor_tid=obj.get("narrative_anchor_tid"),
                  evasion=obj.get("evasion_curve"))


def _witness_dict(w: MotifResult) -> dict:
    """MotifResult as the plain-dict witness narrate() consumes."""
    return {"hops": [{"transfer_id": h.transfer_id, "timestamp": h.timestamp,
                      "from_bank": h.from_bank, "from_account": h.from_account,
                      "to_bank": h.to_bank, "to_account": h.to_account,
                      "amount": h.amount} for h in w.witness],
            "explanation": w.explanation, "top_score": w.top_score,
            "found": w.found}


def compile_english(english: str, engine: Engine, *,
                    complete=spark_complete,
                    max_attempts: int = 3,
                    n_witnesses: int = 3,
                    sample_n: int = 3000,
                    sample_seed: int = 7,
                    anchor_tids: list[int] | None = None,
                    canned: Canned | None = None,
                    narrate_complete=None) -> CompileResult:
    """Compile English to a motif and show its best sample witnesses.

    `anchor_tids` pins exact anchors (the scripted demo); otherwise a
    seeded sample is ranked by top_score. Raises CompileError when the
    model is unreachable (and no canned pair matches) or a pinned
    anchor is absent from the graph.

    `narrate_complete` is the model transport for witness narratives
    (None sketches none — every narrative stays None). Narration fails
    closed per witness: the canned story covers the scripted anchor
    offline, live witnesses get a model story, and anything else —
    model down, bad witness, failed verification — yields None.
    """
    text = english.strip()
    if not text:
        raise CompileError("english must not be empty", 422)
    try:
        result = repair_loop(text, complete, max_attempts=max_attempts)
    except SparkError as exc:
        result = None
        model_error = str(exc)
    else:
        model_error = ""
    source = "model"
    if result is None or result.compiled is None:
        if canned is not None and text == canned.english.strip():
            cand, errs = try_compile(canned.motif)
            if cand is None:  # pragma: no cover - committed pair compiles
                raise CompileError(f"canned motif broken: {errs}")
            result = RepairResult(canned.motif, cand, 0, [], [])
            source = "canned"
        elif result is None:
            raise CompileError(f"model unavailable: {model_error}")
        else:
            return CompileResult(result.text, False, result.attempts,
                                 result.errors, source, [])
    elif result.usage and all(u.get("cached") for u in result.usage):
        source = "cache"
    assert result.compiled is not None
    witnesses = _witnesses(engine, result.compiled, n_witnesses, sample_n,
                           sample_seed, anchor_tids or [])
    narratives, weakest = _narratives(text, result.text, witnesses,
                                      narrate_complete, canned)
    return CompileResult(result.text, True, result.attempts, result.errors,
                         source, witnesses, narratives, weakest)


def _narratives(english: str, motif_text: str, witnesses: list[MotifResult],
                narrate_complete, canned: Canned | None
                ) -> tuple[list[str | None], list[int | None]]:
    """Per-witness stories; the canned anchor needs no model call."""
    stories: list[str | None] = []
    steps: list[int | None] = []
    for w in witnesses:
        weak = narrate_mod.weakest(w.explanation)
        steps.append(weak[2] if weak else None)
        anchor = w.witness[-1].transfer_id if w.witness else None
        if (canned is not None and canned.narrative
                and anchor is not None
                and anchor == canned.narrative_anchor_tid):
            stories.append(canned.narrative)
        elif narrate_complete is None:
            stories.append(None)
        else:
            try:
                res = narrate_mod.narrate(english, motif_text,
                                          _witness_dict(w),
                                          complete=narrate_complete)
            except Exception:
                res = None
            stories.append(res.text if res is not None else None)
            if res is not None and res.text is not None:
                steps[-1] = res.weakest_step
    return stories, steps


def _witnesses(engine: Engine, compiled, n_witnesses: int, sample_n: int,
               seed: int, anchor_tids: list[int]) -> list[MotifResult]:
    graph = engine.graph
    if anchor_tids:
        positions = []
        for tid in anchor_tids:
            pos = graph.pos_of(tid)
            if pos < 0:
                raise CompileError(f"anchor transfer {tid} not in graph", 422)
            positions.append(pos)
    else:
        n = graph.n_edges
        if n == 0 or n_witnesses <= 0:
            return []
        rng = np.random.default_rng(seed)
        sample = np.sort(rng.choice(n, min(sample_n, n), replace=False))
        tops = [(score_any(graph, int(p), compiled.plan, compiled.libs,
                           floor=EXACT_FLOOR).top_score, int(p)) for p in sample]
        tops.sort(key=lambda t: (-t[0], t[1]))
        positions = [p for s, p in tops[:n_witnesses] if s > 0]
    return [engine.score_with(p, compiled.plan, compiled.libs) for p in positions]


@dataclass
class EvasionWitness:
    """One pinned witness re-scored under synthetic evasion noise."""

    anchor_tid: int
    fuzzy_top: float
    crisp_top: float  # exact arm at alpha = emit (the RQ2 fair-exact rule)
    crisp_alpha: float
    found: bool  # fuzzy score at or above emit
    baseline_fuzzy: float  # unperturbed pair, for "0.71 -> 0.52" captions
    baseline_crisp: float
    witness: list[WitnessHop] = field(default_factory=list)  # perturbed hops
    explanation: list[dict] = field(default_factory=list)  # fuzzy degrees


def _witness_frame(hops: list[WitnessHop]) -> pl.DataFrame:
    """Prep-schema frame for witness hops (naive UTC, like the prep)."""
    return pl.DataFrame({
        "transfer_id": [h.transfer_id for h in hops],
        "timestamp": [datetime.fromtimestamp(h.timestamp, tz=timezone.utc)
                      .replace(tzinfo=None) for h in hops],
        "from_bank": [h.from_bank for h in hops],
        "from_account": [h.from_account for h in hops],
        "to_bank": [h.to_bank for h in hops],
        "to_account": [h.to_account for h in hops],
        "amount_usd": [h.amount for h in hops],
    })


def _perturb_witness_frame(df: pl.DataFrame, amount_pct: float,
                           gap_factor: float, seed: int) -> pl.DataFrame:
    """`eval/perturb.py` noise restricted to one witness (same draws).

    Amounts scale per transfer_id from SeedSequence(seed, tid), so one
    witness's noise never depends on which others share the run; gaps
    scale in integer microseconds with order preserved. Level (0, x1)
    is bit-identical to the input.
    """
    tids = df["transfer_id"].to_list()
    us = (df.select(pl.col("timestamp").dt.epoch(time_unit="us"))
          .to_series().to_list())
    t0 = min(us)
    new_us = [t0 + int(round(gap_factor * (u - t0))) for u in us]
    amts = df["amount_usd"].to_list()
    new_amt = [a * (1.0 + float(np.random.default_rng([seed, int(t)])
                                    .uniform(-amount_pct, amount_pct)))
               for a, t in zip(amts, tids)]
    return df.with_columns(
        pl.Series("timestamp",
                  [datetime.fromtimestamp(u / 1_000_000, tz=timezone.utc)
                   .replace(tzinfo=None) for u in new_us],
                  dtype=pl.Datetime("us")),
        pl.Series("amount_usd", new_amt),
    ).sort("timestamp", "transfer_id")


def _frame_hops(df: pl.DataFrame) -> list[WitnessHop]:
    """Witness hops from a (possibly perturbed) frame, oldest first."""
    hops = []
    for row in df.iter_rows(named=True):
        ts = int(row["timestamp"].replace(tzinfo=timezone.utc).timestamp())
        hops.append(WitnessHop(
            transfer_id=int(row["transfer_id"]), timestamp=ts,
            from_bank=row["from_bank"], from_account=row["from_account"],
            to_bank=row["to_bank"], to_account=row["to_account"],
            amount=float(row["amount_usd"])))
    return hops


def _fixed_witness_degrees(graph: TemporalGraph, plan, libs) -> tuple:
    """Per-predicate degrees of the whole tiny graph as one witness.

    The path matcher drops the witness below emit (top_score only), but
    the evasion bars must keep showing the displayed witness sinking, so
    below emit the degrees are evaluated on the fixed witness via the
    matcher's shared explanation helper (byte-identical to search).
    Group plans always carry their witness; only path plans land here.
    """
    from motifql import planner as _planner
    from motifql.matcher import _witness_parts

    if not isinstance(plan, _planner.PathPlan):
        return ()
    order = sorted(range(graph.n_edges),
                   key=lambda e: (int(graph.times[e]),
                                  int(graph.transfer_id[e])))
    _, degs = _witness_parts(graph, tuple(order), plan, libs, False, 0.0)
    return tuple(degs)


def score_evasion(motif_text: str, engine: Engine, *,
                  anchor_tids: list[int],
                  amount_noise_pct: float = 0.0,
                  gap_scale: float = 1.0,
                  seed: int = EVASION_SEED,
                  ) -> tuple[float, list[EvasionWitness]]:
    """Re-score pinned witnesses under synthetic evasion noise.

    Each anchor's live witness is perturbed (amount jitter + order
    preserving gap scaling, same draws as `eval/perturb.py`) and
    re-scored in a tiny witness-only graph through the engine's own
    `score_any`: fuzzy with the exact floor, crisp (exact) at
    alpha = emit. Witness-only search keeps the demo honest about what
    it shows — the displayed witness degrading, not a full replay
    where background candidates could rescue the anchor — and instant
    (no 5M-row rebuild per slider tick). At (0, x1) the perturbed
    frame is bit-identical, so fuzzy_top reproduces the live score
    exactly. Returns (emit, witnesses).
    """
    if not anchor_tids:
        raise CompileError("anchor_tids must not be empty", 422)
    if not 0.0 <= amount_noise_pct <= 1.0:
        raise CompileError("amount_noise_pct must be in [0, 1]", 422)
    if not gap_scale > 0.0:
        raise CompileError("gap_scale must be > 0", 422)
    compiled, errors = try_compile(motif_text)
    if compiled is None:
        raise CompileError("motif does not compile: " + "; ".join(errors),
                           422)
    plan, libs = compiled.plan, compiled.libs
    emit = float(plan.emit)
    out: list[EvasionWitness] = []
    for tid in anchor_tids:
        pos = engine.graph.pos_of(int(tid))
        if pos < 0:
            raise CompileError(f"anchor transfer {tid} not in graph", 422)
        base = engine.score_with(pos, plan, libs)
        if not base.witness:
            out.append(EvasionWitness(int(tid), base.top_score, 0.0, emit,
                                      False, base.top_score, 0.0))
            continue
        frame = _witness_frame(base.witness)
        perturbed = _perturb_witness_frame(frame, amount_noise_pct,
                                           gap_scale, seed)
        tiny = TemporalGraph.from_transfers(perturbed)
        apos = tiny.pos_of(int(tid))
        assert apos >= 0  # the anchor is a witness member by construction
        fuzzy = score_any(tiny, apos, plan, libs, floor=EXACT_FLOOR)
        crisp = score_any(tiny, apos, plan, libs, crisp=True,
                          crisp_alpha=emit)
        degs = (fuzzy.degrees if fuzzy.found
                else _fixed_witness_degrees(tiny, plan, libs))
        tiny0 = TemporalGraph.from_transfers(frame)
        apos0 = tiny0.pos_of(int(tid))
        baseline_fuzzy = score_any(tiny0, apos0, plan, libs,
                                   floor=EXACT_FLOOR).top_score
        baseline_crisp = score_any(tiny0, apos0, plan, libs, crisp=True,
                                   crisp_alpha=emit).top_score
        out.append(EvasionWitness(
            anchor_tid=int(tid), fuzzy_top=fuzzy.top_score,
            crisp_top=crisp.top_score, crisp_alpha=emit,
            found=fuzzy.found, baseline_fuzzy=baseline_fuzzy,
            baseline_crisp=baseline_crisp,
            witness=_frame_hops(perturbed),
            explanation=[{"predicate": label, "degree": d, "step": s}
                         for label, d, s in degs]))
    return emit, out
