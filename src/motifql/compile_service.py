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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from motifql.engine import Engine, MotifResult
from motifql.groups import score_any
from motifql.matcher import EXACT_FLOOR
from motifql.models.repair import RepairResult, repair_loop, try_compile
from motifql.models.spark import SparkError, complete as spark_complete


class CompileError(Exception):
    """The English could not be turned into a motif (model or data)."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class Canned:
    english: str
    motif: str


@dataclass
class CompileResult:
    motif: str
    compiled: bool
    attempts: int
    errors: list[str]
    source: str  # "model" | "cache" | "canned"
    witnesses: list[MotifResult]


def load_canned(path: str | Path) -> Canned | None:
    """Read the committed scripted-demo pair; None when absent."""
    p = Path(path)
    if not p.exists():
        return None
    obj = json.loads(p.read_text())
    return Canned(english=obj["english"], motif=obj["motif"])


def compile_english(english: str, engine: Engine, *,
                    complete=spark_complete,
                    max_attempts: int = 3,
                    n_witnesses: int = 3,
                    sample_n: int = 3000,
                    sample_seed: int = 7,
                    anchor_tids: list[int] | None = None,
                    canned: Canned | None = None) -> CompileResult:
    """Compile English to a motif and show its best sample witnesses.

    `anchor_tids` pins exact anchors (the scripted demo); otherwise a
    seeded sample is ranked by top_score. Raises CompileError when the
    model is unreachable (and no canned pair matches) or a pinned
    anchor is absent from the graph.
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
    return CompileResult(result.text, True, result.attempts, result.errors,
                         source, witnesses)


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
