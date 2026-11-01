"""Learned fuzzy calibration: parameter spaces for fitting membership functions.

Issue #24: the README headline is "you don't have to guess the threshold",
but someone still hand-picks trapezoid shoulders, `~=` widths, quantifier
ramps and `min` vs `product`. This module defines the fittable numeric
parameters per motif (search space, bounds committed in code), samples
them with a seeded RNG, and applies a sampled vector to a parsed document
to produce a fitted document. Motif structure (match/where/having,
quantifier choice, term shapes) is frozen — only numbers move.

Shared terms are copied, not mutated: fitting `round_trip` introduces a
private `patient_fit_rt` term and repoints that motif's soft conditions
at it, so `layering_chain` (which also uses `patient`) is unaffected.
The harness in `eval/calibrate.py` replays fitted documents on train-only
data; the pre-registered space, budget and seed live in
`docs/2026-11-08-issue24-prereg.md`.

Usage:
    from motifql.calibrate import apply_params, default_params, sample_params
    import numpy as np
    doc = parse_file("examples/amlworld.motif")
    rng = np.random.default_rng(24)
    params = sample_params("round_trip", rng)
    fitted = apply_params(doc, "round_trip", params)
    plan = compile_motif(fitted, "round_trip")
"""

from __future__ import annotations

import dataclasses

import numpy as np

from motifql.parser import (
    And, Approx, Comparison, Document, Dur, IsTerm, Not, Num, Or, Pct,
    Quantified, TermDef,
)

#: Bumped whenever a bound or default below changes; the harness records it
#: in run.json so a trace is always attributable to its space.
SPACE_VERSION = 1

#: Hand-tuned reference values (examples/amlworld.motif as of Nov 7).
#: Gap shoulders are hours, tolerances percent, ramps raw counts.
DEFAULTS: dict[str, dict[str, float | str]] = {
    "round_trip": {
        "gap_plateau_h": 24.0, "gap_support_h": 120.0,
        "tol_pct": 20.0, "score": "min", "emit": 0.5,
    },
    "layering_chain": {
        "gap_plateau_h": 24.0, "gap_support_h": 120.0,
        "tol_pct": 25.0, "score": "min", "emit": 0.5,
    },
    "scatter_gather": {
        "several_lo": 1.0, "several_hi": 3.0,
        "sum_tol_pct": 20.0, "score": "product", "emit": 0.5,
    },
}

#: Hard `where` span window per path motif, in hours. Fitted gap supports
#: must stay inside it — a gap shoulder beyond the window could never fire.
WINDOW_H = {"round_trip": 7 * 24.0, "layering_chain": 10 * 24.0}

#: Original shared term each motif's gap/count grades reference, plus the
#: private fitted term that replaces the reference (other motifs keep the
#: original).
TERM_COPY = {
    "round_trip": ("patient", "patient_fit_rt"),
    "layering_chain": ("patient", "patient_fit_lc"),
    "scatter_gather": ("several", "several_fit_sg"),
}

#: Target AMLworld pattern per fittable motif (prep_amlworld TYPE_TO_MOTIF).
TARGET_PATTERN = {
    "round_trip": "CYCLE",
    "layering_chain": "RANDOM",
    "scatter_gather": "SCATTER-GATHER",
}

SCORES = ("min", "product")
EMIT_LO, EMIT_HI = 0.3, 0.7


def default_params(motif: str) -> dict[str, float | str]:
    """Hand-tuned reference vector for `motif` (a fresh copy)."""
    try:
        return dict(DEFAULTS[motif])
    except KeyError:
        raise ValueError(f"no calibration space for motif `{motif}`: "
                         f"choose from {sorted(DEFAULTS)}") from None


def validate_params(motif: str, params: dict) -> None:
    """Raise ValueError unless `params` is a legal vector for `motif`."""
    if motif not in DEFAULTS:
        raise ValueError(f"no calibration space for motif `{motif}`: "
                         f"choose from {sorted(DEFAULTS)}")
    want = set(DEFAULTS[motif])
    got = set(params)
    if got != want:
        raise ValueError(f"`{motif}` wants params {sorted(want)}, got {sorted(got)}")
    if params["score"] not in SCORES:
        raise ValueError(f"score {params['score']!r}: use min or product")
    emit = params["emit"]
    if not isinstance(emit, (int, float)) or not EMIT_LO <= emit <= EMIT_HI:
        raise ValueError(f"emit {emit!r}: outside [{EMIT_LO}, {EMIT_HI}]")
    if motif in ("round_trip", "layering_chain"):
        plat, supp, tol = (params["gap_plateau_h"], params["gap_support_h"],
                           params["tol_pct"])
        for key, val in (("gap_plateau_h", plat), ("gap_support_h", supp),
                         ("tol_pct", tol)):
            if not isinstance(val, (int, float)) or not np.isfinite(val):
                raise ValueError(f"{key} {val!r}: not a finite number")
        if not 6.0 <= plat <= 72.0:
            raise ValueError(f"gap_plateau_h {plat}: outside [6, 72]")
        cap = WINDOW_H[motif]
        if not 48.0 <= supp <= cap:
            raise ValueError(f"gap_support_h {supp}: outside [48, {cap:g}] "
                             f"(the hard where window)")
        if plat > supp:
            raise ValueError(f"gap_plateau_h {plat} above gap_support_h {supp}")
        if not 5.0 <= tol <= 50.0:
            raise ValueError(f"tol_pct {tol}: outside [5, 50]")
    else:  # scatter_gather
        lo, hi, tol = params["several_lo"], params["several_hi"], params["sum_tol_pct"]
        for key, val in (("several_lo", lo), ("several_hi", hi),
                         ("sum_tol_pct", tol)):
            if not isinstance(val, (int, float)) or not np.isfinite(val):
                raise ValueError(f"{key} {val!r}: not a finite number")
        if not 1.0 <= lo <= 3.0:
            raise ValueError(f"several_lo {lo}: outside [1, 3]")
        if not 2.0 <= hi <= 8.0:
            raise ValueError(f"several_hi {hi}: outside [2, 8]")
        if not lo < hi:
            raise ValueError(f"several_lo {lo} not below several_hi {hi}")
        if not 5.0 <= tol <= 50.0:
            raise ValueError(f"sum_tol_pct {tol}: outside [5, 50]")


def sample_params(motif: str, rng: np.random.Generator) -> dict[str, float | str]:
    """One uniform random vector from `motif`'s space (seeded by the caller).

    Gap shoulders stay plausible by construction: the plateau is drawn
    first, then the support above it and inside the hard `where` window.
    """
    if motif not in DEFAULTS:
        raise ValueError(f"no calibration space for motif `{motif}`: "
                         f"choose from {sorted(DEFAULTS)}")
    out: dict[str, float | str] = {
        "score": str(rng.choice(list(SCORES))),
        "emit": float(rng.uniform(EMIT_LO, EMIT_HI)),
    }
    if motif in ("round_trip", "layering_chain"):
        plat = float(rng.uniform(6.0, 72.0))
        supp = float(rng.uniform(max(plat, 48.0), WINDOW_H[motif]))
        out.update(gap_plateau_h=plat, gap_support_h=supp,
                   tol_pct=float(rng.uniform(5.0, 50.0)))
    else:
        lo = float(rng.uniform(1.0, 3.0))
        hi = float(rng.uniform(max(lo + 0.5, 2.0), 8.0))
        out.update(several_lo=lo, several_hi=hi,
                   sum_tol_pct=float(rng.uniform(5.0, 50.0)))
    return out


def width_score(motif: str, params: dict) -> float:
    """Looseness of a vector in [0, ~2]: the regularization tie-break.

    Tiny train sets overfit toward narrow supports that happen to dodge
    the sampled background, so exact objective ties break toward wider
    (looser) fits. Each component is normalized to ~[0, 1] within its
    bound; higher is looser.
    """
    if motif in ("round_trip", "layering_chain"):
        return (params["gap_support_h"] / WINDOW_H[motif]
                + params["tol_pct"] / 50.0)
    return params["several_hi"] / 8.0 + params["sum_tol_pct"] / 50.0


def _rewrite(node: object, orig_term: str, fitted_term: str,
             tol_pct: float) -> tuple[object, int, int]:
    """Rebuild `node` with the fitted term name and tolerance.

    Returns (new node, gap-term rewrites, tolerance rewrites). Only the
    two fittable shapes move: `is <orig_term>` references and `~=` percent
    tolerances. Anything else (comparisons, other terms, absolute
    tolerances) passes through untouched.
    """
    if isinstance(node, IsTerm):
        if node.term == orig_term:
            return dataclasses.replace(node, term=fitted_term), 1, 0
        return node, 0, 0
    if isinstance(node, Approx):
        if isinstance(node.tol, Pct):
            return dataclasses.replace(node, tol=Pct(tol_pct / 100.0)), 0, 1
        return node, 0, 0
    if isinstance(node, And):
        parts: list[tuple[object, int, int]] = [
            _rewrite(i, orig_term, fitted_term, tol_pct) for i in node.items]
        return (And(tuple(p[0] for p in parts)),
                sum(p[1] for p in parts), sum(p[2] for p in parts))
    if isinstance(node, Or):
        parts = [_rewrite(i, orig_term, fitted_term, tol_pct) for i in node.items]
        return (Or(tuple(p[0] for p in parts)),
                sum(p[1] for p in parts), sum(p[2] for p in parts))
    if isinstance(node, Not):
        body, g, t = _rewrite(node.body, orig_term, fitted_term, tol_pct)
        return dataclasses.replace(node, body=body), g, t
    if isinstance(node, Quantified):
        body, g, t = _rewrite(node.body, orig_term, fitted_term, tol_pct)
        return dataclasses.replace(node, body=body), g, t
    if isinstance(node, Comparison):
        return node, 0, 0
    return node, 0, 0


def apply_params(doc: Document, motif_name: str, params: dict) -> Document:
    """Copy of `doc` with `motif_name`'s fittable numbers set to `params`.

    Adds the private fitted term (replacing any earlier fit of the same
    motif) and repoints the motif's soft conditions at it; every other
    motif keeps its original terms and text. Raises ValueError when the
    motif has no fittable `~=` tolerance or term reference left — that
    means the motif structure drifted from the space definition.
    """
    validate_params(motif_name, params)
    motif = next((m for m in doc.motifs if m.name == motif_name), None)
    if motif is None:
        raise ValueError(f"no motif `{motif_name}` in document")
    if motif.soft is None:
        raise ValueError(f"motif `{motif_name}` has no soft clause to fit")
    orig_term, fitted_term = TERM_COPY[motif_name]
    if motif_name in ("round_trip", "layering_chain"):
        new_term = TermDef(
            fitted_term, (), "trapezoid",
            (Dur(0.0), Dur(0.0),
             Dur(float(params["gap_plateau_h"]) * 3600.0),
             Dur(float(params["gap_support_h"]) * 3600.0)))
        tol_pct = float(params["tol_pct"])
    else:
        new_term = TermDef(
            fitted_term, (), "ramp",
            (Num(float(params["several_lo"])), Num(float(params["several_hi"]))))
        tol_pct = float(params["sum_tol_pct"])
    new_soft, n_term, n_tol = _rewrite(motif.soft, orig_term, fitted_term, tol_pct)
    if n_term == 0:
        raise ValueError(f"motif `{motif_name}` no longer references `{orig_term}`: "
                         "structure drifted from the calibration space")
    if n_tol == 0:
        raise ValueError(f"motif `{motif_name}` has no fittable `~=` percent tolerance: "
                         "structure drifted from the calibration space")
    terms = tuple(t for t in doc.terms if t.name != fitted_term) + (new_term,)
    new_motif = dataclasses.replace(
        motif, soft=new_soft, score=str(params["score"]),
        emit=float(params["emit"]))
    motifs = tuple(new_motif if m.name == motif_name else m for m in doc.motifs)
    return Document(terms, doc.quantifiers, motifs)


def _fmt_h(hours: float) -> str:
    return f"{hours:g}h"


def _fmt_pct(pct: float) -> str:
    return f"{pct:g}%"


def render_fitted_file(motif_name: str, params: dict, meta: str = "") -> str:
    """Standalone `.motif` text with the fitted numbers (committable).

    Parses and compiles on its own: the private fitted term plus the
    motif block, whose match/where/having clauses mirror
    `examples/amlworld.motif` verbatim (pinned by
    `test_rendered_default_matches_reference`). Only the fittable
    numbers differ.
    """
    validate_params(motif_name, params)
    header = "// Fitted %s — issue #24 learned fuzzy calibration%s\n" % (
        motif_name, f" ({meta})" if meta else "")
    if motif_name == "round_trip":
        return (
            header
            + f"term patient_fit_rt = trapezoid(0h, 0h, "
            + _fmt_h(float(params["gap_plateau_h"])) + ", "
            + _fmt_h(float(params["gap_support_h"])) + ")\n"
            + "\n"
            + "motif round_trip {\n"
            + "  match (a:account) -[e:transfer*2..5]-> (a)\n"
            + "  where e is time_ordered\n"
            + "    and span(e) <= 7d\n"
            + "  soft  each e: e.gap is patient_fit_rt\n"
            + "    and each e: e.amount ~= prev.amount within "
            + _fmt_pct(float(params["tol_pct"])) + "\n"
            + f"  score {params['score']}\n"
            + f"  emit when score >= {params['emit']:g}\n"
            + "}\n"
        )
    if motif_name == "layering_chain":
        return (
            header
            + f"term patient_fit_lc = trapezoid(0h, 0h, "
            + _fmt_h(float(params["gap_plateau_h"])) + ", "
            + _fmt_h(float(params["gap_support_h"])) + ")\n"
            + "\n"
            + "motif layering_chain {\n"
            + "  match (a:account) -[e:transfer*5..8]-> (z:account)\n"
            + "  where e is time_ordered\n"
            + "    and span(e) <= 10d\n"
            + "    and a != z\n"
            + "  soft  each e: e.gap is patient_fit_lc\n"
            + "    and each e: e.amount ~= prev.amount within "
            + _fmt_pct(float(params["tol_pct"])) + "\n"
            + f"  score {params['score']}\n"
            + f"  emit when score >= {params['emit']:g}\n"
            + "}\n"
        )
    if motif_name == "scatter_gather":
        return (
            header
            + f"term several_fit_sg = ramp({params['several_lo']:g}, "
            + f"{params['several_hi']:g})\n"
            + "\n"
            + "motif scatter_gather {\n"
            + "  match (s:account) -[leg1:transfer]-> (m:account) -[leg2:transfer]-> (d:account)\n"
            + "  group by s, d over m window 7d\n"
            + "  where leg2.time > leg1.time\n"
            + "    and s != d\n"
            + "  having count(distinct m) >= 2\n"
            + "  soft  count(distinct m) is several_fit_sg\n"
            + "    and sum(leg2.amount) ~= sum(leg1.amount) within "
            + _fmt_pct(float(params["sum_tol_pct"])) + "\n"
            + f"  score {params['score']}\n"
            + f"  emit when score >= {params['emit']:g}\n"
            + "}\n"
        )
    raise ValueError(f"no calibration space for motif `{motif_name}`")
