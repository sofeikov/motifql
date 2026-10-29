"""Motif discovery agent tools (Nov 1, DESIGN.md section 7.2).

The agent never sees holdout: it gets train miss clusters plus three
tools over a `DiscoveryCtx` (graph + train member anchors + fixed
background sample). The model talks a text protocol (SHOW / TRY /
PROPOSE / DONE) because the Spark client is plain chat completions;
`run_agent` executes the loop with a bounded step budget and a
train-side gate. Gate-passing proposals are scored on holdout by the
harness (`score_holdout`), never by the agent.

Gate (constants below, tuned from measured background rates before
the runs): a proposal must compile, catch missed-cluster instances
the current motifs miss, and hold background fire at or below the
current-motif level.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from motifql.models import repair

log = logging.getLogger("discovery")

# Gate: catch at least this many missed-cluster instances with
# background fire at or below the median current motif's rate.
GATE_MIN_CATCHES = 2
GATE_BG_MULT = 1.0
MAX_STEPS = 12


@dataclass
class DiscoveryCtx:
    """Everything the agent may touch: train data only."""

    graph: object
    transfer_ids: np.ndarray  # graph position -> transfer_id
    pos_of: dict  # transfer_id -> graph position
    member_pos: dict[int, list[int]]  # instance_id -> member graph positions
    member_tid: dict[int, list[int]]  # instance_id -> member transfer_ids
    bg_pos: list[int]  # fixed background sample (graph positions)
    base_scores: dict = field(default_factory=dict)  # motif -> {pos: top}
    base_bg_fire: dict = field(default_factory=dict)  # motif -> fire rate
    base_emit: dict = field(default_factory=dict)  # motif -> emit threshold


def list_missed(ctx: DiscoveryCtx, motifs: list[str]) -> list[int]:
    """Train instance ids no motif emits on.

    An instance is missed iff no member anchor reaches any motif's
    emit threshold. Current-motif scores come from the ctx build; the
    caller scopes member_pos to one pattern before calling.
    """
    return [iid for iid, positions in ctx.member_pos.items()
            if not any(ctx.base_scores[m].get(p, 0.0) >= ctx.base_emit[m]
                       for m in motifs for p in positions)]


def show_instance(ctx: DiscoveryCtx, df, iid: int, motifs: list[str],
                  max_rows: int = 40) -> str:
    """Member transfer table plus current motifs' best scores per member."""
    tids = ctx.member_tid[iid]
    rows = (df.filter(pl.col("transfer_id").is_in(tids))
            .sort("timestamp").head(max_rows))
    lines = [f"instance {iid}: {len(tids)} members"]
    for r in rows.iter_rows(named=True):
        pos = ctx.pos_of[int(r["transfer_id"])]
        scores = ",".join(f"{m}={ctx.base_scores[m].get(pos, 0.0):.2f}" for m in motifs)
        lines.append(f"  {r['timestamp']} {r['from_account']}->{r['to_account']} "
                     f"${r['amount_usd']:,.0f} [{scores}]")
    if len(tids) > max_rows:
        lines.append(f"  ... +{len(tids) - max_rows} more")
    return "\n".join(lines)


def try_motif(ctx: DiscoveryCtx, text: str, batch_score, cluster: list[int]) -> str:
    """Compile + train replay: recall on the cluster, bg fire, FP sample.

    `batch_score(positions, plan, libs)` returns {pos: top_score};
    batching lets the driver score thousands of anchors per call.
    """
    compiled, errors = repair.try_compile(text)
    if compiled is None:
        return "COMPILE FAILED:\n" + "\n".join(f"- {e}" for e in errors)
    plan = compiled.plan
    emit = plan.emit
    want = sorted({p for iid in cluster for p in ctx.member_pos[iid]} | set(ctx.bg_pos))
    tops = batch_score(want, plan, compiled.libs)
    caught = [iid for iid in cluster
              if max(tops[p] for p in ctx.member_pos[iid]) >= emit]
    means = [max(tops[p] for p in ctx.member_pos[iid]) for iid in cluster]
    bg_hits = sorted(p for p in ctx.bg_pos if tops[p] >= emit)
    out = [f"COMPILED. emit={emit:.2f} cluster recall {len(caught)}/{len(cluster)} "
           f"(mean top {np.mean(means):.3f})",
           f"background fire {len(bg_hits)}/{len(ctx.bg_pos)} "
           f"({100 * len(bg_hits) / max(1, len(ctx.bg_pos)):.3f}%)"]
    for p in bg_hits[:3]:
        out.append(f"  FP tid={ctx.transfer_ids[p]} score={tops[p]:.2f}")
    return "\n".join(out)


@dataclass
class GateResult:
    """Outcome of the train-side gate for one proposal."""

    passed: bool
    detail: str
    catches: list[int] = field(default_factory=list)
    bg_fire: float = 0.0


def gate(ctx: DiscoveryCtx, text: str, batch_score, cluster: list[int],
         motifs: list[str]) -> GateResult:
    """Train-side gate: compile + catch misses + hold background fire."""
    compiled, errors = repair.try_compile(text)
    if compiled is None:
        return GateResult(False, "does not compile: " + "; ".join(errors))
    plan = compiled.plan
    want = sorted({p for iid in cluster for p in ctx.member_pos[iid]} | set(ctx.bg_pos))
    tops = batch_score(want, plan, compiled.libs)
    catches = [iid for iid in cluster
               if max(tops[p] for p in ctx.member_pos[iid]) >= plan.emit]
    bg_fire = sum(tops[p] >= plan.emit for p in ctx.bg_pos) / max(1, len(ctx.bg_pos))
    base_med = float(np.median([ctx.base_bg_fire[m] for m in motifs])) if motifs else 0.0
    need = min(GATE_MIN_CATCHES, len(cluster))
    if len(catches) < need:
        return GateResult(False, f"catches {len(catches)}/{len(cluster)} "
                                 f"(need {need})", catches, bg_fire)
    # Floor of 2 background fires: with a 20k sample, 0 vs 2 is noise,
    # so a zero base median must not veto every proposal.
    allowed = max(base_med * GATE_BG_MULT, 2 / max(1, len(ctx.bg_pos)))
    if bg_fire > allowed + 1e-9:
        return GateResult(False, f"bg fire {bg_fire:.4f} over allowed "
                                 f"{allowed:.4f} (base median {base_med:.4f})",
                          catches, bg_fire)
    return GateResult(True, f"catches {len(catches)}/{len(cluster)} "
                            f"bg fire {bg_fire:.4f} <= {allowed:.4f}",
                      catches, bg_fire)


_ACTION = re.compile(r"\b(SHOW|TRY|PROPOSE|DONE)\b")
_FENCE = re.compile(r"```motifql\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_SHOW_ID = re.compile(r"SHOW\s+(\d+)")


def parse_action(text: str):
    """One action per model turn: (kind, motif_text_or_None, instance_id_or_None).

    Keywords are UPPERCASE (the prompt shows them that way; every live
    reply complies), so lowercase prose can never trigger an action
    ("my last try", "once done" parse as nothing). Matches inside
    ```motifql fences are ignored. A fenced motif is unambiguous
    intent, so with a fence present the last TRY/PROPOSE wins over
    trailing prose keywords ("RATIONALE: ALL DONE"); otherwise the
    LAST keyword wins. The SHOW id / motif block must come AFTER
    the winning keyword.
    """
    spans = [m.span() for m in _FENCE.finditer(text)]
    cands = [m for m in _ACTION.finditer(text)
             if not any(a < m.start() < b for a, b in spans)]
    if not cands:
        return None, None, None
    if spans:
        fenced = [m for m in cands if m.group(1) in ("TRY", "PROPOSE")]
        if fenced:
            cands = fenced
    win = cands[-1]
    kind = win.group(1)
    if kind == "SHOW":
        mm = _SHOW_ID.search(text, win.start())
        return kind, None, int(mm.group(1)) if mm else None
    if kind in ("TRY", "PROPOSE"):
        for fm in _FENCE.finditer(text):
            if fm.start() > win.start():
                return kind, fm.group(1).strip(), None
        return kind, None, None
    return kind, None, None


def agent_system_prompt(cluster_desc: str, gate_desc: str = "") -> str:
    return (repair.system_prompt()
            + "\n\nDISCOVERY TASK. " + cluster_desc
            + ("\n\nGATE. " + gate_desc if gate_desc else "")
            + "\n\nTOOLS. End every reply with exactly one action line:\n"
            "SHOW <instance_id> — member transfers + current scores.\n"
            "TRY then a ```motifql block (motif `cand`) — compile + train replay.\n"
            "PROPOSE then a ```motifql block + RATIONALE: — final proposal.\n"
            "DONE — stop (use when nothing beats the gate).\n"
            "Put the keyword (SHOW/TRY/PROPOSE/DONE, in capitals) AFTER any thinking text.\n"
            "Rules: train data only (you never see holdout); name the motif "
            "`cand`; every TRY/PROPOSE is scored and the numbers come back.")


def run_agent(cluster_desc: str, cluster: list[int], ctx: DiscoveryCtx,
              motifs: list[str], df, batch_score, complete,
              max_steps: int = MAX_STEPS, gate_desc: str = "",
              transcript_out: list | None = None) -> dict:
    """Text-protocol agent loop; returns transcript + best proposal.

    Steps append to `transcript_out` live when given, so the driver
    can persist a partial transcript if the run dies mid-loop.
    """
    messages = [{"role": "system", "content": agent_system_prompt(cluster_desc, gate_desc)},
                {"role": "user", "content": f"Missed cluster ({len(cluster)} instances): "
                                            f"{cluster}. Start with SHOW."}]
    transcript: list[str] = transcript_out if transcript_out is not None else []
    best = {"text": None, "gate": None}
    for step in range(max_steps):
        text, _ = complete(messages)
        transcript.append(f"--- step {step} model ---\n{text}")
        kind, motif_text, iid = parse_action(text)
        try:
            result, g = _dispatch(kind, motif_text, iid, ctx, motifs, df,
                                  batch_score, cluster)
        except Exception as exc:
            # Fail closed: a tool crash becomes feedback, never an
            # escaped exception (and never a silent holdout read).
            result, g = f"tool error: {type(exc).__name__}: {exc}"[:500], None
        if kind == "PROPOSE" and g is not None and g.passed and (
                best["text"] is None or len(g.catches) > len(best["gate"].catches)):
            best = {"text": motif_text, "gate": g}
        if step == max_steps - 2 and kind not in ("PROPOSE", "DONE"):
            result += "\nNEXT IS YOUR LAST STEP: reply PROPOSE or DONE."
        transcript.append(f"--- step {step} tool ---\n{result}")
        messages = messages + [{"role": "assistant", "content": text},
                               {"role": "user", "content": result}]
        if kind == "DONE" or (kind == "PROPOSE" and g is not None and g.passed):
            break
    return {"transcript": transcript, "proposal": best["text"], "gate": best["gate"]}


def _dispatch(kind, motif_text, iid, ctx, motifs, df, batch_score, cluster):
    """Execute one parsed action; returns (result_text, gate_or_None)."""
    if kind == "SHOW" and iid in ctx.member_tid:
        return show_instance(ctx, df, iid, motifs), None
    if kind == "SHOW":
        return f"unknown instance {iid}; cluster is {cluster}.", None
    if kind == "TRY" and motif_text:
        return try_motif(ctx, motif_text, batch_score, cluster), None
    if kind in ("TRY", "PROPOSE") and not motif_text:
        return "no ```motifql block found; send the motif in one.", None
    if kind == "PROPOSE":
        g = gate(ctx, motif_text or "", batch_score, cluster, motifs)
        return f"GATE {'PASS' if g.passed else 'FAIL'}: {g.detail}", g
    if kind == "DONE":
        return "noted.", None
    return "no action parsed; reply SHOW / TRY / PROPOSE / DONE.", None
