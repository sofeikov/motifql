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
    batching lets the driver screen (crisp first, fuzzy refine).
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


_ACTION = re.compile(r"\b(SHOW|TRY|PROPOSE|DONE)\b", re.IGNORECASE)
_FENCE = re.compile(r"```motifql\s*(.*?)```", re.IGNORECASE | re.DOTALL)
_SHOW_ID = re.compile(r"SHOW\s+(\d+)", re.IGNORECASE)


def parse_action(text: str):
    """One action per model turn: (kind, motif_text_or_None, instance_id_or_None).

    The keyword may be glued to prose ("...them.SHOW 123"), so the
    LAST keyword anywhere wins — the model thinks first, acts last.
    A ```motifql fence means TRY/PROPOSE even if prose after it
    mentions done/show ("RATIONALE: this is done").
    """
    ms = list(_ACTION.finditer(text))
    if not ms:
        return None, None, None
    if _FENCE.search(text):
        fenced = [m for m in ms if m.group(1).upper() in ("TRY", "PROPOSE")]
        if fenced:
            kind = fenced[-1].group(1).upper()
            mm = _FENCE.search(text)
            return kind, mm.group(1).strip() if mm else None, None
    kind = ms[-1].group(1).upper()
    if kind == "SHOW":
        mm = _SHOW_ID.search(text)
        return kind, None, int(mm.group(1)) if mm else None
    if kind in ("TRY", "PROPOSE"):
        mm = _FENCE.search(text)
        return kind, mm.group(1).strip() if mm else None, None
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
            "Put the keyword (SHOW/TRY/PROPOSE/DONE) AFTER any thinking text.\n"
            "Rules: train data only (you never see holdout); name the motif "
            "`cand`; every TRY/PROPOSE is scored and the numbers come back.")


def run_agent(cluster_desc: str, cluster: list[int], ctx: DiscoveryCtx,
              motifs: list[str], df, batch_score, complete,
              max_steps: int = MAX_STEPS, gate_desc: str = "") -> dict:
    """Text-protocol agent loop; returns transcript + best proposal."""
    messages = [{"role": "system", "content": agent_system_prompt(cluster_desc, gate_desc)},
                {"role": "user", "content": f"Missed cluster ({len(cluster)} instances): "
                                            f"{cluster}. Start with SHOW."}]
    transcript: list[str] = []
    best = {"text": None, "gate": None}
    for step in range(max_steps):
        text, _ = complete(messages)
        transcript.append(f"--- step {step} model ---\n{text}")
        kind, motif_text, iid = parse_action(text)
        if kind == "SHOW" and iid in ctx.member_tid:
            result = show_instance(ctx, df, iid, motifs)
        elif kind == "SHOW":
            result = f"unknown instance {iid}; cluster is {cluster}."
        elif kind == "TRY" and motif_text:
            result = try_motif(ctx, motif_text, batch_score, cluster)
        elif kind in ("TRY", "PROPOSE") and not motif_text:
            result = "no ```motifql block found; send the motif in one."
        elif kind == "PROPOSE":
            g = gate(ctx, motif_text or "", batch_score, cluster, motifs)
            result = f"GATE {'PASS' if g.passed else 'FAIL'}: {g.detail}"
            if g.passed and (best["text"] is None
                             or len(g.catches) > len(best["gate"].catches)):
                best = {"text": motif_text, "gate": g}
            transcript.append(f"--- step {step} tool ---\n{result}")
            messages = messages + [{"role": "assistant", "content": text},
                                   {"role": "user", "content": result}]
            if g.passed:
                break
            continue
        elif kind == "DONE":
            transcript.append(f"--- step {step} tool ---\nnoted.")
            break
        else:
            result = "no action parsed; reply SHOW / TRY / PROPOSE / DONE."
        transcript.append(f"--- step {step} tool ---\n{result}")
        messages = messages + [{"role": "assistant", "content": text},
                               {"role": "user", "content": result}]
    return {"transcript": transcript, "proposal": best["text"], "gate": best["gate"]}
