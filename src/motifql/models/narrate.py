"""Witness narratives (Nov 8, issue #23, DESIGN.md §7).

One-paragraph investigator stories per witness: what moved, through
which accounts, over how long, how close each condition held, and the
weakest step. Each story leads with a one-line verdict — what matched
and WHY it qualifies, in plain words, no account numbers. Muse Spark
writes both from an extractive witness table; every call goes through
the Spark client, so prompts and replies are cached on disk by prompt
hash exactly like compiler calls.

Fail closed: missing witness data, a model error, a `CANNOT NARRATE`
reply, or text that fails verification (an amount or date not in the
witness) yields NO text — never an invented one.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from motifql.models.spark import complete as spark_complete

log = logging.getLogger("narrate")

SYSTEM = """\
You write investigator narratives for money-flow witnesses: 3-5 plain-text \
sentences, no markdown, no bullets.

A PATH witness is one route: narrate it in transfer order — what moved \
(amounts), through which accounts, over how long, how close each listed \
condition held (quote its degree), and the weakest step as an explicit \
sentence ("The weakest step is hop N, ...").

A GROUP witness is a set of transfers around shared accounts, not a \
route: describe the shape — the shared account(s), the members and their \
amounts, the time window — then the conditions with quoted degrees, then \
the weakest condition as an explicit sentence (naming the hop number \
when the table gives one). Take every count from the DISTINCT line; \
never count members by hand (transfers repeat accounts).

EXTRACTIVE RULES. The witness table below is your only source of facts. \
Use ONLY figures from it: write amounts EXACTLY as shown ($1,000), dates \
EXACTLY as shown (2022-09-03), accounts EXACTLY as shown (001/A). Never \
invent, round differently, or carry a figure from one transfer to \
another. Never add transfers, accounts, or dates not in the table.

If any fact you need is missing from the table, reply with exactly: \
CANNOT NARRATE"""

VERDICT_SYSTEM = """\
You write one-line verdicts for money-flow witnesses: 1-2 plain-text \
sentences, no markdown, no bullets. Say WHAT matched and WHY it \
qualifies, in the analyst's own words (see the ANALYST line): the \
shape (a round trip, a layering chain, a fan-out, a scatter-gather, \
a gather), the key facts (how many hops or intermediaries, whether \
amounts or sums roughly match, the overall score), and the weakest \
condition in plain words with its quoted degree.

FORBIDDEN: account numbers, transfer ids, individual amounts, \
timestamps, and predicate code names. Never write things like \
count(...), DistinctArg, leg1, or gap is patient — translate every \
condition into words ("the balance of senders and recipients", "the \
hop delays", "the amount similarity"). Describe sums qualitatively \
("roughly equal"); never state totals (they are not in the table). \
You may quote the overall score and condition degrees (0.998) as shown.

EXTRACTIVE RULES. The witness table below is your only source of facts. \
Take every count from the DISTINCT line; never count members by hand \
(transfers repeat accounts). Any figure you use must appear in the \
table exactly as shown. Never invent or carry a figure from one \
transfer to another.

If any fact you need is missing from the table, reply with exactly: \
CANNOT NARRATE"""

DIAGNOSIS_SYSTEM = """\
You write unified diagnoses for composite money-flow findings: 2-3 \
plain-text sentences, no markdown, no bullets. Two shapes fired on \
the same anchor — say what the COMBINED structure is and why it \
matters, in the analysts' own words (see the ANALYST lines): which \
shapes fired with their scores, how they link (the SHARED transfers \
belong to both shapes — say what they connect, e.g. "the chain's \
tail feeds the gather"), and the weakest link across both shapes in \
plain words with its quoted degree. Read as ONE criminal structure, \
not two reports.

FORBIDDEN: account numbers, transfer ids, individual amounts, \
timestamps, and predicate code names. Never write things like \
count(...), DistinctArg, leg1, or gap is patient — translate every \
condition into words. Describe sums qualitatively ("roughly equal"); \
never state totals (they are not in the tables). You may quote the \
overall scores and condition degrees (0.998) as shown.

EXTRACTIVE RULES. The witness tables below are your only source of \
facts. Take every count from the DISTINCT lines; never count members \
by hand (transfers repeat accounts). Any figure you use must appear \
in the tables exactly as shown. Never invent or carry a figure from \
one transfer to another.

If any fact you need is missing from the tables, reply with exactly: \
CANNOT NARRATE"""

HOP_KEYS = ("transfer_id", "timestamp", "from_bank", "from_account",
            "to_bank", "to_account", "amount")

_AMOUNT_RE = re.compile(r"\$(\d[\d,]*(\.\d+)?)")
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


@dataclass
class NarrateResult:
    """Outcome of one narration; `text` is None when failed closed."""

    text: str | None
    weakest_step: int | None = None  # 1-based hop, from the degrees
    weakest_predicate: str | None = None
    weakest_degree: float | None = None
    issues: list[str] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


def validate_witness(witness: dict) -> list[str]:
    """Missing-data check; non-empty means narrate() must fail closed."""
    if not isinstance(witness, dict):
        return ["witness is not a dict"]
    hops = witness.get("hops")
    if not isinstance(hops, list) or not hops:
        return ["witness has no hops"]
    issues = []
    for i, hop in enumerate(hops):
        if not isinstance(hop, dict):
            issues.append(f"hop {i + 1} is not a dict")
            continue
        for key in HOP_KEYS:
            if hop.get(key) is None:
                issues.append(f"hop {i + 1} misses {key}")
        for key in ("timestamp", "amount", "transfer_id"):
            value = hop.get(key)
            if value is not None and isinstance(value, bool):
                issues.append(f"hop {i + 1} has a boolean {key}")
            elif value is not None and not isinstance(value, (int, float)):
                issues.append(f"hop {i + 1} has a non-numeric {key}")
    return issues


def weakest(explanation: list[dict]) -> tuple[str, float, int | None] | None:
    """Lowest-degree predicate (ties to the first); None when empty."""
    if not explanation:
        return None
    row = min(explanation, key=lambda e: float(e.get("degree", 1.0)))
    step = row.get("step")
    return (str(row.get("predicate", "")),
            float(row.get("degree", 1.0)),
            int(step) if isinstance(step, (int, float)) else None)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _fmt_money(amount: float) -> str:
    if float(amount).is_integer():
        return f"${int(amount):,}"
    return f"${float(amount):,.2f}"


def _fmt_dur(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 90:
        return f"{seconds:,.0f}s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.1f}m"
    if seconds < 48 * 3600:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def _fmt_ts(epoch: int | float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%MZ")


def is_chain(hops: list[dict]) -> bool:
    """True when every transfer continues the previous account."""
    for prev, hop in zip(hops, hops[1:]):
        if ((hop["from_bank"], hop["from_account"])
                != (prev["to_bank"], prev["to_account"])):
            return False
    return True


def witness_table(witness: dict, english: str, motif_text: str) -> str:
    """Deterministic extractive facts the model must write from."""
    hops = witness["hops"]
    explanation = witness.get("explanation", [])
    top = witness.get("top_score")
    found = "FIRES" if witness.get("found") else "below emit"
    chain = is_chain(hops)
    unit = "HOPS" if chain else "MEMBERS"
    lines = [f"ANALYST: {english.strip()}",
             "",
             f"MOTIF (score {top:.3f}, {found}):" if top is not None
             else f"MOTIF ({found}):",
             motif_text.strip(),
             "",
             f"SHAPE: {'path (one route in order)' if chain else 'group (a set of transfers, not a route)'}",
             f"{unit} ({len(hops)}):"]
    if not chain:
        senders = {(h["from_bank"], h["from_account"]) for h in hops}
        recipients = {(h["to_bank"], h["to_account"]) for h in hops}
        lines.append(f"DISTINCT: {_plural(len(senders), 'sender')}, "
                     f"{_plural(len(recipients), 'recipient')} "
                     f"across {len(hops)} transfers")
    for i, hop in enumerate(hops):
        acct = (f"{hop['from_bank']}/{hop['from_account']} -> "
                f"{hop['to_bank']}/{hop['to_account']}")
        lines.append(f"{i + 1}. {_fmt_ts(hop['timestamp'])} {acct} "
                     f"{_fmt_money(hop['amount'])} "
                     f"(transfer {hop['transfer_id']})")
        if chain and i:
            gap = hop["timestamp"] - hops[i - 1]["timestamp"]
            lines.append(f"   gap after hop {i}: {_fmt_dur(gap)}")
    if len(hops) > 1:
        first = min(h["timestamp"] for h in hops)
        last = max(h["timestamp"] for h in hops)
        lines.append(f"SPAN first to last: {_fmt_dur(last - first)}")
    lines.append("CONDITIONS:")
    if explanation:
        for row in explanation:
            step = f", weakest at hop {row['step']}" if row.get("step") else ""
            lines.append(f"- {row.get('predicate')}: "
                         f"{float(row.get('degree', 0.0)):.3f}{step}")
    else:
        lines.append("- (no soft conditions)")
    weak = weakest(explanation)
    if weak is not None:
        predicate, degree, step = weak
        label = "WEAKEST STEP" if step else "WEAKEST CONDITION"
        at = f" at hop {step}" if step else ""
        lines.append(f"{label}: {predicate} = {degree:.3f}{at}")
    return "\n".join(lines)


def build_messages(english: str, motif_text: str, witness: dict) -> list[dict]:
    """System + witness-table prompts; raises ValueError when invalid."""
    issues = validate_witness(witness)
    if issues:
        raise ValueError(f"cannot narrate: {'; '.join(issues)}")
    if not english.strip():
        raise ValueError("cannot narrate: english is empty")
    if not motif_text.strip():
        raise ValueError("cannot narrate: motif text is empty")
    return [{"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": "Write the investigator narrative for this witness:\n\n"
                        + witness_table(witness, english, motif_text)}]


def _cents(value: float) -> int:
    return int(round(float(value) * 100))


def verify_composite(text: str, witnesses: list[dict]) -> list[str]:
    """Extractive check over the union of composite witness hops."""
    merged = {"hops": [h for w in witnesses for h in w.get("hops", [])]}
    return verify_narrative(text, merged)


def verify_narrative(text: str, witness: dict) -> list[str]:
    """Extractive check: every amount/date in `text` is in the witness.

    Amounts compare in integer cents (narrative `$1,000` vs witness
    1000.0); dates compare as ISO days. Anything else the model wrote
    is style, not fact, and is not checked.
    """
    issues = []
    if not text.strip():
        return ["narrative is empty"]
    hop_amounts = {_cents(h["amount"]) for h in witness.get("hops", [])}
    for raw, _ in _AMOUNT_RE.findall(text):
        if _cents(float(raw.replace(",", ""))) not in hop_amounts:
            issues.append(f"amount ${raw} is not in the witness")
    hop_dates = {_fmt_ts(h["timestamp"])[:10] for h in witness.get("hops", [])}
    for day in _DATE_RE.findall(text):
        if day not in hop_dates:
            issues.append(f"date {day} is not in the witness")
    return issues


def _precheck(english: str, motif_text: str,
              witness: dict) -> list[str]:
    """Shared fail-closed gate; non-empty means return None."""
    issues = validate_witness(witness)
    if not english.strip():
        issues = issues + ["english is empty"]
    if not motif_text.strip():
        issues = issues + ["motif text is empty"]
    return issues


def _complete_and_verify(system: str, user_content: str, witness: dict, *,
                         complete, strict: bool,
                         log_tag: str) -> NarrateResult:
    """One fail-closed model call over an extractive table."""
    try:
        text, usage = complete([{"role": "system", "content": system},
                                {"role": "user", "content": user_content}])
    except Exception as exc:
        log.warning("%s: model call failed: %r", log_tag, exc)
        return NarrateResult(
            None, issues=[f"model unavailable: {exc}"], usage={})
    text = (text or "").strip()
    if not text or "CANNOT NARRATE" in text:
        return NarrateResult(
            None, issues=["model declined (CANNOT NARRATE or empty)"],
            usage=dict(usage))
    problems = verify_narrative(text, witness)
    weak = weakest(witness.get("explanation", []))
    step = weak[2] if weak else None
    predicate = weak[0] if weak else None
    degree = weak[1] if weak else None
    if problems and strict:
        log.warning("%s: verification failed: %s", log_tag, problems)
        return NarrateResult(None, weakest_step=step,
                             weakest_predicate=predicate,
                             weakest_degree=degree,
                             issues=problems, usage=dict(usage))
    return NarrateResult(text, weakest_step=step, weakest_predicate=predicate,
                         weakest_degree=degree,
                         issues=problems, usage=dict(usage))


def narrate(english: str, motif_text: str, witness: dict, *,
            complete=spark_complete, strict: bool = True) -> NarrateResult:
    """Write one witness narrative, failing closed (None, never invented).

    `complete(messages)` returns (text, usage) and hides the transport
    (real client or test stub). With `strict` (default) a narrative
    that fails verification is dropped; with `strict=False` it is kept
    with its issues listed (spot-check input, never demo output).
    """
    issues = _precheck(english, motif_text, witness)
    if issues:
        return NarrateResult(None, issues=issues)
    try:
        messages = build_messages(english, motif_text, witness)
    except ValueError as exc:  # unreachable after validate, kept honest
        return NarrateResult(None, issues=[str(exc)])
    return _complete_and_verify(messages[0]["content"], messages[1]["content"],
                                witness, complete=complete, strict=strict,
                                log_tag="narrate")


def build_verdict_messages(english: str, motif_text: str,
                           witness: dict) -> list[dict]:
    """System + witness-table prompts for the one-line verdict."""
    issues = validate_witness(witness)
    if issues:
        raise ValueError(f"cannot narrate: {'; '.join(issues)}")
    if not english.strip():
        raise ValueError("cannot narrate: english is empty")
    if not motif_text.strip():
        raise ValueError("cannot narrate: motif text is empty")
    return [{"role": "system", "content": VERDICT_SYSTEM},
            {"role": "user",
             "content": "Write the one-line verdict for this witness:\n\n"
                        + witness_table(witness, english, motif_text)}]


def summarize(english: str, motif_text: str, witness: dict, *,
              complete=spark_complete, strict: bool = True) -> NarrateResult:
    """Write the one-line verdict (what matched, why it qualifies).

    Same fail-closed contract as narrate(): the verdict leads the
    investigator paragraph in the UI, and drops to nothing rather
    than inventing facts.
    """
    issues = _precheck(english, motif_text, witness)
    if issues:
        return NarrateResult(None, issues=issues)
    try:
        messages = build_verdict_messages(english, motif_text, witness)
    except ValueError as exc:  # unreachable after validate, kept honest
        return NarrateResult(None, issues=[str(exc)])
    return _complete_and_verify(messages[0]["content"], messages[1]["content"],
                                witness, complete=complete, strict=strict,
                                log_tag="verdict")


def build_diagnosis_messages(english_list: list[str], motif_list: list[str],
                             witness_list: list[dict]) -> list[dict]:
    """System + stacked witness-table prompts for a composite finding."""
    if not (len(english_list) == len(motif_list) == len(witness_list)):
        raise ValueError("cannot narrate: composite inputs misaligned")
    if len(witness_list) < 2:
        raise ValueError("cannot narrate: composite needs 2+ witnesses")
    for english, motif_text, witness in zip(english_list, motif_list,
                                            witness_list):
        issues = validate_witness(witness)
        if issues:
            raise ValueError(f"cannot narrate: {'; '.join(issues)}")
        if not english.strip():
            raise ValueError("cannot narrate: english is empty")
        if not motif_text.strip():
            raise ValueError("cannot narrate: motif text is empty")
    tids = [set(h["transfer_id"] for h in w["hops"]) for w in witness_list]
    shared = set.intersection(*tids) if tids else set()
    tables = []
    for i, (english, motif_text, witness) in enumerate(
            zip(english_list, motif_list, witness_list)):
        tables.append(f"SHAPE {i + 1}:\n"
                      + witness_table(witness, english, motif_text))
    tables.append(f"SHARED: {len(shared)} transfer(s) appear in "
                  f"{len(witness_list)} shapes")
    return [{"role": "system", "content": DIAGNOSIS_SYSTEM},
            {"role": "user",
             "content": "Write the unified diagnosis for this composite "
                        "finding:\n\n" + "\n\n".join(tables)}]


def summarize_composite(english_list: list[str], motif_list: list[str],
                        witness_list: list[dict], *,
                        complete=spark_complete,
                        strict: bool = True) -> NarrateResult:
    """Write one unified diagnosis over several shapes (fail-closed).

    Amounts and dates verify against the union of witness hops; the
    weakest link is the min-degree condition across shapes (its hop
    number is dropped — hops are per-shape).
    """
    try:
        messages = build_diagnosis_messages(english_list, motif_list,
                                            witness_list)
    except ValueError as exc:
        return NarrateResult(None, issues=[str(exc)])
    merged = {"hops": [h for w in witness_list for h in w.get("hops", [])]}
    res = _complete_and_verify(messages[0]["content"], messages[1]["content"],
                               merged, complete=complete, strict=strict,
                               log_tag="diagnosis")
    weaks = [weakest(w.get("explanation", [])) for w in witness_list]
    weaks = [w for w in weaks if w is not None]
    if weaks:
        predicate, degree, _ = min(weaks, key=lambda t: t[1])
        res.weakest_predicate = predicate
        res.weakest_degree = degree
    res.weakest_step = None
    return res
