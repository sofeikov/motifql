"""English → MotifQL compiler with repair loop (Oct 31, DESIGN.md §7.1).

Round 0 sends the English request plus the language pack (grammar, data
model, term/quantifier tables, reference examples). On any failure the
stable compiler messages go back to the model, up to N rounds. An
optional sanity callback (fire rate on a train sample) adds loosen /
tighten feedback. Engine crashes become feedback text, never harness
crashes — a model motif that kills the evaluator is just a bad motif.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from motifql.degrees import libs_from_doc
from motifql.parser import parse_text
from motifql.planner import PlanError, compile_motif

log = logging.getLogger("repair")

ROOT = Path(__file__).resolve().parent.parent.parent.parent
CANDIDATE_NAME = "cand"
MAX_FEEDBACK_CHARS = 2000

_SPEC = """\
You write MotifQL motifs: fuzzy, time-ordered money-flow patterns.

DATA MODEL. Directed multigraph of transfers. Node `account` fields: `id`,
`bank`, `first_seen`, `in_degree`, `out_degree` (account fields are NOT usable
in `soft` — grade edge fields or aggregates instead). Edge `transfer` fields:
`time`, `amount` (USD), `gap` (time since previous step on same path;
in scatter groups `leg2.gap` is the forward delay), `index`.

MOTIF SHAPE.
- Path (cycles, chains): `match (a:account) -[e:transfer*M..N]-> (b)`
  `where e is time_ordered and span(e) <= D [and a != b]`
  `soft <quantified per-step conditions> score min|product emit when score >= T`
  Per-step body over `e`: `e.gap is <term>`, `e.amount ~= prev.amount within P%`.
- Fan: `match (s) -[e:transfer]-> (r)` `group by s over r window D`
  `having count(distinct r) >= K` `soft <aggregates / most r: ...> ...`
- Scatter: `match (s) -[leg1]-> (m) -[leg2]-> (d)` `group by s, d over m`
  `window D` `where leg2.time > leg1.time and s != d`
  `having count(distinct m) >= K`.
- Gather: `match (x) -[g]-> (c), (c) -[p]-> (y)` `group by c over x, y`
  `window D` `where p.time > g.time` `having count(distinct x) >= K [and ...]`.
- Butterfly: four single-edge patterns forming the 2x2 core,
  `where x1 != x2 and y1 != y2 and span(all) <= D`,
  `soft span(all) is <term>`.

TERMS. `trapezoid(a,b,c,d)`: 0 below a, 1 at b..c, 0 past d.
`triangle(a,b,c)`, `ramp(a,b)` (0 at/below a, 1 at/above b), `ramp_down(a,b).
Define every term you use (`term soon = trapezoid(0h,0h,6h,48h)`), or use a
parameter (`term just_below(limit) = trapezoid(0.85*limit, ...)`).
Units follow the graded value: durations for gaps/spans, money for amounts,
plain numbers for counts. Time minus time is a duration.

QUANTIFIERS in soft: `each`/`all` (min), `some` (max), `at least N`,
`all but K` (forgives K outliers), `most`/`few` (mean degree through a
ramp), `@name` (user-defined). `A ~= B within T` is graded closeness;
comparisons are crisp 0/1. `and` combines with the motif t-norm.

RULES. Name the motif `cand`. Every motif needs a time bound (`span <= D`
or group `window D`). `where`/`having` stay crisp (no `~=`, terms, or
most/few). Comparisons need matching units. Return ONLY MotifQL text:
any `term`/`quantifier` lines the motif needs, then the motif block."""


def _examples() -> str:
    want = ("round_trip", "fan_out", "scatter_gather", "layering_chain")
    raw = (ROOT / "examples" / "amlworld.motif").read_text()
    blocks = []
    for name in want:
        match = re.search(rf"^motif {name} \{{.*?^}}$", raw,
                          re.MULTILINE | re.DOTALL)
        if match:
            blocks.append(match.group(0))
    terms = ("term soon = trapezoid(0h, 0h, 6h, 48h)\n"
             "term patient = trapezoid(0h, 0h, 24h, 120h)\n"
             "term several = ramp(1, 3)\nterm many = ramp(3, 8)")
    return terms + "\n\n" + "\n\n".join(blocks)


def system_prompt() -> str:
    grammar = (ROOT / "grammar" / "motifql.lark").read_text()
    return (_SPEC + "\n\nGRAMMAR (motifql.lark):\n" + grammar
            + "\n\nEXAMPLES:\n" + _examples())


def user_prompt(english: str) -> str:
    return f"Write a MotifQL motif for this pattern:\n\n{english.strip()}"


def repair_prompt(english: str, last_motif: str, errors: list[str]) -> str:
    feedback = "\n".join(f"- {e}" for e in errors)[:MAX_FEEDBACK_CHARS]
    return (f"Write a MotifQL motif for this pattern:\n\n{english.strip()}\n\n"
            f"Your last attempt failed to compile or score:\n\n{last_motif.strip()}\n\n"
            f"Fix these errors (do not explain, return only MotifQL):\n{feedback}")


@dataclass
class CompiledCandidate:
    """A candidate that parsed, checked, planned and library-resolved."""

    text: str
    plan: object
    libs: object


def _truncate(msg: str, limit: int = 500) -> str:
    one_line = " ".join(msg.split())
    return one_line if len(one_line) <= limit else one_line[:limit] + "…"


def try_compile(text: str) -> tuple[CompiledCandidate | None, list[str]]:
    """Parse, check and plan `text`; crashes become error strings."""
    try:
        doc = parse_text(text)
    except Exception as exc:  # Lark errors are verbose; keep the head.
        return None, [f"parse error: {_truncate(str(exc))}"]
    names = [m.name for m in doc.motifs]
    if CANDIDATE_NAME not in names:
        return None, [f"no `motif {CANDIDATE_NAME}` block found (got: {names or 'none'})."]
    try:
        plan = compile_motif(doc, CANDIDATE_NAME)
        libs = libs_from_doc(doc)
    except PlanError as exc:
        return None, [f"{exc.code}: {_truncate(f'{exc.problem} Fix: {exc.fix}')}"]
    except Exception as exc:
        return None, [f"compiler crashed: {_truncate(f'{type(exc).__name__}: {exc}')}"]
    return CompiledCandidate(text, plan, libs), []


@dataclass
class RepairResult:
    """Outcome of one repair loop."""

    text: str  # last motif text (best effort)
    compiled: CompiledCandidate | None
    attempts: int  # model calls used
    errors: list[str] = field(default_factory=list)  # final round's errors
    usage: list[dict] = field(default_factory=list)  # per-call usage dicts


def repair_loop(english: str, complete, *, max_attempts: int = 3,
                sanity=None) -> RepairResult:
    """Run the §7.1 loop.

    `complete(messages)` returns (text, usage) and hides the transport
    (real client or test stub). `sanity(compiled)` optionally returns
    extra feedback strings (fire-rate loosen/tighten). Repair stops at
    the first candidate that compiles AND draws no sanity feedback.
    """
    system = system_prompt()
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user_prompt(english)}]
    usage: list[dict] = []
    last_text, last_errors = "", []
    for _ in range(max(1, max_attempts)):
        text, use = complete(messages)
        usage.append(use)
        last_text = text
        compiled, errors = try_compile(text)
        if compiled is not None and sanity is not None:
            try:
                errors = list(sanity(compiled))
            except Exception as exc:
                errors = [f"scoring crashed: {_truncate(f'{type(exc).__name__}: {exc}')}"]
        last_errors = errors
        if not errors and compiled is not None:
            return RepairResult(text, compiled, len(usage), [], usage)
        messages = [{"role": "system", "content": system},
                    {"role": "user",
                     "content": repair_prompt(english, text, errors or ["unknown failure"])}]
    compiled, _ = try_compile(last_text)
    return RepairResult(last_text, compiled, len(usage), last_errors, usage)
