"""MotifQL surface parser: Lark grammar to typed AST (Oct 29, compiler step 1).

Parses a `.motif` file per `grammar/motifql.lark` into frozen dataclasses.
Literals are normalised on the way in: durations to seconds, percentages to
fractions, money to (amount, currency). The AST covers the whole v0 surface
(path and group motifs alike); executing group motifs is Oct 30 work, so
the planner rejects them with a clean error until then.

Note on `grammar/motifql.lark`: the group rule carries key/member
sub-rules (`key_list`, `member_list`) that only restructure the parse tree
— the accepted language is byte-identical, so the Oct 29 grammar freeze
holds. They exist because anonymous `("," NAME)*` repetitions otherwise
erase the keys/members boundary from the tree.

Usage:
    from motifql.parser import parse_file
    doc = parse_file("examples/amlworld.motif")
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

from lark import Lark, Token, Transformer

_GRAMMAR = Path(__file__).resolve().parent.parent.parent / "grammar" / "motifql.lark"

SECS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0, "w": 604800.0}


# -- literals ------------------------------------------------------------

@dataclass(frozen=True)
class Num:
    value: float


@dataclass(frozen=True)
class Dur:
    seconds: float


@dataclass(frozen=True)
class Pct:
    fraction: float


@dataclass(frozen=True)
class Money:
    amount: float
    currency: str


@dataclass(frozen=True)
class Ref:
    parts: tuple[str, ...]


@dataclass(frozen=True)
class Neg:
    body: object


@dataclass(frozen=True)
class DistinctArg:
    ref: Ref


@dataclass(frozen=True)
class AllArgs:
    pass


# -- expressions (shared by where / having / soft; legality is checked later)

@dataclass(frozen=True)
class And:
    items: tuple


@dataclass(frozen=True)
class Or:
    items: tuple


@dataclass(frozen=True)
class Not:
    body: object


@dataclass(frozen=True)
class Comparison:
    op: str
    left: object
    right: object


@dataclass(frozen=True)
class Arith:
    op: str  # "+" | "-" | "*" | "/"
    left: object
    right: object


@dataclass(frozen=True)
class Approx:
    left: object
    right: object
    tol: object  # Pct | Money | Dur | Num


@dataclass(frozen=True)
class IsTerm:
    base: object
    negated: bool
    term: str
    args: tuple


@dataclass(frozen=True)
class Quantified:
    quant: str  # each|all|all_but:K|some|most|few|at_least:N|named:name
    var: str
    body: object


@dataclass(frozen=True)
class Call:
    name: str
    args: tuple


# -- patterns ------------------------------------------------------------

@dataclass(frozen=True)
class NodePat:
    var: str
    label: str | None


@dataclass(frozen=True)
class EdgePat:
    var: str
    label: str | None
    direction: str  # "out" | "in"
    min_hops: int | None
    max_hops: int | None


@dataclass(frozen=True)
class Pattern:
    nodes: tuple[NodePat, ...]
    edges: tuple[EdgePat, ...]  # len(edges) == len(nodes) - 1


# -- top level -----------------------------------------------------------

@dataclass(frozen=True)
class TermDef:
    name: str
    params: tuple[str, ...]
    shape: str
    args: tuple


@dataclass(frozen=True)
class QuantifierDef:
    name: str
    shape: str
    args: tuple


@dataclass(frozen=True)
class GapTermCond:
    """Prunable per-step `e.gap is <term>` (skipped, not failed, on step one)."""

    var: str
    term: str
    term_args: tuple
    negated: bool


@dataclass(frozen=True)
class AmountConsCond:
    """Prunable per-step `e.amount ~= prev.amount within <tol>` (ditto)."""

    var: str
    tol: object  # Pct | Money | Dur | Num
    negated: bool = False


@dataclass(frozen=True)
class GroupClause:
    keys: tuple[str, ...]
    members: tuple[str, ...]
    window_s: float | None


@dataclass(frozen=True)
class Motif:
    name: str
    patterns: tuple[Pattern, ...]
    group: GroupClause | None
    where: object | None
    having: object | None
    soft: object | None
    score: str  # "min" | "product"
    emit: float
    line: int = 0


@dataclass(frozen=True)
class Document:
    terms: tuple[TermDef, ...]
    quantifiers: tuple[QuantifierDef, ...]
    motifs: tuple[Motif, ...]


def _dur(tok: Token) -> float:
    v = tok.value
    for suffix, mult in sorted(SECS.items(), key=lambda kv: -len(kv[0])):
        if v.endswith(suffix):
            return float(v[: -len(suffix)]) * mult
    raise ValueError(f"bad duration: {v}")


class _ToAST(Transformer):
    def NUMBER(self, tok: Token) -> Num:
        return Num(float(tok.value))

    def DURATION(self, tok: Token) -> Dur:
        return Dur(_dur(tok))

    def PERCENT(self, tok: Token) -> Pct:
        return Pct(float(tok.value[:-1]) / 100.0)

    def money(self, children: list) -> Money:
        return Money(float(children[0].value), children[1].value)

    def ref(self, children: list) -> Ref:
        return Ref(tuple(c.value for c in children))

    def distinct(self, children: list) -> DistinctArg:
        return DistinctArg(children[0])

    def all_bindings(self, _children: list) -> AllArgs:
        return AllArgs()

    def call(self, children: list) -> Call:
        name = children[0].value
        args = tuple(children[1]) if len(children) > 1 else ()
        return Call(name, args)

    def call_args(self, children: list) -> list:
        return children

    def params(self, children: list) -> list:
        return [c.value for c in children]

    def term_def(self, children: list) -> TermDef:
        name = children[0].value
        if len(children) == 3:
            params, call = children[1], children[2]
        else:
            params, call = [], children[1]
        return TermDef(name, tuple(params), call.name, call.args)

    def quantifier_def(self, children: list) -> QuantifierDef:
        return QuantifierDef(children[0].value, children[1].name, children[1].args)

    def label(self, children: list) -> str:
        return children[0].value

    def node(self, children: list) -> NodePat:
        var = children[0].value
        label = children[1] if len(children) > 1 else None
        return NodePat(var, label)

    def hops(self, children: list) -> tuple[int, int]:
        return (int(children[0].value), int(children[1].value))

    def edge_spec(self, children: list) -> list:
        return children

    def _edge(self, children: list, direction: str) -> EdgePat:
        spec = children[0]
        var = spec[0].value
        label, hops = None, None
        for item in spec[1:]:
            if isinstance(item, tuple):
                hops = item
            else:
                label = item
        lo, hi = hops if hops else (None, None)
        return EdgePat(var, label, direction, lo, hi)

    def out_edge(self, children: list) -> EdgePat:
        return self._edge(children, "out")

    def in_edge(self, children: list) -> EdgePat:
        return self._edge(children, "in")

    def pattern(self, children: list) -> Pattern:
        return Pattern(tuple(children[0::2]), tuple(children[1::2]))

    def match_clause(self, children: list) -> list:
        return children

    def key_list(self, children: list) -> tuple:
        return ("keys", [c.value for c in children])

    def member_list(self, children: list) -> tuple:
        return ("members", [c.value for c in children])

    def window(self, children: list) -> Dur:
        return children[0]

    def group_clause(self, children: list) -> tuple:
        keys: tuple[str, ...] = ()
        members: tuple[str, ...] = ()
        window = None
        for c in children:
            if isinstance(c, tuple):
                if c[0] == "keys":
                    keys = tuple(c[1])
                else:
                    members = tuple(c[1])
            elif isinstance(c, Dur):
                window = c.seconds
        return ("group", GroupClause(keys, members, window))

    def where_clause(self, children: list) -> tuple:
        return ("where", children[0])

    def having_clause(self, children: list) -> tuple:
        return ("having", children[0])

    def soft_clause(self, children: list) -> tuple:
        return ("soft", children[0])

    def score_clause(self, children: list) -> tuple:
        return ("score", children[0].value)

    def emit_clause(self, children: list) -> tuple:
        return ("emit", float(children[0].value))

    def motif(self, children: list) -> Motif:
        name = children[0].value
        patterns = tuple(children[1])
        group = where = having = soft = None
        score, emit = "min", 0.5
        seen: set[str] = set()
        for clause in children[2:]:
            kind, value = clause
            if kind in seen:
                raise ValueError(f"duplicate `{kind}` clause in motif `{name}`: "
                                 "clauses after `match` may appear at most once each.")
            seen.add(kind)
            if kind == "group":
                group = value
            elif kind == "where":
                where = value
            elif kind == "having":
                having = value
            elif kind == "soft":
                soft = value
            elif kind == "score":
                score = value
            elif kind == "emit":
                emit = value
        return Motif(name, patterns, group, where, having, soft, score, emit)

    # -- expressions --
    def or_(self, children: list) -> Or:
        return Or(tuple(children))

    def and_(self, children: list) -> And:
        return And(tuple(children))

    def not_(self, children: list) -> Not:
        return Not(children[0])

    def true_(self, _children: list) -> bool:
        return True

    def false_(self, _children: list) -> bool:
        return False

    def q_each(self, _c: list) -> str:
        return "each"

    def q_all(self, _c: list) -> str:
        return "all"

    def q_all_but(self, children: list) -> str:
        return f"all_but:{children[0].value}"

    def q_some(self, _c: list) -> str:
        return "some"

    def q_most(self, _c: list) -> str:
        return "most"

    def q_few(self, _c: list) -> str:
        return "few"

    def q_at_least(self, children: list) -> str:
        return f"at_least:{children[0].value}"

    def q_named(self, children: list) -> str:
        return f"named:{children[0].value}"

    def quantified(self, children: list) -> Quantified:
        return Quantified(children[0], children[1].value, children[2])

    def comparison(self, children: list) -> Comparison:
        return Comparison(children[1].value, children[0], children[2])

    def approx(self, children: list) -> Approx:
        return Approx(children[0], children[1], children[2])

    def membership(self, children: list) -> IsTerm:
        base = children[0]
        rest = children[1:]
        negated = False
        if rest and isinstance(rest[0], Token) and rest[0].type == "NEGATE":
            negated = True
            rest = rest[1:]
        term = rest[0].value
        args = tuple(rest[1]) if len(rest) > 1 else ()
        return IsTerm(base, negated, term, args)

    def binop(self, children: list) -> Arith:
        return Arith(children[1].value, children[0], children[2])

    def neg(self, children: list) -> Neg:
        return Neg(children[0])


def parse_text(text: str) -> Document:
    """Parse MotifQL source text into a typed AST document."""
    parser = Lark(
        _GRAMMAR.read_text(),
        parser="earley",
        maybe_placeholders=False,
        propagate_positions=True,
    )
    tree = parser.parse(text)
    conv = _ToAST()
    terms, quants, motifs = [], [], []
    for item in tree.children:
        node = conv.transform(item)
        if isinstance(node, TermDef):
            terms.append(node)
        elif isinstance(node, QuantifierDef):
            quants.append(node)
        elif isinstance(node, Motif):
            line = getattr(item.meta, "line", 0) or 0
            motifs.append(dataclasses.replace(node, line=line))
    return Document(tuple(terms), tuple(quants), tuple(motifs))


def parse_file(path: str | Path) -> Document:
    """Parse a `.motif` file into a typed AST document."""
    return parse_text(Path(path).read_text())
