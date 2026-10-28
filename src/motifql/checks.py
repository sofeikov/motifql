"""MotifQL static checks: LANGUAGE.md section 5 (Oct 29, compiler step 2).

Five checks, each with a stable code, the location, the problem and a
suggested fix — the messages feed straight back to the model in the Oct 31
repair loop, so they say what to do, not just what is wrong:

* E_TIME   — every motif needs a time bound (span <= D in where, or a
             group window).
* E_UNIT   — unknown or mismatched units (amount ~= time, duration term
             on money, tolerance that does not fit the compared values).
* E_CRISP  — fuzzy constructs (~=, graded terms, most/few/named
             quantifiers) in the crisp where/having clauses.
* E_SCOPE  — scoping: unbound variables, prev outside a path quantifier,
             quantifiers over non-members, having without group,
             aggregates outside a group, unknown term/quantifier names.
* E_COST   — estimated expansions per anchor above budget (needs live
             degree stats; skipped without them, wired by the planner).

Usage:
    from motifql.parser import parse_file
    from motifql import checks
    doc = parse_file("examples/amlworld.motif")
    errors = checks.check_document(doc)  # [] means the file compiles
"""

from __future__ import annotations

from dataclasses import dataclass

from motifql.parser import (
    AllArgs, And, Approx, Arith, Call, Comparison, DistinctArg, Document, Dur,
    GroupClause, IsTerm, Money, Motif, Neg, Not, Num, Or, Pct, Quantified,
    Ref,
)

# -- error type ----------------------------------------------------------

@dataclass(frozen=True)
class CheckError:
    code: str      # E_TIME | E_UNIT | E_CRISP | E_SCOPE | E_COST
    motif: str
    line: int
    problem: str
    fix: str

    def message(self) -> str:
        loc = f"{self.motif}:{self.line}" if self.line else self.motif
        return f"[{self.code}] {loc}: {self.problem} Fix: {self.fix}"


# -- scopes --------------------------------------------------------------

# Variable kinds in a motif body.
_ACCOUNT = "account"
_EDGE = "edge"
_PATH = "path"  # variable-length edge: ranges over whole paths
_MEMBER = "member"  # group member var: ranges over member bindings


def _pattern_scope(motif: Motif) -> dict[str, str]:
    """Map every pattern variable to its kind."""
    scope: dict[str, str] = {}
    for pat in motif.patterns:
        for node in pat.nodes:
            scope.setdefault(node.var, _ACCOUNT)
        for edge in pat.edges:
            kind = _PATH if edge.max_hops else _EDGE
            scope.setdefault(edge.var, kind)
    if motif.group is not None:
        for key in motif.group.keys:
            scope.setdefault(key, _ACCOUNT)
        for member in motif.group.members:
            scope[member] = _MEMBER
    return scope


# -- type inference ------------------------------------------------------

_TIME = "time"
_MONEY = "money"
_DURATION = "duration"
_NUMBER = "number"
_BOOL = "bool"
_DEGREE = "degree"  # fuzzy value in [0, 1]

_EDGE_FIELDS = {
    "time": _TIME,
    "amount": _MONEY,
    "amount_paid": _MONEY,
    "currency": _MONEY,
    "payment_format": _MONEY,
    "gap": _DURATION,
    "index": _NUMBER,
}

_CRISP_TERMS = {"time_ordered"}


class _Checker:
    def __init__(self, doc: Document) -> None:
        self.terms = {t.name for t in doc.terms}
        self.term_defs = {t.name: t for t in doc.terms}
        self.quantifiers = {q.name for q in doc.quantifiers}
        self.errors: list[CheckError] = []

    def err(self, code: str, motif: Motif, problem: str, fix: str) -> None:
        self.errors.append(CheckError(code, motif.name, motif.line, problem, fix))

    # -- entry points --
    def check_document(self, doc: Document) -> list[CheckError]:
        for motif in doc.motifs:
            self.check_motif(motif)
        for term in doc.terms:
            self._check_term_def(term)
        return self.errors

    def check_motif(self, motif: Motif) -> None:
        scope = _pattern_scope(motif)
        self._check_time_bound(motif)
        if motif.where is not None:
            self._check_crisp(motif, motif.where, "where")
            self._check_expr(motif, motif.where, scope, crisp=True)
        if motif.having is not None:
            if motif.group is None:
                self.err("E_SCOPE", motif, "`having` without `group`.",
                         "add a `group by ... over ...` clause or move the condition to `where`.")
            else:
                self._check_crisp(motif, motif.having, "having")
                self._check_expr(motif, motif.having, scope, crisp=True)
        if motif.soft is not None:
            self._check_expr(motif, motif.soft, scope, crisp=False)
        for clause in (motif.where, motif.having, motif.soft):
            if clause is not None:
                self._check_prev(motif, clause, scope)

    # -- E_TIME --
    def _check_time_bound(self, motif: Motif) -> None:
        if motif.group is not None and motif.group.window_s is not None:
            return
        if motif.where is not None and _has_span_bound(motif.where):
            return
        self.err("E_TIME", motif, "no time bound: neither `span(...) <= D` in `where` nor a `group ... window D`.",
                 "add e.g. `span(e) <= 7d` to `where`, or a `window` to the `group` clause.")

    # -- E_CRISP --
    def _check_crisp(self, motif: Motif, expr: object, clause: str) -> None:
        for node in _walk(expr):
            if isinstance(node, Approx):
                self.err("E_CRISP", motif, f"`~=` is fuzzy and cannot appear in `{clause}`.",
                         f"move it to `soft`, or use an exact comparison in `{clause}`.")
            elif isinstance(node, IsTerm) and node.term not in _CRISP_TERMS:
                self.err("E_CRISP", motif, f"`is {node.term}` is graded and cannot appear in `{clause}`.",
                         f"move it to `soft`, or keep only `is time_ordered` here.")
            elif isinstance(node, Quantified) and (
                node.quant in ("most", "few")
                or node.quant.startswith("named:")
                or node.quant.startswith("all_but:")
                or node.quant in self.quantifiers
            ):
                # each/all/some/at_least:N are crisp; most/few/all_but/named are fuzzy.
                self.err("E_CRISP", motif, f"quantifier `{node.quant}` is fuzzy and cannot appear in `{clause}`.",
                         "use `each`/`all`/`some`/`at least N` here, or move the condition to `soft`.")

    # -- E_UNIT + E_SCOPE over expressions --
    def _check_expr(self, motif: Motif, expr: object, scope: dict[str, str], crisp: bool) -> None:
        for node in _walk(expr):
            if isinstance(node, Ref):
                self._check_ref(motif, node, scope)
            elif isinstance(node, Quantified):
                self._check_quantified(motif, node, scope)
            elif isinstance(node, Call):
                self._check_call(motif, node, scope)
            elif isinstance(node, IsTerm) and node.term not in _CRISP_TERMS:
                self._check_term_use(motif, node, scope)
        self._check_units(motif, expr, scope)

    def _check_ref(self, motif: Motif, ref: Ref, scope: dict[str, str]) -> None:
        head = ref.parts[0]
        if head == "prev":
            return  # contextual: validated against quantifier scope below
        if head not in scope:
            self.err("E_SCOPE", motif, f"unbound variable `{head}`.",
                     f"bind it in `match` (or `group`) before using it.")

    def _check_quantified(self, motif: Motif, node: Quantified, scope: dict[str, str]) -> None:
        kind = scope.get(node.var)
        if kind not in (_PATH, _MEMBER, _EDGE):
            self.err("E_SCOPE", motif, f"quantifier over `{node.var}`, which is not a path or member variable.",
                     "quantify over a `*m..n` path edge or a group member.")
        if node.quant.startswith("named:"):
            name = node.quant.split(":", 1)[1]
            if name not in self.quantifiers:
                self.err("E_SCOPE", motif, f"unknown quantifier `@{name}`.",
                         f"define `quantifier {name} = ...` or use a built-in.")

    def _check_call(self, motif: Motif, node: Call, scope: dict[str, str]) -> None:
        if node.name == "span":
            if len(node.args) != 1:
                self.err("E_SCOPE", motif, "`span` takes one argument.",
                         "write `span(e)` or `span(all)`.")
        elif node.name in ("count", "sum", "avg", "min", "max"):
            if motif.group is None:
                self.err("E_SCOPE", motif, f"aggregate `{node.name}` outside a group.",
                         "add a `group by ... over ...` clause, or drop the aggregate.")
        elif node.name in self.terms or node.name in ("trapezoid", "triangle", "ramp", "ramp_down"):
            pass  # term shapes and user terms: checked at IsTerm use
        else:
            self.err("E_SCOPE", motif, f"unknown function `{node.name}`.",
                     "use span, count, sum, avg, min, max, or a defined term.")

    def _check_term_use(self, motif: Motif, node: IsTerm, scope: dict[str, str]) -> None:
        if node.term not in self.terms:
            self.err("E_SCOPE", motif, f"unknown term `{node.term}`.",
                     f"define `term {node.term} = ...` or fix the name.")
            return
        want = len(self.term_defs[node.term].params)
        if len(node.args) != want:
            self.err("E_SCOPE", motif, f"term `{node.term}` takes {want} arguments, got {len(node.args)}.",
                     f"pass {want} arguments.")
            return
        if node.args:
            kinds = {self._infer(a, scope) for a in node.args}
            kinds.discard("unknown")
            if len(kinds) > 1:
                self.err("E_UNIT", motif, f"term `{node.term}` arguments mix units {sorted(kinds)}.",
                         "give every argument the same unit.")

    def _check_term_def(self, term) -> None:
        for node in _walk(term.args):
            if isinstance(node, Ref) and node.parts[0] not in term.params:
                self.errors.append(CheckError(
                    "E_SCOPE", f"term {term.name}", 0,
                    f"unbound parameter `{node.parts[0]}` in term `{term.name}`.",
                    f"use one of ({', '.join(term.params)}) or add it to the parameter list."))

    # -- unit inference --
    def _infer(self, node: object, scope: dict[str, str]) -> str:
        if isinstance(node, Num):
            return _NUMBER
        if isinstance(node, Dur):
            return _DURATION
        if isinstance(node, Pct):
            return _NUMBER
        if isinstance(node, Money):
            return _MONEY
        if isinstance(node, bool):
            return _BOOL
        if isinstance(node, Ref):
            return self._infer_ref(node, scope)
        if isinstance(node, Neg):
            return self._infer(node.body, scope)
        if isinstance(node, Comparison) or isinstance(node, (And, Or, Not)):
            return _BOOL
        if isinstance(node, Arith):
            return self._infer_arith(node, scope)
        if isinstance(node, (Approx, IsTerm)):
            return _DEGREE
        if isinstance(node, Quantified):
            return _DEGREE
        if isinstance(node, Call):
            if node.name == "span":
                return _DURATION
            if node.name == "count":
                return _NUMBER
            if node.name in ("sum", "avg", "min", "max"):
                return _MONEY
            return "unknown"
        if isinstance(node, tuple):
            return "unknown"
        return "unknown"

    def _term_domain(self, tdef, call_args: tuple, scope: dict[str, str]) -> str | None:
        """The value domain a term grades, or None when it cannot be told.

        Inferred from the definition's argument literals, with parameters
        bound to the call-site argument units. Pure-number literals (ramp
        breakpoints, scaling factors like `0.85 * limit`) never set the
        domain on their own. Ambiguous definitions skip the check rather
        than risk a false positive.
        """
        subst = {formal: self._infer(actual, scope)
                 for formal, actual in zip(tdef.params, call_args)}
        domains: set[str] = set()
        for arg in tdef.args:
            domains |= self._arg_domains(arg, subst)
        if _TIME in domains:
            return None
        domains.discard(_NUMBER)
        if len(domains) == 1:
            return domains.pop()
        if not domains:
            return _NUMBER
        return None

    def _arg_domains(self, node: object, subst: dict[str, str]) -> set[str]:
        if isinstance(node, Dur):
            return {_DURATION}
        if isinstance(node, Money):
            return {_MONEY}
        if isinstance(node, (Num, Pct)):
            return {_NUMBER}
        if isinstance(node, Ref) and len(node.parts) == 1 and node.parts[0] in subst:
            unit = subst[node.parts[0]]
            return {unit} if unit in (_TIME, _MONEY, _DURATION, _NUMBER) else set()
        if isinstance(node, Arith):
            return self._arg_domains(node.left, subst) | self._arg_domains(node.right, subst)
        if isinstance(node, Neg):
            return self._arg_domains(node.body, subst)
        return set()

    def _infer_arith(self, node, scope: dict[str, str]) -> str:
        lt, rt = self._infer(node.left, scope), self._infer(node.right, scope)
        if "unknown" in (lt, rt):
            return "unknown"
        if node.op == "/" and lt == rt:
            return _NUMBER  # a ratio of like units is a plain number
        if lt == rt:
            return lt
        if _NUMBER in (lt, rt):
            other = rt if lt == _NUMBER else lt
            if other in (_MONEY, _DURATION, _TIME):
                return other  # scaling: 0.85 * limit
        return "unknown"

    def _infer_ref(self, ref: Ref, scope: dict[str, str]) -> str:
        head = ref.parts[0]
        if head == "prev":
            field = ref.parts[1] if len(ref.parts) > 1 else None
            return _EDGE_FIELDS.get(field, "unknown")
        kind = scope.get(head, "unknown")
        if len(ref.parts) == 1:
            return _ACCOUNT if kind == _ACCOUNT else ("unknown" if kind == "unknown" else kind)
        if kind in (_EDGE, _PATH, _MEMBER):
            return _EDGE_FIELDS.get(ref.parts[1], "unknown")
        return "unknown"

    def _check_units(self, motif: Motif, expr: object, scope: dict[str, str]) -> None:
        for node in _walk(expr):
            if isinstance(node, Comparison):
                lt, rt = self._infer(node.left, scope), self._infer(node.right, scope)
                if "unknown" in (lt, rt):
                    continue
                if node.op in ("==", "!="):
                    if lt != rt:
                        self.err("E_UNIT", motif, f"`{node.op}` compares {lt} with {rt}.",
                                 "compare values of the same unit, or compare node identities.")
                elif lt != rt:
                    self.err("E_UNIT", motif, f"`{node.op}` compares {lt} with {rt}.",
                             "give both sides the same unit (time, money, duration or number).")
            elif isinstance(node, Approx):
                lt, rt = self._infer(node.left, scope), self._infer(node.right, scope)
                if "unknown" not in (lt, rt) and lt != rt:
                    self.err("E_UNIT", motif, f"`~=` compares {lt} with {rt}.",
                             "give both sides the same unit.")
                self._check_tol(motif, node, lt)
            elif isinstance(node, Arith):
                lt, rt = self._infer(node.left, scope), self._infer(node.right, scope)
                if "unknown" in (lt, rt):
                    continue
                if self._infer_arith(node, scope) != "unknown":
                    continue
                if {lt, rt} <= {_TIME, _DURATION}:
                    continue  # temporal arithmetic evaluates numerically
                self.err("E_UNIT", motif, f"`{node.op}` mixes {lt} with {rt}.",
                         "use the same unit on both sides (a plain number may scale money or durations).")
            elif isinstance(node, IsTerm) and node.term not in _CRISP_TERMS \
                    and node.term in self.terms:
                base = self._infer(node.base, scope)
                if base in (_ACCOUNT, _BOOL):
                    self.err("E_UNIT", motif, f"`is {node.term}` applies to {base}.",
                             "apply graded terms to numbers, money or durations.")
                elif base in (_TIME, _MONEY, _DURATION, _NUMBER):
                    dom = self._term_domain(self.term_defs[node.term], node.args, scope)
                    if dom is not None and base != dom:
                        self.err("E_UNIT", motif,
                                 f"`is {node.term}` applies a {dom} term to {base}.",
                                 "apply duration terms to gaps and spans, money terms "
                                 "to amounts, number terms to counts.")

    def _check_tol(self, motif: Motif, node: Approx, compared: str) -> None:
        tol = node.tol
        ok = isinstance(tol, Pct)
        if isinstance(tol, Money):
            ok = compared == _MONEY
        elif isinstance(tol, Dur):
            ok = compared in (_DURATION, _TIME)
        elif isinstance(tol, Num):
            ok = compared == _NUMBER
        if not ok:
            self.err("E_UNIT", motif,
                     f"tolerance `{tol}` does not fit compared {compared}.",
                     "use % for anything, money for money, durations for times/durations.")


    def _check_prev(self, motif: Motif, expr: object, scope: dict[str, str]) -> None:
        """`prev` is legal only under a quantifier over a path/member var."""

        def rec(node: object, allowed: bool) -> None:
            if isinstance(node, Quantified):
                kind = scope.get(node.var)
                rec(node.body, allowed or kind in (_PATH, _MEMBER))
                return
            if isinstance(node, Ref) and node.parts[0] == "prev" and not allowed:
                self.err("E_SCOPE", motif, "`prev` outside a quantifier over a path variable.",
                         "use `prev` only as `each e: ... prev ...` over a `*m..n` edge.")
                return
            for child in _children(node):
                rec(child, allowed)

        rec(expr, False)


def _children(node: object) -> list:
    """Direct child nodes of one AST node."""
    if isinstance(node, (And, Or)):
        return list(node.items)
    if isinstance(node, Not):
        return [node.body]
    if isinstance(node, Comparison):
        return [node.left, node.right]
    if isinstance(node, Arith):
        return [node.left, node.right]
    if isinstance(node, Approx):
        return [node.left, node.right, node.tol]
    if isinstance(node, IsTerm):
        return [node.base, *node.args]
    if isinstance(node, Quantified):
        return [node.body]
    if isinstance(node, Call):
        return list(node.args)
    if isinstance(node, DistinctArg):
        return [node.ref]
    if isinstance(node, Neg):
        return [node.body]
    if isinstance(node, (list, tuple)):
        return list(node)
    return []


def _walk(node: object):
    """Yield every AST node in the subtree (pre-order, cycle-free by shape)."""
    stack = [node]
    while stack:
        cur = stack.pop()
        yield cur
        if isinstance(cur, (And, Or)):
            stack.extend(cur.items)
        elif isinstance(cur, Not):
            stack.append(cur.body)
        elif isinstance(cur, Comparison):
            stack.extend((cur.left, cur.right))
        elif isinstance(cur, Arith):
            stack.extend((cur.left, cur.right))
        elif isinstance(cur, Approx):
            stack.extend((cur.left, cur.right, cur.tol))
        elif isinstance(cur, IsTerm):
            stack.append(cur.base)
            stack.extend(cur.args)
        elif isinstance(cur, Quantified):
            stack.append(cur.body)
        elif isinstance(cur, Call):
            stack.extend(cur.args)
        elif isinstance(cur, DistinctArg):
            stack.append(cur.ref)
        elif isinstance(cur, Neg):
            stack.append(cur.body)
        elif isinstance(cur, (list, tuple)):
            stack.extend(cur)


def walk(node: object):
    """Yield every AST node in the subtree (public alias for the planner)."""
    return _walk(node)


def _has_span_bound(where: object) -> bool:
    for node in _walk(where):
        if isinstance(node, Comparison) and node.op in ("<=", "<"):
            if isinstance(node.left, Call) and node.left.name == "span":
                if isinstance(node.right, (Dur, Num)):
                    return True
    return False


def check_document(doc: Document, stats=None) -> list[CheckError]:
    """Run every static check over a parsed document (E_COST needs stats)."""
    checker = _Checker(doc)
    errors = checker.check_document(doc)
    if stats is not None:
        errors.extend(_check_cost(doc, stats))
    return errors


def _check_cost(doc: Document, stats) -> list[CheckError]:
    # Wired by the planner (Oct 29 step 4) with live degree statistics.
    return []
