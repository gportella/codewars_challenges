#! /usr/bin/env python
# reusing

import math
import re


class Num:
    __slots__ = ("value",)

    def __init__(self, value):
        self.value = value

    def __iter__(self):
        return iter(())


class Var:
    __slots__ = ("name",)

    def __init__(self, name="x"):
        self.name = name

    def __iter__(self):
        return iter(())


class Add:
    __slots__ = ("terms",)

    def __init__(self, terms):
        self.terms = terms

    def __iter__(self):
        return iter(self.terms)


class Mul:
    __slots__ = ("factors",)

    def __init__(self, factors):
        self.factors = factors

    def __iter__(self):
        return iter(self.factors)


class Div:
    __slots__ = ("num_factors", "denum_factors")

    def __init__(self, num_factors, denum_factors):
        self.num_factors = num_factors
        self.denum_factors = denum_factors

    def __iter__(self):
        yield self.num_factors, self.denum_factors


class Pow:
    __slots__ = ("base", "exp")

    def __init__(self, base, exp):
        self.base = base
        self.exp = exp

    def __iter__(self):
        yield self.base
        yield self.exp


class Neg:
    __slots__ = ("child",)

    def __init__(self, child):
        self.child = child

    def __iter__(self):
        yield self.child


class Func:
    __slots__ = ("name", "child")

    def __init__(self, name, child):
        self.name = name
        self.child = child

    def __iter__(self):
        yield self.child


TOKEN_RE = re.compile(
    r"\s*(\(|\)|\*\*|[+\-*/^]|\d+(?:\.\d+)?|[A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE
)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
VAR_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
INFIX_TOKEN_RE = re.compile(
    r"\s*(\d+(?:\.\d+)?|[A-Za-z_][A-Za-z0-9_]*|\*\*|[()+\-*/^])"
)


def tokenize(s):
    tokens = []
    pos = 0
    while pos < len(s):
        match = TOKEN_RE.match(s, pos)
        if not match:
            raise SyntaxError(f"Unexpected character at position {pos}: {s[pos]!r}")
        token = match.group(1)
        pos = match.end()
        if not token:
            continue
        if token in ("(", ")"):
            continue
        tokens.append(token)
    return tokens


def tokenize_infix(s):
    tokens = []
    pos = 0
    while pos < len(s):
        match = INFIX_TOKEN_RE.match(s, pos)
        if not match:
            raise SyntaxError(f"Unexpected character at position {pos}: {s[pos]!r}")
        token = match.group(1)
        pos = match.end()
        if token:
            tokens.append(token)
    return tokens


def _make_add(left, right):
    left_terms = left.terms if isinstance(left, Add) else (left,)
    right_terms = right.terms if isinstance(right, Add) else (right,)
    return Add(left_terms + right_terms)


def _make_mul(left, right):
    left_factors = left.factors if isinstance(left, Mul) else (left,)
    right_factors = right.factors if isinstance(right, Mul) else (right,)
    return Mul(left_factors + right_factors)


def _split_div(node):
    if isinstance(node, Div):
        return node.num_factors, node.denum_factors
    return (node,), ()


def _make_div(left, right):
    left_num, left_denum = _split_div(left)
    right_num, right_denum = _split_div(right)
    return Div(left_num + right_denum, left_denum + right_num)


BINARY_OPERATORS = {
    "+": _make_add,
    "-": lambda left, right: _make_add(left, Neg(right)),
    "*": _make_mul,
    "/": _make_div,
    "**": lambda left, right: Pow(left, right),
    "^": lambda left, right: Pow(left, right),
}

UNARY_FUNCTIONS = {"sin", "cos", "ln", "exp", "tan"}


class InfixParser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.i = 0

    def peek(self):
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def _peek_next(self):
        return self.tokens[self.i + 1] if self.i + 1 < len(self.tokens) else None

    def take(self, expected=None):
        token = self.peek()
        if token is None:
            if expected is None:
                return None
            raise SyntaxError(f"Expected {expected!r} but hit end of input")
        if expected is not None and token != expected:
            raise SyntaxError(f"Expected {expected!r} but got {token!r}")
        self.i += 1
        return token

    def parse(self):
        expr = self._expr()
        if self.peek() is not None:
            raise SyntaxError(f"Unexpected trailing token: {self.peek()!r}")
        return expr

    def _expr(self):
        node = self._term()
        while True:
            tok = self.peek()
            if tok not in {"+", "-"}:
                break
            op = self.take()
            right = self._term()
            if op == "+":
                node = _make_add(node, right)
            else:
                node = _make_add(node, Neg(right))
        return node

    def _term(self):
        node = self._power()
        while True:
            tok = self.peek()
            if tok not in {"*", "/"}:
                break
            op = self.take()
            right = self._power()
            if op == "*":
                node = _make_mul(node, right)
            else:
                node = _make_div(node, right)
        return node

    def _power(self):
        node = self._unary()
        tok = self.peek()
        if tok in {"^", "**"}:
            self.take()
            node = Pow(node, self._power())
        return node

    def _unary(self):
        tok = self.peek()
        if tok in {"+", "-"}:
            self.take()
            child = self._unary()
            return child if tok == "+" else Neg(child)

        if tok and VAR_RE.fullmatch(tok) and self._peek_next() == "(":
            name_token = self.take()
            assert name_token is not None
            self.take("(")
            arg = self._expr()
            self.take(")")
            return Func(name_token, arg)

        if tok == "(":
            self.take("(")
            node = self._expr()
            self.take(")")
            return node

        if tok and NUMBER_RE.fullmatch(tok):
            self.take()
            value = float(tok) if "." in tok else int(tok)
            return Num(value)

        if tok and VAR_RE.fullmatch(tok):
            self.take()
            return Var(tok)

        raise SyntaxError(f"Unexpected token in expression: {tok!r}")


class PrefixParser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.i = 0

    def parse(self):
        expr = self._parse_node()
        if self.i != len(self.tokens):
            raise SyntaxError("Extra tokens after valid prefix expression")
        return expr

    def _next_token(self):
        if self.i >= len(self.tokens):
            raise SyntaxError("Unexpected end of input")
        token = self.tokens[self.i]
        self.i += 1
        return token

    def _parse_node(self):
        token = self._next_token()

        if token in BINARY_OPERATORS:
            left = self._parse_node()
            right = self._parse_node()
            return BINARY_OPERATORS[token](left, right)

        if token == "neg":
            return Neg(self._parse_node())

        if token in UNARY_FUNCTIONS:
            return Func(token, self._parse_node())

        if NUMBER_RE.fullmatch(token):
            value = float(token) if "." in token else int(token)
            return Num(value)

        if VAR_RE.fullmatch(token) and token not in UNARY_FUNCTIONS:
            return Var(token)

        raise SyntaxError(f"Unrecognized token: {token}")


def ast_to_prefix(node):
    def render(n):
        if isinstance(n, (Num, Var)):
            return str(n.value) if isinstance(n, Num) else n.name
        if isinstance(n, Neg):
            return f"(- {render(n.child)})"
        if isinstance(n, Func):
            return f"({n.name} {render(n.child)})"
        if isinstance(n, Add):
            parts = " ".join(render(term) for term in n.terms)
            return f"(+ {parts})"
        if isinstance(n, Mul):
            parts = " ".join(render(factor) for factor in n.factors)
            return f"(* {parts})"
        if isinstance(n, Div):
            num = " ".join(render(f) for f in n.num_factors)
            den = " ".join(render(f) for f in n.denum_factors)
            return f"(/ (* {num}) (* {den}))"
        if isinstance(n, Pow):
            return f"(^ {render(n.base)} {render(n.exp)})"
        raise TypeError(f"Unsupported node: {n!r}")

    return render(node)


def eval_expression(node):
    if isinstance(node, Num):
        return node.value
    if isinstance(node, Neg):
        return -eval_expression(node.child)
    if isinstance(node, Func):
        arg = eval_expression(node.child)
        if node.name == "sin":
            return math.sin(arg)
        if node.name == "cos":
            return math.cos(arg)
        if node.name == "ln":
            return math.log(arg)
        if node.name == "exp":
            return math.exp(arg)
        if node.name == "tan":
            return math.tan(arg)
        raise ValueError(f"Unsupported function: {node.name}")
    if isinstance(node, Add):
        return sum(eval_expression(t) for t in node.terms)
    if isinstance(node, Mul):
        v = 1
        for f in node.factors:
            v *= eval_expression(f)
        return v
    if isinstance(node, Div):
        num_groups, den_groups = next(iter(node))
        num_val = 1
        for factor in num_groups:
            num_val *= eval_expression(factor)
        den_val = 1
        for factor in den_groups:
            den_val *= eval_expression(factor)
        return num_val / den_val

    if isinstance(node, Pow):
        return eval_expression(node.base) ** eval_expression(node.exp)
    raise TypeError(f"Unknown node: {node}")


def infix_to_prefix(expr):
    tokens = tokenize_infix(expr)
    parser = InfixParser(tokens)
    ast = parser.parse()
    return ast_to_prefix(ast)


infix_expr = ["-7 * -(6 / 3)", "-7.131343 * 3.34413"]


def calc(expression):
    tokens = tokenize_infix(expression)
    parser_ast = InfixParser(tokens)
    ast = parser_ast.parse()
    evaluated = eval_expression(ast)
    return evaluated


for infx in infix_expr:
    solution = calc(infx)
    print(f"Solution: {solution}")
