#! /usr/bin/env python
""" Differentiate simple polynomial and evaluate at a point"""

import re
from dataclasses import dataclass
from typing import Tuple, Optional

TOKEN_RE = re.compile(
    r"""
    (?P<NUMBER>\d+(?:\.\d+)?) |
    (?P<VAR>[A-Za-z])        |
    (?P<POW>\*\*|[\^])         |
    (?P<MUL>\*)              |
    (?P<PLUS>\+)             |
    (?P<MINUS>-)             |
    (?P<LPAREN>\()           |
    (?P<RPAREN>\))           |
    (?P<WS>\s+)
""",
    re.VERBOSE,
)


@dataclass(frozen=True)
class Num:
    value: int

    def __iter__(self):
        return iter(())


@dataclass(frozen=True)
class Var:
    name: str = "x"

    def __iter__(self):
        return iter(())


@dataclass(frozen=True)
class Add:
    terms: Tuple[object, ...]

    def __iter__(self):
        return iter(self.terms)


@dataclass(frozen=True)
class Mul:
    factors: Tuple[object, ...]

    def __iter__(self):
        return iter(self.factors)


@dataclass(frozen=True)
class Pow:
    base: object
    exp: object

    def __iter__(self):
        yield self.base
        yield self.exp


@dataclass(frozen=True)
class Neg:
    child: object

    def __iter__(self):
        yield self.child


def tokenize(s):
    pos = 0
    for m in TOKEN_RE.finditer(s):
        if m.start() != pos:
            raise SyntaxError(f"Unexpected character at {pos}: {s[pos : m.start()]}")
        pos = m.end()
        kind = m.lastgroup
        text = m.group()
        if kind == "WS":
            continue
        if kind == "NUMBER":
            val = int(text) if "." not in text else float(text)
            yield ("NUMBER", val)
        else:
            yield (kind, text)
    if pos != len(s):
        raise SyntaxError(f"Unexpected trailing input at {pos}: {s[pos:]}")


class Parser:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.i = 0

    def peek(self) -> Optional[tuple]:
        return self.tokens[self.i] if self.i < len(self.tokens) else None

    def take(self, kind=None):
        t = self.peek()
        if t is None:
            return None
        if kind and t[0] != kind:
            return None
        self.i += 1
        return t

    def parse(self):
        node = self.expr()
        if self.peek() is not None:
            raise SyntaxError("Probably unknown tokens we could not parse?")
        return node

    def expr(self):
        left = self.term()
        terms = [left]
        while True:
            t = self.peek()
            if t and t[0] in ("PLUS", "MINUS"):
                op = self.take()[0]
                right = self.term()
                if op == "PLUS":
                    terms.append(right)
                else:
                    terms.append(Neg(right))
            else:
                break
        if len(terms) == 1:
            return terms[0]
        return Add(tuple(terms))

    def term(self):
        factors = [self.power()]
        while True:
            t = self.peek()
            if t is None:
                break

            if t[0] in ("PLUS", "MINUS"):
                break

            if t[0] == "MUL":
                op = self.take()[0]
                right = self.power()
                if op == "MUL":
                    factors.append(right)
                else:
                    factors.append(Pow(right, Num(-1)))
                continue

            if t[0] == "POW" or t[0] == "MINUS":
                break

            if t[0] in ("NUMBER", "VAR", "LPAREN"):
                factors.append(self.power())
                continue
            break

        return factors[0] if len(factors) == 1 else Mul(tuple(factors))

    def power(self):
        base = self.factor()
        t = self.peek()
        if t and t[0] == "POW":
            self.take("POW")
            exp = self.power()
            return Pow(base, exp)
        return base

    def factor(self):
        t = self.peek()
        if t is None:
            raise SyntaxError("Unexpected end of input")
        if t[0] == "NUMBER":
            self.take()
            return Num(int(t[1])) if isinstance(t[1], int) else Num(t[1])
        if t[0] == "VAR":
            self.take()
            return Var(t[1])
        if t[0] == "LPAREN":
            self.take("LPAREN")
            node = self.expr()
            if not self.take("RPAREN"):
                raise SyntaxError("Missing closing parenthesis")
            return node
        if t[0] == "MINUS":
            self.take("MINUS")
            return Neg(self.factor())
        raise SyntaxError(f"Unexpected token: {t}")


def eval_ast(node, xval):
    if isinstance(node, Num):
        return node.value
    if isinstance(node, Var):
        return xval
    if isinstance(node, Neg):
        return -eval_ast(node.child, xval)
    if isinstance(node, Add):
        return sum(eval_ast(t, xval) for t in node.terms)
    if isinstance(node, Mul):
        v = 1
        for f in node.factors:
            v *= eval_ast(f, xval)
        return v
    if isinstance(node, Pow):
        return eval_ast(node.base, xval) ** eval_ast(node.exp, xval)
    raise TypeError(f"Unknown node: {node}")


def derivative(node):
    if isinstance(node, Num):
        return Num(0)
    if isinstance(node, Var):
        return Num(1)
    if isinstance(node, Neg):
        return Neg(derivative(node.child))
    if isinstance(node, Add):
        return Add(tuple(derivative(t) for t in node.terms))

    if isinstance(node, Mul):
        terms = []
        for i, n in enumerate(node.factors):
            dfi = derivative(n)
            if dfi is not Num(0):
                rest_der = Mul(
                    tuple([dfi] + [g for j, g in enumerate(node.factors) if j != i])
                )
                terms.append(rest_der)
        return Add(tuple(terms))

    if isinstance(node, Pow):
        base, exp = node.base, node.exp
        if (
            isinstance(base, Var)
            and isinstance(exp, Num)
            and isinstance(exp.value, int)
        ):
            if exp.value == 0:
                return Num(0)
            return Mul((Num(exp.value), Pow(base, Num(exp.value - 1))))
    else:
        raise NotImplementedError(f"Sorry, normal poly only, don't know {node}")


def differentiate(expr: str, val):
    tokens = tokenize(expr)
    parser = Parser(tokens)
    ast = parser.parse()
    der_ast = derivative(ast)
    return eval_ast(der_ast, val)
