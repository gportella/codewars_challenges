#! /usr/bin/env python

import math
import re
from dataclasses import dataclass
from typing import Tuple, Union


@dataclass(frozen=True)
class Num:
    value: Union[int, float]

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
class Div:
    num_factors: Tuple[object, ...]
    denum_factors: Tuple[object, ...]

    def __iter__(self):
        yield self.num_factors, self.denum_factors


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


@dataclass(frozen=True)
class Func:
    name: str
    child: object

    def __iter__(self):
        yield self.child


TOKEN_RE = re.compile(
    r"\s*(\(|\)|\*\*|[+\-*/^]|\d+(?:\.\d+)?|[A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE
)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
VAR_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


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


def _make_add(left, right):
    left_terms = left.terms if isinstance(left, Add) else (left,)
    right_terms = right.terms if isinstance(right, Add) else (right,)
    return Add(left_terms + right_terms)


# Decide whether _make_mul should pull any Div from left/right so the final shape doesn’t end up with nested Div under multiplication.
def _make_mul(left, right):
    # if left
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


# todo
# Implement evaluation for Div in eval_ast, probably by multiplying numerator factors and dividing by the product of denominator factors.
# Extend derivative handling once you know how you want division to interact symbolically.
# Multiply by a Div: its numerator factors belong in the combined numerator, denominator factors in the combined denominator.
# If both sides are divisions, the result is another division with merged numerators and denominators.
# So “pulling Div from left/right” means recognizing when one operand is a fraction, unpacking it with
# something like _split_div, and collapsing the result into a single Div
# (or a simpler Mul if the denominator is empty). Whether you do that depends on how normalized you want the AST to be.

# Write a helper like to_prefix(node) returning a list of tokens (strings or numbers). Start by matching on the node type.
# For leaf nodes (Num, Var), return [str(value)] or [node.name].
# For unary nodes (Neg, Func), return [op_name] + to_prefix(child). For Neg, use a dedicated keyword like neg.
# For binary-ish nodes:
# Add: operators are +; fold ["+"] + flatten(children) where each child contributes its own prefix sequence.
# Mul: same with "*".
# Div: use ["/"] + to_prefix for numerator factors first, then denominator; if you’re keeping them grouped, you can emit an auxiliary operator like "*" inside each group. Example:
# ["/", *chain("+", ...? actually "*", etc.)].
# Pow: return ["^"] + to_prefix(base) + to_prefix(exp).
# Once you have token lists, join with spaces to get a readable string, e.g.:
# def prefix_string(node):
#     return " ".join(_prefix_tokens(node))

# print(prefix_string(ast))


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


def eval_ast(node, xval):
    if isinstance(node, Num):
        return node.value
    if isinstance(node, Var):
        return xval
    if isinstance(node, Neg):
        return -eval_ast(node.child, xval)
    if isinstance(node, Func):
        arg = eval_ast(node.child, xval)
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
    if isinstance(node, Func):
        name, child = node.name, node.child
        if name == "cos":
            dfi = Mul(tuple([derivative(child), Neg(Func(name="sin", child=child))]))
            return dfi
        if name in ["sin", "cos"]:
            dfi = Mul(tuple([derivative(child), Func(name=name, child=child)]))
            return dfi
        if name == "ln":
            dfi = Mul(tuple([derivative(child), Func(name=name, child=child)]))
            return dfi
        if name == "exp":
            dfi = Mul(tuple([derivative(child), Func(name="exp", child=child)]))
            return dfi
        else:
            raise NotImplementedError(
                f"Derivative for function '{node.name}' is not implemented"
            )
        # if (
        #     isinstance(base, Var)
        #     and isinstance(exp, Num)
        #     and isinstance(exp.value, int)
        # ):
        #     if exp.value == 0:
        #         return Num(0)
        #     return Mul((Num(exp.value), Pow(base, Num(exp.value - 1))))
    # if isinstance(node, Func):
    # raise NotImplementedError(
    # f"Derivative for function '{node.name}' is not implemented"
    # )
    else:
        raise NotImplementedError(f"Sorry, normal poly only, don't know {node}")


def differentiate(expr: str, val):
    tokens = tokenize(expr)
    parser = PrefixParser(tokens)
    ast = parser.parse()
    der_ast = derivative(ast)
    return eval_ast(der_ast, val)


expressions = [
    "(+ x 2)",
    "(+ (* 1 x) (* 2 (+ x 1)))",
    "(cos (+ 1 x))",
    "(cos x)",
    "(^ x 2)",
    # "(^ 1 (^ (+ (exp -1) (^ (- (* 0 x) x) 1)) 1))(* (+ x 3) 5)",
    # "(/ 1 2)",
]

for expr in expressions:
    print(f"Parsing {expr}")
    tokens = tokenize(expr)
    parser = PrefixParser(tokens)
    ast = parser.parse()
    # print(f"Back at you, from AST {ast_to_prefix(ast)}")
    print(f"Parsed Abstract syntax tree: {ast}")
    der_ast = derivative(ast)
    print(f"AST derivada es {der_ast}")
    print(f"Back at you, derivative from AST {ast_to_prefix(der_ast)}")
    # print(f"La derivadad: {der_ast}")
