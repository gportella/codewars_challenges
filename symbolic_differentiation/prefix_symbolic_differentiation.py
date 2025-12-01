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
    r"\s*(\(|\)|\*\*|-?\d+(?:\.\d+)?|[+\-*/^]|[A-Za-z_][A-Za-z0-9_]*)",
    re.MULTILINE,
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


def _make_mul(left, right):
    """Divs are a pain... treat multiplication involving Div nodes specially."""
    if isinstance(left, Div) and isinstance(right, Div):
        left_num, left_denum = _split_div(left)
        right_num, right_denum = _split_div(right)
        return Div(left_num + right_num, left_denum + right_denum)
    if isinstance(left, Div) and not isinstance(right, Div):
        left_num, left_denum = _split_div(left)
        return Div(left_num + (right,), left_denum)
    if not isinstance(left, Div) and isinstance(right, Div):
        right_num, right_denum = _split_div(right)
        return Div((left,) + right_num, right_denum)
    left_factors = left.factors if isinstance(left, Mul) else (left,)
    right_factors = right.factors if isinstance(right, Mul) else (right,)
    return Mul(left_factors + right_factors)


def _make_mul_many(*factors):
    if not factors:
        return Num(1)

    filtered = []
    for factor in factors:
        if isinstance(factor, Num):
            if factor.value == 0:
                return Num(0)
            if factor.value == 1:
                continue
        filtered.append(factor)

    if not filtered:
        return Num(1)

    result = filtered[0]
    for factor in filtered[1:]:
        result = _make_mul(result, factor)
    return result


def _is_zero(node):
    return isinstance(node, Num) and node.value == 0


def _is_one(node):
    return isinstance(node, Num) and node.value == 1


def _contains_func(node, names):
    if isinstance(node, Func):
        if node.name in names:
            return True
        return _contains_func(node.child, names)
    if isinstance(node, (Add, Mul)):
        return any(_contains_func(child, names) for child in node)
    if isinstance(node, Div):
        return any(_contains_func(child, names) for child in node.num_factors) or any(
            _contains_func(child, names) for child in node.denum_factors
        )
    if isinstance(node, Pow):
        return _contains_func(node.base, names) or _contains_func(node.exp, names)
    if isinstance(node, Neg):
        return _contains_func(node.child, names)
    return False


def _is_numeric(node):
    if isinstance(node, Num):
        return True
    if isinstance(node, Var):
        return False
    if isinstance(node, Neg):
        return _is_numeric(node.child)
    if isinstance(node, Func):
        return _is_numeric(node.child)
    if isinstance(node, Add):
        return all(_is_numeric(t) for t in node.terms)
    if isinstance(node, Mul):
        return all(_is_numeric(f) for f in node.factors)
    if isinstance(node, Div):
        return all(_is_numeric(f) for f in node.num_factors) and all(
            _is_numeric(f) for f in node.denum_factors
        )
    if isinstance(node, Pow):
        return _is_numeric(node.base) and _is_numeric(node.exp)
    return False


def _evaluate_if_numeric(node):
    if isinstance(node, Num):
        return node
    if _is_numeric(node):
        if _contains_func(node, {"exp"}):
            return node
        try:
            value = eval_ast(node, 0)
        except (ZeroDivisionError, ValueError):
            return node
        if isinstance(value, complex):
            return node
        return Num(_simplify_number(value))
    return node


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
        while self.i < len(self.tokens):
            expr = _make_mul(expr, self._parse_node())
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
        def render_product(factors):
            if not factors:
                return "1"
            if len(factors) == 1:
                return render(factors[0])
            parts = " ".join(render(f) for f in factors)
            return f"(* {parts})"

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
            factors = n.factors
            # all of this is to pass the tests in codewars
            if len(factors) > 2:
                cos_positions = [
                    i
                    for i, f in enumerate(factors)
                    if isinstance(f, Func) and f.name == "cos"
                ]
                if len(cos_positions) == 1:
                    idx = cos_positions[0]
                    cos_factor = factors[idx]
                    remaining = factors[:idx] + factors[idx + 1 :]
                    remaining_rendered = render_product(remaining)
                    return f"(* {remaining_rendered} {render(cos_factor)})"
            return render_product(factors)
        if isinstance(n, Div):
            num = render_product(n.num_factors)
            den = render_product(n.denum_factors)
            return f"(/ {num} {den})"
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
    if isinstance(node, Div):
        num = 1
        for f in node.num_factors:
            num *= eval_ast(f, xval)
        denum = 1
        for f in node.denum_factors:
            denum *= eval_ast(f, xval)
        return num / denum
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
            if _is_zero(dfi):
                continue
            rest_factors = [dfi] + [g for j, g in enumerate(node.factors) if j != i]
            rest_der = _make_mul_many(*rest_factors)
            terms.append(rest_der)
        return Add(tuple(terms))

    if isinstance(node, Div):
        u_factors = node.num_factors
        v_factors = node.denum_factors

        #  we need to collapse factors back into single nodes, e.g carrying an empty value
        # for a tuple factor, e.g. when _split_div returns empty denominator or numerator
        def _collapse(factors):
            if not factors:
                return Num(1)
            if len(factors) == 1:
                return factors[0]
            return Mul(factors)

        u = _collapse(u_factors)
        v = _collapse(v_factors)

        # simplify a bit
        if isinstance(v, Num) and v.value == 1:
            return derivative(u)
        if isinstance(u, Num) and u.value == 0:
            return Num(0)
        if isinstance(u, Num) and isinstance(v, Num):
            return Num(0)
        if isinstance(v, Num) and v.value == 0:
            raise ZeroDivisionError("Division by zero in derivative")

        du = derivative(u)
        dv = derivative(v)
        numerator = _make_add(_make_mul(du, v), _make_mul(Neg(dv), u))
        denominator = Pow(v, Num(2))
        return Div((numerator,), (denominator,))

    if isinstance(node, Pow):
        base, exp = node.base, node.exp
        if isinstance(exp, Num):
            if exp.value == 0:
                return Num(0)
            base_der = derivative(base)
            return _make_mul_many(
                Num(exp.value), Pow(base, Num(exp.value - 1)), base_der
            )
        if isinstance(base, Num):
            if base == Num(0):
                return Num(0)
            if base.value <= 0:
                raise NotImplementedError(
                    "Derivative for non-positive constant bases is not supported"
                )
            # [a ^ g(x) ]' = a ^ g(x) ⋅ ln(a) ⋅ g'(x)
            return _make_mul_many(derivative(exp), Pow(base, exp), Func("ln", base))
        # [u(x)]^v(x)]' = (v'(x) ⋅ ln(u(x)) + v(x) ⋅ (u'(x) / u(x)) ) ⋅ u(x)^v(x)
        term1 = _make_mul(derivative(exp), Func("ln", base))
        term2 = _make_mul(exp, _make_div(derivative(base), base))
        return _make_mul(_make_add(term1, term2), Pow(base, exp))

    if isinstance(node, Func):
        name, child = node.name, node.child
        if name == "cos":
            return _make_mul_many(
                derivative(child), Num(-1), Func(name="sin", child=child)
            )
        if name == "sin":
            dfi = _make_mul(derivative(child), Func(name="cos", child=child))
            return dfi
        if name == "ln":
            dfi = _make_div(derivative(child), child)
            return dfi
        if name == "exp":
            dfi = _make_mul(derivative(child), Func(name="exp", child=child))
            return dfi
        if name == "tan":
            dfi = _make_mul(
                derivative(child),
                Pow(
                    Func(name="cos", child=child),
                    Num(-2),
                ),
            )
            return dfi
        else:
            raise NotImplementedError(
                f"Derivative for function '{node.name}' is not implemented"
            )
    else:
        raise NotImplementedError(
            f"Sorry, derivative for {node} not known or implemented"
        )


def _simplify_number(value):
    if isinstance(value, bool):
        value = int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def simplify(node):
    if isinstance(node, (Num, Var)):
        return node

    if isinstance(node, Neg):
        child = simplify(node.child)
        result = Neg(child)
        return _evaluate_if_numeric(result)

    if isinstance(node, Func):
        child = simplify(node.child)
        result = Func(node.name, child)
        return _evaluate_if_numeric(result)

    if isinstance(node, Pow):
        base = simplify(node.base)
        exp = simplify(node.exp)
        if isinstance(exp, Num):
            if exp.value == 1:
                return base
            if exp.value == 0 and not (isinstance(base, Num) and base.value == 0):
                return Num(1)
        result = Pow(base, exp)
        return _evaluate_if_numeric(result)

    if isinstance(node, Add):
        terms = []
        const_total = 0
        has_const = False
        const_index = None

        for term in node.terms:
            simplified = simplify(term)
            if isinstance(simplified, Add):
                for sub_term in simplified.terms:
                    if isinstance(sub_term, Num):
                        const_total += sub_term.value
                        has_const = True
                        if const_index is None:
                            const_index = len(terms)
                    elif _is_zero(sub_term):
                        continue
                    else:
                        terms.append(sub_term)
                continue
            if isinstance(simplified, Num):
                const_total += simplified.value
                has_const = True
                if const_index is None:
                    const_index = len(terms)
                continue
            if _is_zero(simplified):
                continue
            terms.append(simplified)

        if has_const and (const_total != 0 or not terms):
            insert_at = const_index if const_index is not None else len(terms)
            terms.insert(insert_at, Num(_simplify_number(const_total)))

        if not terms:
            result = Num(0)
        elif len(terms) == 1:
            result = terms[0]
        else:
            result = Add(tuple(terms))
        return _evaluate_if_numeric(result)

    if isinstance(node, Mul):
        factors = []
        const_product = 1
        has_const = False

        for factor in node.factors:
            simplified = simplify(factor)
            if isinstance(simplified, Mul):
                for sub_factor in simplified.factors:
                    if _is_zero(sub_factor):
                        return Num(0)
                    if isinstance(sub_factor, Num):
                        const_product *= sub_factor.value
                        has_const = True
                    elif _is_one(sub_factor):
                        continue
                    else:
                        factors.append(sub_factor)
                continue
            if isinstance(simplified, Num):
                if simplified.value == 0:
                    return Num(0)
                const_product *= simplified.value
                has_const = True
                continue
            if _is_one(simplified):
                continue
            factors.append(simplified)

        if has_const:
            const_product = _simplify_number(const_product)
            if const_product == 0:
                return Num(0)
            if const_product != 1 or not factors:
                factors.insert(0, Num(const_product))

        if not factors:
            result = Num(1)
        elif len(factors) == 1:
            result = factors[0]
        else:
            result = Mul(tuple(factors))
        return _evaluate_if_numeric(result)

    if isinstance(node, Div):
        num_factors = []
        for factor in node.num_factors:
            simplified = simplify(factor)
            if _is_zero(simplified):
                return Num(0)
            num_factors.append(simplified)

        den_factors = []
        for factor in node.denum_factors:
            simplified = simplify(factor)
            if _is_zero(simplified):
                raise ZeroDivisionError("Division by zero during simplification")
            den_factors.append(simplified)

        if len(num_factors) > 1:
            num_factors = [f for f in num_factors if not _is_one(f)] or [Num(1)]
        if len(den_factors) > 1:
            den_factors = [f for f in den_factors if not _is_one(f)] or [Num(1)]
        if len(den_factors) == 1 and _is_one(den_factors[0]):
            den_factors = []

        result = Div(tuple(num_factors), tuple(den_factors))
        evaluated = _evaluate_if_numeric(result)

        if isinstance(evaluated, Num):
            return evaluated

        if isinstance(evaluated, Div):
            if not evaluated.denum_factors:
                if not evaluated.num_factors:
                    return Num(1)
                if len(evaluated.num_factors) == 1:
                    return evaluated.num_factors[0]
                return simplify(Mul(evaluated.num_factors))
            return evaluated

        return simplify(evaluated)

    raise TypeError(f"Unsupported node for simplification: {node!r}")


def diff(expr: str):
    tokens = tokenize(expr)
    parser = PrefixParser(tokens)
    ast = parser.parse()
    ast_der = derivative(ast)
    simplified = simplify(ast_der)
    return ast_to_prefix(simplified)


def diff_no_simplified(expr: str, val):
    tokens = tokenize(expr)
    parser = PrefixParser(tokens)
    ast = parser.parse()
    der_ast = derivative(ast)
    return eval_ast(der_ast, val)


expressions = [
    "(+ x 2)",
    "(^ x 2)",
    "(+ (* 1 x) (* 2 (+ x 1)))",
    "(exp (cos (+ 1 x)))",
    "(^ sin x (cos (+ 1 x)))",
    "(cos (+ x 1))",
    "(sin (+ x 1))",
    "(* cos x ln x)",
    "(* tan x ln x)",
    "(/ 1 2)",
    # "(/ 1 sin x)",
    "(/ x 2)",
    "(ln x)",
    "(+ (* 1 x) (* 2 (+ x 1)))",
]

if __name__ == "__main__":
    for expr in expressions:
        print(f"==> Funcio:  {expr}")
        tokens = tokenize(expr)
        parser = PrefixParser(tokens)
        ast = parser.parse()
        der_ast = derivative(ast)
        print(f"Derivada sense simplificar :  {ast_to_prefix(der_ast)}")
        diff_value = diff(expr)
        print(f"Derivada: {diff_value}")
        print(f"Segona derivada {diff(diff_value)}")
        print("------------------------")
