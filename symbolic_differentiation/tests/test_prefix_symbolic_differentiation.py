import math

import pytest

from prefix_symbolic_differentiation import (
    _make_mul,
    Add,
    Div,
    Func,
    Mul,
    Num,
    Pow,
    PrefixParser,
    Var,
    ast_to_prefix,
    derivative,
    eval_ast,
    tokenize,
    simplify,
)


class TestMakeMul:
    def test_merges_mul_factors(self):
        left = Mul((Var("x"), Var("y")))
        right = Var("z")

        result = _make_mul(left, right)

        assert isinstance(result, Mul)
        assert result.factors == (Var("x"), Var("y"), Var("z"))

    def test_mul_with_left_div_adds_factor_to_numerator(self):
        left = Div((Var("a"),), (Var("b"),))
        right = Var("c")

        result = _make_mul(left, right)

        assert isinstance(result, Div)
        assert result.num_factors == (Var("a"), Var("c"))
        assert result.denum_factors == (Var("b"),)

    def test_mul_with_right_div_adds_factor_to_numerator(self):
        left = Var("c")
        right = Div((Var("a"),), (Var("b"),))

        result = _make_mul(left, right)

        assert isinstance(result, Div)
        assert result.num_factors == (Var("c"), Var("a"))
        assert result.denum_factors == (Var("b"),)

    def test_mul_of_two_divs_merges_numerator_and_denominator(self):
        left = Div((Var("a"),), (Var("b"),))
        right = Div((Var("c"),), (Var("d"),))

        result = _make_mul(left, right)

        assert isinstance(result, Div)
        assert result.num_factors == (Var("a"), Var("c"))
        assert result.denum_factors == (Var("b"), Var("d"))

    def test_parser_handles_negative_numbers(self):
        tokens = tokenize("(* 2 -4)")
        parser = PrefixParser(tokens)

        ast = parser.parse()

        assert isinstance(ast, Mul)
        assert ast.factors == (Num(2), Num(-4))

    def test_parser_multiplies_adjacent_expressions(self):
        tokens = tokenize("(* 2 -4)(ln x)")
        parser = PrefixParser(tokens)

        ast = parser.parse()

        assert isinstance(ast, Mul)
        assert ast.factors == (Num(2), Num(-4), Func("ln", Var("x")))


class TestEvalDiv:
    def test_evaluates_numeric_division(self):
        node = Div((Num(6),), (Num(3),))

        assert eval_ast(node, 0) == 2

    def test_evaluates_division_with_variable(self):
        node = Div((Var("x"), Num(2)), (Num(4),))

        assert eval_ast(node, 8) == 4


class TestDerivativeDiv:
    def test_derivative_of_exp_over_var(self):
        node = Div((Func("exp", Var("x")),), (Var("x"),))

        derivative_node = derivative(node)

        value = eval_ast(derivative_node, 2)
        expected = math.exp(2) * (2 - 1) / (2**2)
        assert value == pytest.approx(expected)

    def test_derivative_of_cos_over_exp(self):
        numerator = Func("cos", Add((Var("x"), Num(1))))
        denominator = Func("exp", Var("x"))
        node = Div((numerator,), (denominator,))

        derivative_node = derivative(node)

        value = eval_ast(derivative_node, 0)
        expected = (-math.sin(1) - math.cos(1)) / math.exp(0)
        assert value == pytest.approx(expected)


class TestDerivativePrefix:
    def _derivative_prefix(self, expr: str) -> str:
        tokens = tokenize(expr)
        parser = PrefixParser(tokens)
        ast = parser.parse()
        return ast_to_prefix(derivative(ast))

    def test_prefix_of_linear_plus_constant(self):
        assert self._derivative_prefix("(+ x 2)") == "(+ 1 0)"

    def test_prefix_of_weighted_linear_combo(self):
        expected = "(+ (+ 1) (+ (* (+ 1 0) 2)))"
        assert self._derivative_prefix("(+ (* 1 x) (* 2 (+ x 1)))") == expected

    def test_prefix_of_cos_shifted(self):
        assert (
            self._derivative_prefix("(cos (+ 1 x))") == "(* (+ 0 1) -1 (sin (+ 1 x)))"
        )

    def test_prefix_of_cos(self):
        assert self._derivative_prefix("(cos x)") == "(* -1 (sin x))"

    def test_prefix_of_pow(self):
        assert self._derivative_prefix("(^ x 2)") == "(* 2 (^ x 1))"

    def test_prefix_of_ln(self):
        assert self._derivative_prefix("(ln x)") == "(/ 1 x)"


class TestSimplify:
    def test_add_combines_constants_and_drops_zero(self):
        node = Add((Num(2), Var("x"), Num(3), Num(0)))

        simplified = simplify(node)

        assert simplified == Add((Num(5), Var("x")))

    def test_mul_collapses_zero_factor(self):
        node = Mul((Var("x"), Num(0), Func("sin", Var("x"))))

        simplified = simplify(node)

        assert simplified == Num(0)

    def test_mul_combines_constants_and_removes_one(self):
        node = Mul((Num(1), Num(2), Num(3), Var("x")))

        simplified = simplify(node)

        assert simplified == Mul((Num(6), Var("x")))

    def test_numeric_division_folds_to_number(self):
        node = Div(
            (Mul((Num(2),)),),
            (Mul((Pow(Num(2), Num(2)),)),),
        )

        simplified = simplify(node)

        assert simplified == Num(0.5)

    def test_exp_of_numeric_remains_symbolic(self):
        node = Func("exp", Num(-1))

        simplified = simplify(node)

        assert simplified == Func("exp", Num(-1))

    def test_division_keeps_unit_numerator(self):
        node = Div((Num(1),), (Var("x"),))

        simplified = simplify(node)

        assert simplified == Div((Num(1),), (Var("x"),))

    def test_simplify_derivative_of_ln(self):
        tokens = tokenize("(ln x)")
        parser = PrefixParser(tokens)
        ast = parser.parse()

        simplified_derivative = simplify(derivative(ast))

        assert simplified_derivative == Div((Num(1),), (Var("x"),))

    def test_pow_with_exponent_one_reduces_to_base(self):
        node = Pow(Var("x"), Num(1))

        simplified = simplify(node)

        assert simplified == Var("x")

    def test_add_preserves_operand_order_when_constant_last(self):
        node = Add((Var("x"), Num(1)))

        simplified = simplify(node)

        assert simplified == Add((Var("x"), Num(1)))


class TestDerivativeScenarios:
    @staticmethod
    def _assert_derivative(expr: str, expected_prefix: str, order: int = 1):
        tokens = tokenize(expr)
        parser = PrefixParser(tokens)
        ast = parser.parse()

        result = ast
        for _ in range(order):
            result = derivative(result)
        simplified = simplify(result)
        assert ast_to_prefix(simplified) == expected_prefix

    @pytest.mark.parametrize(
        "expected, expr",
        [
            ("0", "5"),
            ("1", "x"),
            ("2", "(+ x x)"),
            ("0", "(- x x)"),
            ("2", "(* x 2)"),
            ("0.5", "(/ x 2)"),
            ("(* 2 x)", "(^ x 2)"),
            ("(* -1 (sin x))", "(cos x)"),
            ("(cos x)", "(sin x)"),
            ("(exp x)", "(exp x)"),
            ("(/ 1 x)", "(ln x)"),
        ],
    )
    def test_first_derivative_cases(self, expected, expr):
        self._assert_derivative(expr, expected)

    @pytest.mark.parametrize(
        "expected, expr",
        [
            ("3", "(+ x (+ x x))"),
            ("1", "(- (+ x x) x)"),
            ("2", "(* 2 (+ x 2))"),
            ("(/ -2 (^ (+ 1 x) 2))", "(/ 2 (+ 1 x))"),
            ("(* -1 (sin (+ x 1)))", "(cos (+ x 1))"),
            ("(cos (+ x 1))", "(sin (+ x 1))"),
            ("(* 2 (cos (* 2 x)))", "(sin (* 2 x))"),
            ("(* 2 (exp (* 2 x)))", "(exp (* 2 x))"),
        ],
    )
    def test_first_derivative_composite_cases(self, expected, expr):
        self._assert_derivative(expr, expected)

    @pytest.mark.parametrize(
        "expected, expr",
        [
            ("(* -1 (sin x))", "(sin x)"),
            ("(exp x)", "(exp x)"),
        ],
    )
    def test_second_derivative_cases(self, expected, expr):
        self._assert_derivative(expr, expected, order=2)
