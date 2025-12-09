#! /usr/bin/env python
import atexit
import os
import re
import sys
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

try:
    import readline
except ImportError:  # pragma: no cover
    readline = None


NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
VAR_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
FN_DECL_RE = re.compile(
    r"""
    ^\s*fn\s+
    (?P<name>[A-Za-z_][A-Za-z0-9_]*)       
    (?P<params>(?:\s+[A-Za-z_][A-Za-z0-9_]*)*) 
    \s*=>\s*
    (?P<body>.+)\s*$
    """,
    re.VERBOSE,
)


CALL_ARG_STOP = {"+", "-", "*", "/", "%", "^", "**", "=", "(", ")", ","}


class ASTNode(Iterable):
    pass


@dataclass
class ExecutionContext:
    variables: Dict[str, Union[int, float]] = field(default_factory=dict)
    functions: Dict[str, "FuncDef"] = field(default_factory=dict)
    labels: Optional[Dict[str, int]] = field(default_factory=dict)
    last_value: Union[int, float, str, None] = None
    error_flag: bool = False
    parent: Optional["ExecutionContext"] = None


class NodeVisitor:
    def visit(self, node: ASTNode) -> Any:
        method = f"visit_{type(node).__name__}"
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node)

    def generic_visit(self, node: ASTNode) -> Any:
        result = None
        for child in node:
            result = self.visit(child)
        return result


class ContainsVarVisitor(NodeVisitor):
    def __init__(self):
        self.found = False

    def visit_Var(self, node: "Var") -> bool:
        self.found = True
        return True

    def generic_visit(self, node: ASTNode) -> Any:
        if self.found:
            return True
        return super().generic_visit(node)


class FunctionBodyValidator(NodeVisitor):
    def __init__(self, allowed: Iterable[str], functions: Dict[str, "FuncDef"]):
        self.allowed = set(allowed)
        self.functions = functions

    def visit_Var(self, node: "Var") -> None:
        if node.name not in self.allowed and node.name not in self.functions:
            msg = f"ERROR: Invalid identifier '{node.name}' in function body."
            raise ValueError(msg)

    def visit_FuncCall(self, node: "FuncCall") -> None:
        for arg in node.args:
            self.visit(arg)


class Evaluator(NodeVisitor):
    def __init__(self, ctx: ExecutionContext):
        self.ctx: ExecutionContext = ctx

    def _root_ctx(self) -> ExecutionContext:
        ctx = self.ctx
        while ctx.parent is not None:
            ctx = ctx.parent
        return ctx

    def visit_Num(self, node):
        return node.value

    def visit_Var(self, node: "Var") -> Union[int, float]:
        try:
            return self.ctx.variables[node.name]
        except KeyError:
            msg = f"ERROR: Invalid identifier. No variable with name '{node.name}' was found."
            sys.exit(msg)

    def visit_Add(self, node):
        return sum(self.visit(term) for term in node.terms)

    def visit_Mul(self, node):
        result = 1
        for factor in node.factors:
            result *= self.visit(factor)
        return result

    def visit_Div(self, node):
        num, den = next(iter(node))
        num_val = 1
        for factor in num:
            num_val *= self.visit(factor)
        den_val = 1
        for factor in den:
            den_val *= self.visit(factor)
        return num_val / den_val

    def visit_Mod(self, node: "Mod") -> Union[int, float]:
        return self.visit(node.left) % self.visit(node.right)

    def visit_Pow(self, node):
        return self.visit(node.base) ** self.visit(node.exp)

    def visit_Neg(self, node):
        return -self.visit(node.child)

    def visit_Func(self, node):
        arg = self.visit(node.child)
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

    def visit_Assign(self, node: "Assign") -> Union[int, float]:
        target, expr = node.var, node.expr
        if not isinstance(target, Var):
            raise SyntaxError("Left side of assignment must be a variable")
        root_ctx = self._root_ctx()
        is_new_var = target.name not in self.ctx.variables
        if is_new_var and target.name in root_ctx.functions:
            msg = f"ERROR: Name conflict. Cannot assign to '{target.name}' because it is a function."
            raise ValueError(msg)
        value = self.visit(expr)
        self.ctx.variables[target.name] = value
        return value

    def visit_FuncDef(self, node: "FuncDef"):
        root_ctx = self._root_ctx()
        if node.name in root_ctx.variables:
            msg = f"ERROR: Name conflict. Cannot declare function '{node.name}' because a variable with the same name already exists."
            raise ValueError(msg)
        seen_params: set[str] = set()
        for param in node.param:
            if param.name in seen_params:
                msg = f"ERROR: Duplicate parameter name '{param.name}' in function '{node.name}'."
                raise ValueError(msg)
            seen_params.add(param.name)
        allowed_names = {param.name for param in node.param}
        allowed_names.add(node.name)
        try:
            FunctionBodyValidator(allowed_names, root_ctx.functions).visit(node.body)
        except ValueError as exc:
            sys.exit(str(exc))
        root_ctx.functions[node.name] = node
        if self.ctx is not root_ctx:
            self.ctx.functions[node.name] = node
        return ""

    def visit_FuncCall(self, node: "FuncCall"):
        fct = self.ctx.functions.get(node.name)
        if fct is None:
            raise ValueError(f"Undefined function: {node.name}")
        if len(fct.param) != len(node.args):
            raise ValueError(
                f"Function '{node.name}' expects {len(fct.param)} arguments, got {len(node.args)}"
            )
        variable_mapping = {
            param.name: self.visit(arg) for param, arg in zip(fct.param, node.args)
        }
        function_mapping = self.ctx.functions.copy()
        evaluator = Evaluator(
            ctx=ExecutionContext(
                variables=variable_mapping,
                functions=function_mapping,
                parent=self.ctx,
            )
        )
        return evaluator.visit(fct.body)


@dataclass(frozen=True)
class FuncDef(ASTNode):
    name: str
    param: List["Var"]
    body: ASTNode

    def __iter__(self):
        yield self.body


@dataclass(frozen=True)
class FuncCall(ASTNode):
    name: str
    args: List[ASTNode]

    def __iter__(self):
        for arg in self.args:
            yield arg


@dataclass(frozen=True)
class Num(ASTNode):
    value: Union[int, float]

    def __iter__(self):
        return iter(())


@dataclass(frozen=True)
class Var(ASTNode):
    name: str = "x"

    def __iter__(self):
        return iter(())


@dataclass(frozen=True)
class Assign(ASTNode):
    var: "Var"
    expr: Union["Assign", ASTNode]

    def __iter__(self):
        return iter(())


@dataclass(frozen=True)
class Add(ASTNode):
    terms: Tuple[object, ...]

    def __iter__(self):
        return iter(self.terms)


@dataclass(frozen=True)
class Mul(ASTNode):
    factors: Tuple[object, ...]

    def __iter__(self):
        return iter(self.factors)


@dataclass(frozen=True)
class Div(ASTNode):
    num_factors: Tuple[object, ...]
    denum_factors: Tuple[object, ...]

    def __iter__(self):
        yield self.num_factors, self.denum_factors


@dataclass(frozen=True)
class Mod(ASTNode):
    left: ASTNode
    right: ASTNode

    def __iter__(self):
        yield self.left
        yield self.right


@dataclass(frozen=True)
class Neg(ASTNode):
    child: object

    def __iter__(self):
        yield self.child


@dataclass(frozen=True)
class Pow(ASTNode):
    base: object
    exp: object

    def __iter__(self):
        yield self.base
        yield self.exp


@dataclass(frozen=True)
class Func(ASTNode):
    name: str
    child: object

    def __iter__(self):
        yield self.child


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


class InfixParser:
    def __init__(self, tokens, ctx: ExecutionContext):
        self.tokens = tokens
        self.ctx = ctx
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
        func_def = self._maybe_func_def()
        if func_def is not None:
            if self.peek() is not None:
                raise SyntaxError(f"Unexpected trailing token: {self.peek()!r}")
            return func_def
        assignment = self._maybe_assignment()
        if assignment is not None:
            if self.peek() is not None:
                raise SyntaxError(f"Unexpected trailing token: {self.peek()!r}")
            return assignment
        expr = self._expr()
        if self.peek() is not None:
            raise SyntaxError(f"Unexpected trailing token: {self.peek()!r}")
        return expr

    def _maybe_func_def(self):
        tok = self.peek()
        if tok == "fn" and (
            matches := FN_DECL_RE.fullmatch(" ".join(self.tokens[self.i :]))
        ):
            self.take("fn")
            fn_name = self.take(matches.group("name"))
            if fn_name is None:
                raise SyntaxError("Expected function name after 'fn'")
            params = matches.group("params").split()
            if params:
                params = [Var(str(self.take(p))) for p in params]
            else:
                params = []
            self.take("=>")
            body = self._expr()
            fdef = FuncDef(fn_name, param=params, body=body)
            return fdef

    def _maybe_assignment(self):
        tok = self.peek()
        if tok and VAR_RE.fullmatch(tok) and self._peek_next() == "=":
            name_token = self.take()
            if name_token is None:
                raise SyntaxError("Expected variable name before '='")
            if not VAR_RE.fullmatch(name_token):
                raise SyntaxError(f"Invalid variable name: {name_token!r}")
            self.take("=")
            nested = self._maybe_assignment()
            if nested is not None:
                expr = nested
            else:
                expr = self._expr()
            return Assign(Var(name_token), expr)
        return self._expr()

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
            if tok not in {"*", "/", "%"}:
                break
            op = self.take()
            right = self._power()
            if op == "*":
                node = _make_mul(node, right)
            elif op == "/":
                node = _make_div(node, right)
            else:
                node = Mod(node, right)
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
            arg = self._maybe_assignment()
            self.take(")")
            return Func(name_token, arg)

        if tok == "(":
            self.take("(")
            node = self._maybe_assignment()
            self.take(")")
            return node

        if tok and NUMBER_RE.fullmatch(tok):
            self.take()
            value = float(tok) if "." in tok else int(tok)
            return Num(value)

        if tok and VAR_RE.fullmatch(tok):
            name_token = self.take()
            assert name_token is not None
            args: List[ASTNode] = []
            if name_token not in self.ctx.functions:
                return Var(name_token)
            arg_count = 0
            while arg_count < self.ctx.functions[name_token].param.__len__():
                next_tok = self.peek()
                if next_tok is None or next_tok in CALL_ARG_STOP:
                    break
                arg = self._maybe_assignment()
                args.append(arg)
                arg_count += 1
            return FuncCall(name_token, args)

        raise SyntaxError(f"Unexpected token in expression: {tok!r}")


def tokenize(expression):
    if expression == "":
        return []

    regex = re.compile(
        r"\s*(=>|\*\*|[-+*\/\%\^=\(\)]|[A-Za-z_][A-Za-z0-9_]*|[0-9]*\.?[0-9]+)\s*"
    )
    tokens = regex.findall(expression)
    return [s for s in tokens if not s.isspace()]


class Interpreter:
    def __init__(self, variables: Optional[Dict[str, Union[int, float]]] = None):
        self.vars: Dict[str, Union[int, float]] = variables or {}
        self.ctx = ExecutionContext(variables=self.vars)

    def input(self, expression):
        tokens = tokenize(expression)
        if not tokens:
            return ""
        ast = InfixParser(tokens, self.ctx).parse()
        evaluator = Evaluator(self.ctx)
        evaluated = evaluator.visit(ast)
        print(evaluated)
        return evaluated


if __name__ == "__main__":
    if readline:
        readline.parse_and_bind("tab: complete")
        history_path = os.path.expanduser("~/.simple_interpreter_history")
        try:
            readline.read_history_file(history_path)
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            corrupted_path = f"{history_path}.corrupt"
            try:
                os.replace(history_path, corrupted_path)
            except OSError:
                try:
                    os.remove(history_path)
                except OSError:
                    pass

        def _persist_history():
            if readline is None:
                return
            try:
                readline.write_history_file(history_path)
            except OSError:
                pass

        atexit.register(_persist_history)

    interpreter = Interpreter()
    while True:
        try:
            expression = input(">>> ")
            if expression.strip().lower() in {"exit", "quit"}:
                break
            interpreter.input(expression)
        except EOFError:
            break
