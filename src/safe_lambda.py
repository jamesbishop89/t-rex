"""
Restricted compiler for configuration lambdas.

The reconciliation configs rely on a small subset of Python for pandas/numpy
expressions. This module validates that subset and evaluates lambdas with a
locked-down global namespace rather than unrestricted eval().
"""

from __future__ import annotations

import ast
import re
from functools import lru_cache
from typing import Any, Callable, Dict, Set

import numpy as np
import pandas as pd


class UnsafeExpressionError(ValueError):
    """Raised when a config expression uses unsupported or unsafe syntax."""


_ALLOWED_GLOBALS: Dict[str, Any] = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "filter": filter,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "list": list,
    "max": max,
    "min": min,
    "next": next,
    "np": np,
    "pd": pd,
    "re": re,
    "round": round,
    "set": set,
    "str": str,
    "sum": sum,
    "tuple": tuple,
}

_SAFE_BUILTINS: Dict[str, Any] = {
    # Pandas/numpy internals can require import access while evaluating
    # otherwise-safe expressions such as Timestamp.strftime().
    "__import__": __import__,
}

_ALLOWED_ROOT_ATTRS: Dict[str, Set[str]] = {
    "np": {"nan", "round", "where"},
    "pd": {"concat", "to_datetime", "to_numeric"},
    "re": {"match"},
    "str": {"isalnum"},
}

_ALLOWED_METHODS: Set[str] = {
    "astype",
    "combine_first",
    "extract",
    "fillna",
    "get",
    "groupby",
    "items",
    "join",
    "match",
    "notna",
    "replace",
    "round",
    "strftime",
    "strip",
    "sum",
    "transform",
    "upper",
    "lower",
}

_ALLOWED_PROPERTIES: Set[str] = {
    "date",
    "dt",
    "nan",
    "str",
}

_ALLOWED_CALLABLE_NAMES: Set[str] = {
    name for name, value in _ALLOWED_GLOBALS.items() if callable(value)
}

_ALLOWED_NODE_TYPES = (
    ast.Add,
    ast.And,
    ast.Attribute,
    ast.BinOp,
    ast.BoolOp,
    ast.Call,
    ast.Compare,
    ast.Constant,
    ast.Dict,
    ast.Eq,
    ast.Expression,
    ast.GeneratorExp,
    ast.Gt,
    ast.GtE,
    ast.IfExp,
    ast.In,
    ast.Is,
    ast.IsNot,
    ast.keyword,
    ast.Lambda,
    ast.List,
    ast.Load,
    ast.Lt,
    ast.LtE,
    ast.Mod,
    ast.Mult,
    ast.Name,
    ast.Not,
    ast.NotEq,
    ast.NotIn,
    ast.Or,
    ast.Pow,
    ast.Slice,
    ast.Store,
    ast.Sub,
    ast.Subscript,
    ast.Tuple,
    ast.UAdd,
    ast.UnaryOp,
    ast.USub,
    ast.Div,
    ast.comprehension,
)


class _SafeLambdaValidator(ast.NodeVisitor):
    """Validates a config lambda against the supported expression subset."""

    def __init__(self) -> None:
        self._lambda_depth = 0
        self._local_names: Set[str] = set()

    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise UnsafeExpressionError(
                f"Unsupported expression element: {node.__class__.__name__}"
            )
        super().generic_visit(node)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        if self._lambda_depth > 0:
            raise UnsafeExpressionError("Nested lambda expressions are not allowed")

        if (
            len(node.args.args) != 1
            or node.args.defaults
            or node.args.kw_defaults
            or node.args.kwonlyargs
            or node.args.posonlyargs
            or node.args.vararg is not None
            or node.args.kwarg is not None
        ):
            raise UnsafeExpressionError("Only single-argument lambdas are supported")

        arg_name = node.args.args[0].arg
        if arg_name.startswith("_"):
            raise UnsafeExpressionError("Lambda parameter names cannot start with '_'")

        previous_locals = set(self._local_names)
        self._local_names.add(arg_name)
        self._lambda_depth += 1
        self.visit(node.body)
        self._lambda_depth -= 1
        self._local_names = previous_locals

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("_"):
            raise UnsafeExpressionError(f"Unsupported name: {node.id}")

        if isinstance(node.ctx, ast.Load):
            if node.id not in self._local_names and node.id not in _ALLOWED_GLOBALS:
                raise UnsafeExpressionError(f"Unsupported name: {node.id}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.visit(node.value)

        if node.attr.startswith("_"):
            raise UnsafeExpressionError(f"Unsupported attribute access: {node.attr}")

        if isinstance(node.value, ast.Name) and node.value.id in _ALLOWED_ROOT_ATTRS:
            allowed_attrs = _ALLOWED_ROOT_ATTRS[node.value.id]
            if node.attr not in allowed_attrs:
                raise UnsafeExpressionError(
                    f"Unsupported attribute access: {node.value.id}.{node.attr}"
                )
            return

        if node.attr not in _ALLOWED_METHODS and node.attr not in _ALLOWED_PROPERTIES:
            raise UnsafeExpressionError(f"Unsupported attribute access: {node.attr}")

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func

        if isinstance(func, ast.Name):
            if func.id not in _ALLOWED_CALLABLE_NAMES:
                raise UnsafeExpressionError(f"Unsupported function call: {func.id}")
        elif isinstance(func, ast.Attribute):
            self.visit(func)

            if isinstance(func.value, ast.Name) and func.value.id in _ALLOWED_ROOT_ATTRS:
                if func.attr not in _ALLOWED_ROOT_ATTRS[func.value.id]:
                    raise UnsafeExpressionError(
                        f"Unsupported function call: {func.value.id}.{func.attr}"
                    )
            elif func.attr not in _ALLOWED_METHODS:
                raise UnsafeExpressionError(f"Unsupported method call: {func.attr}")
        else:
            raise UnsafeExpressionError("Unsupported callable expression")

        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            if keyword.arg and keyword.arg.startswith("_"):
                raise UnsafeExpressionError(
                    f"Unsupported keyword argument: {keyword.arg}"
                )
            self.visit(keyword.value)

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self.visit(node.iter)
        self._bind_target(node.target)
        for condition in node.ifs:
            self.visit(condition)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        previous_locals = set(self._local_names)
        for generator in node.generators:
            self.visit(generator)
        self.visit(node.elt)
        self._local_names = previous_locals

    def _bind_target(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            if target.id.startswith("_"):
                raise UnsafeExpressionError(
                    f"Unsupported comprehension target: {target.id}"
                )
            self._local_names.add(target.id)
            return

        if isinstance(target, (ast.List, ast.Tuple)):
            for element in target.elts:
                self._bind_target(element)
            return

        raise UnsafeExpressionError("Unsupported comprehension target")


@lru_cache(maxsize=256)
def compile_config_lambda(lambda_str: str) -> Callable[[Any], Any]:
    """
    Compile a configuration lambda after validating its AST.

    The compiled lambda is cached because the same config expressions are used
    repeatedly during one reconciliation run.
    """
    normalized_lambda = re.sub(
        r"__import__\((['\"])re\1\)",
        "re",
        lambda_str,
    )

    try:
        tree = ast.parse(normalized_lambda, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError(f"Invalid lambda syntax: {exc}") from exc

    if not isinstance(tree.body, ast.Lambda):
        raise UnsafeExpressionError("Expression must be a lambda")

    validator = _SafeLambdaValidator()
    validator.visit(tree.body)

    safe_globals = {"__builtins__": _SAFE_BUILTINS, **_ALLOWED_GLOBALS}
    compiled = eval(compile(tree, "<config_lambda>", "eval"), safe_globals, {})

    if not callable(compiled):
        raise UnsafeExpressionError("Expression did not compile to a callable")

    return compiled
