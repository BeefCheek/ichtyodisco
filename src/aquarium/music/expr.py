"""Tiny, safe expression language for scores: ``"clip(fish_count / 6, 0.2, 1) * (1 - mean_z)"``.

Only arithmetic, comparisons, ``and/or/not``, ``x if c else y``, numbers / strings and a
whitelist of functions are allowed: a score file is data, it can never run arbitrary code.
Unknown names evaluate to 0, so a score keeps working when a metric is not available yet
(e.g. depth disabled).
"""

from __future__ import annotations

import ast
import logging
import math
from typing import Any

log = logging.getLogger(__name__)


def _clip(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if v < lo else hi if v > hi else v


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _step(edge: float, v: float) -> float:
    return 1.0 if v >= edge else 0.0


def _smoothstep(lo: float, hi: float, v: float) -> float:
    if hi == lo:
        return _step(lo, v)
    x = _clip((v - lo) / (hi - lo))
    return x * x * (3 - 2 * x)


FUNCTIONS: dict[str, Any] = {
    "min": min, "max": max, "abs": abs, "round": round, "clip": _clip, "lerp": _lerp,
    "step": _step, "smoothstep": _smoothstep, "sqrt": lambda v: math.sqrt(max(v, 0.0)),
    "exp": math.exp, "log": lambda v: math.log(max(v, 1e-9)),
}

_ALLOWED = (
    ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.FloorDiv, ast.Mod, ast.Pow, ast.UnaryOp, ast.Not, ast.USub, ast.UAdd, ast.Compare,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.IfExp, ast.Call,
    ast.Name, ast.Load, ast.Constant, ast.Tuple, ast.List,
)


class ExprError(ValueError):
    pass


class _Env(dict):
    def __missing__(self, key: str) -> float:
        return 0.0


class Expr:
    def __init__(self, source: str | float | int | bool):
        self.source = str(source)
        if isinstance(source, bool | int | float):
            self._const: Any = source
            self._code = None
            return
        self._const = None
        try:
            tree = ast.parse(self.source.strip(), mode="eval")
        except SyntaxError as exc:
            raise ExprError(f"invalid expression {self.source!r}: {exc.msg}") from exc
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED):
                raise ExprError(f"{type(node).__name__} not allowed in {self.source!r}")
            if isinstance(node, ast.Call):
                if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
                    raise ExprError(f"only {', '.join(sorted(FUNCTIONS))} can be called: {self.source!r}")
                if node.keywords:
                    raise ExprError(f"keyword arguments not allowed: {self.source!r}")
            if isinstance(node, ast.Name) and node.id.startswith("_"):
                raise ExprError(f"names starting with '_' are not allowed: {self.source!r}")
        self._code = compile(tree, "<score>", "eval")
        self._warned = False

    def __call__(self, env: dict[str, Any]) -> Any:
        if self._code is None:
            return self._const
        scope = _Env(env)
        scope.update(FUNCTIONS)
        try:
            return eval(self._code, {"__builtins__": {}}, scope)  # noqa: S307 - validated AST
        except (ArithmeticError, TypeError, ValueError) as exc:
            if not self._warned:
                log.warning("score expression %r failed: %s (using 0)", self.source, exc)
                self._warned = True
            return 0.0

    def num(self, env: dict[str, Any], default: float = 0.0) -> float:
        v = self(env)
        try:
            f = float(v)
        except (TypeError, ValueError):
            return default
        return f if math.isfinite(f) else default

    def __repr__(self) -> str:
        return f"Expr({self.source!r})"
