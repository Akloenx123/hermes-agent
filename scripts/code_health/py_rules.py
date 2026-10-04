"""Hermes-specific AST rules (the ``HX`` ids in ``config.RULES``).

Each checker is a small function ``(tree, ctx) -> iterable of line numbers``; the table at the
bottom maps rule ids to checkers. Checkers favour precision: a rule that cries wolf gets an
allow comment on every hit and stops meaning anything.
"""

from __future__ import annotations

import ast
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field

_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)
_CAPTURE_CALLS = {
    "get_hermes_home",
    "display_hermes_home",
    "load_config",
    "load_config_readonly",
    "read_raw_config",
    "get_secret",
    "getcwd",
    "expanduser",
    "_float_env",
    "_int_env",
    "_bool_env",
    "_env_float",
    "_env_int",
    "_env_bool",
}
_SYNC_CONFIG_CALLS = {"load_config", "save_config", "read_raw_config", "load_config_readonly"}
_SUBPROCESS_WAITS = {"run", "call", "check_call", "check_output"}
# health: allow HX003 -- the detector's own pattern list
_SHELL_IDENTITY = ("pgrep -f", "ps aux", "ps -ef", "ps -eo")


@dataclass
class Ctx:
    known_env: set[str] = field(default_factory=set)


def _dotted(node: ast.AST) -> str:
    """``os.environ.get`` for an Attribute chain, ``""`` for anything else."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    if isinstance(node, ast.Call):
        inner = _dotted(node.func)
        return ".".join([f"{inner}()", *reversed(parts)]) if inner else ""
    return ""


def _call_name(call: ast.Call) -> str:
    return _dotted(call.func)


def _str_arg(call: ast.Call, index: int = 0) -> str | None:
    if len(call.args) > index:
        arg = call.args[index]
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
    return None


def _has_kw(call: ast.Call, name: str) -> bool:
    return any(kw.arg in (name, None) for kw in call.keywords)


def _env_read_name(node: ast.AST) -> str | None:
    """Env var name for ``os.getenv("X")`` / ``os.environ.get("X")`` / ``os.environ["X"]``."""
    if isinstance(node, ast.Call):
        if _call_name(node) in ("os.getenv", "os.environ.get", "environ.get", "getenv"):
            return _str_arg(node)
        return None
    if isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Load):
        if _dotted(node.value) in ("os.environ", "environ"):
            key = node.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                return key.value
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        if isinstance(node.ops[0], (ast.In, ast.NotIn)):
            if _dotted(node.comparators[0]) in ("os.environ", "environ"):
                left = node.left
                if isinstance(left, ast.Constant) and isinstance(left.value, str):
                    return left.value
    return None


def _walk_skipping(node: ast.AST, skip: tuple[type, ...]) -> Iterator[ast.AST]:
    """``ast.walk`` that does not descend into ``skip`` node types (except the root)."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        for child in ast.iter_child_nodes(current):
            if not isinstance(child, skip):
                stack.append(child)


def hardcoded_home(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            right = node.right
            if (
                isinstance(node.left, ast.Call)
                and _call_name(node.left).endswith("Path.home")
                and isinstance(right, ast.Constant)
                and isinstance(right.value, str)
                and right.value.strip("/").startswith(".hermes")
            ):
                yield node.lineno
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            arg = _str_arg(node)
            if arg and arg.startswith("~/.hermes") and name.endswith(("expanduser", "Path")):
                yield node.lineno


def new_env_var(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    for node in ast.walk(tree):
        name = _env_read_name(node)
        if name and name.startswith("HERMES_") and name not in ctx.known_env:
            yield getattr(node, "lineno", 0)


def argv_identity(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            if isinstance(node.ops[0], (ast.In, ast.NotIn)):
                left = node.left
                target = ast.unparse(node.comparators[0])
                if isinstance(left, ast.Constant) and isinstance(left.value, str):
                    if "cmdline" in target:
                        yield node.lineno
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if any(cmd in node.value for cmd in _SHELL_IDENTITY):
                yield node.lineno


def unscoped_secret_fallback(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler) or node.type is None:
            continue
        if "UnscopedSecretError" not in ast.unparse(node.type):
            continue
        for stmt in node.body:
            if any(_env_read_name(inner) for inner in ast.walk(stmt)):
                yield node.lineno
                break


def _capture_lines(expr: ast.AST) -> Iterator[int]:
    for node in _walk_skipping(expr, (ast.Lambda, *_FUNCS)):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            leaf = name.rsplit(".", 1)[-1]
            if leaf in _CAPTURE_CALLS or name.endswith("Path.home"):
                yield node.lineno
                return
        if _env_read_name(node):
            yield getattr(node, "lineno", 0)
            return


def _is_main_guard(stmt: ast.stmt) -> bool:
    return isinstance(stmt, ast.If) and "__main__" in ast.unparse(stmt.test)


def _import_time_statements(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """Statements that run at import: module/class bodies and their if/try/with blocks."""
    for stmt in body:
        if _is_main_guard(stmt):
            continue
        yield stmt
        if isinstance(stmt, ast.ClassDef):
            yield from _import_time_statements(stmt.body)
        elif isinstance(stmt, (ast.If, ast.Try, ast.With)):
            for name in ("body", "orelse", "finalbody"):
                yield from _import_time_statements(getattr(stmt, name, None) or [])
            for handler in getattr(stmt, "handlers", None) or []:
                yield from _import_time_statements(handler.body)


def import_time_capture(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    for stmt in _import_time_statements(tree.body):
        if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and stmt.value:
            yield from _capture_lines(stmt.value)
        elif isinstance(stmt, _FUNCS):
            for default in [*stmt.args.defaults, *stmt.args.kw_defaults]:
                if default is not None:
                    yield from _capture_lines(default)


def missing_timeout(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _has_kw(node, "timeout"):
            continue
        name = _call_name(node)
        head, _, leaf = name.rpartition(".")
        if head == "subprocess" and leaf in _SUBPROCESS_WAITS:
            yield node.lineno
        elif leaf == "urlopen" and len(node.args) < 3:
            yield node.lineno
        elif leaf == "communicate" and head and not node.args:
            yield node.lineno


def sync_config_in_async(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    for func in ast.walk(tree):
        if not isinstance(func, ast.AsyncFunctionDef):
            continue
        for stmt in func.body:
            for node in _walk_skipping(stmt, (ast.Lambda, *_FUNCS)):
                if isinstance(node, ast.Call):
                    if _call_name(node).rsplit(".", 1)[-1] in _SYNC_CONFIG_CALLS:
                        yield node.lineno


def get_event_loop(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) == "asyncio.get_event_loop":
            yield node.lineno


def _gathers_exceptions(func: ast.AST) -> bool:
    for node in ast.walk(func):
        if isinstance(node, ast.Call) and _call_name(node).endswith("gather"):
            for kw in node.keywords:
                if kw.arg == "return_exceptions" and getattr(kw.value, "value", False) is True:
                    return True
    return False


def gather_exception_check(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    for func in ast.walk(tree):
        if not isinstance(func, _FUNCS) or not _gathers_exceptions(func):
            continue
        for node in ast.walk(func):
            if (
                isinstance(node, ast.Call)
                and _call_name(node) == "isinstance"
                and len(node.args) == 2
                and _dotted(node.args[1]) == "Exception"
            ):
                yield node.lineno


def bool_of_env(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) == "bool" and len(node.args) == 1:
            if isinstance(node.args[0], ast.Call) and _env_read_name(node.args[0]):
                yield node.lineno


def _ladder_key(test: ast.expr) -> str | None:
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        if isinstance(test.ops[0], (ast.Eq, ast.In, ast.Is)):
            right = test.comparators[0]
            if isinstance(right, (ast.Constant, ast.Tuple, ast.Set, ast.List)):
                return ast.dump(test.left)
    return None


def elif_ladder(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    elifs: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or id(node) in elifs:
            continue
        keys, current = [], node
        while isinstance(current, ast.If):
            keys.append(_ladder_key(current.test))
            nxt = current.orelse
            if len(nxt) == 1 and isinstance(nxt[0], ast.If):
                elifs.add(id(nxt[0]))
                current = nxt[0]
            else:
                break
        if len(keys) >= 4 and keys[0] is not None and len(set(keys)) == 1:
            yield node.lineno


def raw_thread(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node) in ("threading.Thread", "Thread"):
            yield node.lineno


CHECKERS: dict[str, Callable[[ast.Module, Ctx], Iterable[int]]] = {
    "HX001": hardcoded_home,
    "HX002": new_env_var,
    "HX003": argv_identity,
    "HX004": unscoped_secret_fallback,
    "HX005": import_time_capture,
    "HX006": missing_timeout,
    "HX007": sync_config_in_async,
    "HX008": get_event_loop,
    "HX009": gather_exception_check,
    "HX010": bool_of_env,
    "HX011": elif_ladder,
    "HX012": raw_thread,
}
