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
    if isinstance(expr, (ast.Lambda, *_FUNCS)):
        return  # a deferred body reads at call time, which is the fix, not the bug
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


def _main_guard(stmt: ast.stmt) -> str | None:
    """``"=="``/``"!="`` for ``if __name__ <op> "__main__":``, else None."""
    test = stmt.test if isinstance(stmt, ast.If) else None
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1 and len(test.comparators) == 1):
        return None
    sides = {ast.unparse(test.left), ast.unparse(test.comparators[0])}
    if sides != {"__name__", "'__main__'"}:
        return None
    return {ast.Eq: "==", ast.NotEq: "!="}.get(type(test.ops[0]))


# Statement blocks that execute when the enclosing block does (``match`` cases via ``cases``).
_BLOCKS = ("body", "orelse", "finalbody")
_COMPOUND = (ast.If, ast.Try, ast.TryStar, ast.With, ast.For, ast.While, ast.Match)


def _import_time_blocks(stmt: ast.stmt) -> Iterator[list[ast.stmt]]:
    guard = _main_guard(stmt)
    if isinstance(stmt, ast.If) and guard is not None:
        # Only the branch that runs on import: ``else`` of ``==``, body of ``!=``.
        yield stmt.orelse if guard == "==" else stmt.body
        return
    for name in _BLOCKS:
        yield getattr(stmt, name, None) or []
    for handler in getattr(stmt, "handlers", None) or []:
        yield handler.body
    for case in getattr(stmt, "cases", None) or []:
        yield case.body


def _import_time_statements(body: list[ast.stmt]) -> Iterator[ast.stmt]:
    """Statements that run at import: module/class bodies and every compound block in them."""
    for stmt in body:
        yield stmt
        if isinstance(stmt, ast.ClassDef):
            yield from _import_time_statements(stmt.body)
        elif isinstance(stmt, _COMPOUND):
            for block in _import_time_blocks(stmt):
                yield from _import_time_statements(block)


def _import_time_exprs(stmt: ast.stmt) -> Iterator[ast.AST]:
    """Expressions a statement evaluates at import (not its deferred function bodies)."""
    if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and stmt.value:
        yield stmt.value
    elif isinstance(stmt, (*_FUNCS, ast.ClassDef)):
        yield from stmt.decorator_list
        if isinstance(stmt, ast.ClassDef):
            yield from (*stmt.bases, *(kw.value for kw in stmt.keywords))
        else:
            yield from (d for d in [*stmt.args.defaults, *stmt.args.kw_defaults] if d is not None)
    elif isinstance(stmt, (ast.For, ast.While)):
        yield stmt.iter if isinstance(stmt, ast.For) else stmt.test
    elif isinstance(stmt, ast.Match):
        yield stmt.subject


def import_time_capture(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    for stmt in _import_time_statements(tree.body):
        for expr in _import_time_exprs(stmt):
            yield from _capture_lines(expr)


def _finite(node: ast.AST | None) -> bool:
    """A deadline expression that is present and not a literal ``None``."""
    return node is not None and not (isinstance(node, ast.Constant) and node.value is None)


def _deadline(call: ast.Call, name: str = "timeout", position: int | None = None) -> bool:
    """True when ``call`` passes a finite deadline: ``name=<not None>``, the positional slot,
    or ``**opts`` (an unknown mapping is trusted; a literal one must name the deadline)."""
    if position is not None and len(call.args) > position:
        return _finite(call.args[position])
    for kw in call.keywords:
        if kw.arg == name:
            return _finite(kw.value)
        if kw.arg is None:
            if not isinstance(kw.value, ast.Dict):
                return True
            for key, value in zip(kw.value.keys, kw.value.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == name:
                    return _finite(value)
    return False


def _awaited_calls(body: list[ast.stmt]) -> Iterator[ast.Call]:
    for stmt in body:
        for node in _walk_skipping(stmt, (ast.Lambda, *_FUNCS)):
            if isinstance(node, ast.Await) and isinstance(node.value, ast.Call):
                yield node.value


def _bounded_calls(tree: ast.Module) -> set[int]:
    """ids of calls an asyncio deadline bounds: the awaitable passed to ``wait_for(x, <finite>)``
    and every call awaited directly inside ``async with asyncio.timeout(<finite>):``."""
    bounded: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _call_name(node).rpartition(".")[2] == "wait_for":
            if node.args and _deadline(node, "timeout", 1):
                bounded.add(id(node.args[0]))
        elif isinstance(node, ast.AsyncWith):
            for item in node.items:
                ctx_call = item.context_expr
                if not isinstance(ctx_call, ast.Call):
                    continue
                leaf = _call_name(ctx_call).rpartition(".")[2]
                slot = {"timeout": "delay", "timeout_at": "when"}.get(leaf)
                if slot and _deadline(ctx_call, slot, 0):
                    bounded.update(id(call) for call in _awaited_calls(node.body))
    return bounded


def missing_timeout(tree: ast.Module, ctx: Ctx) -> Iterable[int]:
    bounded = _bounded_calls(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or id(node) in bounded:
            continue
        head, _, leaf = _call_name(node).rpartition(".")
        if head == "subprocess" and leaf in _SUBPROCESS_WAITS and not _deadline(node):
            yield node.lineno
        elif leaf == "urlopen" and not _deadline(node, "timeout", 2):
            yield node.lineno
        elif leaf == "communicate" and head and not _deadline(node, "timeout", 1):
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
