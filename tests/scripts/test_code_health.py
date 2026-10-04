"""Invariants of the code-health ratchet (scripts/code_health), on a real git repo + pinned ruff."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.code_health.cli import run

REPO = Path(__file__).resolve().parents[2]
_LEGACY = "def legacy(x):\n" + "".join(f"    if x == {i}:\n        return {i}\n" for i in range(21))
_SWALLOW = "def other():\n    try:\n        pass\n    except Exception:\n        pass\n"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                          text=True, encoding="utf-8", timeout=60).stdout.strip()


def _commit(repo: Path, files: dict[str, str | None]) -> str:
    for rel, text in files.items():
        path = repo / rel
        if text is None:
            path.unlink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "step")
    return _git(repo, "rev-parse", "HEAD")


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    (repo / "scripts" / "ci").mkdir(parents=True)
    shutil.copy(REPO / "pyproject.toml", repo / "pyproject.toml")
    shutil.copy(REPO / "scripts/ci/profile_scope_patterns.json", repo / "scripts/ci/")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    return repo, _commit(repo, {"pkg/a.py": _LEGACY + "\n\n" + _SWALLOW})


def _verdict(repo: Path, base: str, files: dict[str, str | None], capsys) -> tuple[int, str]:
    head = _commit(repo, files)
    code = run(repo, base, head)
    out = capsys.readouterr().out
    _git(repo, "reset", "-q", "--hard", base)
    return code, out


def test_ratchet_blocks_growth_new_and_swapped_violations(tmp_path, capsys):
    repo, base = _repo(tmp_path)
    legacy_plus = _LEGACY + "    if x == 99:\n        return 99\n"
    ok = _LEGACY + "\n\n" + _SWALLOW + "\n\ndef added():\n    return 1\n"
    grown = legacy_plus + "\n\n" + _SWALLOW
    fresh = _SWALLOW.replace("pass\n", "return 2\n", 1).replace("other", "fresh")
    swapped = _LEGACY + "\n\ndef other():\n    return 1\n\n\n" + fresh

    assert _verdict(repo, base, {"pkg/a.py": ok}, capsys)[0] == 0  # legacy debt never blocks
    code, out = _verdict(repo, base, {"pkg/a.py": grown}, capsys)
    assert code == 1 and "legacy" in out and "CC 23 > 22" in out
    code, out = _verdict(repo, base, {"pkg/a.py": swapped}, capsys)
    assert code == 1 and "BLE001" in out and "fresh" in out  # fixing one doesn't buy another
    traded = {"pkg/a.py": _LEGACY, "pkg/b.py": fresh}
    code, out = _verdict(repo, base, traded, capsys)
    assert code == 1 and "fresh" in out  # deleting a violation never pays for an unrelated one


def test_moved_code_keeps_its_cap(tmp_path, capsys):
    repo, base = _repo(tmp_path)
    for split in ({"pkg/a.py": _SWALLOW, "pkg/a_legacy.py": _LEGACY},
                  {"pkg/a.py": _LEGACY, "pkg/a_swallow.py": _SWALLOW}):  # hits move with their unit
        code, out = _verdict(repo, base, dict(split), capsys)
        assert code == 0, out
    renamed: dict[str, str | None] = {"pkg/a.py": None, "pkg/b.py": _LEGACY + "\n\n" + _SWALLOW}
    code, out = _verdict(repo, base, renamed, capsys)
    assert code == 0, out


_RUN = "import subprocess\n\n\ndef f(cmd):\n    return subprocess.run(cmd{})\n"
_PROC = "import asyncio\n\n\nasync def f(proc):\n{}\n"
_ENV = "import os\n\n{}\n"
_EXCEPT = "    try:\n        pass\n    except Exception:\n        pass\n"


_STUB = {"pkg/b.py": "def legacy(x):\n    return x\n"}


@pytest.mark.parametrize("extra_base, files, blocks", [
    # one base unit is credit for one head unit: a copy of unchanged debt is new debt
    ({}, {"pkg/a.py": _LEGACY + "\n\n" + _SWALLOW + "\n\n" + _LEGACY.replace("legacy", "copied")}, True),
    # a file rename re-keys each old hit once, so a second swallow in that function is new
    ({}, {"pkg/a.py": None, "pkg/b.py": _LEGACY + "\n\n" + _SWALLOW + _EXCEPT}, True),
    # a function moved over a same-name stub keeps its own cap, not the stub's
    (_STUB, {"pkg/a.py": _SWALLOW, "pkg/b.py": _LEGACY}, False),
    ({}, {"pkg/p.py": _RUN.format(", timeout=None")}, True),  # a disabled deadline is no deadline
    ({}, {"pkg/p.py": _RUN.format(", **{}")}, True),
    ({}, {"pkg/p.py": _PROC.format("    return await asyncio.wait_for(proc.communicate(), None)")}, True),
    ({}, {"pkg/p.py": _PROC.format("    async with asyncio.timeout(1):\n        return await proc.communicate()")}, False),
    # an allow directive inside a string literal waives nothing
    ({}, {"pkg/p.py": _RUN.format("").replace("    return", "    print('health: allow HX006 -- doc')\n    return")}, True),
    ({}, {"pkg/c.py": _ENV.format("if __name__ != '__main__':\n    CACHED = os.getenv('PATH')")}, True),
    ({}, {"pkg/c.py": _ENV.format("for _ in range(1):\n    CACHED = os.getenv('PATH')")}, True),
    ({}, {"pkg/c.py": _ENV.format("current = lambda: os.getenv('PATH')")}, False),  # deferred read
])
def test_verdicts_follow_ownership_deadlines_and_import_execution(tmp_path, capsys, extra_base, files, blocks):
    repo, base = _repo(tmp_path)
    if extra_base:
        base = _commit(repo, extra_base)
    code, out = _verdict(repo, base, files, capsys)
    assert code == (1 if blocks else 0), out
