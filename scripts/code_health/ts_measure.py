"""TypeScript measurement: runs ``ts_units.mjs`` under node with the lockfile's typescript."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

from scripts.code_health.gitio import git

_SCRIPT = Path(__file__).with_name("ts_units.mjs")


def pinned_typescript(repo: Path) -> str:
    lock = json.loads((repo / "package-lock.json").read_text(encoding="utf-8-sig"))
    return lock["packages"]["node_modules/typescript"]["version"]


def _version_at(pkg_dir: Path) -> str | None:
    try:
        return json.loads((pkg_dir / "package.json").read_text(encoding="utf-8-sig"))["version"]
    except (OSError, ValueError, KeyError):
        return None


def _cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "hermes-code-health"


def resolve_typescript(repo: Path) -> Path:
    """Directory of the pinned ``typescript`` package, installing it into a cache if absent."""
    pin = pinned_typescript(repo)
    common = Path(git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir").strip())
    for root in (repo, common.parent):
        cand = root / "node_modules" / "typescript"
        if _version_at(cand) == pin:
            return cand
    prefix = _cache_root() / f"typescript-{pin}"
    cand = prefix / "node_modules" / "typescript"
    if _version_at(cand) != pin:
        npm = shutil.which("npm")
        if not npm:
            raise RuntimeError("code health needs node + npm to measure TypeScript files")
        subprocess.run(
            [npm, "install", "--no-save", "--no-package-lock", "--no-audit", "--no-fund",
             "--prefix", str(prefix), f"typescript@{pin}"],
            check=True, capture_output=True, timeout=300, stdin=subprocess.DEVNULL,
        )
    return cand


def measure_ts(repo: Path, root: Path, paths: list[str]) -> dict[str, dict]:
    if not paths:
        return {}
    node = shutil.which("node")
    if not node:
        raise RuntimeError("code health needs node to measure TypeScript files")
    ts_dir = resolve_typescript(repo)
    proc = subprocess.run(
        [node, str(_SCRIPT), str(ts_dir), str(root)],
        input=json.dumps(paths), capture_output=True, text=True, encoding="utf-8",
        timeout=600, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ts_units.mjs failed: {proc.stderr.strip()[:2000]}")
    return json.loads(proc.stdout)
