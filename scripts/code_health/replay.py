"""Replay the ratchet over recently merged PRs: what would it have blocked?

    python -m scripts.code_health.replay --limit 300 --out <dir>

Writes ``<dir>/replay.jsonl`` (one row per PR) and prints per-rule totals. Use it before
promoting a rule to blocking or changing a target: a rule whose hits on merged PRs are mostly
legitimate code ships advisory until its checker is fixed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from scripts.code_health import gitio
from scripts.code_health.compare import compare
from scripts.code_health.config import in_scope
from scripts.code_health.measure import Measurer
from scripts.code_health.report import apply_allows
from scripts.code_health.ruff_runner import resolve_ruff

_QUERY = """
query($q: String!, $after: String) {
  search(query: $q, type: ISSUE, first: 50, after: $after) {
    pageInfo { hasNextPage endCursor }
    nodes { ... on PullRequest {
      number title mergedAt
      mergeCommit { oid }
      commits(last: 100) { totalCount nodes { commit { messageHeadline } } }
    } }
  }
}"""


def merged_prs(repo: Path, limit: int) -> list[dict]:
    prs: list[dict] = []
    after = None
    while len(prs) < limit:
        args = ["gh", "api", "graphql", "-f", f"query={_QUERY}",
                "-f", "q=repo:NousResearch/hermes-agent is:pr is:merged base:main sort:updated-desc"]
        if after:
            args += ["-f", f"after={after}"]
        out = subprocess.run(args, cwd=repo, capture_output=True, text=True, encoding="utf-8", check=True,
                             timeout=120, stdin=subprocess.DEVNULL).stdout
        data = json.loads(out)["data"]["search"]
        prs += [n for n in data["nodes"] if n.get("mergeCommit")]
        if not data["pageInfo"]["hasNextPage"]:
            break
        after = data["pageInfo"]["endCursor"]
    return prs[:limit]


def pr_range(repo: Path, pr: dict) -> tuple[str, str]:
    """Base/head on main for a rebase-merged PR (its commits' subjects), else the merge parent."""
    head = pr["mergeCommit"]["oid"]
    subjects = {n["commit"]["messageHeadline"] for n in pr["commits"]["nodes"]}
    log = gitio.git(repo, "log", "--first-parent", "--format=%H%x00%s", "-n",
                    str(pr["commits"]["totalCount"] + 1), head).splitlines()
    k = 0
    for line in log:
        _, subject = line.split("\0", 1)
        if subject not in subjects:
            break
        k += 1
    return gitio.resolve_rev(repo, f"{head}~{max(k, 1)}"), head


def replay_one(repo: Path, measurer: Measurer, base: str, head: str) -> list[dict]:
    changes = gitio.changed_files(repo, base, head)
    head_paths = sorted({c.new for c in changes if c.new and in_scope(c.new)})
    base_paths = sorted({c.old for c in changes if c.old and in_scope(c.old)})
    if not head_paths:
        return []
    measurer.ctx.known_env = gitio.known_env_names(repo, base)
    base_m = measurer.measure(base, base_paths)
    head_m = measurer.measure(head, head_paths)
    findings = compare(base_m, head_m, changes)
    apply_allows(findings, head_m)
    return [asdict(f) for f in findings if f.allowed_reason is None]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="code_health.replay")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    repo = gitio.repo_root(Path.cwd())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    measurer = Measurer(repo, resolve_ruff(repo), known_env=set())
    per_rule: Counter[str] = Counter()
    prs_blocked: Counter[str] = Counter()
    with (out_dir / "replay.jsonl").open("w", encoding="utf-8") as fh:
        for pr in merged_prs(repo, args.limit):
            try:
                base, head = pr_range(repo, pr)
                findings = replay_one(repo, measurer, base, head)
            except RuntimeError as exc:
                findings, base, head = [{"error": str(exc)}], "", ""
            row = {"number": pr["number"], "title": pr["title"], "base": base, "head": head,
                   "findings": findings}
            fh.write(json.dumps(row) + "\n")
            rules = {f["rule"] for f in findings if "rule" in f}
            prs_blocked.update(rules)
            per_rule.update(f["rule"] for f in findings if "rule" in f)
    for rule, count in per_rule.most_common():
        print(f"{rule:<11} findings {count:>4}  PRs {prs_blocked[rule]:>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
