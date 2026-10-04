"""Base vs head: the ratchet verdict.

Every unit has its own cap: a function or file already over target may not grow past the value
it has on the base revision; anything new must meet the target. Pattern rules compare multisets
of fingerprints per file, so fixing one violation and adding another still fails.

Code that moves keeps its cap and its existing violations, matched by content: a unit's prior is
the base unit with the same name-independent body hash (same file first, so an anonymous
callback whose source-order ordinal shifted is still itself; then any file, for a moved
function), else the base unit with the same qualname. Pattern hits follow ONLY such a proven
unit move; deleting one function never pays for a violation somewhere else.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from scripts.code_health.config import ADVISORY_GROWTH, RULES_BY_ID, TARGETS
from scripts.code_health.gitio import Change
from scripts.code_health.model import MODULE_SCOPE, FileMeasure, Finding, Hit, Unit

# (base path, base qualname) a head unit was matched to.
Origin = tuple[str, str]


@dataclass
class _Moved:
    """Base units absent from their own file at head: candidates for a cross-file move."""

    units: dict[str, list[tuple[str, Unit]]] = field(default_factory=lambda: defaultdict(list))

    def take(self, body_hash: str) -> tuple[str, Unit] | None:
        bucket = self.units.get(body_hash)
        return bucket.pop() if bucket else None


def _vanished(base: dict[str, FileMeasure], head: dict[str, FileMeasure],
              head_of: dict[str, str | None]) -> _Moved:
    moved = _Moved()
    for bpath, bf in base.items():
        hpath = head_of.get(bpath, bpath)
        hf = head.get(hpath) if hpath else None
        live_hashes = {u.body_hash for u in hf.units.values()} if hf else set()
        for unit in bf.units.values():
            if unit.body_hash not in live_hashes:
                moved.units[unit.body_hash].append((bpath, unit))
    return moved


def _prior(qual: str, unit: Unit, bpath: str | None, bf: FileMeasure | None,
           moved: _Moved) -> tuple[Unit, Origin] | None:
    if bf is not None and bpath is not None:
        same_body = [u for u in bf.units.values() if u.body_hash == unit.body_hash]
        if same_body:
            pick = next((u for u in same_body if u.qualname == qual), same_body[0])
            return pick, (bpath, pick.qualname)
        if qual in bf.units:
            return bf.units[qual], (bpath, qual)
    taken = moved.take(unit.body_hash)
    return (taken[1], (taken[0], taken[1].qualname)) if taken else None


def _metric_findings(path: str, hf: FileMeasure, bpath: str | None, bf: FileMeasure | None,
                     moved: _Moved) -> tuple[list[Finding], dict[str, Origin]]:
    findings: list[Finding] = []
    lines = hf.metrics.get("FILE_LINES", 0)
    if lines > TARGETS["FILE_LINES"]:
        base_lines = bf.metrics.get("FILE_LINES", 0) if bf else 0
        cap = max(TARGETS["FILE_LINES"], base_lines)
        if lines > cap:
            grew_over = base_lines > TARGETS["FILE_LINES"] and "FILE_LINES" in ADVISORY_GROWTH
            findings.append(Finding(path, "FILE_LINES", MODULE_SCOPE, 1, _detail(
                lines, TARGETS["FILE_LINES"], base_lines if bf else None, "lines"),
                blocking=not grew_over))
    origins: dict[str, Origin] = {}
    for qual, unit in hf.units.items():
        matched = _prior(qual, unit, bpath, bf, moved)
        prior = matched[0] if matched else None
        if matched:
            origins[qual] = matched[1]
        for metric, value in unit.metrics.items():
            target = TARGETS[metric]
            if value <= target:
                continue
            was = prior.metrics.get(metric) if prior else None
            cap = max(target, was or 0)
            if value > cap:
                findings.append(Finding(path, metric, qual, unit.line,
                                        _detail(value, target, was, metric)))
    return findings, origins


def _detail(value: int, target: int, was: int | None, what: str) -> str:
    if was is None:
        return f"{what} {value} > target {target} (new code must meet the target)"
    if was <= target:
        return f"{what} {value} > target {target} (was {was})"
    return f"{what} {value} > {was}, its value on main (over target {target}: it may only go down)"


def _base_hits(path: str, bf: FileMeasure | None, origins: dict[str, Origin],
               base: dict[str, FileMeasure]) -> Counter[Hit]:
    """The base hits a head file is compared against: its own base file's, plus those of each
    unit that provably moved or was renamed into it (re-keyed to the unit's head scope)."""
    counter: Counter[Hit] = Counter(bf.hits) if bf else Counter()
    for qual, (opath, oqual) in origins.items():
        if (opath, oqual) == (path, qual) or opath not in base:
            continue
        for hit, count in base[opath].hits.items():
            if hit.scope == oqual:
                counter[Hit(hit.rule, qual, hit.text)] += count
    return counter


def _hit_findings(path: str, hf: FileMeasure, base_hits: Counter[Hit]) -> list[Finding]:
    findings: list[Finding] = []
    for hit, count in sorted(hf.hits.items(), key=lambda kv: hf.hit_lines[kv[0]][0]):
        extra = count - base_hits.get(hit, 0)
        if extra <= 0:
            continue
        rule = RULES_BY_ID[hit.rule]
        for line in hf.hit_lines[hit][-extra:]:
            findings.append(Finding(path, hit.rule, hit.scope, line, rule.title,
                                    blocking=rule.blocking))
    return findings


def compare(base: dict[str, FileMeasure], head: dict[str, FileMeasure],
            changes: list[Change]) -> list[Finding]:
    base_of = {c.new: c.old for c in changes if c.new}
    head_of = {c.old: c.new for c in changes if c.old}
    moved = _vanished(base, head, head_of)
    findings: list[Finding] = []
    for path in sorted(head):
        hf = head[path]
        bpath = base_of.get(path, path)
        bf = base.get(bpath) if bpath else None
        metric, origins = _metric_findings(path, hf, bpath, bf, moved)
        findings += metric
        findings += _hit_findings(path, hf, _base_hits(path, bf, origins, base))
    return findings
