"""Base vs head: the ratchet verdict.

Every unit has its own cap: a function or file already over target may not grow past the value
it has on the base revision; anything new must meet the target. Pattern rules compare multisets
of fingerprints per file, so fixing one violation and adding another still fails.

Code that moves keeps its cap and its existing violations, matched by content: a function by
its name-independent body hash, a pattern hit by (rule, line text) once its enclosing function
has vanished from the old place. Nothing else crosses units.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from scripts.code_health.config import ADVISORY_GROWTH, RULES_BY_ID, TARGETS
from scripts.code_health.gitio import Change
from scripts.code_health.model import MODULE_SCOPE, FileMeasure, Finding, Hit, Unit


@dataclass
class _Pool:
    """Units and hits that disappeared from their base location in this diff."""

    units: dict[str, list[Unit]] = field(default_factory=lambda: defaultdict(list))
    hits: Counter[tuple[str, str]] = field(default_factory=Counter)

    def take_unit(self, body_hash: str) -> Unit | None:
        bucket = self.units.get(body_hash)
        return bucket.pop() if bucket else None

    def take_hit(self, rule: str, text: str) -> bool:
        key = (rule, text)
        if self.hits[key] > 0:
            self.hits[key] -= 1
            return True
        return False


def _build_pool(base: dict[str, FileMeasure], head: dict[str, FileMeasure],
                head_of: dict[str, str | None]) -> _Pool:
    pool = _Pool()
    for bpath, bf in base.items():
        hpath = head_of.get(bpath, bpath)
        hf = head.get(hpath) if hpath else None
        live = set(hf.units) if hf else set()
        for qual, unit in bf.units.items():
            if qual not in live:
                pool.units[unit.body_hash].append(unit)
        for hit, count in bf.hits.items():
            gone = hf is None or (hit.scope != MODULE_SCOPE and hit.scope not in live)
            if gone:
                pool.hits[(hit.rule, hit.text)] += count
    return pool


def _metric_findings(path: str, hf: FileMeasure, bf: FileMeasure | None, pool: _Pool) -> list[Finding]:
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
    for qual, unit in hf.units.items():
        prior = bf.units.get(qual) if bf else None
        if prior is None:
            prior = pool.take_unit(unit.body_hash)
        for metric, value in unit.metrics.items():
            target = TARGETS[metric]
            if value <= target:
                continue
            was = prior.metrics.get(metric) if prior else None
            cap = max(target, was or 0)
            if value > cap:
                findings.append(Finding(path, metric, qual, unit.line,
                                        _detail(value, target, was, metric)))
    return findings


def _detail(value: int, target: int, was: int | None, what: str) -> str:
    if was is None:
        return f"{what} {value} > target {target} (new code must meet the target)"
    if was <= target:
        return f"{what} {value} > target {target} (was {was})"
    return f"{what} {value} > {was}, its value on main (over target {target}: it may only go down)"


def _hit_findings(path: str, hf: FileMeasure, bf: FileMeasure | None, pool: _Pool) -> list[Finding]:
    findings: list[Finding] = []
    base_hits: Counter[Hit] = bf.hits if bf else Counter()
    for hit, count in sorted(hf.hits.items(), key=lambda kv: hf.hit_lines[kv[0]][0]):
        extra = count - base_hits.get(hit, 0)
        while extra > 0 and pool.take_hit(hit.rule, hit.text):
            extra -= 1
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
    pool = _build_pool(base, head, head_of)
    findings: list[Finding] = []
    for path in sorted(head):
        hf = head[path]
        bpath = base_of.get(path, path)
        bf = base.get(bpath) if bpath else None
        findings += _metric_findings(path, hf, bf, pool)
        findings += _hit_findings(path, hf, bf, pool)
    return findings
