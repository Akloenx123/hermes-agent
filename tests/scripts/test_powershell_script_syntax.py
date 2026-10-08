"""The repo's PowerShell scripts must parse — checked with the REAL parser.

install.ps1 is fetched standalone (``irm | iex``) by every Windows user, so
a parse error ships instantly and breaks installs at line 1. Linux CI cannot
execute PowerShell, which is why the other install_ps1 tests are source-regex
probes; this test runs on the ``platforms("windows")`` lane (tests-os.yml), where a
PowerShell host is part of the OS, and asks the actual language parser.

The harness is written to a file and invoked with ``-File`` because inline
``-Command`` needs every ``$`` and quote to survive a shell hop, and
``[ref]$null`` fails inside ``-Command`` with "[ref] cannot be applied to a
variable that does not exist" — an error easy to misread as a finding
against the file under test.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Every PowerShell script a user or CI machine actually runs. Discovered,
# not hardcoded: a new script in scripts/ gets gated automatically.
PS1_SCRIPTS = sorted(
    list(REPO_ROOT.glob("scripts/*.ps1"))
    + list(REPO_ROOT.glob("tests/install/*.ps1"))
    + list(REPO_ROOT.glob("*.ps1"))  # repo-root scripts (setup-hermes.ps1, activate.ps1)
)

_HARNESS = """\
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Paths)
$failed = $false
foreach ($p in $Paths) {
    $tokens = $null
    $errors = $null
    [System.Management.Automation.Language.Parser]::ParseFile(
        $p, [ref]$tokens, [ref]$errors) | Out-Null
    if ($errors.Count) {
        $failed = $true
        Write-Output "PS SYNTAX ERRORS: $p"
        $errors | Select-Object -First 8 | ForEach-Object {
            Write-Output ("  {0}: {1}" -f $_.Extent.StartLineNumber, $_.Message)
        }
    } else {
        Write-Output "PS SYNTAX OK: $p"
    }
}
if ($failed) { exit 1 }
"""


# Scripts users fetch and run as a string, never from disk:
# `& ([scriptblock]::Create((irm <raw url>)))` (website/docs/user-guide/
# windows-native.md, scripts/update-test/HANDOFF.md). Windows PowerShell
# 5.1's irm keeps a UTF-8 BOM as a literal U+FEFF, so `param(` is no longer
# the first statement and the script dies with "The assignment expression is
# not valid". ParseFile above treats the BOM as an encoding marker, so it
# cannot catch this; the BOM has been stripped twice and re-added once.
# (The GUI bootstrap adds a BOM to its *cached* copy on purpose, because it
# runs that copy with -File; see install_script.rs::prepare_cached_script_bytes.)
FETCHED_AS_STRING = (
    REPO_ROOT / "scripts/install.ps1",
    REPO_ROOT / "scripts/update-test/hermes-update-rehearsal.ps1",
)


@pytest.mark.parametrize("script", FETCHED_AS_STRING, ids=lambda p: p.name)
def test_fetched_scripts_have_no_utf8_bom(script: Path) -> None:
    assert not script.read_bytes().startswith(b"\xef\xbb\xbf"), (
        f"{script.relative_to(REPO_ROOT)} starts with a UTF-8 BOM; "
        "[scriptblock]::Create((irm ...)) cannot parse it on PowerShell 5.1"
    )


# Windows PowerShell 5.1 runs a BOM-less script with -File in the system ANSI
# code page, not UTF-8, and the BOM is off the table (see above). On a CJK
# code page (936/932/949) a multi-byte UTF-8 sequence, even inside a comment,
# mis-decodes and the script dies before its first statement: the Desktop
# update hand-off (scripts/desktop-update/windows.ps1) never started on
# Chinese Windows (#134960). Pure ASCII decodes the same under every code
# page, BOM or not. Recursive on purpose: the hand-off scripts live in a
# subdirectory.
SHIPPED_PS1_SCRIPTS = sorted(
    set(REPO_ROOT.glob("*.ps1"))
    | set(REPO_ROOT.glob("scripts/**/*.ps1"))
    | set(REPO_ROOT.glob("apps/desktop/scripts/*.ps1"))
    | set(REPO_ROOT.glob("tests/install/**/*.ps1"))
)


@pytest.mark.parametrize(
    "script", SHIPPED_PS1_SCRIPTS, ids=lambda p: p.relative_to(REPO_ROOT).as_posix()
)
def test_powershell_scripts_are_pure_ascii(script: Path) -> None:
    offenders = [
        number
        for number, line in enumerate(script.read_bytes().splitlines(), start=1)
        if any(byte >= 0x80 for byte in line)
    ]
    assert not offenders, (
        f"{script.relative_to(REPO_ROOT)} has non-ASCII bytes on line(s) "
        f"{offenders}; Windows PowerShell 5.1 reads it in the ANSI code page "
        "and a CJK code page cannot parse it. Use ASCII ('--' for an em-dash, "
        "'-' for box drawing, a \\uXXXX escape in a string)."
    )


def test_ascii_gate_covers_the_update_handoff() -> None:
    names = {p.relative_to(REPO_ROOT).as_posix() for p in SHIPPED_PS1_SCRIPTS}
    assert {"scripts/install.ps1", "scripts/desktop-update/windows.ps1"} <= names


def _powershell_host() -> str | None:
    for name in ("pwsh", "powershell"):
        found = shutil.which(name)
        if found:
            return found
    return None


def test_repo_has_powershell_scripts_to_gate() -> None:
    """The glob must keep finding the scripts this gate exists for."""
    names = {p.name for p in PS1_SCRIPTS}
    assert "install.ps1" in names, PS1_SCRIPTS


@pytest.mark.platforms("windows")
def test_powershell_scripts_parse(tmp_path: Path) -> None:
    host = _powershell_host()
    assert host is not None, "no PowerShell host on a Windows runner"

    harness = tmp_path / "ps-syntax-check.ps1"
    harness.write_text(_HARNESS, encoding="utf-8")

    result = subprocess.run(
        [host, "-NoProfile", "-ExecutionPolicy", "Bypass",
         "-File", str(harness)]
        + [str(p) for p in PS1_SCRIPTS],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"PowerShell parse errors:\n{result.stdout}\n{result.stderr}"
    )
    # Belt and braces: the harness printed a verdict for every script, so a
    # harness that silently checked nothing cannot pass.
    for script in PS1_SCRIPTS:
        assert f"PS SYNTAX OK: {script}" in result.stdout, result.stdout
