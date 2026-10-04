"""Startup requests for PM recovery and rescue of orphaned launchers."""

from __future__ import annotations

import contextlib
import errno
import os
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

# Older installers left renamed launchers behind on failure. The startup
# orphan sweep still restores them. Generation installs never rename live shims.
QUARANTINE_RESTORE_BACKOFF_MS: tuple[int, ...] = (0, 100, 250, 500, 1000)


def restore_quarantined_shims(
    moved: list[tuple[Path, Path]], *, stream=None,
    backoff_ms: tuple[int, ...] = QUARANTINE_RESTORE_BACKOFF_MS,
) -> list[tuple[Path, Path]]:
    """Rename quarantined shims back, retrying a lock instead of giving up.

    A pair is not a failure when ``original`` already exists or ``quarantined`` has gone: the
    installer wrote a fresh shim, or a concurrent sweep won the race. Both are silent, so two
    processes sweeping the same orphan cannot produce a spurious error.
    """
    if stream is None:
        stream = sys.stderr
    failed: list[tuple[Path, Path]] = []
    for original, quarantined in moved:
        last_exc: OSError | None = None
        for delay_ms in backoff_ms:
            try:
                if os.path.exists(original) or not os.path.exists(quarantined):
                    last_exc = None
                    break
                if delay_ms:
                    time.sleep(delay_ms / 1000.0)
                os.rename(quarantined, original)
                last_exc = None
                break
            except OSError as exc:
                last_exc = exc
        if last_exc is None:
            continue
        failed.append((original, quarantined))
        name = os.path.basename(str(original))
        stem = name[:-4] if name.lower().endswith(".exe") else name
        print(
            f"  ✖ FAILED to restore {name} "
            f"({last_exc.__class__.__name__}) — it is still quarantined "
            f"as {os.path.basename(str(quarantined))}.\n"
            f"    `{stem}` will NOT be on PATH until it is put back. Run this, "
            f"then re-run the update:\n"
            f'      move "{quarantined}" "{original}"',
            file=stream,
        )
    return failed


# Set only when this process successfully finishes a deferred core install for an ``update``
# invocation. The CLI import that follows must not resolve external secret sources: a configured
# source can map cryptography._rust and immediately recreate the self-lock marker this fresh
# process just consumed. Process-local on purpose so children do not inherit the exception.
_UPDATE_RETRY_RECOVERED = False


def _should_skip_external_secret_sources() -> bool:
    """True inside any ``hermes update`` process (and its import probes).

    Every dotenv load in the process — ``hermes_cli.main``, ``run_agent``, ``cli`` — consults
    this, so the updater never resolves external secret sources: on Windows they map
    ``cryptography._rust.pyd`` into the process replacing that venv, and everywhere a slow
    ``op``/``bws``/command helper (up to 120s per source) would run inside the updater's
    120s critical-module import probe and be reported as an import-health timeout.
    Profile flags are stripped before ``hermes_cli.main`` loads dotenv, so ``argv[1]`` is
    the authoritative subcommand.
    """
    return _UPDATE_RETRY_RECOVERED or sys.argv[1:2] == ["update"]


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _read_marker_attempts(marker_path: Path) -> int:
    """Attempt counter from a marker's opportunistic JSON body; corrupt/missing → 0."""
    try:
        raw = marker_path.read_text(encoding="utf-8-sig", errors="replace").strip()
    except OSError:
        return 0
    if not raw:
        return 0
    try:
        import json

        return int(json.loads(raw).get("attempts", 0))
    except (ValueError, AttributeError, TypeError):
        for line in reversed(raw.splitlines()):
            key, separator, value = line.partition("=")
            if separator and key.strip() == "attempts":
                try:
                    return max(0, int(value))
                except ValueError:
                    return 0
        return 0


def _process_state(pid: int) -> str | None:
    """Single-letter process state (``ps`` style), or ``None`` when unknowable.

    ``os.kill(pid, 0)`` also succeeds for a ZOMBIE — a process that exited but
    whose parent has not reaped it yet. Reading the state lets callers treat a
    zombie as dead, so a crashed update stage lingering under an un-reaping
    parent cannot keep a stale update marker "live" for the whole age ceiling
    (#77259, #120635, #125932). Best-effort and stdlib-only: on any failure the
    answer is ``None`` and callers keep their signal-0 verdict.
    """
    if sys.platform == "linux":
        try:
            with open(f"/proc/{pid}/stat", "rb") as fh:
                stat = fh.read()
        except OSError:
            return None
        # Field 3 is the state, but comm may contain spaces/parens: anchor on
        # the closing paren of comm instead of splitting on whitespace.
        comm_end = stat.rfind(b")")
        if comm_end < 0:
            return None
        return stat[comm_end + 2 : comm_end + 3].decode("ascii", "replace") or None
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(pid)],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=5, check=False,
            ).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        return out.strip()[:1] or None
    return None


def _pid_is_running(pid: int) -> bool:
    """Best-effort stdlib-only process liveness probe.

    ``os.kill(pid, 0)`` is not a no-op on Windows, so use the Win32 process handle API there. An
    access-denied result counts as live: racing an elevated updater is worse than postponing
    recovery for one launch. A zombie (exited, un-reaped) counts as dead — see
    :func:`_process_state`.
    """
    if pid <= 0:
        return False
    if sys.platform == "win32":
        try:
            import ctypes

            synchronize = 0x00100000
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            kernel32.WaitForSingleObject.restype = ctypes.c_ulong
            kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
            kernel32.CloseHandle.restype = ctypes.c_int
            handle = kernel32.OpenProcess(synchronize, False, pid)
            if not handle:
                return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED
            try:
                return kernel32.WaitForSingleObject(handle, 0) == 258
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return True
    try:
        os.kill(pid, 0)  # windows-footgun: ok — Windows returns above
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    state = _process_state(pid)
    if state is not None and state.upper().startswith("Z"):
        return False  # exited, unreaped — not a live owner
    return True


def _marker_owner_is_live(marker: Path) -> bool:
    """True when a legacy update marker names a process still running."""
    try:
        body = marker.read_text(encoding="utf-8-sig", errors="replace")
    except OSError:
        return False
    for line in body.splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "pid":
            try:
                return _pid_is_running(int(value.strip()))
            except ValueError:
                return False
    return False


# ``hermes update`` writes this into the git dir right before git moves the checkout and removes it
# once git has exited (a kill is the only exit that keeps it). Git rewrites the tree file by file and
# moves HEAD last, so an update killed in between leaves HEAD on the old commit with a prefix of the
# files already new; that mixed tree fails at import in every entry point, ``hermes update`` included.
INTERRUPTED_PULL_MARKER = "hermes-update-pull"
# A fast-forward takes seconds; past this a "live" owner pid is a recycled one.
_INTERRUPTED_PULL_MAX_AGE_SECONDS = 10 * 60
# The user (or a killed updater) is mid-operation: its own state files own the tree.
_GIT_OPERATION_IN_PROGRESS = ("MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "rebase-merge", "rebase-apply")
_REGULAR_FILE_MODES = ("100644", "100755")


def git_operation_in_progress(root: Path) -> str | None:
    """Return the active Git operation currently controlling *root*, if any."""
    git_dir = _git_dir(root)
    for marker in _GIT_OPERATION_IN_PROGRESS:
        state = git_dir / marker
        if not state.exists():
            continue
        if marker == "rebase-apply":
            return "am" if (state / "applying").exists() else "rebase"
        if marker == "rebase-merge":
            return "rebase"
        return marker.removesuffix("_HEAD").lower().replace("_", "-")
    return None


def _git_dir(root: Path) -> Path:
    """``root``'s git dir: ``.git`` itself, or where a linked worktree's ``.git`` file points."""
    dot_git = root / ".git"
    if dot_git.is_file():
        text = dot_git.read_text(encoding="utf-8-sig").strip()
        if text.startswith("gitdir:"):
            return root / text[len("gitdir:"):].strip()
    return dot_git


def interrupted_pull_marker(root: Path) -> Path:
    return _git_dir(root) / INTERRUPTED_PULL_MARKER


def _git_executable(recorded_by_updater: str = "") -> str:
    """The git ``hermes update`` runs (``_subprocess_compat.expose_pm_git``), without installing it.

    The absolute git the killed updater recorded in its marker first: finding PM's copy needs
    ``pm``, which imports ``hermes_constants``, and a merge killed while writing that module (or
    anything else ``pm`` imports) leaves only this path to the repair. Then PATH's git. A Windows install whose only git is the one install.ps1 staged in PM's store
    has none on PATH until the updater exposes it, so this falls back to that copy: PM's recorded
    entry, or the lockfile's pinned entry the installer extracted without recording it. A bare
    ``git`` there dies with WinError 2 and the torn tree this repair exists for stays torn.
    """
    import shutil

    if recorded_by_updater and os.path.isfile(recorded_by_updater):
        return recorded_by_updater
    found = shutil.which("git")
    if found:
        return found
    with contextlib.suppress(Exception):  # no PM, no store, no Windows git package: PATH's answer stands
        import pm
        from pm import paths
        from pm.lock import Lockfile

        recorded = pm.installed_package("git", allow_outdated=True)
        if recorded is not None and recorded.binary is not None and recorded.binary.is_file():
            return str(recorded.binary)
        package, target = pm.get_package("git"), pm.current_target()
        version = Lockfile(paths.lockfile_path()).version("git")
        staged = package.binary(paths.store_root() / package.store_entry(version, target), target)
        if staged is not None and staged.is_file():
            return str(staged)
    return "git"


def _trees_git_could_write(git, pre: str, target: str) -> tuple[list[str], set[str]]:
    """The trees the killed git was moving the checkout to, and the paths whose new content is unknowable.

    A fast-forward or ``reset --hard`` writes ``target``. On a custom branch the updater runs
    ``git merge``, whose files are the merge of both sides: ``merge-tree`` computes the same tree. A
    conflicted path (its markers carry other labels) and, on git < 2.38 (no ``--write-tree``), every
    path both sides changed count as git's whatever their content.
    """
    base = git("merge-base", pre, target).stdout.strip()
    if not base or base == pre:  # fast-forward, or unrelated histories (only a reset can land those)
        return [target], set()
    merged = git("merge-tree", "--write-tree", "-z", "--name-only", "--no-messages", pre, target)
    if merged.returncode in (0, 1):  # 1: conflicts
        tree, *conflicted = merged.stdout.split("\0")
        return [target, tree], set(filter(None, conflicted))
    changed = [set(filter(None, git("diff", "--name-only", "-z", "--no-renames", base, side).stdout.split("\0")))
               for side in (pre, target)]
    return [target], changed[0] & changed[1]


def _hash_worktree(git, paths: list[str]) -> dict[str, str]:
    """Blob ids of the checkout's files, through the repo's clean filters, like ``git add`` would store."""
    listed = [p for p in paths if "\n" not in p]  # --stdin-paths is newline-delimited
    blobs = {}
    if listed:
        hashed = git("hash-object", "--stdin-paths", stdin="\n".join(listed) + "\n")
        if hashed.returncode != 0:
            raise subprocess.SubprocessError(hashed.stderr.strip())
        blobs.update(zip(listed, hashed.stdout.split()))
    for path in set(paths) - set(listed):
        blobs[path] = git("hash-object", "--", path).stdout.strip()
    return blobs


def _paths_git_wrote(git, root: Path, pre: str, target: str) -> tuple[list[str], list[str], set[str]] | None:
    """Paths the killed git already touched on the way to ``target``: (restore from HEAD, delete as added,
    directories git may have created for its added files).

    Git rewrites a file as unlink, create, write, so a kill leaves it missing, empty or cut short:
    all of those count as git's, like the full new blob. Content that matches neither side and is not
    the start of a new blob is the user's own edit (e.g. a re-applied stash) and is left alone.
    ``None``: git no longer knows ``target``.
    """
    if git("rev-parse", "-q", "--verify", f"{target}^{{commit}}").returncode != 0:
        return None
    trees, unknown = _trees_git_could_write(git, pre, target)
    entries = {}  # path -> (pre mode, pre blob or None when git adds it, [(new mode, new blob or None)])
    for tree in trees:
        diff = git("diff", "--raw", "-z", "--no-renames", "--no-abbrev", pre, tree)
        if diff.returncode != 0:
            raise subprocess.SubprocessError(diff.stderr.strip())
        parts = diff.stdout.split("\0")
        for meta, path in zip(parts[::2], parts[1::2]):
            old_mode, new_mode, old_blob, new_blob, status = meta.lstrip(":").split()
            if (old_mode if status == "D" else new_mode) not in _REGULAR_FILE_MODES:
                continue
            entry = entries.setdefault(path, (old_mode, None if status == "A" else old_blob, []))
            entry[2].append((new_mode, None if status == "D" else new_blob))
    worktree_blob = _hash_worktree(git, [path for path in entries if (root / path).is_file()])
    restore, added = [], []
    for path, (old_mode, old_blob, new) in entries.items():
        file, blobs = root / path, {blob for _mode, blob in new if blob}
        if path not in worktree_blob:
            written = old_blob is not None  # unlinked (or deleted), not yet recreated
        elif worktree_blob[path] == old_blob:  # only a mode change tells whether git got here
            written = (sys.platform != "win32" and any(b == old_blob and m != old_mode for m, b in new)
                       and bool(file.stat().st_mode & 0o100) != (old_mode == "100755"))
        elif worktree_blob[path] in blobs or path in unknown:
            written = True
        else:  # git's own file cut short starts one of the new blobs
            content = file.read_bytes()
            written = False
            for blob in blobs:
                shown = git("cat-file", "--filters", f"--path={path}", blob, text=False)
                if shown.returncode != 0:
                    raise subprocess.SubprocessError(shown.stderr.decode(errors="replace").strip())
                written = written or shown.stdout.startswith(content)
        if written:
            (added if old_blob is None else restore).append(path)
    new_dirs = {str(parent) for path, (_m, old_blob, _n) in entries.items() if old_blob is None
                for parent in PurePosixPath(path).parents if parent.parts}
    if new_dirs:
        new_dirs -= set(git("ls-tree", "-r", "-d", "--name-only", "-z", pre).stdout.split("\0"))
    return restore, added, new_dirs


# Launches that start together after a killed update (a restarting gateway or Desktop backend next to
# the user's CLI) restore one at a time: the claim holder owns the git index, the rest wait for it and
# relaunch from its tree. An OS lock, not a pid file: the kernel drops it with its owner, so a dead
# launch's claim is broken without a check-then-unlink race.
_RESTORE_CLAIM = "hermes-update-pull.claim"
_RESTORE_CLAIM_WAIT_SECONDS = 10.0
_merge_advice_shown = False


# flock's EWOULDBLOCK and msvcrt LK_NBLCK's EACCES/EDEADLOCK: another process holds the claim.
_CLAIM_HELD_ERRNOS = frozenset({errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES, errno.EDEADLK})


def _lock_fd(fd: int, lock: bool) -> bool:
    """False only while another launch holds the claim.

    A filesystem that cannot lock at all (ENOLCK on NFS without lockd, EOPNOTSUPP/EINVAL on some SMB
    shares) proceeds unguarded like a read-only git dir: counting it as "held" would stop every launch
    from ever repairing a checkout the restore handled fine before claims existed.
    """
    try:
        if sys.platform == "win32":
            import msvcrt

            os.lseek(fd, 0, os.SEEK_SET)  # msvcrt locks from the file position
            msvcrt.locking(fd, msvcrt.LK_NBLCK if lock else msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, (fcntl.LOCK_EX | fcntl.LOCK_NB) if lock else fcntl.LOCK_UN)
    except OSError as exc:
        return not (lock and exc.errno in _CLAIM_HELD_ERRNOS)
    return True


@contextlib.contextmanager
def _restore_claim(git_dir: Path):
    """Yields True while this launch holds the restore claim, False when another held it past the wait."""
    try:
        fd = os.open(git_dir / _RESTORE_CLAIM, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError:  # read-only git dir: no restore can write there either, the attempt reports why
        yield True
        return
    try:
        deadline = time.monotonic() + _RESTORE_CLAIM_WAIT_SECONDS
        while not _lock_fd(fd, True):
            if time.monotonic() > deadline:
                yield False
                return
            time.sleep(0.05)
        try:
            os.ftruncate(fd, 0)
            os.write(fd, f"pid={os.getpid()}\n".encode())
        except OSError:
            pass
        try:
            yield True
        finally:
            _lock_fd(fd, False)
    finally:
        os.close(fd)


class _Holder(str):
    """A process that may own a git lock, e.g. ``pid 4242 (git commit)``: truthy, and never ``is True``."""


# Git subcommands that can keep ``index.lock`` (or a ref lock) with its fd CLOSED while they wait
# (``commit``/``merge``/``rebase``/``tag``... in the editor). A reader in the tree (``git log`` in a
# pager, ``git status``, ``gitstatusd``, the fsmonitor daemon) is not one: while it really holds a lock
# its open fd names it.
_LOCK_KEEPING_GIT = frozenset({
    "am", "cherry-pick", "commit", "merge", "pull", "rebase", "revert", "stash", "tag", "notes",
})


def _git_subcommand_of(cmdline: bytes) -> str | None:
    args = [a.decode("utf-8", "replace") for a in cmdline.split(b"\0") if a][1:]
    it = iter(args)
    for arg in it:
        if arg in ("-c", "-C", "--git-dir", "--work-tree", "--namespace", "--exec-path", "--config-env"):
            next(it, None)
        elif not arg.startswith("-"):
            return arg
    return None


def _held_open(path: Path, root: Path | None = None) -> _Holder | bool | None:
    """Whether a running process may still own ``path`` (a git lock): the holder, False, or None (unknowable).

    The holder is a truthy :class:`_Holder` naming it (``pid 4242 (git commit)``). False is proof, not
    a guess: Linux reads every process's fds through /proc and also counts any live ``git`` working
    inside ``root`` whose subcommand can keep ``index.lock`` with its fd closed (``commit`` waiting in
    the editor, :data:`_LOCK_KEEPING_GIT`); readers there (a paged ``git log``, ``gitstatusd``, the
    fsmonitor daemon) only count while their fd is on the lock. macOS/BSD ask ``lsof``. Without either
    check the answer is None and the caller never deletes the lock. Windows needs no answer here: it
    refuses to unlink a file another process has open, so the caller's unlink is the probe.
    """
    proc = Path("/proc")
    if (proc / "self" / "fd").is_dir():
        target = os.path.realpath(path)
        inside = os.path.realpath(root) + os.sep if root is not None else None

        def name(pid_dir: Path) -> str:
            try:
                argv = (pid_dir / "cmdline").read_bytes().split(b"\0")
                words = [a.decode("utf-8", "replace") for a in argv if a][:3]
                if words:
                    return " ".join([os.path.basename(words[0]), *words[1:]])
            except OSError:
                pass
            try:
                return (pid_dir / "comm").read_bytes().decode("ascii", "replace").strip()  # /proc: Linux only
            except OSError:
                return "?"

        for pid_dir in proc.glob("[0-9]*"):
            try:
                if any(os.readlink(entry.path) == target for entry in os.scandir(pid_dir / "fd")):
                    return _Holder(f"pid {pid_dir.name} ({name(pid_dir)})")
            except OSError:
                pass
            if inside is None:
                continue
            try:
                comm = (pid_dir / "comm").read_bytes().decode("ascii", "replace").strip()
                cwd = os.readlink(pid_dir / "cwd")
                cmdline = (pid_dir / "cmdline").read_bytes()
            except OSError:
                continue
            if comm == "git" and (cwd + os.sep).startswith(inside) \
                    and _git_subcommand_of(cmdline) in _LOCK_KEEPING_GIT:
                return _Holder(f"pid {pid_dir.name} ({name(pid_dir)})")
        return False
    import shutil

    lsof = shutil.which("lsof") or next((p for p in ("/usr/sbin/lsof", "/usr/bin/lsof") if os.path.isfile(p)), None)
    if lsof is None:
        return None
    try:
        found = subprocess.run([lsof, "-F", "pc", "--", str(path)], capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=20,
                               stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return None
    pids = [line[1:] for line in found.stdout.splitlines() if line.startswith("p")]
    if pids:
        names = [line[1:] for line in found.stdout.splitlines() if line.startswith("c")]
        return _Holder(f"pid {pids[0]} ({names[0] if names else '?'})")
    return False if found.returncode in (0, 1) else None  # lsof exits 1 when nothing has the file open


def _index_lock_holder(git_dir: Path, root: Path | None) -> str:
    """Who keeps ``index.lock`` for the user-facing message: ``pid N (name)`` when it can be named."""
    held = None if sys.platform == "win32" else _held_open(git_dir / "index.lock", root)
    return f"a running git ({held})" if isinstance(held, _Holder) else "a running git"


def _release_dead_index_lock(git_dir: Path, root: Path | None = None) -> bool:
    """Drop a killed git's ``index.lock`` (it refuses every git command) once its owner is PROVEN gone.

    False while a live git may hold it, or when this platform cannot prove it dead: the caller then
    keeps the interrupted-pull marker, so the next launch tries again instead of a rollback being lost.
    """
    lock = git_dir / "index.lock"
    deadline = time.monotonic() + 5
    while lock.exists():
        held = None if sys.platform == "win32" else _held_open(lock, root)
        if held is False or sys.platform == "win32":
            try:
                lock.unlink()
                return True
            except FileNotFoundError:
                return True
            except PermissionError:  # Windows: open in a live process
                pass
        elif held is None:
            return False
        if time.monotonic() > deadline:
            return False
        time.sleep(0.1)
    return True


# The ZIP update's equivalent of the interrupted-pull marker: a journal in the install root naming
# every entry the swap stages/renames and whether it existed before, plus an OS lock its owner holds
# for the whole stage+swap (the kernel drops it with the owner: liveness without pids or ages).
ZIP_SWAP_JOURNAL = ".hermes-update-zip-swap"
_ZIP_STAGING_SUFFIX, _ZIP_OLD_SUFFIX = ".hermes-update-staging", ".hermes-update-old"


class ZipSwapLock:
    """``zip_swap_owner_lock``'s verdict: truthy only while this process holds the kernel lock.

    ``reason`` names a refusal: ``busy`` (a live owner holds it) or why no lock could be taken at all.
    """

    __slots__ = ("owned", "reason")

    def __init__(self, owned: bool, reason: str = "") -> None:
        self.owned, self.reason = owned, reason

    def __bool__(self) -> bool:
        return self.owned


@contextlib.contextmanager
def zip_swap_owner_lock(root: Path, *, wait: float = 0.0):
    """Yields a truthy ``ZipSwapLock`` while this process owns the ZIP swap lock, a falsy one otherwise.

    The lock file is a stable sidecar: never unlinked, so every process locks the same inode (an
    unlink lets a waiter keep the old inode while a newcomer locks a fresh one). No lock, no admission:
    a root where the file cannot be opened or locked refuses with the reason (fail closed)."""
    lock_path = Path(root) / (ZIP_SWAP_JOURNAL + ".lock")
    try:
        fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as exc:
        yield ZipSwapLock(False, f"cannot open the ZIP swap lock {lock_path}: {exc.strerror or exc}")
        return
    try:
        deadline = time.monotonic() + wait
        while True:
            try:
                if sys.platform == "win32":
                    import msvcrt

                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in _CLAIM_HELD_ERRNOS:
                    yield ZipSwapLock(False, f"cannot lock the ZIP swap lock {lock_path}: {exc.strerror or exc}")
                    return
            if time.monotonic() >= deadline:
                yield ZipSwapLock(False, "busy")
                return
            time.sleep(0.05)
        try:
            yield ZipSwapLock(True)
        finally:
            _lock_fd(fd, False)
    finally:
        os.close(fd)


def write_zip_swap_journal(root: Path, phase: str, entries: list) -> None:
    import json

    journal = Path(root) / ZIP_SWAP_JOURNAL
    tmp = journal.with_name(journal.name + ".tmp")
    tmp.write_text(json.dumps({"pid": os.getpid(), "phase": phase, "entries": entries}), encoding="utf-8")
    os.replace(tmp, journal)


def _drop_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        import shutil
        import stat

        try:
            shutil.rmtree(path)
        except OSError:
            # A staged copy keeps its source's modes: a read-only directory refuses to give up its
            # entries. These trees are the update's own copies, so make them writable and retry.
            rwx = stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR
            os.chmod(path, os.stat(path).st_mode | rwx)
            for dirpath, dirnames, files in os.walk(path):
                for name in (*dirnames, *files):
                    child = os.path.join(dirpath, name)
                    if not os.path.islink(child):  # about to be deleted: only the owner bits matter
                        os.chmod(child, os.stat(child).st_mode | (rwx if name in dirnames else stat.S_IWUSR))
            shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def restore_interrupted_zip_swap(project_root: Path | None = None) -> bool:
    """Finish or roll back a ZIP swap whose owner died; True when the tree changed (relaunch).

    ``committed``: every rename landed, only backups remain -> drop them (finish). ``swapping``: put
    each moved-aside entry back and remove entries the swap added (roll back to the old tree, which
    the venv was built for; ``hermes update`` redoes it). ``staging``: nothing live moved -> drop the
    staging copies. Either way no ``*.hermes-update-staging``/``-old`` sibling is left to wedge the
    next run's free-space or dirty-tree checks. A live owner (lock held) is never second-guessed.
    """
    root = _project_root() if project_root is None else Path(project_root)
    journal = root / ZIP_SWAP_JOURNAL
    if not journal.is_file() or _pytest_owns_live_checkout(root):
        return False
    with zip_swap_owner_lock(root) as owned:
        if not owned and owned.reason != "busy":
            print(f"⚠ An interrupted ZIP update cannot be settled: {owned.reason}.", file=sys.stderr)
        if not owned or not journal.is_file():
            return False
        import json

        try:
            data = json.loads(journal.read_text(encoding="utf-8-sig"))
            phase, entries = data["phase"], [(str(n), bool(e)) for n, e in data["entries"]]
        except (OSError, ValueError, KeyError, TypeError):
            phase, entries = "staging", []  # unreadable: never guess which live entry to move
        changed = False
        failed = False
        for name, existed in reversed(entries):
            if not name or "/" in name or "\\" in name or name in (".", ".."):
                continue
            dst = root / name
            staging, old = Path(f"{dst}{_ZIP_STAGING_SUFFIX}"), Path(f"{dst}{_ZIP_OLD_SUFFIX}")
            try:
                if phase == "swapping":
                    # Staging finished before this phase, so a backup now is the one the swap made
                    # from the old entry: it existed, whatever an older journal recorded.
                    existed = existed or old.exists()
                    if existed and old.exists():
                        if old.is_file() and not old.is_symlink() and not dst.is_dir():
                            os.replace(old, dst)  # a file entry never goes missing, not even here
                        else:
                            _drop_path(dst)
                            os.rename(old, dst)
                        changed = True
                    elif not existed and dst.exists() and not staging.exists():
                        _drop_path(dst)
                        changed = True
                elif existed and not dst.exists() and old.exists():
                    os.rename(old, dst)  # the backup is the only copy left
                    changed = True
                for leftover in (staging, old):
                    _drop_path(leftover)
            except OSError as exc:
                failed = True
                print(f"⚠ Could not settle {name} after an interrupted ZIP update: {exc}", file=sys.stderr)
        if failed:
            return changed  # the journal stays: the next launch retries
        journal.unlink(missing_ok=True)
    if changed:
        print("⚠ A previous ZIP `hermes update` was interrupted mid-swap; the old install was put back. "
              "`hermes update` updates it again.", file=sys.stderr)
    return changed


def restore_interrupted_pull(project_root: Path | None = None, *, after_failure: bool = False) -> bool:
    """Put back the files a killed ``hermes update`` had half-moved to the new commit.

    ``after_failure``: the updater itself calls this when git exited non-zero mid-move (a locked or
    read-only file): same restore, and the marker stays whenever the tree is not verified whole.

    Returns True when the tree changed under this process: modules it already imported may be the
    half-written ones, so the caller must relaunch (``relaunch_after_restore``).

    Fast path (no marker) is one or two ``stat`` calls. Acts only when the marker's owner is gone,
    HEAD is still the pre-pull commit and no merge/rebase is in progress; then every path git wrote
    (the target's content, or torn on the way there) returns to HEAD (the commit the venv was built
    for), so the install is whole again and ``hermes update`` redoes the update from the start. Local
    edits are never touched; the updater's autostash (if any) stays in ``git stash list``. Concurrent
    launches take turns (``_restore_claim``); a launch that waited out another's restore relaunches.

    Limits, by design: a torn ``hermes_cli/__init__.py``, ``hermes_bootstrap.py``, ``agent/__init__.py``
    or ``agent/jiter_preload.py`` (imported before the ``hermes-agent`` hook) fails before this runs. A file git also changes that the user deleted, emptied or cut to a prefix of git's version
    looks exactly like git's own half-written file and is restored too, as is a user edit to a
    conflicted path or, on git < 2.38, to a path both sides of a custom-branch merge changed.
    """
    try:
        root = _project_root() if project_root is None else project_root
        marker = interrupted_pull_marker(root)
        if not marker.is_file() and not (Path(root) / ZIP_SWAP_JOURNAL).is_file():
            return False  # fast path: nothing to repair, no lock taken
        if _pytest_owns_live_checkout(root):
            return False
        # A live `hermes update` (or its build/completion/git, after its updater died) owns the
        # checkout: its own transaction settles the tree, and repairing under it races its git.
        busy_note = "⚠ Not repairing the checkout now: {}. Launch again once it finishes."
        if (Path(root) / ZIP_SWAP_JOURNAL).is_file():
            with _checkout_custody(Path(root)) as busy:
                if busy:
                    print(busy_note.format(busy), file=sys.stderr)
                    return False
                if restore_interrupted_zip_swap(root):
                    return True
        if not marker.is_file():
            return False
        with _restore_claim(marker.parent) as claimed:
            if not claimed:
                print("⚠ Another Hermes launch is still repairing the checkout after an interrupted "
                      "`hermes update`; if this one fails, launch again in a moment.", file=sys.stderr)
                return False
            if not marker.is_file():
                return True  # another launch finished while this one started: rerun from its tree
            # The claim orders launches; the checkout lock keeps out an update tree. Claim first, so a
            # launch that waited out another's repair reruns from its tree without contending.
            with _checkout_custody(Path(root)) as busy:
                if busy:
                    print(busy_note.format(busy), file=sys.stderr)
                    return False
                return _restore_holding_claim(root, marker, after_failure=after_failure)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        # Never block launch: the import that follows surfaces any real breakage.
        print(f"⚠ Could not check for an interrupted `hermes update`: {exc}", file=sys.stderr)
    return False


@contextlib.contextmanager
def _checkout_custody(root: Path):
    """Hold the checkout kernel lock (``hermes_cli.update_lock``) for a repair, like an updater (R2).

    Yields ``""`` while this process holds or joined it (the updater's own ``after_failure``
    settle and its children join), else the busy reason: a live update tree owns the checkout and
    the repair leaves the tree alone. The restore's mutating git children inherit the lock (POSIX)
    or join this process's kill-on-close job (Windows) through ``update_custody.run_git``, so a
    launch killed mid-restore keeps the checkout locked until its git is gone.

    A tree too torn to import the lock module still gets its repair (this is the code that untears
    it); no updater can be running from such a tree.
    """
    try:
        from hermes_cli import update_lock
    except Exception as exc:  # noqa: BLE001 - torn tree: repair without the lock rather than never
        print(f"⚠ Repairing without the checkout lock ({type(exc).__name__}: {exc}).", file=sys.stderr)
        yield ""
        return
    holder = update_lock._acquire_checkout(Path(root))
    if holder is not None:
        if not update_lock.checkout_lock_held(Path(root)):
            # Not a holder but no lock at all (a git dir without working locks, e.g. NFS without
            # lockd): no updater can hold it either (they fail closed), so the repair runs unguarded.
            yield ""
            return
        yield update_lock.describe_holder(holder) or "another update holds the checkout lock"
        return
    try:
        yield ""
    finally:
        update_lock._release_checkout()


def _restore_holding_claim(root: Path, marker: Path, *, after_failure: bool = False) -> bool:
    global _merge_advice_shown
    git_dir = marker.parent
    fields = dict(line.partition("=")[::2] for line in marker.read_text(encoding="utf-8-sig").splitlines())
    try:
        owner = int(fields.get("pid", ""))
    except ValueError:
        owner = -1
    # Our own pid is never the owner: this runs at startup, and containers hand a retry the
    # killed updater's pid.
    if (owner != os.getpid() and _pid_is_running(owner)
            and time.time() - marker.stat().st_mtime < _INTERRUPTED_PULL_MAX_AGE_SECONDS):
        return False
    pre, target = fields.get("pre", "").strip(), fields.get("target", "").strip()
    stash = fields.get("stash", "").strip()

    executable = _git_executable(fields.get("git", "").strip())
    try:
        from hermes_cli.update_custody import run_git
    except Exception:  # noqa: BLE001 - torn tree: plain spawns (see _checkout_custody)
        run_git = None

    def git(*args: str, stdin: str | None = None, text: bool = True) -> subprocess.CompletedProcess:
        base = [executable, "--literal-pathspecs", "-C", str(root)]
        kwargs = dict(input=stdin, cwd=str(root), capture_output=True, timeout=120,
                      stdin=None if stdin is not None else subprocess.DEVNULL,
                      **({"text": True, "encoding": "utf-8", "errors": "replace"} if text else {}))
        if run_git is not None:
            return run_git(base, list(args), **kwargs)
        return subprocess.run([*base, *args], **kwargs)

    # A killed git's index.lock goes first, before any git runs (an unresolvable or failing git
    # would otherwise strand it, and it refuses every later git command). Proven-dead only; after a
    # git that EXITED (``after_failure``) a lock now is another git's, never ours to drop.
    foreign_lock = after_failure and (git_dir / "index.lock").exists()
    index_free = foreign_lock or _release_dead_index_lock(git_dir, root)

    rollback = fields.get("rollback", "").strip()
    rollback = rollback if rollback in ("branch", "detach") else ""
    head = git("rev-parse", "HEAD")
    if head.returncode != 0:
        print(f"⚠ Could not read HEAD to repair an interrupted `hermes update` ({head.stderr.strip()}); "
              "the next launch retries.", file=sys.stderr)
        return False
    if rollback and pre and target and head.stdout.strip() == target:
        # A syntax rollback killed before it moved HEAD back: redo that step (HEAD and index, no file),
        # so the restore below lands on ``pre`` (the code the update started from), not the broken tree.
        # Every step is checked, and the killed step's ``index.lock`` goes first, only once its git is
        # proven gone: until HEAD is on ``pre`` this marker is the rollback's only record.
        reason = _redo_rollback_head(git, git_dir, root, pre, rollback, after_failure=after_failure)
        if reason is not None:
            print(f"⚠ An interrupted `hermes update` rollback to {pre[:10]} cannot resume yet ({reason}); "
                  "the next launch retries.", file=sys.stderr)
            return False
        head = git("rev-parse", "HEAD")
    if not pre or not target or head.stdout.strip() != pre:
        marker.unlink()  # git finished (HEAD moved) or the marker is unusable
        return False
    if any((git_dir / name).exists() for name in _GIT_OPERATION_IN_PROGRESS):
        merge_head = git_dir / "MERGE_HEAD"
        if (not _merge_advice_shown and merge_head.is_file()
                and merge_head.read_text(encoding="utf-8-sig").strip() == target):
            # The killed updater's own merge: its conflict markers may sit in startup modules.
            _merge_advice_shown = True
            print(f"⚠ A killed `hermes update` left its merge unfinished. Run `git -C {root} merge --abort`, "
                  "then launch again." + (f" Your local changes are in its stash ({stash})." if stash else ""),
                  file=sys.stderr)
        return False
    # A killed claim holder's own git child can still be writing; scanning under it reads half a tree.
    # After a git that EXITED (``after_failure``) no git of ours is left: a lock now is another git's
    # (often the very reason ours failed), never ours to drop. The scan below only reads.
    if not index_free:
        print(f"⚠ {_index_lock_holder(git_dir, root)} holds the index after an interrupted `hermes update`; "
              "the next launch finishes the restore.", file=sys.stderr)
        return False
    written = _paths_git_wrote(git, root, pre, target)
    if written is None:  # after a gc or re-clone: nothing left to compare against
        if rollback and not _rollback_verified(git, git_dir, pre):
            print(f"⚠ An interrupted `hermes update` rollback to {pre[:10]} is not verified and commit "
                  f"{target[:10]} is gone. Inspect `git -C {root} status`; `git -C {root} reset --hard "
                  f"{pre[:10]}` finishes it.", file=sys.stderr)
            return False
        marker.unlink()
        print(f"⚠ Ignoring a stale interrupted-update marker: commit {target[:10]} is gone.", file=sys.stderr)
        return False
    restore, added, new_dirs = written
    if (restore or added) and foreign_lock:
        print("⚠ Another git holds the index, so the files the update already wrote cannot be put back "
              "now; the next launch restores them.", file=sys.stderr)
        return False
    if restore or added:
        print(("⚠ git stopped partway through writing the new code — " if after_failure else
               "⚠ A previous `hermes update` was killed while git was writing the new code — ")
              + f"restoring the checkout to {pre[:10]}...", file=sys.stderr)
        failed = _put_back_paths(git, root, restore, added)
        if failed:
            # No manual recipe: a reset would also wipe the edits this restore keeps, and the
            # marker stays so the next launch retries.
            reason = (failed.stderr.strip().splitlines() or ["git failed"])[-1]
            print(f"  ✗ Could not restore it automatically ({reason}); the next launch retries.",
                  file=sys.stderr)
            return False
    for rel in sorted(new_dirs, key=lambda d: d.count("/"), reverse=True):
        with contextlib.suppress(OSError):
            (root / rel).rmdir()  # only when empty: an untracked file inside keeps it
    if rollback and not _rollback_verified(git, git_dir, pre):
        # A rollback owns the whole tree (the update parked local edits first): only HEAD on ``pre``
        # with no tracked change left is a finished rollback. Anything less keeps its only record.
        print(f"  ✗ The rollback to {pre[:10]} is not verified yet (tracked files still differ); the next "
              f"launch retries. Inspect `git -C {root} status`; `git -C {root} reset --hard {pre[:10]}` "
              "finishes it by hand.", file=sys.stderr)
        return bool(restore or added)
    marker.unlink()
    if not restore and not added:
        return False  # the killed git never reached the tree: nothing to put back
    print("  ✓ Checkout restored; `hermes update` updates it again.", file=sys.stderr)
    if stash:
        print(f"  Your local changes are still in the update's stash ({stash}).", file=sys.stderr)
    return True


def _redo_rollback_head(git, git_dir: Path, root: Path, pre: str, mode: str, *, after_failure: bool) -> str | None:
    """Move HEAD (and the index) back to ``pre`` for a killed syntax rollback; None, or why not yet.

    The killed step's ``index.lock`` is reclaimed first, and only when its git is proven gone. After a
    git that exited in this very process (``after_failure``) a lock is never ours to judge.
    """
    lock = git_dir / "index.lock"
    if lock.exists() and (after_failure or not _release_dead_index_lock(git_dir, root)):
        return f"{_index_lock_holder(git_dir, root)} may still hold .git/index.lock"
    if mode == "detach":
        moved = git("update-ref", "--no-deref", "HEAD", pre)
        if moved.returncode != 0:
            return f"git update-ref failed: {(moved.stderr.strip().splitlines() or ['?'])[-1]}"
    reset = git("reset", "-q", pre)
    if reset.returncode != 0:
        return f"git reset failed: {(reset.stderr.strip().splitlines() or ['?'])[-1]}"
    head = git("rev-parse", "HEAD")
    if head.returncode != 0 or head.stdout.strip() != pre:
        return "HEAD did not move back"
    return None


def _rollback_verified(git, git_dir: Path, pre: str) -> bool:
    """The whole-tree rollback landed: HEAD is ``pre``, no index lock, no tracked change."""
    head = git("rev-parse", "HEAD")
    if head.returncode != 0 or head.stdout.strip() != pre or (git_dir / "index.lock").exists():
        return False
    status = git("status", "--porcelain", "-z", "--untracked-files=no")
    return status.returncode == 0 and not status.stdout.strip("\0")


def _put_back_paths(git, root: Path, restore: list[str], added: list[str]) -> subprocess.CompletedProcess | None:
    """Return ``restore`` to HEAD and drop ``added``; the failed git run, or None."""
    if restore:
        run = git("restore", "--source=HEAD", "--staged", "--worktree", "--pathspec-from-file=-",
                  "--pathspec-file-nul", stdin="\0".join(restore))
        if run.returncode:
            return run
    if added:
        run = git("rm", "-q", "--cached", "--ignore-unmatch", "--pathspec-from-file=-",
                  "--pathspec-file-nul", stdin="\0".join(added))
        for rel in added:
            (root / rel).unlink(missing_ok=True)
        if run.returncode:
            return run
    return None


def relaunch_after_restore() -> None:
    """Re-run this command from the restored tree; never returns.

    Everything imported so far (this package, ``hermes_bootstrap``, ``hermes_cli.main`` itself) may
    be the killed git's new files, and they would run against the restored old tree.
    """
    argv = [sys.executable, *sys.orig_argv[1:]]
    sys.stdout.flush()
    sys.stderr.flush()
    if sys.platform == "win32":
        # os.execv on Windows spawns and exits, detaching the console's wait on us.
        sys.exit(subprocess.call(argv))  # windows-footgun: ok — interactive child keeps our console
    os.execv(sys.executable, argv)


def _pytest_owns_live_checkout(root: Path) -> bool:
    """Keep lifecycle tests from repairing the checkout that runs the suite.

    Copied installations in tmp_path remain eligible for real recovery tests.
    """
    return "PYTEST_CURRENT_TEST" in os.environ and root == Path(__file__).resolve().parent.parent


def _missing_environment(root: Path) -> bool:
    """True when PM recorded an environment whose site-packages is gone."""
    from pm.environments import runtime_facts_path, selected_venv, site_packages

    if not runtime_facts_path(root).is_file():
        return False
    try:
        return not site_packages(selected_venv(root)).is_dir()
    except RuntimeError:
        return True  # PM validates the recorded state before rebuilding.


def _count_failed_attempt(marker: Path) -> None:
    """Bump ``marker``'s attempt count in its own format so the retry limit can trip."""
    import json

    # Best-effort: an unwritable marker only means the limit trips later.
    with contextlib.suppress(OSError):
        attempts = _read_marker_attempts(marker) + 1
        body = marker.read_text(encoding="utf-8-sig")
        if any(line.startswith("pid=") for line in body.splitlines()):
            lines = [line for line in body.splitlines() if not line.startswith("attempts=")]
            body = "\n".join([*lines, f"attempts={attempts}"]) + "\n"
        else:
            body = json.dumps({"attempts": attempts})
        marker.write_text(body, encoding="utf-8")


def recover_if_needed(project_root: Path | None = None, argv: list[str] | None = None, *, explicit: bool = False) -> bool:
    """Ask PM to restore dependencies before activation; leave failed requests retryable."""
    global _UPDATE_RETRY_RECOVERED

    root = _project_root() if project_root is None else Path(project_root).resolve()
    if not explicit and _pytest_owns_live_checkout(root):
        return False
    from hermes_cli._parser import command_argv

    args = command_argv(sys.argv[1:] if argv is None else argv)
    if not explicit and args[:1] == ["pm"]:
        return False  # PM's command boundary owns the explicit repair.
    from pm.environments import install_state_dir

    missing_marker = install_state_dir(root) / ".repair-incomplete"
    marker_paths = (root / ".update-incomplete", root / ".lazy-refresh-incomplete", missing_marker)
    markers = [path for path in marker_paths if path.is_file()]

    if not (root / "pyproject.toml").is_file() or not (explicit or markers or _missing_environment(root)):
        return False
    lock = _claim_recovery_lock(root)
    if lock is None:
        return False
    try:
        # Recheck after locking: another launch can finish between discovery and claim.
        markers = [path for path in marker_paths if path.is_file()]
        if not markers:
            if not explicit and not _missing_environment(root):
                return False
            missing_marker.write_text('{"attempts": 0}', encoding="utf-8")
            markers = [missing_marker]
        if any(_marker_owner_is_live(marker) for marker in markers):
            return False
        if not explicit and any(_read_marker_attempts(marker) >= _EARLY_CORE_INSTALL_MAX_ATTEMPTS for marker in markers):
            print("hermes: automatic dependency repair retry limit reached; run `hermes pm repair`", file=sys.stderr)
            return False
        from pm.recovery import repair_dependencies

        print("hermes: repairing the recorded dependency environment...", file=sys.stderr)
        repair_dependencies(root)
        for marker in markers:
            marker.unlink(missing_ok=True)
        _UPDATE_RETRY_RECOVERED = args[:1] == ["update"]
        print("hermes: dependency environment repaired", file=sys.stderr)
        return True
    except Exception as exc:
        from pm.environments import install_state_permission_message

        if isinstance(exc, PermissionError) and install_state_permission_message(root, exc):
            raise  # The bootstrap or PM CLI reports the access error once.
        for marker in markers:
            _count_failed_attempt(marker)
        print(f"hermes: dependency repair failed: {exc}; run `hermes pm repair`", file=sys.stderr)
        return False
    finally:
        os.close(lock)


# A failed network or build must not retry on every launch forever.
_EARLY_CORE_INSTALL_MAX_ATTEMPTS = 3


def _claim_recovery_lock(root: Path) -> int | None:
    """Hold a kernel lock in writable state; process exit releases it."""
    from pm.environments import install_state_dir
    from hermes_cli.runtime_state import _lock

    state = install_state_dir(root)
    state.mkdir(parents=True, exist_ok=True)
    fd = os.open(state / ".recovery.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if _lock(fd, wait=False):
            return fd
    except BaseException:
        os.close(fd)
        raise
    os.close(fd)
    return None
