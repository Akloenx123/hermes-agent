"""Regression: a state.db write-lock timeout names the process holding the lock.

Before, ``database is locked (another Hermes process held the state.db write lock for over 60s)``
identified the victim only; every Hermes process has the DB open, so the descriptor scan could not
single out the writer. ``/proc/locks`` can.
"""

import io
import os
import sqlite3
import subprocess
import sys
import textwrap
import time

import pytest

from hermes_state_lockowners import parse_proc_locks, state_db_write_lock_holders


def test_parse_proc_locks_keeps_only_write_locks_on_our_inodes_and_decodes_the_wal_write_byte():
    inodes = {(os.makedev(0x103, 0x02), 4194737): "-shm", (os.makedev(0x103, 0x02), 4228330): ""}
    text = textwrap.dedent("""\
        1: POSIX  ADVISORY  WRITE 594094 103:02:4194737 120 120
        2: POSIX  ADVISORY  READ 594094 103:02:4194737 125 125
        3: POSIX  ADVISORY  READ 99493 103:02:4194737 128 128
        4: OFDLCK ADVISORY  WRITE -1 103:02:4228330 1073741825 1073741825
        5: POSIX  ADVISORY  WRITE 4242 103:02:99999 120 120
        6: -> POSIX  ADVISORY  WRITE 777 103:02:4194737 120 120
    """)
    assert parse_proc_locks(text, inodes) == [
        (594094, "WAL write", "-shm"),
        (-1, "RESERVED", ""),
    ]


def test_parse_proc_locks_matches_on_inode_alone_when_the_device_is_none():
    """btrfs: stat() reports a subvolume device, /proc/locks the superblock device."""
    inodes = {(None, 4194737): "-shm"}
    text = "1: POSIX  ADVISORY  WRITE 594094 00:1e:4194737 120 120\n2: POSIX  ADVISORY  WRITE 1 00:1e:5 120 120\n"
    assert parse_proc_locks(text, inodes) == [(594094, "WAL write", "-shm")]


def test_fstype_of_picks_the_longest_matching_mount(tmp_path, monkeypatch):
    import hermes_state_lockowners as mod

    info = (
        "20 1 0:1e / / rw - btrfs /dev/nvme0n1p2 rw\n"
        "21 20 0:20 /home/a\\040b /home/a\\040b rw - ext4 /dev/sda1 rw\n"
    )
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(info, encoding="utf-8")
    monkeypatch.setattr(mod, "_MOUNTINFO", str(mountinfo))
    assert mod._fstype_of("/home/a b/state.db") == "ext4"
    assert mod._fstype_of("/home/x/state.db") == "btrfs"


@pytest.mark.platforms("linux")
def test_btrfs_device_mismatch_still_names_the_holder(tmp_path, monkeypatch):
    import hermes_state_lockowners as mod

    db = tmp_path / "state.db"
    db.write_bytes(b"")
    st = os.stat(db)
    held = f"1: POSIX  ADVISORY  WRITE {os.getpid()} 00:1e:{st.st_ino} 1073741825 1073741825\n"
    monkeypatch.setattr(mod, "_fstype_of", lambda path: "btrfs")
    monkeypatch.setattr(mod, "open", lambda *a, **k: io.StringIO(held), raising=False)
    lines = state_db_write_lock_holders(db)
    assert len(lines) == 1 and f"PID {os.getpid()} " in lines[0], lines


@pytest.mark.platforms("linux")
def test_holder_skipped_by_one_proc_locks_pass_is_still_named(tmp_path, monkeypatch):
    """/proc/locks is served over several read()s, so churn elsewhere can skip an entry in one pass."""
    import hermes_state_lockowners

    db = tmp_path / "state.db"
    db.write_bytes(b"")
    st = os.stat(db)
    held = (f"1: POSIX  ADVISORY  WRITE {os.getpid()} "
            f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino} 1073741825 1073741825\n")
    passes = iter(["", held, ""])
    monkeypatch.setattr(hermes_state_lockowners, "open", lambda *a, **k: io.StringIO(next(passes)), raising=False)
    lines = state_db_write_lock_holders(db)
    assert len(lines) == 1 and f"PID {os.getpid()} " in lines[0] and "RESERVED" in lines[0], lines


@pytest.mark.platforms("linux")
def test_live_writer_in_another_process_is_named_by_pid(tmp_path):
    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA journal_mode=wal")
    conn.execute("CREATE TABLE t(x)")
    conn.commit()
    conn.close()

    holder = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent(f"""
            import sqlite3, sys, time
            c = sqlite3.connect({str(db)!r}, isolation_level=None)
            c.execute("BEGIN IMMEDIATE")
            c.execute("INSERT INTO t VALUES (1)")
            print("held", flush=True)
            time.sleep(30)
        """)],
        stdout=subprocess.PIPE, text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        deadline = time.monotonic() + 5
        lines = []
        while time.monotonic() < deadline:
            lines = state_db_write_lock_holders(db)
            if lines:
                break
            time.sleep(0.05)
        assert any(f"PID {holder.pid} " in line and "WAL write" in line for line in lines), lines
    finally:
        holder.kill()
        holder.wait()
    assert state_db_write_lock_holders(db) == []
