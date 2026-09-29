"""Tests for the interactive key reader in main.py.

These run against a real pseudo-terminal, because the bug they guard against
only shows up there: a terminal sends an arrow key as the three-byte sequence
ESC [ A/B/C/D, and the game must see exactly one keypress (and never a bare
Escape, which used to quit the game).
"""

import os
import pty
import select
import subprocess
import sys
import time

import pytest

import main
from main import Terminal


@pytest.fixture
def tty():
    """A Terminal attached to a fresh pseudo-terminal, cleaned up afterwards."""
    master, slave = pty.openpty()
    stream = os.fdopen(slave, "r")
    terminal = Terminal(stream=stream)
    with terminal:
        yield master, terminal
    stream.close()
    os.close(master)


def type_bytes(master, data, gap=0.0):
    """Send raw bytes to the terminal, optionally one chunk at a time."""
    if gap:
        for byte in data:
            os.write(master, bytes([byte]))
            time.sleep(gap)
    else:
        os.write(master, data)


def test_plain_letters_come_back_one_at_a_time(tty):
    master, terminal = tty
    type_bytes(master, b"a")
    assert terminal.read_key(0.5) == "a"


def test_arrow_keys_are_one_keypress_each(tty):
    master, terminal = tty
    for sequence, name in (
        (main.ARROW_LEFT, "left"),
        (main.ARROW_RIGHT, "right"),
        (main.ARROW_UP, "up"),
        (main.ARROW_DOWN, "down"),
    ):
        type_bytes(master, sequence.encode())
        assert terminal.read_key(0.5) == sequence, name


def test_arrow_keys_survive_being_split_across_reads(tty):
    """The regression: a slow terminal must not turn an arrow into an Escape."""
    master, terminal = tty
    type_bytes(master, main.ARROW_LEFT.encode(), gap=0.005)
    assert terminal.read_key(0.5) == main.ARROW_LEFT


def test_arrow_keys_are_not_quit_keys():
    """Left/right used to end the game because ESC was in QUIT_KEYS."""
    for sequence in main.ARROW_KEYS:
        assert sequence not in main.QUIT_KEYS
    assert "\x1b" not in main.QUIT_KEYS


def test_arrow_keys_map_to_the_expected_actions():
    from tetris.actions import Action

    assert main.KEY_TO_ACTION[main.ARROW_LEFT] is Action.LEFT
    assert main.KEY_TO_ACTION[main.ARROW_RIGHT] is Action.RIGHT
    assert main.KEY_TO_ACTION[main.ARROW_DOWN] is Action.DOWN
    assert main.KEY_TO_ACTION[main.ARROW_UP] is Action.ROTATE_CW


def test_a_burst_of_keys_is_delivered_in_order(tty):
    master, terminal = tty
    type_bytes(master, b"aadx")
    assert [terminal.read_key(0.5) for _ in range(4)] == ["a", "a", "d", "x"]


def test_arrows_and_letters_mixed_in_one_burst(tty):
    master, terminal = tty
    type_bytes(master, (main.ARROW_LEFT + "a" + main.ARROW_RIGHT + " ").encode())
    assert [terminal.read_key(0.5) for _ in range(4)] == [
        main.ARROW_LEFT, "a", main.ARROW_RIGHT, " ",
    ]


def test_read_key_returns_none_when_nothing_is_typed(tty):
    _, terminal = tty
    assert terminal.read_key(0.1) is None


def test_bare_escape_is_not_quit():
    """A lone Escape (or an unknown sequence) yields ESC, which maps to nothing."""
    master, slave = pty.openpty()
    stream = os.fdopen(slave, "r")
    terminal = Terminal(stream=stream)
    try:
        with terminal:
            os.write(master, b"\x1b")
            assert terminal.read_key(0.2) == "\x1b"
            assert "\x1b" not in main.KEY_TO_ACTION
    finally:
        stream.close()
        os.close(master)


def test_game_keeps_running_when_arrow_keys_are_pressed():
    """End to end: press left/right/up/down, then q -- nothing must quit early."""
    project = os.path.dirname(os.path.abspath(main.__file__))
    master, slave = pty.openpty()
    process = subprocess.Popen(
        [sys.executable, "main.py", "--seed", "3"],
        cwd=project,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env={**os.environ, "TERM": "xterm-256color"},
        close_fds=True,
    )
    os.close(slave)
    output = b""

    def drain(timeout=0.15):
        nonlocal output
        while select.select([master], [], [], timeout)[0]:
            try:
                chunk = os.read(master, 65536)
            except OSError:
                return
            if not chunk:
                return
            output += chunk

    try:
        time.sleep(0.7)
        drain()
        repaints = output.count(b"\x1b[H")
        assert repaints > 0, "the game never drew a frame"

        for keys in (main.ARROW_LEFT, main.ARROW_RIGHT, main.ARROW_UP, main.ARROW_DOWN):
            os.write(master, keys.encode())
            time.sleep(0.35)
            drain()
            # Still running, and still repainting: the arrow was handled, not
            # treated as Escape -> quit.
            assert process.poll() is None, f"the game exited on {keys!r}"
            assert output.count(b"\x1b[H") > repaints
            repaints = output.count(b"\x1b[H")

        os.write(master, b"q")
        assert process.wait(timeout=5) == 0
    finally:
        if process.poll() is None:
            process.kill()
        os.close(master)
