"""Playable colored demo of the engine -- the human frontend.

This drives the exact same `Game` API the AI will use later; all it adds is a
real-time input loop and the colored view from `tetris/render.py`.

    python main.py

Controls
    left / a        move left          right / d   move right
    down / s        soft drop         space       hard drop
    up / w / x      rotate clockwise  z           rotate counter-clockwise
    p               pause             r           restart
    q               quit

Gravity is a timer (`Game.apply_gravity`): the piece falls one cell every
`gravity_delay(level)` seconds, so the game speeds up as you level. If stdin is
not a terminal (a pipe or a file), the loop falls back to the original
prompt-per-turn mode instead of failing.
"""

from __future__ import annotations

import argparse
import os
import random
import select
import sys
import termios
import time
import tty
from typing import List, Optional  # noqa: UP035

from tetris import render
from tetris.actions import Action
from tetris.game import Game

# Up arrow rotates clockwise, like most Tetris implementations.
ARROW_LEFT = "\x1b[D"
ARROW_RIGHT = "\x1b[C"
ARROW_DOWN = "\x1b[B"
ARROW_UP = "\x1b[A"

# The arrow keys we recognise. A terminal sends them as an escape *sequence*
# (three bytes), so they are matched as whole sequences, never one byte at a
# time -- a lone ESC must not be mistaken for an arrow key (or vice versa).
ARROW_KEYS = (ARROW_LEFT, ARROW_RIGHT, ARROW_DOWN, ARROW_UP)

KEY_TO_ACTION = {
    "a": Action.LEFT,
    ARROW_LEFT: Action.LEFT,
    "d": Action.RIGHT,
    ARROW_RIGHT: Action.RIGHT,
    "s": Action.DOWN,
    ARROW_DOWN: Action.DOWN,
    "w": Action.ROTATE_CW,
    "x": Action.ROTATE_CW,
    ARROW_UP: Action.ROTATE_CW,
    "z": Action.ROTATE_CCW,
    " ": Action.DROP,           # hard drop
}
QUIT_KEYS = ("q", "\x03")       # q or Ctrl-C; a bare Escape is ignored
PAUSE_KEYS = ("p",)
RESUME_KEYS = ("p", " ")
RESTART_KEYS = ("r",)

# Input tuning: how long to wait for the rest of an escape sequence, and how
# many bytes to pull out of the tty at a time.
ESCAPE_GRACE = 0.03
READ_SIZE = 64
SEQUENCE_LEN = 3                # ESC [ X

# Gravity: seconds per cell, shrinking by GRAVITY_DECAY per level.
GRAVITY_START = 0.80
GRAVITY_MIN = 0.06
GRAVITY_DECAY = 0.85

# Fallback (non-tty) mode: gravity every N prompts, as the old demo did.
GRAVITY_EVERY = 5


def gravity_delay(level: int) -> float:
    """Seconds between gravity ticks at this level."""
    return max(GRAVITY_MIN, GRAVITY_START * (GRAVITY_DECAY ** (level - 1)))


class _Quit(Exception):
    """Raised internally to leave the game loop."""


class Terminal:
    """Raw single-key input for the duration of a `with` block.

    Puts the tty in cbreak mode (keys arrive immediately, no Enter, but Ctrl-C
    and output post-processing still work), hides the cursor, and always
    restores both -- even if the game loop raises.

    Keys are read straight from the tty into our own buffer and handed out one
    *logical* keypress at a time. That matters because an arrow key is not one
    byte but three (ESC [ A/B/C/D), and they can arrive in separate reads: the
    bytes are reassembled here so `left arrow` is always reported as
    `ARROW_LEFT` and never as a bare Escape.
    """

    def __init__(self, stream=None):
        self._stream = stream if stream is not None else sys.stdin
        self._fd: Optional[int] = None
        self._attrs = None
        self._pending = ""      # bytes read from the tty but not yet returned

    def __enter__(self) -> "Terminal":
        self._fd = self._stream.fileno()
        self._attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self.write("\x1b[?25l")   # hide the cursor while we repaint
        return self

    def __exit__(self, *exc_info) -> None:
        if self._attrs is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._attrs)
        self.write("\x1b[?25h\x1b[0m\n")   # show the cursor again, reset colors

    @staticmethod
    def write(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    def _fill(self, timeout: Optional[float]) -> bool:
        """Read whatever the tty has ready into the buffer (False on timeout)."""
        ready, _, _ = select.select([self._fd], [], [], timeout)
        if not ready:
            return False
        data = os.read(self._fd, READ_SIZE)
        if not data:                    # EOF: no more input will ever arrive
            return False
        self._pending += data.decode("utf-8", "replace")
        return True

    def read_key(self, timeout: Optional[float] = None) -> Optional[str]:
        """One logical keypress, or None after `timeout` seconds.

        `timeout=None` waits indefinitely. Buffered bytes are returned before
        we ever touch the tty, so a burst of keys is never dropped.
        """
        if not self._pending and not self._fill(timeout):
            return None

        if self._pending.startswith("\x1b"):
            # An escape sequence: wait for the remaining bytes (they may still
            # be in flight), then hand back the whole sequence if we know it.
            deadline = time.monotonic() + ESCAPE_GRACE
            while len(self._pending) < SEQUENCE_LEN:
                left = deadline - time.monotonic()
                if left <= 0 or not self._fill(left):
                    break
            if self._pending[:SEQUENCE_LEN] in ARROW_KEYS:
                key, self._pending = (
                    self._pending[:SEQUENCE_LEN],
                    self._pending[SEQUENCE_LEN:],
                )
                return key
            # An unknown sequence (Home, F1, a bare Escape...): report the ESC
            # itself. It maps to no action, so it is simply ignored -- and the
            # bytes behind it are returned as ordinary keys next time round.

        key, self._pending = self._pending[0], self._pending[1:]
        return key


def draw(frame: str) -> None:
    """Repaint the screen in place (cursor home, then clear what's left over)."""
    sys.stdout.write("\x1b[H" + frame + "\x1b[0m\x1b[J")
    sys.stdout.flush()


def play(game: Game, renderer: render.Renderer) -> None:
    """Real-time game loop: keys act immediately, gravity ticks on a timer."""
    with Terminal() as term:
        while True:
            draw(renderer.frame(game))
            if game.is_game_over():
                break

            key = term.read_key(gravity_delay(game.level))
            if key is None:
                game.apply_gravity()
                continue
            if key in QUIT_KEYS:
                raise _Quit
            if key in PAUSE_KEYS:
                _paused(game, renderer, term)
                continue
            if key in RESTART_KEYS:
                game.reset()
                continue

            action = KEY_TO_ACTION.get(key)
            if action is not None and not game.is_game_over():
                game.step(action)


def _paused(game: Game, renderer: render.Renderer, term: Terminal) -> None:
    """Block until the player presses p again (or quits)."""
    while True:
        draw(renderer.frame(game, message="PAUSED -- press p to resume"))
        key = term.read_key(None)
        if key is None:
            continue
        if key in RESUME_KEYS:
            return
        if key in QUIT_KEYS:
            raise _Quit


def play_prompts(game: Game, renderer: render.Renderer) -> None:
    """Fallback loop for piped input: print a frame, read a command, repeat."""
    turns = 0
    print(renderer.frame(game))
    while not game.is_game_over():
        typed = input("> ").lower()
        if typed.strip() in ("q", "quit"):
            break
        # A bare space (hard drop) survives, other commands are stripped.
        action = KEY_TO_ACTION.get(typed if typed == " " else typed.strip())
        if action is not None:
            game.step(action)
        turns += 1
        if turns % GRAVITY_EVERY == 0:
            game.apply_gravity()
        print()
        print(renderer.frame(game))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--plain", action="store_true", help="plain ASCII blocks instead of colors"
    )
    parser.add_argument(
        "--no-ghost", action="store_true", help="hide the landing preview"
    )
    parser.add_argument(
        "--grid", choices=render.GRID_STYLES, default="none",
        help="mark the empty cells ('none', 'dots' or 'lines')",
    )
    parser.add_argument("--seed", type=int, help="seed the piece randomizer")
    args = parser.parse_args(argv)

    color = False if args.plain else render.supports_color()
    if not color:
        print("(no truecolor terminal detected -- using plain blocks)", file=sys.stderr)

    rng = random.Random(args.seed) if args.seed is not None else None
    game = Game(rng)
    renderer = render.Renderer(
        color=color, ghost=not args.no_ghost, grid=args.grid
    )

    if sys.stdin.isatty():
        try:
            play(game, renderer)
        except (_Quit, KeyboardInterrupt):
            pass
        print()
    else:
        play_prompts(game, renderer)
        print(renderer.frame(game))

    print(f"Final score: {game.score}   Lines: {game.lines_cleared}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
