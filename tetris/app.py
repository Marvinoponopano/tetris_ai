"""The frame loop shared by every frontend: human, AI training, AI replay.

Architecture
------------
    tetris/game.py     the engine: board, pieces, scoring, gravity. Pure logic,
                       no I/O, no keyboard, no rendering.
          ^
          |  (Game API: step / apply_gravity / reset / is_game_over / get_state)
          |
    tetris/app.py      `TetrisApp`: timing and mode-independent state. It knows
          ^            about gravity, pause, restart and game over -- and
          |            nothing about keyboards, pixels or AI.
          |
    controller         where the *decisions* come from. Today
    (choose_action)    `HumanController` (in `tetris/pygame_view.py`) turns
                       keystrokes into Actions; an AI policy will implement the
                       same single-method interface later.
    view               where the *pixels* come from: `PygameView` (a window),
                       `tetris/render.py` (the terminal), or nothing at all for
                       headless training.

This module deliberately does not import pygame, and the engine does not import
this module, so the three modes the project wants are just different
(controller, view) pairs::

    human    HumanController + PygameView     play it yourself
    train    AIPolicy        + no view        run as fast as the CPU allows
    viz      AIPolicy        + PygameView     watch the AI play

Nothing in `Game` or `TetrisApp` has to change for the AI to take over: the AI
only has to produce `Action`s from `game.get_state()`, which is exactly what a
human does with arrow keys.
"""

from __future__ import annotations

import time
from typing import Callable, Optional, Protocol  # noqa: UP035

from .actions import Action
from .game import Game

DEFAULT_FPS = 60

# Gravity: milliseconds per cell, shrinking by GRAVITY_DECAY per level.
# (Same curve as the terminal demo in main.py, expressed in milliseconds so it
# can be driven by a frame clock.)
GRAVITY_START_MS = 800.0
GRAVITY_MIN_MS = 60.0
GRAVITY_DECAY = 0.85

# A frame longer than this (a dragged window, a stalled process, a debugger
# breakpoint) is clamped, so the piece never teleports to the floor after a
# hiccup; at 60fps frames are ~16ms, so this only ever matters when something
# went badly wrong.
MAX_FRAME_MS = 500.0


def gravity_interval_ms(level: int) -> float:
    """Milliseconds between gravity ticks at `level` (1 = slowest)."""
    level = max(1, int(level))
    return max(GRAVITY_MIN_MS, GRAVITY_START_MS * (GRAVITY_DECAY ** (level - 1)))


def monotonic_ms() -> float:
    """Wall clock in milliseconds (injectable, so timing is testable)."""
    return time.monotonic() * 1000.0


class Controller(Protocol):
    """Anything that can decide the next move.

    Implemented by `HumanController` today and by the AI policy later; both see
    the same `Game` and answer with the same `Action` enum, which is what keeps
    keyboard input and AI input interchangeable.
    """

    def choose_action(self, game: Game) -> Optional[Action]:
        """The next action to apply, or None to leave the piece alone."""


class TetrisApp:
    """Drives a `Game` at a fixed frame rate, independent of any frontend.

    Parameters
    ----------
    game : Game
        The engine instance being played (already seeded, if you want
        reproducibility).
    controller : Controller, optional
        Where decisions come from. `None` means "gravity only" -- a game that
        falls by itself, which is what a training loop wants while it computes
        the next move.
    fps : int
        Target frame rate. Gravity is derived from the *elapsed* milliseconds
        reported by the frontend, never from the loop count, so a slow frame
        cannot change the game's speed.
    gravity : bool
        While False the piece never falls on its own (the controller is then
        responsible for every move -- the AI case).
    game_factory : callable, optional
        Builds a brand-new `Game`. When given, `restart()` uses it, so a seeded
        run replays the *same* piece sequence -- which is what makes AI
        experiments comparable. Without it, `restart()` simply resets the
        existing game and its randomizer carries on.
    """

    def __init__(
        self,
        game: Game,
        controller: Optional[Controller] = None,
        fps: int = DEFAULT_FPS,
        gravity: bool = True,
        game_factory: Optional[Callable[[], Game]] = None,
    ):
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps!r}")
        self.game = game
        self.controller = controller
        self.fps = fps
        self.gravity_enabled = gravity
        self.game_factory = game_factory
        self.paused = False
        self._accumulator_ms = 0.0

    # ----------------------------------------------------------------- queries

    @property
    def game_over(self) -> bool:
        return self.game.is_game_over()

    @property
    def frame_ms(self) -> float:
        """How long one frame should last (the frontend's clock target)."""
        return 1000.0 / self.fps

    # ---------------------------------------------------------------- commands

    def restart(self) -> None:
        """Start a new game (the R key).

        With a `game_factory` this builds a fresh game, so a seeded run replays
        the same piece sequence; otherwise the current game is reset in place.
        Either way the pause is cleared and gravity's accumulator restarts.
        """
        if self.game_factory is not None:
            self.game = self.game_factory()
        else:
            self.game.reset()
        self._accumulator_ms = 0.0
        self.paused = False

    def toggle_pause(self) -> bool:
        """Flip pause and return the new value (the P key)."""
        self.paused = not self.paused
        return self.paused

    def apply(self, action: Optional[Action]) -> Optional[Action]:
        """Apply one action through the engine, returning what was applied.

        Nothing happens while paused or after game over, which keeps a controller
        from driving a finished game.
        """
        if action is None or self.paused or self.game.is_game_over():
            return None
        self.game.step(action)
        return action

    def update(self, dt_ms: float) -> int:
        """Advance gravity by `dt_ms` and return how many cells were dropped."""
        if not self.gravity_enabled or self.paused or self.game.is_game_over():
            return 0

        self._accumulator_ms += min(max(dt_ms, 0.0), MAX_FRAME_MS)
        ticks = 0
        while not self.game.is_game_over():
            interval = gravity_interval_ms(self.game.level)
            if self._accumulator_ms < interval:
                break
            self._accumulator_ms -= interval
            self.game.apply_gravity()
            ticks += 1
        if self.game.is_game_over():
            self._accumulator_ms = 0.0
        return ticks

    def tick(self, dt_ms: float) -> Optional[Action]:
        """One frame: gravity for `dt_ms`, then one controller decision."""
        self.update(dt_ms)
        if self.controller is None:
            return None
        return self.apply(self.controller.choose_action(self.game))
