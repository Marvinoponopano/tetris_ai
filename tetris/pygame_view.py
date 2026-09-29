"""The pygame frontend: it draws a `Game` and turns keystrokes into `Action`s.

Three deliberately dumb pieces:

* `PygameView` -- pixels only. It *reads* the game (board, active piece, score,
  lines, level, queue) and paints it; it never mutates the game and never looks
  at input.
* `HumanController` -- input only. It turns pygame KEYDOWN/KEYUP events into a
  stream of `Action`s plus a few app commands ("pause", "restart", ...). It
  never draws and never mutates the game.
* `run_human` -- the glue: one `pygame.time.Clock`-driven frame loop that asks
  `TetrisApp` to advance the game, the controller for decisions, and the view to
  paint. Swapping this loop for an AI one changes nothing below it.

Every game call this module makes is one of `app.tick()`, `app.restart()`,
`app.toggle_pause()` or a read-only query -- there is no game rule implemented
here, which is what keeps keyboard input and AI input interchangeable.

Colors come from `tetris/render.py` so the window and the terminal demo agree.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple  # noqa: UP035

import pygame

from . import render
from .actions import Action
from .app import TetrisApp
from .board import HEIGHT, WIDTH
from .pieces import PieceType, rotated_cells

DEFAULT_CELL = 32          # pixels per board cell
MARGIN = 16
BORDER = 3
PANEL_WIDTH = 200
PREVIEW_SCALE = 0.52       # next-piece previews are this fraction of a cell

BACKGROUND: render.Color = (7, 9, 16)
PANEL_BG: render.Color = (11, 14, 24)
# A dimmer rule than the terminal's GRID_DOT: at 1px on a large well even a
# faint line reads strongly, and this one is only meant to help you aim.
GRID_COLOR: render.Color = (20, 24, 39)
FONT_NAMES = "menlo,couriernew,dejavusansmono,monospace,dejavusans"

# Input: which key does which action. The arrow keys are the documented
# controls; the letters mirror the terminal demo (main.py) for people who
# prefer them.
KEY_TO_ACTION: Dict[int, Action] = {
    pygame.K_LEFT: Action.LEFT,
    pygame.K_a: Action.LEFT,
    pygame.K_RIGHT: Action.RIGHT,
    pygame.K_d: Action.RIGHT,
    pygame.K_DOWN: Action.DOWN,
    pygame.K_s: Action.DOWN,
    pygame.K_UP: Action.ROTATE_CW,
    pygame.K_x: Action.ROTATE_CW,
    pygame.K_w: Action.ROTATE_CW,
    pygame.K_z: Action.ROTATE_CCW,
    pygame.K_SPACE: Action.DROP,
}

# Keys that act as an app command rather than a game action.
KEY_TO_COMMAND: Dict[int, str] = {
    pygame.K_r: "restart",
    pygame.K_p: "pause",
    pygame.K_g: "ghost",
    pygame.K_q: "quit",
    pygame.K_ESCAPE: "quit",
}

# Holding these repeats the action (classic DAS/ARR: one immediate move, then a
# pause, then steady repeats).
REPEAT_KEYS = frozenset({pygame.K_LEFT, pygame.K_RIGHT, pygame.K_DOWN, pygame.K_a,
                         pygame.K_d, pygame.K_s})
DAS_MS = 150.0   # delay before the first repeat
ARR_MS = 45.0    # gap between repeats


def landing_cells(game) -> List[Tuple[int, int]]:
    """Where the active piece would come to rest (the ghost preview).

    Read-only: it asks the board for collisions but changes nothing. It mirrors
    the terminal renderer's ghost logic (`tetris/render.py`).
    """
    piece = game.current
    if piece is None:
        return []
    row = piece.row
    while not game.board.collides(piece.absolute_cells(row + 1, piece.col)):
        row += 1
    return piece.absolute_cells(row, piece.col)


class HumanController:
    """Turns pygame events into `Action`s (and app commands).

    `feed()` is called once per frame with that frame's events and the elapsed
    milliseconds; it returns the app commands to run, queues the actions, and
    advances the auto-repeat timers. `choose_action()` then hands the app at most
    one action per frame -- the same single-method interface an AI policy will
    implement.
    """

    def __init__(self, das_ms: float = DAS_MS, arr_ms: float = ARR_MS):
        self.das_ms = das_ms
        self.arr_ms = arr_ms
        self._queue: List[Action] = []
        self._repeat_key: Optional[int] = None
        self._repeat_wait_ms = 0.0

    def feed(self, events: Sequence[pygame.event.Event], dt_ms: float) -> List[str]:
        """Consume one frame of events; return the app commands they asked for."""
        commands: List[str] = []
        for event in events:
            if event.type == pygame.KEYDOWN:
                action = KEY_TO_ACTION.get(event.key)
                if action is not None:
                    self._queue.append(action)
                    if event.key in REPEAT_KEYS:
                        self._repeat_key = event.key
                        self._repeat_wait_ms = self.das_ms
                command = KEY_TO_COMMAND.get(event.key)
                if command is not None:
                    commands.append(command)
            elif event.type == pygame.KEYUP and event.key == self._repeat_key:
                self._repeat_key = None

        self._advance_repeat(dt_ms)
        return commands

    def _advance_repeat(self, dt_ms: float) -> None:
        if self._repeat_key is None:
            return
        self._repeat_wait_ms -= max(dt_ms, 0.0)
        while self._repeat_wait_ms <= 0:
            self._queue.append(KEY_TO_ACTION[self._repeat_key])
            self._repeat_wait_ms += self.arr_ms

    def choose_action(self, game) -> Optional[Action]:
        """The next queued move, or None (held keys fill the queue over time)."""
        return self._queue.pop(0) if self._queue else None

    def clear(self) -> None:
        """Forget queued moves and held keys (used when the game restarts)."""
        self._queue.clear()
        self._repeat_key = None


class PygameView:
    """Draws a `Game` into a pygame window.

    Parameters
    ----------
    cell : int
        Pixels per board cell; the whole window is sized from it.
    ghost : bool
        Draw the outlined landing preview of the active piece.
    grid : str
        "none", "dots" or "lines" -- how the empty well is marked (the same
        names the terminal renderer uses).
    """

    def __init__(
        self,
        cell: int = DEFAULT_CELL,
        ghost: bool = True,
        grid: str = "lines",
        caption: str = "Tetris -- arrows move/rotate, space hard-drops",
        panel_width: int = PANEL_WIDTH,
    ):
        if grid not in render.GRID_STYLES:
            raise ValueError(f"grid must be one of {render.GRID_STYLES!r}")
        self.cell = int(cell)
        self.ghost = ghost
        self.grid = grid
        self.panel_width = int(panel_width)
        self._unit = self.cell / float(DEFAULT_CELL)

        # The overlay wording lives here so a subclass can re-word the hints for
        # its own controls (`AIPygameView` is paused with SPACE, not P).
        self.pause_hint = "press P to resume"
        self.game_over_hint = "press R to restart"

        self.well = pygame.Rect(MARGIN, MARGIN, WIDTH * self.cell, HEIGHT * self.cell)
        size = (self.well.right + MARGIN + self.panel_width + MARGIN, self.well.bottom + MARGIN)
        self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption(caption)

        self.font_title = self._font(21, bold=True)
        self.font_label = self._font(14)
        self.font_value = self._font(19, bold=True)
        self.font_small = self._font(13)
        self.font_banner = self._font(30, bold=True)

    # ------------------------------------------------------------------ fonts

    def _font(self, size: float, bold: bool = False) -> pygame.font.Font:
        return pygame.font.SysFont(FONT_NAMES, max(8, int(round(size * self._unit))), bold=bold)

    def _text(self, text: str, font: pygame.font.Font, color: render.Color) -> pygame.Surface:
        return font.render(text, True, color)

    # -------------------------------------------------------------- geometry

    def cell_rect(self, row: int, col: int) -> pygame.Rect:
        return pygame.Rect(
            self.well.x + col * self.cell,
            self.well.y + row * self.cell,
            self.cell,
            self.cell,
        )

    # --------------------------------------------------------------- drawing

    def draw(self, app: TetrisApp) -> None:
        """Paint one frame (call `flip()` afterwards to show it)."""
        game = app.game
        self.screen.fill(BACKGROUND)
        self._draw_well()
        self._draw_locked(game)
        if not game.is_game_over() and game.current is not None:
            if self.ghost:
                for row, col in landing_cells(game):
                    self._draw_ghost(self.cell_rect(row, col), render.piece_color(game.current.type))
            for row, col in game.current.absolute_cells():
                if 0 <= row < HEIGHT:
                    self._draw_block(self.cell_rect(row, col), render.piece_color(game.current.type))
        self._draw_panel(app)
        if app.paused:
            self._draw_overlay("PAUSED", [self.pause_hint])
        elif game.is_game_over():
            self._draw_overlay(
                "GAME OVER", [f"score {game.score}", self.game_over_hint]
            )

    def flip(self) -> None:
        pygame.display.flip()

    def save(self, path: str) -> str:
        """Write the current frame to a PNG (used by `play.py --screenshot`)."""
        folder = os.path.dirname(os.path.abspath(path))
        if folder:
            os.makedirs(folder, exist_ok=True)
        pygame.image.save(self.screen, path)
        return path

    def _draw_well(self) -> None:
        pygame.draw.rect(self.screen, render.PLAYFIELD, self.well)
        if self.grid == "lines":
            step = self.cell
            for col in range(1, WIDTH):
                x = self.well.x + col * step
                pygame.draw.line(self.screen, GRID_COLOR, (x, self.well.y), (x, self.well.bottom))
            for row in range(1, HEIGHT):
                y = self.well.y + row * step
                pygame.draw.line(self.screen, GRID_COLOR, (self.well.x, y), (self.well.right, y))
        elif self.grid == "dots":
            for row in range(HEIGHT):
                for col in range(WIDTH):
                    rect = self.cell_rect(row, col)
                    pygame.draw.circle(self.screen, GRID_COLOR, (rect.x + 3, rect.y + 3), 1)
        # Drawn *outside* the cells so the top row of blocks is never clipped.
        pygame.draw.rect(
            self.screen, render.FRAME, self.well.inflate(BORDER * 2, BORDER * 2), width=BORDER
        )

    def _draw_locked(self, game) -> None:
        board = game.board.as_list()
        for row in range(min(HEIGHT, len(board))):
            for col in range(min(WIDTH, len(board[row]))):
                if board[row][col]:
                    self._draw_block(self.cell_rect(row, col), render.piece_color(board[row][col]))

    def _draw_block(self, rect: pygame.Rect, color: render.Color) -> None:
        """One beveled tetromino block: light top-left edges, dark bottom-right."""
        edge = max(2, rect.width // 7)
        pygame.draw.rect(self.screen, color, rect)
        light = render.lighten(color, 0.38)
        dark = render.darken(color, 0.45)
        pygame.draw.rect(self.screen, light, (rect.x, rect.y, rect.w, edge))
        pygame.draw.rect(self.screen, light, (rect.x, rect.y, edge, rect.h))
        pygame.draw.rect(self.screen, dark, (rect.x, rect.bottom - edge, rect.w, edge))
        pygame.draw.rect(self.screen, dark, (rect.right - edge, rect.y, edge, rect.h))

    def _draw_ghost(self, rect: pygame.Rect, color: render.Color) -> None:
        """A hollow outline where the piece will land."""
        tint = render.mix(render.PLAYFIELD, color, 0.55)
        pygame.draw.rect(self.screen, tint, rect, width=max(2, rect.width // 9))

    def _draw_panel(self, app: TetrisApp) -> None:
        game = app.game
        x = self.well.right + MARGIN
        y = self.well.y
        width = self.panel_width
        pygame.draw.rect(self.screen, PANEL_BG, (x - MARGIN // 2, y - MARGIN // 2,
                                                width + MARGIN, self.well.height + MARGIN))

        self.screen.blit(self._text("TETRIS", self.font_title, render.LABEL), (x, y))
        y += int(34 * self._unit)
        pygame.draw.line(self.screen, render.GRID_DOT, (x, y), (x + width - 20, y))
        y += int(14 * self._unit)

        for label, value in (
            ("SCORE", game.score),
            ("LINES", game.lines_cleared),
            ("LEVEL", game.level),
        ):
            self.screen.blit(self._text(label, self.font_label, render.DIM), (x, y))
            surface = self._text(str(value), self.font_value, render.VALUE)
            self.screen.blit(surface, (x + width - 20 - surface.get_width(), y - 2))
            y += int(28 * self._unit)

        y += int(14 * self._unit)
        self.screen.blit(self._text("NEXT", self.font_label, render.LABEL), (x, y))
        y += int(22 * self._unit)
        preview_cell = max(6, int(self.cell * PREVIEW_SCALE))
        for piece_type in list(game.queue)[:3]:
            self._draw_preview(x, y, preview_cell, PieceType(piece_type))
            y += 4 * preview_cell + int(12 * self._unit)

        y = self.well.bottom - int(120 * self._unit)
        for line in (
            "\u2190 \u2192   move",
            "\u2193     soft drop",
            "\u2191 / X rotate    Z  ccw",
            "space hard drop",
            "P pause    R restart",
            "G ghost    Esc quit",
        ):
            self.screen.blit(self._text(line, self.font_small, render.DIM), (x, y))
            y += int(15 * self._unit)

    def _draw_preview(self, x: int, y: int, cell: int, piece_type: PieceType) -> None:
        """A small centred drawing of an upcoming piece inside a 4x4 cell box."""
        box = pygame.Rect(x, y, 4 * cell, 4 * cell)
        pygame.draw.rect(self.screen, render.PLAYFIELD, box)
        pygame.draw.rect(self.screen, render.GRID_DOT, box, width=1)

        cells = rotated_cells(piece_type, 0)
        min_row = min(r for r, _ in cells)
        min_col = min(c for _, c in cells)
        height = max(r for r, _ in cells) - min_row + 1
        width = max(c for _, c in cells) - min_col + 1
        off_row = (4 - height) // 2
        off_col = (4 - width) // 2
        for row, col in cells:
            rect = pygame.Rect(
                box.x + (off_col + col - min_col) * cell,
                box.y + (off_row + row - min_row) * cell,
                cell,
                cell,
            )
            self._draw_block(rect, render.PIECE_COLORS[piece_type])

    def _draw_overlay(self, title: str, lines: Sequence[str]) -> None:
        shade = pygame.Surface(self.well.size, pygame.SRCALPHA)
        shade.fill((5, 7, 12, 205))
        self.screen.blit(shade, self.well.topleft)

        centre_x = self.well.centerx
        y = self.well.centery - int(40 * self._unit)
        surface = self._text(title, self.font_banner, render.BANNER)
        self.screen.blit(surface, surface.get_rect(center=(centre_x, y)))
        y += int(42 * self._unit)
        for line in lines:
            surface = self._text(line, self.font_label, render.VALUE)
            self.screen.blit(surface, surface.get_rect(center=(centre_x, y)))
            y += int(24 * self._unit)


def run_human(
    app: TetrisApp,
    view: PygameView,
    controller: HumanController,
    max_frames: Optional[int] = None,
    screenshot: Optional[str] = None,
    record: Optional[str] = None,
    record_every: int = 1,
    stop_after_game_over: Optional[int] = None,
) -> int:
    """Play in a window until the player quits. Returns the number of frames.

    The clock is `pygame.time.Clock`, so the loop runs at `app.fps` and sleeps
    between frames instead of spinning; the game's own gravity comes from the
    elapsed milliseconds `clock.tick()` reports.

    `controller` is anything with `feed(events, dt_ms)` and `choose_action(game)`
    -- the human keyboard or, in `--mode viz`, an AI policy. `record` is a
    directory: one PNG per `record_every` frames, which is what a demo video is
    made of (the app has no video backend, and this keeps the rendering in the
    same place as the screenshot). `stop_after_game_over` ends the loop that many
    frames after the game ends -- a recording of an AI game should be the game,
    not the game plus a frozen board.
    """
    clock = pygame.time.Clock()
    frames = 0
    final_frames = 0
    quitting = False
    try:
        while not quitting:
            dt_ms = clock.tick(app.fps)
            events = pygame.event.get()
            if any(event.type == pygame.QUIT for event in events):
                break

            for command in controller.feed(events, dt_ms):
                if command == "quit":
                    quitting = True
                elif command == "pause":
                    app.toggle_pause()
                elif command == "restart":
                    app.restart()
                    controller.clear()
                elif command == "ghost":
                    view.ghost = not view.ghost
            if quitting:
                break

            app.tick(dt_ms)
            view.draw(app)
            view.flip()

            if record is not None and frames % record_every == 0:
                view.save(os.path.join(record, f"frame_{frames // record_every:05d}.png"))

            frames += 1
            if app.game.is_game_over():
                final_frames += 1
                if (stop_after_game_over is not None
                        and final_frames >= stop_after_game_over):
                    break
            if max_frames is not None and frames >= max_frames:
                break
    finally:
        if screenshot:
            view.save(screenshot)
    return frames
