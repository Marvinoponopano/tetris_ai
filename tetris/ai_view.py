"""The AI viewer: the same game a human plays, drawn with the AI's own numbers.

    python play_ai.py                      # the best trained checkpoint
    python play_ai.py --model checkpoints/best.pt
    python play_ai.py --agent heuristic    # the hand-written baseline

Two things this module is careful about.

**It does not change how the AI decides.** The agent is the same object, called
through the same `choose_action(state, valid_actions)` the training loop and the
evaluation harness call, and the state is read from `TetrisEnv.get_state()`. The
only difference from a headless run is that someone is watching: the loop ticks a
clock and paints a frame. That is the whole point of the split --

    training       headless env   ->  as fast as the CPU allows
    visualization   same env      ->  same decisions, human-readable speed

-- and `tests/test_play_ai.py` pins it down by running the same checkpoint and
seed through both paths and asserting the placements came out identical.

**It does not train, and it never writes.** Checkpoints are opened read-only, so
closing the window (or killing it) cannot damage a model.

Speed is a table (`SPEED_LEVELS`), not a hidden constant: each level is a pair of
"milliseconds between moves" and "engine ticks per frame". The second number is
what makes `Very fast` genuinely fast -- above one, the AI gets several moves in
per painted frame, which is faster than any frame rate could make it otherwise.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import pygame

from tetris import render
from tetris.app import TetrisApp
from tetris.pieces import PieceType
from tetris.policy_controller import (   # the pacing table is pygame-free
    DEFAULT_SPEED,
    SPEED_LEVELS,
    speed_count,
    speed_detail,
)
from tetris.pygame_view import MARGIN, PANEL_BG, PygameView

AI_PANEL_WIDTH = 300         # wider than the human panel: it has more to say
AI_CELL = 34                 # bigger than the human default, for a watchable board
PREVIEW_SCALE = 0.42         # next-piece boxes, smaller: the AI section needs room

# Every pixel step in the panel is written in "units" of DEFAULT_CELL so the whole
# layout scales with --cell, exactly like the human panel does.
TITLE_STEP = 34
RULE_STEP = 14
ROW_STEP = 26
SMALL_STEP = 22
LEGEND_STEP = 16
LEGEND_ROWS = 4
MOVE_ROWS = 6
MOVE_STEP = 19
ROW_INSET = 12               # keep values off the panel's right edge


def _number(value: object, digits: int = 2) -> str:
    """Format an optional float for the panel (`-` when there is nothing)."""
    if value is None:
        return "-"
    try:
        return f"{float(value):+.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


@dataclass(frozen=True)
class AIStatus:
    """Everything the AI panel shows, collected in one place.

    The loop builds it (it knows the clock and the viewer state), the view draws
    it (it knows about fonts), and neither reaches into the other. All of it is
    plain data, so a test can assert on the panel's contents without a display.
    """

    status: str                       # PLAYING / PAUSED / GAME OVER
    speed: str                        # the level's label, e.g. "Normal"
    speed_detail: str                 # e.g. "120 ms/move"
    agent: str                        # "dqn", "heuristic", ...
    model: str                        # what was loaded, or a built-in's name
    action: Optional[str] = None      # the engine action in flight
    current_piece: str = "-"          # the falling piece's letter
    report: Optional[Dict[str, object]] = None   # the last locked piece
    decisions: int = 0
    pieces: int = 0
    actions: int = 0
    abandoned: int = 0
    epsilon: Optional[float] = None   # the agent's exploration rate, if it has one

    def lines_of(self, debug: bool = False) -> List[str]:
        """The panel as text -- what a test reads instead of the pixels."""
        move = self.report or {}
        rows = [
            f"status: {self.status}",
            f"speed: {self.speed} ({self.speed_detail})",
            f"agent: {self.agent}",
            f"model: {self.model}",
            f"action: {self.action or '-'}",
            f"piece: {self.current_piece}",
            f"pieces: {self.pieces}  decisions: {self.decisions}",
            f"actions: {self.actions}  abandoned: {self.abandoned}",
        ]
        if self.epsilon is not None:
            rows.append(f"epsilon: {self.epsilon:.3f}")
        if move:
            rows += [
                f"chosen rotation: {move.get('rotation')}",
                f"chosen column: {move.get('column')}",
                f"{move.get('value_label', 'value')}: {_number(move.get('value'))}",
                f"reward: {_number(move.get('reward'))}",
                f"lines cleared: {move.get('lines_cleared')}",
            ]
        if debug:
            rows.append("debug overlay: on")
        return rows

    def debug_lines(self) -> List[str]:
        """The overlay's rows, in the wording the debug view asks for."""
        move = self.report or {}
        return [
            f"chosen rotation : {move.get('rotation', '-')}",
            f"chosen column   : {move.get('column', '-')}",
            f"{move.get('value_label', 'value')} : {_number(move.get('value'))}",
            f"reward          : {_number(move.get('reward'))}",
            f"lines cleared   : {move.get('lines_cleared', '-')}",
        ]


class AIPygameView(PygameView):
    """`PygameView` with an AI panel: what the agent chose, and what it got.

    The board side is inherited unchanged -- well, grid, locked blocks, the
    falling piece and its ghost, the pause/game-over overlays -- because the
    point of watching an AI play is to watch *this* game. Only the panel and the
    optional debug box are new.
    """

    def __init__(
        self,
        cell: int = AI_CELL,
        ghost: bool = True,
        grid: str = "lines",
        panel_width: int = AI_PANEL_WIDTH,
        debug: bool = False,
    ) -> None:
        super().__init__(
            cell=cell,
            ghost=ghost,
            grid=grid,
            caption="Tetris AI -- SPACE pause, R restart, +/- speed, ESC quit",
            panel_width=panel_width,
        )
        self.grid_style = grid
        self.debug = bool(debug)
        self.status: Optional[AIStatus] = None
        self.pause_hint = "SPACE resumes"
        self.game_over_hint = "R restarts, ESC quits"

    # ------------------------------------------------------------------ drawing

    def draw(self, app: TetrisApp, status: Optional[AIStatus] = None) -> None:
        """Paint one frame; with no `status` the human panel is drawn instead."""
        if status is not None:
            self.status = status
        super().draw(app)                 # board, pieces, panel, overlays
        if self.debug and self.status is not None:
            self._draw_debug(self.status)

    def _draw_panel(self, app: TetrisApp) -> None:
        """The AI panel: what the AI is doing, the game, and the last move."""
        if self.status is None:
            super()._draw_panel(app)
            return

        status = self.status
        game = app.game
        x = self.well.right + MARGIN
        width = self.panel_width
        unit = self._unit
        y = self.well.y

        pygame.draw.rect(self.screen, PANEL_BG, (x - MARGIN // 2, y - MARGIN // 2,
                                                width + MARGIN, self.well.height + MARGIN))

        # --- header -------------------------------------------------------
        self.screen.blit(self._text("TETRIS \u00b7 AI", self.font_title, render.LABEL), (x, y))
        y += int(TITLE_STEP * unit)
        self._rule(x, y, width)
        y += int(RULE_STEP * unit)

        # --- what the AI is doing ----------------------------------------
        y = self._row(y, x, width, "STATUS", status.status, self.font_value)
        y = self._row(y, x, width, "SPEED", status.speed, self.font_value)
        y = self._row(y, x, width, "ACTION", status.action or "-", self.font_value)
        y = self._row(y, x, width, "AGENT", status.agent, self.font_value)
        y = self._row(y, x, width, "MODEL", status.model, self.font_small, small=True,
                      step=SMALL_STEP)
        y += int(4 * unit)
        self._rule(x, y, width)
        y += int(RULE_STEP * unit)

        # --- the game -----------------------------------------------------
        y = self._row(y, x, width, "SCORE", str(game.score), self.font_value)
        y = self._row(y, x, width, "LINES", str(game.lines_cleared), self.font_value)
        y = self._row(y, x, width, "LEVEL", str(game.level), self.font_value)
        y = self._row(y, x, width, "PIECES", str(status.pieces), self.font_value)
        if status.epsilon is not None:
            y = self._row(y, x, width, "EPSILON", f"{status.epsilon:.3f}",
                          self.font_small, small=True, step=SMALL_STEP)
        y += int(6 * unit)
        self._rule(x, y, width)
        y += int(RULE_STEP * unit)

        # --- next pieces (side by side, so they cost one row not three) ----
        self.screen.blit(self._text("NEXT", self.font_label, render.LABEL), (x, y))
        y += int(20 * unit)
        preview_cell = max(6, int(self.cell * PREVIEW_SCALE))
        for index, piece_type in enumerate(list(game.queue)[:3]):
            self._draw_preview(x + index * (4 * preview_cell + int(10 * unit)), y,
                               preview_cell, PieceType(piece_type))

        self._draw_move_block(status)
        self._draw_legend()

    def _draw_move_block(self, status: AIStatus) -> None:
        """The last locked piece, in the terms the reward is written in."""
        x = self.well.right + MARGIN
        width = self.panel_width
        unit = self._unit
        y = self._move_block_top()

        self.screen.blit(self._text("LAST MOVE", self.font_label, render.LABEL), (x, y))
        y += int(20 * unit)
        move = status.report
        if not move:
            self.screen.blit(
                self._text("no piece locked yet", self.font_small, render.DIM), (x, y)
            )
            return

        value = _number(move.get("value"))
        if move.get("legal"):
            value = f"{value} ({move['legal']} legal)"
        # The piece row carries a colour swatch: the letter "O" on its own reads
        # as a zero, and the colour is what the eye uses to tell the pieces apart.
        name = str(move.get("piece"))
        swatch = max(8, int(14 * unit))
        text = self._text(name, self.font_small, render.VALUE)
        self.screen.blit(self._text("piece", self.font_small, render.DIM),
                         (x, y + int(2 * unit)))
        self.screen.blit(text, (x + width - int(ROW_INSET * unit) - swatch
                                - int(6 * unit) - text.get_width(), y))
        try:
            self._draw_block(pygame.Rect(x + width - int(ROW_INSET * unit) - swatch, y,
                                        swatch, swatch),
                             render.piece_color(PieceType[name]))
        except KeyError:                      # a name this build does not know
            self.screen.blit(text, (x + width - int(ROW_INSET * unit) - text.get_width(), y))
        y += int(MOVE_STEP * unit)

        rows = [
            ("rotation", str(move.get("rotation"))),
            ("column", str(move.get("column"))),
            (str(move.get("value_label") or "value"), value),
            ("reward", _number(move.get("reward"))),
            ("board", f"{move.get('lines_cleared')}L  {move.get('holes')}H  "
                      f"{move.get('height')}Ht"),
        ]
        for label, text in rows:
            y = self._row(y, x, width, label, text, self.font_small, small=True,
                          step=MOVE_STEP)

    def _draw_legend(self) -> None:
        """The keys a spectator of an AI game still wants."""
        x = self.well.right + MARGIN
        unit = self._unit
        y = self.well.bottom - int((LEGEND_ROWS * LEGEND_STEP + 4) * unit)
        for line in (
            "SPACE  pause / resume",
            "+ / -  slower / faster",
            "R  restart      G  ghost",
            "D  debug info   ESC  quit",
        ):
            self.screen.blit(self._text(line, self.font_small, render.DIM), (x, y))
            y += int(LEGEND_STEP * unit)

    def _draw_debug(self, status: AIStatus) -> None:
        """The optional overlay: the choice and its value, over the board.

        Drawn on the playfield rather than the panel so it is readable at a
        glance while the piece is moving, which is when the numbers matter.
        """
        rows = status.debug_lines()
        font = self.font_small
        pad = max(6, int(8 * self._unit))
        width = max(font.size(row)[0] for row in rows) + 2 * pad
        height = len(rows) * font.get_linesize() + 2 * pad
        box = pygame.Rect(self.well.x + pad, self.well.y + pad, width, height)
        shade = pygame.Surface(box.size, pygame.SRCALPHA)
        shade.fill((5, 7, 12, 205))
        self.screen.blit(shade, box.topleft)
        pygame.draw.rect(self.screen, render.GRID_DOT, box, width=1)
        y = box.y + pad
        for row in rows:
            self.screen.blit(self._text(row, font, render.VALUE), (box.x + pad, y))
            y += font.get_linesize()

    # ------------------------------------------------------------------ pieces

    def _move_block_top(self) -> int:
        """Where the LAST MOVE block starts (anchored above the legend)."""
        reserved = LEGEND_ROWS * LEGEND_STEP + 8 + 20 + MOVE_ROWS * MOVE_STEP + 10
        return self.well.bottom - int(reserved * self._unit)

    def _rule(self, x: int, y: int, width: int) -> None:
        pygame.draw.line(self.screen, render.GRID_DOT, (x, y), (x + width, y))

    def _row(
        self,
        y: int,
        x: int,
        width: int,
        label: str,
        value: str,
        font,
        small: bool = False,
        step: int = ROW_STEP,
    ) -> int:
        """A label on the left, its value right-aligned; returns the next y."""
        unit = self._unit
        label_font = self.font_small if small else self.font_label
        label_surface = self._text(label, label_font, render.DIM)
        self.screen.blit(label_surface, (x, y + int(2 * unit)))
        room = (width - int(ROW_INSET * unit) - label_surface.get_width()
                - int(8 * unit))
        surface = self._text(self._fit(str(value), font, room), font, render.VALUE)
        self.screen.blit(
            surface, (x + width - int(ROW_INSET * unit) - surface.get_width(), y)
        )
        return y + int(step * unit)

    def _fit(self, text: str, font, limit: int) -> str:
        """Shorten `text` to fit `limit` pixels, marking the cut with an ellipsis.

        A panel row must never overlap its own label, whatever an agent calls
        itself or wherever a checkpoint lives.
        """
        if limit <= 0 or font.size(text)[0] <= limit:
            return text if limit > 0 else ""
        while text and font.size(text + "\u2026")[0] > limit:
            text = text[:-1]
        return text + "\u2026"


def _epsilon_of(controller) -> Optional[float]:
    """The agent's exploration rate, if it publishes one (a DQN does)."""
    value = getattr(getattr(controller, "agent", None), "epsilon", None)
    return None if value is None else float(value)


def _label_of(controller) -> str:
    return str(getattr(getattr(controller, "agent", None), "name", "agent"))


def run_ai(
    app: TetrisApp,
    view: AIPygameView,
    controller,
    speed: int = DEFAULT_SPEED,
    model: str = "",
    max_frames: Optional[int] = None,
    screenshot: Optional[str] = None,
    record: Optional[str] = None,
    record_every: int = 1,
    stop_after_game_over: Optional[int] = None,
    logger=None,
) -> int:
    """Let the AI play in a window until the viewer is closed. Returns frames.

    The keys are the viewer's, not the player's: the AI moves the pieces, so the
    keyboard is left for what a spectator needs (`SPACE` pause, `R` restart,
    `+`/`-` speed, `G` ghost, `D` debug, `ESC` quit).

    One frame is one clock tick and one paint, but `Very fast` runs several engine
    ticks per frame, so the AI can move faster than the screen refreshes. Gravity
    is left to the app (`--gravity`), and while paused nothing ticks at all -- the
    AI must not plan while the board is frozen.
    """
    clock = pygame.time.Clock()
    speed = max(0, min(int(speed), speed_count() - 1))
    controller.think_ms = SPEED_LEVELS[speed][1]

    frames = 0
    final_frames = 0
    quitting = False
    was_over = False
    last_action: Optional[str] = None

    def announce() -> None:
        if logger is not None:
            logger(f"speed {SPEED_LEVELS[speed][0]}: {speed_detail(speed)}")

    announce()
    try:
        while not quitting:
            dt_ms = clock.tick(app.fps)
            events = pygame.event.get()
            if any(event.type == pygame.QUIT for event in events):
                break

            for event in events:
                if event.type != pygame.KEYDOWN:
                    continue
                key = event.key
                if key in (pygame.K_ESCAPE, pygame.K_q):
                    quitting = True
                elif key == pygame.K_SPACE:
                    app.toggle_pause()
                elif key == pygame.K_r:
                    app.restart()
                    controller.clear()
                    was_over = False
                    final_frames = 0
                    last_action = None
                elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    if speed < speed_count() - 1:
                        speed += 1
                        controller.think_ms = SPEED_LEVELS[speed][1]
                        announce()
                elif key in (pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_KP_MINUS):
                    if speed > 0:
                        speed -= 1
                        controller.think_ms = SPEED_LEVELS[speed][1]
                        announce()
                elif key == pygame.K_g:
                    view.ghost = not view.ghost
                elif key == pygame.K_d:
                    view.debug = not view.debug
            if quitting:
                break

            if not app.paused:
                # The pacing clock lives in the controller (`think_ms`); feeding it
                # the frame's elapsed time is what turns "120 ms/move" into a move
                # every 120 ms instead of one per frame.
                controller.feed((), dt_ms)
                for index in range(SPEED_LEVELS[speed][2]):
                    action = app.tick(dt_ms if index == 0 else 0.0)
                    if action is not None:
                        last_action = str(action.value)
                    if app.game.is_game_over():
                        break

            game = app.game
            status = AIStatus(
                status=("PAUSED" if app.paused
                        else "GAME OVER" if game.is_game_over() else "PLAYING"),
                speed=SPEED_LEVELS[speed][0],
                speed_detail=speed_detail(speed),
                agent=_label_of(controller),
                model=model,
                action=last_action,
                current_piece=(game.current.type.name if game.current is not None else "-"),
                report=controller.snapshot()["report"],
                decisions=controller.decisions,
                pieces=controller.pieces,
                actions=controller.actions,
                abandoned=controller.abandoned,
                epsilon=_epsilon_of(controller),
            )

            view.draw(app, status)
            view.flip()
            if record is not None and frames % record_every == 0:
                view.save(os.path.join(record, f"frame_{frames // record_every:05d}.png"))

            frames += 1
            if game.is_game_over():
                if not was_over:
                    was_over = True
                    if logger is not None:
                        logger(
                            f"game over: score {game.score}  lines {game.lines_cleared}  "
                            f"pieces {controller.pieces}  "
                            f"abandoned plans {controller.abandoned}"
                        )
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


__all__ = [
    "AI_CELL",
    "AI_PANEL_WIDTH",
    "AIStatus",
    "AIPygameView",
    "DEFAULT_SPEED",
    "SPEED_LEVELS",
    "run_ai",
    "speed_count",
    "speed_detail",
]
