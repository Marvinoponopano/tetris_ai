"""A colored, game-like terminal view for the engine.

The engine (`board.py`, `pieces.py`, `game.py`) is pure logic and deliberately
has no I/O. This module is the *view*: it turns a `Game` into terminal output
that looks like an actual Tetris playfield -- square colored blocks with a
glossy bevel, a framed well, a ghost piece showing where the active piece will
land, and a side panel with the score, lines, level and the next pieces.

How the blocks are drawn
------------------------
A terminal character cell is about twice as tall as it is wide, and the
half-block glyphs (U+2580 UPPER HALF BLOCK / U+2584 LOWER HALF BLOCK) let ONE
character show TWO independent square pixels (foreground = top, background =
bottom). So two characters side by side are a 2x2 pixel square -- exactly one
board cell::

    board cell (row, col)  ->  canvas pixels (col*2 .. col*2+1, row*2 .. row*2+1)
                           ->  terminal chars (col*2 .. col*2+1, line row)

The playfield is therefore 10 cells * 2 chars = 20 columns wide and 20 lines
tall, which is the same 1:2 aspect as the real 10x20 board, with square cells.

Each block gets a 24-bit color from `PIECE_COLORS` and a bevel: the top-left
pixel is lightened, the bottom-right darkened, so neighboring blocks stay
visually separate. Empty cells are the well background (optionally with a faint
grid). The landing preview ("ghost") is a dim dotted silhouette of the piece in
its own color, so it reads as a hint rather than a locked block.

Everything degrades gracefully: `color=False` switches to plain ASCII (`#`),
for terminals that cannot do truecolor or for logs and tests.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

from .board import EMPTY, HEIGHT, WIDTH
from .pieces import PieceType, rotated_cells

Color = Tuple[int, int, int]
WHITE: Color = (255, 255, 255)
BLACK: Color = (0, 0, 0)

# --------------------------------------------------------------------- palette

# The classic Guideline colors, one per tetromino.
PIECE_COLORS: Dict[PieceType, Color] = {
    PieceType.I: (0, 240, 240),    # cyan
    PieceType.O: (240, 240, 0),    # yellow
    PieceType.T: (160, 0, 240),    # purple
    PieceType.S: (0, 240, 0),      # green
    PieceType.Z: (240, 0, 0),      # red
    PieceType.J: (0, 60, 240),     # blue
    PieceType.L: (240, 160, 0),    # orange
}

PLAYFIELD: Color = (13, 16, 28)     # empty cell background
GRID_DOT: Color = (26, 32, 50)      # faint mark on each empty cell
FRAME: Color = (92, 110, 168)       # the well's border
LABEL: Color = (150, 170, 220)      # panel headings
VALUE: Color = (236, 240, 252)      # panel numbers
DIM: Color = (108, 122, 158)        # secondary text
BANNER: Color = (255, 96, 96)       # game-over / paused banner

# Bevel: how far each pixel is blended toward white (light) or black (shadow).
BEVEL_LIGHT = 0.42
BEVEL_MID_LIGHT = 0.12
BEVEL_DARK = 0.30
BEVEL_MID_DARK = 0.50

# The landing preview ("ghost") is a *dotted silhouette* rather than a solid
# square: only these pixels of each ghost cell are painted, so it reads as a
# hint instead of an actual block. The tint is how much piece color is blended
# into the well background first.
GHOST_TINT = 0.60
GHOST_PATTERN = ((0, 0),)

# Grid styles for the empty part of the well: "none" (flat, cleanest),
# "dots" (a faint mark per cell) or "lines" (faint rules along cell edges).
GRID_STYLES = ("none", "dots", "lines")

# Preview panel: how many board cells of the next piece we show. Sizing all
# three previews to the same box keeps the panel rectangular.
PREVIEW_CELLS_W = 4
PREVIEW_CELLS_H = 4
PREVIEW_COUNT = 3

# ------------------------------------------------------------------ geometry

CHAR_PIXELS = 2      # vertical pixels per character (half blocks)
PX_W = 2             # canvas pixels per board cell, horizontally (= 2 chars)
PX_H = 2             # canvas pixels per board cell, vertically  (= 1 line)
CELL_CHARS = PX_W    # terminal columns per board cell
CELL_LINES = PX_H // CHAR_PIXELS  # terminal lines per board cell

FULL_BLOCK = "\u2588"
UPPER_HALF = "\u2580"
LOWER_HALF = "\u2584"

RESET = "\x1b[0m"

# ------------------------------------------------------------------- color math


def mix(a: Color, b: Color, t: float) -> Color:
    """Blend `t` (0..1) of color `b` into color `a`."""
    t = max(0.0, min(1.0, t))
    return (
        int(round(a[0] + (b[0] - a[0]) * t)),
        int(round(a[1] + (b[1] - a[1]) * t)),
        int(round(a[2] + (b[2] - a[2]) * t)),
    )


def lighten(color: Color, amount: float = BEVEL_LIGHT) -> Color:
    """The highlight pixel of a block."""
    return mix(color, WHITE, amount)


def darken(color: Color, amount: float = BEVEL_DARK) -> Color:
    """The shadow pixel of a block."""
    return mix(color, BLACK, amount)


def piece_color(piece_type) -> Color:
    """The color of a piece, accepting a `PieceType` or a raw 1..7 id."""
    return PIECE_COLORS[PieceType(int(piece_type))]


def supports_color(stream=None) -> bool:
    """True when `stream` (default stdout) looks like a truecolor terminal."""
    stream = stream if stream is not None else sys.stdout
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    try:
        if not stream.isatty():
            return False
    except (AttributeError, ValueError):
        return False
    term = os.environ.get("TERM", "")
    if term in ("", "dumb"):
        return False
    return True


def strip_ansi(text: str) -> str:
    """Remove every SGR escape sequence (handy for width math and tests)."""
    out: List[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\x1b" and text[i + 1: i + 2] == "[":
            j = i + 2
            while j < len(text) and text[j] not in "mK":
                j += 1
            i = j + 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def visible_len(text: str) -> int:
    """Number of printed columns in a string that may contain SGR codes."""
    return len(strip_ansi(text))


def _sgr_fg(color: Color) -> str:
    return f"\x1b[38;2;{color[0]};{color[1]};{color[2]}m"


def _sgr_bg(color: Color) -> str:
    return f"\x1b[48;2;{color[0]};{color[1]};{color[2]}m"


def _paint(text: str, color: Color) -> str:
    return f"{_sgr_fg(color)}{text}{RESET}"


# -------------------------------------------------------------------- canvas


class Canvas:
    """A small RGB pixel buffer that renders itself with half-block glyphs.

    One character shows two vertically stacked square pixels, so a canvas of
    `width` x `height` pixels prints as `width` columns x `height/2` lines.
    """

    def __init__(self, width: int, height: int, background: Color = PLAYFIELD):
        self.width = width
        self.height = height
        self.background = background
        self._pixels: List[List[Color]] = [
            [background] * width for _ in range(height)
        ]

    def set(self, x: int, y: int, color: Color) -> None:
        """Color the pixel at (x, y); out-of-bounds writes are ignored."""
        if 0 <= x < self.width and 0 <= y < self.height:
            self._pixels[y][x] = color

    def get(self, x: int, y: int) -> Color:
        """The color at (x, y), or the background outside the canvas."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self._pixels[y][x]
        return self.background

    def fill(self, x: int, y: int, w: int, h: int, color: Color) -> None:
        """Color a rectangle of pixels."""
        for yy in range(y, y + h):
            for xx in range(x, x + w):
                self.set(xx, yy, color)

    # ------------------------------------------------------------------ output

    def to_lines(self, color: bool = True) -> List[str]:
        """Render the canvas: one list entry per terminal line."""
        lines: List[str] = []
        for y in range(0, self.height, CHAR_PIXELS):
            lines.append(self._line(y, color))
        return lines

    def _line(self, y: int, color: bool) -> str:
        if not color:
            # ASCII fallback: '#' wherever either pixel differs from the
            # background, ' ' for empty pixels (what "#" rendering did before).
            return "".join(
                "#" if (self.get(x, y) != self.background
                         or self.get(x, y + 1) != self.background) else " "
                for x in range(self.width)
            )

        parts: List[str] = []
        painted: Optional[Tuple[Color, Color]] = None
        for x in range(self.width):
            top = self.get(x, y)
            bottom = self.get(x, y + 1)
            glyph, fg, bg = _pair_glyph(top, bottom, self.background)
            if painted != (fg, bg):
                parts.append(RESET + _sgr_fg(fg) + _sgr_bg(bg))
                painted = (fg, bg)
            parts.append(glyph)
        parts.append(RESET)
        return "".join(parts)


def _pair_glyph(
    top: Color, bottom: Color, background: Color
) -> Tuple[str, Color, Color]:
    """Pick the glyph + colors for one character from the two pixels it covers.

    Using a full block when both pixels match (instead of a half block with a
    matching background) keeps terminals from showing a seam between cells.
    """
    if top == bottom:
        return FULL_BLOCK, top, top
    if top == background:
        return LOWER_HALF, bottom, bottom
    if bottom == background:
        return UPPER_HALF, top, top
    return UPPER_HALF, top, bottom


# ------------------------------------------------------------------ renderer


class Renderer:
    """Draws a `Game` as a framed playfield plus an info panel.

    Parameters
    ----------
    color : bool
        Emit 24-bit ANSI colors. When False the output is plain ASCII.
    ghost : bool
        Draw a faint preview of where the active piece will land.
    panel : bool
        Draw the side panel (score / lines / level / next pieces).
    indent : str
        Leading text on every line, useful for terminal padding.
    """

    def __init__(
        self,
        color: bool = True,
        ghost: bool = True,
        panel: bool = True,
        indent: str = "  ",
        grid: str = "none",
    ):
        self.color = color
        self.ghost = ghost
        self.panel = panel
        self.indent = indent
        if grid not in GRID_STYLES:
            raise ValueError(f"grid must be one of {GRID_STYLES!r}, got {grid!r}")
        self.grid = grid

    # ------------------------------------------------------------- playfield

    @property
    def playfield_width(self) -> int:
        """Printed columns of the playfield canvas (border excluded)."""
        return WIDTH * CELL_CHARS

    def board_canvas(self, game) -> Canvas:
        """A pixel canvas with the locked cells, ghost and active piece drawn."""
        canvas = Canvas(self.playfield_width, HEIGHT * PX_H)
        self._draw_grid(canvas, WIDTH, HEIGHT)

        grid = game.board.as_list()
        for row in range(min(HEIGHT, len(grid))):
            for col in range(min(WIDTH, len(grid[row]))):
                value = grid[row][col]
                if value != EMPTY:
                    self._draw_cell(canvas, row, col, piece_color(value))

        piece = getattr(game, "current", None)
        if piece is not None and not game.is_game_over():
            if self.ghost:
                for row, col in self._landing_cells(game):
                    if self._cell_is_empty(grid, row, col):
                        self._draw_ghost(canvas, row, col, piece_color(piece.type))
            for row, col in piece.absolute_cells():
                self._draw_cell(canvas, row, col, piece_color(piece.type))
        return canvas

    def board_lines(self, game) -> List[str]:
        """The playfield, framed, as a list of terminal lines."""
        width = self.playfield_width
        if self.color:
            horizontal = _paint("\u2550" * width, FRAME)
            top = _paint("\u2554", FRAME) + horizontal + _paint("\u2557", FRAME)
            bottom = _paint("\u255a", FRAME) + horizontal + _paint("\u255d", FRAME)
            side = _paint("\u2551", FRAME)
        else:
            horizontal = "-" * width
            top = "+" + horizontal + "+"
            bottom = "+" + horizontal + "+"
            side = "|"

        lines = [self.indent + top]
        for line in self.board_canvas(game).to_lines(self.color):
            lines.append(self.indent + side + line + side)
        lines.append(self.indent + bottom)
        return lines

    def _landing_cells(self, game) -> List[Tuple[int, int]]:
        """Where the active piece would land if it dropped straight down."""
        piece = game.current
        row = piece.row
        while not game.board.collides(piece.absolute_cells(row + 1, piece.col)):
            row += 1
        return piece.absolute_cells(row, piece.col)

    @staticmethod
    def _cell_is_empty(grid, row: int, col: int) -> bool:
        if not (0 <= row < len(grid) and 0 <= col < len(grid[row])):
            return False
        return grid[row][col] == EMPTY

    def _draw_grid(self, canvas: Canvas, cols: int, rows: int) -> None:
        """Faintly mark the empty cells so the well reads as a grid."""
        if self.grid == "none":
            return
        for row in range(rows):
            for col in range(cols):
                x, y = col * PX_W, row * PX_H
                if self.grid == "dots":
                    canvas.set(x, y, GRID_DOT)
                else:  # "lines": a continuous rule along each cell's left edge
                    canvas.set(x, y, GRID_DOT)
                    canvas.set(x, y + 1, GRID_DOT)

    def _draw_cell(self, canvas: Canvas, row: int, col: int, color: Color) -> None:
        """Draw one solid block: bright on the top-left, dark on the bottom-right."""
        if not 0 <= row < HEIGHT:
            return  # still hidden above the well while spawning
        x, y = col * PX_W, row * PX_H
        canvas.set(x, y, lighten(color, BEVEL_LIGHT))
        canvas.set(x + 1, y, lighten(color, BEVEL_MID_LIGHT))
        canvas.set(x, y + 1, darken(color, BEVEL_DARK))
        canvas.set(x + 1, y + 1, darken(color, BEVEL_MID_DARK))

    def _draw_ghost(self, canvas: Canvas, row: int, col: int, color: Color) -> None:
        """Draw the stippled landing shadow of the active piece."""
        if not 0 <= row < HEIGHT:
            return
        tint = mix(canvas.background, color, GHOST_TINT)
        x, y = col * PX_W, row * PX_H
        for dx, dy in GHOST_PATTERN:
            canvas.set(x + dx, y + dy, tint)

    # ------------------------------------------------------------------ panel

    def panel_lines(self, game) -> List[str]:
        """The info panel, padded so every line is the same printed width."""
        lines: List[str] = []
        lines.append(self._text("TETRIS", LABEL))
        lines.append(self._text("\u2500" * 14, DIM))
        lines.append(self._stat("SCORE", str(game.score)))
        lines.append(self._stat("LINES", str(game.lines_cleared)))
        lines.append(self._stat("LEVEL", str(game.level)))
        lines.append("")
        lines.append(self._text("NEXT", LABEL))
        for piece_type in list(game.queue)[:PREVIEW_COUNT]:
            lines.extend(self._preview_lines(PieceType(piece_type)))

        width = max(visible_len(line) for line in lines) if lines else 0
        padded = [self._pad(line, width) for line in lines]
        return padded

    def _preview_lines(self, piece_type: PieceType) -> List[str]:
        """A small boxed drawing of one upcoming piece (always 4 lines)."""
        canvas = Canvas(PREVIEW_CELLS_W * PX_W, PREVIEW_CELLS_H * PX_H)
        self._draw_grid(canvas, PREVIEW_CELLS_W, PREVIEW_CELLS_H)

        cells = rotated_cells(piece_type, 0)
        min_row = min(r for r, _ in cells)
        min_col = min(c for _, c in cells)
        height = max(r for r, _ in cells) - min_row + 1
        width = max(c for _, c in cells) - min_col + 1
        off_row = (PREVIEW_CELLS_H - height) // 2
        off_col = (PREVIEW_CELLS_W - width) // 2
        for row, col in cells:
            self._draw_cell(
                canvas,
                off_row + row - min_row,
                off_col + col - min_col,
                PIECE_COLORS[piece_type],
            )

        lines = canvas.to_lines(self.color)
        lines[1] = lines[1] + "  " + self._text(piece_type.name, LABEL)
        return lines

    def _stat(self, label: str, value: str) -> str:
        return self._text(f"{label:<8}", DIM) + self._text(f"{value:>6}", VALUE)

    def _text(self, text: str, color: Color) -> str:
        if not text:
            return text
        return _paint(text, color) if self.color else text

    def _pad(self, line: str, width: int) -> str:
        """Right-pad to a printed width (ANSI codes do not count)."""
        missing = width - visible_len(line)
        return line + " " * missing if missing > 0 else line

    # ------------------------------------------------------------------ frame

    def lines(self, game) -> List[str]:
        """The playfield and the panel side by side (all lines equal width)."""
        board = self.board_lines(game)
        if not self.panel:
            return board

        panel = self.panel_lines(game)
        # Bring the panel up to the playfield's height, then pair line by line.
        while len(panel) < len(board):
            panel.append("")
        gap = self.indent if panel else ""
        width = max(visible_len(line) for line in panel) if panel else 0
        return [
            self._pad(left, visible_len(left)) + gap + self._pad(right, width)
            for left, right in zip(board, panel)
        ]

    def frame(self, game, message: Optional[str] = None) -> str:
        """A full screen: playfield + panel, plus a status banner and controls."""
        lines = list(self.lines(game))
        banner = message
        if banner is None and game.is_game_over():
            banner = f"GAME OVER -- score {game.score}"
        if banner:
            lines.append("")
            lines.append(self.indent + self._text(banner, BANNER))
        lines.append("")
        lines.append(
            self.indent
            + self._text(
                "move \u2190 \u2192   soft drop \u2193   rotate \u2191 / x   ccw z   "
                "hard drop [space]   pause p   quit q",
                DIM,
            )
        )
        return "\n".join(lines) + (RESET if self.color else "")


# --------------------------------------------------------------- convenience


def frame(game, color: Optional[bool] = None, ghost: bool = True) -> str:
    """Render a whole game with defaults picked for the current terminal."""
    return Renderer(color=supports_color() if color is None else color,
                    ghost=ghost).frame(game)


def board(game, color: Optional[bool] = None, ghost: bool = True) -> str:
    """Render just the framed playfield (a drop-in for the old ASCII view)."""
    colored = supports_color() if color is None else color
    renderer = Renderer(color=colored, ghost=ghost, panel=False)
    return "\n".join(renderer.board_lines(game)) + (RESET if colored else "")
