"""Tetromino definitions, rotation, and the Piece object.

Design decisions
----------------

1. Every piece is stored as a tuple of `(row, col)` offsets around a
   "rotation center" cell at (0, 0). A coordinate list is much easier to
   reason about for collision detection than a 4x4 matrix, and it avoids
   hardcoding 4 orientations x 7 pieces = 28 shapes by hand.

2. Rotation is ONE generic formula, shared by every piece::

       clockwise:         (row, col) -> (col, -row)
       counter-clockwise: (row, col) -> (-col, row)

   Because board rows grow *downward*, this sign gives the correct
   "clockwise as seen on screen" direction. No per-piece special cases.

3. The O piece is the single exception: rotating a 2x2 block around a corner
   cell would make it "hop" to a different spot, so O simply does not rotate
   (this matches real Tetris).

`PieceType` is an `IntEnum`, so a locked cell on the board can store a plain
integer 1..7 (0 = empty). That makes the board trivially convertible to a
NumPy array later.
"""

from enum import IntEnum
from typing import Dict, List, Optional, Tuple

# A cell is a (row_offset, col_offset) relative to the rotation center (0, 0).
Cell = Tuple[int, int]


class PieceType(IntEnum):
    """The seven tetrominoes. Their integer value (1..7) is what the board stores."""

    I = 1
    O = 2
    T = 3
    S = 4
    Z = 5
    J = 6
    L = 7

    @property
    def base_cells(self) -> Tuple[Cell, ...]:
        """The spawn orientation as (row, col) offsets around (0, 0)."""
        return BASE_CELLS[self]


# Spawn orientation of each piece, centered on the rotation center (0, 0).
# "Row" grows downward (row 0 = top), so negative rows are above the center.
BASE_CELLS = {
    PieceType.I: ((0, -2), (0, -1), (0, 0), (0, 1)),
    PieceType.O: ((-1, -1), (-1, 0), (0, -1), (0, 0)),
    PieceType.T: ((-1, 0), (0, -1), (0, 0), (0, 1)),
    PieceType.S: ((-1, 0), (-1, 1), (0, -1), (0, 0)),
    PieceType.Z: ((-1, -1), (-1, 0), (0, 0), (0, 1)),
    PieceType.J: ((-1, -1), (0, -1), (0, 0), (0, 1)),
    PieceType.L: ((-1, 1), (0, -1), (0, 0), (0, 1)),
}

# Standard Tetris colors (for a renderer later).
COLORS: Dict[PieceType, str] = {
    PieceType.I: "cyan",
    PieceType.O: "yellow",
    PieceType.T: "purple",
    PieceType.S: "green",
    PieceType.Z: "red",
    PieceType.J: "blue",
    PieceType.L: "orange",
}


def rotate_cw(cells: Tuple[Cell, ...]) -> Tuple[Cell, ...]:
    """Rotate a set of cells 90 degrees clockwise around (0, 0)."""
    return tuple((col, -row) for row, col in cells)


def rotate_ccw(cells: Tuple[Cell, ...]) -> Tuple[Cell, ...]:
    """Rotate a set of cells 90 degrees counter-clockwise around (0, 0)."""
    return tuple((-col, row) for row, col in cells)


def rotated_cells(piece_type: PieceType, rotation: int) -> Tuple[Cell, ...]:
    """The piece's cells after `rotation` clockwise quarter-turns (0..3)."""
    cells = piece_type.base_cells
    for _ in range(rotation % 4):
        cells = rotate_cw(cells)
    return cells


class Piece:
    """A tetromino with a type, a rotation state, and a position on the board.

    `row` / `col` are the board coordinates of the *rotation center* cell.
    The actual cells are derived from `type` + `rotation`, so there is a single
    source of truth and the two can never drift apart.
    """

    def __init__(
        self,
        piece_type: PieceType,
        row: int = 0,
        col: int = 0,
        rotation: int = 0,
    ):
        self.type = piece_type
        self.row = row
        self.col = col
        self.rotation = rotation % 4

    @property
    def cells(self) -> Tuple[Cell, ...]:
        """The piece's cells in its current orientation (relative to center)."""
        return rotated_cells(self.type, self.rotation)

    def rotate_cw(self) -> "Piece":
        if self.type is PieceType.O:
            return Piece(self.type, self.row, self.col, self.rotation)
        return Piece(self.type, self.row, self.col, self.rotation + 1)

    def rotate_ccw(self) -> "Piece":
        if self.type is PieceType.O:
            return Piece(self.type, self.row, self.col, self.rotation)
        return Piece(self.type, self.row, self.col, self.rotation - 1)

    def absolute_cells(
        self,
        row: Optional[int] = None,
        col: Optional[int] = None,
    ) -> List[Cell]:
        """The piece's cells translated into board coordinates."""
        r0 = self.row if row is None else row
        c0 = self.col if col is None else col
        return [(r0 + dr, c0 + dc) for dr, dc in self.cells]

    def __repr__(self) -> str:
        return f"Piece({self.type.name}, row={self.row}, col={self.col}, rot={self.rotation})"
