"""The 10x20 game board and collision rules.

Design decisions
----------------

1. The board is a plain Python `list` of `list`s: `grid[row][col]`.

       * `row` 0 is the TOP visible row, `row` 19 is the floor.
       * `col` 0..9 are the columns.

   This is how a human reads a Tetris board, and it converts to a NumPy array
   with a single `np.array(grid)` call (nothing to transpose).

2. A cell holds `EMPTY` (0) or the integer id (1..7) of the piece that locked
   there. Storing the piece id -- rather than just 1 -- lets a renderer color
   locked cells and gives an ML system richer features for free.

3. Collision logic lives here because it only depends on the board and a list
   of candidate cells. Cells with `row < 0` are legal: a piece may still be
   "hidden" above the visible board while it spawns.
"""

EMPTY = 0

WIDTH = 10
HEIGHT = 20


class Board:
    def __init__(self, width: int = WIDTH, height: int = HEIGHT):
        self.width = width
        self.height = height
        self.grid = [[EMPTY] * width for _ in range(height)]

    # ------------------------------------------------------------------ queries

    def collides(self, cells) -> bool:
        """True if any cell is out of bounds or already occupied.

        Rows above the board (row < 0) are always legal.
        """
        for row, col in cells:
            if col < 0 or col >= self.width:
                return True
            if row >= self.height:
                return True
            if row >= 0 and self.grid[row][col] != EMPTY:
                return True
        return False

    # --------------------------------------------------------------- mutations

    def place(self, cells, value: int) -> None:
        """Write `value` into every cell (used to lock a piece in place).

        Cells above the visible board (row < 0) are silently ignored.
        """
        for row, col in cells:
            if 0 <= row < self.height:
                self.grid[row][col] = value

    def clear_lines(self) -> int:
        """Remove every full row and return how many were cleared.

        Rows above a cleared row shift down; empty rows are added at the top.
        """
        kept = [row for row in self.grid if not self._row_is_full(row)]
        cleared = self.height - len(kept)
        empty_rows = [[EMPTY] * self.width for _ in range(cleared)]
        self.grid = empty_rows + kept
        return cleared

    # ------------------------------------------------------------------- helpers

    def _row_is_full(self, row) -> bool:
        return all(cell != EMPTY for cell in row)

    def as_list(self):
        """A deep copy of the grid (safe to hand to callers / ML code)."""
        return [row[:] for row in self.grid]

    def reset(self) -> None:
        """Empty the board (used by `Game.reset`)."""
        self.grid = [[EMPTY] * self.width for _ in range(self.height)]
