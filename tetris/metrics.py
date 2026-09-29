"""Board measurements: what the reward function and the heuristic agent look at.

These take a board in any of the shapes the project already uses -- the engine's
`list[list[int]]`, a numpy array of piece ids, or a 0/1 occupancy grid -- and
return plain numbers or a fresh array. Nothing here mutates its input and
nothing here knows about the environment, pygame, the terminal or any agent, so
every measurement can be reasoned about (and tested) on its own.

The six features the heuristic agent scores are collected by `extract_features`;
each one also has a function of its own, and `extract_features` simply computes
them together because they all start from the same column heights.

Vocabulary used below
    * column height  -- rows from a column's highest filled cell down to the floor
    * hole           -- an empty cell with a filled cell somewhere above it
    * bumpiness      -- total height difference between neighbouring columns
    * well           -- a column lower than the blocks (or walls) beside it
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple  # noqa: UP035

import numpy as np

from .board import EMPTY, HEIGHT, WIDTH

Board = np.ndarray

#: The feature names, in the order the heuristic weights use them.
FEATURE_NAMES: Tuple[str, ...] = (
    "aggregate_height",
    "max_height",
    "holes",
    "bumpiness",
    "completed_lines",
    "wells",
)


def occupancy(board) -> Board:
    """A (height, width) float32 grid: 1.0 where a block sits, 0.0 where it is empty.

    Accepts the engine's board (ids 1..7), a numpy equivalent, or an existing
    occupancy grid (`True`/`1.0`), so a measurement can be taken from any point
    in the pipeline. Always a fresh array: callers may freely write into the
    result without touching the board they passed in.
    """
    grid = np.asarray(board)
    if grid.ndim != 2:
        raise ValueError(f"a board must be 2D, got shape {grid.shape}")
    return (grid != EMPTY).astype(np.float32)


def filled_cells(board) -> np.ndarray:
    """A boolean (height, width) grid: True where a block sits."""
    grid = np.asarray(board)
    if grid.ndim != 2:
        raise ValueError(f"a board must be 2D, got shape {grid.shape}")
    return grid != EMPTY


# ---------------------------------------------------------------- stack shape


def column_heights(occupancy: Board) -> np.ndarray:
    """The height of every column, as an int array of length `width`.

    A column's height is the distance from its highest filled cell to the floor,
    so an empty column is 0 and a column filled to the top is `height`. A block
    floating above empty space still counts: it is in the way.
    """
    filled = filled_cells(occupancy)
    height = filled.shape[0]
    first_filled = np.where(filled.any(axis=0), filled.argmax(axis=0), height)
    return (height - first_filled).astype(int)


def board_height(occupancy: Board) -> int:
    """How tall the stack is: rows from the highest filled cell down to the floor.

    Zero for an empty board; the full height (20) when a block sits on the top row.
    """
    return max_column_height(occupancy)


def max_column_height(occupancy: Board) -> int:
    """The tallest column of the board (`board_height` is the same number)."""
    heights = column_heights(occupancy)
    return int(heights.max()) if heights.size else 0


def aggregate_height(occupancy: Board) -> int:
    """The sum of all column heights: roughly how much of the well is filled.

    Painful when non-zero, fatal when large, and the simplest way to say "the
    stack is getting high". Compared with `max_column_height` it tracks the
    whole surface, so piling up one tower costs less than building a plateau.
    """
    return int(column_heights(occupancy).sum())


def bumpiness(occupancy: Board) -> int:
    """Total height difference between neighbouring columns.

    A jagged surface is hard to place pieces on: flat-bottomed pieces leave gaps
    under them, and every spike has to be built over before the row can be
    cleared.
    """
    heights = column_heights(occupancy)
    if heights.size < 2:
        return 0
    return int(np.abs(np.diff(heights)).sum())


def top_rows(occupancy: Board) -> np.ndarray:
    """For each column, the row of its highest filled cell (or `height` if empty).

    The landing row of a piece dropped into a column follows from these, which
    is how the heuristic simulates a placement without walking the piece down
    the board.
    """
    filled = filled_cells(occupancy)
    height = filled.shape[0]
    return np.where(filled.any(axis=0), filled.argmax(axis=0), height).astype(int)


# ------------------------------------------------------------ rows and holes


def covered_cells(occupancy: Board) -> np.ndarray:
    """Boolean grid of holes: empty cells with a filled cell above them."""
    filled = filled_cells(occupancy)
    return np.maximum.accumulate(filled, axis=0) & ~filled


def count_holes(occupancy: Board) -> int:
    """Empty cells that have a filled cell somewhere above them in the column.

    These are the cells a player can no longer reach without clearing lines, so
    the default reward penalises creating new ones and the heuristic avoids
    them hard: every hole can cost a whole column's worth of future line clears.
    """
    return int(covered_cells(occupancy).sum())


def completed_lines(occupancy: Board) -> int:
    """How many rows are completely filled (they have not been cleared yet).

    The point of the game: each completed row is progress, and four at once is
    worth far more than four separate rows.
    """
    filled = filled_cells(occupancy)
    return int(filled.all(axis=1).sum())


# ---------------------------------------------------------------------- wells


def well_depths(occupancy: Board) -> np.ndarray:
    """How much deeper each column is than the blocks flanking it, as ints.

    A column is a well where it is lower than its neighbours, and the depth is
    the step down to it. The walls count as solid sides, so a gap pressed
    against the wall with a taller column beside it is a well too -- exactly the
    slot an I piece is dropped into.

    Deep wells are usually a nuisance (nothing else fits in them, so they wait
    for a long piece), which is why the heuristic scores them negatively.
    """
    heights = column_heights(occupancy)
    if heights.size == 0:
        return heights.astype(int)
    # The wall is as solid as the tallest thing in the well, so a gap against
    # the wall with a taller column beside it still counts as a well.
    wall = np.array([filled_cells(occupancy).shape[0]])
    left = np.concatenate([wall, heights[:-1]])
    right = np.concatenate([heights[1:], wall])
    return np.maximum(0, np.minimum(left, right) - heights).astype(int)


def wells(occupancy: Board) -> int:
    """Total well depth across the board (see `well_depths`)."""
    return int(well_depths(occupancy).sum())


# ------------------------------------------------------------------ together


@dataclass(frozen=True)
class BoardFeatures:
    """The six numbers the heuristic agent's weights are applied to.

    Every field is also a function of its own in this module, so each can be
    tested in isolation; `extract_features` computes them from one shared pass.
    """

    aggregate_height: int
    max_height: int
    holes: int
    bumpiness: int
    completed_lines: int
    wells: int

    def as_dict(self) -> Dict[str, int]:
        """The features as a plain dict, keyed like the heuristic weights."""
        return {name: getattr(self, name) for name in FEATURE_NAMES}

    def __repr__(self) -> str:  # a one-line summary reads better than six lines
        pairs = " ".join(f"{name}={getattr(self, name)}" for name in FEATURE_NAMES)
        return f"BoardFeatures({pairs})"


def extract_features(board) -> BoardFeatures:
    """Every feature of one board (see `BoardFeatures`).

    Takes a board in any of the shapes `occupancy` accepts and returns plain
    ints, so the result can be compared, logged and scored with no numpy
    knowledge. The column heights are computed once and shared by all six.
    """
    grid = occupancy(board)
    heights = column_heights(grid)
    holes = covered_cells(grid)
    height, width = grid.shape

    wall = np.array([height])
    left = np.concatenate([wall, heights[:-1]]) if width else heights
    right = np.concatenate([heights[1:], wall]) if width else heights
    depths = np.maximum(0, np.minimum(left, right) - heights) if heights.size else heights

    return BoardFeatures(
        aggregate_height=int(heights.sum()),
        max_height=int(heights.max()) if heights.size else 0,
        holes=int(holes.sum()),
        bumpiness=int(np.abs(np.diff(heights)).sum()) if heights.size > 1 else 0,
        completed_lines=int(grid.all(axis=1).sum()) if width else 0,
        wells=int(depths.sum()),
    )


def empty_board() -> Board:
    """A fresh all-empty occupancy grid (handy in tests and for padding states)."""
    return np.zeros((HEIGHT, WIDTH), dtype=np.float32)
