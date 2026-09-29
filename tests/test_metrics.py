"""Tests for the board measurements the reward function uses."""

import numpy as np
import pytest

from tetris.board import EMPTY, HEIGHT, WIDTH, Board
from tetris.metrics import board_height, count_holes, empty_board, occupancy
from tetris.pieces import PieceType


def build(*rows: str) -> np.ndarray:
    """An occupancy grid from ASCII art, '.' empty and '#' filled.

    Missing rows are padded at the top, so tests can describe just the bottom
    of the board and keep the fixtures readable.
    """
    grid = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    for offset, row in enumerate(rows):
        line = row.replace(" ", "")
        assert len(line) <= WIDTH, f"row too wide: {row!r}"
        for col, char in enumerate(line):
            grid[HEIGHT - len(rows) + offset, col] = 1.0 if char == "#" else 0.0
    return grid


# ------------------------------------------------------------------ occupancy


def test_occupancy_makes_a_float_grid_from_engine_ids():
    board = Board()
    board.place([(HEIGHT - 1, 0), (HEIGHT - 1, 1)], int(PieceType.T))
    grid = occupancy(board.as_list())
    assert grid.shape == (HEIGHT, WIDTH)
    assert grid.dtype == np.float32
    assert set(np.unique(grid)) == {0.0, 1.0}
    assert grid[HEIGHT - 1, 0] == 1.0 and grid[HEIGHT - 1, 1] == 1.0
    assert grid.sum() == 2


def test_occupancy_is_idempotent_and_does_not_mutate_its_input():
    board = Board()
    board.place([(0, 5)], int(PieceType.I))
    before = board.as_list()
    grid = occupancy(board.as_list())
    assert (occupancy(grid) == grid).all()
    assert board.as_list() == before


def test_occupancy_rejects_non_2d_input():
    with pytest.raises(ValueError):
        occupancy(np.zeros((HEIGHT, WIDTH, 2)))


def test_empty_board_is_empty():
    grid = empty_board()
    assert grid.shape == (HEIGHT, WIDTH)
    assert grid.sum() == 0 and grid.dtype == np.float32


# --------------------------------------------------------------------- height


def test_an_empty_board_has_no_height():
    assert board_height(empty_board()) == 0


def test_height_counts_rows_from_the_highest_block_to_the_floor():
    assert board_height(build("#########.")) == 1
    assert board_height(build("#########.", "#########.")) == 2
    assert board_height(build("##........", "##........", "##........")) == 3


def test_height_is_the_full_board_when_a_block_sits_on_the_top_row():
    assert board_height(build("#########.", *[".........."] * (HEIGHT - 1))) == HEIGHT


def test_a_block_high_up_makes_the_stack_tall_even_with_a_gap_below():
    # One floating block on row 5 is still 15 rows of "stack" by this measure.
    assert board_height(build(*[".........."] * 4 + ["#........."] + [".........."] * 14)) == 15


# ---------------------------------------------------------------------- holes


def test_a_flat_surface_has_no_holes():
    assert count_holes(empty_board()) == 0
    assert count_holes(build("#########.")) == 0
    assert count_holes(build("#########.", "#########.")) == 0


def test_a_pit_is_not_a_hole_until_something_covers_it():
    # Empty cell at (19, 1) with open sky above it: reachable, not a hole.
    assert count_holes(build("#.#.......")) == 0


def test_covered_empty_cells_are_holes():
    # A roof over an empty cell makes exactly one hole...
    assert count_holes(build("###.......", "#.#.......")) == 1
    # ...and every empty cell under the roof counts, not just the first.
    assert count_holes(build("####......", "#...#.....")) == 3


def test_a_hole_can_be_several_rows_deep():
    assert count_holes(build("##........", "#.........", "#.........")) == 2


# --------------------------------------------------------------- the two together


def test_measurements_read_the_board_the_engine_produced():
    board = Board()
    board.place([(HEIGHT - 1, col) for col in range(WIDTH)], int(PieceType.O))
    assert board_height(occupancy(board.as_list())) == 1
    assert count_holes(occupancy(board.as_list())) == 0

    assert board.clear_lines() == 1
    assert board_height(occupancy(board.as_list())) == 0
    assert count_holes(occupancy(board.as_list())) == 0
