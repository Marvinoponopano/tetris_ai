"""Tests for the board grid and collision rules."""

from tetris.board import EMPTY, HEIGHT, WIDTH, Board


def test_board_dimensions():
    board = Board()
    assert len(board.grid) == HEIGHT
    assert all(len(row) == WIDTH for row in board.grid)


def test_board_starts_empty():
    board = Board()
    assert all(cell == EMPTY for row in board.grid for cell in row)


def test_collides_left_wall():
    board = Board()
    assert board.collides([(0, -1)])       # left of the board
    assert not board.collides([(0, 0)])    # first column is fine


def test_collides_right_wall():
    board = Board()
    assert board.collides([(0, WIDTH)])    # one past the right edge
    assert not board.collides([(0, WIDTH - 1)])


def test_collides_bottom():
    board = Board()
    assert board.collides([(HEIGHT, 0)])   # below the floor
    assert not board.collides([(HEIGHT - 1, 0)])


def test_above_board_is_legal():
    board = Board()
    assert not board.collides([(-5, 0)])   # hidden rows above are legal


def test_collides_occupied_cell():
    board = Board()
    board.grid[5][5] = 1
    assert board.collides([(5, 5)])
    assert not board.collides([(6, 5)])


def test_place_writes_values():
    board = Board()
    board.place([(0, 0), (0, 1)], 7)
    assert board.grid[0][0] == 7
    assert board.grid[0][1] == 7
    assert board.grid[0][2] == EMPTY


def test_place_ignores_cells_above_board():
    board = Board()
    board.place([(-1, 0), (0, 0)], 3)
    assert board.grid[0][0] == 3  # only the visible cell lands


def test_clear_single_line():
    board = Board()
    for col in range(WIDTH):
        board.grid[HEIGHT - 1][col] = 1  # fill the bottom row
    assert board.clear_lines() == 1
    assert all(cell == EMPTY for cell in board.grid[HEIGHT - 1])


def test_clear_multiple_lines():
    board = Board()
    for row in range(HEIGHT - 2, HEIGHT):
        for col in range(WIDTH):
            board.grid[row][col] = 1
    assert board.clear_lines() == 2
    assert all(cell == EMPTY for cell in board.grid[HEIGHT - 1])
    assert all(cell == EMPTY for cell in board.grid[HEIGHT - 2])


def test_clear_lines_shifts_rows_above_down():
    board = Board()
    board.grid[0][0] = 7
    for col in range(WIDTH):
        board.grid[HEIGHT - 1][col] = 1
    board.clear_lines()
    assert board.grid[1][0] == 7   # the marked cell shifted down one row
    assert board.grid[0][0] == EMPTY


def test_as_list_returns_copy():
    board = Board()
    snapshot = board.as_list()
    snapshot[0][0] = 99
    assert board.grid[0][0] == EMPTY  # original is untouched
