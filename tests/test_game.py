"""Tests for the Game environment: spawning, movement, locking, scoring, ..."""

import random

from tetris.actions import Action
from tetris.board import EMPTY, HEIGHT, WIDTH
from tetris.game import Game
from tetris.pieces import Piece, PieceType


def test_reset_starts_playable_game():
    game = Game()
    assert not game.is_game_over()
    assert game.current is not None
    state = game.get_state()
    assert state.current_piece_type is not None
    assert state.score == 0


def test_spawn_is_inside_horizontal_bounds():
    game = Game()
    for _, dc in game.current.cells:
        col = game.current.col + dc
        assert 0 <= col < WIDTH


def test_spawn_top_cell_at_row_zero():
    game = Game()
    top_row = min(r for r, _ in game.current.absolute_cells())
    assert top_row == 0


def test_move_left_and_right():
    game = Game()
    start_col = game.current.col
    game.step(Action.RIGHT)
    assert game.current.col == start_col + 1
    game.step(Action.LEFT)
    assert game.current.col == start_col


def test_cannot_move_through_left_wall():
    game = Game()
    game.current = Piece(PieceType.I, row=5, col=2)  # leftmost cell at col 0
    game.step(Action.LEFT)
    assert game.current.col == 2  # unchanged


def test_cannot_move_through_right_wall():
    game = Game()
    game.current = Piece(PieceType.I, row=5, col=8)  # rightmost cell at col 9
    game.step(Action.RIGHT)
    assert game.current.col == 8  # unchanged


def test_rotate_changes_orientation():
    game = Game()
    # Use a fixed T in the open middle of the board: rotation must always work
    # here regardless of which piece happened to spawn (O would not rotate).
    game.current = Piece(PieceType.T, row=5, col=4)
    before = game.current.cells
    game.step(Action.ROTATE_CW)
    assert game.current.cells != before
    assert game.current.rotation == 1


def test_rotation_wall_kick():
    game = Game()
    # A vertical I against the left wall: rotating to horizontal would put a
    # cell out of bounds, so the wall kick must shift it right by one.
    game.current = Piece(PieceType.I, row=5, col=0, rotation=1)
    game.step(Action.ROTATE_CW)
    assert game.current.rotation == 2  # now horizontal
    assert game.current.col == 1       # kicked right


def test_soft_drop_then_lock():
    game = Game()
    game.current = Piece(PieceType.O, row=HEIGHT - 2, col=4)
    game.step(Action.DOWN)
    assert game.current.row == HEIGHT - 1  # fell one cell
    game.step(Action.DOWN)
    # It could not fall further, so it locked into the board and a new piece
    # spawned near the top.
    assert game.board.grid[HEIGHT - 1][4] == int(PieceType.O)
    assert game.board.grid[HEIGHT - 1][3] == int(PieceType.O)
    assert game.current is not None
    assert game.current.row < HEIGHT - 1


def test_gravity_falls_and_locks():
    game = Game()
    game.current = Piece(PieceType.O, row=HEIGHT - 2, col=4)
    game.apply_gravity()
    assert game.current.row == HEIGHT - 1
    game.apply_gravity()  # cannot fall -> locks
    assert game.board.grid[HEIGHT - 1][4] == int(PieceType.O)


def test_hard_drop_locks_and_spawns():
    game = Game()
    game.step(Action.DROP)
    # The first piece landed on the floor, and a new piece has spawned.
    assert any(cell != EMPTY for cell in game.board.grid[HEIGHT - 1])
    assert game.current is not None
    assert not game.is_game_over()


def test_line_clear_updates_score_and_lines():
    game = Game()
    # Fill the bottom row except a 2-wide gap under where an O will land.
    for col in range(WIDTH):
        game.board.grid[HEIGHT - 1][col] = int(PieceType.I)
    game.board.grid[HEIGHT - 1][4] = EMPTY
    game.board.grid[HEIGHT - 1][5] = EMPTY
    game.current = Piece(PieceType.O, row=HEIGHT - 1, col=5)
    score_before = game.score
    game.step(Action.DOWN)  # cannot move -> locks and clears the line
    assert game.lines_cleared == 1
    assert game.score - score_before == 100  # 1 line x level 1


def test_game_over_when_spawn_is_blocked():
    game = Game()
    # Fill the top two rows so no piece can fit.
    for row in range(2):
        for col in range(WIDTH):
            game.board.grid[row][col] = int(PieceType.I)
    game.current = None
    game._spawn_piece()
    assert game.is_game_over()


def test_game_over_lock_out():
    game = Game()
    # A piece that locks entirely above the visible board ends the game.
    game.current = Piece(PieceType.I, row=-4, col=0)
    game._lock_piece()
    assert game.is_game_over()


def test_seven_bag_randomizer():
    game = Game(rng=random.Random(42))
    seen = [game.current.type]
    for _ in range(6):
        game.step(Action.DROP)
        seen.append(game.current.type)
    # The first seven pieces are exactly one of each type.
    assert sorted(t.value for t in seen) == list(range(1, 8))


def test_get_valid_actions_at_left_wall():
    game = Game()
    game.current = Piece(PieceType.I, row=5, col=2)  # leftmost at col 0
    valid = game.get_valid_actions()
    assert Action.LEFT not in valid
    assert Action.RIGHT in valid
    assert Action.DOWN in valid
    assert Action.DROP in valid


def test_get_valid_actions_game_over_returns_empty():
    game = Game()
    game.game_over = True
    assert game.get_valid_actions() == []


def test_step_returns_state_reward_done_info():
    game = Game()
    result = game.step(Action.DROP)
    assert len(result) == 4
    state, reward, done, info = result
    assert state.game_over is False
    assert isinstance(reward, int)
    assert done is False
    assert "lines_cleared" in info


def test_plain_move_has_zero_reward():
    game = Game()
    _, reward, _, _ = game.step(Action.RIGHT)
    assert reward == 0


def test_board_with_active_piece_overlays_piece():
    game = Game()
    state = game.get_state()
    merged = state.board_with_active_piece()
    filled = sum(1 for row in merged for cell in row if cell != EMPTY)
    assert filled == 4  # exactly the four cells of the active piece
    # The underlying locked board is unchanged.
    assert all(cell == EMPTY for row in state.board for cell in row)
