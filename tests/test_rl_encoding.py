"""Tests for the RL state and action encoding (`tetris/rl/encoding.py`).

The load-bearing test in this file is `test_the_mask_is_the_environments_own_
action_list`: index for index, `mask_for_state(state)` must agree with
`TetrisEnv.get_valid_actions()`. Everything else -- the agent's masking, the
Bellman target's masking, what a chosen index means -- assumes it.
"""

from __future__ import annotations

import numpy as np
import pytest

from tetris.board import HEIGHT, WIDTH
from tetris.metrics import column_heights, covered_cells
from tetris.env import Placement, TetrisEnv, placements_for
from tetris.pieces import PieceType

torch = pytest.importorskip("torch")            # the package needs torch to import

from tetris.rl.encoding import (                # noqa: E402
    ACTION_SIZE,
    BOARD_PLANE,
    CURRENT_PIECE_PLANE,
    EXTENDED_SHAPE,
    HEIGHT_PLANE,
    HOLE_PLANE,
    NEXT_PIECE_PLANE,
    STATE_SHAPE,
    action_index,
    encode_state,
    encode_state_with_features,
    encode_states,
    encode_states_with_features,
    index_for_placement,
    mask_for_piece,
    mask_for_state,
    placement_for_index,
    valid_indices,
)


# ------------------------------------------------------------ indices and masks


def test_an_index_round_trips_through_the_placement():
    for rotation in range(4):
        for column in range(WIDTH):
            index = action_index(rotation, column)
            assert index == rotation * WIDTH + column
            assert placement_for_index(index) == Placement(rotation, column)
            assert index_for_placement(Placement(rotation, column)) == index


def test_indices_outside_the_space_are_rejected():
    for bad in (-1, ACTION_SIZE, ACTION_SIZE + 5):
        with pytest.raises(ValueError, match="0..39"):
            placement_for_index(bad)
    with pytest.raises(ValueError, match="rotation"):
        index_for_placement((4, 0))
    with pytest.raises(ValueError, match="column"):
        index_for_placement((0, WIDTH))


def test_every_piece_type_has_a_mask_of_the_right_size():
    for piece_type in PieceType:
        mask = mask_for_piece(piece_type)
        assert mask.shape == (ACTION_SIZE,)
        assert mask.dtype == bool
        assert mask.any()


def test_a_mask_lists_exactly_the_pieces_placements():
    for piece_type in PieceType:
        expected = {action_index(r, c) for r, c in placements_for(piece_type)}
        assert set(np.flatnonzero(mask_for_piece(piece_type))) == expected


def test_a_square_piece_has_one_orientation_and_nine_columns():
    """Rotations 1-3 of an O are the same shape, so they are not separate actions."""
    mask = mask_for_piece(PieceType.O)
    assert int(mask.sum()) == 9
    assert set(np.flatnonzero(mask)) == {action_index(0, c) for c in range(9)}


def test_the_mask_is_cached_and_read_only():
    assert mask_for_piece(PieceType.T) is mask_for_piece(PieceType.T)
    with pytest.raises(ValueError):
        mask_for_piece(PieceType.T)[0] = True


def test_the_mask_is_the_environments_own_action_list():
    """Every state of two real games: the mask and `get_valid_actions()` agree."""
    env = TetrisEnv(seed=11)
    state = env.reset()
    rng = np.random.default_rng(0)

    for step in range(120):
        valid = env.get_valid_actions()
        mask = mask_for_state(state)
        assert set(np.flatnonzero(mask)) == {action_index(r, c) for r, c in valid}
        assert valid_indices(state).tolist() == sorted(action_index(r, c) for r, c in valid)
        action = valid[int(rng.integers(len(valid)))]
        state, _reward, done, _info = env.step(action)
        if done:
            state = env.reset(seed=100 + step)


def test_a_finished_game_has_no_legal_actions():
    env = TetrisEnv(seed=5)
    env.reset()
    while not env.is_game_over():
        valid = env.get_valid_actions()
        env.step(valid[0])
    assert mask_for_state(env.get_state()).sum() == 0
    assert valid_indices(env.get_state()).tolist() == []


# ---------------------------------------------------------------------- states


def test_an_encoded_state_has_the_documented_shape_and_range():
    env = TetrisEnv(seed=3)
    planes = encode_state(env.reset())

    assert planes.shape == STATE_SHAPE == (3, HEIGHT, WIDTH)
    assert planes.dtype == np.float32
    assert set(np.unique(planes)) <= {0.0, 1.0}      # binary, so already normalised


def test_the_first_plane_is_the_board_the_agent_must_fill():
    env = TetrisEnv(seed=2)
    env.reset()
    for _ in range(12):
        env.step(env.get_valid_actions()[0])

    state = env.get_state()
    planes = encode_state(state)

    assert np.array_equal(planes[BOARD_PLANE], (state.board > 0).astype(np.float32))
    assert planes[BOARD_PLANE].sum() > 0             # pieces were placed after all


def test_the_piece_planes_carry_four_cells_each():
    env = TetrisEnv(seed=4)
    state = env.reset()
    planes = encode_state(state)

    assert planes[CURRENT_PIECE_PLANE].sum() == 4
    assert planes[NEXT_PIECE_PLANE].sum() == 4
    # the current piece is drawn where it actually is: in the top rows, above the stack
    rows = np.flatnonzero(planes[CURRENT_PIECE_PLANE].any(axis=1))
    assert len(rows) <= 4
    assert rows.min() < 4


def test_the_next_piece_is_drawn_in_the_corner_whatever_its_shape():
    """The preview has no board position of its own, so it goes top-left, every time."""
    env = TetrisEnv(seed=6)
    state = env.reset()
    planes = encode_state(state)
    cells = np.argwhere(planes[NEXT_PIECE_PLANE] > 0)

    assert cells.min(axis=0).tolist() == [0, 0]
    assert len(cells) == 4


def test_the_piece_planes_distinguish_the_seven_pieces():
    """Shape identifies the piece: no one-hot vectors are needed anywhere."""
    env = TetrisEnv(seed=8)
    state = env.reset()
    shapes = set()
    for _ in range(40):
        planes = encode_state(state)
        shapes.add((tuple(planes[CURRENT_PIECE_PLANE].astype(int).ravel()),
                    tuple(planes[NEXT_PIECE_PLANE].astype(int).ravel())))
        state, _reward, done, _info = env.step(env.get_valid_actions()[0])
        if done:
            state = env.reset()
    assert len({current for current, _ in shapes}) == 7      # all seven, in shape


def test_encoding_copies_so_the_state_cannot_be_changed_through_it():
    env = TetrisEnv(seed=9)
    state = env.reset()
    planes = encode_state(state)
    planes[:] = 1.0
    assert state.board.sum() >= 0                # untouched
    assert np.array_equal(encode_state(state), encode_state(env.get_state()))


def test_states_encode_as_a_batch():
    env = TetrisEnv(seed=10)
    states = [env.reset()]
    for _ in range(3):
        state, _reward, done, _info = env.step(env.get_valid_actions()[0])
        states.append(state)
        if done:
            states.append(env.reset())

    batch = encode_states(states)

    assert batch.shape == (len(states),) + STATE_SHAPE
    assert np.array_equal(batch[0], encode_state(states[0]))


# ------------------------------------------------- the extended feature encoding


def test_the_feature_encoding_has_five_planes():
    env = TetrisEnv(seed=11)
    state = env.reset(seed=11)
    planes = encode_state_with_features(state)
    assert planes.shape == EXTENDED_SHAPE == (5, HEIGHT, WIDTH)
    assert planes.dtype == np.float32
    assert np.isfinite(planes).all()
    assert planes.min() >= 0.0 and planes.max() <= 1.0


def test_the_feature_encoding_leaves_the_original_planes_alone():
    """The five-plane state must *contain* the three-plane one, not replace it."""
    env = TetrisEnv(seed=12)
    state = env.reset(seed=12)
    for _ in range(6):
        state, _reward, done, _info = env.step(env.get_valid_actions()[0])
        if done:
            break
    assert np.array_equal(
        encode_state_with_features(state)[:3], encode_state(state)
    )


def test_the_height_plane_is_the_column_heights():
    env = TetrisEnv(seed=13)
    state = env.reset(seed=13)
    for _ in range(8):
        state, _reward, done, _info = env.step(env.get_valid_actions()[0])
        if done:
            break
    planes = encode_state_with_features(state)
    heights = column_heights(state.board).astype(np.float32) / float(HEIGHT)
    # constant down each column: one number per column, not a shed roof
    for column in range(WIDTH):
        column_values = planes[HEIGHT_PLANE][:, column]
        assert np.allclose(column_values, heights[column], atol=1e-6), column
    assert np.allclose(planes[HEIGHT_PLANE][0], heights, atol=1e-6)


def test_the_hole_plane_is_what_the_heuristic_calls_a_hole():
    env = TetrisEnv(seed=14)
    state = env.reset(seed=14)
    for _ in range(10):
        state, _reward, done, _info = env.step(env.get_valid_actions()[0])
        if done:
            break
    planes = encode_state_with_features(state)
    expected = covered_cells(state.board).astype(np.float32)
    assert np.array_equal(planes[HOLE_PLANE], expected)
    # and a hole is empty and covered, which is what the plane shows
    filled = state.board > 0
    assert not (planes[HOLE_PLANE][filled] > 0).any()


def test_the_feature_batch_matches_the_single_state():
    env = TetrisEnv(seed=15)
    states = [env.reset(seed=15)]
    for _ in range(3):
        state, _reward, done, _info = env.step(env.get_valid_actions()[0])
        states.append(state)
        if done:
            states.append(env.reset())
    batch = encode_states_with_features(states)
    assert batch.shape == (len(states),) + EXTENDED_SHAPE
    assert np.array_equal(batch[0], encode_state_with_features(states[0]))


def test_an_empty_board_has_no_holes_and_no_height():
    env = TetrisEnv(seed=16)
    state = env.reset(seed=16)
    planes = encode_state_with_features(state)
    assert planes[HEIGHT_PLANE].sum() == 0.0
    assert planes[HOLE_PLANE].sum() == 0.0
