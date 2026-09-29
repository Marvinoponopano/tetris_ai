"""Tests for the reinforcement-learning environment (`tetris.env`).

These check the *contract* an agent depends on: the shapes and ranges of the
observation, the legality of every action the environment offers, the reward
wiring, termination, and seeded determinism. Nothing here trains anything, and
nothing here needs a display.
"""

import pathlib
import random
import subprocess
import sys

import numpy as np
import pytest

from tetris.board import HEIGHT, WIDTH
from tetris.env import (
    FEATURE_VECTOR_SIZE,
    N_PIECE_TYPES,
    Placement,
    RewardConfig,
    TetrisEnv,
    TetrisState,
    _orientation_for,
)
from tetris.metrics import occupancy
from tetris.pieces import Piece, PieceType

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------- helpers


def force_piece(env: TetrisEnv, piece_type: PieceType, rotation: int = 0) -> None:
    """Make the environment's active piece a specific tetromino.

    The piece queue is random, so tests that care about a particular shape put
    one there directly. Only the piece *type* (and rotation) matter: `step()`
    repositions whatever piece is active before dropping it.
    """
    env.game.current = Piece(piece_type, row=0, col=4, rotation=rotation)


def fill(env: TetrisEnv, cells, piece_type: PieceType = PieceType.J) -> None:
    """Lock blocks into the board, to set up a position."""
    env.game.board.place(list(cells), int(piece_type))


def bottom_row(columns) -> list:
    return [(HEIGHT - 1, col) for col in columns]


def placed_cells(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """The cells that changed from empty to filled, as (row, col) pairs."""
    return np.argwhere(after > before)


def top_out(env: TetrisEnv) -> None:
    """Fill the board so the next placement cannot fit -> game over."""
    fill(env, [(row, col) for row in range(HEIGHT) for col in range(1, WIDTH)])
    env.game.current = Piece(PieceType.O, row=0, col=4)
    env.step((0, 1))


def play(env: TetrisEnv, rng: random.Random, max_pieces: int = 500):
    """Play an episode with a random legal policy; return `(steps, total reward, info)`."""
    total = 0.0
    steps = 0
    info = {}
    while not env.is_game_over() and steps < max_pieces:
        state, reward, done, info = env.step(rng.choice(env.get_valid_actions()))
        total += reward
        steps += 1
        if done:
            break
    return steps, total, info


# -------------------------------------------------------------------- reset


def test_reset_returns_a_state_with_an_empty_board():
    env = TetrisEnv(seed=1)
    state = env.reset()

    assert isinstance(state, TetrisState)
    assert state.board.shape == (HEIGHT, WIDTH)
    assert state.board.sum() == 0
    assert not state.game_over
    assert env.pieces_placed == 0


def test_reset_puts_exactly_four_falling_cells_on_the_board():
    state = TetrisEnv(seed=1).reset()
    assert state.board_with_piece.sum() == 4
    assert len(state.current_piece) == 4
    assert len({tuple(cell) for cell in state.current_piece}) == 4
    assert list(state.current_piece_type).count(1.0) == 1


def test_reset_starts_a_playable_episode_with_a_next_piece():
    state = TetrisEnv(seed=7).reset()
    assert state.current_piece_name in {piece.name for piece in PieceType}
    assert state.next_piece_name in {piece.name for piece in PieceType}
    assert state.queue.size >= 7
    assert int(state.queue[0]) == int(state.next_piece_type.argmax()) + 1
    assert state.score == 0 and state.lines_cleared == 0 and state.level == 1


def test_reset_clears_previous_progress():
    env = TetrisEnv(seed=3)
    env.step((0, 0))
    env.step((0, 0))
    assert env.pieces_placed == 2

    state = env.reset()
    assert env.pieces_placed == 0
    assert state.score == 0
    assert state.board.sum() == 0


def test_reset_can_switch_to_another_episode():
    env = TetrisEnv(seed=1)
    first = env.reset().queue.copy()
    env.reset(seed=2)
    assert env.seed == 2
    assert not np.array_equal(first, env.reset().queue)


# -------------------------------------------------------------------- step


def test_step_returns_state_reward_done_and_info():
    env = TetrisEnv(seed=5)
    env.reset()
    result = env.step((0, 0))

    assert isinstance(result, tuple) and len(result) == 4
    state, reward, done, info = result
    assert isinstance(state, TetrisState)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_step_locks_the_piece_and_spawns_the_next_one():
    env = TetrisEnv(seed=11)
    state = env.reset()
    next_before = state.next_piece_name

    state, reward, done, info = env.step((0, 0))

    assert state.board.sum() == 4          # the piece landed on an empty board
    assert env.pieces_placed == 1
    assert state.current_piece_name == next_before
    assert info["piece"] is not None
    assert not done


def test_step_reports_the_placement_it_made():
    env = TetrisEnv(seed=11)
    env.reset()
    _, _, _, info = env.step({"rotation": 0, "column": 6})
    assert info["placement"] == (0, 6)


def test_all_three_action_spellings_are_the_same_move():
    boards = []
    for action in (Placement(0, 3), (0, 3), {"rotation": 0, "column": 3}):
        env = TetrisEnv(seed=99)
        env.reset()
        force_piece(env, PieceType.T)
        state, _, _, _ = env.step(action)
        boards.append(state.board)
    assert np.array_equal(boards[0], boards[1])
    assert np.array_equal(boards[1], boards[2])


def test_step_rejects_actions_it_cannot_play():
    env = TetrisEnv(seed=5)
    env.reset()

    for bad in (
        (4, 0),                                     # rotation out of range
        (-1, 0),
        (0, 10),                                    # column off the right wall
        (0, -1),
        (0.5, 3),                                   # not whole numbers
        (0, "three"),
        (0,),                                       # wrong arity
        (0, 3, 4),
        {"rotation": 0},                            # missing a key
        None,
        "place it",
    ):
        with pytest.raises(ValueError):
            env.step(bad)


def test_the_column_range_depends_on_the_piece_width():
    env = TetrisEnv(seed=5)
    env.reset()

    force_piece(env, PieceType.I)
    env.step((0, 6))                                # an I is 4 wide: 0..6 is legal

    env.reset()                                     # step() spawned a new piece
    force_piece(env, PieceType.I)
    with pytest.raises(ValueError, match="use 0..6"):
        env.step((0, 7))

    env.reset()
    force_piece(env, PieceType.I, rotation=1)
    env.step((1, 9))                                # standing up it is 1 wide

    env.reset()
    force_piece(env, PieceType.I, rotation=1)
    with pytest.raises(ValueError, match="use 0..9"):
        env.step((1, 10))


def test_stepping_a_finished_episode_is_a_harmless_no_op():
    env = TetrisEnv(seed=4)
    top_out(env)
    assert env.is_game_over()

    state, reward, done, info = env.step((0, 0))
    assert done and reward == 0.0
    assert info["game_over"] is True
    assert info["placement"] is None
    assert np.array_equal(state.board, env.get_state().board)


# ------------------------------------------------------------ valid actions


def test_valid_actions_are_placements_that_compare_to_tuples():
    env = TetrisEnv(seed=8)
    env.reset()
    actions = env.get_valid_actions()

    assert actions, "a fresh game always has placements available"
    assert all(isinstance(action, Placement) for action in actions)
    assert (0, 0) in actions
    assert actions[0] == (actions[0].rotation, actions[0].column)
    assert [tuple(action) for action in actions] == sorted(tuple(a) for a in actions)


@pytest.mark.parametrize(
    "piece_type,widths",
    [
        (PieceType.O, {0: 2}),                     # one distinct orientation, 2 wide
        (PieceType.I, {0: 4, 1: 1}),               # flat is 4 wide, standing up 1
        (PieceType.S, {0: 3, 1: 2}),               # upright S/Z/J/L/T are 2 wide
        (PieceType.Z, {0: 3, 1: 2}),
        (PieceType.T, {0: 3, 1: 2, 2: 3, 3: 2}),
        (PieceType.J, {0: 3, 1: 2, 2: 3, 3: 2}),
        (PieceType.L, {0: 3, 1: 2, 2: 3, 3: 2}),
    ],
)
def test_valid_actions_cover_every_orientation_of_every_piece(piece_type, widths):
    env = TetrisEnv(seed=8)
    env.reset()
    force_piece(env, piece_type)

    actual = {}
    for rotation, column in env.get_valid_actions():
        actual.setdefault(rotation, []).append(column)

    assert set(actual) == set(widths)
    for rotation, width in widths.items():
        assert actual[rotation] == list(range(WIDTH - width + 1))


@pytest.mark.parametrize("piece_type", list(PieceType))
def test_every_valid_action_lands_where_it_says_it_will(piece_type):
    env = TetrisEnv(seed=0)
    env.reset()
    force_piece(env, piece_type)
    actions = env.get_valid_actions()
    assert actions

    for action in actions:
        env.reset()
        force_piece(env, piece_type)
        before = occupancy(env.game.board.as_list())

        state, reward, done, info = env.step(action)
        landed = placed_cells(before, state.board)

        assert len(landed) == 4, f"{action} should lock exactly four cells"
        assert not done
        assert landed[:, 1].min() == action.column, f"{action} moved the piece sideways"
        width = _orientation_for(piece_type, action.rotation).width
        assert landed[:, 1].max() - landed[:, 1].min() + 1 == width


def test_valid_actions_are_empty_once_the_game_is_over():
    env = TetrisEnv(seed=4)
    top_out(env)
    assert env.get_valid_actions() == []


def test_valid_actions_reflect_the_board_only_through_the_piece():
    # A placement is legitimate even in a bad column: the environment filters
    # what is impossible, not what is unwise.
    env = TetrisEnv(seed=4)
    env.reset()
    force_piece(env, PieceType.O)
    on_empty_board = env.get_valid_actions()

    fill(env, bottom_row(range(WIDTH)))     # a full bottom row waiting to clear
    assert env.get_valid_actions() == on_empty_board


# -------------------------------------------------------------------- state


def test_state_arrays_have_the_documented_shapes_and_dtypes():
    state = TetrisEnv(seed=2).reset()

    expected = {
        "board": ((HEIGHT, WIDTH), np.float32),
        "board_with_piece": ((HEIGHT, WIDTH), np.float32),
        "current_piece": ((4, 2), np.int8),
        "current_piece_type": ((N_PIECE_TYPES,), np.float32),
        "current_rotation": ((4,), np.float32),
        "next_piece_type": ((N_PIECE_TYPES,), np.float32),
        "next_piece": ((4, 2), np.int8),
    }
    for field, (shape, dtype) in expected.items():
        array = getattr(state, field)
        assert array.shape == shape, f"{field} has shape {array.shape}"
        assert array.dtype == dtype, f"{field} has dtype {array.dtype}"

    assert state.queue.ndim == 1 and state.queue.dtype == np.int8
    assert isinstance(state.current_column, int)
    assert isinstance(state.game_over, bool)


def test_state_is_binary_occupancy_for_the_board_arrays():
    env = TetrisEnv(seed=2)
    env.reset()
    env.step((0, 0))
    state = env.get_state()

    assert set(np.unique(state.board)) <= {0.0, 1.0}
    assert set(np.unique(state.board_with_piece)) <= {0.0, 1.0}


def test_the_state_board_is_the_engine_board():
    env = TetrisEnv(seed=2)
    env.reset()
    env.step((0, 4))
    state = env.get_state()
    assert np.array_equal(state.board, occupancy(env.game.board.as_list()))


def test_the_state_shows_the_piece_that_is_about_to_move():
    env = TetrisEnv(seed=2)
    state = env.reset()

    assert state.current_piece_name is not None
    assert int(state.current_piece_type.argmax()) + 1 == int(env.game.current.type)
    assert state.current_column == state.current_piece[:, 1].min()
    assert state.board.sum() + 4 == state.board_with_piece.sum()
    assert state.current_rotation.sum() == 1
    assert int(state.current_rotation.argmax()) == env.game.current.rotation


def test_the_state_goes_quiet_when_the_game_is_over():
    env = TetrisEnv(seed=4)
    top_out(env)
    state = env.get_state()

    assert state.game_over
    assert state.current_piece_name is None
    assert state.current_column == -1
    assert state.current_piece.sum() == 0
    assert state.current_piece_type.sum() == 0
    assert state.current_rotation.sum() == 0
    assert state.board_with_piece.sum() == state.board.sum()


def test_feature_vector_is_flat_fixed_length_and_finite():
    env = TetrisEnv(seed=2)
    env.reset()
    vector = env.get_state().feature_vector()

    assert vector.shape == (FEATURE_VECTOR_SIZE,)
    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()

    env.step((0, 0))
    assert env.get_state().feature_vector().shape == (FEATURE_VECTOR_SIZE,)


def test_describe_names_every_field_with_its_shape():
    text = TetrisEnv(seed=2).reset().describe()
    for field in ("board", "current_piece", "next_piece_type", "queue", "game_over"):
        assert field in text
    assert f"({HEIGHT}x{WIDTH})" in text


# ------------------------------------------------------------- determinism


def test_two_environments_with_the_same_seed_deal_the_same_pieces():
    a = TetrisEnv(seed=42)
    b = TetrisEnv(seed=42)
    assert np.array_equal(a.reset().queue, b.reset().queue)

    for _ in range(30):
        state_a, _, _, _ = a.step((0, 2))
        state_b, _, _, _ = b.step((0, 2))
        assert np.array_equal(state_a.board, state_b.board)
        assert np.array_equal(state_a.queue, state_b.queue)
        assert state_a.current_piece_name == state_b.current_piece_name


def test_a_seeded_environment_replays_the_same_episode_after_reset():
    env = TetrisEnv(seed=13)
    first_queue = env.reset().queue.copy()
    for _ in range(5):
        env.step((0, 1))
    assert np.array_equal(env.reset().queue, first_queue)


def test_different_seeds_deal_different_pieces():
    assert not np.array_equal(TetrisEnv(seed=1).reset().queue, TetrisEnv(seed=2).reset().queue)


def test_an_unseeded_environment_is_a_fresh_episode_each_time():
    a = TetrisEnv()
    b = TetrisEnv()
    assert not np.array_equal(a.reset().queue, b.reset().queue)


def test_a_seeded_policy_run_is_reproducible_episode_to_episode():
    def episode(seed: int):
        env = TetrisEnv(seed=seed)
        env.reset()
        return play(env, random.Random(0), max_pieces=60)[:2]

    assert episode(21) == episode(21)


# ------------------------------------------------------------------- reward


def test_reward_is_a_float_and_matches_the_configured_formula():
    env = TetrisEnv(seed=6)
    env.reset()
    force_piece(env, PieceType.O)

    _, reward, _, info = env.step((0, 4))       # flat landing on an empty board
    expected = RewardConfig().height * 2        # the stack grew by two rows
    assert reward == pytest.approx(expected)
    assert info["height_added"] == 2
    assert info["holes_created"] == 0


def test_clearing_a_line_pays_the_per_line_reward():
    env = TetrisEnv(seed=6)
    env.reset()
    fill(env, bottom_row(range(4, WIDTH)))      # columns 0..3 are left open for the I
    force_piece(env, PieceType.I)

    state, reward, done, info = env.step((0, 0))    # the I completes the bottom row

    assert info["lines_cleared"] == 1
    assert state.board.sum() == 0                   # the row cleared and nothing is left
    assert state.lines_cleared == 1
    assert reward == pytest.approx(RewardConfig().per_line)


def test_clearing_several_lines_pays_more_than_clearing_one():
    env = TetrisEnv(seed=6)
    env.reset()
    fill(env, bottom_row(range(1, WIDTH)))
    fill(env, [(HEIGHT - 2, col) for col in range(1, WIDTH)])
    force_piece(env, PieceType.I)

    _, reward, _, info = env.step((1, 0))           # standing the I up clears both

    assert info["lines_cleared"] == 2
    assert reward > RewardConfig().per_line


def test_creating_a_hole_is_penalised():
    def reward_for(column: int):
        env = TetrisEnv(seed=6)
        env.reset()
        fill(env, [(HEIGHT - 1, 0), (HEIGHT - 1, 2)])
        force_piece(env, PieceType.O)
        _, reward, _, info = env.step((0, column))
        return reward, info

    holed, holed_info = reward_for(0)               # the O buries the gap at column 1
    clean, clean_info = reward_for(4)               # the same piece, no hole

    assert holed_info["holes_created"] == 1
    assert clean_info["holes_created"] == 0
    assert holed < clean
    # One new hole plus the two rows the O added on top of the single-row stack.
    assert holed == pytest.approx(RewardConfig().hole + 2 * RewardConfig().height)


def test_a_taller_placement_is_penalised_more_than_a_flat_one():
    def reward_for(rotation: int) -> float:
        env = TetrisEnv(seed=6)
        env.reset()
        force_piece(env, PieceType.I)
        _, reward, _, _ = env.step((rotation, 4))
        return reward

    assert reward_for(1) < reward_for(0)            # standing an I up grows the stack more


def test_game_over_costs_a_lot():
    config = RewardConfig(per_line=0.0, hole=0.0, height=0.0)
    env = TetrisEnv(seed=4, reward=config)
    env.reset()
    fill(env, [(row, col) for row in range(HEIGHT) for col in range(1, WIDTH)])
    force_piece(env, PieceType.O)

    state, reward, done, info = env.step((0, 0))

    assert done and info["game_over"]
    assert reward == pytest.approx(config.game_over)


def test_the_reward_function_can_be_replaced():
    env = TetrisEnv(seed=6, reward=lambda outcome: 1.0)
    env.reset()
    for action in [(0, 0), (0, 2), (0, 4)]:
        _, reward, _, _ = env.step(action)
        assert reward == 1.0


def test_reward_weights_are_the_knobs():
    env = TetrisEnv(seed=6, reward=RewardConfig(per_line=100.0, height=0.0, hole=0.0))
    env.reset()
    fill(env, bottom_row(range(4, WIDTH)))
    force_piece(env, PieceType.I)

    _, reward, _, _ = env.step((0, 0))
    assert reward == pytest.approx(100.0)


# ------------------------------------------------- the survival term (per_piece)


def test_the_survival_term_is_off_by_default():
    """`per_piece=0` means every reward this project produced before is unchanged."""
    assert RewardConfig().per_piece == 0.0
    env = TetrisEnv(seed=6)                       # the default reward, unmodified
    env.reset()
    plain = TetrisEnv(seed=6, reward=RewardConfig(per_piece=0.0))
    plain.reset()

    for action in [(0, 3), (0, 5), (0, 1)]:
        assert env.step(action)[1] == plain.step(action)[1]


def test_surviving_a_piece_pays_the_survival_reward():
    config = RewardConfig(per_line=0.0, hole=0.0, height=0.0, per_piece=2.0)
    env = TetrisEnv(seed=6, reward=config)
    env.reset()

    _, reward, done, info = env.step((0, 3))

    assert not done and not info["game_over"]
    assert reward == pytest.approx(2.0)


def test_the_piece_that_ends_the_game_is_not_paid_for():
    """Dying is never profitable through `per_piece`: it is not a survived piece."""
    config = RewardConfig(per_line=0.0, hole=0.0, height=0.0, game_over=-100.0,
                          per_piece=2.0)
    env = TetrisEnv(seed=4, reward=config)
    env.reset()
    fill(env, [(row, col) for row in range(HEIGHT) for col in range(1, WIDTH)])
    force_piece(env, PieceType.O)

    _, reward, done, info = env.step((0, 0))

    assert done and info["game_over"]
    assert reward == pytest.approx(-100.0)        # no +2 for the fatal piece


def test_the_default_reward_punishes_a_longer_life_and_the_survival_term_repairs_it():
    """The defect `per_piece` exists to fix, measured on one policy.

    40 random games per reward, the *same* action sequences (only the reward
    differs). With the default weights a game that lasts longer earns less --
    every per-step term is a penalty and the terminal one is paid whenever the
    game ends, so the cheapest episode is the shortest one. A positive
    `per_piece` flips that: the longer game is worth more.
    """

    def episodes(config: RewardConfig) -> list:
        rows = []
        for seed in range(40):
            env = TetrisEnv(seed=seed, reward=config)
            env.reset()
            _, total, _ = play(env, random.Random(seed), max_pieces=400)
            rows.append((env.pieces_placed, total))
        return rows

    def correlation(rows: list) -> float:
        pieces = np.array([count for count, _ in rows], dtype=float)
        rewards = np.array([total for _, total in rows], dtype=float)
        return float(np.corrcoef(pieces, rewards)[0, 1])

    assert correlation(episodes(RewardConfig())) < 0
    assert correlation(episodes(RewardConfig(per_piece=5.0))) > 0


def test_info_carries_the_documented_debugging_fields():
    env = TetrisEnv(seed=6)
    env.reset()
    _, reward, done, info = env.step((0, 3))

    assert {"lines_cleared", "score", "game_over"} <= set(info)
    assert {"holes", "height", "placement", "piece", "pieces_placed", "valid_actions"} <= set(info)
    assert info["reward"] == reward
    assert info["score"] == env.get_state().score
    assert info["valid_actions"] == len(env.get_valid_actions())


# --------------------------------------------------------------- termination


def test_a_game_eventually_ends_and_says_so():
    env = TetrisEnv(seed=4)
    env.reset()
    steps, total, info = play(env, random.Random(1), max_pieces=300)

    assert env.is_game_over()
    assert info["game_over"] is True
    assert steps <= 300, "a random policy should top out well before 300 pieces"
    assert env.get_state().game_over


def test_done_is_true_exactly_when_the_game_is_over():
    env = TetrisEnv(seed=4)
    env.reset()
    rng = random.Random(2)
    while not env.is_game_over():
        state, reward, done, info = env.step(rng.choice(env.get_valid_actions()))
        assert done == env.is_game_over() == state.game_over


def test_the_final_reward_is_negative():
    # Even a line-clearing final placement cannot outweigh the game-over penalty.
    env = TetrisEnv(seed=4)
    env.reset()
    rng = random.Random(3)
    last_reward = 0.0
    while not env.is_game_over():
        _, last_reward, done, _ = env.step(rng.choice(env.get_valid_actions()))
        if done:
            break
    assert last_reward < 0


def test_a_full_episode_is_reproducible_from_a_seed():
    def episode(seed: int) -> tuple:
        env = TetrisEnv(seed=seed)
        env.reset()
        return play(env, random.Random(7), max_pieces=100)

    assert episode(5) == episode(5)


# ------------------------------------------------------- no display required


def test_importing_the_headless_modules_does_not_import_pygame():
    modules = "tetris.env, tetris.agents, tetris.evaluate, tetris.heuristic, tetris.config, tetris.experiment"
    result = subprocess.run(
        [sys.executable, "-c", f"import sys, {modules}; print('pygame' in sys.modules)"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_an_episode_runs_without_a_display():
    # SDL is forced to its dummy driver: if anything on the environment's path
    # touched pygame or tried to open a window, this subprocess would fail.
    code = (
        "import os; os.environ['SDL_VIDEODRIVER'] = 'dummy'; "
        "from tetris.env import TetrisEnv; "
        "env = TetrisEnv(seed=1); env.reset(); "
        "[env.step((0, i % 5)) for i in range(10)]; "
        "print(env.pieces_placed, env.game.score)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=str(PROJECT_ROOT), capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr

    placed, score = result.stdout.split()
    assert placed == "10" and int(score) > 0
