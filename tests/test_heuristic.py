"""Tests for the heuristic layer: board features, simulation and scoring.

Three things are checked here:

* every feature function on its own, on boards small enough to reason about;
* `simulate_placement` against the *real engine* -- a simulated placement must
  leave exactly the board an actual one does, or the agent is planning on a
  board that does not exist;
* the weights: the score is the weighted sum, and changing a weight changes
  the decision.
"""

import random

import numpy as np
import pytest

from tetris.board import HEIGHT, WIDTH, Board
from tetris.env import TetrisEnv, normalized_cells, orientations
from tetris.heuristic import (
    WEIGHT_NAMES,
    HeuristicWeights,
    best_placement,
    clear_lines,
    evaluate_board,
    feature_rows,
    simulate_placement,
)
from tetris.metrics import (
    FEATURE_NAMES,
    BoardFeatures,
    aggregate_height,
    board_height,
    bumpiness,
    column_heights,
    completed_lines,
    count_holes,
    empty_board,
    extract_features,
    max_column_height,
    occupancy,
    top_rows,
    well_depths,
    wells,
)
from tetris.pieces import Piece, PieceType


# ------------------------------------------------------------------- helpers


def from_heights(heights) -> np.ndarray:
    """A board whose columns are filled from the floor up to the given heights."""
    grid = empty_board()
    for col, height in enumerate(heights):
        if height:
            grid[HEIGHT - height:, col] = 1.0
    return grid


def with_cells(cells, base=None) -> np.ndarray:
    grid = empty_board() if base is None else base.copy()
    for row, col in cells:
        grid[row, col] = 1.0
    return grid


def random_cells(rng: random.Random, blocks: int = 25, tall: int = 8) -> list:
    """A random low pile of blocks: realistic enough for pieces to land on."""
    cells = set()
    for _ in range(blocks):
        cells.add((rng.randrange(HEIGHT - tall, HEIGHT), rng.randrange(WIDTH)))
    return sorted(cells)


def env_ready_for(cells=(), piece_type: PieceType = PieceType.O, seed: int = 0) -> TetrisEnv:
    """An environment whose board holds `cells` and whose piece is `piece_type`."""
    env = TetrisEnv(seed=seed)
    env.reset()
    if len(cells):
        env.game.board.place(list(cells), int(PieceType.J))
    env.game.current = Piece(piece_type, row=0, col=4)
    return env


def env_on(grid, piece_type: PieceType = PieceType.O, seed: int = 0) -> TetrisEnv:
    """The same, for a board given as a grid rather than a list of cells."""
    cells = [(int(row), int(col)) for row, col in np.argwhere(np.asarray(grid))]
    return env_ready_for(cells, piece_type, seed)


# ------------------------------------------------------- shared piece geometry


def test_normalized_cells_start_at_the_origin():
    """The helper the environment and the agent share: bounding box at (0, 0)."""
    for piece_type in PieceType:
        for orientation in orientations(piece_type):
            cells = normalized_cells(piece_type, orientation.rotation)
            assert min(row for row, _ in cells) == 0
            assert min(col for _, col in cells) == 0
            assert len(cells) == 4


# ------------------------------------------------------------ column heights


def test_an_empty_board_has_no_height():
    assert list(column_heights(empty_board())) == [0] * WIDTH
    assert aggregate_height(empty_board()) == 0
    assert max_column_height(empty_board()) == 0


def test_column_heights_measure_from_the_highest_block_to_the_floor():
    heights = [1, 3, 0, 5, 0, 0, 2, 0, 0, 0]
    grid = from_heights(heights)
    assert list(column_heights(grid)) == heights


def test_a_floating_block_still_counts_as_height():
    # A block on row 5 with nothing under it is 15 rows of "height": it is in the way.
    grid = with_cells([(5, 2)])
    assert column_heights(grid)[2] == 15
    assert max_column_height(grid) == 15


def test_aggregate_height_is_the_sum_of_the_column_heights():
    grid = from_heights([1, 2, 3, 4, 0, 0, 0, 0, 0, 0])
    assert aggregate_height(grid) == sum(column_heights(grid)) == 10


def test_max_column_height_is_the_tallest_column():
    grid = from_heights([1, 9, 3, 0, 0, 0, 0, 0, 0, 0])
    assert max_column_height(grid) == 9
    assert board_height(grid) == 9          # the reward's name for the same thing


# --------------------------------------------------------------- bumpiness


def test_a_flat_surface_has_no_bumpiness():
    assert bumpiness(empty_board()) == 0
    assert bumpiness(from_heights([3, 3, 3, 3, 3, 3, 3, 3, 3, 3])) == 0


def test_bumpiness_sums_the_steps_between_neighbours():
    #    1 -> 3 (+2) -> 3 (0) -> 0 (-3) -> ... -> 0
    assert bumpiness(from_heights([1, 3, 3, 0, 0, 0, 0, 0, 0, 0])) == 5


def test_bumpiness_notices_a_single_spike():
    assert bumpiness(from_heights([0, 0, 4, 0, 0, 0, 0, 0, 0, 0])) == 8


# ---------------------------------------------------------- completed lines


def test_a_board_with_no_full_rows_has_no_completed_lines():
    assert completed_lines(empty_board()) == 0
    grid = from_heights([1] * WIDTH)
    grid[HEIGHT - 1, 3] = 0.0
    assert completed_lines(grid) == 0


def test_a_flat_row_is_a_completed_line():
    assert completed_lines(from_heights([1] * WIDTH)) == 1


def test_completed_lines_counts_every_full_row():
    grid = from_heights([1] * WIDTH)
    grid[HEIGHT - 2, :] = 1.0
    assert completed_lines(grid) == 2


def test_a_row_with_one_gap_is_not_completed():
    grid = from_heights([1] * WIDTH)
    grid[HEIGHT - 1, 4] = 0.0
    assert completed_lines(grid) == 0
    assert count_holes(grid) == 0        # and it is not a hole either: still open above


# -------------------------------------------------------------------- wells


def test_a_flat_surface_has_no_wells():
    assert list(well_depths(from_heights([2] * WIDTH))) == [0] * WIDTH
    assert wells(from_heights([2] * WIDTH)) == 0


def test_an_interior_gap_is_a_well():
    grid = from_heights([4, 0, 4, 0, 0, 0, 0, 0, 0, 0])
    assert well_depths(grid)[1] == 4
    assert wells(grid) == 4


def test_a_gap_against_the_wall_is_a_well():
    # The wall is a solid side, so this is the slot an I piece drops into.
    grid = from_heights([0, 4, 4, 0, 0, 0, 0, 0, 0, 0])
    assert well_depths(grid)[0] == 4
    assert wells(grid) == 4


def test_a_shallow_step_is_a_shallow_well():
    grid = from_heights([3, 2, 3, 0, 0, 0, 0, 0, 0, 0])
    assert well_depths(grid)[1] == 1
    assert wells(grid) == 1


def test_a_wide_gap_between_two_towers_is_not_a_well():
    # A well is a one-column slot. In a gap several columns wide the middle
    # columns have an empty neighbour, so nothing is deeper than *both* sides.
    grid = from_heights([4, 0, 0, 0, 4, 0, 0, 0, 0, 0])
    assert list(well_depths(grid)[:5]) == [0, 0, 0, 0, 0]
    assert wells(grid) == 0


# ---------------------------------------------------------- top_rows helper


def test_top_rows_finds_the_highest_filled_row_of_every_column():
    grid = from_heights([0, 2, 0, 1, 0, 0, 0, 0, 0, 0])
    tops = top_rows(grid)
    assert tops[1] == HEIGHT - 2
    assert tops[3] == HEIGHT - 1
    assert tops[0] == HEIGHT          # empty column: "nothing in the way"


# ------------------------------------------------------------ extract_features


def test_extract_features_matches_the_individual_functions():
    rng = random.Random(17)
    for _ in range(25):
        grid = with_cells(random_cells(rng, blocks=rng.randrange(5, 40)))
        features = extract_features(grid)
        assert features.aggregate_height == aggregate_height(grid)
        assert features.max_height == max_column_height(grid)
        assert features.holes == count_holes(grid)
        assert features.bumpiness == bumpiness(grid)
        assert features.completed_lines == completed_lines(grid)
        assert features.wells == wells(grid)


def test_extract_features_accepts_the_engine_board_format():
    env = env_ready_for([(HEIGHT - 1, 0), (HEIGHT - 2, 0), (HEIGHT - 1, 4)], PieceType.O)
    from_engine = extract_features(env.game.board.as_list())
    from_array = extract_features(occupancy(env.game.board.as_list()))
    assert from_engine == from_array
    assert from_engine.aggregate_height == 3
    assert from_engine.max_height == 2


def test_extract_features_names_match_the_documented_list():
    features = extract_features(from_heights([1, 2, 3, 0, 0, 0, 0, 0, 0, 0]))
    assert tuple(features.as_dict()) == FEATURE_NAMES
    assert set(features.as_dict()) == {
        "aggregate_height",
        "max_height",
        "holes",
        "bumpiness",
        "completed_lines",
        "wells",
    }
    assert all(isinstance(value, int) for value in features.as_dict().values())
    assert "aggregate_height=" in repr(features)


def test_features_do_not_mutate_the_board():
    grid = from_heights([2, 0, 3, 0, 0, 0, 0, 0, 0, 0])
    before = grid.copy()
    extract_features(grid)
    assert np.array_equal(grid, before)
    assert isinstance(extract_features(grid), BoardFeatures)


# ---------------------------------------------------------------- simulation


def test_simulating_a_placement_leaves_the_board_untouched():
    grid = from_heights([2, 2, 0, 0, 0, 0, 0, 0, 0, 0])
    before = grid.copy()

    simulation = simulate_placement(grid, PieceType.O, 0, 0)

    assert np.array_equal(grid, before), "the board we simulated on was modified"
    assert simulation.board is not grid
    assert simulation.cleared == 0


def test_an_o_piece_lands_on_the_floor():
    simulation = simulate_placement(empty_board(), PieceType.O, 0, 0)
    landed = simulation.board

    assert simulation.board.sum() == 4
    assert {(int(r), int(c)) for r, c in np.argwhere(landed)} == {
        (HEIGHT - 1, 0), (HEIGHT - 1, 1), (HEIGHT - 2, 0), (HEIGHT - 2, 1),
    }


def test_the_column_is_the_leftmost_cell_of_the_piece():
    # An I placed at column 3 covers columns 3, 4, 5, 6 -- not 0..3 or 4..7.
    simulation = simulate_placement(empty_board(), PieceType.I, 0, 3)
    assert {int(c) for _, c in np.argwhere(simulation.board)} == {3, 4, 5, 6}


def test_a_piece_rests_on_top_of_the_stack():
    grid = from_heights([3, 3, 0, 0, 0, 0, 0, 0, 0, 0])
    simulation = simulate_placement(grid, PieceType.O, 0, 0)
    assert list(column_heights(simulation.board))[:2] == [5, 5]


def test_a_piece_can_land_on_the_floor_beside_a_tower():
    grid = from_heights([6, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    simulation = simulate_placement(grid, PieceType.O, 0, 5)
    assert column_heights(simulation.board)[5] == 2


def test_a_completed_row_is_cleared_and_reported():
    grid = from_heights([1] * WIDTH)
    grid[HEIGHT - 1, 0:4] = 0.0                      # a four-wide gap for an I

    simulation = simulate_placement(grid, PieceType.I, 0, 0)

    assert simulation.cleared == 1
    assert simulation.board.sum() == 0               # the row went, nothing else left
    assert simulation.features().completed_lines == 0


def test_lines_clear_and_the_rest_of_the_stack_shifts_down():
    grid = from_heights([1] * WIDTH)
    grid[HEIGHT - 1, 0:4] = 0.0
    grid[HEIGHT - 2, 9] = 1.0                        # a lone block one row up

    simulation = simulate_placement(grid, PieceType.I, 0, 0)

    assert simulation.cleared == 1
    assert simulation.board[HEIGHT - 1, 9] == 1.0    # it fell one row
    assert simulation.board[HEIGHT - 2, 9] == 0.0


def test_two_rows_can_clear_at_once():
    grid = from_heights([1] * WIDTH)                 # a full bottom row
    grid[HEIGHT - 2, :] = 1.0                        # and a full row above it
    grid[HEIGHT - 1, 0] = 0.0                        # with a one-wide slot at the left
    grid[HEIGHT - 2, 0] = 0.0

    simulation = simulate_placement(grid, PieceType.I, 1, 0)   # standing up

    assert simulation.cleared == 2
    assert simulation.board.sum() == 2                          # the top half of the I


# ------------------------------------------------- simulation vs the engine


@pytest.mark.parametrize("piece_type", list(PieceType))
def test_a_simulated_placement_matches_the_real_engine(piece_type):
    """The heart of the agent's correctness: planning must predict reality.

    For a handful of random boards, every legal placement is simulated and then
    actually played in the environment. The resulting boards must be identical,
    cell for cell, and the line counts must agree.
    """
    rng = random.Random(4)
    for _ in range(3):
        cells = random_cells(rng)
        prototype = env_ready_for(cells, piece_type)
        board = occupancy(prototype.game.board.as_list())
        actions = prototype.get_valid_actions()
        assert actions

        for action in actions:
            real = env_ready_for(cells, piece_type)
            state, reward, done, info = real.step(action)

            simulated = simulate_placement(board, piece_type, action.rotation, action.column)
            engine_board = occupancy(real.game.board.as_list())

            assert simulated.cleared == info["lines_cleared"], f"{action}"
            assert np.array_equal(simulated.board, engine_board), f"{action}"


def test_a_piece_that_cannot_fit_above_the_board_behaves_like_the_engine():
    # A board filled to the top: the piece locks out above the well, as in the engine.
    cells = [(row, col) for row in range(HEIGHT) for col in range(1, WIDTH)]
    prototype = env_ready_for(cells, PieceType.O)
    board = occupancy(prototype.game.board.as_list())

    real = env_ready_for(cells, PieceType.O)
    state, reward, done, info = real.step((0, 1))
    simulated = simulate_placement(board, PieceType.O, 0, 1)

    assert done, "the engine should have ended the game"
    assert simulated.board.sum() == board.sum()          # nothing was added
    assert np.array_equal(simulated.board, occupancy(real.game.board.as_list()))


def test_clear_lines_matches_the_engine_board():
    rng = random.Random(5)
    for _ in range(25):
        grid = with_cells(random_cells(rng, blocks=rng.randrange(0, 60), tall=HEIGHT))
        if rng.random() < 0.5:                            # sometimes make a full row
            grid[rng.randrange(HEIGHT), :] = 1.0

        engine = Board()
        engine.grid = [[int(cell) for cell in row] for row in grid]
        expected = engine.clear_lines()

        result, cleared = clear_lines(grid.copy())

        assert cleared == expected
        assert np.array_equal(result, occupancy(engine.as_list()))


# ------------------------------------------------------------------ weights


def test_the_default_weights_reward_lines_and_punish_the_rest():
    weights = HeuristicWeights()
    assert weights.lines_cleared > 0
    assert weights.holes < 0
    assert weights.aggregate_height < 0
    assert weights.max_height < 0
    assert weights.bumpiness < 0
    assert weights.wells < 0


def test_weights_can_be_listed_as_a_dict():
    listed = HeuristicWeights().as_dict()
    assert set(listed) == set(WEIGHT_NAMES)
    assert set(listed) == {
        "lines_cleared", "holes", "aggregate_height", "max_height", "bumpiness", "wells"
    }
    assert listed["lines_cleared"] == HeuristicWeights().lines_cleared


def test_a_dict_of_overrides_keeps_the_other_defaults():
    weights = HeuristicWeights.from_dict({"holes": -99.0})
    assert weights.holes == -99.0
    assert weights.bumpiness == HeuristicWeights().bumpiness


def test_an_unknown_weight_is_an_error():
    with pytest.raises(ValueError, match="unknown weight"):
        HeuristicWeights.from_dict({"holse": -1.0})


def test_from_any_accepts_weights_a_dict_or_nothing():
    assert HeuristicWeights.from_any(None) == HeuristicWeights()
    assert HeuristicWeights.from_any({"wells": -3.0}).wells == -3.0
    custom = HeuristicWeights(holes=-1.0)
    assert HeuristicWeights.from_any(custom) is custom
    with pytest.raises(TypeError):
        HeuristicWeights.from_any("holes=-1")


def test_the_score_uses_the_completed_lines_the_board_shows_by_default():
    features = extract_features(from_heights([1] * WIDTH))     # one full row
    assert features.completed_lines == 1
    weights = HeuristicWeights()
    assert weights.score(features) == pytest.approx(
        weights.score(features, lines_cleared=features.completed_lines)
    )
    assert weights.score(features, lines_cleared=0) == pytest.approx(
        weights.score(features) - weights.lines_cleared
    )


def test_the_score_is_the_weighted_sum_of_the_features():
    features = BoardFeatures(
        aggregate_height=10, max_height=4, holes=2, bumpiness=3, completed_lines=1, wells=1
    )
    weights = HeuristicWeights(
        lines_cleared=5.0, holes=-7.0, aggregate_height=-0.5,
        max_height=-1.0, bumpiness=-0.25, wells=-2.0,
    )
    expected = (5.0 * 2 + -7.0 * 2 + -0.5 * 10 + -1.0 * 4 + -0.25 * 3 + -2.0 * 1)
    assert weights.score(features, lines_cleared=2) == pytest.approx(expected)
    # With no line count given it falls back to the board's own full rows.
    in_board = (5.0 * 1 + -7.0 * 2 + -0.5 * 10 + -1.0 * 4 + -0.25 * 3 + -2.0 * 1)
    assert weights.score(features) == pytest.approx(in_board)


def test_evaluate_board_is_the_features_times_the_weights():
    rng = random.Random(6)
    grid = with_cells(random_cells(rng))
    weights = HeuristicWeights(holes=-3.0, bumpiness=-2.0)

    features = extract_features(grid)
    expected = (
        weights.lines_cleared * features.completed_lines
        + weights.holes * features.holes
        + weights.aggregate_height * features.aggregate_height
        + weights.max_height * features.max_height
        + weights.bumpiness * features.bumpiness
        + weights.wells * features.wells
    )
    assert evaluate_board(grid, weights) == pytest.approx(expected)


def test_evaluate_board_accepts_a_dict_of_weights():
    grid = from_heights([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    assert evaluate_board(grid, {"holes": -1.0}) == evaluate_board(
        grid, HeuristicWeights(holes=-1.0)
    )


def test_evaluate_board_can_count_lines_that_are_already_gone():
    grid = from_heights([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    weights = HeuristicWeights(lines_cleared=9.0)
    assert evaluate_board(grid, weights, lines_cleared=3) - evaluate_board(grid, weights) == pytest.approx(
        27.0
    )


def test_a_clean_low_board_scores_higher_than_a_messy_one():
    clean = from_heights([2, 2, 2, 2, 2, 0, 0, 0, 0, 0])
    messy = with_cells([(HEIGHT - 1, c) for c in (0, 2)] + [(HEIGHT - 4, 0), (HEIGHT - 5, 1)])
    assert evaluate_board(clean) > evaluate_board(messy)

    flat = from_heights([3] * WIDTH)
    jagged = from_heights([0, 6, 0, 6, 0, 6, 0, 6, 0, 6])
    assert evaluate_board(flat) > evaluate_board(jagged)


# ------------------------------------------------------------- best_placement


def test_best_placement_returns_the_highest_scoring_action():
    env = env_on(from_heights([2, 3, 1, 0, 0, 0, 0, 0, 0, 0]), PieceType.T)
    actions = env.get_valid_actions()

    action, scored = best_placement(
        occupancy(env.game.board.as_list()), PieceType.T, actions
    )

    assert action in actions
    assert len(scored) == len(actions)
    assert action == max(scored, key=lambda pair: pair[1])[0]


def test_best_placement_keeps_the_first_action_when_scores_tie():
    # Every O placement on an empty board scores the same -> the order decides.
    actions = [(0, 0), (0, 8)]
    empty = empty_board()
    assert best_placement(empty, PieceType.O, actions)[0] == (0, 0)
    assert best_placement(empty, PieceType.O, list(reversed(actions)))[0] == (0, 8)


def test_clearing_lines_is_preferred_even_before_the_line_reward():
    # Clearing rows leaves a smaller, flatter board, and the height features
    # already like that: the line weight makes clearing *more* attractive, it is
    # not what makes it attractive in the first place.
    grid = from_heights([1] * WIDTH)               # a full bottom row
    grid[HEIGHT - 2, :] = 1.0                      # a full row above it
    grid[HEIGHT - 1, 0] = 0.0                      # a one-wide slot
    grid[HEIGHT - 2, 0] = 0.0
    actions = [(1, 0), (0, 1)]                     # upright I clears; flat I builds on top

    rows = feature_rows(grid, PieceType.I, actions, HeuristicWeights())
    cleared = {action: cleared for action, cleared, _features, _score in rows}

    assert cleared[(1, 0)] == 2 and cleared[(0, 1)] == 0
    assert best_placement(grid, PieceType.I, actions, HeuristicWeights(lines_cleared=0.0))[0] == (1, 0)
    assert best_placement(grid, PieceType.I, actions, HeuristicWeights())[0] == (1, 0)


def test_the_line_weight_makes_clearing_more_attractive():
    grid = from_heights([1] * WIDTH)
    grid[HEIGHT - 1, 0:4] = 0.0
    actions = [(0, 0), (0, 5)]

    rows = feature_rows(grid, PieceType.I, actions, HeuristicWeights())
    clearing, not_clearing = rows[0], rows[1]

    assert clearing[1] == 1 and not_clearing[1] == 0
    assert clearing[3] > not_clearing[3]


def test_absurd_weights_really_do_change_the_decision():
    # The weights are the whole policy: make holes *good* and the agent seeks
    # them out. (Proof that the choice follows the weights, not something else.)
    grid = from_heights([1] * WIDTH)
    grid[HEIGHT - 1, 5] = 0.0                      # a one-wide pit at column 5
    actions = [(0, 0), (0, 4)]                     # (0, 4) bridges the pit -> a hole
    sane = HeuristicWeights()
    hole_lover = HeuristicWeights(
        lines_cleared=0.0, holes=+1.0, aggregate_height=0.0,
        max_height=0.0, bumpiness=0.0, wells=0.0,
    )

    def score_by_action(weights):
        return {
            action: score
            for action, _cleared, _features, score in feature_rows(
                grid, PieceType.O, actions, weights
            )
        }

    sane_scores, hole_scores = score_by_action(sane), score_by_action(hole_lover)

    assert best_placement(grid, PieceType.O, actions, sane)[0] == (0, 0)
    assert best_placement(grid, PieceType.O, actions, hole_lover)[0] == (0, 4)
    assert sane_scores[(0, 0)] > sane_scores[(0, 4)]
    assert hole_scores[(0, 4)] > hole_scores[(0, 0)]


def test_feature_rows_lists_every_action_with_its_features():
    env = env_ready_for(piece_type=PieceType.O)
    actions = env.get_valid_actions()

    rows = feature_rows(occupancy(env.game.board.as_list()), PieceType.O, actions)

    assert [action for action, _cleared, _features, _score in rows] == list(actions)
    for action, cleared, features, score in rows:
        assert cleared == 0
        assert isinstance(features, BoardFeatures)
        assert score == pytest.approx(
            HeuristicWeights().score(features, cleared)
        )
