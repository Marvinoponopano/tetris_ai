"""Tests for the lookahead search behind `HeuristicAgent(lookahead_depth=...)`.

What is checked here:

* the move lists the search generates for *future* pieces are the same lists the
  environment would hand out for them;
* depth 0 is exactly the old no-lookahead behaviour, and a deeper search never
  narrows down the decision it is taking;
* the tree has the shape and the cost the docs claim (nodes, beam);
* the search refuses moves that end the game -- a lock-out, or a board where the
  next piece cannot even appear -- and *that* is why looking ahead changes moves;
* the death test agrees with the real engine.
"""

import random
from dataclasses import replace

import numpy as np
import pytest

from tetris.agents import HeuristicAgent, upcoming_pieces
from tetris.board import HEIGHT, WIDTH
from tetris.env import TetrisEnv, placements_for
from tetris.heuristic import (
    DEAD,
    DEFAULT_BEAM,
    HeuristicWeights,
    best_placement,
    search,
    simulate_placement,
    spawn_blocked,
)
from tetris.metrics import empty_board, occupancy
from tetris.pieces import Piece, PieceType

# --------------------------------------------------------------------- boards


def skyline(*heights) -> np.ndarray:
    """A board whose columns are filled from the floor up to the given heights."""
    grid = empty_board()
    for col, height in enumerate(heights):
        if height:
            grid[max(0, HEIGHT - height):, col] = 1.0
    return grid


#: A board where the middle is a deep well and the two towers beside it reach the
#: ceiling -- 17 rows of headroom in columns 4-5, nothing anywhere else:
#:
#:     ..##..##..
#:     ..##..##..          (three rows of tower above the well)
#:     ..######..
#:     ...  17 rows ...
#:     ##########
MIDDLE_WELL = skyline(1, 1, 20, 20, 17, 17, 20, 20, 1, 1)

#: A small hill with a flat left shoulder -- the position in the "changed move"
#: test below:
#:
#:     ..........
#:     .....#....
#:     ...###....
#:     ..#####...
HILL = skyline(0, 0, 1, 2, 3, 2, 1, 0, 0, 0)

#: Two towers, 19 and 18 rows high, with a 12-row gap between them and low
#: ground on both sides. A reachable position: a Z really can be dealt here.
#:
#:     ...#..#...
#:     ..##..##..
#:     ..##..##..        (ten rows of the same)
#:     ..######..
#:     ##########
TOWERS = skyline(1, 1, 18, 19, 12, 12, 19, 18, 1, 1)


def env_ready(cells=(), piece_type: PieceType = PieceType.T, seed: int = 0) -> TetrisEnv:
    """An environment with `cells` on the board and `piece_type` in play."""
    env = TetrisEnv(seed=seed)
    env.reset()
    if len(cells):
        env.game.board.place(list(cells), int(PieceType.J))
    env.game.current = Piece(piece_type, row=0, col=4)
    return env


def env_on(board, piece_type: PieceType = PieceType.T, seed: int = 0) -> TetrisEnv:
    cells = [(int(r), int(c)) for r, c in np.argwhere(np.asarray(board))]
    return env_ready(cells, piece_type, seed)


# ------------------------------------------------- the moves of future pieces


@pytest.mark.parametrize("piece_type", list(PieceType))
def test_placements_for_is_what_the_environment_would_offer(piece_type):
    """Future pieces must get the same move list the env gives the current one."""
    env = env_ready(piece_type=piece_type)
    assert list(placements_for(piece_type)) == env.get_valid_actions()


def test_placements_for_never_needs_an_environment():
    # The whole point: the search asks about pieces that are not on the board yet.
    assert len(placements_for(PieceType.O)) == WIDTH - 1
    assert len(placements_for(PieceType.I)) == 17
    assert all(isinstance(action, tuple) for action in placements_for(PieceType.T))


def test_upcoming_pieces_reads_the_current_piece_then_the_queue():
    env = env_ready(piece_type=PieceType.S, seed=5)
    state = env.get_state()
    queue = [int(piece) for piece in state.queue]

    assert upcoming_pieces(state, 0) == (PieceType.S,)
    assert upcoming_pieces(state, 1) == (PieceType.S, PieceType(queue[0]))
    assert upcoming_pieces(state, 2) == (
        PieceType.S, PieceType(queue[0]), PieceType(queue[1])
    )
    # queue[0] is the next piece: the state says so twice.
    assert PieceType(queue[0]) == PieceType(int(np.argmax(state.next_piece_type)) + 1)


def test_upcoming_pieces_stops_at_the_end_of_the_queue():
    env = env_ready(piece_type=PieceType.Z, seed=6)
    state = replace(env.get_state(), queue=np.array([int(PieceType.L)], dtype=np.int8))
    assert upcoming_pieces(state, 3) == (PieceType.Z, PieceType.L)


# ------------------------------------------- depth 0 is the old no-lookahead agent


def test_depth_zero_is_the_plain_best_placement():
    actions = env_on(HILL, PieceType.T).get_valid_actions()
    expected_action, expected_scores = best_placement(
        HILL, PieceType.T, actions, HeuristicWeights()
    )

    result = search(HILL, (PieceType.T,), depth=0, actions=actions)

    assert result.action == expected_action
    assert list(result.scores) == expected_scores
    assert result.nodes == len(actions)          # one simulation per move, no more


def test_depth_zero_ignores_the_pieces_that_follow():
    actions = env_on(HILL, PieceType.T).get_valid_actions()
    alone = search(HILL, (PieceType.T,), depth=0, actions=actions)
    with_next = search(HILL, (PieceType.T, PieceType.I, PieceType.O), depth=0,
                       actions=actions)
    assert list(alone.scores) == list(with_next.scores)


# ------------------------------------------------------------- tree and cost


def test_depth_one_is_the_exhaustive_product():
    # An empty board: 9 placements for an O at every step, so 9 + 9*9 simulations.
    result = search(empty_board(), (PieceType.O, PieceType.O), depth=1)
    assert result.nodes == 9 + 9 * 9


def test_depth_two_is_bounded_by_the_beam():
    # 9 * (9 + beam * 9) + 9: the middle ply is where the beam bites.
    for beam in (1, 2, 4):
        result = search(empty_board(), (PieceType.O, PieceType.O, PieceType.O),
                        depth=2, beam=beam)
        assert result.nodes == 9 + 9 * (9 + beam * 9)


def test_a_wider_beam_explores_at_least_as_much():
    sizes = [
        search(HILL, (PieceType.T, PieceType.I, PieceType.O), depth=2, beam=beam).nodes
        for beam in (1, 2, 3, 6)
    ]
    assert sizes == sorted(sizes)
    assert sizes[0] < sizes[-1]


def test_every_current_move_is_scored_however_deep_the_search_goes():
    actions = env_on(HILL, PieceType.L).get_valid_actions()
    for depth in (0, 1, 2):
        result = search(HILL, (PieceType.L, PieceType.S, PieceType.O), depth=depth,
                        actions=actions)
        assert [action for action, _score in result.scores] == list(actions)
        assert len(result.scores) == len(actions)


def test_lookahead_does_not_cut_the_move_it_is_taking():
    # The beam is a limit on the *continuations*, never on the current decision:
    # with a beam of 1 the move lists scored at the root are still complete.
    actions = env_on(HILL, PieceType.T).get_valid_actions()
    narrow = search(HILL, (PieceType.T, PieceType.I), depth=1, beam=1, actions=actions)
    wide = search(HILL, (PieceType.T, PieceType.I), depth=1, beam=99, actions=actions)
    assert list(narrow.scores) == list(wide.scores)   # depth 1 is exhaustive anyway

    narrow2 = search(HILL, (PieceType.T, PieceType.I, PieceType.O), depth=2, beam=1,
                     actions=actions)
    assert len(narrow2.scores) == len(actions)


def test_the_search_is_deterministic_and_leaves_the_board_alone():
    before = HILL.copy()
    first = search(HILL, (PieceType.T, PieceType.I, PieceType.O), depth=2, beam=3)
    second = search(HILL, (PieceType.T, PieceType.I, PieceType.O), depth=2, beam=3)

    assert first.action == second.action
    assert list(first.scores) == list(second.scores)
    assert np.array_equal(HILL, before), "the search wrote to the board it was given"


def test_depth_and_beam_are_validated():
    with pytest.raises(ValueError, match="depth must be 0 or more"):
        search(HILL, (PieceType.T,), depth=-1)
    with pytest.raises(ValueError, match="beam must be at least 1"):
        search(HILL, (PieceType.T,), depth=1, beam=0)
    with pytest.raises(ValueError, match="at least the current piece"):
        search(HILL, ())


# ------------------------------------------------ looking ahead changes the move


def test_lookahead_changes_the_move_on_a_hand_built_board():
    """The T goes left instead of right, because the I that follows prefers it."""
    actions = env_on(HILL, PieceType.T).get_valid_actions()

    greedy = search(HILL, (PieceType.T, PieceType.I), depth=0, actions=actions)
    deliberate = search(HILL, (PieceType.T, PieceType.I), depth=1, actions=actions)

    assert greedy.action == (0, 7)              # flat on the right shoulder
    assert deliberate.action == (2, 0)          # upside down on the left, for the I
    scores = dict(deliberate.scores)
    assert scores[deliberate.action] > scores[greedy.action]
    # and the greedy move's immediate score was genuinely the best one
    assert greedy.best_score == max(score for _action, score in greedy.scores)


def test_lookahead_refuses_a_move_that_kills_the_next_piece():
    """The pipeline in one position: the greedy move seals the gap, the S dies."""
    actions = placements_for(PieceType.Z)
    greedy_action = (1, 3)              # an upright Z in the left tower's shadow

    greedy = search(TOWERS, (PieceType.Z, PieceType.S), depth=0, actions=actions)
    deliberate = search(TOWERS, (PieceType.Z, PieceType.S), depth=1, actions=actions)

    assert greedy.action == greedy_action
    assert deliberate.action == (1, 6)   # into the gap between the towers
    assert dict(deliberate.scores)[greedy_action] == DEAD, (
        "the greedy move leaves no room for the S, so it must score dead"
    )
    # Dead lines are cut, not explored: 17 moves, minus the 5 fatal ones that
    # were abandoned, times the 17 moves of the piece that follows.
    assert deliberate.nodes == 17 + 12 * 17

    # And the move it plays instead really does leave the S somewhere to appear.
    after = simulate_placement(TOWERS, PieceType.Z, *deliberate.action)
    assert not spawn_blocked(after.board, PieceType.S)


def test_dead_moves_are_reported_but_never_chosen():
    # Column 0-4 are filled to the ceiling: an O placed there locks out (DEAD),
    # the same O placed on the right is fine. The search takes the right.
    board = skyline(20, 20, 20, 20, 20, 0, 0, 0, 0, 0)
    result = search(board, (PieceType.O,), depth=0)

    dead = [action for action, score in result.scores if score == DEAD]
    assert dead, "this board must have lock-out placements"
    assert result.action not in dead
    assert result.best_score > DEAD


def test_a_board_with_no_playable_move_still_returns_something():
    # Everything is full to the ceiling: every move locks out, the game is over
    # whatever we do -- the search says so and still picks an action.
    board = skyline(*([20] * WIDTH))
    result = search(board, (PieceType.O,), depth=0)
    assert all(score == DEAD for _action, score in result.scores)
    assert result.action is not None


def test_a_dead_continuation_rubs_off_on_the_move_that_leads_to_it():
    # Same board, but now the search must plan the next piece too: the only
    # placements available are in the left tower's shadow, so the next piece dies.
    board = skyline(0, 2, 2, 2, 2, 2, 2, 2, 2, 2)
    result = search(board, (PieceType.I, PieceType.O), depth=1)
    for action, score in result.scores:
        if simulate_placement(board, PieceType.I, action.rotation, action.column).topped_out:
            assert score == DEAD
        else:
            assert score > DEAD


# ------------------------------------------------------ the death model itself


def test_spawn_blocked_is_false_on_an_empty_board():
    for piece_type in PieceType:
        assert not spawn_blocked(empty_board(), piece_type)


def test_spawn_blocked_sees_a_tower_in_the_middle():
    # An O always arrives centred (columns 4-5), so that pair being full is fatal
    # even with both side columns wide open.
    assert spawn_blocked(skyline(0, 0, 0, 0, 20, 20, 0, 0, 0, 0), PieceType.O)
    # Every piece spawns across columns 3-6, an O only across 4-5: a tower at
    # column 3 is fatal for the I and fine for the O.
    assert spawn_blocked(skyline(0, 0, 0, 20, 0, 0, 0, 0, 0, 0), PieceType.I)
    assert not spawn_blocked(skyline(0, 0, 0, 20, 0, 0, 0, 0, 0, 0), PieceType.O)


@pytest.mark.parametrize("piece_type", list(PieceType))
def test_spawn_blocked_agrees_with_the_engine(piece_type):
    """The real check: play a move, then ask the engine if it ended the game.

    The search decides whether a line of play continues by asking this predicate,
    so it has to agree with `Game._spawn_piece` -- including the cases where the
    sides are open and only the middle is piled up.
    """
    boards = [empty_board(), HILL, MIDDLE_WELL,
              skyline(4, 4, 4, 4, 6, 6, 4, 4, 4, 4), skyline(2, 5, 5, 5, 5, 5, 5, 5, 5, 2)]
    rng = random.Random(3)
    for _ in range(6):                      # plus random low piles
        grid = empty_board()
        for _ in range(rng.randrange(4, 30)):
            grid[rng.randrange(HEIGHT - 4, HEIGHT), rng.randrange(WIDTH)] = 1.0
        boards.append(grid)

    for board in boards:
        for current in PieceType:
            action = next(
                (candidate for candidate in placements_for(current)
                 if not simulate_placement(
                     board, current, candidate.rotation, candidate.column).topped_out),
                None,
            )
            if action is None:
                continue

            simulation = simulate_placement(board, current, action.rotation, action.column)
            env = env_on(board, current)
            env.game.queue.insert(0, piece_type)          # the piece arrives next
            _state, _reward, done, _info = env.step(action)

            assert bool(done) == spawn_blocked(simulation.board, piece_type), (
                f"{current.name} at {action} on a board, then {piece_type.name}: "
                f"engine says game over={done}, the search's model says "
                f"{spawn_blocked(simulation.board, piece_type)}"
            )


def test_an_impossible_placement_is_a_lock_out():
    # Columns 0-4 are filled to the ceiling: an O placed there settles entirely
    # above the board, so nothing is added -- just like the engine's lock-out.
    board = skyline(20, 20, 20, 20, 20, 0, 0, 0, 0, 0)
    simulation = simulate_placement(board, PieceType.O, 0, 0)
    assert simulation.topped_out
    assert simulation.cleared == 0
    assert simulation.board.sum() == board.sum()        # nothing was placed
    assert not simulate_placement(board, PieceType.O, 0, 8).topped_out


# ------------------------------------------------------------------ the agent


def test_the_agent_looks_as_far_ahead_as_it_was_told():
    env = env_ready(piece_type=PieceType.T, seed=7)
    state = env.get_state()
    for depth in (0, 1, 2):
        agent = HeuristicAgent(lookahead_depth=depth)
        agent.choose_action(state, env.get_valid_actions())
        assert agent.lookahead_depth == depth
        assert agent.last_decision.depth == depth
        assert len(agent.last_decision.pieces) == depth + 1


def test_the_agent_rejects_nonsense_configuration():
    with pytest.raises(ValueError, match="lookahead_depth must be 0 or more"):
        HeuristicAgent(lookahead_depth=-1)
    with pytest.raises(ValueError, match="beam must be at least 1"):
        HeuristicAgent(beam=0)


def test_the_agent_plays_the_move_the_search_found():
    env = env_ready(piece_type=PieceType.T, seed=8)
    state = env.get_state()
    valid = env.get_valid_actions()
    agent = HeuristicAgent(lookahead_depth=1)

    action = agent.choose_action(state, valid)

    assert action == agent.last_decision.action
    assert [move for move, _score in agent.last_scores] == list(valid)
    assert dict(agent.last_scores)[action] == agent.last_decision.best_score


def test_the_decision_can_be_printed_for_debugging(capsys):
    env = env_ready(piece_type=PieceType.T, seed=9)
    agent = HeuristicAgent(lookahead_depth=1, debug=True)
    agent.choose_action(env.get_state(), env.get_valid_actions())

    printed = capsys.readouterr().out
    assert "AI selected:" in printed
    assert "Rotation: " in printed
    assert "Column: " in printed
    assert "Evaluation: " in printed
    assert "depth 1 lookahead" in printed
    assert "runners-up:" in printed


def test_debug_is_silent_unless_asked_for(capsys):
    env = env_ready(piece_type=PieceType.T, seed=10)
    HeuristicAgent(lookahead_depth=1).choose_action(
        env.get_state(), env.get_valid_actions()
    )
    assert capsys.readouterr().out == ""


def test_the_decision_text_says_which_moves_end_the_game():
    result = search(skyline(20, 20, 20, 20, 20, 0, 0, 0, 0, 0), (PieceType.O,), depth=0)
    text = str(result)
    assert "AI selected:" in text
    assert "moves end the game" in text
    assert "next: none" in text                       # depth 0 knows no next piece


def test_the_decision_text_handles_a_position_with_no_moves():
    result = search(empty_board(), (PieceType.O,), depth=0, actions=[])
    assert result.action is None
    assert "nothing" in str(result)


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_the_agent_plays_a_whole_legal_reproducible_game(depth):
    def play() -> tuple:
        env = TetrisEnv(seed=21)
        state = env.reset()
        agent = HeuristicAgent(lookahead_depth=depth)
        for _ in range(25):
            if env.is_game_over():
                break
            valid = env.get_valid_actions()
            action = agent.choose_action(state, valid)
            assert action in valid
            state, _reward, _done, _info = env.step(action)
        return env.pieces_placed, state.score, state.lines_cleared

    first = play()
    assert first[0] == 25                     # it survived the 25 pieces we asked for
    assert first == play()                    # and replayed them identically


@pytest.mark.parametrize("depth", [1, 2])
def test_looking_ahead_still_clears_lines(depth):
    env = TetrisEnv(seed=31)
    state = env.reset()
    agent = HeuristicAgent(lookahead_depth=depth)
    for _ in range(120):
        if env.is_game_over():
            break
        state, _reward, _done, _info = env.step(
            agent.choose_action(state, env.get_valid_actions())
        )
    assert state.lines_cleared > 20
    assert env.pieces_placed >= 100


def test_the_agent_can_be_reconfigured_by_hand():
    agent = HeuristicAgent(lookahead_depth=1)
    agent.lookahead_depth = 2                 # the knobs are plain attributes
    env = env_ready(piece_type=PieceType.O, seed=11)
    agent.choose_action(env.get_state(), env.get_valid_actions())
    assert agent.last_decision.depth == 2
