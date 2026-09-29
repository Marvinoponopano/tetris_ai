"""Tests for the agents (`tetris.agents`)."""

import pathlib
import random
import subprocess
import sys

import numpy as np
import pytest

from tetris.agents import Agent, HeuristicAgent, RandomAgent
from tetris.env import TetrisEnv
from tetris.heuristic import HeuristicWeights
from tetris.metrics import count_holes, extract_features, max_column_height, occupancy
from tetris.pieces import Piece, PieceType

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


def env_with(piece_type: PieceType = None) -> TetrisEnv:
    """A fresh environment, optionally with a chosen active piece."""
    env = TetrisEnv(seed=0)
    env.reset()
    if piece_type is not None:
        env.game.current = Piece(piece_type, row=0, col=4)
    return env


def draws(agent: RandomAgent, valid, times: int = 20):
    return [agent.choose_action(None, valid) for _ in range(times)]


# ----------------------------------------------------------------- interface


def test_a_random_agent_satisfies_the_agent_interface():
    assert isinstance(RandomAgent(), Agent)
    assert isinstance(RandomAgent(seed=1), RandomAgent)
    assert RandomAgent().name == "random"


def test_choose_action_returns_one_of_the_valid_actions():
    env = env_with()
    agent = RandomAgent(seed=3)
    valid = env.get_valid_actions()

    assert all(action in valid for action in draws(agent, valid, 50))


@pytest.mark.parametrize("piece_type", list(PieceType))
def test_every_piece_type_can_be_played(piece_type):
    env = env_with(piece_type)
    agent = RandomAgent(seed=4)
    valid = env.get_valid_actions()

    assert all(action in valid for action in draws(agent, valid, 20))


def test_a_single_valid_action_is_always_chosen():
    agent = RandomAgent(seed=5)
    only = [(0, 2)]
    assert draws(agent, only, 10) == only * 10


def test_choose_action_rejects_an_empty_action_list():
    with pytest.raises(ValueError, match="game is over"):
        RandomAgent(seed=0).choose_action(None, [])


# ------------------------------------------------------------------ randomness


def test_the_same_seed_gives_the_same_choices():
    env = env_with(PieceType.T)
    valid = env.get_valid_actions()
    first = draws(RandomAgent(seed=7), valid, 30)
    second = draws(RandomAgent(seed=7), valid, 30)
    assert first == second


def test_different_seeds_give_different_choices():
    env = env_with(PieceType.T)
    valid = env.get_valid_actions()
    assert draws(RandomAgent(seed=1), valid, 60) != draws(RandomAgent(seed=2), valid, 60)


def test_reset_restarts_the_random_stream():
    env = env_with(PieceType.T)
    valid = env.get_valid_actions()
    agent = RandomAgent(seed=11)

    before = draws(agent, valid, 10)
    agent.reset(seed=11)
    assert draws(agent, valid, 10) == before


def test_reset_without_a_seed_still_produces_valid_actions():
    env = env_with()
    valid = env.get_valid_actions()
    agent = RandomAgent(seed=12)

    agent.reset()
    assert all(action in valid for action in draws(agent, valid, 10))


def test_an_injected_rng_behaves_like_a_seed():
    env = env_with(PieceType.T)
    valid = env.get_valid_actions()
    assert draws(RandomAgent(seed=13), valid, 25) == draws(
        RandomAgent(rng=random.Random(13)), valid, 25
    )


def test_the_choice_is_spread_over_the_options():
    # Two options, 200 draws: both must show up (missing one is a 2^-199 fluke).
    valid = [(0, 0), (1, 5)]
    chosen = set(draws(RandomAgent(seed=14), valid, 200))
    assert chosen == set(valid)


# ---------------------------------------------------------------- whole games


def test_it_can_play_a_complete_game():
    env = TetrisEnv(seed=1)
    agent = RandomAgent(seed=2)
    state = env.reset()
    actions = 0

    while not env.is_game_over():
        valid = env.get_valid_actions()
        action = agent.choose_action(state, valid)
        assert action in valid
        state, reward, done, info = env.step(action)
        actions += 1

    assert actions == env.pieces_placed > 0
    assert state.game_over


def test_a_seeded_game_is_reproducible():
    def play() -> tuple:
        env = TetrisEnv(seed=99)
        agent = RandomAgent(seed=100)
        state = env.reset()
        while not env.is_game_over():
            action = agent.choose_action(state, env.get_valid_actions())
            state, reward, done, info = env.step(action)
        return env.pieces_placed, state.score, state.lines_cleared

    assert play() == play()


# ------------------------------------------------------- no display required


@pytest.mark.parametrize("module", ["tetris.agents", "tetris.evaluate", "tetris.heuristic"])
def test_the_agent_and_evaluation_paths_do_not_import_pygame(module):
    result = subprocess.run(
        [sys.executable, "-c", f"import sys, {module}; print('pygame' in sys.modules)"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


# ------------------------------------------------------------ HeuristicAgent


def play(agent, placements: int = 60, seed: int = 21):
    """Play up to `placements` moves; returns (pieces, score, lines) for the run."""
    env = TetrisEnv(seed=seed)
    state = env.reset()
    for _ in range(placements):
        if env.is_game_over():
            break
        state, _reward, _done, _info = env.step(
            agent.choose_action(state, env.get_valid_actions())
        )
    return env.pieces_placed, state.score, state.lines_cleared


def test_a_heuristic_agent_satisfies_the_agent_interface():
    assert isinstance(HeuristicAgent(), Agent)
    assert HeuristicAgent().name == "heuristic"


@pytest.mark.parametrize("piece_type", list(PieceType))
def test_the_heuristic_agent_only_plays_legal_placements(piece_type):
    env = env_with(piece_type)
    valid = env.get_valid_actions()

    assert HeuristicAgent().choose_action(env.get_state(), valid) in valid


def test_the_heuristic_agent_does_not_modify_the_state_it_is_given():
    env = env_with(PieceType.S)
    state = env.get_state()
    before = state.board.copy()

    HeuristicAgent().choose_action(state, env.get_valid_actions())

    assert np.array_equal(state.board, before)
    assert np.array_equal(state.board, occupancy(env.game.board.as_list()))


def test_the_heuristic_agent_honours_its_weights():
    agent = HeuristicAgent({"holes": -1e6, "aggregate_height": 0.0, "max_height": 0.0,
                            "bumpiness": 0.0, "wells": 0.0, "lines_cleared": 0.0})
    assert agent.weights.holes == -1e6

    env = env_with(PieceType.L)
    action = agent.choose_action(env.get_state(), env.get_valid_actions())
    assert action in env.get_valid_actions()


def test_it_records_the_score_of_every_candidate_placement():
    env = env_with(PieceType.T)
    valid = env.get_valid_actions()
    agent = HeuristicAgent()

    chosen = agent.choose_action(env.get_state(), valid)

    assert [action for action, _score in agent.last_scores] == list(valid)
    assert chosen == max(agent.last_scores, key=lambda pair: pair[1])[0]
    assert len(set(score for _action, score in agent.last_scores)) > 1


def test_ties_are_broken_by_the_order_of_the_valid_actions():
    # On an empty board every O placement scores the same, so the first wins.
    env = env_with(PieceType.O)
    valid = env.get_valid_actions()

    assert HeuristicAgent().choose_action(env.get_state(), valid) == valid[0]
    assert HeuristicAgent().choose_action(env.get_state(), valid) == valid[0]


def test_choose_action_rejects_an_empty_action_list_here_too():
    with pytest.raises(ValueError, match="game is over"):
        HeuristicAgent().choose_action(None, [])


def test_the_heuristic_choice_does_not_depend_on_a_random_seed():
    env = env_with(PieceType.J)
    state = env.get_state()
    valid = env.get_valid_actions()
    first = HeuristicAgent().choose_action(state, valid)
    for _ in range(5):
        assert HeuristicAgent().choose_action(state, valid) == first


def test_the_heuristic_agent_plays_the_same_game_twice_identically():
    assert play(HeuristicAgent()) == play(HeuristicAgent())


def test_it_clears_far_more_lines_than_random_on_the_same_pieces():
    random_pieces, _score, random_lines = play(RandomAgent(seed=21))
    heuristic_pieces, _score, heuristic_lines = play(HeuristicAgent())

    assert heuristic_lines > random_lines + 3, (random_lines, heuristic_lines)
    assert heuristic_pieces > random_pieces


def test_a_height_loving_agent_builds_a_tower():
    # The weights *are* the policy, so an agent told to like tall stacks builds one.
    greedy = HeuristicAgent(
        {"lines_cleared": 0.0, "holes": 0.0, "aggregate_height": 1.0,
         "max_height": 1.0, "bumpiness": 0.0, "wells": 0.0}
    )
    env = TetrisEnv(seed=3)
    state = env.reset()
    for _ in range(30):
        if env.is_game_over():
            break
        state, _reward, _done, _info = env.step(
            greedy.choose_action(state, env.get_valid_actions())
        )
    tower = max_column_height(state.board)

    env = TetrisEnv(seed=3)
    state = env.reset()
    for _ in range(30):
        if env.is_game_over():
            break
        state, _reward, _done, _info = env.step(
            HeuristicAgent().choose_action(state, env.get_valid_actions())
        )
    sane = max_column_height(state.board)

    assert tower > sane + 3, (tower, sane)


def test_the_heuristic_agent_keeps_the_board_tidy():
    env = TetrisEnv(seed=8)
    state = env.reset()
    agent = HeuristicAgent()
    for _ in range(120):
        if env.is_game_over():
            break
        state, _reward, _done, _info = env.step(
            agent.choose_action(state, env.get_valid_actions())
        )

    features = extract_features(state.board)
    assert features.holes == count_holes(state.board) <= 4
    assert features.max_height < 12


def test_the_weights_it_reports_are_the_ones_it_uses():
    weights = HeuristicWeights(holes=-123.0)
    agent = HeuristicAgent(weights)
    assert agent.weights is weights
    assert agent.weights.as_dict()["holes"] == -123.0
