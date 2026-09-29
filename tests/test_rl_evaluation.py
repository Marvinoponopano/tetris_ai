"""Tests for the greedy mid-training evaluation (`tetris/rl/evaluation.py`).

The evaluation is the measurement every later claim rests on, so what is
checked here is that it measures the policy and nothing else: greedy, on fixed
seeds, without moving the agent's own state.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from tetris.agents import RandomAgent  # noqa: E402
from tetris.rl.config import TrainingConfig  # noqa: E402
from tetris.rl.dqn import DQNAgent  # noqa: E402
from tetris.rl.evaluation import (  # noqa: E402
    DEFAULT_EVAL_SEED,
    EVAL_FIELDS,
    evaluate_greedy,
)


def agent(**overrides) -> DQNAgent:
    config = TrainingConfig.from_dict(
        {"episodes": 1, "max_pieces": 8, "batch_size": 4, "hidden_sizes": (8,),
         "epsilon_start": 0.7, "epsilon_end": 0.05, "epsilon_decay_steps": 10,
         "seed": 3, **overrides}
    )
    built = DQNAgent(config=config, seed=3)
    built.set_epsilon(0.7)
    built.decisions = 5
    return built


# ------------------------------------------------------------------ the shape


def test_an_evaluation_row_carries_exactly_the_declared_fields():
    snapshot = evaluate_greedy(agent(), num_games=2, seed=11, max_pieces=6, logger=None)

    assert tuple(snapshot.row()) == EVAL_FIELDS
    assert snapshot.num_games == 2 and snapshot.seed == 11 and snapshot.max_pieces == 6
    assert len(snapshot.games) == 2
    for name in ("score_mean", "lines_mean", "pieces_mean", "lines_per_100_pieces"):
        assert name in snapshot.stats, name
    assert json.loads(json.dumps(snapshot.as_dict()))["games"]


def test_the_row_records_both_the_agents_counters_and_the_schedule():
    snapshot = evaluate_greedy(agent(), num_games=1, seed=11, max_pieces=5, episode=17,
                               logger=None)

    row = snapshot.row()
    assert row["episode"] == 17
    assert row["decisions"] == 5                    # the agent's own counter, not the eval's
    assert row["games"] == 1
    assert row["epsilon"] == pytest.approx(0.7)     # what it had while training


def test_the_summary_says_what_was_measured():
    text = evaluate_greedy(agent(), num_games=2, seed=11, max_pieces=6, episode=4,
                           logger=None).summary()

    assert "after episode 4" in text
    assert "greedy" in text and "seed 11" in text
    assert "score" in text and "lines cleared" in text


# ------------------------------------------------------- it does not disturb


def test_evaluation_puts_the_exploration_rate_back():
    trained = agent()
    evaluate_greedy(trained, num_games=2, seed=11, max_pieces=6, logger=None)

    assert trained.epsilon == pytest.approx(0.7)


def test_evaluation_does_not_move_the_agents_counters_or_weights():
    trained = agent()
    before = {name: parameter.detach().clone() for name, parameter in
              trained.online.state_dict().items()}
    decisions, steps, episode = trained.decisions, trained.steps, trained.episode

    evaluate_greedy(trained, num_games=2, seed=11, max_pieces=6, logger=None)

    assert (trained.decisions, trained.steps, trained.episode) == (decisions, steps, episode)
    for name, parameter in trained.online.state_dict().items():
        assert torch.equal(parameter, before[name]), name


def test_evaluation_runs_at_epsilon_zero():
    """If it explored, two evaluations of the same weights could disagree."""
    trained = agent()
    first = evaluate_greedy(trained, num_games=3, seed=11, max_pieces=8, logger=None)
    second = evaluate_greedy(trained, num_games=3, seed=11, max_pieces=8, logger=None)

    assert [game.score for game in first.games] == [game.score for game in second.games]
    assert first.stats["pieces_mean"] == second.stats["pieces_mean"]


# ------------------------------------------------------------- the seeds


def test_the_same_seed_gives_the_same_games_and_another_seed_does_not():
    trained = agent()
    one = evaluate_greedy(trained, num_games=3, seed=100, max_pieces=8, logger=None)
    again = evaluate_greedy(trained, num_games=3, seed=100, max_pieces=8, logger=None)
    other = evaluate_greedy(trained, num_games=3, seed=200, max_pieces=8, logger=None)

    assert [game.seed for game in one.games] == [100, 101, 102]
    assert [game.score for game in one.games] == [game.score for game in again.games]
    assert [game.seed for game in other.games] == [200, 201, 202]


def test_the_default_seed_is_the_shared_evaluation_seed():
    snapshot = evaluate_greedy(agent(), num_games=1, max_pieces=5, logger=None)
    assert snapshot.seed == DEFAULT_EVAL_SEED


def test_any_agent_with_choose_action_can_be_evaluated():
    """The harness needs `choose_action` and `name`; nothing about DQN."""
    snapshot = evaluate_greedy(RandomAgent(seed=1), num_games=3, seed=7, max_pieces=30,
                              logger=None)

    assert snapshot.stats["games"] == 3
    assert 0.0 < snapshot.pieces_mean < 30
    assert snapshot.training_epsilon == 0.0        # no epsilon to remember
