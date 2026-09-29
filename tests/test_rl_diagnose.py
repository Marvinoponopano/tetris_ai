"""Tests for the diagnosis (`tetris/rl/diagnose.py`).

A diagnosis is only useful if every number in it is the number it claims to be,
so these tests check the arithmetic: the reward decomposition has to add up to
the episode reward, the action counts to the decisions, and the encoding to what
`tetris.rl.encoding` says it encodes.
"""

from __future__ import annotations

import json
import math
import pathlib

import pytest

torch = pytest.importorskip("torch")

from tetris.env import RewardConfig  # noqa: E402
from tetris.rl.config import TrainingConfig  # noqa: E402
from tetris.rl.diagnose import diagnose_agent, diagnose_checkpoint, diagnose_untrained  # noqa: E402
from tetris.rl.dqn import DQNAgent  # noqa: E402
from tetris.rl.training import train  # noqa: E402

TINY = dict(episodes=2, max_pieces=10, batch_size=4, min_buffer_size=4, hidden_sizes=(8,),
            target_update_steps=3, checkpoint_every=2, log_every=10, seed=0,
            epsilon_decay_steps=40)


def tiny(**overrides) -> TrainingConfig:
    return TrainingConfig.from_dict({**TINY, **overrides})


def diagnose(**overrides):
    payload = dict(games=2, seed=5, max_pieces=12, lookahead_decisions=2)
    payload.update(overrides)
    return diagnose_untrained(tiny(), **payload)


# ------------------------------------------------------------- it is complete


def test_a_diagnosis_covers_every_part_of_the_pipeline():
    diagnosis = diagnose()

    for block in ("reward", "survival", "actions", "q_values", "heuristic", "state",
                  "exploration"):
        assert getattr(diagnosis, block), block
    assert diagnosis.games == 2 and diagnosis.max_pieces == 12
    assert diagnosis.decisions > 0


def test_the_diagnosis_is_json_serialisable_and_savable(tmp_path):
    diagnosis = diagnose()
    path = diagnosis.save(tmp_path / "diagnosis.json")
    data = json.loads(pathlib.Path(path).read_text())

    assert data["label"] == "untrained"
    assert data["state"]["states"] > 0
    assert set(data["reward"]["components"]) == {
        "per_line", "hole", "height", "game_over", "per_piece"}


def test_the_summary_names_every_block_it_reports():
    text = diagnose().summary()

    for heading in ("reward -- where the episode reward actually comes from", "survival",
                    "actions", "Q-values", "against the heuristic agent", "state encoding",
                    "exploration"):
        assert heading in text, heading


# ------------------------------------------------------------- it adds up


def test_the_reward_components_add_up_to_the_episode_reward():
    diagnosis = diagnose()
    components = diagnosis.reward["components"]

    assert sum(components.values()) == pytest.approx(
        diagnosis.reward["episode_mean"] * diagnosis.games, abs=1e-6
    )
    assert sum(diagnosis.reward["shares"].values()) == pytest.approx(1.0, abs=1e-6)


def test_the_components_use_the_agents_own_reward_weights():
    config = tiny(reward={"per_line": 40.0, "game_over": -7.0})
    diagnosis = diagnose_untrained(config, games=1, seed=5, max_pieces=10,
                                   lookahead_decisions=0)
    components = diagnosis.reward["components"]

    # a piece cap of 10 pieces clears no lines: the per-line term must stay 0
    assert components["per_line"] == pytest.approx(0.0)
    assert diagnosed_reward_config(diagnosis)["per_line"] == 40.0
    assert diagnosed_reward_config(diagnosis)["game_over"] == -7.0


def diagnosed_reward_config(diagnosis) -> dict:
    """The reward weights the diagnosed agent was training with."""
    agent = DQNAgent(config=TrainingConfig.from_dict(diagnosis.config), seed=0)
    reward = agent.config.reward_config() or RewardConfig()
    return {"per_line": reward.per_line, "hole": reward.hole, "height": reward.height,
            "game_over": reward.game_over}


def test_the_action_counts_add_up_to_the_decisions():
    diagnosis = diagnose()
    actions = diagnosis.actions

    assert sum(actions["counts"]) == diagnosis.decisions
    assert sum(actions["rotation_counts"]) == diagnosis.decisions
    assert sum(actions["column_counts"]) == diagnosis.decisions


def test_the_greedy_policy_never_picks_an_illegal_action():
    assert diagnose().actions["illegal_selections"] == 0


# --------------------------------------------------------- the state encoding


def test_the_current_piece_plane_is_populated_for_every_state():
    """An empty piece plane would mean the network cannot tell what it is placing."""
    diagnosis = diagnose()

    assert diagnosis.state["states_without_a_piece"] == 0
    # four cells of the falling piece, always: every tetromino has four
    assert diagnosis.state["plane_sums_mean"][1] == pytest.approx(4.0)
    assert diagnosis.state["plane_sums_mean"][2] == pytest.approx(4.0)   # the preview
    assert diagnosis.state["empty_masks"] == 0


def test_the_survival_block_agrees_with_the_pieces_it_played():
    diagnosis = diagnose()

    assert diagnosis.survival["pieces_mean"] <= diagnosis.max_pieces
    assert diagnosis.survival["deaths"] <= diagnosis.games
    assert 0.0 <= diagnosis.survival["peak_column_mean"] <= 20.0


def test_a_game_ending_move_is_counted_rather_than_averaged():
    """The heuristic marks a losing move `-inf`; that must not reach the averages."""
    diagnosis = diagnose_untrained(tiny(), games=2, seed=5, max_pieces=60,
                                   lookahead_decisions=0)
    heuristic = diagnosis.heuristic

    for name in ("regret_mean", "regret_share_mean", "agreement_depth0",
                 "agreement_chance"):
        assert math.isfinite(heuristic[name]), name
    assert heuristic["suicidal_choices"] >= 0
    assert heuristic["forced_deaths"] >= 0
    assert diagnosis.survival["deaths"] > 0          # the untrained policy tops out


def test_the_agreement_chance_is_one_over_the_legal_moves():
    """A piece offers between 9 (O) and 32 (T) placements, so chance is 1/9..1/32."""
    diagnosis = diagnose()

    assert 1 / 32 <= diagnosis.heuristic["agreement_chance"] <= 1 / 9


def test_the_q_vs_heuristic_correlation_is_none_when_the_q_values_are_all_equal():
    """An untrained network outputs a constant, so no correlation is defined."""
    diagnosis = diagnose_untrained(tiny(), games=1, seed=5, max_pieces=30,
                                   lookahead_decisions=0)

    assert diagnosis.heuristic["q_score_correlation"] is None


def test_the_q_vs_heuristic_correlation_is_defined_for_a_network_that_ranks():
    agent = DQNAgent(config=tiny(), seed=0)
    with torch.no_grad():                       # break the symmetry of a fresh net
        for parameter in agent.online.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.2)

    diagnosis = diagnose_agent(agent, games=1, seed=5, max_pieces=20, lookahead_decisions=0)

    correlation = diagnosis.heuristic["q_score_correlation"]
    assert correlation is not None and -1.0 <= correlation <= 1.0


# ----------------------------------------------------------- the exploration


def test_the_exploration_block_reports_the_schedule_at_the_checkpoints_counters():
    config = tiny(epsilon_start=1.0, epsilon_end=0.0, epsilon_decay_steps=10)
    diagnosis = diagnose_untrained(config, games=1, seed=5, max_pieces=8,
                                   lookahead_decisions=0)

    exploration = diagnosis.exploration
    assert exploration["epsilon_start"] == 1.0
    assert exploration["epsilon_decay_steps"] == 10
    assert exploration["epsilon_at_checkpoint"] == pytest.approx(1.0)   # decisions = 0
    assert exploration["schedule"][10_000] == pytest.approx(0.0)       # fully annealed


# --------------------------------------------------------------- a checkpoint


def test_a_trained_checkpoint_can_be_diagnosed(tmp_path):
    agent = DQNAgent(config=tiny(), seed=0)
    agent.decisions = 17
    path = agent.save(tmp_path / "checkpoint_2.pt")

    diagnosis = diagnose_checkpoint(path, games=1, seed=5, max_pieces=10,
                                    lookahead_decisions=0)

    assert diagnosis.checkpoint == str(path)
    assert diagnosis.agent_decisions == 17
    assert diagnosis.decisions > 0
    assert diagnosis.learn_steps == 0
    assert diagnosis.config["seed"] == 0


def test_an_agent_that_trained_for_a_while_is_diagnosed_the_same_way(tmp_path):
    result = train(tiny(), output_dir=tmp_path, logger=None)

    diagnosis = diagnose_agent(result.agent, label="trained", games=1, seed=5,
                              max_pieces=10, lookahead_decisions=0)

    assert diagnosis.episode == 2
    assert diagnosis.decisions > 0
    assert diagnosis.reward["per_step_mean"] != 0.0
