"""Tests for the training loop: the log, the checkpoints, resuming, determinism.

The runs here are deliberately tiny (a handful of episodes, a few pieces each),
so the whole file is a second of CPU. What is being checked is the *bookkeeping*
around the learning -- the learning itself is checked in `test_rl_dqn.py`.
"""

from __future__ import annotations

import csv
import json

import pytest

torch = pytest.importorskip("torch")

from tetris.env import RewardConfig  # noqa: E402
from tetris.rl.config import TrainingConfig  # noqa: E402
from tetris.rl.dqn import DQNAgent  # noqa: E402
from tetris.rl.training import (  # noqa: E402
    CHECKPOINT_NAME,
    CONFIG_NAME,
    LOG_FIELDS,
    LOG_NAME,
    SUMMARY_NAME,
    TrainingResult,
    train,
)

TINY = dict(episodes=3, max_pieces=6, batch_size=8, min_buffer_size=8, hidden_sizes=(16,),
            target_update_steps=4, checkpoint_every=2, log_every=1, seed=5,
            epsilon_decay_steps=40)


def tiny(**overrides) -> TrainingConfig:
    return TrainingConfig.from_dict({**TINY, **overrides})


def run(tmp_path, **overrides) -> TrainingResult:
    return train(tiny(**overrides), output_dir=tmp_path, logger=None)


def rows_of(result: TrainingResult):
    with open(result.log_path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ------------------------------------------------------------------- the files


def test_a_run_writes_the_log_the_checkpoints_and_a_summary(tmp_path):
    result = run(tmp_path)

    assert (tmp_path / LOG_NAME).exists()
    assert (tmp_path / SUMMARY_NAME).exists()
    assert (tmp_path / CONFIG_NAME).exists()
    assert [path.name for path in sorted(tmp_path.glob("checkpoint_*.pt"))] == [
        "checkpoint_2.pt", "checkpoint_3.pt",
    ]
    assert len(result.episodes) == 3
    assert len(result.episodes) == 3
    assert result.checkpoints == sorted(str(tmp_path / name) for name in
                                        ("checkpoint_2.pt", "checkpoint_3.pt"))


def test_the_config_that_was_used_is_written_next_to_the_results(tmp_path):
    result = run(tmp_path, episodes=2, max_pieces=5)
    written = json.loads((tmp_path / CONFIG_NAME).read_text())

    assert written["episodes"] == 2 and written["max_pieces"] == 5
    assert TrainingConfig.from_dict(written) == tiny(episodes=2, max_pieces=5)
    assert result.config == tiny(episodes=2, max_pieces=5)


def test_every_episode_is_logged_with_the_columns_asked_for(tmp_path):
    rows = rows_of(run(tmp_path))

    assert len(rows) == 3
    assert tuple(rows[0]) == LOG_FIELDS
    for column in ("episode", "reward", "score", "lines", "pieces", "epsilon",
                   "loss_mean", "decisions", "duration_seconds"):
        assert column in rows[0], column

    assert [int(row["episode"]) for row in rows] == [1, 2, 3]     # 1-based, contiguous
    for row in rows:
        assert int(row["pieces"]) == 6                  # the episode cap
        assert float(row["reward"]) <= 0                # a fresh agent is bad at Tetris
        assert float(row["score"]) >= 0
        assert 0.0 <= float(row["epsilon"]) <= 1.0
        assert int(row["decisions"]) == 6


def test_epsilon_in_the_log_follows_the_schedule(tmp_path):
    rows = rows_of(run(tmp_path, epsilon_start=1.0, epsilon_end=0.0, epsilon_decay_steps=18))

    epsilons = [float(row["epsilon"]) for row in rows]

    assert epsilons[0] > epsilons[-1]
    assert all(later <= earlier + 1e-9 for earlier, later in zip(epsilons, epsilons[1:]))
    assert epsilons[-1] == pytest.approx(0.0, abs=1e-6)


def test_the_loss_is_empty_during_the_warm_up_and_a_number_after(tmp_path):
    """No gradient step happens before the buffer holds `min_buffer_size`."""
    rows = rows_of(run(tmp_path, episodes=3, max_pieces=6, min_buffer_size=12,
                       batch_size=8))

    assert rows[0]["loss_mean"] == ""                   # 6 transitions: too few
    assert rows[-1]["loss_mean"] != ""                  # by the 18th it has learned twice


def test_the_summary_agrees_with_the_log(tmp_path):
    result = run(tmp_path)
    summary = json.loads((tmp_path / SUMMARY_NAME).read_text())

    assert summary["episodes_trained"] == len(result.episodes)
    assert summary["best_score"] == max(int(row["score"]) for row in result.episodes)
    assert summary["total_decisions"] == sum(int(row["decisions"]) for row in result.episodes)
    assert summary["config"]["seed"] == 5
    assert summary["environment"]["python_version"]
    assert summary["environment"]["torch_version"].startswith("2.")
    assert summary["final_epsilon"] == pytest.approx(result.final_epsilon)
    assert summary["total_learn_steps"] == result.total_learn_steps
    assert summary["reward_first_window"] == pytest.approx(result.reward_trend()[0])


def test_the_result_describes_itself(tmp_path):
    text = str(run(tmp_path))

    assert "3 episode(s)" in text and "checkpoint" in text
    assert "reward" in text and "best score" in text


def test_an_episode_that_reaches_the_piece_cap_is_not_a_game_over(tmp_path):
    """Truncation is not death: the last transition must still bootstrap."""
    rows = rows_of(run(tmp_path, max_pieces=4))

    assert all(int(row["pieces"]) == 4 for row in rows)
    assert all(row["game_over"] in {"True", "False"} for row in rows)
    assert any(row["game_over"] == "False" for row in rows)   # someone survived the cap


# --------------------------------------------------------------- reproducibility


def test_the_same_config_and_seed_give_the_same_run_twice(tmp_path):
    first = rows_of(run(tmp_path / "a"))
    second = rows_of(run(tmp_path / "b"))

    assert [row["reward"] for row in first] == [row["reward"] for row in second]
    assert [row["score"] for row in first] == [row["score"] for row in second]
    assert [row["loss_mean"] for row in first] == [row["loss_mean"] for row in second]


def test_a_different_seed_gives_a_different_run(tmp_path):
    first = rows_of(run(tmp_path / "a", seed=5))
    second = rows_of(run(tmp_path / "b", seed=6))

    assert [row["score"] for row in first] != [row["score"] for row in second]


# ---------------------------------------------------------------------- resuming


def test_training_resumes_from_a_checkpoint_without_repeating_episodes(tmp_path):
    run(tmp_path, episodes=2, checkpoint_every=2)
    checkpoint = tmp_path / "checkpoint_2.pt"

    resumed = train(tiny(episodes=4), output_dir=tmp_path, resume_from=checkpoint,
                    logger=None)
    rows = rows_of(resumed)

    assert len(resumed.episodes) == 2                    # episodes 3 and 4, nothing else
    assert [int(row["episode"]) for row in rows] == [1, 2, 3, 4]   # no duplicates
    assert resumed.config.episodes == 4
    assert resumed.agent.episode == 4


def test_a_resumed_run_keeps_the_optimizer_and_the_epsilon_it_had(tmp_path):
    first = run(tmp_path, episodes=2, checkpoint_every=2)
    epsilon_at_stop = first.agent.epsilon

    resumed = train(tiny(episodes=3), output_dir=tmp_path,
                    resume_from=tmp_path / "checkpoint_2.pt", logger=None)

    assert resumed.agent.decisions == first.agent.decisions + 6      # picked up where it left
    assert resumed.agent.epsilon <= epsilon_at_stop
    assert any(state for state in resumed.agent.optimizer.state.values())


def test_a_resume_may_change_the_hyperparameters_and_the_new_ones_govern(tmp_path):
    """`--resume x.pt --episodes 300 --learning-rate 5e-4` is the point of resume."""
    run(tmp_path, episodes=2, checkpoint_every=2)

    resumed = train(tiny(episodes=4, learning_rate=0.5), output_dir=tmp_path,
                    resume_from=tmp_path / "checkpoint_2.pt", logger=None)

    assert resumed.config.learning_rate == 0.5
    assert resumed.agent.config.learning_rate == 0.5
    assert resumed.agent.optimizer.param_groups[0]["lr"] == pytest.approx(0.5)
    assert len(resumed.episodes) == 2                 # episodes 3 and 4


# ---------------------------------------------------------------- the agent seam


def test_a_passed_in_agent_carries_its_own_config(tmp_path):
    config = tiny(episodes=2)
    agent = DQNAgent(config=config, seed=5)

    result = train(agent=agent, output_dir=tmp_path, logger=None)

    assert result.agent is agent
    assert result.config == config
    assert len(result.episodes) == 2


def test_a_passed_in_agent_is_told_to_follow_the_config_it_is_given(tmp_path):
    agent = DQNAgent(config=tiny(episodes=9), seed=5)

    result = train(tiny(episodes=2), agent=agent, output_dir=tmp_path, logger=None)

    assert result.config.episodes == 2
    assert agent.config.episodes == 2                  # adopted before the run


def test_the_environment_is_built_with_the_configured_reward(tmp_path):
    result = run(tmp_path, reward={"per_line": 100.0})
    assert result.config.reward_config() == RewardConfig(per_line=100.0)


def test_shorter_episodes_mean_more_games_per_second(tmp_path):
    """Throughput is the number that decides whether a run is worth starting."""
    short = run(tmp_path / "short", max_pieces=2)
    long = run(tmp_path / "long", max_pieces=6)

    assert short.total_decisions < long.total_decisions
    assert short.elapsed_seconds <= long.elapsed_seconds + 0.05


def test_learning_starts_only_once_the_buffer_has_enough(tmp_path):
    result = run(tmp_path, min_buffer_size=8, batch_size=8, episodes=1, max_pieces=6)

    assert result.total_learn_steps == 0                   # 6 transitions < 8
    assert all(row["loss_mean"] is None for row in result.episodes)


def test_a_fresh_run_replaces_the_episode_log(tmp_path):
    """A re-run replaces the episode log; it does not stack rows on top of it.

    `EpisodeLogger` opens the file in append mode, which is right for a resumed
    run and wrong for a fresh one: without this, the second run of a seed reads
    back with the first run's rows above it and every episode curve is a mixture
    of two runs. This bit the queue arm, whose first attempt was interrupted and
    whose log then carried 1,228 stale rows.
    """
    config = tiny(episodes=2, checkpoint_every=99, log_every=99)
    train(config, output_dir=tmp_path, logger=None)
    log = tmp_path / LOG_NAME
    fresh = _log_rows(log)
    assert fresh == 2

    train(config, output_dir=tmp_path, logger=None)
    assert _log_rows(log) == fresh, "a fresh run must not append to the old log"


def test_a_resumed_run_appends_to_the_episode_log(tmp_path):
    """A run that starts at episode 3 keeps the rows the first run wrote."""
    config = tiny(episodes=2, checkpoint_every=99, log_every=99)
    train(config, output_dir=tmp_path, logger=None)
    log = tmp_path / LOG_NAME
    before = _log_rows(log)

    resumed = DQNAgent(config=TrainingConfig.from_dict({**config.to_dict(), "episodes": 4}), seed=0)
    resumed.episode = 2                        # as a resume from a checkpoint leaves it
    train(TrainingConfig.from_dict({**config.to_dict(), "episodes": 4}),
          output_dir=tmp_path, agent=resumed, logger=None)

    assert _log_rows(log) > before, "a resumed run continues its own log"


def _log_rows(path) -> int:
    """Data rows in the episode log, i.e. not counting its header."""
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return len(lines) - 1 if lines else 0
