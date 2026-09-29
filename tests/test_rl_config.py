"""Tests for `TrainingConfig`: validation, the epsilon schedule, identity."""

from __future__ import annotations

import json

import pytest

from tetris.env import RewardConfig

torch = pytest.importorskip("torch")

from tetris.rl.config import DEVICES, REWARD_FIELDS, TrainingConfig  # noqa: E402


# ------------------------------------------------------------------- defaults


def test_defaults_are_a_small_cpu_run():
    config = TrainingConfig()
    assert config.episodes == 200
    assert config.batch_size == 64
    assert config.buffer_size == 50_000
    assert 0.0 < config.gamma < 1.0
    assert config.epsilon_start > config.epsilon_end
    assert config.device == "cpu"
    assert config.reward is None                 # the environment's own default


def test_a_config_round_trips_through_a_json_file(tmp_path):
    config = TrainingConfig(episodes=7, batch_size=8, hidden_sizes=(32, 16),
                            reward={"per_line": 20.0}, label="round trip")
    path = tmp_path / "train.json"
    config.save(path)

    assert json.loads(path.read_text())["episodes"] == 7
    assert TrainingConfig.load(path) == config
    assert TrainingConfig.load(path).hidden_sizes == (32, 16)   # a tuple again, not a list
    assert TrainingConfig.load_all(path) == (config,)


def test_the_dict_has_exactly_the_declared_fields():
    assert tuple(TrainingConfig().to_dict()) == TrainingConfig.CONFIG_FIELDS


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError, match="learning_rat"):
        TrainingConfig.from_dict({"episodes": 1, "learning_rat": 0.1})
    with pytest.raises(ValueError, match="nope"):
        TrainingConfig().with_overrides(nope=1)


@pytest.mark.parametrize("overrides, message", [
    ({"episodes": 0}, "episodes"),
    ({"batch_size": 0}, "batch_size"),
    ({"buffer_size": 0}, "buffer_size"),
    ({"learning_rate": 0.0}, "learning_rate"),
    ({"gamma": 0.0}, "gamma"),
    ({"gamma": 1.5}, "gamma"),
    ({"epsilon_start": 1.5}, "epsilon_start"),
    ({"epsilon_end": -0.1}, "epsilon_end"),
    ({"target_update_steps": 0}, "target_update_steps"),
    ({"min_buffer_size": 0}, "min_buffer_size"),
    ({"max_pieces": 0}, "max_pieces"),
    ({"checkpoint_every": 0}, "checkpoint_every"),
    ({"hidden_sizes": ()}, "hidden_sizes"),
    ({"hidden_sizes": (0,)}, "hidden"),
    ({"seed": "0"}, "seed"),
    ({"device": "gpu"}, "device"),
    ({"reward": {"per_line": 1, "nope": 2}}, "nope"),
])
def test_bad_values_are_rejected_with_a_useful_message(overrides, message):
    with pytest.raises(ValueError, match=message):
        TrainingConfig.from_dict(overrides)


def test_epsilon_end_may_not_exceed_epsilon_start():
    with pytest.raises(ValueError, match="exploration should shrink"):
        TrainingConfig(epsilon_start=0.1, epsilon_end=0.5)


def test_the_warm_up_must_fit_in_the_buffer():
    """Otherwise the first gradient step never arrives, silently."""
    with pytest.raises(ValueError, match="learning would never start"):
        TrainingConfig(buffer_size=100, min_buffer_size=200)


def test_hidden_sizes_accept_a_single_number_and_a_list():
    assert TrainingConfig(hidden_sizes=128).hidden_sizes == (128,)
    assert TrainingConfig(hidden_sizes=[64, 64]).hidden_sizes == (64, 64)


def test_the_reward_weights_must_be_the_environments_own():
    config = TrainingConfig(reward={"per_line": 12.0, "game_over": -50})
    assert set(config.reward) == {"per_line", "game_over"}       # stored sorted, as given
    assert set(REWARD_FIELDS) >= set(config.reward)
    assert config.reward_config() == RewardConfig(per_line=12.0, game_over=-50)
    assert TrainingConfig().reward_config() is None


# ------------------------------------------------------------ epsilon schedule


def test_epsilon_falls_in_a_straight_line_and_then_stops():
    config = TrainingConfig(epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=100)

    assert config.epsilon_for_step(0) == pytest.approx(1.0)
    assert config.epsilon_for_step(25) == pytest.approx(0.775)
    assert config.epsilon_for_step(50) == pytest.approx(0.55)
    assert config.epsilon_for_step(99) == pytest.approx(0.1 + 0.9 * 0.01)
    assert config.epsilon_for_step(100) == pytest.approx(0.1)
    assert config.epsilon_for_step(10_000) == pytest.approx(0.1)


def test_epsilon_never_increases():
    config = TrainingConfig(epsilon_start=0.9, epsilon_end=0.2, epsilon_decay_steps=250)
    values = [config.epsilon_for_step(step) for step in range(0, 500, 7)]
    assert all(later <= earlier + 1e-12 for earlier, later in zip(values, values[1:]))
    assert values[0] == pytest.approx(0.9)
    assert values[-1] == pytest.approx(0.2)


def test_a_negative_step_counts_as_the_start():
    assert TrainingConfig().epsilon_for_step(-5) == 1.0


# -------------------------------------------------------------------- identity


def test_config_id_ignores_the_presentation_fields():
    base = TrainingConfig(episodes=10)
    assert base.with_overrides(label="x").config_id == base.config_id
    assert base.with_overrides(log_every=1).config_id == base.config_id
    assert base.with_overrides(checkpoint_every=5).config_id == base.config_id


def test_config_id_changes_when_learning_could_change():
    base = TrainingConfig(episodes=10, seed=1)
    for change in ({"episodes": 11}, {"learning_rate": 2e-3}, {"batch_size": 8},
                   {"gamma": 0.9}, {"epsilon_decay_steps": 5}, {"hidden_sizes": (64,)},
                   {"target_update_steps": 7}, {"seed": 2},
                   {"reward": {"per_line": 11.0}}):
        assert base.with_overrides(**change).config_id != base.config_id, change


def test_describe_reads_like_a_caption():
    text = TrainingConfig(episodes=50, batch_size=32).describe()
    assert "dqn" in text and "50 episodes" in text
    assert "batch 32" in text and "gamma" in text and "eps" in text
    assert str(TrainingConfig(episodes=50, batch_size=32)) == text


def test_devices_are_the_known_ones():
    assert "cpu" in DEVICES and "auto" in DEVICES
    for device in DEVICES:
        assert TrainingConfig(device=device).device == device
