"""Tests for the configuration system: round-trips, validation, identity.

The point of these tests is that a config is *reliable*: what you write to a
file is what you read back, a typo is an error rather than a silently ignored
key, and `config_id` identifies an experiment by everything that can change its
results (and nothing that cannot).
"""

from __future__ import annotations

import json

import pytest

from tetris.config import (
    CONFIG_FIELDS,
    ExperimentConfig,
    dependency_versions,
    environment_info,
    loads_configs,
    parse_override,
)
from tetris.heuristic import DEFAULT_BEAM, HeuristicWeights


# ------------------------------------------------------------------ defaults


def test_defaults_are_a_small_random_run():
    config = ExperimentConfig()
    assert config.agent == "random"
    assert config.num_games == 100
    assert config.seed is None
    assert config.lookahead_depth == 0
    assert config.beam == DEFAULT_BEAM
    assert config.weights is None
    assert config.max_pieces is not None


def test_every_field_survives_a_dict_round_trip():
    config = ExperimentConfig(
        agent="heuristic",
        num_games=7,
        seed=42,
        max_pieces=200,
        lookahead_depth=1,
        beam=3,
        weights={"holes": -8.0},
        label="round trip",
    )
    again = ExperimentConfig.from_dict(config.to_dict())
    assert again == config
    assert again.to_dict() == config.to_dict()


def test_to_dict_has_exactly_the_known_fields():
    config = ExperimentConfig()
    assert tuple(config.to_dict()) == CONFIG_FIELDS


def test_json_round_trip_through_a_file(tmp_path):
    config = ExperimentConfig(agent="heuristic", num_games=5, seed=3, lookahead_depth=1)
    path = tmp_path / "nested" / "config.json"     # parent made for us
    written = config.save(path)

    assert written == str(path)
    assert json.loads(path.read_text())["agent"] == "heuristic"
    assert ExperimentConfig.load(path) == config
    assert ExperimentConfig.load_all(path) == (config,)


def test_a_file_can_hold_a_list_of_configs(tmp_path):
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps([
        {"agent": "random", "num_games": 2, "seed": 1},
        {"agent": "heuristic", "num_games": 2, "seed": 2, "lookahead_depth": 1},
    ]))

    configs = ExperimentConfig.load_all(path)

    assert [config.seed for config in configs] == [1, 2]
    assert configs[1].lookahead_depth == 1
    with pytest.raises(ValueError, match="list of configs"):
        ExperimentConfig.load(path)


def test_configs_can_be_read_from_a_json_string():
    configs = loads_configs('[{"agent": "random", "num_games": 3}]')
    assert len(configs) == 1
    assert configs[0].num_games == 3
    assert len(loads_configs('{"agent": "random", "num_games": 3}')) == 1


def test_a_broken_config_file_names_itself(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json")
    with pytest.raises(ValueError, match="broken.json"):
        ExperimentConfig.load(path)


# ---------------------------------------------------------------- validation


def test_unknown_keys_are_rejected():
    """A typo must not silently run a different experiment."""
    with pytest.raises(ValueError, match="typo"):
        ExperimentConfig.from_dict({"num_games": 10, "typo_key": 1})


def test_unknown_keys_are_rejected_by_overrides_too():
    with pytest.raises(ValueError, match="nope"):
        ExperimentConfig().with_overrides(nope=1)


@pytest.mark.parametrize("overrides, message", [
    ({"agent": ""}, "agent"),
    ({"agent": 7}, "agent"),
    ({"num_games": 0}, "num_games"),
    ({"num_games": 2.5}, "num_games"),
    ({"seed": "42"}, "seed"),
    ({"max_pieces": 0}, "max_pieces"),
    ({"lookahead_depth": -1}, "lookahead_depth"),
    ({"lookahead_depth": 1.5}, "lookahead_depth"),
    ({"beam": 0}, "beam"),
])
def test_bad_values_are_rejected_with_a_useful_message(overrides, message):
    with pytest.raises(ValueError, match=message):
        ExperimentConfig.from_dict(overrides)


def test_seed_and_cap_may_be_none():
    config = ExperimentConfig(seed=None, max_pieces=None)
    assert config.seed is None
    assert config.max_pieces is None
    assert ExperimentConfig.from_dict(config.to_dict()) == config


def test_weights_are_completed_and_canonicalised():
    """A partial weight dict becomes the whole vector: a saved config is explicit."""
    config = ExperimentConfig(weights={"holes": -8, "bumpiness": -2.5})

    assert config.weights["holes"] == -8.0
    assert config.weights["bumpiness"] == -2.5
    assert set(config.weights) == set(HeuristicWeights().as_dict())   # nothing missing
    assert list(config.weights) == sorted(config.weights)             # stable order
    assert all(isinstance(value, float) for value in config.weights.values())

    with pytest.raises(ValueError):
        ExperimentConfig(weights={"holls": -8})


def test_weights_accept_a_heuristic_weights_object():
    weights = HeuristicWeights.from_any({"holes": -3})
    assert ExperimentConfig(weights=weights).weights["holes"] == -3.0


# ------------------------------------------------------------------ identity


def test_config_id_is_stable_and_result_sensitive():
    base = ExperimentConfig(agent="heuristic", num_games=10, seed=1, lookahead_depth=1)

    assert base.config_id == ExperimentConfig.from_dict(base.to_dict()).config_id
    assert base.config_id != base.with_overrides(seed=2).config_id
    assert base.config_id != base.with_overrides(lookahead_depth=0).config_id
    assert base.config_id != base.with_overrides(num_games=11).config_id
    assert base.config_id != base.with_overrides(weights={"holes": -8.0}).config_id


def test_config_id_ignores_the_label():
    """The label is for humans: it must not split one experiment into two ids."""
    plain = ExperimentConfig(agent="heuristic", num_games=4)
    labelled = plain.with_overrides(label="the good one")
    assert labelled.label == "the good one"
    assert labelled.config_id == plain.config_id


def test_describe_reads_like_a_caption():
    config = ExperimentConfig(agent="heuristic", num_games=30, seed=123,
                              max_pieces=400, lookahead_depth=1)
    text = config.describe()
    assert "heuristic" in text
    assert "depth 1" in text
    assert "30 games" in text
    assert "seed 123" in text
    assert "cap 400" in text
    assert str(config) == text


def test_describe_mentions_none_for_an_unseeded_run():
    assert "seed none" in ExperimentConfig(seed=None).describe()
    assert "no cap" in ExperimentConfig(max_pieces=None).describe()


# ----------------------------------------------------------------- overrides


def test_with_overrides_returns_a_new_validated_config():
    base = ExperimentConfig(agent="heuristic", num_games=10, seed=1)
    changed = base.with_overrides(num_games=20, seed=2)

    assert (changed.num_games, changed.seed) == (20, 2)
    assert (base.num_games, base.seed) == (10, 1)          # untouched
    with pytest.raises(ValueError, match="num_games"):
        base.with_overrides(num_games=0)


def test_from_any_accepts_config_dict_and_path(tmp_path):
    config = ExperimentConfig(agent="random", num_games=2)
    path = config.save(tmp_path / "c.json")

    assert ExperimentConfig.from_any(config) is config
    assert ExperimentConfig.from_any({"agent": "random", "num_games": 2}) == config
    assert ExperimentConfig.from_any(path) == config
    assert ExperimentConfig.from_any(tmp_path / "c.json") == config
    with pytest.raises(ValueError, match="expected"):
        ExperimentConfig.from_any(42)


@pytest.mark.parametrize("text, expected", [
    ("num_games=10", ("num_games", 10)),
    ("seed=null", ("seed", None)),
    ("lookahead_depth=0", ("lookahead_depth", 0)),
    ("label=my run", ("label", "my run")),
    ('weights={"holes": -8}', ("weights", {"holes": -8})),
    ("seed=0012", ("seed", 12)),
])
def test_parse_override_reads_json_when_it_can(text, expected):
    assert parse_override(text) == expected


@pytest.mark.parametrize("text", ["num_games", "=10", ""])
def test_parse_override_rejects_non_pairs(text):
    with pytest.raises(ValueError, match="key=value"):
        parse_override(text)


# ---------------------------------------------------------------- provenance


def test_environment_info_records_what_the_run_was_made_with():
    info = environment_info()
    for key in ("python_version", "python_implementation", "platform", "machine",
                "numpy", "pygame"):
        assert key in info
    assert info["python_version"].count(".") == 2
    assert info["numpy"] is not None            # the one hard dependency


def test_dependency_versions_reports_none_for_something_absent():
    versions = dependency_versions(("numpy", "a-package-that-is-not-installed"))
    assert versions["numpy"] is not None
    assert versions["a-package-that-is-not-installed"] is None
