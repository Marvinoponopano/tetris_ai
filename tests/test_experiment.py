"""Tests for the experiment framework: running, logging, comparing, profiling.

Runs here are deliberately tiny (a couple of games, a handful of pieces) so the
suite stays fast; the framework is about *records*, not about long games. The
most important test in this file is the pair of reproducibility ones: the same
config and seed must give identical `stats` and `diagnostics` every time, while
`timing` is allowed to move.
"""

from __future__ import annotations

import json
import statistics

import pytest

from tetris.agents import RandomAgent
from tetris.config import ExperimentConfig
from tetris.experiment import (
    COMPARISON_FIELDS,
    LOG_NAME,
    Metered,
    append_run_log,
    compare_runs,
    comparison_rows,
    gather_runs,
    load_run_log,
    run_experiment,
    save_comparison_csv,
)

SMALL = {"agent": "random", "num_games": 2, "seed": 1, "max_pieces": 10}
GREEDY = {"agent": "heuristic", "num_games": 2, "seed": 1, "max_pieces": 6,
          "lookahead_depth": 0}


def small(**overrides) -> ExperimentConfig:
    """A tiny random-agent config, with any field overridden."""
    return ExperimentConfig.from_dict({**SMALL, **overrides})


# ------------------------------------------------------------------- records


def test_a_result_records_everything_an_experiment_has_to_report():
    """The required record, field by field: agent, config, games, seed, numbers."""
    config = ExperimentConfig(**SMALL)
    result = run_experiment(config, logger=None)

    # agent name, configuration, number of games, seed
    assert result.agent == "random"
    assert result.config == config
    assert result.config_id == config.config_id
    assert result.num_games == 2
    assert result.seed == 1

    # average / median / maximum score, average / maximum lines, average pieces
    for key in ("score_mean", "score_median", "score_max", "lines_mean", "lines_max",
                "pieces_mean"):
        assert key in result.stats
    assert result.stats["score_mean"] <= result.stats["score_max"]
    assert result.stats["lines_max"] <= 10

    # runtime, and what the run was made with
    assert result.timing["elapsed_seconds"] > 0
    assert result.environment["python_version"]
    assert result.games and len(result.games) == 2
    assert result.started_at and result.finished_at


def test_the_same_config_and_seed_reproduce_stats_and_diagnostics():
    """Determinism: only the clock is allowed to differ between identical runs."""
    config = ExperimentConfig(**GREEDY)

    first = run_experiment(config, logger=None)
    second = run_experiment(config, logger=None)

    assert first.stats == second.stats
    assert first.diagnostics == second.diagnostics
    assert [game.as_dict() for game in first.games] == \
           [game.as_dict() for game in second.games]
    assert first.config_id == second.config_id          # same experiment
    assert first.run_id != second.run_id                # different records


def test_timing_is_not_part_of_the_deterministic_numbers():
    result = run_experiment(small(), logger=None)

    assert not set(result.timing) & set(result.stats)
    assert result.timing["games_per_second"] > 0
    assert result.timing["seconds_per_game"] > 0
    assert result.timing["pieces_per_second"] >= result.timing["games_per_second"]


def test_throughput_reflects_how_much_the_agent_did():
    result = run_experiment(small(agent="heuristic", lookahead_depth=0, max_pieces=6),
                            logger=None)

    pieces = sum(game.pieces for game in result.games)
    assert result.diagnostics["decisions"] == pieces
    assert result.timing["decisions_per_second"] > 0
    assert result.timing["pieces_per_second"] == pytest.approx(
        pieces / result.timing["elapsed_seconds"], rel=1e-3)


def test_a_searching_agent_reports_the_boards_it_scored():
    greedy = run_experiment(ExperimentConfig(**GREEDY), logger=None)
    blind = run_experiment(small(), logger=None)

    assert greedy.diagnostics["boards_scored"] > 0
    assert greedy.timing["boards_per_decision"] > 1
    assert blind.diagnostics["boards_scored"] == 0       # random scores no boards
    assert blind.timing["boards_per_decision"] == 0


def test_the_peak_of_the_stack_is_recorded_per_game():
    result = run_experiment(ExperimentConfig(**GREEDY), logger=None)
    assert result.diagnostics["peak_column_mean"] > 0
    assert result.diagnostics["peak_column_max"] >= result.diagnostics["peak_column_mean"]


def test_an_agent_with_a_name_of_its_own_keeps_it():
    class AlwaysFirst:
        name = "always-first"

        def choose_action(self, state, valid_actions):
            return valid_actions[0]

    result = run_experiment(small(), agent=AlwaysFirst(), logger=None)

    assert result.agent == "always-first"
    assert result.config.agent == "random"        # the config still says what was asked


def test_a_stateful_learning_agent_fits_in_the_framework(tmp_path):
    """The seam for training: any agent object is run, metered and logged as it is.

    Nothing here knows how the agent learns -- it only has to choose actions. A
    training loop can therefore live inside the agent and still get the config,
    the seed control, the run log and the throughput measurement for free.
    """
    class Counting:
        name = "counting"
        last_decision = None

        def __init__(self):
            self.experiences = 0

        def reset(self, seed=None):
            pass

        def choose_action(self, state, valid_actions):
            self.experiences += len(valid_actions)      # it "sees" each option
            return valid_actions[-1]

    learner = Counting()
    config = ExperimentConfig(agent="counting", num_games=3, seed=4, max_pieces=5)
    result = run_experiment(config, agent=learner, save_dir=tmp_path, logger=None)

    assert learner.experiences > 0
    assert result.agent == "counting"
    assert result.diagnostics["decisions"] == sum(game.pieces for game in result.games)
    row = json.loads((tmp_path / LOG_NAME).read_text().splitlines()[0])
    assert row["agent"] == "counting"
    assert row["stats"]["score_mean"] == result.stats["score_mean"]


def test_an_unknown_agent_name_is_an_error():
    with pytest.raises(KeyError, match="clairvoyant"):
        run_experiment(small(agent="clairvoyant"), logger=None)


def test_run_experiment_accepts_a_config_like_argument(tmp_path):
    """A dict or a path works, so notebooks and scripts need no ceremony."""
    assert run_experiment(SMALL, logger=None).agent == "random"

    config = ExperimentConfig(**SMALL)
    path = config.save(tmp_path / "config.json")
    assert run_experiment(path, logger=None).config_id == config.config_id


# ------------------------------------------------------------------- metering


def test_metered_counts_decisions_and_forwards_the_agent():
    class Toy:
        name = "toy"
        last_decision = None

        def __init__(self):
            self.seeds = []

        def reset(self, seed=None):
            self.seeds.append(seed)

        def choose_action(self, state, valid_actions):
            self.last_decision = type("D", (), {"nodes": 7})()
            return valid_actions[0]

    toy = Toy()
    metered = Metered(toy)
    state = type("S", (), {"board": [[0] * 10 for _ in range(20)]})()
    valid = [(0, 0), (0, 1)]

    metered.reset(seed=3)
    assert metered.choose_action(state, valid) == (0, 0)
    assert metered.decisions == 1
    assert metered.boards_scored == 7
    assert toy.seeds == [3]                       # reset reached the agent
    assert metered.name == "toy"


def test_metered_banks_one_peak_per_game():
    """Peaks are collected per game: reset() closes the game that just ended."""
    board = [[0] * 10 for _ in range(20)]
    board[19][4] = 1                               # one cell, so a peak of 1
    state = type("S", (), {"board": board})()
    agent = Metered(RandomAgent(seed=1))

    for _ in range(2):                             # two games of two decisions
        agent.choose_action(state, [(0, 0), (0, 1)])
        agent.choose_action(state, [(0, 0), (0, 1)])
        agent.reset(seed=0)
    agent.flush()                                  # the last game is banked too

    assert agent.game_peaks == [1.0, 1.0]
    assert agent.decisions == 4


# ------------------------------------------------------------------- saving


def test_results_are_saved_as_machine_readable_json(tmp_path):
    config = ExperimentConfig(**GREEDY)
    result = run_experiment(config, save_dir=tmp_path, logger=None)

    assert result.result_path is not None
    saved = json.loads((tmp_path / f"run-{result.run_id}.json").read_text())

    assert saved["agent"] == "heuristic"
    assert saved["config"] == config.to_dict()
    assert saved["num_games"] == 2
    assert saved["seed"] == 1
    assert saved["stats"]["score_mean"] == result.stats["score_mean"]
    assert saved["timing"]["games_per_second"] == result.timing["games_per_second"]
    assert saved["environment"]["python_version"] == result.environment["python_version"]
    assert len(saved["games"]) == 2               # every game, for pairing later
    assert set(saved["games"][0]) == {"game", "seed", "score", "lines_cleared",
                                      "pieces", "capped"}


def test_the_run_log_is_one_json_line_per_run(tmp_path):
    first = run_experiment(ExperimentConfig(**SMALL), save_dir=tmp_path, logger=None)
    second = run_experiment(ExperimentConfig(**SMALL), save_dir=tmp_path, logger=None)

    log = tmp_path / LOG_NAME
    rows = load_run_log(log)

    assert log.exists()
    assert len(rows) == 2
    assert {row["run_id"] for row in rows} == {first.run_id, second.run_id}
    assert all(row["agent"] == "random" for row in rows)
    assert all("games" not in row for row in rows)      # the log is a summary
    assert all(row["result_path"] for row in rows)


def test_two_runs_of_one_config_get_their_own_files(tmp_path):
    """Runs must not overwrite each other, even in the same second."""
    config = ExperimentConfig(**SMALL)
    one = run_experiment(config, save_dir=tmp_path, logger=None)
    two = run_experiment(config, save_dir=tmp_path, logger=None)

    assert one.config_id == two.config_id
    assert one.run_id != two.run_id
    files = sorted(path.name for path in tmp_path.glob("run-*.json"))
    assert files == sorted([f"run-{one.run_id}.json", f"run-{two.run_id}.json"])


def test_the_run_id_is_safe_for_a_file_name():
    result = run_experiment(small(), logger=None)
    assert all(character.isalnum() or character == "-" for character in result.run_id)
    assert result.run_id.endswith(result.config_id)


def test_append_run_log_can_write_to_an_empty_path(tmp_path):
    result = run_experiment(small(), logger=None)
    path = append_run_log(result, tmp_path / "deep" / "runs.jsonl")
    assert json.loads(open(path).read().splitlines()[0])["run_id"] == result.run_id


# ------------------------------------------------------------------ gathering


def test_runs_can_be_gathered_from_a_result_a_log_or_a_directory(tmp_path):
    config = ExperimentConfig(**SMALL)
    result = run_experiment(config, save_dir=tmp_path, logger=None)

    assert len(gather_runs(result)) == 1
    assert len(gather_runs(tmp_path / LOG_NAME)) == 1
    assert len(gather_runs(tmp_path)) == 1             # log and run file are one run
    assert len(gather_runs(tmp_path / f"run-{result.run_id}.json")) == 1
    assert len(gather_runs([result, tmp_path / LOG_NAME])) == 1
    assert gather_runs([]) == []


def test_a_run_of_two_configs_lists_both_in_order(tmp_path):
    configs = [ExperimentConfig(**SMALL),
               ExperimentConfig(agent="heuristic", num_games=1, seed=2,
                                max_pieces=5, lookahead_depth=0)]
    for config in configs:
        run_experiment(config, save_dir=tmp_path, logger=None)

    rows = comparison_rows(gather_runs(tmp_path))

    assert [row["agent"] for row in rows] == ["random", "heuristic"]
    assert all(row["num_games"] for row in rows)


# ----------------------------------------------------------------- comparing


def test_compare_runs_prints_the_columns_a_comparison_needs(tmp_path):
    run_experiment(ExperimentConfig(**SMALL), save_dir=tmp_path, logger=None)
    run_experiment(ExperimentConfig(**GREEDY), save_dir=tmp_path, logger=None)

    rows = comparison_rows(gather_runs(tmp_path))
    table = compare_runs(tmp_path)

    for key in COMPARISON_FIELDS:
        assert key in rows[0]
    assert "random" in table and "heuristic" in table
    assert "score mean" in table and "games/s" in table
    assert len(table.splitlines()) == len(rows) + 2         # header + rule


def test_compare_runs_accepts_a_live_result():
    result = run_experiment(small(), logger=None)
    assert "random" in compare_runs(result)
    assert compare_runs([]) == "no runs to compare"


def test_a_comparison_can_be_saved_as_csv(tmp_path):
    run_experiment(ExperimentConfig(**SMALL), save_dir=tmp_path, logger=None)
    run_experiment(ExperimentConfig(**GREEDY), save_dir=tmp_path, logger=None)

    rows = comparison_rows(gather_runs(tmp_path))
    path = save_comparison_csv(rows, tmp_path / "compared.csv")
    lines = open(path).read().splitlines()

    assert lines[0].split(",") == list(COMPARISON_FIELDS)
    assert len(lines) == 3                                   # header + two runs
    assert "," in lines[1]


def test_compare_runs_writes_the_csv_when_asked(tmp_path):
    run_experiment(ExperimentConfig(**SMALL), save_dir=tmp_path, logger=None)
    table = compare_runs(tmp_path, save_csv=tmp_path / "out.csv")

    assert "random" in table
    assert (tmp_path / "out.csv").exists()


def test_a_missing_number_prints_as_a_dash_rather_than_crashing():
    result = run_experiment(small(), logger=None)
    row = result.log_row()
    row.pop("stats")
    row.pop("timing")
    assert "-" in compare_runs(row)


# ------------------------------------------------------------------ profiling


def test_profiling_writes_a_report_next_to_the_results(tmp_path):
    config = ExperimentConfig(agent="heuristic", num_games=1, seed=3, max_pieces=6,
                              lookahead_depth=0)
    result = run_experiment(config, save_dir=tmp_path, profile=True, logger=None)

    assert result.profile is not None
    report = open(result.profile).read()

    assert result.run_id in report
    assert "cProfile" in report
    assert "function calls" in report
    assert "choose_action" in report            # the agent's work is in there


def test_profiling_without_a_save_dir_still_writes_a_report(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = run_experiment(ExperimentConfig(**SMALL), profile=True, logger=None)

    assert result.profile == f"profile-{result.run_id}.txt"
    assert (tmp_path / result.profile).exists()


def test_profiling_is_off_by_default():
    assert run_experiment(small(), logger=None).profile is None


# ------------------------------------------------------------------- printing


def test_summary_mentions_the_config_the_numbers_and_the_speed():
    result = run_experiment(ExperimentConfig(**GREEDY), logger=None)
    text = result.summary()

    assert result.run_id in text
    assert "heuristic" in text and "depth 0" in text
    assert "score" in text and "lines cleared" in text and "pieces" in text
    assert "games/s" in text
    assert "boards scored" in text              # because this agent searches
    assert str(result) == text


def test_summary_of_a_blind_agent_has_no_search_line():
    text = run_experiment(small(), logger=None).summary()
    assert "boards scored" not in text


def test_the_stats_in_the_file_are_the_stats_of_the_games_in_the_file(tmp_path):
    """The strongest check that a run was saved correctly: recompute it from `games`."""
    result = run_experiment(ExperimentConfig(**GREEDY), save_dir=tmp_path, logger=None)
    saved: dict = json.loads(open(result.result_path).read())

    scores = [game["score"] for game in saved["games"]]
    lines = [game["lines_cleared"] for game in saved["games"]]
    pieces = [game["pieces"] for game in saved["games"]]

    assert len(scores) == saved["num_games"]
    assert saved["stats"]["score_mean"] == pytest.approx(statistics.fmean(scores))
    assert saved["stats"]["score_median"] == pytest.approx(statistics.median(scores))
    assert saved["stats"]["score_max"] == max(scores)
    assert saved["stats"]["lines_mean"] == pytest.approx(statistics.fmean(lines))
    assert saved["stats"]["lines_max"] == max(lines)
    assert saved["stats"]["pieces_mean"] == pytest.approx(statistics.fmean(pieces))
    assert saved["timing"]["elapsed_seconds"] > 0
    assert saved["environment"]["python_version"] == result.environment["python_version"]


def test_a_result_can_be_rebuilt_from_its_own_json(tmp_path):
    config = ExperimentConfig(**SMALL)
    result = run_experiment(config, save_dir=tmp_path, logger=None)
    saved: dict = json.loads(open(result.result_path).read())

    rebuilt = comparison_rows([saved])[0]

    assert rebuilt["agent"] == result.agent
    assert rebuilt["score_mean"] == result.stats["score_mean"]
    assert rebuilt["run_id"] == result.run_id
    assert rebuilt["config_id"] == result.config_id
    assert rebuilt["games_per_second"] == result.timing["games_per_second"]
