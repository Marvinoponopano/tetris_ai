"""Tests for the evaluation command line (`evaluate.py` at the project root)."""

import argparse
import json

import pytest

from evaluate import AGENTS, build_agent, main, non_negative_int


def test_the_baselines_are_registered():
    """`random` and `heuristic` are the baselines; `dqn` joins them when PyTorch is
    installed (it is built lazily, so the registry entry is always there)."""
    assert {"random", "heuristic"} <= set(AGENTS)
    assert "dqn" in AGENTS


def test_the_registry_builds_the_agent_that_was_asked_for():
    assert build_agent("random").name == "random"
    assert build_agent("heuristic").name == "heuristic"
    assert build_agent("random", seed=1).rng.random() == build_agent("random", seed=1).rng.random()


def test_an_unknown_agent_is_rejected():
    with pytest.raises(KeyError, match="unknown agent"):
        build_agent("clairvoyant")


def test_the_registry_passes_the_search_settings_through():
    agent = build_agent("heuristic", lookahead=2, beam=3)
    assert agent.lookahead_depth == 2
    assert agent.beam == 3
    assert build_agent("heuristic").lookahead_depth == 0     # default: no lookahead
    # agents with nothing to search accept the arguments and ignore them
    assert build_agent("random", lookahead=2, beam=3).name == "random"


def test_lookahead_is_validated_by_the_parser():
    assert non_negative_int("0") == 0
    assert non_negative_int("2") == 2
    with pytest.raises(argparse.ArgumentTypeError):
        non_negative_int("-1")


def test_the_lookahead_depth_reaches_the_agent(capsys, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate.py", "--agent", "heuristic", "--lookahead", "1", "--games", "1",
         "--max-pieces", "3", "--seed", "5", "--quiet"],
    )

    main()

    printed = capsys.readouterr().out
    assert "lookahead depth 1, beam" in printed


def test_tracing_prints_the_decision_the_agent_took(capsys, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate.py", "--agent", "heuristic", "--lookahead", "1", "--games", "1",
         "--max-pieces", "3", "--seed", "5", "--quiet", "--trace", "2"],
    )

    main()

    printed = capsys.readouterr().out
    assert "--- game 1, move 1 " in printed
    assert printed.count("AI selected:") == 2        # --trace 2
    assert "Rotation: " in printed and "Column: " in printed
    assert "Evaluation: " in printed


def test_tracing_can_draw_the_board(capsys, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate.py", "--agent", "heuristic", "--games", "1", "--max-pieces", "2",
         "--seed", "5", "--quiet", "--trace", "1", "--trace-board"],
    )

    main()

    printed = capsys.readouterr().out
    assert "#" in printed                             # the board, as the AI saw it


def test_nothing_is_traced_by_default(capsys, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate.py", "--agent", "heuristic", "--games", "1", "--max-pieces", "3",
         "--seed", "5", "--quiet"],
    )

    main()

    assert "AI selected:" not in capsys.readouterr().out


def test_a_short_evaluation_runs_and_saves_json(tmp_path, capsys, monkeypatch):
    target = tmp_path / "results.json"
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate.py", "--agent", "heuristic", "--games", "2", "--seed", "5",
         "--max-pieces", "20", "--save", str(target)],
    )

    main()

    assert "Game 2/2" in capsys.readouterr().out
    saved = json.loads(target.read_text())
    assert saved["agent"] == "heuristic"
    assert saved["num_games"] == 2
    assert len(saved["games"]) == 2
    assert saved["stats"]["pieces_mean"] == 20          # capped, and said so


def test_a_csv_can_be_written_instead(tmp_path, monkeypatch):
    target = tmp_path / "results.csv"
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate.py", "--games", "2", "--seed", "5", "--quiet", "--save", str(target)],
    )

    main()

    lines = target.read_text().strip().splitlines()
    assert lines[0].startswith("game,seed,score,lines_cleared,pieces,capped")
    assert len(lines) == 3                              # header + two games


def test_quiet_prints_only_the_summary(capsys, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["evaluate.py", "--games", "2", "--seed", "5", "--quiet", "--max-pieces", "20"],
    )

    main()

    printed = capsys.readouterr().out
    assert "Game 1/2" not in printed
    assert "random" in printed


@pytest.mark.parametrize(
    "argv",
    [
        ["evaluate.py", "--agent", "clairvoyant"],
        ["evaluate.py", "--games", "0"],
        ["evaluate.py", "--games", "-3"],
        ["evaluate.py", "--max-pieces", "0"],
    ],
)
def test_bad_arguments_exit_with_an_error(argv, monkeypatch):
    monkeypatch.setattr("sys.argv", argv)
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 2
