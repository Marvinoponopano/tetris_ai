"""Tests for the baseline comparison (`tetris/rl/compare.py`).

A comparison is only worth something if it is paired: the same games, per game.
So what is pinned here is the pairing arithmetic, the sign test, and that the
whole thing runs through the project's own experiment framework.
"""

from __future__ import annotations

import json
import math
import pathlib
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tetris.evaluate import GameResult  # noqa: E402
from tetris.rl.compare import (  # noqa: E402
    ComparisonProtocol,
    best_checkpoints,
    checkpoint_facts,
    compare_agents,
    format_comparison,
    paired_comparison,
    sign_test_p,
)


def games(scores, agent="agent", lines=None, pieces=None) -> SimpleNamespace:
    """A stand-in for an `ExperimentResult`: only the fields used here."""
    lines = lines or [0] * len(scores)
    pieces = pieces or [10] * len(scores)
    return SimpleNamespace(
        agent=agent,
        games=tuple(
            GameResult(game=index, seed=100 + index, score=score, lines_cleared=line,
                       pieces=piece, capped=False)
            for index, (score, line, piece) in enumerate(zip(scores, lines, pieces))
        ),
    )


# ------------------------------------------------------------- the sign test


@pytest.mark.parametrize(
    "wins, losses, expected",
    [
        (0, 0, None),
        (1, 0, 1.0),                     # one game proves nothing
        (5, 0, 2 * (1 / 32)),            # all five, two-sided
        (10, 0, 2 / 1024),
        (5, 5, 1.0),                     # a perfect coin
        (3, 1, 0.625),
    ],
)
def test_the_sign_test_is_the_exact_two_sided_binomial(wins, losses, expected):
    result = sign_test_p(wins, losses)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected, abs=1e-9)
        assert 0.0 <= result <= 1.0


def test_the_sign_test_never_exceeds_one():
    for wins in range(0, 21):
        for losses in range(0, 21):
            value = sign_test_p(wins, losses)
            assert value is None or value <= 1.0


# -------------------------------------------------------------- the pairing


def test_paired_comparison_matches_games_by_index():
    candidate = games([100, 50, 300], agent="dqn")
    baseline = games([10, 100, 100], agent="random")

    paired = paired_comparison(candidate, baseline)

    assert paired["games"] == 3
    assert paired["score_delta_mean"] == pytest.approx((90 - 50 + 200) / 3)
    assert paired["score_wins"] == 2 and paired["score_losses"] == 1
    assert paired["baseline"] == "random"


def test_paired_comparison_uses_only_the_games_both_agents_played():
    candidate = games([100, 200])
    baseline = SimpleNamespace(
        agent="random",
        games=tuple(GameResult(game=index, seed=index, score=0, lines_cleared=0, pieces=1,
                              capped=False) for index in (0, 1, 2, 3)),
    )

    paired = paired_comparison(candidate, baseline)

    assert paired["games"] == 2
    assert paired["score_delta_mean"] == pytest.approx(150.0)


def test_paired_comparison_reports_lines_and_pieces_too():
    candidate = games([100], lines=[4], pieces=[40])
    baseline = games([10], lines=[1], pieces=[25])

    paired = paired_comparison(candidate, baseline)

    assert paired["lines_delta_mean"] == pytest.approx(3.0)
    assert paired["pieces_delta_mean"] == pytest.approx(15.0)


# --------------------------------------------------------------- the tables


def test_the_comparison_table_shows_the_paired_numbers():
    rows = [
        {"label": "dqn-B-seed0", "agent": "dqn",
         "stats": {"games": 20, "score_mean": 900.0, "score_stdev": 100.0,
                   "score_median": 850.0, "lines_mean": 1.5, "pieces_mean": 60.0},
         "diagnostics": {"lines_per_100_pieces": 2.5},
         "paired_vs_primary": {"score_delta_mean": 500.0, "score_wins": 15,
                               "score_losses": 5, "sign_test_p": 0.041}},
        {"label": "random", "agent": "random",
         "stats": {"games": 20, "score_mean": 400.0, "score_stdev": 50.0,
                   "score_median": 390.0, "lines_mean": 0.0, "pieces_mean": 18.0},
         "diagnostics": {"lines_per_100_pieces": 0.0}, "paired_vs_primary": None},
    ]
    text = format_comparison(rows)

    assert "score mean" in text and "d_score" in text
    assert "900.0" in text and "500.0" in text and "15/5" in text and "0.041" in text
    assert "random" in text


def test_best_checkpoints_reads_the_runs_own_index(tmp_path):
    for seed, value in ((0, 100.0), (1, 200.0)):
        directory = tmp_path / f"seed{seed}" / "best"
        directory.mkdir(parents=True)
        checkpoint = directory / f"best_ep000100_score_mean_{value:.3f}.pt"
        checkpoint.write_bytes(b"not really a checkpoint")
        (directory / "best.json").write_text(
            json.dumps({"metric": "score_mean",
                        "best": {"path": str(checkpoint), "value": value}})
        )

    found = best_checkpoints(tmp_path, prefix="A")

    assert sorted(found) == ["A-seed0", "A-seed1"]
    assert found["A-seed0"].name == "best_ep000100_score_mean_100.000.pt"
    assert found["A-seed1"].name == "best_ep000100_score_mean_200.000.pt"


def test_checkpoint_facts_survive_a_file_that_is_not_a_checkpoint(tmp_path):
    path = tmp_path / "broken.pt"
    path.write_bytes(b"nonsense")

    assert checkpoint_facts(path) == {}


# ------------------------------------------------------- the whole comparison


def test_a_comparison_runs_the_baselines_and_writes_it_down(tmp_path):
    protocol = ComparisonProtocol(num_games=2, seed=5, max_pieces=6)
    payload = compare_agents(
        checkpoints=None,
        protocol=protocol,
        output_dir=tmp_path,
        baselines=("random", "heuristic"),
        logger=None,
    )

    assert (tmp_path / "comparison.json").exists()
    assert (tmp_path / "comparison.csv").exists()
    assert set(payload["agents"]) == {"random", "heuristic"}
    for entry in payload["agents"].values():
        assert len(entry["games"]) == 2
    assert payload["agents"]["heuristic"]["paired_vs_primary"]["games"] == 2
    assert payload["protocol"] == {"num_games": 2, "seed": 5, "max_pieces": 6}
    assert "score mean" in payload["table"]


def test_a_trained_checkpoint_can_be_compared(tmp_path):
    from tetris.rl.config import TrainingConfig
    from tetris.rl.dqn import DQNAgent

    agent = DQNAgent(config=TrainingConfig.from_dict(
        {"episodes": 1, "max_pieces": 4, "batch_size": 2, "hidden_sizes": (8,), "seed": 0}
    ), seed=0)
    checkpoint = agent.save(tmp_path / "checkpoint_1.pt")

    payload = compare_agents(
        checkpoints={"dqn-test": checkpoint},
        protocol=ComparisonProtocol(num_games=2, seed=5, max_pieces=6),
        output_dir=tmp_path / "out",
        baselines=("random",),
        logger=None,
    )

    assert "dqn-test" in payload["agents"]
    entry = payload["agents"]["dqn-test"]
    assert entry["agent"] == "dqn"
    assert entry["checkpoint_facts"]["episode"] == 0
    assert payload["primary_baseline"] == "random"


def test_a_missing_primary_baseline_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="primary baseline"):
        compare_agents(
            checkpoints=None,
            protocol=ComparisonProtocol(num_games=1, seed=1, max_pieces=3),
            output_dir=tmp_path,
            baselines=("heuristic",),
            primary_baseline="random",
            logger=None,
        )
