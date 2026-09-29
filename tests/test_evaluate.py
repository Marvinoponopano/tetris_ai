"""Tests for the evaluation harness (`tetris.evaluate`).

The important properties are: the numbers describe the games that were played,
a seed makes a whole run reproducible (even when the agent object is reused),
every agent sees the same piece sequences, and nothing here needs a display.
"""

import json
import csv
import pathlib
import statistics

import pytest

from tetris.agents import RandomAgent
from tetris.evaluate import (
    DEFAULT_MAX_PIECES,
    GAME_FIELDS,
    EvaluationResult,
    GameResult,
    evaluate_agent,
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


class ScriptedAgent:
    """Always takes the first legal action, and records what it was shown."""

    name = "scripted"

    def __init__(self):
        self.resets = []
        self.queues = []
        self.states = 0

    def reset(self, seed=None):
        self.resets.append(seed)

    def choose_action(self, state, valid_actions):
        self.states += 1
        self.queues.append(tuple(int(piece) for piece in state.queue))
        return valid_actions[0]


class LastAgent(ScriptedAgent):
    """A different policy on the same games: always the last legal action."""

    name = "last"

    def choose_action(self, state, valid_actions):
        self.states += 1
        self.queues.append(tuple(int(piece) for piece in state.queue))
        return valid_actions[-1]


class StatelessAgent:
    """No `reset` at all -- the harness must cope without one."""

    name = "stateless"

    def choose_action(self, state, valid_actions):
        return valid_actions[0]


# ------------------------------------------------------------------- basics


def test_evaluation_returns_one_result_per_game():
    result = evaluate_agent(RandomAgent(seed=0), num_games=8, seed=1, logger=None)

    assert isinstance(result, EvaluationResult)
    assert result.num_games == 8
    assert result.seed == 1
    assert len(result.games) == 8
    assert all(isinstance(game, GameResult) for game in result.games)
    assert [game.game for game in result.games] == list(range(8))
    assert result.elapsed_seconds > 0


def test_the_result_reports_the_agent_name():
    assert evaluate_agent(RandomAgent(), num_games=2, logger=None).agent == "random"
    assert evaluate_agent(ScriptedAgent(), num_games=2, logger=None).agent == "scripted"


def test_every_game_is_played_to_the_end():
    result = evaluate_agent(RandomAgent(seed=0), num_games=5, seed=2, logger=None)

    assert all(game.pieces > 0 for game in result.games)
    assert all(game.score > 0 for game in result.games)
    assert not any(game.capped for game in result.games)
    assert result.stats["games_capped"] == 0


def test_an_agent_without_reset_is_fine():
    result = evaluate_agent(StatelessAgent(), num_games=3, seed=3, logger=None)
    assert len(result.games) == 3


# ------------------------------------------------------------ result maths


def test_stats_are_computed_from_the_games_that_were_played():
    result = evaluate_agent(RandomAgent(seed=0), num_games=20, seed=4, logger=None)
    scores = [game.score for game in result.games]
    lines = [game.lines_cleared for game in result.games]
    pieces = [game.pieces for game in result.games]
    stats = result.stats

    assert stats["games"] == 20
    assert stats["score_mean"] == pytest.approx(statistics.fmean(scores))
    assert stats["score_median"] == pytest.approx(statistics.median(scores))
    assert stats["score_min"] == min(scores)
    assert stats["score_max"] == max(scores)
    assert stats["score_stdev"] == pytest.approx(statistics.stdev(scores))
    assert stats["lines_mean"] == pytest.approx(statistics.fmean(lines))
    assert stats["lines_max"] == max(lines)
    assert stats["lines_total"] == sum(lines)
    assert stats["pieces_mean"] == pytest.approx(statistics.fmean(pieces))
    assert stats["pieces_max"] == max(pieces)


def test_stats_include_the_requested_figures():
    stats = evaluate_agent(RandomAgent(seed=0), num_games=5, seed=5, logger=None).stats
    assert {
        "score_mean",          # average score
        "score_median",        # median score
        "score_max",           # maximum score
        "score_min",           # minimum score
        "lines_mean",          # average lines cleared
        "lines_max",           # maximum lines cleared
        "pieces_mean",         # average pieces survived
        "game_length_mean",    # game length
    } <= set(stats)


def test_game_length_is_the_piece_count_here():
    # One action places one piece, so the two figures coincide by construction.
    stats = evaluate_agent(RandomAgent(seed=0), num_games=6, seed=6, logger=None).stats
    assert stats["game_length_mean"] == stats["pieces_mean"]
    assert stats["game_length_max"] == stats["pieces_max"]


def test_the_summary_mentions_the_key_figures():
    result = evaluate_agent(RandomAgent(seed=0), num_games=4, seed=7, logger=None)
    text = result.summary()

    assert "random" in text
    assert "score" in text and "lines cleared" in text and "pieces" in text
    assert "game length" in text
    assert str(int(result.stats["score_mean"]))[:1] in text
    assert str(result) == text


def test_a_single_game_has_a_zero_spread():
    stats = evaluate_agent(RandomAgent(seed=0), num_games=1, seed=8, logger=None).stats
    assert stats["score_stdev"] == 0.0


# --------------------------------------------------------------- determinism


def games_of(result: EvaluationResult):
    return [(g.game, g.seed, g.score, g.lines_cleared, g.pieces) for g in result.games]


def test_the_same_seed_gives_the_same_numbers():
    first = evaluate_agent(RandomAgent(seed=0), num_games=10, seed=123, logger=None)
    second = evaluate_agent(RandomAgent(seed=0), num_games=10, seed=123, logger=None)

    assert games_of(first) == games_of(second)
    assert first.stats == second.stats


def test_reusing_the_same_agent_object_is_still_reproducible():
    # The harness resets the agent before each game, so an agent that has
    # already played does not carry its random stream into the next run.
    agent = RandomAgent(seed=0)
    first = evaluate_agent(agent, num_games=6, seed=99, logger=None)
    second = evaluate_agent(agent, num_games=6, seed=99, logger=None)

    assert games_of(first) == games_of(second)


def test_the_agent_is_reset_before_every_game_with_that_game_seed():
    agent = ScriptedAgent()
    evaluate_agent(agent, num_games=4, seed=50, logger=None)
    assert agent.resets == [50, 51, 52, 53]


def test_an_unseeded_run_does_not_seed_the_agent():
    agent = ScriptedAgent()
    result = evaluate_agent(agent, num_games=3, logger=None)
    assert agent.resets == [None, None, None]
    assert all(game.seed is None for game in result.games)
    assert result.seed is None


def test_different_seeds_play_different_games():
    first = evaluate_agent(RandomAgent(seed=0), num_games=10, seed=1, logger=None)
    second = evaluate_agent(RandomAgent(seed=0), num_games=10, seed=2, logger=None)
    assert games_of(first) != games_of(second)


def test_different_agents_face_the_same_piece_sequences():
    first, second = ScriptedAgent(), LastAgent()
    evaluate_agent(first, num_games=1, seed=77, logger=None)
    evaluate_agent(second, num_games=1, seed=77, logger=None)

    # The two agents die at different points, so compare the part of the game
    # they both lived through: same seed means the same pieces, in the same
    # order, whatever either of them decided to do with them.
    shared = min(len(first.queues), len(second.queues))
    assert shared > 5
    assert first.queues[0] == second.queues[0]
    assert first.queues[:shared] == second.queues[:shared]


def test_the_game_seeds_run_from_the_given_seed():
    result = evaluate_agent(RandomAgent(seed=0), num_games=5, seed=1000, logger=None)
    assert [game.seed for game in result.games] == [1000, 1001, 1002, 1003, 1004]


# ------------------------------------------------------------------ logging


def test_progress_is_logged_about_every_tenth_game():
    lines = []
    evaluate_agent(RandomAgent(seed=0), num_games=100, seed=9, logger=lines.append)
    assert lines == [f"Game {game}/100" for game in range(10, 101, 10)]


def test_progress_logging_includes_the_last_game_when_it_is_not_a_multiple():
    lines = []
    evaluate_agent(ScriptedAgent(), num_games=25, logger=lines.append)
    assert lines == [f"Game {game}/25" for game in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25]]


def test_log_every_can_be_chosen():
    lines = []
    evaluate_agent(ScriptedAgent(), num_games=6, log_every=3, logger=lines.append)
    assert lines == ["Game 3/6", "Game 6/6"]


def test_logging_can_be_switched_off(capsys):
    evaluate_agent(RandomAgent(seed=0), num_games=5, seed=10, logger=None)
    assert capsys.readouterr().out == ""


def test_evaluating_does_not_print_unless_asked(capsys):
    evaluate_agent(RandomAgent(seed=0), num_games=3, seed=11, logger=None)
    result = evaluate_agent(RandomAgent(seed=0), num_games=3, seed=11, logger=None)
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "random" in result.summary()      # only printing shows anything


# ------------------------------------------------------------------ the cap


def test_max_pieces_stops_a_game_and_flags_it():
    result = evaluate_agent(
        RandomAgent(seed=0), num_games=4, seed=12, max_pieces=5, logger=None
    )
    assert all(game.pieces == 5 for game in result.games)
    assert all(game.capped for game in result.games)
    assert result.stats["games_capped"] == 4
    assert result.max_pieces == 5


def test_a_game_that_ends_by_itself_is_not_capped():
    result = evaluate_agent(RandomAgent(seed=0), num_games=5, seed=13, logger=None)
    assert not any(game.capped for game in result.games)
    assert all(game.pieces < DEFAULT_MAX_PIECES for game in result.games)


def test_bad_arguments_are_rejected():
    agent = RandomAgent(seed=0)
    for kwargs in ({"num_games": 0}, {"num_games": -3}, {"num_games": 1, "max_pieces": 0},
                   {"num_games": 1, "log_every": 0}):
        with pytest.raises(ValueError):
            evaluate_agent(agent, logger=None, **kwargs)


# -------------------------------------------------------------------- saving


def test_the_result_serialises_to_a_plain_dict():
    result = evaluate_agent(RandomAgent(seed=0), num_games=3, seed=14, logger=None)
    data = result.as_dict()

    assert data["agent"] == "random"
    assert data["num_games"] == 3 and data["seed"] == 14
    assert data["stats"]["games"] == 3
    assert len(data["games"]) == 3
    assert set(data["games"][0]) == set(GAME_FIELDS)
    assert len(json.dumps(data)) > 100          # JSON-serialisable, no numpy types


def test_saving_json_round_trips(tmp_path):
    result = evaluate_agent(RandomAgent(seed=0), num_games=4, seed=15, logger=None)
    path = tmp_path / "results.json"

    assert result.save(path) == str(path)
    data = json.loads(path.read_text())

    assert data["stats"]["score_mean"] == pytest.approx(result.stats["score_mean"])
    assert [game["score"] for game in data["games"]] == [g.score for g in result.games]


def test_saving_csv_writes_a_header_and_one_row_per_game(tmp_path):
    result = evaluate_agent(RandomAgent(seed=0), num_games=5, seed=16, logger=None)
    path = tmp_path / "results.csv"
    result.save(path)

    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert list(rows[0]) == list(GAME_FIELDS)
    assert len(rows) == 5
    assert [int(row["score"]) for row in rows] == [game.score for game in result.games]
    assert [int(row["game"]) for row in rows] == [0, 1, 2, 3, 4]
    assert rows[0]["seed"] == "16"


def test_saving_rejects_an_unknown_format(tmp_path):
    result = evaluate_agent(RandomAgent(seed=0), num_games=1, seed=17, logger=None)
    with pytest.raises(ValueError, match="use .json or .csv"):
        result.save(tmp_path / "results.txt")


def test_the_cap_is_recorded_in_the_saved_result(tmp_path):
    result = evaluate_agent(RandomAgent(seed=0), num_games=2, seed=18, max_pieces=4, logger=None)
    path = tmp_path / "capped.json"
    result.save(path)
    data = json.loads(path.read_text())

    assert data["max_pieces"] == 4
    assert all(game["capped"] for game in data["games"])
