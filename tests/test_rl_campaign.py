"""Tests for the campaign runner (`tetris/rl/campaign.py`).

A campaign is the thing that makes a training result mean something, so these
tests are about its bookkeeping: the periodic evaluation, the best-checkpoint
rule, the aggregate's arithmetic, and the promise that re-running a seed
replaces its record instead of stacking a second copy on it.
"""

from __future__ import annotations

import json
import pathlib

import pytest

torch = pytest.importorskip("torch")

from tetris.rl.campaign import (  # noqa: E402
    AGGREGATE_NAME,
    BEST_INDEX_NAME,
    CAMPAIGN_NAME,
    METRICS_NAME,
    SUMMARY_NAME,
    BestTracker,
    Campaign,
    SeedRun,
    read_metrics,
    split_metrics,
)
from tetris.rl.config import TrainingConfig  # noqa: E402
from tetris.rl.dqn import DQNAgent  # noqa: E402
from tetris.rl.evaluation import EvaluationSnapshot  # noqa: E402

TINY = dict(episodes=6, max_pieces=8, batch_size=4, min_buffer_size=4, hidden_sizes=(8,),
            target_update_steps=3, checkpoint_every=6, log_every=1, seed=0,
            epsilon_decay_steps=20)


def tiny(**overrides) -> TrainingConfig:
    return TrainingConfig.from_dict({**TINY, **overrides})


def campaign(tmp_path, **overrides) -> Campaign:
    payload = dict(name="T", config=tiny(), seeds=(0,), eval_every=2, eval_games=2,
                   eval_max_pieces=10, output_dir=tmp_path)
    payload.update(overrides)
    return Campaign(**payload)


def snapshot(episode, score=100.0, lines=1.0, pieces=10.0, loss=1.0) -> EvaluationSnapshot:
    return EvaluationSnapshot(
        stats={"score_mean": score, "lines_mean": lines, "pieces_mean": pieces,
               "score_median": score, "lines_per_100_pieces": 10.0, "loss_mean": loss},
        num_games=2, seed=1, max_pieces=10, episode=episode,
    )


# ------------------------------------------------------------------- the files


def test_a_campaign_writes_a_spec_and_an_aggregate(tmp_path):
    result = campaign(tmp_path).run(logger=None)

    directory = tmp_path / "T"
    assert (directory / CAMPAIGN_NAME).exists()
    assert (directory / AGGREGATE_NAME).exists()
    assert (directory / "seed0" / SUMMARY_NAME).exists()
    spec = json.loads((directory / CAMPAIGN_NAME).read_text())
    assert spec["name"] == "T" and spec["seeds"] == [0] and spec["metric"] == "score_mean"
    assert result.aggregate_path.endswith(AGGREGATE_NAME)


def test_the_metrics_stream_holds_training_rows_and_evaluation_rows(tmp_path):
    campaign(tmp_path).run(logger=None)
    rows = read_metrics(tmp_path / "T" / "seed0" / METRICS_NAME)
    training, evaluation = split_metrics(rows)

    assert len(training) == 6                                  # one per episode
    assert [row["episode"] for row in training] == [1, 2, 3, 4, 5, 6]
    for column in ("reward", "score", "lines", "pieces", "epsilon", "loss_mean"):
        assert column in training[0], column

    # episode 0 (untrained), then every 2 episodes and the last one
    assert [row["episode"] for row in evaluation] == [0, 2, 4, 6]
    for row in evaluation:
        for column in ("score_mean", "lines_mean", "pieces_mean", "games", "seed"):
            assert column in row, column


def test_the_untrained_policy_is_measured_before_the_first_episode(tmp_path):
    """"Improvement" has to be measured from the policy the run started with."""
    result = campaign(tmp_path).run(logger=None)
    evaluations = result.runs[0].evaluations

    assert evaluations[0]["episode"] == 0
    assert result.runs[0].first_eval()["episode"] == 0


def test_a_seed_run_is_deterministic(tmp_path):
    first = campaign(tmp_path / "a").run(logger=None)
    second = campaign(tmp_path / "b").run(logger=None)

    def numbers(run):
        training, _ = split_metrics(read_metrics(pathlib.Path(run.directory) / METRICS_NAME))
        return (
            [(row["episode"], row["reward"], row["score"], row["epsilon"]) for row in training],
            [(row["episode"], round(row["score_mean"], 6)) for row in run.evaluations],
        )

    assert numbers(first.runs[0]) == numbers(second.runs[0])


def test_re_running_a_seed_replaces_its_metrics_instead_of_appending(tmp_path):
    campaign(tmp_path).run(logger=None)
    campaign(tmp_path).run(logger=None)
    rows = read_metrics(tmp_path / "T" / "seed0" / METRICS_NAME)

    assert len(rows) == 6 + 4                                  # not doubled


def test_parallel_seeds_produce_the_same_thing_as_sequential_ones(tmp_path):
    sequential = campaign(tmp_path / "one", seeds=(0, 1), eval_every=3).run(parallel=1,
                                                                           logger=None)
    parallel = campaign(tmp_path / "many", seeds=(0, 1), eval_every=3).run(parallel=2,
                                                                          logger=None)

    def scores(result):
        return {run.seed: [round(row["score_mean"], 6) for row in run.evaluations]
                for run in result.runs}

    assert scores(sequential) == scores(parallel)
    assert sequential.aggregate["verdict"] == parallel.aggregate["verdict"]


# ------------------------------------------------------- the best checkpoint


def test_the_best_tracker_writes_a_new_file_for_every_improvement(tmp_path):
    tracker = BestTracker(tmp_path / "best", metric="score_mean")
    agent = DQNAgent(config=tiny(), seed=0)

    first = tracker.consider(snapshot(10, score=100.0), agent)
    worse = tracker.consider(snapshot(20, score=90.0), agent)
    better = tracker.consider(snapshot(30, score=150.0), agent)

    assert first is not None and better is not None
    assert worse is None                                       # not an improvement: no file
    files = sorted(path.name for path in (tmp_path / "best").glob("*.pt"))
    assert files == ["best_ep000010_score_mean_100.000.pt",
                     "best_ep000030_score_mean_150.000.pt"]
    assert first != better                                     # never overwritten


def test_the_best_index_says_which_checkpoint_is_best_and_why(tmp_path):
    tracker = BestTracker(tmp_path / "best", metric="score_mean")
    agent = DQNAgent(config=tiny(), seed=0)
    tracker.consider(snapshot(5, score=100.0), agent)
    tracker.consider(snapshot(10, score=250.0), agent)

    index = json.loads((tmp_path / "best" / BEST_INDEX_NAME).read_text())
    assert index["metric"] == "score_mean" and index["higher_is_better"] is True
    assert index["best"]["episode"] == 10 and index["best"]["value"] == 250.0
    assert [row["episode"] for row in index["improvements"]] == [5, 10]


def test_a_saved_best_checkpoint_carries_the_evaluation_that_made_it_best(tmp_path):
    tracker = BestTracker(tmp_path / "best", metric="score_mean")
    agent = DQNAgent(config=tiny(), seed=0)
    path = tracker.consider(snapshot(7, score=321.0), agent, experiment="T", seed=0)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["best_value"] == 321.0
    assert payload["evaluation"]["score_mean"] == 321.0
    assert payload["experiment"] == "T" and payload["seed"] == 0


def test_a_campaign_keeps_the_best_checkpoint_of_the_unoptimised_agent_too(tmp_path):
    """If training makes the policy worse, the honest "best" is the first one."""
    result = campaign(tmp_path).run(logger=None)
    best = result.runs[0].best

    assert best["best"]["path"]
    assert best["improvements"][0]["episode"] == 0


def test_the_tracker_can_minimise_instead(tmp_path):
    tracker = BestTracker(tmp_path / "best", metric="loss_mean", higher_is_better=False)
    agent = DQNAgent(config=tiny(), seed=0)

    assert tracker.consider(snapshot(1, loss=5.0), agent) is not None
    assert tracker.consider(snapshot(2, loss=6.0), agent) is None
    assert tracker.consider(snapshot(3, loss=4.0), agent) is not None
    assert tracker.best_value == 4.0


# -------------------------------------------------------------- the aggregate


def run_with(evaluations) -> SeedRun:
    """A `SeedRun` built from hand-made evaluations: the aggregate, in isolation."""
    return SeedRun(
        seed=0,
        directory="/nowhere",
        training={"episodes_trained": 10, "best_score": 1, "total_decisions": 10,
                  "total_learn_steps": 5},
        evaluations=list(evaluations),
    )


def evaluation_row(episode, score, lines=0.0, pieces=10.0):
    return {"episode": episode, "score_mean": score, "lines_mean": lines,
            "score_median": score, "pieces_mean": pieces, "lines_per_100_pieces": lines * 10}


def test_the_aggregate_reports_first_against_tail_over_seeds():
    runs = [
        run_with([evaluation_row(0, 10.0), evaluation_row(50, 30.0), evaluation_row(100, 40.0)]),
        run_with([evaluation_row(0, 20.0), evaluation_row(50, 50.0), evaluation_row(100, 60.0)]),
    ]
    aggregate = campaign(pathlib.Path("/tmp/never")).aggregate(runs, window=1)
    trend = aggregate["metrics"]["score_mean"]

    assert trend["first_mean"] == pytest.approx(15.0)
    assert trend["tail_mean"] == pytest.approx(50.0)
    assert trend["delta_mean"] == pytest.approx(35.0)
    assert trend["improved_seeds"] == 2 and trend["n_seeds"] == 2
    assert trend["verdict"] == "improving"
    assert aggregate["verdict"] == "improving"
    assert len(aggregate["per_seed"]) == 2            # two seeds, kept apart


def test_the_tail_is_the_mean_of_the_last_evaluations_not_just_the_last():
    """A run's last evaluation is one row: the tail is what "ended up" should mean."""
    run = run_with([evaluation_row(0, 10.0), evaluation_row(25, 20.0),
                    evaluation_row(50, 30.0), evaluation_row(75, 40.0),
                    evaluation_row(100, 200.0)])

    assert run.tail_eval()["score_mean"] == pytest.approx((30 + 40 + 200) / 3)
    assert run.final_eval()["score_mean"] == 200.0
    assert run.best_eval()["score_mean"] == 200.0
    assert campaign(pathlib.Path("/tmp/never")).aggregate([run], window=3)["metrics"]["score_mean"]["tail_mean"] == pytest.approx(90.0)


def test_a_verdict_is_never_given_without_the_numbers_behind_it():
    runs = [run_with([evaluation_row(0, 10.0), evaluation_row(10, 5.0)])]
    aggregate = campaign(pathlib.Path("/tmp/never")).aggregate(runs, window=1)

    assert aggregate["verdict"] == "flat-or-down"
    assert aggregate["metrics"]["score_mean"]["deltas"] == [-5.0]
    assert aggregate["per_seed"][0]["first"]["score_mean"] == 10.0


def test_one_seed_going_the_other_way_makes_the_verdict_mixed():
    runs = [
        run_with([evaluation_row(0, 10.0), evaluation_row(10, 20.0)]),
        run_with([evaluation_row(0, 10.0), evaluation_row(10, 4.0)]),
    ]
    aggregate = campaign(pathlib.Path("/tmp/never")).aggregate(runs, window=1)

    assert aggregate["metrics"]["score_mean"]["delta_mean"] == pytest.approx(2.0)
    assert aggregate["metrics"]["score_mean"]["improved_seeds"] == 1
    assert aggregate["metrics"]["score_mean"]["delta_sd"] == pytest.approx(11.31, abs=0.01)
    assert aggregate["verdict"] == "mixed"


def test_a_run_with_no_evaluations_does_not_break_the_aggregate():
    aggregate = campaign(pathlib.Path("/tmp/never")).aggregate([run_with([])])

    assert aggregate["metrics"]["score_mean"]["first_mean"] == 0.0
    assert aggregate["verdict"] == "flat-or-down"


def test_the_campaign_describes_itself():
    text = campaign(pathlib.Path("/tmp/never")).describe()
    assert "6 episodes" in text and "1 seed" in text and "best by score_mean" in text


# ------------------------------------------------------------------ validation


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"name": ""}, "needs a name"),
        ({"seeds": ()}, "at least one seed"),
        ({"seeds": ("zero",)}, "whole numbers"),
        ({"eval_every": 0}, "eval_every"),
        ({"eval_games": 0}, "eval_games"),
        ({"metric": "reward"}, "metric must be one of"),
    ],
)
def test_a_bad_campaign_is_rejected(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        campaign(tmp_path, **overrides)


def test_the_spec_round_trips():
    original = campaign(pathlib.Path("experiments/rl"), seeds=(1, 2), variant="double-dqn")
    rebuilt = Campaign.from_spec(json.loads(json.dumps(original.spec())))

    assert rebuilt.name == original.name
    assert rebuilt.seeds == original.seeds
    assert rebuilt.metric == original.metric
    assert rebuilt.variant == original.variant
    assert rebuilt.eval_games == original.eval_games
    assert rebuilt.config == original.config


# ------------------------------------------------------------- one run at a time


def test_a_campaign_refuses_to_run_while_another_one_holds_its_directory(tmp_path):
    """Two runs appending to one metrics.jsonl make a record of neither run."""
    import os

    from tetris.rl.campaign import LOCK_NAME

    running = campaign(tmp_path)
    lock = running.directory / LOCK_NAME
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text(f"{os.getpid()} 2026-09-28T00:00:00+00:00\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="already running"):
        campaign(tmp_path).run(logger=None)

    assert not list(running.directory.glob(f"seed*/{METRICS_NAME}")), \
        "the refused run must not have written any metrics"
    lock.unlink()


def test_a_lock_left_by_a_dead_run_is_taken_over(tmp_path):
    from tetris.rl.campaign import LOCK_NAME

    stale = campaign(tmp_path)
    lock = stale.directory / LOCK_NAME
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("999999 2026-09-28T00:00:00+00:00\n", encoding="utf-8")   # nobody

    run = campaign(tmp_path).run(logger=None)

    assert len(run.runs) == len(campaign(tmp_path).seeds)
    assert not lock.exists(), "the lock is released when the run finishes"


def test_the_lock_is_released_after_a_run(tmp_path):
    from tetris.rl.campaign import LOCK_NAME

    campaign(tmp_path).run(logger=None)

    assert not (campaign(tmp_path).directory / LOCK_NAME).exists()
