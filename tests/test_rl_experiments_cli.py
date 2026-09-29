"""Tests for the experiment plan and its command line.

`experiments/rl_plan.py` is the definition of what "experiment A/B/C/D" means, so
a broken plan has to fail here rather than after an hour of training. The CLI is
loaded from its path (like `experiments/run.py`) because `experiments/` is a
folder of scripts, not a package.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

torch = pytest.importorskip("torch")

ROOT = pathlib.Path(__file__).resolve().parent.parent
HERE = ROOT / "experiments"


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plan = load("rl_plan", "rl_plan.py")
cli = load("rl_experiments", "rl_experiments.py")


# ------------------------------------------------------------------- the plan


def test_the_plan_grows_from_a_to_c():
    sizes = [plan.EXPERIMENTS[name]["episodes"] for name in ("A", "B", "C")]
    assert sizes == sorted(sizes) and sizes[0] < sizes[-1]


def test_the_exploration_schedule_fits_inside_each_experiment():
    """A run whose epsilon never anneals measures a partly random agent."""
    for name, spec in plan.EXPERIMENTS.items():
        config = plan.training_config(name)
        assert config.epsilon_decay_steps <= config.episodes * config.max_pieces, name


def test_every_experiment_builds_a_valid_campaign():
    from tetris.rl.variants import VARIANTS

    for name in plan.EXPERIMENTS:
        campaign = plan.campaign(name)
        assert campaign.name == name.upper()
        assert campaign.seeds and len(campaign.seeds) >= 3
        assert campaign.eval_every >= 1 and campaign.eval_games >= 1
        assert campaign.metric == "score_mean"
        assert campaign.variant is None or campaign.variant in VARIANTS
        assert campaign.config.episodes == plan.EXPERIMENTS[name]["episodes"]


def test_the_plan_can_be_shrunk_for_a_quick_run():
    campaign = plan.campaign("B", seeds=[0], episodes=3, eval_every=1, eval_games=1)

    assert campaign.seeds == (0,)
    assert campaign.config.episodes == 3
    assert campaign.eval_every == 1


def test_an_unknown_experiment_is_rejected():
    with pytest.raises(KeyError, match="unknown experiment"):
        plan.campaign("Z")


def test_the_plan_table_shows_every_experiment():
    text = plan.table()

    for name in plan.EXPERIMENTS:
        assert f"{name}" in text
    assert "episodes" in text and "variant" in text and "purpose" in text
    assert "reward" in text


def test_the_targeted_experiments_change_one_knob_each():
    """D and E are only worth running if each is the control with *one* thing moved."""
    # `to_result_signature` is the part that can change a number (not the label).
    control = plan.training_config("C").to_result_signature()

    # D: the same training config, a different agent class.
    assert plan.training_config("D").to_result_signature() == control
    assert plan.EXPERIMENTS["D"]["variant"] != plan.EXPERIMENTS["C"]["variant"]

    # E: the same agent class, one reward weight added.
    changed = [name for name, value in plan.training_config("E").to_result_signature().items()
               if value != control[name]]
    assert changed == ["reward"]
    assert plan.EXPERIMENTS["E"]["variant"] == plan.EXPERIMENTS["C"]["variant"]
    assert plan.training_config("E").reward_config().per_piece > 0

    # F: one knob again, the exploration schedule.
    changed_f = [name for name, value in plan.training_config("F").to_result_signature().items()
                 if value != control[name]]
    assert changed_f == ["epsilon_decay_steps"]
    assert plan.training_config("F").epsilon_decay_steps < plan.training_config("C").epsilon_decay_steps

    # G: the two fixes together -- the one arm that is deliberately not single-knob.
    changed_g = sorted(name for name, value in plan.training_config("G").to_result_signature().items()
                       if value != control[name])
    assert changed_g == ["epsilon_decay_steps", "reward"]


ROUND_TWO = ("I", "J", "K", "L", "N")


def test_each_round_two_arm_is_g_with_one_thing_moved():
    """H..N are only worth running if each is G with *one* thing moved.

    The control for this round is G, not C: G is the best agent the project has,
    so an arm that changes G in two ways answers two questions at once and
    answers neither.
    """
    from tetris.rl.variants import VARIANTS

    control = plan.training_config("G").to_result_signature()
    control_variant = plan.EXPERIMENTS["G"]["variant"]

    # H: the reward ratio, and nothing else. The other weights are untouched, so
    # the arm tests "lines are worth more than the survival they cost" rather
    # than "reward was changed".
    changed = [name for name, value in plan.training_config("H").to_result_signature().items()
               if value != control[name]]
    assert changed == ["reward"]
    assert plan.EXPERIMENTS["H"]["variant"] == control_variant

    before = plan.training_config("G").reward_config()
    after = plan.training_config("H").reward_config()
    assert after.per_line > before.per_line and after.per_piece < before.per_piece
    assert (after.hole, after.height, after.game_over) == (
        before.hole, before.height, before.game_over
    )

    # I..N: the same training config with a different agent -- the same shape of
    # change D had against C.
    for name in ROUND_TWO:
        assert plan.training_config(name).to_result_signature() == control, name
        assert plan.EXPERIMENTS[name]["variant"] != control_variant, name
        assert plan.EXPERIMENTS[name]["variant"] in VARIANTS, name


def test_every_round_two_arm_is_the_same_size_as_its_control():
    """Same episodes, seeds and evaluation schedule, or the rows are not comparable."""
    control = plan.campaign("G")
    for name in ("H",) + ROUND_TWO:
        arm = plan.campaign(name)
        assert arm.config.episodes == control.config.episodes, name
        assert arm.seeds == control.seeds, name
        assert arm.eval_every == control.eval_every, name
        assert arm.eval_games == control.eval_games, name
        assert arm.eval_seed == control.eval_seed, name
        assert arm.eval_max_pieces == control.eval_max_pieces, name


def test_the_round_two_arms_are_the_changes_the_diagnosis_asked_for():
    """Each variant the plan names must exist, and be the one it claims."""
    from tetris.rl.variants import VARIANTS

    expected = {
        "I": "n-step",
        "J": "feature-planes",
        "K": "dueling",
        "L": "prioritized",
        "N": "queue-planes",
    }
    for name, variant in expected.items():
        assert plan.EXPERIMENTS[name]["variant"] == variant
        assert variant in VARIANTS


def test_every_experiment_shares_the_evaluation_protocol():
    """A curve and the comparison table have to be the same measurement."""
    seeds = {plan.campaign(name).eval_seed for name in plan.EXPERIMENTS}
    caps = {plan.campaign(name).eval_max_pieces for name in plan.EXPERIMENTS}

    assert len(seeds) == 1 and len(caps) == 1


# -------------------------------------------------------------------- the CLI


def test_the_cli_prints_the_plan(capsys):
    assert cli.main(["plan"]) == 0

    printed = capsys.readouterr().out
    assert "planned experiments" in printed
    assert "A" in printed and "D" in printed


def test_the_cli_says_when_there_is_nothing_to_aggregate(tmp_path, capsys):
    assert cli.main(["--data-dir", str(tmp_path), "aggregate"]) == 0

    assert "no aggregate.json yet" in capsys.readouterr().err


def test_the_cli_runs_a_shrunk_experiment_and_aggregates_it(tmp_path, capsys):
    code = cli.main([
        "--data-dir", str(tmp_path), "run", "--experiment", "A",
        "--seeds", "0", "--episodes", "2", "--eval-every", "1", "--eval-games", "1",
    ])

    assert code == 0
    directory = tmp_path / "A" / "seed0"
    assert (directory / "metrics.jsonl").exists()
    assert (directory / "best" / "best.json").exists()
    assert json.loads((tmp_path / "A" / "aggregate.json").read_text())["experiment"] == "A"

    assert cli.main(["--data-dir", str(tmp_path), "aggregate", "--experiment", "A"]) == 0
    printed = capsys.readouterr().out
    assert "first(ep0)" in printed and "verdict" in printed


def test_the_cli_diagnoses_an_untrained_agent(capsys):
    code = cli.main(["diagnose", "--untrained", "--games", "1", "--max-pieces", "6",
                     "--lookahead-decisions", "0"])

    assert code == 0
    printed = capsys.readouterr().out
    assert "diagnosis: untrained" in printed
    assert "where the episode reward actually comes from" in printed


def test_the_cli_diagnoses_a_checkpoint_and_saves_it(tmp_path, capsys):
    from tetris.rl.config import TrainingConfig
    from tetris.rl.dqn import DQNAgent

    agent = DQNAgent(config=TrainingConfig.from_dict(
        {"episodes": 1, "max_pieces": 4, "batch_size": 2, "hidden_sizes": (8,), "seed": 0}
    ), seed=0)
    checkpoint = agent.save(tmp_path / "checkpoint_1.pt")
    target = tmp_path / "diagnosis.json"

    code = cli.main(["diagnose", "--checkpoint", str(checkpoint), "--games", "1",
                     "--max-pieces", "6", "--lookahead-decisions", "0", "--save", str(target)])

    assert code == 0
    saved = json.loads(target.read_text())
    assert saved["checkpoint"] == str(checkpoint)
    assert "diagnosis: checkpoint_1" in capsys.readouterr().out


def test_the_cli_needs_a_checkpoint_to_diagnose(capsys):
    assert cli.main(["diagnose"]) == 2
    assert "pass --checkpoint" in capsys.readouterr().err


def test_the_cli_compares_the_baselines_alone(tmp_path, capsys):
    code = cli.main(["compare", "--baselines", "random", "heuristic", "--baselines-only",
                     "--games", "2", "--seed", "5", "--max-pieces", "6",
                     "--output-dir", str(tmp_path)])

    assert code == 0
    printed = capsys.readouterr().out
    assert "score mean" in printed and "d_score" in printed
    payload = json.loads((tmp_path / "comparison.json").read_text())
    assert set(payload["agents"]) == {"random", "heuristic"}
    assert payload["protocol"]["seed"] == 5


def test_the_cli_plots_a_finished_campaign(tmp_path, capsys):
    cli.main(["--data-dir", str(tmp_path), "run", "--experiment", "A", "--seeds", "0",
              "--episodes", "2", "--eval-every", "1", "--eval-games", "1"])
    capsys.readouterr()

    code = cli.main(["--data-dir", str(tmp_path), "plot", "--experiment", "A", "--window", "2"])

    assert code == 0
    assert "01_training_reward.png" in capsys.readouterr().out
    assert (tmp_path / "A" / "plots" / "04_loss.png").exists()


def test_the_cli_needs_to_be_told_what_to_plot(capsys):
    assert cli.main(["plot"]) == 2
    assert "nothing to plot" in capsys.readouterr().err


def test_the_cli_reports_what_it_has(tmp_path, capsys):
    cli.main(["--data-dir", str(tmp_path), "run", "--experiment", "A", "--seeds", "0",
              "--episodes", "2", "--eval-every", "1", "--eval-games", "1"])
    capsys.readouterr()

    assert cli.main(["--data-dir", str(tmp_path), "report"]) == 0

    printed = capsys.readouterr().out
    assert "trailing mean" in printed or "01_training_reward.png" in printed
    assert "verdict" in printed


def test_the_cli_says_what_a_comparison_needs(tmp_path, capsys):
    assert cli.main(["--data-dir", str(tmp_path), "compare"]) == 2
    assert "no checkpoints to compare" in capsys.readouterr().err


def test_a_failing_command_exits_nonzero_with_a_message(tmp_path, capsys):
    code = cli.main(["diagnose", "--checkpoint", str(tmp_path / "missing.pt"),
                     "--games", "1", "--max-pieces", "4"])

    assert code == 1
    assert "failed" in capsys.readouterr().err


def test_the_aggregate_table_prints_the_trends_and_the_other_metrics():
    documents = [
        {
            "experiment": "B", "episodes": 2000, "seeds": [0, 1], "metric": "score_mean",
            "metrics": {
                "score_mean": {"first_mean": 400.0, "tail_mean": 500.0, "delta_mean": 100.0,
                               "delta_sd": 10.0, "improved_seeds": 2, "n_seeds": 2,
                               "verdict": "improving"},
                "lines_mean": {"first_mean": 0.0, "tail_mean": 1.0, "delta_mean": 1.0,
                               "improved_seeds": 2, "n_seeds": 2},
            },
        }
    ]
    text = cli.aggregate_table(documents)

    assert "first(ep0)" in text and "improving" in text
    assert "lines_mean" in text and "1.000" in text
