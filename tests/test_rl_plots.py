"""Tests for the figures (`tetris/rl/plots.py`).

The figures are the part of a report nobody re-checks, so two properties are
pinned here: the smoothing is causal (a point can never be influenced by a
measurement that had not happened yet), and every figure is written with the
raw measurements in it -- the smoothed line is drawn on top, never instead.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

torch = pytest.importorskip("torch")
matplotlib = pytest.importorskip("matplotlib")

from tetris.rl.campaign import Campaign  # noqa: E402
from tetris.rl.config import TrainingConfig  # noqa: E402
from tetris.rl.plots import (  # noqa: E402
    FIGURE_NAMES,
    load_campaign_metrics,
    plot_campaign,
    plot_comparison,
    trailing_mean,
)


def tiny_campaign(tmp_path, seeds=(0,)) -> pathlib.Path:
    config = TrainingConfig.from_dict(
        {"episodes": 6, "max_pieces": 6, "batch_size": 4, "min_buffer_size": 4,
         "hidden_sizes": (8,), "target_update_steps": 3, "checkpoint_every": 6,
         "log_every": 6, "seed": 0, "epsilon_decay_steps": 20}
    )
    campaign = Campaign(name="T", config=config, seeds=seeds, eval_every=3, eval_games=2,
                        eval_max_pieces=8, output_dir=tmp_path)
    campaign.run(parallel=1, logger=None)
    return tmp_path / "T"


# ----------------------------------------------------------- the smoothing


def test_a_window_of_one_changes_nothing():
    values = [3.0, 1.0, 4.0, 1.0, 5.0]
    assert list(trailing_mean(values, 1)) == values


def test_the_trailing_mean_averages_only_the_past():
    values = [1.0, 2.0, 3.0, 4.0]
    smoothed = trailing_mean(values, 2)

    assert list(smoothed) == [1.0, 1.5, 2.5, 3.5]      # never looks forward


def test_changing_a_late_measurement_never_changes_an_earlier_smoothed_value():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    before = trailing_mean(values, 3)
    after = trailing_mean(values[:3] + [99.0, 99.0], 3)

    assert list(before[:3]) == list(after[:3])
    assert before[-1] != after[-1]


def test_the_first_points_are_average_of_what_exists():
    assert list(trailing_mean([2.0, 4.0, 8.0], 10)) == [2.0, 3.0, 14.0 / 3]


def test_empty_input_is_empty_output():
    assert trailing_mean([], 5).size == 0


# ------------------------------------------------------------- the metrics


def test_the_metrics_are_read_back_split_by_kind(tmp_path):
    directory = tiny_campaign(tmp_path)
    training, evaluation = load_campaign_metrics(directory)

    assert set(training) == {0} and set(evaluation) == {0}
    assert len(training[0]) == 6
    assert [row["episode"] for row in evaluation[0]] == [0, 3, 6]
    assert all(row["kind"] == "evaluation" for row in evaluation[0])


# ------------------------------------------------------------- the figures


def test_a_campaign_gets_the_four_figures(tmp_path):
    directory = tiny_campaign(tmp_path, seeds=(0, 1))
    written = plot_campaign(directory, window=3, title="T")

    assert len(written) == 4
    names = sorted(pathlib.Path(path).name for path in written)
    assert names == sorted(FIGURE_NAMES.values())
    for path in written:
        target = pathlib.Path(path)
        assert target.exists() and target.stat().st_size > 1000
        assert target.parent == directory / "plots"


def test_the_figures_can_be_written_somewhere_else(tmp_path):
    directory = tiny_campaign(tmp_path)
    written = plot_campaign(directory, output_dir=tmp_path / "figures", window=2)

    assert all(pathlib.Path(path).parent == tmp_path / "figures" for path in written)


def test_a_panel_draws_the_raw_measurements_and_the_smoothed_line():
    """The rule the module is built around: never one without the other."""
    import matplotlib.pyplot as plt

    from tetris.rl.plots import _panel

    figure, axis = plt.subplots()
    series = {0: ([1, 2, 3, 4], [10.0, 20.0, 5.0, 30.0])}
    _panel(axis, series, 3, "reward", "title", "episode reward", "trailing mean (window 3)")

    lines = axis.get_lines()
    labels = [line.get_label() for line in lines]
    assert any(label.startswith("raw") for label in labels)
    assert any("trailing mean (window 3)" in label for label in labels)

    raw, smoothed = lines[0], lines[1]
    assert list(raw.get_ydata()) == [10.0, 20.0, 5.0, 30.0]     # the measurements
    assert list(smoothed.get_ydata()) == list(trailing_mean([10.0, 20.0, 5.0, 30.0], 3))
    assert list(smoothed.get_ydata()) != list(raw.get_ydata())  # actually smoothed
    plt.close(figure)


def test_several_seeds_get_a_line_each_and_a_cross_seed_mean():
    import matplotlib.pyplot as plt

    from tetris.rl.plots import _panel

    figure, axis = plt.subplots()
    series = {0: ([1, 2], [1.0, 1.0]), 1: ([1, 2], [3.0, 3.0])}
    _panel(axis, series, 2, "reward", "title", "episode reward", "trailing mean (window 2)")

    labels = [line.get_label() for line in axis.get_lines()]
    assert any("seed 0" in label for label in labels)
    assert any("seed 1" in label for label in labels)
    assert any("mean of 2 seeds" in label for label in labels)
    plt.close(figure)


def test_a_campaign_without_metrics_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="nothing to plot"):
        plot_campaign(tmp_path / "missing")


# --------------------------------------------------------- the baselines


def comparison_payload() -> dict:
    def agent(score, sd, per_100):
        return {"stats": {"score_mean": score, "score_stdev": sd},
                "diagnostics": {"lines_per_100_pieces": per_100}}

    return {
        "protocol": {"num_games": 20, "seed": 7, "max_pieces": 1000},
        "agents": {
            "random": agent(410.0, 80.0, 0.0),
            "heuristic": agent(910_000.0, 1000.0, 39.6),
            "dqn-B-seed0": agent(700.0, 120.0, 1.2),
        },
    }


def test_the_baseline_figure_is_written_from_a_saved_comparison(tmp_path):
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(comparison_payload()))

    written = plot_comparison(path, tmp_path / "05_baselines.png")

    assert pathlib.Path(written).exists()
    assert pathlib.Path(written).stat().st_size > 1000


def test_the_baseline_figure_writes_where_it_is_told():
    written = plot_comparison(comparison_payload(), pathlib.Path("/tmp/rl-test-baselines.png"))
    assert pathlib.Path(written) == pathlib.Path("/tmp/rl-test-baselines.png")
    assert pathlib.Path(written).stat().st_size > 1000


def test_a_log_axis_is_chosen_from_the_spread_not_assumed():
    from tetris.rl.plots import _spans_orders

    assert _spans_orders([410.0, 910_000.0])            # random vs heuristic
    assert not _spans_orders([400.0, 900.0, 1200.0])    # all in the same range
    assert not _spans_orders([0.0, 0.0])                # nothing to plot


def test_a_reference_far_outside_the_curves_is_not_drawn_on_them():
    """930,000 against ~550 would flatten the very curves the figure is about."""
    from tetris.rl.plots import _fits_on_the_axis

    curves = [248.75, 560.2, 677.5]

    assert _fits_on_the_axis(499.8, curves)             # the random baseline: same scale
    assert _fits_on_the_axis(2500.0, curves)            # 3.7x the tallest point: fits
    assert not _fits_on_the_axis(930_266.2, curves)     # the heuristic: three orders out
    assert not _fits_on_the_axis(5000.0, curves)
    assert not _fits_on_the_axis(499.8, [])             # nothing to compare against


def test_the_campaign_figures_accept_baselines_including_a_far_one(tmp_path):
    """An off-scale reference must not stop the figures being written."""
    directory = tiny_campaign(tmp_path)
    written = plot_campaign(
        directory,
        output_dir=tmp_path / "plots",
        baselines={"random": {"score_mean": 499.8},
                   "heuristic": {"score_mean": 930_266.2}},
        title="T",
    )

    assert len(written) == 4
    for path in written:
        assert pathlib.Path(path).exists() and pathlib.Path(path).stat().st_size > 0


def test_the_lines_per_100_fallback_computes_from_the_games():
    from tetris.rl.plots import _lines_per_100

    entry = {"games": [{"pieces": 10, "lines_cleared": 1}, {"pieces": 10, "lines_cleared": 3}]}
    assert _lines_per_100(entry) == pytest.approx(20.0)
    assert _lines_per_100({"games": []}) == 0.0
