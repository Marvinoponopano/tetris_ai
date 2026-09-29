"""Figures for a campaign: what the numbers did, with the numbers still visible.

Every figure follows one rule, because reading a smoothed curve and believing it
is the easiest way to fool yourself about a training run:

    the raw measurements are always drawn, in light grey, and the smoothed line
    is drawn *on top of them*, never instead of them.

The smoothing itself is a **trailing** mean (each point is the average of the
last `window` measurements, ending at that point), not a centred one: a centred
window averages in measurements that had not happened yet, which is fine for
describing a finished run but wrong for "what did the agent know at episode N".

The four figures the task asks for, plus the baseline comparison:

    01_training_reward.png    reward per training episode (raw + trailing mean)
    02_lines_per_episode.png  lines cleared per training episode
    03_evaluation_score.png   greedy score on fixed seeds, per seed, with error
                              bars, plus lines per 100 pieces (cap-robust)
    04_loss.png               Huber loss per training episode
    05_baselines.png          the same protocol for dqn vs random / heuristic /
                              lookahead heuristic (written by `plot_comparison`)

Each figure is labelled with how many raw measurements it draws and what the
window is, so a figure can never be quoted without its smoothing being obvious.
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple  # noqa: UP035

import matplotlib

matplotlib.use("Agg")   # no display: these are files, and this makes that explicit

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .campaign import METRICS_NAME, read_metrics, split_metrics  # noqa: E402

#: The moving-average window, in episodes, used unless the caller says otherwise.
DEFAULT_WINDOW = 20

#: Drawn raw, so a figure shows the measurements and their trend together.
RAW_STYLE = dict(color="0.72", linewidth=0.6, alpha=0.9, zorder=1)
MEAN_STYLE = dict(linewidth=1.8, zorder=3)

FIGURE_NAMES = {
    "reward": "01_training_reward.png",
    "lines": "02_lines_per_episode.png",
    "evaluation": "03_evaluation_score.png",
    "loss": "04_loss.png",
}


def trailing_mean(values: Sequence[float], window: int = DEFAULT_WINDOW) -> np.ndarray:
    """The mean of the last `window` values, at every position (causal).

    `min_periods=1`: the first points are the mean of what exists so far, so the
    smoothed line covers the whole run instead of starting at `window`.
    """
    numbers = np.asarray([float(value) for value in values], dtype=float)
    if window <= 1 or numbers.size == 0:
        return numbers
    window = int(window)
    cumulative = np.concatenate([[0.0], np.cumsum(numbers)])
    smoothed = np.empty_like(numbers)
    for index in range(numbers.size):
        start = max(0, index - window + 1)
        smoothed[index] = (cumulative[index + 1] - cumulative[start]) / (index + 1 - start)
    return smoothed


def _seed_label(seed: int, many: bool) -> str:
    return f"seed {seed}" if many else "training run"


def load_campaign_metrics(
    directory: pathlib.Path,
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[int, List[Dict[str, Any]]]]:
    """`(training rows by seed, evaluation rows by seed)` for a campaign directory.

    Accepts the campaign directory (`<output_dir>/<name>`) and reads every
    `seed*/metrics.jsonl` under it. The seed comes from the directory name --
    the only place it is unambiguous, since an evaluation row carries both the
    run's seed and the seed its games were played on.
    """
    root = pathlib.Path(directory)
    training: Dict[int, List[Dict[str, Any]]] = {}
    evaluation: Dict[int, List[Dict[str, Any]]] = {}
    paths = sorted(root.glob(f"seed*/{METRICS_NAME}"))
    if not paths and (root / METRICS_NAME).exists():        # a single seed, not nested
        paths = [root / METRICS_NAME]
    for path in paths:
        rows = read_metrics(path)
        train_rows, eval_rows = split_metrics(rows)
        match = re.fullmatch(r"seed(-?\d+)", path.parent.name)
        if match:
            seed = int(match.group(1))
        else:
            seed = int(rows[0].get("seed", 0)) if rows else 0
        training[seed] = train_rows
        evaluation[seed] = eval_rows
    return training, evaluation


def _fits_on_the_axis(value: float, values: Sequence[float], factor: float = 5.0) -> bool:
    """Whether a reference line can be drawn without flattening the curves.

    The heuristic agents score ~930,000 where a DQN scores ~550: drawn on the
    same linear axis they turn every training curve into a flat line on the
    floor, which is exactly the kind of figure this module exists to avoid. A
    reference too far outside the data is *named* in the corner instead (see
    `plot_campaign`), with its value, so it is still on the record.
    """
    numbers = [float(v) for v in values if v is not None]
    if not numbers:
        return False
    low, high = min(numbers), max(numbers)
    if high <= 0:
        return value <= 0
    floor = low / factor if low > 0 else 0.0
    return floor <= value <= high * factor


def _panel(
    axis,
    series: Dict[Any, Tuple[Sequence[float], Sequence[float]]],
    window: int,
    ylabel: str,
    title: str,
    raw_label: str,
    smoothed_label: str,
    references: Optional[Dict[str, float]] = None,
    log_y: bool = False,
) -> None:
    """Draw one raw-plus-trailing-mean panel (see the module docstring)."""
    colours = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
    many = len(series) > 1
    all_smoothed: List[np.ndarray] = []
    drawn_points = 0

    for position, (key, (xs, ys)) in enumerate(sorted(series.items(), key=lambda item: str(item[0]))):
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        if ys.size == 0:
            continue
        colour = colours[position % len(colours)]
        axis.plot(xs, ys, label=(f"raw {raw_label}" if position == 0 else None), **RAW_STYLE)
        smoothed = trailing_mean(ys, window)
        axis.plot(
            xs, smoothed, color=colour,
            label=(f"{_seed_label(key, many)} -- {smoothed_label}" if many
                   else smoothed_label),
            **MEAN_STYLE,
        )
        all_smoothed.append(smoothed)
        drawn_points += int(ys.size)

    if len(all_smoothed) > 1:
        width = min(len(values) for values in all_smoothed)
        stacked = np.vstack([values[:width] for values in all_smoothed])
        axis.plot(
            np.arange(1, width + 1),
            stacked.mean(axis=0),
            color="black", linewidth=2.2, linestyle="--", zorder=4,
            label=f"mean of {len(all_smoothed)} seeds",
        )

    for name, value in (references or {}).items():
        axis.axhline(value, color="crimson", linestyle=":", linewidth=1.2, alpha=0.8,
                     label=f"{name} ({value:,.1f})")

    axis.set_title(title, fontsize=11)
    axis.set_xlabel("training episode")
    axis.set_ylabel(ylabel)
    if log_y:
        axis.set_yscale("log")
    axis.grid(True, alpha=0.25)
    handle, labels = axis.get_legend_handles_labels()
    if labels:
        axis.legend(fontsize=7, loc="best")
    return drawn_points


def _finish(figure, path: pathlib.Path, note: str) -> str:
    figure.text(0.01, 0.005, note, fontsize=7, color="0.35", ha="left", va="bottom")
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return str(path)


def _series(training: Dict[int, List[Dict[str, Any]]], field: str):
    """`{seed: (episodes, values)}`, skipping rows where the field is null."""
    series = {}
    for seed, rows in training.items():
        xs, ys = [], []
        for row in rows:
            value = row.get(field)
            if value is None:
                continue
            xs.append(float(row.get("episode", len(xs) + 1)))
            ys.append(float(value))
        series[seed] = (xs, ys)
    return series


def plot_campaign(
    directory,
    output_dir=None,
    window: int = DEFAULT_WINDOW,
    baselines: Optional[Dict[str, Dict[str, float]]] = None,
    title: Optional[str] = None,
) -> List[str]:
    """Write the four training figures for a campaign; return the paths written."""
    root = pathlib.Path(directory)
    target = pathlib.Path(output_dir) if output_dir is not None else root / "plots"
    training, evaluation = load_campaign_metrics(root)
    if not training:
        raise FileNotFoundError(f"no metrics.jsonl under {root}: nothing to plot")
    label = title or root.name
    written: List[str] = []

    # ------------------------------------------------------------ 1. reward
    series = _series(training, "reward")
    figure, axis = plt.subplots(figsize=(10, 4.6))
    points = _panel(
        axis, series, window, "episode reward", f"{label}: training reward per episode",
        "episode reward", f"trailing mean (window {window})",
    )
    written.append(_finish(
        figure, target / FIGURE_NAMES["reward"],
        f"{points} raw per-episode measurements, one light line per seed; bold = trailing "
        f"mean over {window} episodes (never drawn without the raw values).",
    ))

    # ------------------------------------------------------------- 2. lines
    series = _series(training, "lines")
    figure, axis = plt.subplots(figsize=(10, 4.6))
    points = _panel(
        axis, series, window, "lines cleared", f"{label}: lines cleared per episode",
        "lines per episode", f"trailing mean (window {window})",
    )
    written.append(_finish(
        figure, target / FIGURE_NAMES["lines"],
        f"{points} raw per-episode measurements; bold = trailing mean over {window} "
        "episodes. A zero line here means the policy never cleared a line at all.",
    ))

    # -------------------------------------------------------- 3. evaluation
    figure, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True)
    if evaluation and any(evaluation.values()):
        colours = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
        many = len(evaluation) > 1
        for position, seed in enumerate(sorted(evaluation)):
            rows = [row for row in evaluation[seed] if row.get("episode") is not None]
            if not rows:
                continue
            xs = np.array([float(row["episode"]) for row in rows])
            scores = np.array([float(row["score_mean"]) for row in rows])
            spread = np.array([float(row.get("score_stdev") or 0.0) for row in rows])
            games = np.array([max(1, int(row.get("games") or 1)) for row in rows])
            error = spread / np.sqrt(games)                     # standard error
            colour = colours[position % len(colours)]
            top.errorbar(xs, scores, yerr=error, color=colour, marker="o", markersize=3,
                         linewidth=1.2, capsize=2, alpha=0.9,
                         label=_seed_label(seed, many) if many else "greedy score")
            lines_per_100 = np.array([float(row.get("lines_per_100_pieces") or 0.0) for row in rows])
            bottom.plot(xs, lines_per_100, color=colour, marker="o", markersize=3,
                        linewidth=1.2, label=_seed_label(seed, many) if many else "lines/100")

        # The baselines are reference lines -- but only the ones that fit. An
        # agent scoring 930,000 drawn against curves of ~550 would flatten those
        # curves to the floor, so an off-scale baseline is named in the corner
        # with its value instead of being drawn.
        reference_lines: Dict[str, float] = {}
        off_scale: Dict[str, float] = {}
        curve_scores = [float(row["score_mean"]) for rows in evaluation.values()
                        for row in rows if row.get("score_mean") is not None]
        for name, values in (baselines or {}).items():
            if "score_mean" not in values:
                continue
            value = float(values["score_mean"])
            if _fits_on_the_axis(value, curve_scores):
                reference_lines[name] = value
            else:
                off_scale[name] = value
        for name, value in reference_lines.items():
            top.axhline(value, color="crimson", linestyle=":", linewidth=1.2, alpha=0.8,
                        label=f"{name} ({value:,.0f})")
        if off_scale:
            top.text(
                0.012, 0.97,
                "off this axis:\n" + "\n".join(
                    f"{name} {value:,.0f}" for name, value in sorted(off_scale.items())),
                transform=top.transAxes, va="top", ha="left", fontsize=7, color="crimson",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="crimson",
                          alpha=0.85),
            )
        top.set_title(f"{label}: greedy evaluation on fixed seeds (no exploration)", fontsize=11)
        top.set_ylabel("mean score")
        bottom.set_ylabel("lines per 100 pieces")
        bottom.set_xlabel("training episode")
        for axis in (top, bottom):
            axis.grid(True, alpha=0.25)
            axis.legend(fontsize=7, loc="best")
        note = ("every point is one evaluation: greedy (epsilon 0) games on the same seeded "
                "piece sequences; error bars are the standard error over the games. No smoothing.")
        if off_scale:
            note += (" The baselines off the axis are named in the corner (they score "
                     f"{min(off_scale.values()):,.0f}+, thousands of times the curves); "
                     "drawing them here would flatten the curves this figure is about.")
        else:
            note += " The crimson dotted lines are the baselines on the same protocol."
    else:
        top.set_title(f"{label}: greedy evaluation (no rows yet)", fontsize=11)
        note = "no evaluation rows were found in this campaign"
    written.append(_finish(figure, target / FIGURE_NAMES["evaluation"], note))

    # -------------------------------------------------------------- 4. loss
    series = _series(training, "loss_mean")
    figure, axis = plt.subplots(figsize=(10, 4.6))
    points = _panel(
        axis, series, window, "Huber loss (per gradient step)",
        f"{label}: training loss per episode", "mean loss per episode",
        f"trailing mean (window {window})",
    )
    written.append(_finish(
        figure, target / FIGURE_NAMES["loss"],
        f"{points} raw per-episode measurements (episodes before the replay warm-up have no "
        f"gradient step and are absent); bold = trailing mean over {window} episodes.",
    ))
    return written


def _spans_orders(values: Sequence[float], factor: float = 20.0) -> bool:
    """True when the values are spread over more than `factor` (a log axis earns it).

    Used for the score panel of the baseline figure: a random agent scores in
    the hundreds and the heuristic agent in the hundreds of thousands, and on a
    linear axis the DQN curve would be a flat line on the floor. The choice is
    made from the data (and the axis is labelled), never assumed.
    """
    positive = [float(value) for value in values if float(value) > 0]
    if not positive:
        return False
    return max(positive) > factor * max(1e-9, min(positive))


def plot_comparison(comparison, path=None, log_scores: Optional[bool] = None) -> str:
    """The baseline figure: the same protocol, every agent, two panels.

    Panel 1 is the mean score with the game-to-game spread, on a log axis when
    the agents span orders of magnitude (which they do: a random agent scores in
    the hundreds and the heuristic agent in the hundreds of thousands). Panel 2
    is lines per 100 pieces, which is the only metric that stays comparable when
    one agent is capped at `max_pieces` and another dies in twenty pieces.
    """
    data = comparison
    if not isinstance(data, dict):
        data = json.loads(pathlib.Path(data).read_text(encoding="utf-8"))
    agents = data.get("agents", {})
    target = pathlib.Path(path) if path is not None else pathlib.Path("05_baselines.png")

    names = [name for name, entry in agents.items() if entry.get("stats")]
    scores = [float(agents[name]["stats"].get("score_mean") or 0.0) for name in names]
    score_sd = [float(agents[name]["stats"].get("score_stdev") or 0.0) for name in names]
    per_100 = [float(agents[name]["diagnostics"].get("lines_per_100_pieces") or 0.0)
               if "diagnostics" in agents[name] else _lines_per_100(agents[name])
               for name in names]

    positive = [value for value in scores if value > 0]
    if log_scores is None:
        log_scores = _spans_orders(positive)

    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 0.55 * max(4, len(names)) + 2))
    positions = np.arange(len(names))
    colours = ["C3" if name.startswith("random") else
               "C0" if name.startswith("dqn") else "C2" for name in names]

    left.barh(positions, scores, xerr=score_sd, color=colours, alpha=0.85, capsize=3)
    left.set_yticks(positions)
    left.set_yticklabels(names, fontsize=8)
    left.set_xlabel("mean score (bar = mean, whisker = 1 sd over games)")
    left.set_title("mean score" + (" (log axis)" if log_scores else ""), fontsize=11)
    if log_scores:
        left.set_xscale("log")
        left.set_xlim(left=max(1.0, min(positive) / 4) if positive else 1.0)
    left.grid(True, axis="x", alpha=0.25)

    right.barh(positions, per_100, color=colours, alpha=0.85)
    right.set_yticks(positions)
    right.set_yticklabels(names, fontsize=8)
    right.set_xlabel("lines cleared per 100 pieces placed")
    right.set_title("lines per 100 pieces (cap-independent)", fontsize=11)
    right.grid(True, axis="x", alpha=0.25)

    protocol = data.get("protocol", {})
    figure.suptitle(
        "the same evaluation protocol for every agent: "
        f"{protocol.get('num_games', '?')} games, seed {protocol.get('seed', '?')}, "
        f"cap {protocol.get('max_pieces', '?')}",
        fontsize=11,
    )
    return _finish(
        figure, target,
        "bars are means over complete games; the whisker is the game-to-game spread. Every "
        "agent played the same seeded piece sequences (game i = seed + i).",
    )


def _lines_per_100(entry: Dict[str, Any]) -> float:
    """`lines per 100 pieces` from the raw per-game rows, when not precomputed."""
    games = entry.get("games") or []
    pieces = sum(int(game.get("pieces", 0)) for game in games)
    lines = sum(int(game.get("lines_cleared", 0)) for game in games)
    return round(100.0 * lines / pieces, 4) if pieces else 0.0


__all__ = [
    "DEFAULT_WINDOW",
    "FIGURE_NAMES",
    "load_campaign_metrics",
    "plot_campaign",
    "plot_comparison",
    "trailing_mean",
]
