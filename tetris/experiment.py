"""Run, log and compare experiments -- the same way, every time.

The pieces:

* `ExperimentConfig` (in `tetris.config`) says what to run.
* `run_experiment(config)` builds the agent, metres it, evaluates it, and
  returns an `ExperimentResult` that carries the config, the numbers, the
  timing, the per-game rows and the machine it ran on.
* `ExperimentResult.save()` writes that to JSON; `append_run_log()` adds a
  one-line summary to a shared `runs.jsonl`; `compare_runs()` prints the log as
  a table.

    from tetris.config import ExperimentConfig
    from tetris.experiment import run_experiment, compare_runs

    config = ExperimentConfig(agent="heuristic", num_games=30, seed=42,
                              lookahead_depth=1)
    result = run_experiment(config, save_dir="experiments/results")
    print(result)                       # the run, as a small table
    print(compare_runs("experiments/results/runs.jsonl"))

`evaluate_agent(agent, ...)` in `tetris.evaluate` is still the low-level way in,
for an agent object that has no config (or no name in the registry):

    from tetris.evaluate import evaluate_agent
    evaluate_agent(MyAgent(), num_games=10, seed=1)

Two rules make the saved numbers trustworthy:

* **`stats` and `diagnostics` are deterministic** -- they depend only on the
  config and the seed, so two runs of the same config can be diffed. Anything
  measured with a clock lives in `timing`, which will differ every time.
* **The environment is recorded with the results** (Python, platform, library
  versions), because a number without the machine that produced it is not
  reproducible.
"""

from __future__ import annotations

import cProfile
import csv
import dataclasses
import json
import pathlib
import pstats
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import (  # noqa: UP035
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from .agents import build_agent
from .config import ConfigLike, ExperimentConfig, environment_info
from .evaluate import GameResult, evaluate_agent
from .metrics import max_column_height

Path = Union[str, pathlib.Path]

#: The shared run log: one JSON object per line, one line per run.
LOG_NAME = "runs.jsonl"

#: The columns of a comparison, in order. These are the numbers an experiment
#: has to report -- everything else is detail.
COMPARISON_FIELDS = (
    "run_id",
    "agent",
    "config_id",
    "num_games",
    "seed",
    "max_pieces",
    "score_mean",
    "score_median",
    "score_max",
    "lines_mean",
    "lines_max",
    "pieces_mean",
    "games_per_second",
    "elapsed_seconds",
)


def timestamp() -> str:
    """Now, in UTC, to the microsecond.

    Microseconds, not seconds, because `run_id` is built from this: two runs of
    the same config must never be handed the same id (or the same file name).
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class Metered:
    """An agent wrapper that counts what the agent did, without changing it.

    `evaluate_agent` only needs `choose_action` and `name`, so this can wrap any
    agent. It adds the numbers that make throughput readable: how many
    decisions and how many boards (for an agent that searches) per second, and
    how high the stack had to be played.

    Forwarding rules: `name` comes from the wrapped agent, and the harness's
    per-game `reset(seed)` is passed through when the agent defines one.
    """

    def __init__(self, agent: Any) -> None:
        self.agent = agent
        self.decisions = 0
        self.boards_scored = 0
        self.game_peaks: List[float] = []
        self._tallest = 0.0

    @property
    def name(self) -> str:
        return getattr(self.agent, "name", type(self.agent).__name__)

    def reset(self, seed: Optional[int] = None) -> None:
        """Bank the finished game's peak, then reset the agent for the next one."""
        if self._tallest:
            self.game_peaks.append(self._tallest)
            self._tallest = 0.0
        reset = getattr(self.agent, "reset", None)
        if callable(reset):
            reset(seed)

    def choose_action(self, state, valid_actions):
        self._tallest = max(self._tallest, float(max_column_height(state.board)))
        action = self.agent.choose_action(state, valid_actions)
        self.decisions += 1
        decision = getattr(self.agent, "last_decision", None)
        if decision is not None:
            self.boards_scored += int(decision.nodes)
        return action

    def flush(self) -> None:
        """Bank the last game's peak (`evaluate_agent` never resets after it)."""
        if self._tallest:
            self.game_peaks.append(self._tallest)
            self._tallest = 0.0

    def __repr__(self) -> str:
        return f"<Metered {self.name} decisions={self.decisions} boards={self.boards_scored}>"


@dataclass(frozen=True)
class ExperimentResult:
    """Everything one run produced: the config, the numbers, the machine.

    `stats` and `diagnostics` are deterministic; `timing` is not. `games` holds
    one `GameResult` per game, which is what a plot or a paired comparison needs.
    """

    config: ExperimentConfig
    agent: str
    stats: Dict[str, float]
    diagnostics: Dict[str, float]
    timing: Dict[str, float]
    games: Tuple[GameResult, ...]
    environment: Dict[str, Any]
    config_id: str
    started_at: str
    finished_at: str
    profile: Optional[str] = None
    result_path: Optional[str] = None

    # ------------------------------------------------------------- identity

    @property
    def run_id(self) -> str:
        """A filesystem-safe id: when it ran, and which config it ran."""
        stamp = self.started_at.replace("-", "").replace(":", "")
        stamp = stamp.replace(".", "").split("+")[0]
        return f"{stamp}-{self.config_id}"

    @property
    def num_games(self) -> int:
        return self.config.num_games

    @property
    def seed(self) -> Optional[int]:
        return self.config.seed

    # ------------------------------------------------------------- output

    def as_dict(self, include_games: bool = True) -> Dict[str, Any]:
        """The whole run as plain data, ready for `json.dump`."""
        data: Dict[str, Any] = {
            "run_id": self.run_id,
            "agent": self.agent,
            "config_id": self.config_id,
            "config": self.config.to_dict(),
            "num_games": self.num_games,
            "seed": self.seed,
            "max_pieces": self.config.max_pieces,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "stats": self.stats,
            "diagnostics": self.diagnostics,
            "timing": self.timing,
            "environment": self.environment,
            "profile": self.profile,
            "result_path": self.result_path,
        }
        if include_games:
            data["games"] = [game.as_dict() for game in self.games]
        return data

    def log_row(self) -> Dict[str, Any]:
        """The one-line summary that goes into the shared run log."""
        return self.as_dict(include_games=False)

    def summary(self) -> str:
        """The run as a small table: what it was, what it produced, what it cost."""
        stats, timing = self.stats, self.timing
        lines = [
            f"run {self.run_id}",
            f"  config         {self.config.describe()}",
            f"  config_id      {self.config_id}",
            f"  score          mean {stats['score_mean']:12,.1f}"
            f"   median {stats['score_median']:10,.0f}   max {stats['score_max']:12,.0f}",
            f"  lines cleared  mean {stats['lines_mean']:12.2f}   max    {stats['lines_max']:10,.0f}",
            f"  pieces         mean {stats['pieces_mean']:12.1f}",
            f"  capped games   {stats['games_capped']:.0f}/{self.num_games}",
            f"  timing         {timing['elapsed_seconds']:.2f}s"
            f"   {timing['games_per_second']:.2f} games/s"
            f"   {timing['pieces_per_second']:.0f} pieces/s",
        ]
        if self.diagnostics.get("boards_scored"):
            lines.append(
                f"  search         {self.diagnostics['boards_scored']:,.0f} boards scored"
                f"   {timing['boards_per_second']:,.0f} boards/s"
                f"   {timing['boards_per_decision']:,.1f} per decision"
            )
        if self.diagnostics.get("peak_column_mean"):
            lines.append(
                f"  peak column    mean {self.diagnostics['peak_column_mean']:.2f}"
                f"   max {self.diagnostics['peak_column_max']:.0f}"
            )
        if self.profile:
            lines.append(f"  profile        {self.profile}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def save(self, path: Path) -> str:
        """Write the run -- config, stats, timing, environment, every game -- to JSON."""
        target = pathlib.Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")
        return str(target)


# --------------------------------------------------------------- the runner


def run_experiment(
    config: ConfigLike,
    agent: Any = None,
    save_dir: Optional[Path] = None,
    profile: bool = False,
    log_every: Optional[int] = None,
    logger: Optional[Callable[[str], None]] = print,
) -> ExperimentResult:
    """Evaluate one agent as `config` describes it, and record everything.

    `agent` may be passed in (any object with `choose_action`); otherwise it is
    built from the config's `agent` name through the registry in
    `tetris.agents`, which is the seam a learning agent plugs into later.

    With `save_dir` the result is written to `<save_dir>/run-<run_id>.json` and
    a summary line is appended to `<save_dir>/runs.jsonl`. With `profile=True` a
    cProfile report is written next to it (`<save_dir or .>/profile-<run_id>.txt`).
    """
    config = ExperimentConfig.from_any(config)
    if agent is None:
        agent = build_agent(
            config.agent,
            seed=config.seed,
            weights=config.weights,
            lookahead=config.lookahead_depth,
            beam=config.beam,
            checkpoint=config.checkpoint,
        )
    metered = Metered(agent)

    started_at = timestamp()
    profiler = cProfile.Profile() if profile else None
    if profiler is not None:
        profiler.enable()
    measured = evaluate_agent(
        metered,
        num_games=config.num_games,
        seed=config.seed,
        max_pieces=config.max_pieces,
        log_every=log_every,
        logger=logger,
    )
    if profiler is not None:
        profiler.disable()
    metered.flush()
    finished_at = timestamp()

    result = ExperimentResult(
        config=config,
        agent=metered.name,
        stats=dict(measured.stats),
        diagnostics=_diagnostics(metered, measured.games),
        timing=_timing(measured.elapsed_seconds, metered, measured.games),
        games=measured.games,
        environment=environment_info(),
        config_id=config.config_id,
        started_at=started_at,
        finished_at=finished_at,
    )

    if profiler is not None:
        result = dataclasses.replace(result, profile=write_profile(profiler, result, save_dir))
    if save_dir is not None:
        directory = pathlib.Path(save_dir)
        path = result.save(directory / f"run-{result.run_id}.json")
        result = dataclasses.replace(result, result_path=path)
        append_run_log(result, directory / LOG_NAME)
    return result


def _diagnostics(metered: Metered, games: Sequence[GameResult]) -> Dict[str, float]:
    """Deterministic extras: what the agent had to deal with, not how long it took."""
    pieces = sum(game.pieces for game in games)
    lines = sum(game.lines_cleared for game in games)
    capped = sum(1 for game in games if game.capped)
    peaks = metered.game_peaks
    return {
        "boards_scored": float(metered.boards_scored),
        "decisions": float(metered.decisions),
        "deaths": float(len(games) - capped),
        "lines_per_100_pieces": round(100 * lines / pieces, 3) if pieces else 0.0,
        "peak_column_mean": round(statistics.fmean(peaks), 3) if peaks else 0.0,
        "peak_column_max": float(max(peaks)) if peaks else 0.0,
    }


def _timing(elapsed: float, metered: Metered, games: Sequence[GameResult]) -> Dict[str, float]:
    """Wall-clock facts. These differ between runs of the same config, by design."""
    pieces = sum(game.pieces for game in games)
    played = len(games)
    safe = elapsed if elapsed > 0 else float("nan")

    def per_second(count: float) -> float:
        return round(count / safe, 3) if elapsed > 0 else 0.0

    return {
        "elapsed_seconds": round(elapsed, 6),
        "seconds_per_game": round(elapsed / played, 6) if played else 0.0,
        "games_per_second": per_second(played),
        "pieces_per_second": per_second(pieces),
        "decisions_per_second": per_second(metered.decisions),
        "boards_per_second": per_second(metered.boards_scored),
        "boards_per_decision": round(metered.boards_scored / metered.decisions, 3)
        if metered.decisions
        else 0.0,
    }


def write_profile(profiler: cProfile.Profile, result: ExperimentResult,
                  save_dir: Optional[Path] = None, limit: int = 20) -> str:
    """Write a readable cProfile report for a run, and return its path.

    Sorted by cumulative time, top `limit` rows: the same view `python -m
    cProfile` gives, next to the results instead of in the terminal.
    """
    directory = pathlib.Path(save_dir) if save_dir is not None else pathlib.Path(".")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"profile-{result.run_id}.txt"

    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative")
    stream = _StringSink()
    stats.stream = stream
    stats.print_stats(limit)
    header = (
        f"cProfile of {result.config.describe()}\n"
        f"run {result.run_id}, {result.num_games} games\n\n"
    )
    target.write_text(header + stream.text + "\n", encoding="utf-8")
    return str(target)


class _StringSink:
    """A minimal text sink for `pstats` (it wants `.write`, not a real file)."""

    def __init__(self) -> None:
        self.parts: List[str] = []

    def write(self, text: str) -> None:
        self.parts.append(text)

    @property
    def text(self) -> str:
        return "".join(self.parts)


# ------------------------------------------------------- the log and compare


def append_run_log(result: ExperimentResult, path: Path) -> str:
    """Append one run to a JSON-lines log, creating it if needed."""
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result.log_row(), sort_keys=True) + "\n")
    return str(target)


def load_run_log(path: Path) -> List[Dict[str, Any]]:
    """Read a JSON-lines run log back into a list of rows."""
    rows = []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def gather_runs(source: Union[Path, ExperimentResult, Dict[str, Any], Sequence[Any]]) -> List[Dict[str, Any]]:
    """Accept runs in whatever form the caller has, and return log-shaped dicts.

    A path may be a `runs.jsonl` log, a single `run-*.json`, or a directory of
    them; a result object or an already-saved dict is taken as it is. This is
    what lets `compare_runs` work on a live result, a log, or a folder.
    """
    if isinstance(source, ExperimentResult):
        return [source.log_row()]
    if isinstance(source, dict):
        return [source]
    if isinstance(source, (str, pathlib.Path)):
        path = pathlib.Path(source)
        if path.is_dir():
            rows: List[Dict[str, Any]] = []
            log = path / LOG_NAME
            if log.exists():
                rows.extend(load_run_log(log))
            for item in sorted(path.glob("run-*.json")):
                rows.append(json.loads(item.read_text(encoding="utf-8")))
            return _dedupe(rows)
        if path.suffix == ".jsonl":
            return load_run_log(path)
        return [json.loads(path.read_text(encoding="utf-8"))]

    rows = []
    for item in source:                      # a list of any of the above
        rows.extend(gather_runs(item))
    return _dedupe(rows)


def _dedupe(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop runs that appear twice (a log line and its `run-*.json`, say).

    Only rows that carry a `run_id` are deduplicated: a run is identified by
    that id, and the same run recorded twice is not two experiments. Rows with
    no id (hand-built dicts) are all kept.
    """
    seen = set()
    unique = []
    for row in rows:
        run_id = row.get("run_id")
        if run_id is not None:
            if run_id in seen:
                continue
            seen.add(run_id)
        unique.append(row)
    return unique


def comparison_rows(runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten runs into the comparison columns, newest last."""
    rows = []
    for run in runs:
        stats = run.get("stats", {})
        timing = run.get("timing", {})
        config = run.get("config", {})
        row = {
            "run_id": run.get("run_id"),
            "agent": run.get("agent"),
            "config_id": run.get("config_id"),
            "num_games": run.get("num_games"),
            "seed": run.get("seed"),
            "max_pieces": run.get("max_pieces", config.get("max_pieces")),
        }
        for key in ("score_mean", "score_median", "score_max", "lines_mean", "lines_max",
                    "pieces_mean", "games_capped"):
            row[key] = stats.get(key)
        row["games_per_second"] = timing.get("games_per_second")
        row["elapsed_seconds"] = timing.get("elapsed_seconds")
        rows.append(row)
    return rows


def format_comparison(rows: Sequence[Dict[str, Any]]) -> str:
    """The comparison table: one line per run, the columns of COMPARISON_FIELDS."""
    if not rows:
        return "no runs to compare"
    header = (
        f"{'agent':>10} {'games':>6} {'seed':>5} {'cap':>6} "
        f"{'score mean':>12} {'median':>10} {'max':>12} "
        f"{'lines mean':>10} {'lines max':>9} {'pieces':>7} {'games/s':>8} {'secs':>7}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        seed = "none" if row.get("seed") is None else row["seed"]
        cap = "none" if row.get("max_pieces") is None else row["max_pieces"]
        lines.append(
            f"{str(row.get('agent'))[:10]:>10} {row.get('num_games') or 0:>6} "
            f"{str(seed):>5} {str(cap):>6} "
            f"{_num(row.get('score_mean'), ',.1f'):>12} "
            f"{_num(row.get('score_median'), ',.0f'):>10} "
            f"{_num(row.get('score_max'), ',.0f'):>12} "
            f"{_num(row.get('lines_mean'), '.2f'):>10} "
            f"{_num(row.get('lines_max'), ',.0f'):>9} "
            f"{_num(row.get('pieces_mean'), ',.1f'):>7} "
            f"{_num(row.get('games_per_second'), '.2f'):>8} "
            f"{_num(row.get('elapsed_seconds'), '.1f'):>7}"
        )
    return "\n".join(lines)


def compare_runs(source, save_csv: Optional[Path] = None) -> str:
    """Load runs from anywhere, print them side by side, optionally save as CSV."""
    rows = comparison_rows(gather_runs(source))
    if save_csv is not None:
        save_comparison_csv(rows, save_csv)
    return format_comparison(rows)


def save_comparison_csv(rows: Sequence[Dict[str, Any]], path: Path) -> str:
    """Write comparison rows to CSV, one row per run."""
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COMPARISON_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in COMPARISON_FIELDS})
    return str(target)


def _num(value: Any, spec: str) -> str:
    """Format a number for the table, tolerating None/missing."""
    if value is None:
        return "-"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


__all__ = [
    "COMPARISON_FIELDS",
    "LOG_NAME",
    "ExperimentResult",
    "Metered",
    "append_run_log",
    "compare_runs",
    "comparison_rows",
    "format_comparison",
    "gather_runs",
    "load_run_log",
    "run_experiment",
    "save_comparison_csv",
    "timestamp",
    "write_profile",
]
