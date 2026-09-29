"""Run a training experiment properly: several seeds, evaluated as it goes.

One training run is an anecdote. This module is what turns a hyper-parameter
question into an answer: it runs the *same* configuration over several seeds,
pauses each one on a schedule to measure it without exploration, keeps the best
checkpoint by that measurement, and reports the whole thing as an aggregate.

The cycle, per seed:

    evaluate the untrained policy          (episode 0 -- the "before" number)
        |
    train N episodes
        |
        +-- after every `eval_every` episodes (and at the end):
              freeze exploration  ->  play `eval_games` games on fixed seeds
              ->  one evaluation row  (score, lines, pieces)
              ->  if it beats the best so far, save a NEW checkpoint

    ... then aggregate across seeds: first (episode 0) vs last, per metric.

What ends up on disk, under `<output_dir>/<experiment>/`:

    campaign.json           the spec: config, seeds, evaluation protocol, variant
    aggregate.json          the cross-seed verdict and every per-seed number
    seed<k>/metrics.jsonl   one JSON object per line, flushed as it is produced:
                            kind=training   (reward, score, lines, pieces, loss, eps)
                            kind=evaluation (greedy score/lines/pieces on fixed seeds)
                            `seed` is always the training run's seed; an
                            evaluation row also carries `eval_seed`, the seed its
                            games were played on.
    seed<k>/summary.json    that seed's run: training summary, every evaluation
    seed<k>/training/       what `tetris.rl.training.train` always writes
                            (train_log.csv, checkpoints, summary, config)
    seed<k>/best/           the best checkpoints, one file per improvement
    seed<k>/best/best.json  an index of them (which is best, and why)

Two rules are worth stating because they are easy to get wrong:

* **The best checkpoint is defined, not felt.** The metric is named in the spec
  (`metric`, default `score_mean`) and read from an evaluation row, i.e. greedy
  games on fixed seeds. It is not the training reward, which is noisy and
  measured while the agent is still exploring.
* **Nothing is overwritten.** Each improvement is written to its own file whose
  name carries the episode and the metric value, so the history of a run stays
  readable and a later, worse evaluation can never destroy an earlier, better
  checkpoint. Only the index (`best.json`) is rewritten, and it is a pointer,
  not a checkpoint.

`parallel=...` runs the seeds in separate processes. They are independent --
different seeds, different directories, no shared state -- which is the cheap
way to use a multi-core machine, and the reason each seed writes its own files.
"""

from __future__ import annotations

import json
import os
import pathlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union  # noqa: UP035

import numpy as np

from ..config import environment_info
from ..evaluate import DEFAULT_MAX_PIECES
from .config import TrainingConfig
from .evaluation import (
    DEFAULT_EVAL_GAMES,
    DEFAULT_EVAL_SEED,
    EVAL_FIELDS,
    EvaluationSnapshot,
    evaluate_greedy,
)
from .training import TrainingResult, train
from .variants import build_variant_agent

Path = Union[str, pathlib.Path]

#: What "best" means unless the spec says otherwise: the mean score of the
#: greedy evaluation. Higher is better, and it is a number this project can
#: compare against the baselines on the same seeds.
DEFAULT_METRIC = "score_mean"

#: Every metric an aggregate reports as a trend (all read from evaluation rows).
#: Mean, median and max are all here because they disagree in a way that matters:
#: a mean score is dragged around by one long game, the median says what a
#: typical game looks like, and the max is the best game the policy found --
#: which is where a policy that clears lines only occasionally shows up first.
TREND_METRICS = (
    "score_mean",
    "score_median",
    "score_max",
    "lines_mean",
    "lines_max",
    "lines_per_100_pieces",
    "pieces_mean",
    "pieces_max",
)

#: How many of the last evaluations the "tail" value averages over. The final
#: evaluation of a run is one row and therefore noisy; the tail is the honest
#: summary of where a run ended up.
TAIL_WINDOW = 3

METRICS_NAME = "metrics.jsonl"
SUMMARY_NAME = "summary.json"
CAMPAIGN_NAME = "campaign.json"
AGGREGATE_NAME = "aggregate.json"
#: Held for as long as a campaign is running, so a second one cannot interleave
#: its rows into the same metrics file.
LOCK_NAME = ".running"
TRAINING_DIR = "training"
BEST_DIR = "best"
BEST_INDEX_NAME = "best.json"


def timestamp() -> str:
    """Now, in UTC, to the microsecond (as everywhere else in this project)."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _process_is_alive(pid: int) -> bool:
    """Whether `pid` is a live process this user can signal (the lock's question)."""
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (TypeError, ValueError, OSError):
        return True      # cannot tell: assume somebody is there, and refuse
    return True


def _write_json(path: Path, payload: Any) -> str:
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return str(target)


class _MetricsWriter:
    """Appends one JSON object per line, flushing each one.

    The same rule as the training log: a run that dies at episode 900 still has
    900 rows on disk, and a curve can be looked at while the run is going.

    `reset=True` truncates the file first. A campaign run is from scratch, not a
    resume, so re-running a seed *replaces* its record; appending to the old one
    would leave two identical curves stacked on top of each other and make every
    later aggregate wrong.
    """

    def __init__(self, path: Path, reset: bool = False) -> None:
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w" if reset else "a", encoding="utf-8")

    def append(self, row: Dict[str, Any]) -> None:
        self._handle.write(json.dumps(row, sort_keys=False) + "\n")
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "_MetricsWriter":
        return self

    def __exit__(self, *exception) -> None:
        self.close()


def _safe(value: float) -> str:
    """A number as a filename fragment (no separators, no plus signs)."""
    text = f"{float(value):.3f}"
    return text.replace("-", "neg")


def read_metrics(path: Path) -> List[Dict[str, Any]]:
    """Read a `metrics.jsonl` back into rows (training and evaluation mixed)."""
    rows = []
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def split_metrics(rows: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """`(training_rows, evaluation_rows)` from the mixed stream."""
    training, evaluation = [], []
    for row in rows:
        (evaluation if row.get("kind") == "evaluation" else training).append(row)
    return training, evaluation


class BestTracker:
    """The best checkpoint so far, by one named evaluation metric.

    Every improvement is written to its own file; the index is the only thing
    rewritten. `consider` is the whole policy, and it is deliberately strict:
    a later evaluation has to be *better* than the best one so far, so a plateau
    does not fill the directory with near-identical files.
    """

    def __init__(
        self,
        directory: Path,
        metric: str = DEFAULT_METRIC,
        higher_is_better: bool = True,
    ) -> None:
        self.directory = pathlib.Path(directory)
        self.metric = str(metric)
        self.higher_is_better = bool(higher_is_better)
        self.improvements: List[Dict[str, Any]] = []
        self.best_value: Optional[float] = None
        self.best: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ policy

    def is_better(self, value: float) -> bool:
        """Strictly better than the best so far (the first one always is)."""
        if self.best_value is None:
            return True
        return value > self.best_value if self.higher_is_better else value < self.best_value

    def consider(
        self, snapshot: EvaluationSnapshot, agent: Any, **extra: Any
    ) -> Optional[str]:
        """Save a checkpoint if `snapshot` beats the best; return its path (or None)."""
        value = snapshot.metric(self.metric)
        if not self.is_better(value):
            return None

        episode = -1 if snapshot.episode is None else int(snapshot.episode)
        name = f"best_ep{episode:06d}_{self.metric}_{_safe(value)}.pt"
        path = agent.save(
            self.directory / name,
            evaluation=snapshot.as_dict(),
            best_metric=self.metric,
            best_value=value,
            **extra,
        )
        record = {
            "episode": snapshot.episode,
            "metric": self.metric,
            "value": value,
            "path": path,
            "checkpoint": name,
            "score_mean": snapshot.score_mean,
            "lines_mean": snapshot.lines_mean,
            "pieces_mean": snapshot.pieces_mean,
            "training_epsilon": snapshot.training_epsilon,
            "decisions": snapshot.decisions,
            "learn_steps": snapshot.learn_steps,
        }
        self.best_value = value
        self.best = record
        self.improvements.append(record)
        self._write_index()
        return path

    def _write_index(self) -> str:
        return _write_json(
            self.directory / BEST_INDEX_NAME,
            {
                "metric": self.metric,
                "higher_is_better": self.higher_is_better,
                "best": self.best,
                "improvements": self.improvements,
            },
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "higher_is_better": self.higher_is_better,
            "best": self.best,
            "improvements": list(self.improvements),
            "directory": str(self.directory),
        }


@dataclass
class SeedRun:
    """What one (experiment, seed) run produced, and where it put it."""

    seed: int
    directory: str
    training: Dict[str, Any]
    evaluations: List[Dict[str, Any]] = field(default_factory=list)
    best: Dict[str, Any] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    started_at: str = ""
    finished_at: str = ""
    metrics_path: str = ""
    summary_path: str = ""

    # ------------------------------------------------------------------ values

    @property
    def episodes_trained(self) -> int:
        return int(self.training.get("episodes_trained", 0))

    def first_eval(self) -> Optional[Dict[str, Any]]:
        return self.evaluations[0] if self.evaluations else None

    def final_eval(self) -> Optional[Dict[str, Any]]:
        return self.evaluations[-1] if self.evaluations else None

    def best_eval(self, metric: str = DEFAULT_METRIC) -> Optional[Dict[str, Any]]:
        """The evaluation with the best `metric` (the one that was checkpointed)."""
        if not self.evaluations:
            return None
        return max(self.evaluations, key=lambda row: float(row.get(metric) or 0.0))

    def tail_eval(self, metric: str = DEFAULT_METRIC,
                  window: int = TAIL_WINDOW) -> Optional[Dict[str, Any]]:
        """The mean of the last `window` evaluations, as one row-like dict.

        The final evaluation is a single measurement and therefore noisy at the
        game-to-game level; averaging the last few is what "where did this run
        end up?" should mean.
        """
        if not self.evaluations:
            return None
        tail = self.evaluations[-max(1, int(window)):]
        keys = [key for key in EVAL_FIELDS if key not in {"episode", "seed", "max_pieces"}]
        averaged = {
            key: float(np.mean([float(row.get(key) or 0.0) for row in tail])) for key in keys
        }
        averaged["episode"] = tail[-1].get("episode")
        averaged["evaluations_averaged"] = len(tail)
        return averaged

    def deltas(self, metrics: Sequence[str] = TREND_METRICS,
               window: int = TAIL_WINDOW) -> Dict[str, float]:
        """`tail - first` per metric: did this run end up better than it started?"""
        first, tail = self.first_eval(), self.tail_eval(window=window)
        if first is None or tail is None:
            return {}
        return {
            name: float(tail.get(name) or 0.0) - float(first.get(name) or 0.0)
            for name in metrics
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "directory": self.directory,
            "training": self.training,
            "evaluations": self.evaluations,
            "best": self.best,
            "elapsed_seconds": self.elapsed_seconds,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "metrics_path": self.metrics_path,
            "summary_path": self.summary_path,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SeedRun":
        return cls(
            seed=int(data["seed"]),
            directory=str(data.get("directory", "")),
            training=dict(data.get("training", {})),
            evaluations=list(data.get("evaluations", [])),
            best=dict(data.get("best", {})),
            elapsed_seconds=float(data.get("elapsed_seconds", 0.0)),
            started_at=str(data.get("started_at", "")),
            finished_at=str(data.get("finished_at", "")),
            metrics_path=str(data.get("metrics_path", "")),
            summary_path=str(data.get("summary_path", "")),
        )

    def summary(self) -> str:
        first, final = self.first_eval(), self.final_eval()
        best = self.best.get("best") or {}
        lines = [
            f"seed {self.seed}: {self.episodes_trained} episodes in {self.elapsed_seconds:.0f}s",
        ]
        if first and final:
            lines.append(
                f"  eval score  first {first['score_mean']:>10,.1f}   "
                f"final {final['score_mean']:>10,.1f}   "
                f"best {best.get('score_mean', 0.0):>10,.1f}"
            )
            lines.append(
                f"  eval lines  first {first['lines_mean']:>10.2f}   "
                f"final {final['lines_mean']:>10.2f}   "
                f"best {best.get('lines_mean', 0.0):>10.2f}"
            )
            lines.append(
                f"  eval pieces first {first['pieces_mean']:>10.1f}   "
                f"final {final['pieces_mean']:>10.1f}   "
                f"best {best.get('pieces_mean', 0.0):>10.1f}"
            )
        if best.get("checkpoint"):
            lines.append(f"  best checkpoint {best['checkpoint']} "
                         f"({self.best.get('metric')} {best.get('value')})")
        return "\n".join(lines)


@dataclass
class CampaignResult:
    """A campaign's outcome: the spec, every seed's run, and the aggregate."""

    campaign: "Campaign"
    runs: List[SeedRun]
    aggregate: Dict[str, Any]
    output_dir: str
    spec_path: str
    aggregate_path: str
    started_at: str
    finished_at: str
    elapsed_seconds: float

    @property
    def name(self) -> str:
        return self.campaign.name

    @property
    def verdict(self) -> str:
        return str(self.aggregate.get("verdict", "unknown"))

    def summary(self) -> str:
        headline = (
            f"{self.name}: {len(self.runs)} seed(s) x "
            f"{self.campaign.config.episodes} episodes, "
            f"{self.elapsed_seconds:.0f}s, metric {self.campaign.metric}"
        )
        lines = [headline, "-" * len(headline)]
        for run in sorted(self.runs, key=lambda item: item.seed):
            lines.append(run.summary())
        lines.append("")
        lines.append("aggregate (tail vs first evaluation, mean over seeds)")
        for name, trend in self.aggregate.get("metrics", {}).items():
            lines.append(
                f"  {name:<22} {trend['first_mean']:>12,.3f} -> {trend['tail_mean']:>12,.3f}"
                f"   delta {trend['delta_mean']:>+12,.3f} (sd {trend['delta_sd']:.3f})"
                f"   improved in {trend['improved_seeds']}/{trend['n_seeds']} seeds"
            )
        lines.append(f"  verdict: {self.verdict}")
        lines.append(f"  aggregate -> {self.aggregate_path}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def save(self, path: Optional[Path] = None) -> str:
        return _write_json(path or self.aggregate_path, self.aggregate)


@dataclass
class Campaign:
    """One experiment: a training config, a set of seeds, an evaluation protocol.

    ::

        campaign = Campaign(
            name="B",
            config=TrainingConfig(episodes=2000, epsilon_decay_steps=30_000),
            seeds=(0, 1, 2),
            eval_every=250,
        )
        result = campaign.run(parallel=3)
        print(result)

    `eval_every`, `eval_games`, `eval_seed` and `eval_max_pieces` are the
    evaluation protocol (the same for every seed, or the seeds are not
    comparable). `metric` names which evaluation number decides "best".
    """

    name: str
    config: TrainingConfig
    seeds: Tuple[int, ...] = (0,)
    eval_every: int = 250
    eval_games: int = DEFAULT_EVAL_GAMES
    eval_seed: int = DEFAULT_EVAL_SEED
    eval_max_pieces: Optional[int] = DEFAULT_MAX_PIECES
    metric: str = DEFAULT_METRIC
    output_dir: Path = "experiments/rl"
    variant: Optional[str] = None
    log_every: int = 50
    note: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"a campaign needs a name, got {self.name!r}")
        if isinstance(self.config, dict) or isinstance(self.config, (str, pathlib.Path)):
            self.config = TrainingConfig.from_any(self.config)
        if self.metric not in EVAL_FIELDS:
            raise ValueError(
                f"metric must be one of the evaluation fields ({', '.join(EVAL_FIELDS)}), "
                f"got {self.metric!r}"
            )
        seeds = (self.seeds,) if isinstance(self.seeds, int) else tuple(self.seeds)
        if not seeds:
            raise ValueError("a campaign needs at least one seed")
        for seed in seeds:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise ValueError(f"seeds must be whole numbers, got {seed!r}")
        self.seeds = tuple(int(seed) for seed in seeds)
        if self.eval_every < 1:
            raise ValueError(f"eval_every must be 1 or more, got {self.eval_every}")
        if self.eval_games < 1:
            raise ValueError(f"eval_games must be 1 or more, got {self.eval_games}")
        if self.log_every < 1:
            raise ValueError(f"log_every must be 1 or more, got {self.log_every}")
        self.output_dir = pathlib.Path(self.output_dir)

    # ------------------------------------------------------------------- spec

    @property
    def directory(self) -> pathlib.Path:
        return pathlib.Path(self.output_dir) / self.name

    def seed_config(self, seed: int) -> TrainingConfig:
        """The training config for one seed: the same experiment, a different deal."""
        return self.config.with_overrides(seed=int(seed))

    def spec(self) -> Dict[str, Any]:
        """The campaign as plain JSON data -- everything needed to re-run it."""
        return {
            "name": self.name,
            "variant": self.variant,
            "seeds": list(self.seeds),
            "metric": self.metric,
            "eval_every": self.eval_every,
            "eval_games": self.eval_games,
            "eval_seed": self.eval_seed,
            "eval_max_pieces": self.eval_max_pieces,
            "log_every": self.log_every,
            "note": self.note,
            "output_dir": str(self.output_dir),
            "config": self.config.to_dict(),
            "config_id": self.config.config_id,
            "environment": environment_info(),
            "written_at": timestamp(),
        }

    @classmethod
    def from_spec(cls, spec: Dict[str, Any]) -> "Campaign":
        """Rebuild a campaign from `spec()` (what the parallel workers receive)."""
        return cls(
            name=str(spec["name"]),
            config=TrainingConfig.from_dict(spec["config"]),
            seeds=tuple(spec["seeds"]),
            eval_every=int(spec["eval_every"]),
            eval_games=int(spec["eval_games"]),
            eval_seed=int(spec["eval_seed"]),
            eval_max_pieces=spec.get("eval_max_pieces"),
            metric=str(spec["metric"]),
            output_dir=spec.get("output_dir", "experiments/rl"),
            variant=spec.get("variant"),
            log_every=int(spec.get("log_every", 50)),
            note=spec.get("note"),
        )

    def describe(self) -> str:
        """One line for a table or a log."""
        return (
            f"{self.name}: {self.config.episodes} episodes x {len(self.seeds)} seed(s) "
            f"{list(self.seeds)}, evaluate every {self.eval_every} over "
            f"{self.eval_games} games (seed {self.eval_seed}, cap {self.eval_max_pieces}), "
            f"best by {self.metric}"
            + (f", variant {self.variant}" if self.variant else "")
        )

    def __str__(self) -> str:
        return self.describe()

    # -------------------------------------------------------------- one seed

    def run_seed(self, seed: int, logger: Optional[Any] = print) -> SeedRun:
        """Train one seed, evaluating on schedule. Writes everything it produces.

        This is the unit of work: it needs no shared state, so it can run in its
        own process (which is what `parallel=` does).
        """
        config = self.seed_config(seed)
        directory = self.directory / f"seed{seed}"
        training_dir = directory / TRAINING_DIR
        directory.mkdir(parents=True, exist_ok=True)

        tracker = BestTracker(directory / BEST_DIR, metric=self.metric)
        evaluations: List[Dict[str, Any]] = []
        started_at = timestamp()
        started = time.perf_counter()

        def say(message: str) -> None:
            if logger is not None:
                logger(message)

        say(f"[seed {seed}] {self.name}: training {config.episodes} episodes "
            f"({config.describe()})")

        def evaluate(episode: Optional[int], agent: Any, stream: _MetricsWriter) -> None:
            """One greedy evaluation: log it, keep it, checkpoint it if it is best."""
            snapshot = evaluate_greedy(
                agent,
                num_games=self.eval_games,
                seed=self.eval_seed,
                max_pieces=self.eval_max_pieces,
                episode=episode,
                train_seconds=time.perf_counter() - started,
                logger=None,
            )
            # `snapshot.row()` carries the *evaluation* seed; in the campaign's
            # stream `seed` is the training run's seed (every row of a seed
            # directory describes one run), so the evaluation's own seed is kept
            # under its own name instead of overwriting it.
            row = snapshot.row()
            row["eval_seed"] = row.pop("seed")
            row["per_game"] = [game.as_dict() for game in snapshot.games]
            stream.append({"kind": "evaluation", "experiment": self.name,
                           "seed": seed, **row})
            evaluations.append({"kind": "evaluation", "experiment": self.name,
                                "seed": seed, **row})
            where = "untrained" if episode in (None, 0) else f"@{episode}"
            say(
                f"  [seed {seed}] eval {where:>9}: score {snapshot.score_mean:>10,.1f}  "
                f"lines {snapshot.lines_mean:>7.2f}  pieces {snapshot.pieces_mean:>7.1f}  "
                f"({snapshot.elapsed_seconds:.1f}s, eps {snapshot.training_epsilon:.3f})"
            )
            saved = tracker.consider(snapshot, agent, experiment=self.name, seed=seed)
            if saved is not None:
                say(f"  [seed {seed}] new best checkpoint -> {saved}")

        with _MetricsWriter(directory / METRICS_NAME, reset=True) as metrics:

            def on_start(agent: Any) -> None:
                # Episode 0: the untrained policy, measured the same way every
                # later evaluation is. Without this row, "improvement" would be
                # measured from an already partly trained agent.
                evaluate(0, agent, metrics)

            def on_episode(episode: int, row: Dict[str, Any], agent: Any) -> None:
                metrics.append({"kind": "training", "experiment": self.name,
                                "seed": seed, **row})
                if episode % self.eval_every and episode != config.episodes:
                    return
                evaluate(episode, agent, metrics)

            training = train(
                config,
                agent=build_variant_agent(self.variant, config, seed),
                output_dir=training_dir,
                logger=None,
                on_episode=on_episode,
                on_start=on_start,
            )

        elapsed = time.perf_counter() - started
        run = SeedRun(
            seed=seed,
            directory=str(directory),
            training=training.as_dict(),
            evaluations=evaluations,
            best=tracker.as_dict(),
            elapsed_seconds=elapsed,
            started_at=started_at,
            finished_at=timestamp(),
            metrics_path=str(directory / METRICS_NAME),
            summary_path=str(directory / SUMMARY_NAME),
        )
        _write_json(directory / SUMMARY_NAME, run.as_dict())
        say(f"[seed {seed}] done in {elapsed:.0f}s"
            + (f" -- best {self.metric} {tracker.best_value:.3f}"
               if tracker.best_value is not None else " -- no evaluation ran"))
        return run

    # -------------------------------------------------------------- all seeds

    def run(
        self,
        parallel: int = 1,
        logger: Optional[Any] = print,
        threads: Optional[int] = None,
    ) -> CampaignResult:
        """Run every seed and aggregate. `parallel` > 1 uses that many processes.

        The campaign takes a lock on its output directory first: two runs writing
        the same `metrics.jsonl` interleave their rows and the file becomes a
        record of neither run, so the second one is refused instead. A lock left
        by a killed run (its process is gone) is taken over.
        """
        self.directory.mkdir(parents=True, exist_ok=True)
        lock = self._acquire_lock()
        try:
            return self._run_locked(parallel=parallel, logger=logger, threads=threads)
        finally:
            lock.unlink(missing_ok=True)

    def _lock_path(self) -> pathlib.Path:
        return self.directory / LOCK_NAME

    def _acquire_lock(self) -> pathlib.Path:
        """Claim this campaign's directory, or say who is already in it."""
        lock = self._lock_path()
        if lock.exists():
            holder = lock.read_text(encoding="utf-8").strip()
            pid = int(holder.split()[0]) if holder.split() else -1
            if pid > 0 and _process_is_alive(pid):
                raise RuntimeError(
                    f"campaign {self.name} is already running (pid {pid}) and writing "
                    f"{self.directory}: two runs sharing one metrics file produce a "
                    f"record of neither. Wait for it, or remove {lock} if that process "
                    f"is gone."
                )
        lock.write_text(f"{os.getpid()} {timestamp()}\n", encoding="utf-8")
        return lock

    def _run_locked(
        self,
        parallel: int = 1,
        logger: Optional[Any] = print,
        threads: Optional[int] = None,
    ) -> CampaignResult:
        """The body of `run()`, once the lock is held."""
        spec = self.spec()
        spec_path = _write_json(self.directory / CAMPAIGN_NAME, spec)

        started_at = timestamp()
        started = time.perf_counter()
        if logger is not None:
            logger(self.describe())

        if parallel and parallel > 1 and len(self.seeds) > 1:
            runs = self._run_parallel(int(parallel), logger, threads)
        else:
            runs = [self.run_seed(seed, logger=logger) for seed in self.seeds]

        aggregate = self.aggregate(runs)
        aggregate_path = _write_json(self.directory / AGGREGATE_NAME, aggregate)
        result = CampaignResult(
            campaign=self,
            runs=runs,
            aggregate=aggregate,
            output_dir=str(self.directory),
            spec_path=spec_path,
            aggregate_path=aggregate_path,
            started_at=started_at,
            finished_at=timestamp(),
            elapsed_seconds=time.perf_counter() - started,
        )
        if logger is not None:
            logger("")
            logger(result.summary())
        return result

    def _run_parallel(
        self,
        parallel: int,
        logger: Optional[Any],
        threads: Optional[int],
    ) -> List[SeedRun]:
        """One process per seed; results come back as plain dicts."""
        worker_threads = threads
        if worker_threads is None:
            worker_threads = max(1, (os.cpu_count() or 1) // parallel)
        payloads = [
            {"spec": self.spec(), "seed": seed, "threads": worker_threads}
            for seed in self.seeds
        ]
        runs: List[SeedRun] = []
        with ProcessPoolExecutor(max_workers=min(parallel, len(self.seeds))) as pool:
            futures = {pool.submit(_run_seed_worker, payload): payload["seed"]
                       for payload in payloads}
            for future in as_completed(futures):
                seed = futures[future]
                run = SeedRun.from_dict(future.result())
                runs.append(run)
                if logger is not None:
                    logger(run.summary())
        return sorted(runs, key=lambda item: item.seed)

    # -------------------------------------------------------------- aggregate

    def aggregate(
        self,
        runs: Sequence[SeedRun],
        metrics: Sequence[str] = TREND_METRICS,
        window: int = TAIL_WINDOW,
    ) -> Dict[str, Any]:
        """First vs last evaluation, averaged over seeds -- and an honest verdict.

        The verdict per metric is deliberately conservative:

        ``improving``
            the mean tail-minus-first is positive *and* every seed improved.
        ``mixed``
            the mean is positive but at least one seed did not improve.
        ``flat-or-down``
            the mean is not positive.

        A single seed's delta is never reported without the spread, and the
        per-seed rows are kept in the aggregate so the claim can be checked.
        """
        per_seed: List[Dict[str, Any]] = []
        for run in sorted(runs, key=lambda item: item.seed):
            first, final, tail = run.first_eval(), run.final_eval(), run.tail_eval(window=window)
            best = run.best.get("best") or {}
            per_seed.append(
                {
                    "seed": run.seed,
                    "episodes_trained": run.episodes_trained,
                    "evaluations": len(run.evaluations),
                    "first": {name: (first or {}).get(name) for name in metrics},
                    "final": {name: (final or {}).get(name) for name in metrics},
                    "tail": {name: (tail or {}).get(name) for name in metrics},
                    "deltas": run.deltas(metrics=metrics, window=window),
                    "best": {
                        "episode": best.get("episode"),
                        "value": best.get("value"),
                        "path": best.get("path"),
                        "score_mean": best.get("score_mean"),
                        "lines_mean": best.get("lines_mean"),
                        "pieces_mean": best.get("pieces_mean"),
                    },
                    "best_training_score": run.training.get("best_score"),
                    "elapsed_seconds": run.elapsed_seconds,
                }
            )

        summary: Dict[str, Any] = {}
        for name in metrics:
            firsts = [float((row["first"] or {}).get(name) or 0.0) for row in per_seed]
            tails = [float((row["tail"] or {}).get(name) or 0.0) for row in per_seed]
            deltas = [float(row["deltas"].get(name, 0.0)) for row in per_seed]
            improved = int(sum(1 for delta in deltas if delta > 0))
            mean_delta = float(np.mean(deltas)) if deltas else 0.0
            summary[name] = {
                "first_mean": float(np.mean(firsts)) if firsts else 0.0,
                "tail_mean": float(np.mean(tails)) if tails else 0.0,
                "delta_mean": mean_delta,
                "delta_sd": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                "deltas": deltas,
                "improved_seeds": improved,
                "n_seeds": len(per_seed),
                "verdict": _verdict(mean_delta, improved, len(per_seed)),
            }

        episodes_total = sum(run.episodes_trained for run in runs)
        return {
            "experiment": self.name,
            "variant": self.variant,
            "metric": self.metric,
            "episodes": self.config.episodes,
            "seeds": [run.seed for run in sorted(runs, key=lambda item: item.seed)],
            "eval": {
                "every": self.eval_every,
                "games": self.eval_games,
                "seed": self.eval_seed,
                "max_pieces": self.eval_max_pieces,
                "tail_window": window,
            },
            "per_seed": per_seed,
            "metrics": summary,
            "verdict": _verdict(
                float(summary[self.metric]["delta_mean"]),
                int(summary[self.metric]["improved_seeds"]),
                len(per_seed),
            ),
            "training": {
                "episodes_total": episodes_total,
                "decisions_total": sum(int(run.training.get("total_decisions", 0)) for run in runs),
                "learn_steps_total": sum(int(run.training.get("total_learn_steps", 0))
                                         for run in runs),
                "elapsed_seconds": round(sum(run.elapsed_seconds for run in runs), 3),
                "best_training_score": max(
                    (int(run.training.get("best_score", 0)) for run in runs), default=0
                ),
            },
            "config": self.config.to_dict(),
            "config_id": self.config.config_id,
            "note": self.note,
            "environment": environment_info(),
            "started_at": min((run.started_at for run in runs), default=""),
            "finished_at": timestamp(),
            "elapsed_seconds": round(sum(run.elapsed_seconds for run in runs), 3),
        }


def _verdict(mean_delta: float, improved: int, n_seeds: int) -> str:
    """The words that are allowed to describe the numbers (see `aggregate`)."""
    if n_seeds and improved == n_seeds and mean_delta > 0:
        return "improving"
    if mean_delta > 0:
        return "mixed"
    return "flat-or-down"


def _run_seed_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Module-level entry point for one seed in its own process."""
    import torch

    threads = payload.get("threads")
    if threads:
        torch.set_num_threads(int(threads))
    campaign = Campaign.from_spec(payload["spec"])
    return campaign.run_seed(int(payload["seed"]), logger=None).as_dict()


def run_campaigns(
    campaigns: Iterable[Campaign],
    parallel: int = 1,
    logger: Optional[Any] = print,
) -> List[CampaignResult]:
    """Run several campaigns one after another, returning every result."""
    results = []
    for campaign in campaigns:
        if logger is not None:
            logger("")
            logger("=" * 72)
        results.append(campaign.run(parallel=parallel, logger=logger))
    return results


__all__ = [
    "LOCK_NAME",
    "AGGREGATE_NAME",
    "BEST_DIR",
    "BEST_INDEX_NAME",
    "CAMPAIGN_NAME",
    "DEFAULT_METRIC",
    "METRICS_NAME",
    "SUMMARY_NAME",
    "TAIL_WINDOW",
    "TRAINING_DIR",
    "TREND_METRICS",
    "BestTracker",
    "Campaign",
    "CampaignResult",
    "SeedRun",
    "read_metrics",
    "run_campaigns",
    "split_metrics",
    "timestamp",
]
