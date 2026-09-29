"""The training loop: episodes in, a log file and checkpoints out.

One episode is one game. The loop is the standard interaction cycle -- observe,
choose, act, store, learn -- with three details that matter:

1. **Every episode is seeded**: game *i* uses `TetrisEnv.reset(seed=seed + i)`.
   Episodes therefore differ from one another but the whole run is reproducible:
   the same config and seed produce the same piece sequences, the same
   exploration, the same gradient steps. Without that, a training run is an
   anecdote.
2. **Learning starts after a warm-up** (`min_buffer_size` transitions). The first
   updates otherwise fit a handful of near-identical positions, which is how a
   DQN learns to be confidently wrong.
3. **Truncation is not termination.** Reaching `max_pieces` ends the episode but
   is not a game over, so its last transition still bootstraps (`done=0`); only
   a real game over sets `done=1`.

What is written, and where:

    <output_dir>/checkpoint_<episode>.pt   weights + optimizer + counters (resumable)
    <output_dir>/train_log.csv             one row per episode
    <output_dir>/training_summary.json     config, provenance, outcome
    <output_dir>/config.json               the config as used, for the record

Do not expect a good player out of a short run. The point of this stage is a
pipeline that provably learns *something*: the loss falls, the reward per episode
drifts up, and a checkpoint reloads into an agent that plays greedy without
crashing.
"""

from __future__ import annotations

import csv
import json
import pathlib
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, Union  # noqa: UP035

import numpy as np
import torch

from ..config import ConfigLike, environment_info
from ..env import TetrisEnv
from .config import TrainingConfig
from .dqn import DQNAgent
from .encoding import mask_for_state, placement_for_index

Path = Union[str, pathlib.Path]

#: The columns of the training log, one row per episode.
LOG_FIELDS = (
    "episode",
    "reward",
    "score",
    "lines",
    "pieces",
    "epsilon",
    "loss_mean",
    "loss_last",
    "learn_steps",
    "decisions",
    "duration_seconds",
    "game_over",
    "valid_actions",
)

CHECKPOINT_NAME = "checkpoint_{episode}.pt"
LOG_NAME = "train_log.csv"
SUMMARY_NAME = "training_summary.json"
CONFIG_NAME = "config.json"


def set_seeds(seed: Optional[int]) -> None:
    """Seed python, numpy and torch so a run is reproducible end to end.

    `torch.use_deterministic_algorithms` is deliberately *not* enabled: it can
    refuse perfectly ordinary operations on CPU for a reproducibility guarantee
    these models do not need (the run is already deterministic on one machine
    with one thread count).
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


@dataclass
class TrainingResult:
    """What a training run produced: the config, the episodes, the files."""

    config: TrainingConfig
    episodes: List[Dict[str, Any]]
    checkpoints: List[str]
    output_dir: str
    log_path: str
    summary_path: str
    started_at: str
    finished_at: str
    elapsed_seconds: float
    total_decisions: int
    total_learn_steps: int
    #: The trained agent itself -- what a caller usually wants next. It is not
    #: part of `as_dict()`: the summary describes the run, not the weights.
    agent: Any = None
    environment: Dict[str, Any] = field(default_factory=dict)

    @property
    def best_score(self) -> int:
        return max((int(row["score"]) for row in self.episodes), default=0)

    @property
    def final_epsilon(self) -> float:
        return float(self.episodes[-1]["epsilon"]) if self.episodes else 0.0

    def reward_trend(self, window: int = 10) -> Tuple[float, float]:
        """Mean reward of the first and last `window` episodes: is it improving?"""
        rewards = [float(row["reward"]) for row in self.episodes]
        if not rewards:
            return 0.0, 0.0
        size = max(1, min(int(window), len(rewards) // 2 or 1))
        return (
            float(np.mean(rewards[:size])),
            float(np.mean(rewards[-size:])),
        )

    def as_dict(self) -> Dict[str, Any]:
        first, last = self.reward_trend()
        return {
            "config": self.config.to_dict(),
            "config_id": self.config.config_id,
            "description": self.config.describe(),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "episodes_trained": len(self.episodes),
            "total_decisions": self.total_decisions,
            "total_learn_steps": self.total_learn_steps,
            "best_score": self.best_score,
            "final_epsilon": self.final_epsilon,
            "reward_first_window": round(first, 3),
            "reward_last_window": round(last, 3),
            "checkpoints": list(self.checkpoints),
            "output_dir": self.output_dir,
            "log": self.log_path,
            "environment": self.environment,
        }

    def save(self, path: Optional[Path] = None) -> str:
        """Write the summary JSON (defaults to the run's own summary path)."""
        target = pathlib.Path(path or self.summary_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")
        return str(target)

    def summary(self) -> str:
        first, last = self.reward_trend()
        arrow = "up" if last > first else "flat/down"
        lines = [
            f"trained {len(self.episodes)} episode(s) in {self.elapsed_seconds:.1f}s",
            f"  config         {self.config.describe()}",
            f"  reward         {first:8.2f} -> {last:8.2f} (first/last window, {arrow})",
            f"  best score     {self.best_score}",
            f"  decisions      {self.total_decisions:,}   "
            f"gradient steps {self.total_learn_steps:,}",
            f"  epsilon        {self.final_epsilon:.3f}",
            f"  checkpoints    {len(self.checkpoints)} in {self.output_dir}",
            f"  log            {self.log_path}",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


class EpisodeLogger:
    """Appends one row per episode to a CSV, flushing as it goes.

    Flushing every row is what makes the file usable while the run is still
    going -- a run that crashes at episode 900 still has 900 rows on disk.
    """

    def __init__(self, path: Path) -> None:
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=list(LOG_FIELDS))
        if self.path.stat().st_size == 0:
            self._writer.writeheader()
            self._handle.flush()

    def append(self, row: Dict[str, Any]) -> None:
        self._writer.writerow({name: row.get(name) for name in LOG_FIELDS})
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "EpisodeLogger":
        return self

    def __exit__(self, *exception) -> None:
        self.close()


def train(
    config: Optional[ConfigLike] = None,
    agent: Optional[DQNAgent] = None,
    output_dir: Path = "checkpoints",
    resume_from: Optional[Path] = None,
    logger: Optional[Callable[[str], None]] = print,
    on_episode: Optional[Callable[[int, Dict[str, Any], Any], None]] = None,
    on_start: Optional[Callable[[Any], None]] = None,
) -> TrainingResult:
    """Run a DQN training session described by `config`.

    `agent` may be passed in (anything with the `DQNAgent` interface, e.g. a
    custom network); otherwise one is built from the config, or loaded from
    `resume_from` -- in which case training continues from the episode that
    checkpoint stopped at and writes no duplicate log rows.

    `on_episode` is an optional hook, called as `on_episode(episode, row, agent)`
    after every episode with the very row that was just written to the log. It
    is the seam a campaign uses to pause and evaluate the agent on a schedule
    (see `tetris.rl.campaign`). It runs last, after that episode's log line,
    checkpoint and progress line, so whatever it costs lands in the run's total
    time but never in a row's `duration_seconds`.

    `on_start` is called once with the finished agent, before the first episode
    -- the only moment at which the *untrained* policy can be measured, which is
    what "before" has to mean in a before/after comparison.

    Both hooks must leave training alone: the replay buffer, the counters, the
    RNG and the weights training sees after they return have to be the ones it
    would have had anyway.
    """
    if resume_from is not None:
        agent = DQNAgent.load(resume_from)
    if agent is not None:
        # Weights (and, on a resume, the optimizer's state) come from the agent;
        # the config describes *this* run, so an explicit one wins and the agent
        # is told to follow it. Without one, the agent's own config is the run's.
        if config is not None:
            agent.adopt(TrainingConfig.from_any(config))
        config = agent.config
    else:
        config = TrainingConfig() if config is None else TrainingConfig.from_any(config)
        set_seeds(config.seed)
        agent = DQNAgent(
            config=config, device=_resolve_device(config.device), seed=int(config.seed or 0)
        )

    directory = pathlib.Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    env = TetrisEnv(seed=config.seed, reward=config.reward_config())
    # The agent builds its own buffer: the reference agent gets the uniform one
    # it always got, and an arm that replays by priority gets that instead (see
    # the seams on `DQNAgent`).
    buffer = agent.build_buffer(config)
    # Sampling uses the agent's own generator: it is saved in checkpoints, so a
    # resumed run continues the same random stream instead of restarting it.
    rng = getattr(agent, "rng", None) or np.random.default_rng(config.seed)

    log_path = directory / LOG_NAME
    summary_path = directory / SUMMARY_NAME
    (directory / CONFIG_NAME).write_text(config.to_json(), encoding="utf-8")

    started_at = _timestamp()
    started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    checkpoints: List[str] = []
    total_decisions = 0
    total_learn_steps = 0
    first_episode = agent.episode

    # A *fresh* run replaces the episode log instead of appending to it: a re-run
    # of a seed must not leave the previous attempt's rows above the new ones,
    # which is what "a re-run overwrites its artifacts" has to mean for a log
    # whose rows are read back for episode curves. A *resumed* run
    # (`first_episode > 0`) still appends, because its earlier rows are its own.
    if first_episode == 0 and log_path.exists():
        log_path.unlink()

    if logger is not None:
        logger(f"training for {config.episodes - first_episode} more episode(s) "
               f"({config.describe()})")

    if on_start is not None:
        on_start(agent)

    with EpisodeLogger(log_path) as episode_log:
        for episode in range(first_episode, config.episodes):
            state = env.reset(seed=_episode_seed(config.seed, episode))
            episode_started = time.perf_counter()
            total_reward = 0.0
            losses: List[float] = []
            learn_steps_before = agent.steps
            decisions_before = agent.decisions
            legal_counts: List[int] = []

            while True:
                mask = mask_for_state(state)
                encoded = agent.encode(state)
                legal_counts.append(int(mask.sum()))
                action = agent.explore_action(encoded, mask, rng)
                next_state, reward, done, info = env.step(placement_for_index(action))

                next_mask = mask_for_state(next_state)
                if not next_mask.any() and not done:
                    raise RuntimeError(
                        f"episode {episode}: no legal action left but the game is not over; "
                        "the environment and the action mask disagree"
                    )

                # `done` is the *game over* flag: the piece cap truncates the
                # episode but leaves a future worth bootstrapping from.
                agent.remember(
                    buffer, encoded, action, reward, agent.encode(next_state), next_mask, done
                )
                agent.note_decision()

                if len(buffer) >= config.min_buffer_size:
                    losses.append(agent.train_step(agent.sample_minibatch(buffer, rng)))

                total_reward += float(reward)
                state = next_state
                if done or info["pieces_placed"] >= config.max_pieces:
                    break

            # Both exits land here: game over and the piece cap. An agent that
            # looks several steps ahead flushes its window, so no transition is
            # ever built across two episodes.
            agent.end_episode(buffer)

            row = {
                "episode": episode + 1,
                "reward": round(total_reward, 4),
                "score": int(info["score"]),
                "lines": int(env.game.lines_cleared),
                "pieces": int(info["pieces_placed"]),
                "epsilon": round(agent.epsilon, 6),
                "loss_mean": round(float(np.mean(losses)), 6) if losses else None,
                "loss_last": round(float(losses[-1]), 6) if losses else None,
                "learn_steps": int(agent.steps - learn_steps_before),
                "decisions": int(agent.decisions - decisions_before),
                "duration_seconds": round(time.perf_counter() - episode_started, 4),
                "game_over": bool(done),
                "valid_actions": round(float(np.mean(legal_counts)), 2) if legal_counts else 0.0,
            }
            rows.append(row)
            episode_log.append(row)
            total_decisions += row["decisions"]
            total_learn_steps += row["learn_steps"]

            agent.episode = episode + 1
            if (episode + 1) % config.checkpoint_every == 0 or episode + 1 == config.episodes:
                path = agent.save(directory / CHECKPOINT_NAME.format(episode=episode + 1))
                checkpoints.append(path)
                if logger is not None:
                    logger(f"  checkpoint {path}")

            if logger is not None and (episode + 1) % config.log_every == 0:
                loss_text = f"{row['loss_mean']:.4f}" if row["loss_mean"] is not None else "  --  "
                logger(
                    f"  episode {episode + 1:>5}/{config.episodes}  "
                    f"reward {row['reward']:>9.1f}  score {row['score']:>7}  "
                    f"lines {row['lines']:>4}  pieces {row['pieces']:>4}  "
                    f"eps {row['epsilon']:.3f}  loss {loss_text}"
                )

            if on_episode is not None:
                on_episode(episode + 1, row, agent)

    elapsed = time.perf_counter() - started
    result = TrainingResult(
        config=config,
        episodes=rows,
        checkpoints=checkpoints,
        output_dir=str(directory),
        log_path=str(log_path),
        summary_path=str(summary_path),
        started_at=started_at,
        finished_at=_timestamp(),
        elapsed_seconds=elapsed,
        total_decisions=total_decisions,
        total_learn_steps=total_learn_steps,
        agent=agent,
        environment=dict(environment_info(), torch_version=str(torch.__version__)),
    )
    result.save(summary_path)
    if logger is not None:
        logger("")
        logger(str(result))
    return result


def _episode_seed(seed: Optional[int], episode: int) -> Optional[int]:
    """The seed of one episode: the run's seed, offset by the episode number."""
    return None if seed is None else int(seed) + int(episode)


def _resolve_device(name: str) -> str:
    """`auto` -> mps/cuda when the build has it, otherwise cpu."""
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


__all__ = [
    "CHECKPOINT_NAME",
    "CONFIG_NAME",
    "LOG_FIELDS",
    "LOG_NAME",
    "SUMMARY_NAME",
    "EpisodeLogger",
    "TrainingResult",
    "set_seeds",
    "train",
]
