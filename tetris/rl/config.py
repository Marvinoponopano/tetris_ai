"""Training configuration: the knobs of a DQN run, in the same shape as the rest.

Like `ExperimentConfig`, this is a frozen dataclass that round-trips through
JSON, rejects unknown keys, validates at construction, and carries a
`config_id` hash of everything that can change the results::

    {
      "episodes": 200,
      "learning_rate": 0.001,
      "batch_size": 64,
      "buffer_size": 50000,
      "gamma": 0.99,
      "epsilon_start": 1.0,
      "epsilon_end": 0.05,
      "epsilon_decay_steps": 20000,
      "target_update_steps": 500,
      "min_buffer_size": 1000,
      "max_pieces": 200,
      "hidden_sizes": [256, 256],
      "seed": 0,
      "device": "cpu",
      "label": "first DQN run"
    }

`epsilon_for_step` is the exploration schedule and lives here rather than in
the training loop, so the curve is one small tested function:

    epsilon(step) = max(epsilon_end, epsilon_start - (start - end) * step / decay)

`checkpoint_every` and `log_every` are part of the file but not of `config_id`:
they decide what gets *written*, not what gets learned.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple  # noqa: UP035

from ..config import JsonConfig
from ..env import RewardConfig

#: Where the agent may run. `auto` means "mps if this torch has it, else cpu".
DEVICES = ("cpu", "mps", "cuda", "auto")

#: The weights a reward may set, from the environment's `RewardConfig`.
REWARD_FIELDS = tuple(field.name for field in dataclasses.fields(RewardConfig))


@dataclass(frozen=True)
class TrainingConfig(JsonConfig):
    """Everything a DQN training run needs to be described and reproduced.

    `episodes` is the *total* number of episodes to have trained by the end, so
    resuming from episode 100 with `episodes=300` trains 200 more. `min_buffer`
    is the warm-up: no gradient step happens until that many transitions are
    stored, which keeps the first updates from being a fit to a handful of
    near-identical states.
    """

    #: The fields of this config, in the order they are written out.
    CONFIG_FIELDS = (
        "episodes",
        "learning_rate",
        "batch_size",
        "buffer_size",
        "gamma",
        "epsilon_start",
        "epsilon_end",
        "epsilon_decay_steps",
        "target_update_steps",
        "min_buffer_size",
        "max_pieces",
        "hidden_sizes",
        "reward",
        "seed",
        "device",
        "checkpoint_every",
        "log_every",
        "label",
    )

    #: Fields that do not change what is learned, and so are left out of `config_id`.
    COSMETIC_FIELDS = ("checkpoint_every", "log_every", "label")

    episodes: int = 200
    learning_rate: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 50_000
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20_000
    target_update_steps: int = 500
    min_buffer_size: int = 1_000
    max_pieces: int = 200
    hidden_sizes: Tuple[int, ...] = (256, 256)
    reward: Optional[Dict[str, float]] = None
    seed: Optional[int] = 0
    device: str = "cpu"
    checkpoint_every: int = 100
    log_every: int = 10
    label: Optional[str] = None

    def __post_init__(self) -> None:
        self._check_positive_int("episodes", self.episodes)
        self._check_positive_int("batch_size", self.batch_size)
        self._check_positive_int("buffer_size", self.buffer_size)
        self._check_positive_int("epsilon_decay_steps", self.epsilon_decay_steps)
        self._check_positive_int("target_update_steps", self.target_update_steps)
        self._check_positive_int("min_buffer_size", self.min_buffer_size)
        self._check_positive_int("max_pieces", self.max_pieces)
        self._check_positive_int("checkpoint_every", self.checkpoint_every)
        self._check_positive_int("log_every", self.log_every)

        if not isinstance(self.learning_rate, (int, float)) or self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate!r}")
        if not isinstance(self.gamma, (int, float)) or not 0.0 < self.gamma <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {self.gamma!r}")
        for name in ("epsilon_start", "epsilon_end"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")
        if self.epsilon_end > self.epsilon_start:
            raise ValueError(
                f"epsilon_end ({self.epsilon_end}) must not exceed "
                f"epsilon_start ({self.epsilon_start}): exploration should shrink"
            )
        if self.min_buffer_size > self.buffer_size:
            raise ValueError(
                f"min_buffer_size ({self.min_buffer_size}) cannot exceed buffer_size "
                f"({self.buffer_size}): learning would never start"
            )
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError(f"seed must be an integer or null, got {self.seed!r}")
        if self.device not in DEVICES:
            raise ValueError(f"device must be one of {DEVICES}, got {self.device!r}")

        sizes = self.hidden_sizes
        if isinstance(sizes, (int, float)):
            sizes = (sizes,)
        if not sizes:
            raise ValueError("hidden_sizes must list at least one layer size")
        try:
            sizes = tuple(int(size) for size in sizes)
        except (TypeError, ValueError):
            raise ValueError(f"hidden_sizes must be whole numbers, got {self.hidden_sizes!r}")
        if any(size < 1 for size in sizes):
            raise ValueError(f"hidden sizes must be 1 or more, got {self.hidden_sizes!r}")
        object.__setattr__(self, "hidden_sizes", sizes)

        if self.reward is not None:
            unknown = sorted(set(self.reward) - set(REWARD_FIELDS))
            if unknown:
                raise ValueError(
                    f"unknown reward weight(s) {', '.join(repr(k) for k in unknown)}; "
                    f"known weights are {', '.join(REWARD_FIELDS)}"
                )
            object.__setattr__(
                self, "reward", {name: float(value) for name, value in sorted(self.reward.items())}
            )

    @staticmethod
    def _check_positive_int(name: str, value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an integer of 1 or more, got {value!r}")

    # ------------------------------------------------------------ the schedule

    def epsilon_for_step(self, step: int) -> float:
        """The exploration rate after `step` decisions: a straight line down.

        Starts at `epsilon_start`, reaches `epsilon_end` at
        `epsilon_decay_steps` and stays there afterwards.
        """
        step = max(0, int(step))
        span = self.epsilon_start - self.epsilon_end
        if self.epsilon_decay_steps <= 0 or span <= 0:
            return float(self.epsilon_end)
        fraction = min(1.0, step / float(self.epsilon_decay_steps))
        return float(self.epsilon_start - span * fraction)

    def reward_config(self) -> Optional[RewardConfig]:
        """The environment reward this config asks for (None = the default)."""
        if self.reward is None:
            return None
        return RewardConfig(**self.reward)

    def describe(self) -> str:
        """One line for a table or a log."""
        bits = [
            f"dqn {self.episodes} episodes",
            "seed none" if self.seed is None else f"seed {self.seed}",
            f"lr {self.learning_rate:g}",
            f"batch {self.batch_size}",
            f"buffer {self.buffer_size:,}",
            f"gamma {self.gamma:g}",
            f"eps {self.epsilon_start:g}->{self.epsilon_end:g} over "
            f"{self.epsilon_decay_steps:,}",
            f"target every {self.target_update_steps:,}",
            "x".join(str(size) for size in self.hidden_sizes),
            f"{self.device}",
        ]
        if self.reward is not None:
            bits.append("[" + ", ".join(f"{k}={v:g}" for k, v in self.reward.items()) + "]")
        return ", ".join(bits)

    def __str__(self) -> str:
        return self.describe()


__all__ = ["DEVICES", "REWARD_FIELDS", "TrainingConfig"]
