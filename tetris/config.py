"""Experiment configuration, and the provenance that goes with it.

A configuration is a small set of settings that can be written to a JSON file,
read back, overridden on the command line, and quoted in the results. It holds
exactly the things that change the *numbers* of a run, so two runs with the same
config and the same seed are the same experiment::

    {
      "agent": "heuristic",
      "num_games": 1000,
      "seed": 42,
      "max_pieces": 1000,
      "lookahead_depth": 1,
      "beam": 4,
      "weights": {"holes": -8.0},
      "label": "heuristic, one piece of lookahead"
    }

    from tetris.config import ExperimentConfig

    config = ExperimentConfig(agent="heuristic", num_games=1000, seed=42,
                              lookahead_depth=1)
    config.save("experiments/configs/mine.json")     # JSON out
    same = ExperimentConfig.load("experiments/configs/mine.json")

    config = ExperimentConfig.from_dict({"num_games": 10, "seed": 7})
    smaller = config.with_overrides(num_games=5, seed=8)   # a new config
    config.config_id                                        # short, stable hash
    config.describe()                                       # one line for a table

Anything presentation-only (`log_every`, where to save, whether to profile) is
deliberately *not* here: it does not change a single result, and keeping it out
is what makes `config_id` a reliable "was this the same experiment?" check.

`environment_info()` is the other half of reproducibility: what the run was made
with -- Python, platform, and the versions of the libraries it depends on.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import platform
import sys
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, Iterable, Optional, Tuple, Union  # noqa: UP035

from .evaluate import DEFAULT_MAX_PIECES
from .heuristic import DEFAULT_BEAM, HeuristicWeights

Path = Union[str, pathlib.Path]

class JsonConfig:
    """JSON round-tripping shared by the config dataclasses in this project.

    A config is a frozen dataclass with `CONFIG_FIELDS` (every field, in the
    order it is written out) and `COSMETIC_FIELDS` (the fields that do not
    change a result, such as a human label). Everything that is the same for
    every kind of config lives here: dict/JSON/path in, dict/JSON/path out,
    validated overrides, and the `config_id` hash that identifies an experiment
    by everything that *can* change its results.
    """

    CONFIG_FIELDS: ClassVar[Tuple[str, ...]] = ()
    COSMETIC_FIELDS: ClassVar[Tuple[str, ...]] = ()
    # ------------------------------------------------------------- conversion

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JsonConfig":
        """Build a config from a plain dict, rejecting unknown keys.

        Unknown keys are an error rather than being ignored: a config with a
        typo in it silently runs the wrong experiment, which is exactly the
        mistake this system exists to prevent.
        """
        if not isinstance(data, dict):
            raise ValueError(f"a config must be a dict, got {type(data).__name__}")
        unknown = sorted(set(data) - set(cls.CONFIG_FIELDS))
        if unknown:
            raise ValueError(
                f"unknown config key(s) {', '.join(repr(key) for key in unknown)}; "
                f"known keys are {', '.join(cls.CONFIG_FIELDS)}"
            )
        return cls(**data)

    @classmethod
    def from_any(cls, value: "ConfigLike") -> "JsonConfig":
        """Accept a config, a dict, or a path to a JSON config file."""
        if isinstance(value, cls):
            return value
        if isinstance(value, dict):
            return cls.from_dict(value)
        if isinstance(value, (str, pathlib.Path)):
            return cls.load(value)
        raise ValueError(
            f"expected a {cls.__name__}, a dict or a path, "
            f"got {type(value).__name__}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """The config as plain JSON-ready data, one key per field."""
        return {name: getattr(self, name) for name in self.CONFIG_FIELDS}

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False) + "\n"

    def save(self, path: Path) -> str:
        """Write the config to `path` (as JSON). Returns the path as a string."""
        target = pathlib.Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json(), encoding="utf-8")
        return str(target)

    @classmethod
    def load(cls, path: Path) -> "JsonConfig":
        """Read one config from a JSON file."""
        data = _read_json(path)
        if isinstance(data, list):
            raise ValueError(
                f"{path} holds a list of configs: use load_all() for that"
            )
        return cls.from_dict(data)

    @classmethod
    def load_all(cls, path: Path) -> Tuple["ExperimentConfig", ...]:
        """Read one config, or a whole list of them, from a JSON file."""
        data = _read_json(path)
        if isinstance(data, list):
            return tuple(cls.from_dict(item) for item in data)
        return (cls.from_dict(data),)

    def with_overrides(self, **overrides: Any) -> "JsonConfig":
        """A copy with some fields changed (and re-validated)."""
        unknown = sorted(set(overrides) - set(self.CONFIG_FIELDS))
        if unknown:
            raise ValueError(
                f"unknown config key(s) {', '.join(repr(key) for key in unknown)}; "
                f"known keys are {', '.join(self.CONFIG_FIELDS)}"
            )
        return dataclasses.replace(self, **overrides)

    # -------------------------------------------------------------- identity

    def to_result_signature(self) -> Dict[str, Any]:
        """The part of the config that decides the results (no label)."""
        return {
            name: getattr(self, name)
            for name in self.CONFIG_FIELDS
            if name not in self.COSMETIC_FIELDS
        }

    @property
    def config_id(self) -> str:
        """A short stable hash of everything that can change the results."""
        payload = json.dumps(self.to_result_signature(), sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]



@dataclass(frozen=True)
class ExperimentConfig(JsonConfig):
    """One experiment: which agent, how many games, and how it should play them.

    `agent` is a name from `tetris.agents.AGENTS` (or any label, if the agent
    object is passed in directly). `num_games` and `seed` are what the harness
    needs: game *i* is played on `TetrisEnv(seed=seed + i)`, so the same seed
    means the same piece sequences for every agent. `max_pieces` is the safety
    cap; `lookahead_depth`, `beam` and `weights` configure the heuristic agent
    and are ignored by agents that have no use for them.

    Weights are stored *complete*: `{"holes": -8}` becomes the whole weight
    vector with the defaults filled in, sorted. A saved config therefore says
    exactly which weights were used, instead of relying on what the defaults
    happened to be at the time. `HeuristicWeights` is the authority on weight
    names, so a typo fails here rather than quietly doing nothing.
    """

    #: The fields of this config, in the order they are written out.
    CONFIG_FIELDS = (
        "agent",
        "num_games",
        "seed",
        "max_pieces",
        "lookahead_depth",
        "beam",
        "weights",
        "checkpoint",
        "label",
    )

    #: Fields that do not change the results, and so are left out of `config_id`.
    COSMETIC_FIELDS = ("label",)

    agent: str = "random"
    num_games: int = 100
    seed: Optional[int] = None
    max_pieces: Optional[int] = DEFAULT_MAX_PIECES
    lookahead_depth: int = 0
    beam: int = DEFAULT_BEAM
    weights: Optional[Dict[str, float]] = None
    checkpoint: Optional[str] = None
    label: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.agent, str) or not self.agent:
            raise ValueError(f"agent must be a non-empty name, got {self.agent!r}")
        if not isinstance(self.num_games, int) or self.num_games < 1:
            raise ValueError(f"num_games must be 1 or more, got {self.num_games!r}")
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError(f"seed must be an integer or null, got {self.seed!r}")
        if self.max_pieces is not None and (
            not isinstance(self.max_pieces, int) or self.max_pieces < 1
        ):
            raise ValueError(
                f"max_pieces must be 1 or more (or null for no cap), "
                f"got {self.max_pieces!r}"
            )
        if not isinstance(self.lookahead_depth, int) or self.lookahead_depth < 0:
            raise ValueError(
                f"lookahead_depth must be 0 or more, got {self.lookahead_depth!r}"
            )
        if not isinstance(self.beam, int) or self.beam < 1:
            raise ValueError(f"beam must be 1 or more, got {self.beam!r}")
        if self.checkpoint is not None and (
            not isinstance(self.checkpoint, str) or not self.checkpoint
        ):
            raise ValueError(
                f"checkpoint must be a path (or null), got {self.checkpoint!r}"
            )

        # Weights are validated by the thing that uses them, and stored in a
        # canonical form so two equal configs hash equal.
        checked = HeuristicWeights.from_any(self.weights)
        object.__setattr__(
            self,
            "weights",
            None if self.weights is None else {name: float(v) for name, v in
                                               sorted(checked.as_dict().items())},
        )

    def _weights_caption(self) -> str:
        """The weights, shown as the differences from the defaults (or `default`)."""
        if self.weights is None:
            return "default weights"
        defaults = HeuristicWeights().as_dict()
        changed = {name: value for name, value in self.weights.items()
                   if value != defaults.get(name)}
        if not changed:
            return "default weights"
        overrides = ", ".join(f"{name}={value:g}" for name, value in sorted(changed.items()))
        return f"[{overrides}]"

    def describe(self) -> str:
        """One line for a table or a log: what this config asks for."""
        bits = [self.agent]
        if self.agent == "heuristic":
            bits.append(f"depth {self.lookahead_depth}")
            if self.lookahead_depth >= 2:
                bits.append(f"beam {self.beam}")
            bits.append(self._weights_caption())
        if self.agent == "dqn":
            name = pathlib.Path(self.checkpoint).name if self.checkpoint else "no checkpoint"
            bits.append(name)
        bits.append(f"{self.num_games} games")
        bits.append("seed none" if self.seed is None else f"seed {self.seed}")
        bits.append(f"cap {self.max_pieces}" if self.max_pieces else "no cap")
        return ", ".join(bits)

    def __str__(self) -> str:
        return self.describe()



#: Kept as module-level names for callers that import them from here.
CONFIG_FIELDS = ExperimentConfig.CONFIG_FIELDS
COSMETIC_FIELDS = ExperimentConfig.COSMETIC_FIELDS

ConfigLike = Union[ExperimentConfig, Dict[str, Any], str, pathlib.Path]


def _read_json(path: Path) -> Any:
    """Read a config file, naming the file when the JSON is broken."""
    text = pathlib.Path(path).read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as problem:
        raise ValueError(f"{path}: not valid JSON ({problem})") from problem


# --------------------------------------------------------------- the machine


def dependency_versions(names: Iterable[str] = ("numpy", "pygame")) -> Dict[str, Optional[str]]:
    """Installed versions of `names`, without importing them (None if absent).

    `importlib.metadata` reads the installed distribution, so this works for
    optional dependencies like pygame without dragging in a graphical library
    (and without printing its banner).
    """
    from importlib import metadata

    versions: Dict[str, Optional[str]] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except Exception:            # not installed, or no metadata at all
            versions[name] = None
    return versions


def environment_info() -> Dict[str, Any]:
    """What produced a run: interpreter, platform, and library versions.

    Together with the config and the seed this is what makes a result
    reproducible and comparable later -- a number without this is just a number.
    """
    info: Dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    info.update(dependency_versions())
    return info


def parse_override(text: str) -> Tuple[str, Any]:
    """Parse one `key=value` command-line override into `(key, value)`.

    The value is read as JSON when it looks like JSON (`10`, `true`, `null`,
    `{"holes": -8}`), then as an integer (`0012`), and otherwise kept as a plain
    string -- so numbers stay numbers without quoting them in the shell.
    """
    key, separator, raw = text.partition("=")
    key = key.strip()
    if not separator or not key:
        raise ValueError(f"expected key=value, got {text!r}")
    try:
        return key, json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        return key, int(raw)
    except ValueError:
        return key, raw


def loads_configs(text: str) -> Tuple[ExperimentConfig, ...]:
    """Read configs straight from a JSON string (used by the CLI)."""
    data = json.loads(text)
    items = data if isinstance(data, list) else [data]
    return tuple(ExperimentConfig.from_dict(item) for item in items)


__all__ = [
    "CONFIG_FIELDS",
    "COSMETIC_FIELDS",
    "ConfigLike",
    "ExperimentConfig",
    "dependency_versions",
    "environment_info",
    "loads_configs",
    "parse_override",
]
