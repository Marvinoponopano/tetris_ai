"""Run an agent over many complete games and collect statistics.

    from tetris.agents import RandomAgent
    from tetris.evaluate import evaluate_agent

    result = evaluate_agent(RandomAgent(seed=0), num_games=100, seed=123)
    print(result)                 # a small table
    print(result.stats)           # the numbers as a dict
    result.save("results.json")   # or results.csv

There is no pygame on this path: games are driven straight through
`tetris.env.TetrisEnv`, so hundreds or thousands of them run in seconds.

Reproducibility
---------------
`seed=` fixes the piece sequences: game *i* is played on the pieces that
`TetrisEnv(seed=seed + i)` deals, and the agent is reset with that same seed
before the game starts. Two agents evaluated with the same `seed` and
`num_games` therefore face byte-identical episodes, which is what makes their
numbers comparable. Without a seed every game is fresh randomness.

`agent.reset(seed)` is called when the agent defines it (see `tetris.agents`),
so reusing one agent object across evaluations is safe.

Game length
-----------
One action places one piece, so a game's length *is* the number of pieces it
survived (this is the same number as `pieces`); both are reported because the
two are conventionally distinct, and they will diverge only if a future
environment adds frame-level control.
"""

from __future__ import annotations

import csv
import json
import pathlib
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union  # noqa: UP035

from .env import TetrisEnv

#: Stop a game after this many pieces. A random agent never gets near it; the
#: cap exists so that a *good* agent (Tetris can be played forever) cannot hang
#: an evaluation. Capped games are flagged in the results.
DEFAULT_MAX_PIECES = 1000

Path = Union[str, pathlib.Path]

#: The per-game columns, in order -- used by the CSV writer.
GAME_FIELDS = ("game", "seed", "score", "lines_cleared", "pieces", "capped")


@dataclass(frozen=True)
class GameResult:
    """What one complete game produced."""

    game: int            # 0-based index within the evaluation
    seed: Optional[int]  # the seed of this game's piece sequence (None = unseeded)
    score: int
    lines_cleared: int
    pieces: int
    capped: bool         # True if the game was stopped by max_pieces, not by topping out

    def as_dict(self) -> Dict[str, object]:
        return {
            "game": self.game,
            "seed": self.seed,
            "score": self.score,
            "lines_cleared": self.lines_cleared,
            "pieces": self.pieces,
            "capped": self.capped,
        }


@dataclass(frozen=True)
class EvaluationResult:
    """The games an agent played, plus the statistics a comparison needs."""

    agent: str
    num_games: int
    seed: Optional[int]
    games: Tuple[GameResult, ...]
    elapsed_seconds: float
    max_pieces: Optional[int] = DEFAULT_MAX_PIECES

    @property
    def stats(self) -> Dict[str, float]:
        """Mean / median / min / max of the run, as plain Python numbers.

        Every value here is determined by the seeds alone -- no wall-clock -- so
        two runs of the same agent on the same seed compare equal, and results
        can be diffed. (`elapsed_seconds` is on the result instead, and the
        throughput in `summary()` is derived from it.)

        `game_length_*` and `pieces_*` hold the same values by construction:
        one action places one piece, so a game's length is the number of pieces
        it survived. See the module docstring.
        """
        scores = [game.score for game in self.games]
        lines = [game.lines_cleared for game in self.games]
        pieces = [game.pieces for game in self.games]
        return {
            "games": len(self.games),
            "score_mean": statistics.fmean(scores),
            "score_median": float(statistics.median(scores)),
            "score_min": float(min(scores)),
            "score_max": float(max(scores)),
            "score_stdev": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "lines_mean": statistics.fmean(lines),
            "lines_max": float(max(lines)),
            "lines_total": float(sum(lines)),
            "pieces_mean": statistics.fmean(pieces),
            "pieces_max": float(max(pieces)),
            "game_length_mean": statistics.fmean(pieces),
            "game_length_max": float(max(pieces)),
            "games_capped": float(sum(1 for game in self.games if game.capped)),
        }

    def summary(self) -> str:
        """A small table: the numbers, without the per-game detail."""
        stats = self.stats
        seed = "none" if self.seed is None else self.seed
        lines = [
            f"agent {self.agent!r} -- {self.num_games} games, seed {seed}",
            f"  score          mean {stats['score_mean']:8.1f}   median {stats['score_median']:6.0f}"
            f"   min {stats['score_min']:5.0f}   max {stats['score_max']:6.0f}"
            f"   sd {stats['score_stdev']:6.1f}",
            f"  lines cleared  mean {stats['lines_mean']:8.2f}   max    {stats['lines_max']:6.0f}",
            f"  pieces         mean {stats['pieces_mean']:8.1f}   max    {stats['pieces_max']:6.0f}",
            f"  game length    mean {stats['game_length_mean']:8.1f}   max    {stats['game_length_max']:6.0f}"
            f"   (pieces = game length here)",
        ]
        if stats["games_capped"]:
            lines.append(f"  capped games   {stats['games_capped']:.0f}/{self.num_games} reached {self.max_pieces} pieces")
        speed = len(self.games) / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0
        lines.append(f"  ran in {self.elapsed_seconds:.2f}s ({speed:.0f} games/s)")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def as_dict(self) -> Dict[str, object]:
        """The whole result, ready for `json.dump`."""
        return {
            "agent": self.agent,
            "num_games": self.num_games,
            "seed": self.seed,
            "max_pieces": self.max_pieces,
            "elapsed_seconds": self.elapsed_seconds,
            "stats": self.stats,
            "games": [game.as_dict() for game in self.games],
        }

    def save(self, path: Path) -> str:
        """Write the result to `path`; the suffix picks the format (.json/.csv)."""
        target = pathlib.Path(path)
        suffix = target.suffix.lower()
        if suffix == ".json":
            self.save_json(target)
        elif suffix == ".csv":
            self.save_csv(target)
        else:
            raise ValueError(f"unknown format {suffix!r}: use .json or .csv")
        return str(target)

    def save_json(self, path: Path) -> str:
        """Stats and every game, as one JSON object."""
        target = pathlib.Path(path)
        target.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")
        return str(target)

    def save_csv(self, path: Path) -> str:
        """One row per game (the stats are the columns, so they belong in JSON)."""
        target = pathlib.Path(path)
        with target.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(GAME_FIELDS))
            writer.writeheader()
            for game in self.games:
                writer.writerow(game.as_dict())
        return str(target)


def evaluate_agent(
    agent,
    num_games: int = 100,
    seed: Optional[int] = None,
    *,
    max_pieces: Optional[int] = DEFAULT_MAX_PIECES,
    log_every: Optional[int] = None,
    logger: Optional[Callable[[str], None]] = print,
) -> EvaluationResult:
    """Play `num_games` complete games with `agent` and collect the numbers.

    ::

        result = evaluate_agent(RandomAgent(seed=0), num_games=100, seed=123)
        result.stats["score_mean"]

    * `seed` -- fixes the piece sequences (game `i` uses `seed + i`) and the
      agent's own randomness, so different agents can be compared on identical
      games. `None` means fresh randomness every game.
    * `max_pieces` -- safety cap per game; games that hit it are flagged
      `capped` (see `DEFAULT_MAX_PIECES`). `None` plays until the game really ends.
    * `log_every` -- print progress every N games. Defaults to ~10% of the run.
    * `logger` -- where progress goes; pass `None` for silence.

    Runs without pygame, at roughly the engine's own speed: the loop calls
    `env.step()` and nothing else.
    """
    if num_games < 1:
        raise ValueError(f"num_games must be at least 1, got {num_games}")
    if max_pieces is not None and max_pieces < 1:
        raise ValueError(f"max_pieces must be at least 1 or None, got {max_pieces}")
    if log_every is None:
        log_every = max(1, num_games // 10)
    elif log_every < 1:
        raise ValueError(f"log_every must be at least 1, got {log_every}")

    env = TetrisEnv(seed=seed)
    games: List[GameResult] = []
    started = time.perf_counter()

    for index in range(num_games):
        game_seed = None if seed is None else seed + index

        reset = getattr(agent, "reset", None)
        if callable(reset):
            reset(seed=game_seed)

        state = env.reset(seed=game_seed)
        while not env.is_game_over() and env.pieces_placed < max_pieces:
            action = agent.choose_action(state, env.get_valid_actions())
            state, reward, done, info = env.step(action)

        games.append(
            GameResult(
                game=index,
                seed=game_seed,
                score=state.score,
                lines_cleared=state.lines_cleared,
                pieces=env.pieces_placed,
                capped=not env.is_game_over(),
            )
        )

        played = index + 1
        if logger is not None and (played % log_every == 0 or played == num_games):
            logger(f"Game {played}/{num_games}")

    return EvaluationResult(
        agent=getattr(agent, "name", type(agent).__name__),
        num_games=num_games,
        seed=seed,
        games=tuple(games),
        elapsed_seconds=time.perf_counter() - started,
        max_pieces=max_pieces,
    )
