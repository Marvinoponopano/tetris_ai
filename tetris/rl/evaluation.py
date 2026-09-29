"""Evaluate a trained or half-trained agent the way training cannot: greedily.

Training reward is noisy and is measured while the agent is still exploring, so
it is a poor answer to "is this getting better?". The answer this module
produces is a *separate*, quieter measurement:

    pause training at episode N
        -> freeze exploration (epsilon = 0, the greedy policy)
        -> play a fixed set of seeded games
        -> score, lines, pieces  ->  one evaluation row
        -> restore epsilon and carry on

Three rules make those rows comparable to each other and to the baselines:

* **No exploration.** `epsilon` is forced to 0 for the duration of the
  evaluation and put back afterwards, so the number is the policy the agent
  would actually play, not its exploration.
* **Fixed seeds.** Every evaluation uses the same `seed`, and game *i* is
  `TetrisEnv(seed=seed + i)` (`tetris.evaluate`), so evaluation 1 and evaluation
  12 play byte-identical piece sequences. A difference between two rows is a
  difference in the agent, never in the luck of the deal.
* **The same piece cap as the baselines** (`max_pieces`), so a score here can be
  put next to `random` / `heuristic` from `tetris.experiment` without adjusting
  anything.

The evaluation does not perturb training: it reads the weights, never writes
them, uses its own environment, and leaves the counters and the epsilon schedule
untouched. There is a test that asserts a run with evaluations interleaved is
bit-for-bit the same as a run without them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple  # noqa: UP035

from ..evaluate import DEFAULT_MAX_PIECES, GameResult, evaluate_agent

#: The seed every evaluation uses unless told otherwise. Shared on purpose: the
#: in-training evaluations and the final baseline comparison are one protocol,
#: so a curve and a table can be read together.
DEFAULT_EVAL_SEED = 20260928

#: Games per evaluation. Small enough to run every few hundred episodes, large
#: enough that the game-to-game spread does not hide a trend.
DEFAULT_EVAL_GAMES = 8

#: The columns of an evaluation row, in the order they are written out.
EVAL_FIELDS = (
    "episode",
    "decisions",
    "learn_steps",
    "epsilon",
    "train_seconds",
    "games",
    "seed",
    "max_pieces",
    "score_mean",
    "score_median",
    "score_max",
    "score_stdev",
    "lines_mean",
    "lines_max",
    "pieces_mean",
    "pieces_max",
    "lines_per_100_pieces",
    "games_capped",
    "eval_seconds",
)


@dataclass(frozen=True)
class EvaluationSnapshot:
    """One greedy evaluation of an agent, with the context that produced it.

    `episode` is the number of training episodes finished when this ran (None
    for an evaluation of an untrained or already-trained policy), which is the
    x-axis of every "over time" curve.
    """

    stats: Dict[str, float]
    num_games: int
    seed: int
    max_pieces: Optional[int] = DEFAULT_MAX_PIECES
    episode: Optional[int] = None
    decisions: int = 0
    learn_steps: int = 0
    train_seconds: float = 0.0
    #: The exploration rate the agent *had* (the schedule's value), restored
    #: after the evaluation. The evaluation itself ran at epsilon 0.
    training_epsilon: float = 0.0
    elapsed_seconds: float = 0.0
    games: Tuple[GameResult, ...] = field(default=())

    # ------------------------------------------------------------------ values

    def metric(self, name: str, default: float = 0.0) -> float:
        """One number out of the stats (0.0 when it is missing)."""
        return float(self.stats.get(name, default))

    @property
    def score_mean(self) -> float:
        return self.metric("score_mean")

    @property
    def lines_mean(self) -> float:
        return self.metric("lines_mean")

    @property
    def pieces_mean(self) -> float:
        return self.metric("pieces_mean")

    def row(self) -> Dict[str, Any]:
        """The flat row: what a `metrics.jsonl` line for this evaluation holds."""
        row: Dict[str, Any] = {
            "episode": self.episode,
            "decisions": self.decisions,
            "learn_steps": self.learn_steps,
            "epsilon": round(self.training_epsilon, 6),
            "train_seconds": round(self.train_seconds, 3),
            "games": self.num_games,
            "seed": self.seed,
            "max_pieces": self.max_pieces,
        }
        for name in EVAL_FIELDS:
            if name in row:
                continue
            row[name] = round(self.metric(name), 6)
        row["eval_seconds"] = round(self.elapsed_seconds, 4)
        return {name: row.get(name) for name in EVAL_FIELDS}

    def as_dict(self) -> Dict[str, Any]:
        """The row plus the per-game detail (what a saved evaluation holds)."""
        data = dict(self.row())
        data["per_game"] = [game.as_dict() for game in self.games]
        return data

    def summary(self) -> str:
        """A few lines: the numbers, and what they were measured on."""
        episode = "untrained" if self.episode is None else f"after episode {self.episode}"
        lines = [
            f"evaluation ({episode}, greedy) -- {self.num_games} games, "
            f"seed {self.seed}, cap {self.max_pieces}",
            f"  score          mean {self.metric('score_mean'):10,.1f}   "
            f"median {self.metric('score_median'):8,.0f}   "
            f"max {self.metric('score_max'):10,.0f}",
            f"  lines cleared  mean {self.lines_mean:10.2f}   "
            f"max {self.metric('lines_max'):8,.0f}   "
            f"({self.metric('lines_per_100_pieces'):.2f} per 100 pieces)",
            f"  pieces         mean {self.pieces_mean:10.1f}   "
            f"max {self.metric('pieces_max'):8,.0f}",
            f"  capped games   {self.metric('games_capped'):.0f}/{self.num_games}"
            f"   (epsilon during training {self.training_epsilon:.3f})",
            f"  took {self.elapsed_seconds:.2f}s",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def evaluate_greedy(
    agent,
    num_games: int = DEFAULT_EVAL_GAMES,
    seed: int = DEFAULT_EVAL_SEED,
    max_pieces: Optional[int] = DEFAULT_MAX_PIECES,
    *,
    episode: Optional[int] = None,
    train_seconds: float = 0.0,
    logger: Optional[Callable[[str], None]] = None,
) -> EvaluationSnapshot:
    """Score `agent` with exploration off, on fixed seeds, without changing it.

    :: 

        snapshot = evaluate_greedy(agent, num_games=10)
        snapshot.score_mean, snapshot.lines_mean        # the numbers

    Any agent with `choose_action` / `name` works; a `DQNAgent` also contributes
    its decision and gradient-step counters, which is what makes the row
    self-describing (see `EvaluationSnapshot`).
    """
    setter = getattr(agent, "set_epsilon", None)
    previous = getattr(agent, "epsilon", None)
    if callable(setter):
        setter(0.0)

    started = time.perf_counter()
    try:
        result = evaluate_agent(
            agent,
            num_games=num_games,
            seed=seed,
            max_pieces=max_pieces,
            logger=logger,
        )
    finally:
        if callable(setter) and previous is not None:
            setter(float(previous))
    elapsed = time.perf_counter() - started

    stats = dict(result.stats)
    pieces = sum(game.pieces for game in result.games)
    stats["lines_per_100_pieces"] = round(100 * stats["lines_total"] / pieces, 4) if pieces else 0.0

    return EvaluationSnapshot(
        stats=stats,
        num_games=num_games,
        seed=int(seed),
        max_pieces=max_pieces,
        episode=episode,
        decisions=int(getattr(agent, "decisions", 0)),
        learn_steps=int(getattr(agent, "steps", 0)),
        train_seconds=float(train_seconds),
        training_epsilon=float(previous if previous is not None else 0.0),
        elapsed_seconds=elapsed,
        games=tuple(result.games),
    )


__all__ = [
    "DEFAULT_EVAL_GAMES",
    "DEFAULT_EVAL_SEED",
    "EVAL_FIELDS",
    "EvaluationSnapshot",
    "evaluate_greedy",
]
