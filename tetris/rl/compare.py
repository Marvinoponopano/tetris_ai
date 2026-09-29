"""Put the trained agent next to the baselines, on exactly the same games.

Two agents can only be compared on identical episodes, so this module runs every
one of them through the project's own experiment framework
(`tetris.experiment.run_experiment`) with the same config, the same seed and the
same piece cap -- which is the seed the in-training evaluations use too, so a
training curve and this table are the same experiment.

    random              picks a legal placement uniformly      (the floor)
    heuristic           scores every placement, no lookahead    (a strong planner)
    lookahead           the same, one piece ahead              (the ceiling)
    dqn                 a saved checkpoint, greedy

Three things make the numbers worth reading:

* **Same piece sequences.** Game *i* is `TetrisEnv(seed=seed + i)` for every
  agent, so the comparison is paired game by game, not just on the averages.
* **Paired deltas, not just means.** For each game the module reports
  `dqn - baseline`; a mean difference over 20 *different* games is a much weaker
  claim than a mean difference over the same 20 games, especially for a baseline
  as noisy as the random agent.
* **A sign test.** With a handful of games, the honest question is "how often
  did the agent win?" -- so a two-sided exact binomial p-value is reported next
  to the win/loss counts. It is not a substitute for more games, and it is
  labelled as what it is.
"""

from __future__ import annotations

import json
import math
import pathlib
import statistics
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union  # noqa: UP035

import torch

from ..config import ExperimentConfig, environment_info
from ..evaluate import DEFAULT_MAX_PIECES
from ..experiment import ExperimentResult, compare_runs, run_experiment, timestamp
from .evaluation import DEFAULT_EVAL_SEED

Path = Union[str, pathlib.Path]

#: Games per agent in the baseline comparison. The in-training evaluations use a
#: prefix of these same seeds (`eval_games`), so nothing here is a new protocol.
DEFAULT_COMPARISON_GAMES = 20

#: The baselines, in the order they are run: label -> ExperimentConfig fields.
BASELINES: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("random", {"agent": "random"}),
    ("heuristic", {"agent": "heuristic", "lookahead_depth": 0}),
    ("lookahead", {"agent": "heuristic", "lookahead_depth": 1}),
)


@dataclass(frozen=True)
class ComparisonProtocol:
    """What every agent in a comparison is measured with."""

    num_games: int = DEFAULT_COMPARISON_GAMES
    seed: int = DEFAULT_EVAL_SEED
    max_pieces: Optional[int] = DEFAULT_MAX_PIECES

    def as_dict(self) -> Dict[str, Any]:
        return {
            "num_games": self.num_games,
            "seed": self.seed,
            "max_pieces": self.max_pieces,
        }

    def describe(self) -> str:
        return (f"{self.num_games} games, seed {self.seed}, "
                f"cap {self.max_pieces if self.max_pieces else 'none'}")


def sign_test_p(wins: int, losses: int) -> Optional[float]:
    """Two-sided exact binomial p-value for `wins` out of `wins + losses`.

    Ties are dropped, as a sign test does. `None` when there is nothing to test.
    """
    total = int(wins) + int(losses)
    if total == 0:
        return None
    extreme = max(int(wins), int(losses))
    tail = sum(math.comb(total, k) for k in range(extreme, total + 1)) / (2 ** total)
    return min(1.0, 2 * tail)


def paired_comparison(candidate: ExperimentResult, baseline: ExperimentResult) -> Dict[str, Any]:
    """Per-game deltas between two results that played the same seeds.

    Games are matched by index, which is the same as matching by seed inside an
    evaluation (`tetris.evaluate`): game *i* is `seed + i` for both agents.
    """
    base = {game.game: game for game in baseline.games}
    shared = [game for game in candidate.games if game.game in base]
    if not shared:
        return {"games": 0}

    score_deltas = [game.score - base[game.game].score for game in shared]
    lines_deltas = [game.lines_cleared - base[game.game].lines_cleared for game in shared]
    pieces_deltas = [game.pieces - base[game.game].pieces for game in shared]
    wins = sum(1 for delta in score_deltas if delta > 0)
    losses = sum(1 for delta in score_deltas if delta < 0)
    return {
        "baseline": baseline.agent,
        "games": len(shared),
        "score_delta_mean": round(statistics.fmean(score_deltas), 4),
        "score_delta_median": float(statistics.median(score_deltas)),
        "score_delta_sd": round(statistics.stdev(score_deltas), 4)
        if len(score_deltas) > 1 else 0.0,
        "score_wins": wins,
        "score_losses": losses,
        "score_ties": len(shared) - wins - losses,
        "sign_test_p": sign_test_p(wins, losses),
        "lines_delta_mean": round(statistics.fmean(lines_deltas), 4),
        "pieces_delta_mean": round(statistics.fmean(pieces_deltas), 4),
    }


def checkpoint_facts(path: Path) -> Dict[str, Any]:
    """The training facts a checkpoint carries (episode, its evaluation, seed)."""
    try:
        payload = torch.load(pathlib.Path(path), map_location="cpu", weights_only=False)
    except Exception:                                  # a file that is not a checkpoint
        return {}
    evaluation = payload.get("evaluation") or {}
    return {
        "checkpoint": str(path),
        "episode": payload.get("episode"),
        "decisions": payload.get("decisions"),
        "learn_steps": payload.get("steps"),
        "seed": (payload.get("config") or {}).get("seed"),
        "config_id": payload.get("config_id"),
        "best_metric": payload.get("best_metric"),
        "best_value": payload.get("best_value"),
        "eval_score_mean": evaluation.get("score_mean"),
        "eval_lines_mean": evaluation.get("lines_mean"),
    }


def best_checkpoints(directory: Path, prefix: Optional[str] = None) -> Dict[str, Path]:
    """The best checkpoint of every seed under a campaign directory.

    Reads each `seed*/best/best.json` (written by `tetris.rl.campaign`) and takes
    the checkpoint it points at, so "the trained agent" is defined by the run's
    own record rather than by a glob.
    """
    root = pathlib.Path(directory)
    found: Dict[str, Path] = {}
    for index in sorted(root.glob("seed*/best/best.json")):
        record = json.loads(index.read_text(encoding="utf-8"))
        best = record.get("best") or {}
        path = best.get("path")
        if not path:
            continue
        seed = index.parent.parent.name
        label = f"{prefix or root.name}-{seed}"
        found[label] = pathlib.Path(path)
    return found


def compare_agents(
    checkpoints: Optional[Dict[str, Path]] = None,
    protocol: Optional[ComparisonProtocol] = None,
    output_dir: Path = "experiments/rl/comparison",
    baselines: Sequence[str] = ("random", "heuristic", "lookahead"),
    primary_baseline: str = "random",
    logger: Optional[Any] = print,
) -> Dict[str, Any]:
    """Evaluate every agent -- baselines and checkpoints -- and pair them up.

    Returns the whole comparison as a dict and writes it next to its CSV: the
    protocol, every agent's numbers and per-game rows, the paired deltas against
    `primary_baseline`, and where the runs' own JSON went.
    """
    protocol = protocol or ComparisonProtocol()
    directory = pathlib.Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoints = {label: pathlib.Path(path) for label, path in (checkpoints or {}).items()}
    results: Dict[str, ExperimentResult] = {}
    facts: Dict[str, Dict[str, Any]] = {}

    for label, fields in BASELINES:
        if label not in baselines:
            continue
        if logger is not None:
            logger(f"[comparison] {label}: {protocol.describe()}")
        config = ExperimentConfig(**fields, num_games=protocol.num_games,
                                  seed=protocol.seed, max_pieces=protocol.max_pieces,
                                  label=f"comparison baseline {label}")
        results[label] = run_experiment(config, save_dir=directory, logger=None)

    for label, path in checkpoints.items():
        if logger is not None:
            logger(f"[comparison] {label}: {protocol.describe()} ({path.name})")
        config = ExperimentConfig(agent="dqn", checkpoint=str(path),
                                  num_games=protocol.num_games, seed=protocol.seed,
                                  max_pieces=protocol.max_pieces, label=label)
        results[label] = run_experiment(config, save_dir=directory, logger=None)
        facts[label] = checkpoint_facts(path)

    if primary_baseline not in results:
        raise ValueError(
            f"the primary baseline {primary_baseline!r} was not run; "
            f"choose from {sorted(results)}"
        )

    baseline = results[primary_baseline]
    paired = {
        label: paired_comparison(result, baseline)
        for label, result in results.items()
        if label != primary_baseline
    }

    rows: List[Dict[str, Any]] = []
    for label, result in results.items():
        stats = dict(result.stats)
        rows.append(
            {
                "label": label,
                "agent": result.agent,
                "run_id": result.run_id,
                "config_id": result.config_id,
                "stats": stats,
                "diagnostics": dict(result.diagnostics),
                "timing": dict(result.timing),
                "games": [game.as_dict() for game in result.games],
                "result_path": result.result_path,
                "paired_vs_primary": paired.get(label),
                "checkpoint_facts": facts.get(label),
            }
        )

    payload: Dict[str, Any] = {
        "protocol": protocol.as_dict(),
        "primary_baseline": primary_baseline,
        "labels": list(results),
        "agents": {row["label"]: row for row in rows},
        "table": format_comparison(rows),
        "experiment_table": compare_runs(list(results.values())),
        "environment": environment_info(),
        "written_at": timestamp(),
    }
    _write(directory / "comparison.json", payload)
    _write_csv(directory / "comparison.csv", rows)
    if logger is not None:
        logger("")
        logger(payload["table"])
    return payload


def format_comparison(rows: Sequence[Dict[str, Any]]) -> str:
    """The comparison table: one line per agent, with the paired deltas.

    Every column the evaluation protocol records is here, because they answer
    different questions and can disagree: a *mean* score is dragged around by one
    long game, the *median* says what a typical game looks like, and the *max* is
    the best game the policy ever played -- which is where a policy that
    occasionally finds a line clear shows up even while its mean is flat. Lines
    and pieces (the game length) get the same treatment.
    """
    header = (
        f"{'agent':>22} {'games':>6} {'score mean':>12} {'sd':>9} {'median':>9} "
        f"{'max':>9} {'lines mean':>11} {'lines max':>10} {'lines/100':>10} "
        f"{'pieces':>8} {'pieces max':>11} "
        f"{'d_score':>10} {'wins':>8} {'p':>7}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        stats = row.get("stats") or {}
        diagnostics = row.get("diagnostics") or {}
        paired = row.get("paired_vs_primary") or {}
        wins = (f"{paired.get('score_wins', 0)}/{paired.get('score_losses', 0)}"
                if paired else "-")
        p_value = paired.get("sign_test_p")
        lines.append(
            f"{str(row.get('label'))[:22]:>22} {int(stats.get('games', 0)):>6} "
            f"{stats.get('score_mean', 0.0):>12,.1f} {stats.get('score_stdev', 0.0):>9,.1f} "
            f"{stats.get('score_median', 0.0):>9,.0f} "
            f"{stats.get('score_max', 0.0):>9,.0f} "
            f"{stats.get('lines_mean', 0.0):>11,.2f} "
            f"{stats.get('lines_max', 0.0):>10,.0f} "
            f"{diagnostics.get('lines_per_100_pieces', 0.0):>10,.2f} "
            f"{stats.get('pieces_mean', 0.0):>8,.1f} "
            f"{stats.get('pieces_max', 0.0):>11,.0f} "
            f"{paired.get('score_delta_mean', 0.0):>10,.1f} "
            f"{wins:>8} "
            f"{('%.3f' % p_value) if p_value is not None else '-':>7}"
        )
    lines.append(
        "score/lines/pieces: mean, median, max over the games; lines/100 = lines per "
        "100 pieces (efficiency, so a long game cannot inflate it);\n"
        "d_score = mean per-game score difference against the primary baseline "
        "(same piece sequences); wins/losses = games won/lost on score; p = two-sided "
        "sign test"
    )
    return "\n".join(lines)


def _write(path: Path, payload: Any) -> str:
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return str(target)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> str:
    import csv

    # every metric the evaluation protocol records, so the CSV behind the table
    # carries the same columns the table prints (max score and max pieces were
    # missing here while the stats had them)
    fields = ("label", "agent", "games", "score_mean", "score_median", "score_stdev",
              "score_max", "lines_mean", "lines_max", "pieces_mean", "pieces_max",
              "lines_per_100_pieces", "score_delta_mean", "score_wins", "score_losses",
              "sign_test_p", "run_id")
    target = pathlib.Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            stats = row.get("stats") or {}
            diagnostics = row.get("diagnostics") or {}
            paired = row.get("paired_vs_primary") or {}
            writer.writerow(
                {
                    "label": row.get("label"),
                    "agent": row.get("agent"),
                    "games": stats.get("games"),
                    "score_mean": stats.get("score_mean"),
                    "score_median": stats.get("score_median"),
                    "score_stdev": stats.get("score_stdev"),
                    "score_max": stats.get("score_max"),
                    "lines_mean": stats.get("lines_mean"),
                    "lines_max": stats.get("lines_max"),
                    "pieces_mean": stats.get("pieces_mean"),
                    "lines_per_100_pieces": diagnostics.get("lines_per_100_pieces"),
                    "score_delta_mean": paired.get("score_delta_mean"),
                    "score_wins": paired.get("score_wins"),
                    "score_losses": paired.get("score_losses"),
                    "sign_test_p": paired.get("sign_test_p"),
                    "run_id": row.get("run_id"),
                }
            )
    return str(target)


__all__ = [
    "BASELINES",
    "DEFAULT_COMPARISON_GAMES",
    "ComparisonProtocol",
    "best_checkpoints",
    "checkpoint_facts",
    "compare_agents",
    "format_comparison",
    "paired_comparison",
    "sign_test_p",
]
