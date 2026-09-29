"""Train a DQN agent to play Tetris.

    python train.py --episodes 200
    python train.py --episodes 200 --learning-rate 5e-4 --batch-size 64 --gamma 0.99
    python train.py --config configs/dqn.json
    python train.py --episodes 200 --epsilon-decay-steps 5000 --checkpoint-every 50
    python train.py --resume checkpoints/checkpoint_100.pt --episodes 300
    python train.py --resume checkpoints/checkpoint_200.pt --evaluate 20

Every run writes into `--checkpoint-dir` (default `checkpoints/`):

    checkpoint_<episode>.pt    weights + optimizer + counters, resumable
    train_log.csv              one row per episode (reward, score, lines, pieces,
                               epsilon, loss, duration)
    training_summary.json      config, provenance and outcome of the run
    config.json                the config as used

Do not expect a strong player from a short run. The goal of this stage is a
pipeline that runs end to end: loss computed, replay sampled, checkpoints
written, checkpoints reloadable, and the result measurable by the existing
evaluation framework.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import List, Optional  # noqa: UP035

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tetris.rl.config import DEVICES, TrainingConfig  # noqa: E402
from tetris.rl.dqn import load_agent_for_evaluation  # noqa: E402
from tetris.rl.training import train  # noqa: E402

DEFAULT_CHECKPOINT_DIR = ROOT / "checkpoints"


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a whole number, got {value!r}")
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {value!r}")
    return number


def positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a number, got {value!r}")
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0, got {value!r}")
    return number


def probability(value: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"must be a number, got {value!r}")
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1, got {value!r}")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="train.py", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--config", type=pathlib.Path,
                        help="a JSON TrainingConfig; individual flags override it")
    parser.add_argument("--episodes", type=positive_int,
                        help="total episodes to have trained by the end")
    parser.add_argument("--learning-rate", type=positive_float, dest="learning_rate")
    parser.add_argument("--batch-size", type=positive_int, dest="batch_size")
    parser.add_argument("--buffer-size", type=positive_int, dest="buffer_size")
    parser.add_argument("--gamma", type=positive_float)
    parser.add_argument("--epsilon-start", type=probability, dest="epsilon_start")
    parser.add_argument("--epsilon-end", type=probability, dest="epsilon_end")
    parser.add_argument("--epsilon-decay-steps", type=positive_int,
                        dest="epsilon_decay_steps")
    parser.add_argument("--target-update", type=positive_int, dest="target_update_steps",
                        help="gradient steps between target-network refreshes")
    parser.add_argument("--min-buffer", type=positive_int, dest="min_buffer_size",
                        help="transitions to collect before the first gradient step")
    parser.add_argument("--max-pieces", type=positive_int, dest="max_pieces",
                        help="cap on pieces per episode")
    parser.add_argument("--hidden", type=int, nargs="+", dest="hidden_sizes",
                        metavar="N", help="hidden layer sizes, e.g. --hidden 256 256")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=DEVICES)
    parser.add_argument("--checkpoint-every", type=positive_int, dest="checkpoint_every",
                        help="episodes between checkpoints")
    parser.add_argument("--log-every", type=positive_int, dest="log_every",
                        help="episodes between progress lines")
    parser.add_argument("--checkpoint-dir", type=pathlib.Path,
                        default=DEFAULT_CHECKPOINT_DIR, dest="checkpoint_dir")
    parser.add_argument("--resume", type=pathlib.Path,
                        help="a checkpoint to continue training from")
    parser.add_argument("--evaluate", type=positive_int, nargs="?", const=20, metavar="GAMES",
                        help="after training, evaluate the agent greedily over GAMES "
                             "games through the experiment framework (default 20)")
    parser.add_argument("--quiet", action="store_true",
                        help="no per-episode progress")
    return parser


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
    """A config from the file (if any) with the given flags applied on top."""
    overrides = {
        name: getattr(args, name)
        for name in TrainingConfig.CONFIG_FIELDS
        if name != "config" and getattr(args, name, None) is not None
    }
    if args.config is not None:
        config = TrainingConfig.load(args.config)
        return config.with_overrides(**overrides)
    return TrainingConfig.from_dict(overrides)


def evaluate_checkpoint(path, games: int, checkpoint_dir: pathlib.Path) -> str:
    """Score a saved agent with the project's own evaluation framework."""
    from tetris.config import ExperimentConfig
    from tetris.experiment import compare_runs, run_experiment

    agent = load_agent_for_evaluation(path)
    config = ExperimentConfig(
        agent="dqn",
        num_games=games,
        seed=123,
        max_pieces=1000,
        checkpoint=str(path),
        label=f"evaluated {pathlib.Path(path).name}",
    )
    result = run_experiment(config, agent=agent, save_dir=checkpoint_dir / "evaluation")
    return compare_runs(result)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = config_from_args(args)
    except (ValueError, KeyError) as problem:
        print(f"bad configuration: {problem}", file=sys.stderr)
        return 1

    try:
        result = train(
            config,
            output_dir=args.checkpoint_dir,
            resume_from=args.resume,
            logger=None if args.quiet else print,
        )
    except (ValueError, RuntimeError) as problem:
        print(f"training failed: {problem}", file=sys.stderr)
        return 1

    if args.evaluate:
        target = result.checkpoints[-1] if result.checkpoints else args.resume
        print()
        if target is None:                       # nothing was trained and nothing loaded
            print("nothing to evaluate: no checkpoint was written")
            return 0
        print(f"evaluating {target} over {args.evaluate} games")
        print(evaluate_checkpoint(target, args.evaluate, args.checkpoint_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
