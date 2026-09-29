"""Watch the trained Tetris AI play, in a window, at whatever speed you like.

    python play_ai.py                          # the best checkpoint on disk
    python play_ai.py --model checkpoints/best.pt
    python play_ai.py --agent heuristic        # the hand-written baseline
    python play_ai.py --speed very-fast --debug

The board is 10x20 cells of 34 px (`--cell`), next to a panel that shows the game
(score, lines, level, pieces, the next three pieces) and the AI itself: what it is
doing right now, which placement it chose for the last piece, what that choice was
worth to it, and what the move cost in reward. `--debug` adds an overlay on the
board with the same numbers in the form they are usually asked for -- chosen
rotation, chosen column, Q-value, reward, lines cleared.

Keyboard (the AI moves the pieces, so the keys are the spectator's):

    SPACE  pause / resume        R  restart
    +  /  -   faster / slower    G  toggle the ghost piece
    D  debug overlay             ESC / Q  quit

Speed is one of `slow`, `normal`, `fast`, `very-fast` (`--speed`), and the panel
names the one in use. `very-fast` also takes several engine moves per painted
frame, so the AI is not limited by the frame rate.

The AI here is *exactly* the agent the training and evaluation code uses -- the
same object, the same `choose_action(state, valid_actions)`, the same
`TetrisEnv.get_state()`. Nothing about the model is changed by watching it: the
checkpoint is opened read-only, and the viewer prints whether the file's hash
still matches when it exits.

A config file can replace the flags (`--config play_ai.json`), which is the easy
way to keep a favourite model and speed around:

    {"model": "experiments/rl/G/seed0/best/best_ep003250_score_mean_822.500.pt",
     "speed": "normal", "debug": true}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, Optional

from tetris.app import DEFAULT_FPS, TetrisApp
from tetris.env import RewardConfig
from tetris.policy_controller import (
    DEFAULT_SPEED,
    SPEED_LEVELS,
    PolicyController,
    best_checkpoint,
    best_records,
    build_agent,
    campaign_of,
    find_checkpoint,
    reward_config_for,
    speed_count,
    speed_detail,
)
from play import build_game, positive_int  # the launcher's own seeding helpers

AGENTS = ("dqn", "heuristic", "random")
SPEED_NAMES = {"slow": 0, "normal": 1, "fast": 2, "very-fast": 3,
               "veryfast": 3, "instant": 3, "max": 3}
KEYS = ("SPACE pause/resume, R restart, +/- speed, G ghost, D debug, ESC quit")


def speed_value(value: str) -> int:
    """`--speed`: a name from the table, or its index."""
    text = str(value).strip().lower().replace("_", "-").replace(" ", "-")
    if text in SPEED_NAMES:
        return SPEED_NAMES[text]
    if text.isdigit() and 0 <= int(text) < speed_count():
        return int(text)
    raise argparse.ArgumentTypeError(
        f"speed must be one of {', '.join(name for name, _, _ in SPEED_LEVELS)} "
        f"or 0-{speed_count() - 1}, not {value!r}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="play_ai.py",
        description="Watch the trained Tetris AI play.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Keys: " + KEYS,
    )
    parser.add_argument("--model", "--checkpoint", dest="model", default=None,
                        help="a saved model to load (default: the best one on disk)")
    parser.add_argument("--agent", choices=AGENTS, default="dqn",
                        help="who plays (default dqn; --model implies dqn)")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"),
                        default="auto",
                        help="DQN device (default: auto, prefers mps/cuda)")
    parser.add_argument("--lookahead", type=int, default=0,
                        help="search depth for --agent heuristic (0 = one ply)")
    parser.add_argument("--speed", type=speed_value, default=DEFAULT_SPEED,
                        help="slow | normal | fast | very-fast (default normal)")
    parser.add_argument("--seed", type=int, default=None,
                        help="piece sequence seed (same seed = same game)")
    parser.add_argument("--cell", type=positive_int, default=None,
                        help="pixels per board cell (default 34: a big, "
                             "watchable board)")
    parser.add_argument("--fps", type=positive_int, default=DEFAULT_FPS,
                        help=f"frames per second (default {DEFAULT_FPS})")
    parser.add_argument("--gravity", action="store_true",
                        help="let pieces fall between moves (off by default: the AI "
                             "should place a piece, not race gravity for it)")
    parser.add_argument("--no-ghost", dest="ghost", action="store_false",
                        help="hide the landing preview")
    parser.add_argument("--grid", choices=("none", "dots", "lines"), default="lines",
                        help="how the empty well is marked (default lines)")
    parser.add_argument("--debug", action="store_true",
                        help="overlay the chosen rotation/column, value, reward, lines")
    parser.add_argument("--reward-per-piece", type=float, default=None,
                        help="override the per-piece reward weight the panel reports "
                             "(default: read from the campaign that trained the model)")
    parser.add_argument("--headless", action="store_true",
                        help="SDL's dummy video driver, so no window opens")
    parser.add_argument("--frames", type=positive_int, default=None,
                        help="stop after N frames")
    parser.add_argument("--until-game-over", action="store_true",
                        help="stop one second after the game ends")
    parser.add_argument("--screenshot", default=None,
                        help="save the last frame to a PNG")
    parser.add_argument("--record", default=None,
                        help="write one PNG per frame into this directory")
    parser.add_argument("--record-every", type=positive_int, default=1,
                        help="record one frame in N (default 1: all of them)")
    parser.add_argument("--verbose", action="store_true",
                        help="log every decision and every locked piece")
    parser.add_argument("--config", default=None,
                        help="a JSON file whose keys are these option names")
    return parser


def load_config(path: str) -> Dict[str, object]:
    """Read a `--config` file: a flat JSON object of option names to values."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            values = json.load(handle)
    except OSError as error:
        raise SystemExit(f"cannot read config {path!r}: {error}")
    except ValueError as error:
        raise SystemExit(f"config {path!r} is not valid JSON: {error}")
    if not isinstance(values, dict):
        raise SystemExit(f"config {path!r} must be a JSON object of option names")
    return {str(key).replace("-", "_"): value for key, value in values.items()}


def parse_args(argv=None) -> argparse.Namespace:
    """Two passes, so `--config` supplies defaults and the flags still win."""
    parser = build_parser()
    args, _unknown = parser.parse_known_args(argv)
    if args.config is None:
        return parser.parse_args(argv)
    parser.set_defaults(**load_config(args.config))
    return parser.parse_args(argv)


def fingerprint(path: str) -> Dict[str, object]:
    """Size, mtime and digest of a file -- proof the viewer did not write to it."""
    stat = os.stat(path)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return {"size": stat.st_size, "mtime": stat.st_mtime, "sha256": digest.hexdigest()}


def missing_model_message(path: str, root: str = "experiments/rl") -> str:
    """What to say when `--model` names a file that is not there.

    The campaigns keep their checkpoints in `experiments/rl/<campaign>/seed<N>/best/`,
    so a guessed path like `checkpoints/best.pt` is the most likely thing to be
    wrong -- listing what does exist (and pointing at the no-argument default) is
    more use than a bare "not found".
    """
    lines = [f"no model at {path!r}."]
    try:
        available = best_records(root)
    except OSError:
        available = []
    if available:
        lines.append("")
        lines.append("The campaigns left these checkpoints (best first):")
        for record in available[:5]:
            lines.append(f"    {record.path}")
            lines.append(f"        {record.source}, score {record.score_mean:.1f}, "
                         f"episode {record.episode}")
        if len(available) > 5:
            lines.append(f"    ... and {len(available) - 5} more")
        lines.append("")
        lines.append("Run with no --model at all to use the best one automatically, "
                     "or pass --agent heuristic/random.")
    else:
        lines.append(
            "No checkpoints under "
            f"{root!r} either: train one with "
            "`python experiments/rl_experiments.py run --experiment G --parallel 5`, "
            "or pass --agent heuristic/random for a non-learning opponent."
        )
    return "\n".join(lines)


def resolve_agent(args) -> Dict[str, object]:
    """Build the agent, and work out what to label it and how to score its reward."""
    if args.agent != "dqn" and args.model is not None:
        print(f"note: --model only applies to the dqn agent; "
              f"--agent {args.agent} ignores it", file=sys.stderr)

    info: Dict[str, object] = {"label": args.agent, "short": "built-in",
                               "checkpoint": None, "before": None}
    if args.agent != "dqn":
        agent = build_agent(args.agent, lookahead=args.lookahead)
        info["agent"] = agent
        reward = RewardConfig()
        if args.reward_per_piece is not None:
            reward = RewardConfig(per_piece=float(args.reward_per_piece))
        info["reward"] = reward
        info["weights"] = f"defaults (per_piece={reward.per_piece:g})"
        return info

    checkpoint = args.model
    found = None
    if checkpoint is None:
        found = best_checkpoint()
        checkpoint = found.path
        info["label"] = f"{found.source} ep{found.episode} score {found.score_mean:.1f}"
        info["short"] = found.source
    else:
        info["short"] = os.path.basename(str(checkpoint))
        # A named model may still be one of the campaign's, and the campaign knows
        # which reward it was trained under -- worth recovering rather than showing
        # the defaults and calling it the same thing.
        found = find_checkpoint(str(checkpoint))
        if found is not None:
            info["label"] = (f"{found.source} ep{found.episode} "
                             f"score {found.score_mean:.1f}")
    if not os.path.isfile(checkpoint):
        raise SystemExit(missing_model_message(checkpoint))
    info["checkpoint"] = checkpoint
    info["before"] = fingerprint(checkpoint)
    agent = build_agent("dqn", checkpoint=checkpoint, device=args.device)
    info["agent"] = agent

    campaign = campaign_of(checkpoint)
    reward, source = (reward_config_for(f"{campaign}/seed0") if campaign
                      else (RewardConfig(), None))
    if args.reward_per_piece is not None:
        reward = RewardConfig(per_piece=float(args.reward_per_piece))
        source = "the --reward-per-piece flag"
    info["reward"] = reward
    info["weights"] = (f"{source} (per_line={reward.per_line:g}, hole={reward.hole:g}, "
                       f"height={reward.height:g}, per_piece={reward.per_piece:g})"
                       if source else f"defaults (per_piece={reward.per_piece:g})")
    return info


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.headless:
        # Must happen before pygame initialises its video backend.
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    info = resolve_agent(args)
    checkpoint = info["checkpoint"]
    if checkpoint is not None:
        print(f"model: {checkpoint}")
    print(f"agent: {info['label']} ({args.agent})")
    device = getattr(info["agent"], "device", None)
    if device is not None:
        print(f"device: {device}")
    print(f"reward: {info['weights']}")
    print(f"speed: {SPEED_LEVELS[args.speed][0]} ({speed_detail(args.speed)})")
    print(f"keys: {KEYS}")

    controller = PolicyController(
        info["agent"],
        think_ms=SPEED_LEVELS[args.speed][1],
        reward_config=info["reward"],
        logger=print if args.verbose else None,
    )
    game = build_game(args.seed)
    app = TetrisApp(
        game,
        controller,
        fps=args.fps,
        gravity=args.gravity,
        game_factory=(lambda: build_game(args.seed)),
    )
    # Imported here, not at module level: this is the only pygame in the program,
    # so --help, the config path and the tests run without a display.
    import pygame
    from tetris.ai_view import AI_CELL, AIPygameView, run_ai

    pygame.init()
    view = AIPygameView(cell=args.cell or AI_CELL, ghost=args.ghost, grid=args.grid,
                        debug=args.debug)

    frames = run_ai(
        app,
        view,
        controller,
        speed=args.speed,
        model=str(info["short"]),
        max_frames=args.frames,
        screenshot=args.screenshot,
        record=args.record,
        record_every=args.record_every,
        stop_after_game_over=(args.fps if args.until_game_over else None),
        logger=print,
    )

    live = app.game
    print(f"frames: {frames}   score: {live.score}   lines: {live.lines_cleared}   "
          f"pieces: {controller.pieces}")
    print(f"agent: {controller.describe()}")

    if checkpoint is not None:
        after = fingerprint(checkpoint)
        if after == info["before"]:
            print(f"model untouched: {after['sha256'][:16]}... "
                  f"({after['size']} bytes)")
        else:
            print("WARNING: the model file changed while the viewer ran!",
                  file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
