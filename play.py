"""Launch Tetris in one of its three modes.

    python play.py                       # human mode: play it in a window
    python play.py --seed 42             # ...with a reproducible piece sequence
    python play.py --mode viz            # the trained AI plays, in the window
    python play.py --mode viz --agent heuristic --lookahead 1
    python play.py --mode viz --checkpoint experiments/rl/G/seed0/best/best_ep003250_score_mean_822.500.pt
    python play.py --mode train          # see the note below

All modes drive the same engine through `tetris.app.TetrisApp`; they differ only
in which controller decides the moves and whether a view is attached:

    human    HumanController + PygameView
    train    AI policy       + no view       (the campaign CLI owns this)
    viz      AI policy       + PygameView

Controls

    left / a        move left          right / d   move right
    down / s        soft drop         space       hard drop
    up / w / x      rotate clockwise  z           rotate counter-clockwise
    p               pause             r           restart
    g               toggle the ghost piece        esc / q   quit

In `viz` mode the keys that still make sense are p / r / g / esc: the AI drives
the piece, and `--think-ms` sets how long it takes between two moves, so the
rotation and the slide are visible instead of instant.

`--mode train` intentionally has no home here. Training a DQN in this window
would be one unmeasured run; the entry point that produces comparable numbers is
the campaign CLI (`experiments/rl_experiments.py run --experiment G`), which
runs several seeds, evaluates each one as it trains, and keeps the best
checkpoint. `--mode viz` then plays whatever that left behind.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from typing import List, Optional  # noqa: UP035

from tetris import render
from tetris.app import DEFAULT_FPS, TetrisApp
from tetris.game import Game

MODES = ("human", "viz", "train")
AGENTS = ("dqn", "heuristic", "random")
CONTROLS = (
    "left/right move, down soft drop, up rotate, space hard drop, "
    "z rotate ccw, p pause, r restart, g ghost, esc quit"
)
DEFAULT_THINK_MS = 80.0


def build_game(seed: Optional[int] = None) -> Game:
    """A `Game`, deterministic when a seed is given.

    The seed goes into the engine's own randomizer (the 7-bag queue), so the
    same seed always deals the same pieces -- and `restart()` replays them.
    """
    rng = random.Random(seed) if seed is not None else None
    return Game(rng)


def positive_int(value: str) -> int:
    """An argparse type: any integer above zero (used by --fps/--frames)."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value!r}")
    return number


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=MODES, default="human",
                        help="human (default), or the AI modes from part 2")
    parser.add_argument("--seed", type=int, help="seed the piece randomizer")
    parser.add_argument("--fps", type=positive_int, default=DEFAULT_FPS,
                        help=f"target frame rate (default {DEFAULT_FPS})")
    parser.add_argument("--cell", type=int, default=0,
                        help="pixels per board cell (0 = the default size)")
    parser.add_argument("--grid", choices=render.GRID_STYLES, default="lines",
                        help="mark the empty cells ('none', 'dots' or 'lines')")
    parser.add_argument("--no-ghost", action="store_true",
                        help="hide the landing preview")
    parser.add_argument("--headless", action="store_true",
                        help="use SDL's dummy video driver, so no window opens")
    parser.add_argument("--frames", type=positive_int,
                        help="quit after N frames (for tests)")
    parser.add_argument("--screenshot", metavar="PATH",
                        help="write the final frame to a PNG")
    parser.add_argument("--agent", choices=AGENTS, default="dqn",
                        help="who plays in --mode viz (default: the trained DQN)")
    parser.add_argument("--checkpoint", metavar="PATH",
                        help="a .pt for --agent dqn (default: the best one on disk)")
    parser.add_argument("--lookahead", type=int, default=0,
                        help="extra pieces the heuristic plans ahead (--agent heuristic)")
    parser.add_argument("--think-ms", type=float, default=DEFAULT_THINK_MS,
                        help=f"milliseconds between two AI moves (default {DEFAULT_THINK_MS:g})")
    parser.add_argument("--gravity", action="store_true",
                        help="let the piece fall on its own in --mode viz (with it on the "
                             "piece can lock before the policy's placement is carried out, "
                             "so the move you see is not always the one it chose)")
    parser.add_argument("--verbose", action="store_true",
                        help="log every AI decision to the terminal")
    parser.add_argument("--record", metavar="DIR",
                        help="write every rendered frame to DIR as frame_00000.png")
    parser.add_argument("--record-every", type=positive_int, default=1,
                        help="record one frame in N (default 1: all of them)")
    parser.add_argument("--until-game-over", action="store_true",
                        help="stop once the game is over (after a second on the final board)")
    args = parser.parse_args(argv)

    if args.headless:
        # Must happen before pygame initialises its video backend.
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    if args.mode == "train":
        raise SystemExit(
            "'train' mode has no window by design: a single training run in this\n"
            "window would be one unmeasured run. The campaign CLI is the entry\n"
            "point that produces comparable numbers -- many seeds, evaluated as\n"
            "they train, best checkpoint kept:\n"
            "\n"
            "    python experiments/rl_experiments.py plan\n"
            "    python experiments/rl_experiments.py run --experiment G --parallel 5\n"
            "\n"
            "then watch the result with:  python play.py --mode viz"
        )

    # Imported here so --help, the tests and the AI controller never need a display.
    import pygame

    from tetris.pygame_view import DEFAULT_CELL, HumanController, PygameView, run_human

    game = build_game(args.seed)
    # The factory makes restart() build a fresh game, so a seeded run always
    # replays the same piece sequence (handy when comparing AI runs later).
    factory = lambda: build_game(args.seed)  # noqa: E731 -- a one-line closure

    if args.mode == "viz":
        from tetris.policy_controller import PolicyController, best_checkpoint, build_agent

        checkpoint = args.checkpoint
        if args.agent == "dqn" and checkpoint is None:
            found = best_checkpoint()
            checkpoint = found.path
            print(f"best trained checkpoint: {found}")

        agent = build_agent(args.agent, checkpoint=checkpoint, lookahead=args.lookahead)
        controller = PolicyController(
            agent,
            think_ms=args.think_ms,
            logger=print if args.verbose else None,
        )
        # Gravity stays off unless asked for: with it on the piece can lock
        # before the policy's placement has been carried out, and the window
        # would then show a *different* move than the one the policy chose.
        app = TetrisApp(
            game,
            controller,
            fps=args.fps,
            gravity=args.gravity,
            game_factory=factory,
        )
        pygame.init()
        try:
            view = PygameView(
                cell=args.cell or DEFAULT_CELL,
                ghost=not args.no_ghost,
                grid=args.grid,
            )
            print(f"AI mode -- agent {agent.name}, thinking every {args.think_ms:g} ms, "
                  f"gravity {'on' if args.gravity else 'off'}")
            print("p pause, r restart, g ghost, esc quit")
            frames = run_human(
                app,
                view,
                controller,
                max_frames=args.frames,
                screenshot=args.screenshot,
                record=args.record,
                record_every=args.record_every,
                stop_after_game_over=args.fps if args.until_game_over else None,
            )
        finally:
            pygame.quit()

        live = app.game
        print(f"frames: {frames}   score: {live.score}   lines: {live.lines_cleared}   "
              f"pieces: {controller.pieces}")
        print(f"agent decisions: {controller.describe()}")
        return 0

    app = TetrisApp(game, HumanController(), fps=args.fps, game_factory=factory)
    pygame.init()
    try:
        view = PygameView(
            cell=args.cell or DEFAULT_CELL,
            ghost=not args.no_ghost,
            grid=args.grid,
        )
        print(f"Human mode -- {CONTROLS}")
        if args.seed is not None:
            print(f"seed {args.seed} (the same seed always deals the same pieces)")
        frames = run_human(
            app,
            view,
            app.controller,
            max_frames=args.frames,
            screenshot=args.screenshot,
            record=args.record,
            record_every=args.record_every,
        )
    finally:
        pygame.quit()

    print(f"frames: {frames}   score: {game.score}   lines: {game.lines_cleared}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
