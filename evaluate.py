"""Evaluate a Tetris agent over many games, headless.

    python evaluate.py                            # 100 games, random agent
    python evaluate.py --agent heuristic          # the planning agent
    python evaluate.py --agent heuristic --lookahead 1 --games 30
    python evaluate.py --agent heuristic --trace 5 --trace-board
    python evaluate.py --save results.json        # keep the numbers
    python evaluate.py --save results.csv --quiet # one row per game, no progress

No pygame, no window: games run straight through `tetris.env.TetrisEnv`, so a
thousand of them take a few seconds (the heuristic agent is slower -- it scores
every legal placement, times the lookahead).

`--agent` picks from the registry below; `--lookahead` and `--beam` configure
the heuristic agent's search. `--trace` prints the moves it chose (and, with
`--trace-board`, the board it chose them on) without leaving the terminal.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional  # noqa: UP035

from tetris.agents import AGENTS, build_agent
from tetris.env import TetrisEnv
from tetris.evaluate import DEFAULT_MAX_PIECES, evaluate_agent
from tetris.heuristic import DEFAULT_BEAM
from tetris.render import Renderer, supports_color


def positive_int(value: str) -> int:
    """An argparse type: any integer above zero (used by --games/--log-every)."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value!r}")
    return number


def non_negative_int(value: str) -> int:
    """An argparse type: zero or more (used by --lookahead, which may be 0)."""
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError(f"must be zero or more, got {value!r}")
    return number


def trace_decisions(agent, seed: Optional[int], count: int, max_pieces: int,
                    board: bool = False) -> None:
    """Play one game and print the agent's first `count` decisions.

    The first game of an evaluation, played step by step: what is printed is
    what game 1 did. With `board=True` the position each move was chosen on is
    drawn too, by the same renderer the terminal game uses.
    """
    env = TetrisEnv(seed=seed)
    state = env.reset()
    renderer = Renderer(color=supports_color()) if board else None

    for move in range(1, count + 1):
        if env.is_game_over() or env.pieces_placed >= max_pieces:
            break
        action = agent.choose_action(state, env.get_valid_actions())
        print(f"--- game 1, move {move} " + "-" * 40)
        if renderer is not None:
            print("\n".join(renderer.board_lines(env.game)))
        print(agent.last_decision)
        state, _reward, _done, _info = env.step(action)
    print("-" * 60)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--agent", choices=sorted(AGENTS), default="random",
                        help="which agent to evaluate (default random)")
    parser.add_argument("--games", type=positive_int, default=100,
                        help="how many complete games to play (default 100)")
    parser.add_argument("--seed", type=int,
                        help="seed the piece sequences and the agent (same seed -> same games)")
    parser.add_argument("--max-pieces", type=positive_int, default=DEFAULT_MAX_PIECES,
                        help=f"stop a game after N pieces (default {DEFAULT_MAX_PIECES})")
    parser.add_argument("--log-every", type=positive_int,
                        help="print progress every N games (default: about every 10%%)")
    parser.add_argument("--lookahead", type=non_negative_int, default=0, metavar="N",
                        help="heuristic agent: pieces to look ahead, 0/1/2 (default 0)")
    parser.add_argument("--beam", type=positive_int, default=DEFAULT_BEAM, metavar="N",
                        help=f"heuristic agent: continuations kept per node "
                             f"(default {DEFAULT_BEAM}, only used from lookahead 2)")
    parser.add_argument("--trace", type=positive_int, default=0, metavar="N",
                        help="print the agent's first N decisions of game 1 (default 0)")
    parser.add_argument("--trace-board", action="store_true",
                        help="with --trace, also draw the board each move was chosen on")
    parser.add_argument("--checkpoint", metavar="PATH",
                        help="dqn agent: the saved .pt file to play with")
    parser.add_argument("--save", metavar="PATH",
                        help="write the results to PATH (.json or .csv)")
    parser.add_argument("--quiet", action="store_true",
                        help="no per-game progress, just the summary")
    args = parser.parse_args(argv)

    agent = build_agent(
        args.agent, seed=args.seed, lookahead=args.lookahead, beam=args.beam,
        checkpoint=args.checkpoint,
    )
    seed = "none" if args.seed is None else args.seed
    print(f"evaluating '{agent.name}' over {args.games} games (seed {seed})")
    if agent.name == "heuristic":
        print(f"lookahead depth {args.lookahead}, beam {args.beam}")

    if args.trace:
        trace_decisions(agent, args.seed, args.trace, args.max_pieces, args.trace_board)

    result = evaluate_agent(
        agent,
        num_games=args.games,
        seed=args.seed,
        max_pieces=args.max_pieces,
        log_every=args.log_every,
        logger=None if args.quiet else print,
    )

    print(result.summary())
    if args.save:
        print(f"wrote {result.save(args.save)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
