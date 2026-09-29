"""Compare the heuristic agent at lookahead depths 0, 1 and 2 on the same games.

    python experiments/lookahead.py                                  # the default run
    python experiments/lookahead.py --games 10 --pieces 200          # a quick one
    python experiments/lookahead.py --depths 2 --beam 8 --games 5 --pieces 150

Every depth is evaluated with the same `seed` and the same piece cap, so each
depth plays *the same piece sequences* game by game (see `tetris.evaluate`): the
comparison is run-by-run, not just on average.

Per depth this records the numbers the harness reports (average / median / worst
/ best score, average and best lines, average game length, capped games) plus
three the search needs: how high the stack had to be played (peak column height),
how many boards per decision the search scored, and the wall-clock cost per
decision.

Results are written next to this script:

    lookahead_depth0.json, lookahead_depth1.json, lookahead_depth2.json
        the harness's own per-game rows, for every game of that depth
    lookahead_summary.json
        one entry per configuration -- what the printed table shows
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time
from typing import Dict, List, Optional  # noqa: UP035

# Running this file directly puts `experiments/` on sys.path, not the project
# root, so the package has to be reachable by hand.
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tetris.agents import HeuristicAgent  # noqa: E402
from tetris.evaluate import DEFAULT_MAX_PIECES, evaluate_agent  # noqa: E402
from tetris.heuristic import DEFAULT_BEAM  # noqa: E402
from tetris.metrics import max_column_height  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent


class Watchful:
    """An agent wrapper that records what the search cost and what it faced.

    `evaluate_agent` resets the agent before every game, which is where the
    per-game peak is banked. Everything the harness itself needs (`name`,
    `choose_action`) is forwarded, so this stays a drop-in.
    """

    def __init__(self, agent, label: str) -> None:
        self.agent = agent
        self.name = label
        self.boards_scored = 0
        self.decisions = 0
        self.peaks: List[float] = []
        self._tallest = 0.0

    def reset(self, seed=None) -> None:
        if self._tallest:
            self.peaks.append(self._tallest)
        self._tallest = 0.0
        reset = getattr(self.agent, "reset", None)
        if reset is not None:
            reset(seed)

    def choose_action(self, state, valid_actions):
        self._tallest = max(self._tallest, float(max_column_height(state.board)))
        action = self.agent.choose_action(state, valid_actions)
        self.decisions += 1
        decision = getattr(self.agent, "last_decision", None)
        if decision is not None:
            self.boards_scored += decision.nodes
        return action

    def flush(self) -> None:
        """Bank the peak of a game the harness never reset away from."""
        if self._tallest:
            self.peaks.append(self._tallest)
            self._tallest = 0.0


def run(
    depth: int,
    beam: int,
    games: int,
    pieces: int,
    seed: Optional[int],
    quiet: bool = True,
):
    """Evaluate one configuration and return `(summary, watched)`."""
    label = f"heuristic/lookahead={depth}" + (f"+beam={beam}" if depth >= 2 else "")
    watched = Watchful(HeuristicAgent(lookahead_depth=depth, beam=beam), label)

    result = evaluate_agent(
        watched, num_games=games, seed=seed, max_pieces=pieces,
        logger=None if quiet else print,
    )
    watched.flush()

    stats = result.stats
    summary = dict(stats)
    peaks = watched.peaks
    if len(peaks) != games:                       # the pairing is the point
        raise RuntimeError(f"recorded {len(peaks)} peaks for {games} games")
    summary.update(
        depth=depth,
        beam=beam,
        agent=label,
        seed=seed,
        max_pieces=pieces,
        elapsed_seconds=round(result.elapsed_seconds, 3),
        seconds_per_game=round(result.elapsed_seconds / games, 3),
        milliseconds_per_decision=round(
            1000 * result.elapsed_seconds / max(1, watched.decisions), 3
        ),
        boards_scored=watched.boards_scored,
        boards_per_decision=round(watched.boards_scored / max(1, watched.decisions), 1),
        peak_column_per_game=[float(peak) for peak in peaks],
        peak_column_mean=round(statistics.fmean(peaks), 2),
        peak_column_median=float(statistics.median(peaks)),
        peak_column_min=float(min(peaks)),
        peak_column_max=float(max(peaks)),
        deaths=games - int(stats["games_capped"]),
        lines_per_100_pieces=round(
            100 * stats["lines_total"] / max(1.0, sum(g.pieces for g in result.games)), 3
        ),
    )
    return summary, result


def table(summaries: List[Dict[str, object]]) -> str:
    """The comparison, one line per configuration."""
    header = (
        f"{'depth':>5} {'beam':>4} {'games':>5} {'score mean':>12} {'score max':>11} "
        f"{'lines mean':>10} {'lines max':>9} {'lines/100':>9} {'len mean':>8} "
        f"{'peak':>5} {'dead':>5} {'ms/dec':>8} {'total s':>8}"
    )
    rows = [header, "-" * len(header)]
    for summary in summaries:
        rows.append(
            f"{summary['depth']:>5} {summary['beam']:>4} {summary['games']:>5.0f} "
            f"{summary['score_mean']:>12,.0f} {summary['score_max']:>11,.0f} "
            f"{summary['lines_mean']:>10.2f} {summary['lines_max']:>9.0f} "
            f"{summary['lines_per_100_pieces']:>9.2f} "
            f"{summary['game_length_mean']:>8.1f} "
            f"{summary['peak_column_mean']:>5.1f} {summary['deaths']:>5.0f} "
            f"{summary['milliseconds_per_decision']:>8.1f} "
            f"{summary['elapsed_seconds']:>8.1f}"
        )
    return "\n".join(rows)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--depths", default="0,1,2",
                        help="comma separated lookahead depths (default 0,1,2)")
    parser.add_argument("--beam", type=int, default=2, metavar="N",
                        help=f"continuations per node (default 2; DEFAULT_BEAM is "
                             f"{DEFAULT_BEAM}, and only depth >= 2 uses it)")
    parser.add_argument("--games", type=int, default=30,
                        help="games per depth (default 30)")
    parser.add_argument("--pieces", type=int, default=400, metavar="N",
                        help=f"pieces per game before it is stopped (default 400)")
    parser.add_argument("--seed", type=int, default=123,
                        help="base seed: every depth plays these same games (default 123)")
    parser.add_argument("--quiet", action="store_true",
                        help="only the table, no per-depth progress")
    args = parser.parse_args(argv)

    depths = [int(part) for part in args.depths.split(",") if part.strip()]
    summaries: List[Dict[str, object]] = []

    for depth in depths:
        label = f"lookahead {depth}" + (f" (beam {args.beam})" if depth >= 2 else "")
        if not args.quiet:
            print(f"\n=== {label}: {args.games} games x {args.pieces} pieces, "
                  f"seed {args.seed} ===", flush=True)
        started = time.perf_counter()
        summary, result = run(depth, args.beam, args.games, args.pieces, args.seed)
        summaries.append(summary)

        path = result.save(HERE / f"lookahead_depth{depth}.json")
        print(f"{label}: {summary['score_mean']:,.0f} mean score, "
              f"{summary['lines_mean']:.1f} mean lines, "
              f"{summary['lines_per_100_pieces']:.2f} lines/100 pieces, "
              f"peak {summary['peak_column_mean']:.1f}, "
              f"{summary['milliseconds_per_decision']:.1f} ms/decision, "
              f"{time.perf_counter() - started:.0f}s"
              + (f"  (per-game rows -> {path})" if not args.quiet else ""),
              flush=True)

    summary_path = HERE / "lookahead_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")

    print()
    print(table(summaries))
    print(f"\nsaved {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
