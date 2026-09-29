"""A heuristic board evaluation: features in, one number out.

This is the non-ML decision layer. Given a board it can say how good that board
looks, and given a placement it can work out the board that placement would
leave behind -- without touching the real game.

The pieces
    * `extract_features(board)`   -- the six measurements (in `tetris.metrics`)
    * `HeuristicWeights`          -- one weight per feature: the whole policy
    * `evaluate_board(board, ...)`-- features x weights, as a single float
    * `simulate_placement(...)`   -- the board a placement would produce
    * `HeuristicAgent`            -- in `tetris.agents`, uses the three above

Simulation, and why it is safe
------------------------------
`simulate_placement` works on the occupancy array the environment already hands
out in `state.board`:

1. copy it (a fresh array -- the caller's board is never written to),
2. drop the piece into its column (the landing row comes straight from the
   column tops, so no walking the piece down the board),
3. clear any full rows exactly the way `Board.clear_lines` does,
4. hand back the occupancy plus how many lines went.

That is the same "drop from above, stop at first contact" rule the environment
uses, and `tests/test_heuristic.py` checks the simulator against the real engine
placement by placement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union  # noqa: UP035

import numpy as np

from .board import HEIGHT, WIDTH
from .env import normalized_cells, placements_for
from .game import spawn_position
from .metrics import (
    Board,
    BoardFeatures,
    extract_features,
    occupancy,
    top_rows,
)
from .pieces import Piece, PieceType

#: The weight names, in the order they are listed. `completed_lines` is
#: deliberately absent: it is the *feature* a board shows, while the weight is
#: named for the lines a placement clears -- same quantity, seen from the side
#: the decision is made on (see `HeuristicWeights.score`).
WEIGHT_NAMES: Tuple[str, ...] = (
    "lines_cleared",
    "aggregate_height",
    "max_height",
    "holes",
    "bumpiness",
    "wells",
)

WeightMap = Union["HeuristicWeights", Dict[str, float], None]


# ------------------------------------------------------------------ simulation


@dataclass(frozen=True)
class Simulation:
    """The position a placement would leave behind. Nothing real is touched."""

    board: Board      # occupancy after the piece locked and lines cleared
    cleared: int      # rows that would be cleared by this placement
    topped_out: bool = False   # the piece settled above the board: the game ends

    def features(self) -> BoardFeatures:
        """The features of the resulting board (its `completed_lines` are gone)."""
        return extract_features(self.board)


def clear_lines(grid: Board) -> Tuple[Board, int]:
    """Remove full rows and pad the top with empty ones: `(grid, cleared)`.

    Mirrors `Board.clear_lines` so a simulated board looks exactly like the one
    the engine would produce.
    """
    full = grid.all(axis=1) if grid.shape[1] else np.zeros(grid.shape[0], dtype=bool)
    cleared = int(full.sum())
    if not cleared:
        return grid, 0
    kept = grid[~full]
    padding = np.zeros((cleared, grid.shape[1]), dtype=grid.dtype)
    return np.concatenate([padding, kept], axis=0), cleared


def _blocked(board, row: int, col: int) -> bool:
    """True if (row, col) is off the board or filled -- for one cell, no copying."""
    if row < 0:
        return False                      # above the board is always free
    if col < 0 or col >= WIDTH or row >= HEIGHT:
        return True
    return bool(board[row][col])


def spawn_blocked(board, piece_type: PieceType) -> bool:
    """True if `piece_type` could not appear on `board`: the engine ends the game.

    A new piece arrives centred at the top (`game.spawn_position`) and the engine
    is over if it collides there, so a tower in the middle kills a board whose
    side columns are still open. Planning ahead has to know that, or it will
    happily plan a move whose next piece never gets to move.
    """
    piece = Piece(piece_type)
    piece.row, piece.col = spawn_position(piece)
    return any(_blocked(board, row, col) for row, col in piece.absolute_cells())


def simulate_placement(
    board, piece_type: PieceType, rotation: int, column: int
) -> Simulation:
    """Drop `piece_type` into `column` on a copy of `board`.

    `column` is the board column of the piece's leftmost cell, exactly as in a
    `Placement`, so a simulated move and a real one mean the same thing. The
    board passed in is never modified.
    """
    grid = occupancy(board)                     # always a fresh array
    height, width = grid.shape
    cells = normalized_cells(piece_type, rotation)

    tops = top_rows(grid)                       # first filled row per column
    rows = np.array([row for row, _ in cells])
    columns = np.array([col for _, col in cells]) + column
    if columns.size and (columns.min() < 0 or columns.max() >= width):
        raise ValueError(
            f"column {column} puts the piece outside the 0..{width - 1} board"
        )

    # The piece falls until one of its cells would enter the top of its column.
    base = int((tops[columns] - 1 - rows).min())
    resting_rows = base + rows
    on_board = resting_rows >= 0               # cells above the well are not stored
    grid[resting_rows[on_board], columns[on_board]] = 1.0

    # A piece that settles entirely above the board locks out: the engine ends
    # the game there, having placed nothing.
    topped_out = not on_board.any()

    grid, cleared = clear_lines(grid)
    return Simulation(board=grid, cleared=cleared, topped_out=bool(topped_out))


# ------------------------------------------------------------------- weights


@dataclass(frozen=True)
class HeuristicWeights:
    """One weight per feature. The entire hand-written policy lives here.

    The defaults are a *reasonable starting point, not an optimum*. They were
    picked by hand and then compared against a handful of alternatives (the
    published Fahey and El-Tetris weights among them) over 1000-piece games on
    seeds 1/600, and the winner checked on 900/901 -- see "Choosing the weights"
    in the README for what that search did and did not show. Change any weight
    -- or pass a plain dict -- and the agent's behaviour changes with it::

        HeuristicWeights(holes=-8.0, bumpiness=-0.4)
        HeuristicWeights.from_dict({"holes": -8.0, "lines_cleared": 4.0})

    Signs say what the agent wants: a positive weight means "more of this is
    good" (lines cleared), a negative one means "more of this is bad" (holes,
    height, bumpiness, wells).
    """

    lines_cleared: float = 4.0          # per line completed by the placement
    holes: float = -8.0                 # per hole on the resulting board
    aggregate_height: float = -1.0      # per cell of total stack height
    max_height: float = -1.0            # per cell of the tallest column
    bumpiness: float = -1.0             # per cell of surface unevenness
    wells: float = -1.0                 # per cell of well depth

    def as_dict(self) -> Dict[str, float]:
        """The weights as a plain dict, keyed by weight name (see `WEIGHT_NAMES`)."""
        return {name: float(getattr(self, name)) for name in WEIGHT_NAMES}

    @classmethod
    def from_dict(cls, values: Dict[str, float]) -> "HeuristicWeights":
        """Weights from a dict; unknown keys are an error, missing ones default."""
        unknown = set(values) - set(WEIGHT_NAMES)
        if unknown:
            raise ValueError(
                f"unknown weight(s) {sorted(unknown)}; known: {sorted(WEIGHT_NAMES)}"
            )
        return cls(**{name: float(value) for name, value in values.items()})

    @classmethod
    def from_any(cls, value: WeightMap) -> "HeuristicWeights":
        """Accept an `HeuristicWeights`, a dict of overrides, or nothing at all."""
        if value is None:
            return cls()
        if isinstance(value, HeuristicWeights):
            return value
        if isinstance(value, dict):
            return cls.from_dict(value)
        raise TypeError(f"weights must be HeuristicWeights, a dict or None, got {value!r}")

    def score(self, features: BoardFeatures, lines_cleared: Optional[int] = None) -> float:
        """The weighted sum: one number saying how good this board looks.

        The line term uses `lines_cleared` if the caller says how many lines the
        move cleared; otherwise it falls back to the full rows the board itself
        still shows (`features.completed_lines`). The agent simulates boards that
        have *already* had their rows removed, so it passes the count in.
        """
        lines = features.completed_lines if lines_cleared is None else lines_cleared
        return float(
            self.lines_cleared * lines
            + self.holes * features.holes
            + self.aggregate_height * features.aggregate_height
            + self.max_height * features.max_height
            + self.bumpiness * features.bumpiness
            + self.wells * features.wells
        )


def evaluate_board(board, weights: WeightMap = None, lines_cleared: Optional[int] = None) -> float:
    """How good `board` looks to a heuristic, as one float (higher is better).

    ::

        evaluate_board(board)                       # default weights
        evaluate_board(board, {"holes": -10.0})     # tweak one weight
        evaluate_board(board, lines_cleared=2)      # board *after* those lines went

    `board` may be anything `occupancy` accepts. Full rows the board still shows
    count through the `completed_lines` feature; pass `lines_cleared` when they
    are already gone (which is what the agent simulates), so the same quantity
    is not counted twice.
    """
    return HeuristicWeights.from_any(weights).score(
        extract_features(board), lines_cleared
    )


# ---------------------------------------------------------------- the planning


def best_placement(board, piece_type: PieceType, actions, weights: WeightMap = None):
    """Score every action and return `(best_action, [(action, score), ...])`.

    Ties keep the first action in the order given, which is deterministic
    (`get_valid_actions` is ordered by rotation then column), so the choice is
    reproducible without any randomness. The full list is returned so a caller
    can inspect *why* a move was picked. This is `search(..., depth=0)` -- the
    no-lookahead case, kept as its own name because it is the common one.
    """
    result = search(board, (piece_type,), weights, depth=0, actions=actions)
    return result.action, list(result.scores)


# ------------------------------------------------------------------- lookahead


#: How many continuations a search keeps per node below the current move. The
#: current move is always chosen from *every* legal placement; this only limits
#: how much of the following pieces' move lists is explored, because the full
#: product grows by a factor of ~30 per extra piece (see `search`).
DEFAULT_BEAM = 4


@dataclass(frozen=True)
class SearchResult:
    """What a lookahead search found, including what it looked at.

    `scores` holds one `(action, score)` pair per action given to the search, in
    that order, so it lines up with `valid_actions` and with `agent.last_scores`.
    `nodes` is how many placements were simulated -- the honest price of the
    decision, and the number to watch when raising the depth.
    """

    action: Optional[Placement]                 # the best root action (None if none)
    scores: Tuple[Tuple[Placement, float], ...]  # root actions and their totals
    depth: int
    beam: int
    nodes: int
    pieces: Tuple[PieceType, ...]                # the pieces searched, current first

    @property
    def best_score(self) -> float:
        return max(score for _action, score in self.scores)

    def ranked(self) -> Tuple[Tuple[Placement, float], ...]:
        """The same pairs, best first -- what a tie-broken debug view wants."""
        return tuple(sorted(self.scores, key=lambda pair: -pair[1]))

    def __str__(self) -> str:
        """The debugging view: which move the search picked, and why it stood out."""
        if self.action is None:
            return "AI selected:\n  nothing -- no legal placement"
        next_pieces = ", ".join(piece.name for piece in self.pieces[1:]) or "none"
        lines = [
            "AI selected:",
            f"Rotation: {self.action.rotation}",
            f"Column: {self.action.column}",
            f"Evaluation: {self.best_score:.2f}",
            f"  (depth {self.depth} lookahead, beam {self.beam}, "
            f"{self.nodes} placements scored, best of {len(self.scores)} legal moves, "
            f"next: {next_pieces})",
        ]
        runners_up = [
            f"({action.rotation}, {action.column}) {score:.2f}"
            for action, score in self.ranked()[1:4]
        ]
        if runners_up:
            lines.append(f"  runners-up: {', '.join(runners_up)}")
        dead = sum(1 for _action, score in self.scores if score == DEAD)
        if dead:
            lines.append(f"  {dead} of {len(self.scores)} moves end the game")
        return "\n".join(lines)


#: Score of a line of play that cannot continue (a lock-out, or a board the next
#: piece cannot even appear on). Any playable alternative beats it, and if every
#: move scores this, the game is over whatever we do.
DEAD = float("-inf")


def _search_value(
    board,
    pieces: Tuple[PieceType, ...],
    depth: int,
    weights: "HeuristicWeights",
    beam: int,
    nodes: List[int],
) -> float:
    """The best total reachable from `board`, with `pieces[0]` about to be placed.

    A *total* is the sum of the immediate evaluations along a line of play: the
    score of the board each placement leaves behind, added up. Leaves score every
    legal placement and keep the best; a node higher up keeps only its `beam`
    best-scoring children, because the exhaustive product would be
    `len(actions) ** depth` placements -- about 33,000 of them at depth 2.
    """
    piece_type = pieces[0]
    children = []
    for action in placements_for(piece_type):
        simulation = simulate_placement(board, piece_type, action.rotation, action.column)
        score = (
            DEAD if simulation.topped_out
            else weights.score(simulation.features(), simulation.cleared)
        )
        children.append((action, score, simulation))
    nodes[0] += len(children)

    if depth <= 0 or len(pieces) == 1:
        return max(score for _action, score, _simulation in children)

    children.sort(key=lambda child: -child[1])
    best = DEAD
    for _action, score, simulation in children[:beam]:
        if score == DEAD:
            continue                        # unplayable: the line is worthless anyway
        if spawn_blocked(simulation.board, pieces[1]):
            continue                        # the game ends after this placement
        continuation = _search_value(
            simulation.board, pieces[1:], depth - 1, weights, beam, nodes
        )
        best = max(best, score + continuation)
    return best


def search(
    board,
    pieces,
    weights: WeightMap = None,
    depth: int = 0,
    beam: int = DEFAULT_BEAM,
    actions=None,
) -> SearchResult:
    """Score every current placement, looking `depth` pieces ahead.

    `pieces` is the piece sequence starting with the one being placed now, so
    `depth=0` scores only the current piece, `depth=1` also considers the best
    placement of the next piece, `depth=2` the one after that. `actions` is what
    to try for the current piece (the environment's `valid_actions`); the pieces
    after it generate their own moves with `placements_for`.

    Each line of play is scored as the *sum* of the immediate board evaluations
    along it: the score of the board a move leaves, plus the best score reachable
    from there. The highest total wins.

    * The current move is always chosen from all of `actions` -- depth never
      narrows down the decision being made.
    * Depth 1 is exact: every current move is followed by the best of *all* next
      moves. From depth 2 the intermediate plies keep only their `beam` best
      moves, because the exhaustive tree has one leaf per combination of moves:
      34x34 = 1,156 for a T then a J, 34x34x34 = 39,304 for a third piece. A
      beam of 4 scores 5,814 placements for that same three-piece line.
    * A line that cannot continue -- a lock-out, or a board where the next piece
      cannot even appear -- scores `DEAD` (`-inf`) instead of counting as good
      just because the board looks tidy.
    """
    weights = HeuristicWeights.from_any(weights)
    if depth < 0:
        raise ValueError(f"depth must be 0 or more, got {depth}")
    if beam < 1:
        raise ValueError(f"beam must be at least 1, got {beam}")

    pieces = tuple(pieces)
    if not pieces:
        raise ValueError("search needs at least the current piece")
    if actions is None:
        actions = placements_for(pieces[0])

    nodes = [0]
    children = []
    for action in actions:
        rotation, column = action          # a Placement, or any (rotation, column) pair
        simulation = simulate_placement(board, pieces[0], rotation, column)
        score = (
            DEAD if simulation.topped_out
            else weights.score(simulation.features(), simulation.cleared)
        )
        children.append((action, score, simulation))
    nodes[0] += len(children)

    if depth == 0 or len(pieces) == 1 or not children:
        scores = tuple((action, score) for action, score, _sim in children)
    else:
        totals = []
        for action, score, simulation in children:
            if score == DEAD or spawn_blocked(simulation.board, pieces[1]):
                totals.append((action, DEAD))
                continue
            continuation = _search_value(
                simulation.board, pieces[1:], depth - 1, weights, beam, nodes
            )
            totals.append((action, score + continuation))
        scores = tuple(totals)

    best_action = None
    if scores:
        best_action = max(scores, key=lambda pair: pair[1])[0]
    return SearchResult(
        action=best_action,
        scores=scores,
        depth=depth,
        beam=beam,
        nodes=nodes[0],
        pieces=pieces,
    )


def feature_rows(board, piece_type: PieceType, actions, weights: WeightMap = None):
    """A table of per-action features and scores: handy when tuning weights.

    Returns `[(action, lines_cleared, BoardFeatures, score), ...]`, one row per
    action, in the order given.
    """
    weights = HeuristicWeights.from_any(weights)
    rows = []
    for action in actions:
        rotation, column = action          # a Placement, or any (rotation, column) pair
        simulation = simulate_placement(board, piece_type, rotation, column)
        features = simulation.features()
        rows.append((action, simulation.cleared, features, weights.score(features, simulation.cleared)))
    return rows


__all__ = [
    "DEAD",
    "DEFAULT_BEAM",
    "WEIGHT_NAMES",
    "HeuristicWeights",
    "SearchResult",
    "Simulation",
    "best_placement",
    "clear_lines",
    "evaluate_board",
    "feature_rows",
    "search",
    "simulate_placement",
    "spawn_blocked",
]
