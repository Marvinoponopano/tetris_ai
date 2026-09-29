"""A reinforcement-learning environment for the Tetris engine.

    env = TetrisEnv(seed=42)
    state = env.reset()
    while not done:
        action = agent.choose_action(state, env.get_valid_actions())
        state, reward, done, info = env.step(action)

This module is the *only* thing an agent needs: it imports numpy and the engine
and nothing else -- no pygame, no terminal, no rendering. Thousands of episodes
can run in a loop with no window open.

Design decisions
----------------

1. **High-level actions, not keystrokes.** An action is a *placement*::

       Placement(rotation=1, column=5)      # or the tuple (1, 5)
       {"rotation": 1, "column": 5}         # a dict works too

   `column` is the board column the piece's *leftmost* cell ends up in, so
   column 0 is flush against the left wall and every column in
   `range(WIDTH - piece_width + 1)` is meaningful. `step()` then rotates the
   piece, puts it at that column just above the well, hard-drops it, lets the
   engine lock it, clear lines and spawn the next piece. One action = one
   piece placed. The agent never has to learn "hold left for four frames"
   nor wall-kick mechanics -- frames stay available through `Game.step()` for
   anyone who wants the harder, frame-level problem later.

2. **A placement is always a straight, unobstructed drop.** Because the piece is
   placed above the board and dropped, it stops on the first thing it touches:
   it can never tunnel into a cave or pass through the stack, and the result is
   independent of any path. This is the standard "plan a placement" abstraction
   used for Tetris RL.

3. **The engine keeps every rule.** The environment never clears lines, locks
   pieces or spawns anything itself; it sets the piece position and calls
   `Game.step(Action.DROP)`. Scoring, line clears, lock-out and the 7-bag
   randomizer all stay in `Game`, so the environment and the human game can
   never disagree about the rules.

4. **`get_valid_actions()` is the action space.** It lists every distinct
   orientation of the current piece at every column where it fits between the
   walls -- exactly the placements that can physically be made right now.
   `step()` rejects anything else (out-of-range rotation or column) with a
   `ValueError` instead of silently playing a different move. Note that a
   *valid* placement can still be a bad one (see `TetrisEnv.get_valid_actions`).

5. **Seeded determinism.** `TetrisEnv(seed=42)` replays the same piece sequence
   on every `reset()`, so two agents can be compared on identical episodes.
   Pass `reset(seed=...)` (or leave the env unseeded) for variety.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, fields
from functools import lru_cache
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

from .actions import Action
from .board import HEIGHT, WIDTH
from .game import Game
from .metrics import board_height, count_holes, occupancy
from .pieces import Piece, PieceType, rotated_cells

# The 7 piece ids in one-hot order: `PieceType.I` is index 0 ... `PieceType.L` index 6.
PIECE_TYPES: Tuple[PieceType, ...] = tuple(PieceType)
N_PIECE_TYPES = len(PIECE_TYPES)

# Length of `TetrisState.feature_vector()`: the board, one-hots for the current
# piece type and rotation, the current column, and the next piece type.
FEATURE_VECTOR_SIZE = HEIGHT * WIDTH + 2 * N_PIECE_TYPES + 4 + 1


class Placement(NamedTuple):
    """A high-level action: put the piece in this orientation at this column.

    A `NamedTuple`, so it compares equal to the plain tuple `(rotation, column)`,
    unpacks as `rotation, column = action`, and still offers `.rotation` /
    `.column` attributes.

    `column` is where the piece's leftmost cell lands (0 = against the left wall).
    """

    rotation: int
    column: int


@dataclass(frozen=True)
class Orientation:
    """One distinct orientation of a piece, normalised to its top-left corner.

    `cells` are (row, col) offsets with `min(row) == min(col) == 0`, so the
    piece's bounding box starts at the origin and `width` tells us which
    columns it can occupy.
    """

    rotation: int          # canonical rotation index (what get_valid_actions reports)
    cells: Tuple[Tuple[int, int], ...]
    width: int
    height: int

    @property
    def columns(self) -> range:
        """Every column this orientation fits in, flush against either wall."""
        return range(WIDTH - self.width + 1)


def _normalized(cells) -> Tuple[Tuple[int, int], ...]:
    """The cells shifted so the bounding box starts at (0, 0)."""
    min_row = min(row for row, _ in cells)
    min_col = min(col for _, col in cells)
    return tuple(sorted((row - min_row, col - min_col) for row, col in cells))


def normalized_cells(piece_type: PieceType, rotation: int) -> Tuple[Tuple[int, int], ...]:
    """A piece's cells in `rotation`, shifted so its bounding box starts at (0, 0).

    `column` in a placement means "the board column of the piece's leftmost
    cell", so both the environment (moving a piece) and the heuristic agent
    (simulating one) need the shape anchored at its top-left corner. This is
    that shape, from the same `rotated_cells` the engine rotates with.
    """
    return _normalized(rotated_cells(piece_type, rotation))


@lru_cache(maxsize=None)
def orientations(piece_type: PieceType) -> Tuple[Orientation, ...]:
    """Every *distinct* orientation of a piece, in rotation order.

    Rotations that produce the same shape (the O piece has one, I/S/Z have two)
    are collapsed to the first rotation that produces them, so the action space
    has no two actions with identical effects.
    """
    seen: Dict[Tuple[Tuple[int, int], ...], Orientation] = {}
    for rotation in range(4):
        cells = _normalized(rotated_cells(piece_type, rotation))
        if cells in seen:
            continue
        seen[cells] = Orientation(
            rotation=rotation,
            cells=cells,
            width=max(col for _, col in cells) + 1,
            height=max(row for row, _ in cells) + 1,
        )
    return tuple(seen[cells] for cells in sorted(seen))


@lru_cache(maxsize=None)
def placements_for(piece_type: PieceType) -> Tuple[Placement, ...]:
    """Every placement `piece_type` can make, as `(rotation, column)` pairs.

    One placement per distinct orientation of the piece, at every column where
    that orientation fits between the walls -- exactly the list
    `TetrisEnv.get_valid_actions()` returns while that piece is the active one,
    but without needing an environment. That is what lets a search generate the
    moves of the *next* pieces while it is still deciding about this one.
    """
    return tuple(
        Placement(orientation.rotation, column)
        for orientation in orientations(piece_type)
        for column in orientation.columns
    )


def _orientation_for(piece_type: PieceType, rotation: int) -> Orientation:
    """The distinct `Orientation` whose shape this rotation produces."""
    cells = _normalized(rotated_cells(piece_type, rotation))
    for orientation in orientations(piece_type):
        if orientation.cells == cells:
            return orientation
    raise AssertionError(f"no orientation matches {piece_type!r} rotation {rotation!r}")


def _canonical_rotation(piece_type: PieceType, rotation: int) -> int:
    """The rotation index `get_valid_actions` uses for this shape."""
    return _orientation_for(piece_type, rotation).rotation


def _one_hot(index: int, size: int) -> np.ndarray:
    vector = np.zeros(size, dtype=np.float32)
    if 0 <= index < size:
        vector[index] = 1.0
    return vector


def _piece_one_hot(piece_type: Optional[int]) -> np.ndarray:
    """One-hot over the seven piece ids (`None` -> all zeros)."""
    if piece_type is None:
        return np.zeros(N_PIECE_TYPES, dtype=np.float32)
    return _one_hot(int(piece_type) - 1, N_PIECE_TYPES)


@dataclass(frozen=True, eq=False)
class TetrisState:
    """What the agent observes: numpy arrays plus a few scalars.

    Everything here is a copy -- the state stays valid after further steps.

    ====================  ==============  ==================================================
    field                 shape           meaning
    ====================  ==============  ==================================================
    board                 (20, 10) f32    locked cells, 1.0 = filled
    board_with_piece      (20, 10) f32    the same, plus the falling piece
    current_piece         (4, 2)   i8     the falling piece's cells, board (row, col)
    current_piece_type    (7,)     f32    one-hot piece id   (all zeros = no piece)
    current_rotation      (4,)     f32    one-hot rotation 0..3
    current_column        int             leftmost occupied column (-1 = no piece)
    next_piece_type       (7,)     f32    one-hot id of the piece that spawns next
    next_piece            (4, 2)   i8     that piece's cells in spawn orientation
    queue                 (n,)     i8     upcoming piece ids, queue[0] == next
    score / lines_cleared / level   int
    game_over                       bool
    ====================  ==============  ==================================================
    """

    board: np.ndarray
    board_with_piece: np.ndarray
    current_piece: np.ndarray
    current_piece_type: np.ndarray
    current_rotation: np.ndarray
    current_column: int
    next_piece_type: np.ndarray
    next_piece: np.ndarray
    queue: np.ndarray
    score: int
    lines_cleared: int
    level: int
    game_over: bool

    @property
    def current_piece_name(self) -> Optional[str]:
        """`"T"`, `"I"`, ... or None when there is no active piece."""
        if not self.current_piece_type.any():
            return None
        return PIECE_TYPES[int(self.current_piece_type.argmax())].name

    @property
    def next_piece_name(self) -> Optional[str]:
        if not self.next_piece_type.any():
            return None
        return PIECE_TYPES[int(self.next_piece_type.argmax())].name

    def feature_vector(self) -> np.ndarray:
        """A flat float32 vector for a plain MLP policy (length `FEATURE_VECTOR_SIZE`).

        Board occupancy, the current piece's type and rotation, its column
        (normalised to 0..1) and the next piece's type. CNN-style policies will
        prefer `board` / `board_with_piece` directly; an agent is free to build
        its own vector from the fields instead.
        """
        column = self.current_column / (WIDTH - 1) if self.current_column >= 0 else 0.0
        return np.concatenate(
            [
                self.board.ravel(),
                self.current_piece_type,
                self.current_rotation,
                np.array([column], dtype=np.float32),
                self.next_piece_type,
            ]
        ).astype(np.float32)

    def describe(self) -> str:
        """Field names with shapes and values -- for inspecting a state by eye."""
        rows: List[str] = []
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, np.ndarray):
                shape = "x".join(str(n) for n in value.shape)
                if value.size <= 12:
                    detail = np.array2string(value.ravel(), precision=2)
                else:
                    detail = f"sum={value.sum():g}"
                rows.append(f"{field.name:<19} ({shape}) {value.dtype} {detail}")
            else:
                rows.append(f"{field.name:<19} {value!r}")
        return "\n".join(rows)

    def __repr__(self) -> str:
        return self.describe()


@dataclass(frozen=True)
class StepOutcome:
    """What one placement did -- the facts a reward function is allowed to use.

    The environment measures these; the reward function only decides what they
    are worth. That split is what makes the reward easy to change.
    """

    lines_cleared: int
    score_gained: int
    holes_before: int
    holes_after: int
    height_before: int
    height_after: int
    game_over: bool
    pieces_placed: int

    @property
    def holes_created(self) -> int:
        """Newly covered empty cells (negative if the placement opened holes up)."""
        return self.holes_after - self.holes_before

    @property
    def height_added(self) -> int:
        """Cells the stack grew by (negative if lines cleared brought it down)."""
        return self.height_after - self.height_before


@dataclass(frozen=True)
class RewardConfig:
    """The default, deliberately simple reward -- tune the weights or replace it.

    ::

        score = per_line * lines_cleared
              + hole   * max(0, holes created)
              + height * max(0, cells of new height)
              + per_piece                          (every step that did not end it)
              + game_over                          (only on the final step)

    Only *increases* are penalised, as the name of each weight suggests. Pass
    any of these numbers differently, or hand `TetrisEnv(reward=...)` a function
    of your own taking a `StepOutcome` and returning a float.

    `per_piece` is 0 by default, so the reward is exactly what it always was.
    Set it positive to pay for *staying alive* -- without it every per-step term
    is a penalty and the terminal penalty is paid whenever the episode ends, so
    an agent that never clears a line can only lose by playing on and the
    cheapest episode is the shortest one. It is paid on the steps that do *not*
    end the episode; the piece that ends the game is placed, but not survived.
    """

    per_line: float = 10.0       # per line cleared
    hole: float = -1.0           # per hole created
    height: float = -0.5         # per cell the stack grows
    game_over: float = -100.0    # once, on the step that ends the episode
    per_piece: float = 0.0       # per piece survived (steps that did not end it)

    def __call__(self, outcome: StepOutcome) -> float:
        reward = self.per_line * outcome.lines_cleared
        reward += self.hole * max(0, outcome.holes_created)
        reward += self.height * max(0, outcome.height_added)
        if outcome.game_over:
            reward += self.game_over
        else:
            reward += self.per_piece
        return float(reward)


class TetrisEnv:
    """A gym-style Tetris environment with placement actions.

    ::

        env = TetrisEnv(seed=42)
        state = env.reset()
        while not done:
            action = agent.choose_action(state, env.get_valid_actions())
            state, reward, done, info = env.step(action)

    `seed=None` (the default) gives a fresh piece sequence every episode.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        reward: Optional[Callable[[StepOutcome], float]] = None,
    ):
        self.seed = seed
        self.reward_fn = RewardConfig() if reward is None else reward
        self.game = Game()          # replaced by reset(); keeps `env.game` always valid
        self.pieces_placed = 0
        self.reset()

    # ---------------------------------------------------------------- lifecycle

    def reset(self, seed: Optional[int] = None) -> TetrisState:
        """Start a new episode and return its first state.

        A seeded environment replays the same piece sequence on every reset, so
        two agents on `TetrisEnv(seed=42)` see identical episodes. Pass
        `seed=...` for a different (still deterministic) episode, or construct
        the env without a seed for fresh randomness each time.
        """
        if seed is not None:
            self.seed = seed
        rng = random.Random(self.seed) if self.seed is not None else random.Random()
        self.game = Game(rng)
        self.pieces_placed = 0
        return self.get_state()

    def step(
        self, action
    ) -> Tuple[TetrisState, float, bool, dict]:
        """Place the current piece and return `(next_state, reward, done, info)`.

        `action` is a `Placement`, a `(rotation, column)` tuple, or a dict with
        those keys. An out-of-range action raises `ValueError`. Stepping an
        already-finished episode changes nothing and returns reward 0.0.
        """
        if not self.game.is_game_over():
            placement = self.normalize_action(action)
            outcome, info = self._play(placement)
            reward = float(self.reward_fn(outcome))
            info["reward"] = reward
        else:
            # Terminal no-op: keep the info keys identical so training loops can
            # log every step uniformly.
            info = {
                "lines_cleared": 0,
                "score": self.game.score,
                "score_gained": 0,
                "game_over": True,
                "holes": count_holes(occupancy(self.game.board.as_list())),
                "height": board_height(occupancy(self.game.board.as_list())),
                "holes_created": 0,
                "height_added": 0,
                "placement": None,
                "piece": None,
                "pieces_placed": self.pieces_placed,
                "valid_actions": 0,
                "reward": 0.0,
            }
            reward = 0.0
        return self.get_state(), reward, self.game.is_game_over(), info

    # -------------------------------------------------------------------- state

    def get_state(self) -> TetrisState:
        """The current observation, as numpy arrays (see `TetrisState`)."""
        snapshot = self.game.get_state()
        board = occupancy(snapshot.board)
        board_with_piece = occupancy(snapshot.board_with_active_piece())

        if snapshot.current_piece_type is None or self.game.current is None:
            cells = np.zeros((4, 2), dtype=np.int8)
            rotation = np.zeros(4, dtype=np.float32)
            column = -1
        else:
            piece = self.game.current
            cells = np.array(piece.absolute_cells(), dtype=np.int8)
            rotation = _one_hot(piece.rotation, 4)
            column = int(cells[:, 1].min())

        next_type = snapshot.next_piece_type
        if next_type is None:
            next_cells = np.zeros((4, 2), dtype=np.int8)
        else:
            next_cells = np.array(
                rotated_cells(PieceType(next_type), 0), dtype=np.int8
            )

        return TetrisState(
            board=board,
            board_with_piece=board_with_piece,
            current_piece=cells,
            current_piece_type=_piece_one_hot(snapshot.current_piece_type),
            current_rotation=rotation,
            current_column=column,
            next_piece_type=_piece_one_hot(next_type),
            next_piece=next_cells,
            queue=np.array(snapshot.queue, dtype=np.int8),
            score=int(snapshot.score),
            lines_cleared=int(snapshot.lines_cleared),
            level=int(snapshot.level),
            game_over=bool(snapshot.game_over),
        )

    def get_valid_actions(self) -> List[Placement]:
        """Every placement the current piece can make: `[(rotation, column), ...]`.

        One action per distinct orientation of the piece, at every column where
        it fits between the walls -- e.g. for a T piece, rotations 0..3 at
        columns 0..7 (32 actions); for an O piece only rotation 0 at columns
        0..8 (9 actions), because its other rotations are the same shape.

        These are the placements that are *physically* possible. A valid
        placement can still be a bad one -- stacking in a bad column, or even
        topping out -- exactly as it can for a human player. Returns `[]` once
        the game is over.
        """
        if self.game.is_game_over() or self.game.current is None:
            return []
        return list(placements_for(self.game.current.type))

    def normalize_action(self, action) -> Placement:
        """Coerce an action to a `Placement`, raising `ValueError` if it is illegal.

        Accepts a `Placement`, a `(rotation, column)` pair or a dict with
        `rotation` / `column` keys. Rotations outside 0..3, columns outside the
        wall-to-wall range for that orientation, and non-integer values are all
        rejected.
        """
        if isinstance(action, dict):
            missing = {"rotation", "column"} - set(action)
            if missing:
                raise ValueError(
                    f"action dict is missing {sorted(missing)}; got keys {sorted(action)}"
                )
            rotation, column = action["rotation"], action["column"]
        elif isinstance(action, (tuple, list)):
            if len(action) != 2:
                raise ValueError(
                    f"an action is (rotation, column); got {len(action)} values: {action!r}"
                )
            rotation, column = action
        else:
            raise ValueError(
                f"an action is (rotation, column) or a dict with those keys; got {action!r}"
            )

        rotation = self._as_int("rotation", rotation)
        column = self._as_int("column", column)

        if not 0 <= rotation <= 3:
            raise ValueError(f"rotation must be 0..3, got {rotation}")
        if self.game.is_game_over() or self.game.current is None:
            raise ValueError("the game is over; call reset() before stepping")

        piece_type = self.game.current.type
        width = _orientation_for(piece_type, rotation).width
        highest = WIDTH - width
        if not 0 <= column <= highest:
            raise ValueError(
                f"column {column} is out of range for a {piece_type.name} in rotation "
                f"{rotation}: use 0..{highest}"
            )
        return Placement(rotation, column)

    def is_game_over(self) -> bool:
        return self.game.is_game_over()

    # ---------------------------------------------------------------- internals

    @staticmethod
    def _as_int(name: str, value) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer, float, np.floating)):
            raise ValueError(f"{name} must be a number, got {value!r}")
        if float(value) != int(value):
            raise ValueError(f"{name} must be a whole number, got {value!r}")
        return int(value)

    def _play(self, placement: Placement) -> Tuple[StepOutcome, dict]:
        """Apply one placement and measure what it did."""
        board_before = occupancy(self.game.board.as_list())
        holes_before = count_holes(board_before)
        height_before = board_height(board_before)
        score_before = self.game.score
        lines_before = self.game.lines_cleared
        piece_type = self.game.current.type

        self._place_piece(piece_type, placement)
        self.pieces_placed += 1

        board_after = occupancy(self.game.board.as_list())
        outcome = StepOutcome(
            lines_cleared=self.game.lines_cleared - lines_before,
            score_gained=self.game.score - score_before,
            holes_before=holes_before,
            holes_after=count_holes(board_after),
            height_before=height_before,
            height_after=board_height(board_after),
            game_over=self.game.is_game_over(),
            pieces_placed=self.pieces_placed,
        )
        info = {
            "lines_cleared": outcome.lines_cleared,
            "score": self.game.score,
            "score_gained": outcome.score_gained,
            "game_over": outcome.game_over,
            "holes": outcome.holes_after,
            "height": outcome.height_after,
            "holes_created": max(0, outcome.holes_created),
            "height_added": max(0, outcome.height_added),
            "placement": (
                _canonical_rotation(piece_type, placement.rotation),
                placement.column,
            ),
            "piece": piece_type.name,
            "pieces_placed": self.pieces_placed,
            "valid_actions": len(self.get_valid_actions()),
        }
        return outcome, info

    def _place_piece(self, piece_type: PieceType, placement: Placement) -> None:
        """Rotate, move, hard-drop, lock, clear lines, spawn -- via the engine.

        The piece is put at the requested column with every cell above the well,
        then the engine's own hard drop does the rest. The drop therefore stops
        at the first thing it touches (no tunnelling, no path to get wrong), and
        every rule -- scoring, line clears, lock-out, the 7-bag queue -- stays
        in `Game`.
        """
        cells = rotated_cells(piece_type, placement.rotation)
        min_row = min(row for row, _ in cells)
        min_col = min(col for _, col in cells)
        height = max(row for row, _ in cells) - min_row + 1

        self.game.current = Piece(
            piece_type,
            row=-height - min_row,          # top cell at row -height, i.e. above the well
            col=placement.column - min_col,  # leftmost cell at `placement.column`
            rotation=placement.rotation,
        )
        self.game.step(Action.DROP)

    def __repr__(self) -> str:
        return (
            f"<TetrisEnv seed={self.seed!r} pieces_placed={self.pieces_placed} "
            f"score={self.game.score} game_over={self.game.is_game_over()}>"
        )
