"""The state and action encoding an RL agent sees -- plain NumPy, no PyTorch.

Two encodings live here, and between them they are the whole interface between
the environment and a network:

**State** -- `encode_state` turns a `TetrisState` into a `(3, 20, 10)` float32
tensor of binary planes:

    0  board occupancy      the locked stack
    1  current piece        the falling piece, at the cells it actually occupies
    2  next piece           the piece that spawns next, drawn at the top-left
                            corner (its only meaningful position is "after this")

The piece planes carry the piece's *shape*, which is all the agent needs about
it: the seven tetrominoes have seven distinguishable shapes, so shape + channel
identifies the piece and its rotation without any one-hot vectors. Everything is
0.0/1.0, so nothing needs rescaling.

**Action** -- `(rotation, column)` flattened to a single integer::

    index = rotation * WIDTH + column        # 0..39

A piece has at most four distinct orientations (`orientations()`) and at most
ten columns, so 40 outputs cover every placement any piece can make. Which of
the 40 are *legal* depends only on the current piece type, so `action_mask` is a
lookup: for an O piece rotations 1-3 are the same shape as rotation 0 and are
excluded, exactly as `placements_for()` excludes them. The mask is what lets a
network output all 40 Q-values and still only ever choose a real move.

`action_mask_for(state)` agrees, index for index, with
`TetrisEnv.get_valid_actions()` -- there is a test that asserts it for every
piece type, because the whole action mapping rests on that agreement.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np

from ..board import HEIGHT, WIDTH
from ..metrics import column_heights, covered_cells
from ..pieces import PieceType, rotated_cells
from ..env import Placement, TetrisState, placements_for

#: 4 rotations x 10 columns: every placement, legal or not.
ACTION_SIZE = 4 * WIDTH

#: The shape of one encoded state.
STATE_SHAPE = (3, HEIGHT, WIDTH)

#: The plane indices, named so callers do not have to remember them.
BOARD_PLANE = 0
CURRENT_PIECE_PLANE = 1
NEXT_PIECE_PLANE = 2

#: The extended encoding (`encode_state_with_features`): the three planes above,
#: plus the two measurements of stack shape a human player reads off the board
#: without thinking.
EXTENDED_SHAPE = (5, HEIGHT, WIDTH)

#: Column heights, as a fraction of the board height, held down the column.
HEIGHT_PLANE = 3

#: Cells that are empty but covered by a filled cell above them -- a hole.
HOLE_PLANE = 4

#: The encoding that also shows the piece *after* the next one
#: (`encode_state_with_queue`): the five planes above plus that piece's shape.
QUEUE_SHAPE = (6, HEIGHT, WIDTH)

#: The piece after next, drawn at the spawn corner like `NEXT_PIECE_PLANE`.
QUEUE_PLANE = 5


def action_index(rotation: int, column: int) -> int:
    """The flat index of a `(rotation, column)` placement."""
    return int(rotation) * WIDTH + int(column)


def placement_for_index(index: int) -> Placement:
    """The placement a flat action index means."""
    if not 0 <= int(index) < ACTION_SIZE:
        raise ValueError(f"action index must be 0..{ACTION_SIZE - 1}, got {index!r}")
    return Placement(int(index) // WIDTH, int(index) % WIDTH)


def index_for_placement(placement) -> int:
    """The flat index of a `Placement` (or any `(rotation, column)` pair)."""
    rotation, column = placement
    if not 0 <= int(rotation) < 4:
        raise ValueError(f"rotation must be 0..3, got {rotation!r}")
    if not 0 <= int(column) < WIDTH:
        raise ValueError(f"column must be 0..{WIDTH - 1}, got {column!r}")
    return action_index(rotation, column)


@lru_cache(maxsize=None)
def mask_for_piece(piece_type: PieceType) -> np.ndarray:
    """The 40-entry boolean mask of placements this piece type can make.

    Cached (there are only seven) and returned read-only: callers should copy it
    if they intend to modify it.
    """
    mask = np.zeros(ACTION_SIZE, dtype=bool)
    for placement in placements_for(piece_type):
        mask[action_index(placement.rotation, placement.column)] = True
    mask.flags.writeable = False
    return mask


def mask_for_state(state: TetrisState) -> np.ndarray:
    """The legal placements for the piece currently falling, as a 40-entry mask.

    An empty mask means there is no active piece (a finished game). Otherwise it
    has at least nine entries: every piece fits somewhere on an empty board.
    """
    piece_type = _piece_type_of(state)
    if piece_type is None:
        return np.zeros(ACTION_SIZE, dtype=bool)
    return mask_for_piece(piece_type)


def valid_indices(state: TetrisState) -> np.ndarray:
    """The flat indices of the legal placements, ascending."""
    return np.flatnonzero(mask_for_state(state))


def _piece_type_of(state: TetrisState):
    """The `PieceType` of the falling piece, or None if there is not one."""
    if state.current_piece_type is None or not np.any(state.current_piece_type):
        return None
    return PieceType(int(np.argmax(state.current_piece_type)) + 1)


def encode_state(state: TetrisState) -> np.ndarray:
    """A `(3, 20, 10)` float32 tensor for one state (see the module docstring)."""
    planes = np.zeros(STATE_SHAPE, dtype=np.float32)
    planes[BOARD_PLANE] = (state.board > 0).astype(np.float32)
    if _piece_type_of(state) is not None:
        _draw_cells(planes[CURRENT_PIECE_PLANE], state.current_piece)
    if np.any(state.next_piece_type):
        _draw_cells(planes[NEXT_PIECE_PLANE], _spawn_cells(state.next_piece))
    return planes


def encode_states(states) -> np.ndarray:
    """A `(n, 3, 20, 10)` float32 batch, for feeding a network directly."""
    batch = np.zeros((len(states),) + STATE_SHAPE, dtype=np.float32)
    for index, state in enumerate(states):
        batch[index] = encode_state(state)
    return batch


def encode_state_with_features(state: TetrisState) -> np.ndarray:
    """A `(5, 20, 10)` tensor: `encode_state` plus the stack's own geometry.

    Two extra planes, both taken from `tetris.metrics` -- the very functions the
    heuristic agent scores with:

    * `HEIGHT_PLANE` -- each column filled with its height as a fraction of the
      board, so "this column is 14 tall" is one number in one place rather than
      something to be inferred from 20 cells.
    * `HOLE_PLANE` -- the cells that are empty but covered (`covered_cells`),
      which is the state the heuristic's hole penalty is about.

    Why: `encode_state` shows the network *where the blocks are*, and then the
    network has to work out the geometry itself, out of the same gradient steps
    it needs for everything else. Eighteen thousand decisions in, G still plays
    like an agent that cannot see how tall a column is (diagnosis: 4.08 lines
    per 100 pieces against the heuristic's 39.76, with the same information
    available). The heuristic is not smarter about arithmetic -- it is handed
    these two numbers for free. This plane set hands them to the network.

    Cost: 200 more input numbers and no new decisions. Everything else in the
    pipeline reads `state_shape` off the network, so the replay buffer, the
    checkpoints and the viewer follow the wider shape automatically.
    """
    planes = np.zeros(EXTENDED_SHAPE, dtype=np.float32)
    planes[:3] = encode_state(state)
    heights = column_heights(state.board).astype(np.float32) / float(HEIGHT)
    planes[HEIGHT_PLANE] = heights[None, :]
    planes[HOLE_PLANE] = covered_cells(state.board).astype(np.float32)
    return planes


def encode_states_with_features(states) -> np.ndarray:
    """A `(n, 5, 20, 10)` float32 batch (see `encode_state_with_features`)."""
    batch = np.zeros((len(states),) + EXTENDED_SHAPE, dtype=np.float32)
    for index, state in enumerate(states):
        batch[index] = encode_state_with_features(state)
    return batch


def encode_state_with_queue(state: TetrisState) -> np.ndarray:
    """A `(6, 20, 10)` tensor: the five feature planes plus the piece after next.

    `TetrisState.queue` has held the upcoming pieces all along and the encoding
    has never shown them: the agent sees the piece in hand and the one that
    follows, and nothing beyond. So it cannot do what a human does on the last
    two pieces of a board -- "this column needs an I, and there is one three
    pieces away, so park the stack flat until it arrives" -- and a single
    placement that is bad for the current piece can be the best move in the game.

    The plane is the same convention `NEXT_PIECE_PLANE` already uses: the piece's
    shape, drawn at the top-left spawn corner, which is how a shape identifies a
    tetromino without a one-hot vector.

    This is the cheapest form of lookahead available here, and it is worth being
    precise about why: the action space is whole placements, so "looking one move
    ahead" would mean evaluating all 40 placements and then every placement
    reachable from each -- 40 x 40 network calls per decision. Showing the queue
    gives the network the *information* a lookahead would use, and lets it decide
    what to do with it at the cost of one plane.
    """
    planes = np.zeros(QUEUE_SHAPE, dtype=np.float32)
    planes[:5] = encode_state_with_features(state)
    upcoming = _upcoming_cells(state)
    if upcoming is not None:
        _draw_cells(planes[QUEUE_PLANE], _spawn_cells(upcoming))
    return planes


def _upcoming_cells(state: TetrisState) -> Optional[np.ndarray]:
    """The cells of the piece after next, in spawn orientation, or None.

    `queue[0]` is the next piece (already drawn in its own plane), so this is
    `queue[1]`. Missing or short queues (an engine built without one) give None
    rather than an exception: the plane is then simply empty, which is the honest
    encoding of "nothing is known about what comes next".
    """
    queue = getattr(state, "queue", None)
    if queue is None or len(np.asarray(queue).reshape(-1)) < 2:
        return None
    piece_id = int(np.asarray(queue).reshape(-1)[1])
    if piece_id < 1 or piece_id > len(PieceType):
        return None
    return np.asarray(rotated_cells(PieceType(piece_id), 0), dtype=np.int64)


def encode_states_with_queue(states) -> np.ndarray:
    """A `(n, 6, 20, 10)` float32 batch (see `encode_state_with_queue`)."""
    batch = np.zeros((len(states),) + QUEUE_SHAPE, dtype=np.float32)
    for index, state in enumerate(states):
        batch[index] = encode_state_with_queue(state)
    return batch


def _spawn_cells(cells: np.ndarray) -> np.ndarray:
    """A preview piece shifted to the top-left corner, where it belongs on a plane."""
    if cells is None or len(cells) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    rows = cells[:, 0] - cells[:, 0].min()
    columns = cells[:, 1] - cells[:, 1].min()
    return np.stack([rows, columns], axis=1)


def _draw_cells(plane: np.ndarray, cells: np.ndarray) -> None:
    """Set the given `(row, column)` cells to 1.0, ignoring anything off-board."""
    for row, column in np.asarray(cells, dtype=np.int64).reshape(-1, 2):
        if 0 <= row < HEIGHT and 0 <= column < WIDTH:
            plane[row, column] = 1.0


__all__ = [
    "ACTION_SIZE",
    "BOARD_PLANE",
    "CURRENT_PIECE_PLANE",
    "EXTENDED_SHAPE",
    "HEIGHT_PLANE",
    "HOLE_PLANE",
    "QUEUE_PLANE",
    "QUEUE_SHAPE",
    "NEXT_PIECE_PLANE",
    "STATE_SHAPE",
    "action_index",
    "encode_state",
    "encode_state_with_features",
    "encode_state_with_queue",
    "encode_states",
    "encode_states_with_features",
    "encode_states_with_queue",
    "index_for_placement",
    "mask_for_piece",
    "mask_for_state",
    "placement_for_index",
    "valid_indices",
]
