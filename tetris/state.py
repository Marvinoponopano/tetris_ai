"""A snapshot of the game state, shaped for an AI to consume.

The AI parts later will mostly read two things:

    * `state.board`                 -- the locked cells (2D list of ints)
    * `state.board_with_active_piece()` -- the same, plus the falling piece

Everything else (score, next piece, queue) is metadata the agent can use for
better decisions or for reward shaping.
"""

from dataclasses import dataclass
from typing import List, Optional

from .pieces import PieceType, rotated_cells


@dataclass(frozen=True)
class GameState:
    """An immutable snapshot of the game at a single moment.

    `board` contains only *locked* cells: 0 = empty, 1..7 = the piece id that
    locked there. The active (falling) piece is described separately by the
    `current_piece_*` fields so the two never get mixed up.
    """

    board: List[List[int]]
    current_piece_type: Optional[int]      # None once the game is over
    current_piece_row: Optional[int]
    current_piece_col: Optional[int]
    current_piece_rotation: Optional[int]  # 0..3 clockwise quarter-turns
    next_piece_type: Optional[int]         # piece id that spawns next
    queue: List[int]                       # upcoming ids; queue[0] is next
    score: int
    lines_cleared: int
    level: int
    game_over: bool

    def board_with_active_piece(self) -> List[List[int]]:
        """The board with the falling piece overlaid (handy for ML features)."""
        merged = [row[:] for row in self.board]
        if self.current_piece_type is not None and not self.game_over:
            cells = rotated_cells(
                PieceType(self.current_piece_type), self.current_piece_rotation
            )
            for dr, dc in cells:
                r = self.current_piece_row + dr
                c = self.current_piece_col + dc
                if 0 <= r < len(merged) and 0 <= c < len(merged[0]):
                    merged[r][c] = self.current_piece_type
        return merged
