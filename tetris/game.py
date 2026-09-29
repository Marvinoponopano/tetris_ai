"""The Game class -- the environment an AI (or a human) drives.

Design decisions
----------------

1. `Game` owns the board, the active piece, the piece queue, and the score,
   and exposes a gym-style API::

       reset() / step(action) / get_state() / get_valid_actions() / is_game_over()

   so later an agent can be driven with a loop like::

       state = game.get_state()
       action = agent.choose_action(state)
       state, reward, done, info = game.step(action)

2. `step()` applies EXACTLY ONE action and does NOT advance gravity on its
   own. That keeps the dynamics deterministic and easy for an agent to plan
   against. Gravity is provided separately by `apply_gravity()` (fall one
   cell, locking if blocked), which a human game loop calls on a timer.

3. The piece queue uses the "7-bag" randomizer: shuffle all seven pieces, deal
   them out, reshuffle. It is fairer than uniform random (no long droughts)
   and still trivial to implement.

4. Rotation uses simple wall kicks: if a rotation would collide, we try
   nudging the piece left/right/up by one or two cells before giving up.
"""

import random
from typing import List, Optional, Tuple

from .actions import Action
from .board import EMPTY, HEIGHT, WIDTH, Board
from .pieces import Piece, PieceType
from .state import GameState

# Wall-kick offsets (row, col) tried in order when a rotation collides.
WALL_KICKS = ((0, 0), (0, -1), (0, 1), (0, -2), (0, 2), (-1, 0))

# Points per number of lines cleared in one lock, times the current level.
LINE_SCORES = {1: 100, 2: 300, 3: 500, 4: 800}


def spawn_position(piece: Piece) -> Tuple[int, int]:
    """Where a piece appears: centred horizontally, its top cell on row 0.

    The rule `Game._spawn_piece` uses, exposed as a plain function so anything
    that plans ahead can ask where a piece *would* appear -- including whether it
    could appear at all on a given board.
    """
    cells = piece.cells
    min_row = min(r for r, _ in cells)
    min_col = min(c for _, c in cells)
    max_col = max(c for _, c in cells)
    width = max_col - min_col + 1
    row = -min_row                        # top cell lands on row 0
    col = (WIDTH - width) // 2 - min_col  # center it horizontally
    return row, col


class Game:
    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng if rng is not None else random.Random()
        self.reset()

    # -------------------------------------------------------------- public API

    def reset(self) -> GameState:
        """Start a brand-new game and return its initial state."""
        self.board = Board()
        self.score = 0
        self.lines_cleared = 0
        self.level = 1
        self.game_over = False
        self.queue: List[PieceType] = []
        self.current: Optional[Piece] = None
        self._refill_queue()
        self._spawn_piece()
        return self.get_state()

    def step(self, action: Action) -> Tuple[GameState, int, bool, dict]:
        """Apply one action.

        Returns `(next_state, reward, done, info)`:

            * reward -- the score gained by this action (0 for plain moves)
            * done   -- True once the game is over
            * info   -- a small dict of diagnostics (e.g. lines cleared)
        """
        if self.game_over:
            return self.get_state(), 0, True, {"lines_cleared": 0}

        score_before = self.score
        lines_before = self.lines_cleared

        if action == Action.LEFT:
            self._move(-1)
        elif action == Action.RIGHT:
            self._move(1)
        elif action == Action.DOWN:
            self._move_down(score_per_cell=1)   # soft drop earns +1 per cell
        elif action == Action.DROP:
            self._hard_drop()
        elif action == Action.ROTATE_CW:
            self._rotate(direction=1)
        elif action == Action.ROTATE_CCW:
            self._rotate(direction=-1)
        else:
            raise ValueError(f"Unknown action: {action!r}")

        reward = self.score - score_before
        info = {"lines_cleared": self.lines_cleared - lines_before}
        return self.get_state(), reward, self.game_over, info

    def get_state(self) -> GameState:
        """A snapshot of the current game, shaped for an AI to consume."""
        if self.game_over or self.current is None:
            cur_type = cur_row = cur_col = cur_rot = None
        else:
            cur_type = int(self.current.type)
            cur_row = self.current.row
            cur_col = self.current.col
            cur_rot = self.current.rotation

        next_type = int(self.queue[0]) if self.queue else None
        return GameState(
            board=self.board.as_list(),
            current_piece_type=cur_type,
            current_piece_row=cur_row,
            current_piece_col=cur_col,
            current_piece_rotation=cur_rot,
            next_piece_type=next_type,
            queue=[int(p) for p in self.queue],
            score=self.score,
            lines_cleared=self.lines_cleared,
            level=self.level,
            game_over=self.game_over,
        )

    def get_valid_actions(self) -> List[Action]:
        """The actions that are legal right now (useful for the AI later)."""
        if self.game_over:
            return []
        valid: List[Action] = []
        if not self.board.collides(
            self.current.absolute_cells(self.current.row, self.current.col - 1)
        ):
            valid.append(Action.LEFT)
        if not self.board.collides(
            self.current.absolute_cells(self.current.row, self.current.col + 1)
        ):
            valid.append(Action.RIGHT)
        if not self.board.collides(
            self.current.absolute_cells(self.current.row + 1, self.current.col)
        ):
            valid.append(Action.DOWN)
        valid.append(Action.DROP)  # always legal while the game runs
        if self._rotation_works(1):
            valid.append(Action.ROTATE_CW)
        if self._rotation_works(-1):
            valid.append(Action.ROTATE_CCW)
        return valid

    def is_game_over(self) -> bool:
        return self.game_over

    def apply_gravity(self) -> None:
        """Advance gravity by one cell; lock the piece if it cannot move.

        A human game loop calls this on a timer. The AI simply never calls it
        and instead drives every move through `step()`.
        """
        if not self.game_over:
            self._move_down(score_per_cell=0)  # gravity earns no soft-drop points

    def render(self) -> str:
        """A small text rendering of the board (for humans and debugging)."""
        grid = self.board.as_list()
        if self.current is not None and not self.game_over:
            for r, c in self.current.absolute_cells():
                if 0 <= r < HEIGHT:
                    grid[r][c] = int(self.current.type)
        border = "+" + "-" * WIDTH + "+"
        lines = [border]
        for row in grid:
            lines.append("|" + "".join("#" if cell else "." for cell in row) + "|")
        lines.append(border)
        return "\n".join(lines)

    # --------------------------------------------------------------- internals

    def _move(self, dcol: int) -> None:
        """Shift the active piece left (-1) or right (+1) if possible."""
        self._try_move(0, dcol)

    def _try_move(self, drow: int, dcol: int) -> bool:
        new_row = self.current.row + drow
        new_col = self.current.col + dcol
        if self.board.collides(self.current.absolute_cells(new_row, new_col)):
            return False
        self.current.row = new_row
        self.current.col = new_col
        return True

    def _move_down(self, score_per_cell: int) -> None:
        """Fall one row; lock if blocked. Optionally award soft-drop points."""
        if self._try_move(1, 0):
            self.score += score_per_cell
        else:
            self._lock_piece()

    def _hard_drop(self) -> None:
        """Fall straight to the bottom (+2 points per cell) and lock."""
        distance = 0
        while self._try_move(1, 0):
            distance += 1
        self.score += distance * 2
        self._lock_piece()

    def _rotate(self, direction: int) -> None:
        """Rotate, applying wall kicks if the naive rotation collides."""
        if direction > 0:
            rotated = self.current.rotate_cw()
        else:
            rotated = self.current.rotate_ccw()
        for drow, dcol in WALL_KICKS:
            r = self.current.row + drow
            c = self.current.col + dcol
            if not self.board.collides(rotated.absolute_cells(r, c)):
                rotated.row = r
                rotated.col = c
                self.current = rotated
                return

    def _rotation_works(self, direction: int) -> bool:
        """True if a rotation succeeds with some wall kick (does not mutate)."""
        rotated = (
            self.current.rotate_cw() if direction > 0 else self.current.rotate_ccw()
        )
        return any(
            not self.board.collides(
                rotated.absolute_cells(self.current.row + dr, self.current.col + dc)
            )
            for dr, dc in WALL_KICKS
        )

    def _lock_piece(self) -> None:
        """Lock the active piece, clear any full lines, then spawn the next."""
        cells = self.current.absolute_cells()
        self.board.place(cells, int(self.current.type))
        cleared = self.board.clear_lines()
        self._award_lines(cleared)

        # Lock-out: a piece that settles entirely above the visible board means
        # the player has stacked to the very top -> game over.
        if all(r < 0 for r, _ in cells):
            self.game_over = True
            return

        self._spawn_piece()

    def _award_lines(self, cleared: int) -> None:
        if cleared:
            self.score += LINE_SCORES[cleared] * self.level
            self.lines_cleared += cleared
            self.level = self.lines_cleared // 10 + 1

    def _spawn_piece(self) -> None:
        """Take the next piece from the queue and place it at the top."""
        if not self.queue:
            self._refill_queue()
        piece_type = self.queue.pop(0)
        self._refill_queue()

        piece = Piece(piece_type)
        piece.row, piece.col = self._spawn_position(piece)

        if self.board.collides(piece.absolute_cells()):
            # The new piece cannot even fit -- the board is too full.
            self.game_over = True
        self.current = piece

    def _spawn_position(self, piece: Piece) -> Tuple[int, int]:
        """Center the piece horizontally, with its top cell on the top row."""
        return spawn_position(piece)

    def _refill_queue(self) -> None:
        """Keep the queue topped up to 7 using the '7-bag' randomizer."""
        while len(self.queue) < 7:
            bag = list(PieceType)
            self.rng.shuffle(bag)
            self.queue.extend(bag)
