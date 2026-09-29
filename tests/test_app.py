"""Tests for the frontend-independent app loop (tetris/app.py).

No pygame here: `TetrisApp` is deliberately display-agnostic, so these tests
drive it with a scripted stand-in controller (the shape the AI policy will take)
and plain numbers instead of a clock.
"""

import random

from tetris.actions import Action
from tetris.app import (
    GRAVITY_MIN_MS,
    GRAVITY_START_MS,
    TetrisApp,
    gravity_interval_ms,
)
from tetris.board import HEIGHT, WIDTH
from tetris.game import Game
from tetris.pieces import Piece, PieceType


class ScriptedController:
    """Hands out a fixed list of actions, one per frame (an AI stand-in)."""

    def __init__(self, actions=()):
        self.actions = list(actions)
        self.calls = 0

    def choose_action(self, game):
        self.calls += 1
        return self.actions.pop(0) if self.actions else None


def fresh_app(**kwargs) -> TetrisApp:
    kwargs.setdefault("game", Game(random.Random(1234)))
    return TetrisApp(**kwargs)


def play_until_game_over(app: TetrisApp, max_pieces: int = 200) -> int:
    """Hard-drop pieces without moving them; they stack up in the middle."""
    pieces = 0
    while not app.game_over and pieces < max_pieces:
        app.apply(Action.DROP)
        pieces += 1
    return pieces


# ------------------------------------------------------------------- gravity


def test_gravity_gets_faster_with_level_but_has_a_floor():
    assert gravity_interval_ms(1) == GRAVITY_START_MS
    assert gravity_interval_ms(2) < gravity_interval_ms(1)
    assert gravity_interval_ms(50) == GRAVITY_MIN_MS
    assert gravity_interval_ms(0) == GRAVITY_START_MS     # level is clamped


def advance(app: TetrisApp, total_ms: float, frame_ms: float = 100.0) -> int:
    """Feed `total_ms` of time in realistic frame-sized chunks; return ticks."""
    ticks = 0
    remaining = total_ms
    while remaining > 0:
        ticks += app.update(min(frame_ms, remaining))
        remaining -= frame_ms
    return ticks


def test_gravity_needs_a_full_interval_to_move_a_cell():
    app = fresh_app()
    start = app.game.current.row
    assert advance(app, GRAVITY_START_MS - 100) == 0       # 700ms is not enough
    assert app.game.current.row == start
    assert advance(app, 200) == 1                          # the next 100ms tips it
    assert app.game.current.row == start + 1


def test_a_long_frame_is_clamped_so_the_piece_never_teleports():
    app = fresh_app()
    row = app.game.current.row
    app.update(10_000)                                     # e.g. a dragged window
    assert app.game.current.row == row


def test_gravity_can_be_disabled_for_the_ai():
    app = fresh_app(gravity=False)
    row = app.game.current.row
    assert app.update(5_000) == 0
    assert app.game.current.row == row


def test_gravity_stops_at_game_over():
    app = fresh_app()
    play_until_game_over(app)
    assert app.game_over
    assert app.update(GRAVITY_START_MS) == 0


# ------------------------------------------------------------------- actions


def test_actions_go_through_the_engine():
    app = fresh_app()
    col = app.game.current.col
    assert app.apply(Action.LEFT) is Action.LEFT
    assert app.game.current.col == col - 1


def test_none_and_illegal_actions_are_ignored():
    app = fresh_app()
    assert app.apply(None) is None
    app.game.game_over = True
    assert app.apply(Action.LEFT) is None


def test_pause_blocks_both_gravity_and_actions():
    app = fresh_app()
    row, col = app.game.current.row, app.game.current.col
    assert app.toggle_pause() is True
    assert app.update(GRAVITY_START_MS * 3) == 0
    assert app.apply(Action.LEFT) is None
    assert (app.game.current.row, app.game.current.col) == (row, col)
    assert app.toggle_pause() is False                     # and it resumes


def test_restart_resets_the_game_and_the_pause():
    app = fresh_app()
    app.apply(Action.DROP)
    app.toggle_pause()
    app.restart()
    assert app.game.score == 0
    assert app.game.lines_cleared == 0
    assert app.game.level == 1
    assert not app.paused
    assert not app.game_over
    assert all(cell == 0 for row in app.game.board.as_list() for cell in row)


# ---------------------------------------------------------------- controller


def test_tick_applies_one_controller_action_per_frame():
    controller = ScriptedController([Action.LEFT, Action.LEFT, Action.RIGHT])
    app = fresh_app(controller=controller)
    start = app.game.current.col

    assert app.tick(0) is Action.LEFT
    assert app.game.current.col == start - 1
    assert app.tick(0) is Action.LEFT
    assert app.game.current.col == start - 2
    assert app.tick(0) is Action.RIGHT
    assert app.game.current.col == start - 1
    assert app.tick(0) is None                             # the script ran out
    assert controller.calls == 4


def test_tick_without_a_controller_only_applies_gravity():
    app = fresh_app()
    start = app.game.current.row
    for _ in range(8):                                     # 8 x 100ms = the interval
        assert app.tick(100) is None
    assert app.game.current.row == start + 1


def test_a_controller_cannot_move_the_piece_after_game_over():
    controller = ScriptedController([Action.LEFT] * 500)
    app = fresh_app(controller=controller)
    play_until_game_over(app)
    assert app.tick(0) is None


# ------------------------------------------------------------- game over flow


def test_pieces_stack_until_the_game_ends():
    app = fresh_app()
    pieces = play_until_game_over(app)
    assert app.game_over
    assert pieces < 200                                    # the stack really tops out
    assert any(
        cell != 0 for row in app.game.board.as_list() for cell in row
    ), "the board should hold the stack that ended the game"


def test_frames_never_busy_wait_at_a_fixed_rate():
    app = fresh_app(fps=60)
    assert app.frame_ms == 1000.0 / 60
    assert fresh_app(fps=30).frame_ms == 1000.0 / 30
    for bad in (0, -1):
        try:
            fresh_app(fps=bad)
        except ValueError as exc:
            assert "fps" in str(exc)
        else:
            raise AssertionError("expected a ValueError for a non-positive fps")


# --------------------------------------------------------------- determinism


def test_a_seed_makes_the_piece_sequence_repeatable():
    first = Game(random.Random(99))
    second = Game(random.Random(99))
    assert first.queue == second.queue
    assert first.current.type == second.current.type
    assert Game(random.Random(1)).queue != Game(random.Random(2)).queue


def test_a_seeded_app_with_a_factory_replays_the_same_game():
    """`play.py` passes a factory, which is what makes seeded runs comparable."""
    app = TetrisApp(
        Game(random.Random(7)),
        fps=60,
        game_factory=lambda: Game(random.Random(7)),
    )
    first_round = []
    for _ in range(3):
        app.apply(Action.DROP)
        first_round.append(app.game.board.as_list())

    app.restart()
    assert app.game.score == 0                             # fresh game
    assert all(cell == 0 for row in app.game.board.as_list() for cell in row)
    second_round = []
    for _ in range(3):
        app.apply(Action.DROP)
        second_round.append(app.game.board.as_list())

    assert first_round == second_round


def test_without_a_factory_restart_keeps_the_same_game_object():
    app = fresh_app()
    before = app.game
    app.apply(Action.DROP)
    app.restart()
    assert app.game is before                             # reset in place
    assert app.game.score == 0
    assert all(cell == 0 for row in app.game.board.as_list() for cell in row)


def test_two_apps_with_the_same_seed_stay_in_step():
    script = [Action.LEFT, Action.ROTATE_CW, Action.DROP, Action.RIGHT, Action.DROP]
    apps = []
    for _ in range(2):
        controller = ScriptedController(script)
        app = TetrisApp(Game(random.Random(5)), controller, fps=60)
        for _ in range(len(script)):
            app.tick(GRAVITY_START_MS)                     # identical frame times
        apps.append(app)

    assert apps[0].game.board.as_list() == apps[1].game.board.as_list()
    assert apps[0].game.score == apps[1].game.score
    assert apps[0].game.queue == apps[1].game.queue


def test_line_clears_score_through_the_app():
    app = fresh_app()
    # Fill every column but the first, then drop a flat I piece over that gap:
    # the row completes, clears, and scores.
    for col in range(1, WIDTH):
        app.game.board.grid[HEIGHT - 1][col] = int(PieceType.L)
    app.game.current = Piece(PieceType.I, HEIGHT - 1, 2)   # covers columns 0..3
    app.apply(Action.DROP)

    assert app.game.lines_cleared == 1
    assert app.game.score >= 100
    assert all(cell == 0 for cell in app.game.board.as_list()[HEIGHT - 1])
