"""Tests for the pygame frontend's input wiring and view geometry.

Rendering is deliberately *not* pixel-tested -- that is what looking at the
window is for. What is tested here is the wiring that must not break: which key
means which action, how events turn into actions one frame at a time, that held
keys repeat, and that the view draws without raising.

Everything runs on SDL's dummy video driver, so no window opens.
"""

import os
import random

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")   # before pygame is imported
pygame = pytest.importorskip("pygame")

from tetris.actions import Action                                  # noqa: E402
from tetris.app import TetrisApp                                   # noqa: E402
from tetris.board import HEIGHT, WIDTH                             # noqa: E402
from tetris.game import Game                                       # noqa: E402
from tetris.pieces import Piece, PieceType                         # noqa: E402
from tetris.pygame_view import (                                   # noqa: E402
    KEY_TO_ACTION,
    KEY_TO_COMMAND,
    HumanController,
    PygameView,
    landing_cells,
    run_human,
)


@pytest.fixture(scope="module", autouse=True)
def display():
    pygame.init()
    yield
    pygame.quit()


@pytest.fixture
def app():
    return TetrisApp(Game(random.Random(5)), HumanController(), fps=60)


@pytest.fixture
def view():
    return PygameView(cell=20, grid="none")


def keydown(key):
    return pygame.event.Event(pygame.KEYDOWN, key=key)


def keyup(key):
    return pygame.event.Event(pygame.KEYUP, key=key)


def press(key):
    """Queue a full key press (down and up) as pygame would deliver it."""
    pygame.event.post(keydown(key))
    pygame.event.post(keyup(key))


def step(app, view, frames=1):
    """Run the real loop for a few frames, consuming whatever is queued."""
    return run_human(app, view, app.controller, max_frames=frames)


# --------------------------------------------------------------- key mapping


def test_the_documented_controls_map_to_the_right_actions():
    assert KEY_TO_ACTION[pygame.K_LEFT] is Action.LEFT
    assert KEY_TO_ACTION[pygame.K_RIGHT] is Action.RIGHT
    assert KEY_TO_ACTION[pygame.K_DOWN] is Action.DOWN
    assert KEY_TO_ACTION[pygame.K_UP] is Action.ROTATE_CW
    assert KEY_TO_ACTION[pygame.K_SPACE] is Action.DROP


def test_letter_keys_mirror_the_terminal_frontend():
    assert KEY_TO_ACTION[pygame.K_a] is Action.LEFT
    assert KEY_TO_ACTION[pygame.K_d] is Action.RIGHT
    assert KEY_TO_ACTION[pygame.K_s] is Action.DOWN
    assert KEY_TO_ACTION[pygame.K_w] is Action.ROTATE_CW
    assert KEY_TO_ACTION[pygame.K_x] is Action.ROTATE_CW
    assert KEY_TO_ACTION[pygame.K_z] is Action.ROTATE_CCW


def test_command_keys_are_not_game_actions():
    assert KEY_TO_COMMAND[pygame.K_r] == "restart"
    assert KEY_TO_COMMAND[pygame.K_p] == "pause"
    assert KEY_TO_COMMAND[pygame.K_g] == "ghost"
    assert KEY_TO_COMMAND[pygame.K_q] == "quit"
    assert KEY_TO_COMMAND[pygame.K_ESCAPE] == "quit"
    assert not set(KEY_TO_COMMAND) & set(KEY_TO_ACTION)   # no key does both


# ---------------------------------------------------------------- controller


def test_events_become_one_action_per_frame(app):
    controller = app.controller
    assert controller.choose_action(app.game) is None       # nothing yet

    controller.feed([keydown(pygame.K_LEFT)], 0)
    assert controller.choose_action(app.game) is Action.LEFT
    assert controller.choose_action(app.game) is None       # one per frame


def test_command_keys_are_reported_to_the_loop(app):
    controller = app.controller
    assert controller.feed([keydown(pygame.K_r)], 0) == ["restart"]
    assert controller.feed([keydown(pygame.K_p)], 0) == ["pause"]
    assert controller.feed([keydown(pygame.K_ESCAPE)], 0) == ["quit"]
    assert controller.feed([keydown(pygame.K_g)], 0) == ["ghost"]
    assert controller.feed([keydown(pygame.K_LEFT)], 0) == []   # an action, not a command


def test_holding_a_key_repeats_after_the_delay():
    controller = HumanController(das_ms=100, arr_ms=50)
    controller.feed([keydown(pygame.K_LEFT)], 0)
    assert controller.choose_action(None) is Action.LEFT     # immediate move

    controller.feed([], 50)                                  # still inside the delay
    assert controller.choose_action(None) is None

    controller.feed([], 60)                                  # past the delay: one repeat
    assert controller.choose_action(None) is Action.LEFT
    assert controller.choose_action(None) is None

    controller.feed([], 50)                                  # and every gap after that
    assert controller.choose_action(None) is Action.LEFT


def test_releasing_a_key_stops_the_repeat():
    controller = HumanController(das_ms=10, arr_ms=10)
    controller.feed([keydown(pygame.K_RIGHT)], 0)
    assert controller.choose_action(None) is Action.RIGHT

    controller.feed([keyup(pygame.K_RIGHT)], 100)
    assert controller.choose_action(None) is None


def test_clear_forgets_queued_moves():
    controller = HumanController()
    controller.feed([keydown(pygame.K_LEFT), keydown(pygame.K_LEFT)], 0)
    controller.clear()
    assert controller.choose_action(None) is None


def test_a_controller_satisfies_the_ai_shaped_interface(app):
    """The AI will implement exactly this: one method, game in, action out."""
    controller = app.controller
    controller.feed([keydown(pygame.K_UP), keydown(pygame.K_SPACE)], 0)
    chosen = [controller.choose_action(app.game) for _ in range(2)]
    assert chosen == [Action.ROTATE_CW, Action.DROP]


# ---------------------------------------------------------------------- view


def test_view_geometry_matches_the_engine_board(view):
    assert view.well.size == (WIDTH * view.cell, HEIGHT * view.cell)
    assert view.cell_rect(0, 0).topleft == view.well.topleft
    assert view.cell_rect(HEIGHT - 1, WIDTH - 1).bottomright == view.well.bottomright
    assert view.cell_rect(0, 1).x == view.well.x + view.cell


def test_view_draws_a_game_without_raising(app, view):
    step(app, view)                                          # paint one frame
    view.draw(app)
    view.flip()
    colours = {
        view.screen.get_at((x, y))[:3]
        for x in range(0, view.screen.get_width(), 4)
        for y in range(0, view.screen.get_height(), 4)
    }
    assert len(colours) > 10                                 # not a blank screen


def test_view_draws_paused_and_game_over_states(app, view):
    for _ in range(200):                                     # stack until it ends
        if app.game_over:
            break
        app.apply(Action.DROP)
    assert app.game_over
    view.draw(app)                                           # game-over overlay

    app.toggle_pause()
    view.draw(app)                                           # paused overlay
    view.flip()


def test_view_rejects_an_unknown_grid():
    with pytest.raises(ValueError):
        PygameView(grid="sparkles")


def test_view_saves_a_screenshot(app, view, tmp_path):
    target = tmp_path / "frame.png"
    view.draw(app)
    assert view.save(str(target)) == str(target)
    assert target.exists() and target.stat().st_size > 0


# -------------------------------------------------------------- ghost helper


def test_landing_cells_rest_on_the_floor_of_an_empty_board(app):
    cells = app.game.current.absolute_cells()
    landing = landing_cells(app.game)

    assert sorted(col for _, col in landing) == sorted(col for _, col in cells)
    fallen = max(row for row, _ in landing) - max(row for row, _ in cells)
    assert fallen > 0                                        # it has further to go
    assert sorted((row - fallen, col) for row, col in landing) == sorted(cells)
    assert max(row for row, _ in landing) == HEIGHT - 1      # resting on the floor
    assert all(cell == 0 for row in app.game.board.as_list() for cell in row)


def test_landing_cells_stack_on_top_of_locked_blocks(app):
    columns = {col for _, col in app.game.current.absolute_cells()}
    for col in columns:
        app.game.board.place([(HEIGHT - 1, col)], int(PieceType.Z))
    landing = landing_cells(app.game)
    assert max(row for row, _ in landing) == HEIGHT - 2      # one row higher


def test_landing_cells_is_empty_without_a_piece(app):
    app.game.current = None
    assert landing_cells(app.game) == []


def test_landing_cells_does_not_change_the_game(app):
    before = app.game.board.as_list()
    landing_cells(app.game)
    assert app.game.board.as_list() == before


# -------------------------------------------------------- end to end in a loop


def test_the_loop_moves_the_piece_on_arrow_keys(app, view):
    step(app, view)                                          # first frame
    start = app.game.current.col

    press(pygame.K_LEFT)
    step(app, view)
    assert app.game.current.col == start - 1

    press(pygame.K_RIGHT)
    step(app, view)
    assert app.game.current.col == start


def test_the_loop_rotates_on_the_up_arrow(app, view):
    step(app, view)
    start = app.game.current.rotation
    press(pygame.K_UP)
    step(app, view)
    assert app.game.current.rotation == (start + 1) % 4


def test_the_loop_soft_drops_and_scores_on_the_down_arrow(app, view):
    step(app, view)
    row = app.game.current.row
    press(pygame.K_DOWN)
    step(app, view)
    assert app.game.current.row == row + 1
    assert app.game.score == 1                               # soft drop pays 1/cell


def test_the_loop_hard_drops_on_space(app, view):
    step(app, view)
    press(pygame.K_SPACE)
    step(app, view)
    assert app.game.score > 0                                # 2 points per cell
    assert any(cell != 0 for row in app.game.board.as_list() for cell in row)


def test_the_loop_restarts_on_r(app, view):
    step(app, view)
    press(pygame.K_SPACE)
    step(app, view)
    assert app.game.score > 0

    press(pygame.K_r)
    step(app, view)
    assert app.game.score == 0
    assert all(cell == 0 for row in app.game.board.as_list() for cell in row)


def test_the_loop_pauses_on_p(app, view):
    step(app, view)
    press(pygame.K_p)
    step(app, view)
    assert app.paused

    row = app.game.current.row
    press(pygame.K_DOWN)
    step(app, view)
    assert app.game.current.row == row                       # frozen while paused
    assert app.update(5_000) == 0                            # and gravity is frozen

    press(pygame.K_p)
    step(app, view)
    assert not app.paused


def test_the_loop_quits_on_escape(app, view):
    step(app, view)
    press(pygame.K_ESCAPE)
    assert step(app, view, frames=5) == 0                     # left before frame 1


def test_the_loop_quits_on_the_window_close_button(app, view):
    step(app, view)
    pygame.event.post(pygame.event.Event(pygame.QUIT))
    assert step(app, view, frames=5) == 0


def test_the_ghost_toggle_only_changes_the_view(app, view):
    step(app, view)
    board = app.game.board.as_list()
    press(pygame.K_g)
    step(app, view)
    assert view.ghost is False
    assert app.game.board.as_list() == board                  # game untouched


def test_a_piece_that_is_left_alone_still_falls(app, view):
    # Level 20 => one cell every 60ms, so a handful of frames is enough to see
    # the loop feeding elapsed time into gravity (at level 1 it would take ~1s).
    app.game.level = 20
    step(app, view)
    start = app.game.current.row
    for _ in range(20):
        step(app, view)
    assert app.game.current.row > start or app.game_over
