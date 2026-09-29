"""Tests for the colored terminal view (tetris/render.py).

The renderer is the only module that touches "pixels", so these tests check two
things: that the color/bevel math lands on the right canvas pixels, and that the
assembled frame has the geometry a terminal needs (rectangular lines, one line
of output per board row, no stray escapes when colors are off).
"""

import random

from tetris import render
from tetris.actions import Action
from tetris.board import HEIGHT, WIDTH
from tetris.game import Game
from tetris.pieces import PieceType

ESC = "\x1b"


def fresh_game(seed: int = 1234) -> Game:
    return Game(random.Random(seed))


def ghost_pixels(canvas, game) -> int:
    """How many canvas pixels hold the landing-preview tint."""
    tint = render.mix(
        render.PLAYFIELD, render.PIECE_COLORS[game.current.type], render.GHOST_TINT
    )
    return sum(
        canvas.get(x, y) == tint
        for y in range(HEIGHT * render.PX_H)
        for x in range(WIDTH * render.PX_W)
    )


# ------------------------------------------------------------------- palette


def test_every_piece_type_has_a_color():
    assert set(render.PIECE_COLORS) == set(PieceType)
    assert len(set(render.PIECE_COLORS.values())) == len(PieceType)  # all distinct


def test_piece_color_accepts_raw_ids():
    assert render.piece_color(int(PieceType.I)) == render.PIECE_COLORS[PieceType.I]
    assert render.piece_color(PieceType.I) == render.PIECE_COLORS[PieceType.I]


def test_lighten_and_darken_move_toward_white_and_black():
    color = render.PIECE_COLORS[PieceType.I]
    assert render.lighten(color) > color               # brighter
    assert render.darken(color) < color                # darker
    assert render.lighten(color, 1.0) == (255, 255, 255)
    assert render.darken(color, 1.0) == (0, 0, 0)


def test_mix_clamps_out_of_range_amounts():
    assert render.mix((0, 0, 0), (100, 100, 100), -1) == (0, 0, 0)
    assert render.mix((0, 0, 0), (100, 100, 100), 5) == (100, 100, 100)


# -------------------------------------------------------------------- canvas


def test_two_pixels_become_one_half_block_character():
    canvas = render.Canvas(1, 2)
    canvas.set(0, 0, (255, 0, 0))
    canvas.set(0, 1, (0, 0, 255))
    line = canvas.to_lines(color=True)[0]
    assert line == (
        f"{ESC}[0m{ESC}[38;2;255;0;0m{ESC}[48;2;0;0;255m" f"{render.UPPER_HALF}{ESC}[0m"
    )


def test_matching_pixels_become_a_full_block():
    canvas = render.Canvas(1, 2)
    canvas.fill(0, 0, 1, 2, (10, 200, 30))
    line = canvas.to_lines(color=True)[0]
    assert render.FULL_BLOCK in line
    assert "38;2;10;200;30" in line and "48;2;10;200;30" in line


def test_half_block_is_used_when_one_pixel_is_the_background():
    canvas = render.Canvas(1, 2)
    canvas.set(0, 0, (255, 0, 0))                       # top pixel only
    assert render.UPPER_HALF in canvas.to_lines(color=True)[0]

    canvas.set(0, 0, render.PLAYFIELD)                  # bottom pixel only
    canvas.set(0, 1, (255, 0, 0))
    assert render.LOWER_HALF in canvas.to_lines(color=True)[0]


def test_canvas_geometry():
    canvas = render.Canvas(6, 4)
    assert len(canvas.to_lines()) == 2                  # 2 vertical pixels/char
    assert all(len(render.strip_ansi(line)) == 6 for line in canvas.to_lines())


def test_canvas_pixels_outside_are_ignored_and_read_as_background():
    canvas = render.Canvas(2, 2)
    canvas.set(99, 99, (255, 0, 0))                     # no crash, no effect
    assert canvas.get(-1, 0) == render.PLAYFIELD
    assert canvas.get(0, 0) == render.PLAYFIELD


def test_ascii_fallback_has_no_escapes():
    canvas = render.Canvas(2, 2)
    canvas.fill(0, 0, 1, 2, (255, 0, 0))
    line = canvas.to_lines(color=False)[0]
    assert ESC not in line
    assert line == "# "


# --------------------------------------------------------- board / playfield


def test_playfield_is_framed_and_rectangular():
    renderer = render.Renderer(color=True)
    lines = renderer.board_lines(fresh_game())
    assert len(lines) == HEIGHT + 2                     # one line per board row
    widths = {render.visible_len(line) for line in lines}
    assert widths == {WIDTH * render.CELL_CHARS + 2 + len(renderer.indent)}


def test_board_cells_are_two_characters_wide():
    assert render.CELL_CHARS == 2
    assert render.CELL_LINES == 1                       # one terminal line per cell
    assert render.Renderer().playfield_width == WIDTH * 2


def test_locked_cell_is_drawn_with_its_own_color_and_bevel():
    game = fresh_game()
    game.current = None                                 # no falling piece
    game.board.place([(5, 3)], int(PieceType.T))
    canvas = render.Renderer(color=True).board_canvas(game)

    t_color = render.PIECE_COLORS[PieceType.T]
    x, y = 3 * render.PX_W, 5 * render.PX_H
    assert canvas.get(x, y) == render.lighten(t_color, render.BEVEL_LIGHT)
    assert canvas.get(x + 1, y) == render.lighten(t_color, render.BEVEL_MID_LIGHT)
    assert canvas.get(x, y + 1) == render.darken(t_color, render.BEVEL_DARK)
    assert canvas.get(x + 1, y + 1) == render.darken(t_color, render.BEVEL_MID_DARK)


def test_empty_cells_stay_background():
    game = fresh_game()
    game.current = None
    canvas = render.Renderer(color=True).board_canvas(game)
    assert all(
        canvas.get(x, y) == render.PLAYFIELD
        for y in range(HEIGHT * render.PX_H)
        for x in range(WIDTH * render.PX_W)
    )


def test_active_piece_is_drawn():
    game = fresh_game()
    canvas = render.Renderer(color=True).board_canvas(game)
    color = render.PIECE_COLORS[game.current.type]
    for row, col in game.current.absolute_cells():
        assert canvas.get(col * render.PX_W, row * render.PX_H) == render.lighten(
            color, render.BEVEL_LIGHT
        )


def test_active_piece_above_the_board_is_not_drawn():
    game = fresh_game()
    game.current.row = -3                               # entirely hidden
    canvas = render.Renderer(color=True).board_canvas(game)
    top_row = [canvas.get(x, 0) for x in range(WIDTH * render.PX_W)]
    assert all(pixel == render.PLAYFIELD for pixel in top_row)


def test_board_without_a_current_piece_still_renders():
    game = fresh_game()
    game.current = None
    assert render.Renderer(color=True).board_lines(game)


# --------------------------------------------------------------------- ghost


def test_ghost_previews_the_landing_position():
    game = fresh_game()
    renderer = render.Renderer(color=True, ghost=True)
    canvas = renderer.board_canvas(game)
    tint = render.mix(
        render.PLAYFIELD, render.PIECE_COLORS[game.current.type], render.GHOST_TINT
    )

    # One dot per cell (GHOST_PATTERN), for all four cells of the piece.
    assert ghost_pixels(canvas, game) == 4 * len(render.GHOST_PATTERN)

    # On an empty board the piece lands on the floor: its bottom cell is on the
    # last row, so every cell of the preview sits `floor_offset` rows lower.
    bottom = max(row for row, _ in game.current.absolute_cells())
    floor_offset = HEIGHT - 1 - bottom
    for row, col in game.current.absolute_cells():
        assert canvas.get(col * render.PX_W, (row + floor_offset) * render.PX_H) == tint


def test_ghost_can_be_turned_off():
    game = fresh_game()
    renderer = render.Renderer(color=True, ghost=False)
    assert ghost_pixels(renderer.board_canvas(game), game) == 0


def test_solid_blocks_win_over_the_ghost():
    game = fresh_game()
    # A full wall in the middle of the well: the piece cannot fall, so its
    # landing cells are its own spawn cells and the ghost is fully painted over.
    game.board.place([(row, 5) for row in range(HEIGHT)], int(PieceType.Z))
    renderer = render.Renderer(color=True, ghost=True)
    canvas = renderer.board_canvas(game)
    assert ghost_pixels(canvas, game) == 0

    # The rows the piece does not cover still show the wall's own colors.
    covered = {row for row, _ in game.current.absolute_cells()}
    wall = render.PIECE_COLORS[PieceType.Z]
    for row in range(HEIGHT):
        if row in covered:
            continue
        assert canvas.get(5 * render.PX_W, row * render.PX_H) == render.lighten(
            wall, render.BEVEL_LIGHT
        )


# ------------------------------------------------------------------ grid/misc


def test_grid_styles_fill_the_well_differently():
    game = fresh_game()
    game.current = None
    plain = render.Renderer(color=True, grid="none").board_canvas(game)
    dotted = render.Renderer(color=True, grid="dots").board_canvas(game)
    ruled = render.Renderer(color=True, grid="lines").board_canvas(game)

    assert all(
        plain.get(x, y) == render.PLAYFIELD
        for y in range(HEIGHT * render.PX_H)
        for x in range(WIDTH * render.PX_W)
    )
    assert dotted.get(0, 0) == render.GRID_DOT
    assert dotted.get(1, 0) == render.PLAYFIELD          # only the cell corner
    assert ruled.get(0, 0) == render.GRID_DOT
    assert ruled.get(0, 1) == render.GRID_DOT            # both pixels -> a rule


def test_unknown_grid_style_is_rejected():
    try:
        render.Renderer(grid="sparkles")
    except ValueError as exc:
        assert "grid" in str(exc)
    else:
        raise AssertionError("expected a ValueError")


# --------------------------------------------------------------------- panel


def test_panel_lists_score_lines_level_and_the_next_pieces():
    game = fresh_game()
    game.score, game.lines_cleared, game.level = 4200, 27, 3
    panel = render.Renderer(color=True).panel_lines(game)
    text = "\n".join(render.strip_ansi(line) for line in panel)
    assert "SCORE" in text and "4200" in text
    assert "LINES" in text and "27" in text
    assert "LEVEL" in text and "3" in text
    assert "NEXT" in text


def test_panel_shows_one_preview_block_per_upcoming_piece():
    game = fresh_game()
    panel = render.Renderer(color=True).panel_lines(game)
    assert len(panel) == 7 + render.PREVIEW_COUNT * 4    # header + one box each

    for index, piece_type in enumerate(game.queue[: render.PREVIEW_COUNT]):
        box = panel[7 + 4 * index: 11 + 4 * index]
        text = "\n".join(render.strip_ansi(line) for line in box)
        assert render.FULL_BLOCK in text                 # the mini drawing itself
        assert PieceType(piece_type).name in text

        # ...and it is drawn in that piece's own color.
        light = render.lighten(render.PIECE_COLORS[PieceType(piece_type)])
        assert f"38;2;{light[0]};{light[1]};{light[2]}m" in "\n".join(box)


def test_panel_can_be_disabled():
    renderer = render.Renderer(color=True, panel=False)
    lines = renderer.lines(fresh_game())
    assert len(lines) == HEIGHT + 2                      # playfield only


# --------------------------------------------------------------------- frame


def test_frame_lines_are_square():
    game = fresh_game()
    lines = render.Renderer(color=True).lines(game)
    assert len({render.visible_len(line) for line in lines}) == 1


def test_frame_is_colored_when_asked_and_plain_otherwise():
    game = fresh_game()
    colored = render.Renderer(color=True).frame(game)
    plain = render.Renderer(color=False).frame(game)
    assert f"{ESC}[38;2;" in colored
    assert ESC not in plain
    assert render.strip_ansi(colored) != colored


def test_plain_frame_keeps_the_same_geometry():
    game = fresh_game()
    colored = render.Renderer(color=True).frame(game)
    plain = render.Renderer(color=False).frame(game)
    assert len(colored.splitlines()) == len(plain.splitlines())


def test_frame_shows_a_banner_and_controls_hint():
    game = fresh_game()
    text = render.strip_ansi(render.Renderer(color=True).frame(game))
    assert "move" in text and "quit" in text             # controls hint
    assert "GAME OVER" not in text

    game.game_over = True
    assert "GAME OVER" in render.strip_ansi(render.Renderer(color=True).frame(game))
    assert "PAUSED" in render.strip_ansi(
        render.Renderer(color=True).frame(game, message="PAUSED")
    )


def test_convenience_helpers_render_a_game():
    game = fresh_game()
    assert render.strip_ansi(render.frame(game, color=True))
    assert render.strip_ansi(render.board(game, color=True))
    assert ESC not in render.frame(game, color=False)


def test_strip_ansi_and_visible_len_agree():
    text = f"{ESC}[38;2;1;2;3mhello{ESC}[0m"
    assert render.strip_ansi(text) == "hello"
    assert render.visible_len(text) == 5


def test_locked_piece_keeps_its_color_after_a_hard_drop():
    game = fresh_game()
    piece_type = game.current.type
    game.step(Action.DROP)                               # lock it on the floor
    canvas = render.Renderer(color=True, ghost=False).board_canvas(game)
    expected = render.lighten(render.PIECE_COLORS[piece_type], render.BEVEL_LIGHT)
    assert any(
        canvas.get(x, y) == expected
        for y in range(HEIGHT * render.PX_H)
        for x in range(WIDTH * render.PX_W)
    )
