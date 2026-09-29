"""Tests for tetromino definitions and rotation."""

from tetris.pieces import (
    Piece,
    PieceType,
    rotate_ccw,
    rotate_cw,
    rotated_cells,
)

ALL_TYPES = list(PieceType)


def test_every_piece_has_four_cells():
    for piece_type in ALL_TYPES:
        assert len(piece_type.base_cells) == 4, f"{piece_type.name} should have 4 cells"


def test_four_clockwise_rotations_return_to_start():
    for piece_type in ALL_TYPES:
        cells = piece_type.base_cells
        for _ in range(4):
            cells = rotate_cw(cells)
        assert set(cells) == set(piece_type.base_cells)


def test_clockwise_then_counter_clockwise_return_to_start():
    for piece_type in ALL_TYPES:
        cells = rotate_cw(piece_type.base_cells)
        cells = rotate_ccw(cells)
        assert set(cells) == set(piece_type.base_cells)


def test_o_piece_does_not_rotate():
    o = Piece(PieceType.O)
    assert o.rotate_cw().cells == o.cells
    assert o.rotate_ccw().cells == o.cells


def test_t_piece_rotates_clockwise_to_point_right():
    up = set(PieceType.T.base_cells)
    right = set(rotate_cw(PieceType.T.base_cells))
    # The left tip of the "up" T is at (0, -1); after rotating clockwise it
    # disappears and a bottom tip appears at (1, 0).
    assert (0, -1) in up
    assert (0, -1) not in right
    assert (1, 0) in right
    assert (1, 0) not in up


def test_absolute_cells_translate_by_position():
    piece = Piece(PieceType.I, row=5, col=3)
    # A horizontal I at (5, 3) covers relative cols -2..1 -> absolute cols 1..4.
    assert set(piece.absolute_cells()) == {(5, 1), (5, 2), (5, 3), (5, 4)}


def test_rotation_wraps_modulo_four():
    piece = Piece(PieceType.T)
    for _ in range(8):
        piece = piece.rotate_cw()
    assert piece.rotation == 0
    assert piece.cells == rotated_cells(PieceType.T, 0)
