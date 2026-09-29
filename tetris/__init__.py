"""Tetris game engine -- a clean, AI-friendly environment.

Public API:
    Game       -- the environment (reset / step / get_state / ...)
    Action     -- the moves a player (or agent) can make
    GameState  -- an immutable snapshot of the game
    Board      -- the 10x20 grid and collision rules
    Piece, PieceType -- tetromino definitions and rotation
"""

from .actions import Action
from .board import Board
from .pieces import Piece, PieceType
from .state import GameState
from .game import Game

__all__ = ["Action", "Board", "Piece", "PieceType", "GameState", "Game"]
