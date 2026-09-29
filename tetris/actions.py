"""Actions a player (or, later, an AI agent) can take each step.

Keeping actions in their own tiny module makes the public interface explicit:
the rest of the engine talks about `Action.LEFT`, `Action.DROP`, etc., and
never about raw keyboard codes or numeric enums baked into game logic. When we
add an AI later, its policy only needs to choose from these same values.
"""

from enum import Enum


class Action(Enum):
    """Every move available to the player."""

    LEFT = "left"              # shift the piece one column left
    RIGHT = "right"            # shift the piece one column right
    DOWN = "down"              # soft drop: one row down, lock if blocked
    DROP = "drop"              # hard drop: fall straight down and lock
    ROTATE_CW = "rotate_cw"    # rotate clockwise
    ROTATE_CCW = "rotate_ccw"  # rotate counter-clockwise
