"""Drive the engine from a whole-placement policy, one action per frame.

`play.py --mode viz` pairs this controller with `PygameView`, so an AI plays in
the same window a human plays in. The *agent* side of the contract is unchanged
-- `choose_action(state, valid_actions) -> Placement`, exactly the call the
evaluation harness makes -- but the frontend speaks `Action` (`LEFT`,
`ROTATE_CW`, `DROP`, ...), and those are not the same language:

    the policy              picks a whole placement   (rotation, column)
    the engine              takes one action per frame (rotate / slide / drop)

So this controller holds the plan for the piece in flight and emits the moves
that carry it out: rotate clockwise until the piece is in the planned
orientation, slide until its leftmost cell is over the planned column, then hard
drop. The window therefore shows *how* the policy plays instead of a piece
teleporting to its destination -- which is the point of a visual demo.

Two things it deliberately does not do:

* **It never calls `TetrisEnv.step()`.** The app owns the game and applies engine
  actions; the controller only reads that game through the environment's own
  `get_state()` / `get_valid_actions()` (`env.game = game` before each read), so
  there is one source of truth for what is legal and no second simulation to
  drift away from the one on screen.
* **It never assumes a move worked.** A rotation or a slide can be blocked by the
  stack, and the engine then ignores it (the O piece never rotates at all). A
  repeated no-op request is detected and the piece is dropped where it stands, so
  a blocked plan degrades into a legal move instead of an infinite loop.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from tetris.actions import Action
from tetris.env import Placement, RewardConfig, StepOutcome, TetrisEnv
from tetris.metrics import board_height, count_holes, occupancy

# A rotation or slide that changed nothing twice in a row is blocked (spawn height
# is the top of the well, so a tall stack is what does it). After that many
# no-ops the piece is dropped where it stands rather than retried forever.
STUCK_LIMIT = 2

# Hard per-piece budget on rotate/slide actions. A correct plan needs at most 3
# rotations and 8 slides; anything past that is a bug or a blocked plan, and the
# demo must never hang on either.
MOVE_LIMIT = 12

#: How a viewer may pace this controller: `(label, think_ms, ticks per frame)`.
#: The first three levels slow the *thinking*; the last also lets a viewer take
#: several actions inside one painted frame, so the policy can outrun the frame
#: rate instead of being capped by it. It lives here, next to `think_ms`, because
#: it is a property of how the controller is driven -- and this module stays free
#: of pygame and torch, so a launcher can import the table without a display.
SPEED_LEVELS = (
    ("Slow", 400.0, 1),
    ("Normal", 120.0, 1),
    ("Fast", 20.0, 1),
    ("Very fast", 0.0, 6),
)
DEFAULT_SPEED = 1            # "Normal"


def speed_count() -> int:
    """How many speeds a viewer cycles through (the bound for a `+`/`-` key)."""
    return len(SPEED_LEVELS)


def speed_detail(index: int) -> str:
    """The pacing of one speed level, in words, for a panel or a log line."""
    _label, think_ms, ticks = SPEED_LEVELS[index]
    moves = "as fast as it can" if think_ms <= 0 else f"{think_ms:g} ms/move"
    if ticks > 1:
        moves += f", {ticks} moves/frame"
    return moves

# The app commands an AI game still wants from the keyboard. These are the same
# four `tetris.pygame_view.KEY_TO_COMMAND` maps; they are repeated here so this
# module can be imported without pygame (the viewer keeps its own table).
COMMAND_KEYS = ("restart", "pause", "ghost", "quit")


class BestCheckpoint(NamedTuple):
    """The best checkpoint found on disk, and what it scored."""

    path: str
    score_mean: float
    episode: int
    source: str

    def __str__(self) -> str:
        return (f"{self.path} (score {self.score_mean:.1f} at episode "
                f"{self.episode}, from {self.source})")


def best_records(root: str = "experiments/rl") -> "List[BestCheckpoint]":
    """Every campaign record under `root`, highest score first.

    Each campaign keeps the best model per seed next to a `best.json` that says
    what it scored, so "the trained agent" is a lookup rather than a path someone
    has to remember. The file *name* is resolved against the directory it was
    found in (a `best.json` also stores the absolute path from the machine that
    trained it, which stops being true as soon as the project moves).
    """
    import json
    import pathlib

    found: List[BestCheckpoint] = []
    for record_path in sorted(pathlib.Path(root).glob("*/seed*/best/best.json")):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):        # a partially written or hand-edited
            continue                          # record must not break the demo
        entry = record.get("best") or {}
        name = entry.get("checkpoint")
        if not name or "score_mean" not in entry:
            continue
        found.append(
            BestCheckpoint(
                path=str(record_path.parent / name),
                score_mean=float(entry["score_mean"]),
                episode=int(entry.get("episode", -1)),
                source=f"{record_path.parent.parent.parent.name}/"
                       f"{record_path.parent.parent.name}",
            )
        )
    return sorted(found, key=lambda item: item.score_mean, reverse=True)


def best_checkpoint(root: str = "experiments/rl") -> BestCheckpoint:
    """The highest-scoring checkpoint under `root`, or raise if there is none."""
    found = best_records(root)
    if not found:
        raise FileNotFoundError(
            f"no trained checkpoint under {root!r}: run "
            "`python experiments/rl_experiments.py run --experiment G --parallel 5` "
            "first, or pass --agent heuristic/random for a non-learning opponent"
        )
    return found[0]


def build_agent(name: str, checkpoint: Optional[str] = None, lookahead: int = 0):
    """An agent by name. Heavy imports (torch) happen only for the DQN.

    Every option satisfies the same contract -- `choose_action(state,
    valid_actions) -> Placement` -- which is why the window, the evaluation
    harness and the campaign can all play the same opponent.
    """
    if name == "random":
        from tetris.agents import RandomAgent

        return RandomAgent()
    if name == "heuristic":
        from tetris.agents import HeuristicAgent

        return HeuristicAgent(lookahead_depth=lookahead)
    if name == "dqn":
        from tetris.rl.dqn import load_agent_for_evaluation

        path = checkpoint or best_checkpoint().path
        return load_agent_for_evaluation(path)
    raise ValueError(f"unknown agent {name!r}; expected random, heuristic or dqn")


def find_checkpoint(path: str, root: str = "experiments/rl") -> Optional[BestCheckpoint]:
    """The campaign record for a checkpoint the caller named itself, if there is one.

    `--model some/path.pt` should still be able to answer "which reward was this
    trained on?" -- the answer is recorded next to the model, in the campaign's
    `best.json`. Matching is by resolved path (and, failing that, by file name),
    so a checkpoint reached through a symlink or copied about still finds its
    record when the file itself is the one that was kept.
    """
    import json
    import pathlib

    wanted = pathlib.Path(path)
    try:
        target = str(wanted.resolve())
    except OSError:
        target = str(wanted)
    for record_path in sorted(pathlib.Path(root).glob("*/seed*/best/best.json")):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        entry = record.get("best") or {}
        name = entry.get("checkpoint")
        if not name or "score_mean" not in entry:
            continue
        candidate = record_path.parent / name
        try:
            matches = str(candidate.resolve()) == target
        except OSError:
            matches = False
        if not matches and candidate.name != wanted.name:
            continue
        return BestCheckpoint(
            path=str(candidate),
            score_mean=float(entry["score_mean"]),
            episode=int(entry.get("episode", -1)),
            source=f"{record_path.parent.parent.parent.name}/{record_path.parent.parent.name}",
        )
    return None


def campaign_of(path: str, root: str = "experiments/rl") -> Optional[str]:
    """Which campaign a checkpoint belongs to, given its path.

    The reward weights live with the campaign, not with the model file, so a
    viewer handed `--model <path>` needs to get from one to the other. A recorded
    `best.json` wins; failing that, the campaigns' own directory shape
    (`<campaign>/seed<N>/best/<file>.pt`) is read off the path, so a checkpoint
    still resolves after the tree moves house or the file is renamed.
    """
    found = find_checkpoint(path, root)
    if found is not None:
        return found.source.split("/")[0]
    import pathlib

    parts = pathlib.PurePath(str(path)).parts
    for index, part in enumerate(parts):
        if index >= 1 and part.startswith("seed") and part[4:].isdigit():
            return parts[index - 1]
    return None


def reward_config_for(
    source: str, root: str = "experiments/rl"
) -> "Tuple[RewardConfig, Optional[str]]":
    """The reward weights the campaign that produced a checkpoint was trained on.

    A viewer that reports a reward per move is reporting a *number*, and the
    number only means something next to the weights it came from: `per_piece=5`
    turns a board-filling move from a penalty into a small profit. The campaign
    records its own config next to its results, so read it rather than assume.
    Falls back to the defaults, and says so, when there is no record to read.
    """
    import json
    import pathlib
    from dataclasses import fields

    campaign = str(source).split("/")[0]
    path = pathlib.Path(root) / campaign / "campaign.json"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return RewardConfig(), None
    weights = (record.get("config") or {}).get("reward") or {}
    known = {field.name for field in fields(RewardConfig)}
    try:
        return RewardConfig(**{key: float(value) for key, value in weights.items()
                              if key in known}), str(path)
    except (TypeError, ValueError):
        return RewardConfig(), None


class MoveReport(NamedTuple):
    """What the last locked piece did, in the terms the reward is written in.

    This is the viewer's window into the same numbers training sees: the board is
    measured with the environment's own `count_holes` / `board_height` /
    `occupancy`, and the reward is the environment's own `RewardConfig` applied to
    the resulting `StepOutcome`. So the reward on screen is the reward that would
    have gone into the replay buffer, not a re-implementation of it that could
    quietly drift.
    """

    piece: str
    rotation: int
    column: int
    lines_cleared: int
    score_gained: int
    holes: int
    holes_created: int
    height: int
    height_added: int
    reward: float
    game_over: bool
    value: Optional[float] = None       # the agent's own evaluation of its choice
    value_label: str = "value"          # what that number is: Q-value, search score, ...
    epsilon: Optional[float] = None
    legal: Optional[int] = None


class PolicyController:
    """Emits the engine actions that carry out a policy's placements.

    Parameters
    ----------
    agent :
        Anything with `choose_action(state, valid_actions) -> Placement`; a
        `tetris.agents` baseline or a `tetris.rl.dqn.DQNAgent` both work.
    think_ms : float
        Minimum milliseconds between two actions. Above zero the piece visibly
        rotates and slides instead of arriving at its column in one frame; zero
        is as fast as the frame loop allows (which is what a test wants).
    env : TetrisEnv, optional
        Used only for its `get_state()` / `get_valid_actions()`, pointed at the
        game on screen. A fresh unseeded env is fine -- it is a reader here, not
        the game being played.
    reward_config : RewardConfig, optional
        Only used to *report* the reward of a piece (`MoveReport.reward`); the
        controller never feeds it anywhere. Pass the config the agent was trained
        with (the campaign records it) and the number on screen is comparable to
        the training curve.
    """

    def __init__(
        self,
        agent,
        think_ms: float = 0.0,
        env: Optional[TetrisEnv] = None,
        logger=None,
        reward_config: Optional[RewardConfig] = None,
    ) -> None:
        self.agent = agent
        self.think_ms = max(float(think_ms), 0.0)
        self.env = TetrisEnv() if env is None else env
        self.logger = logger
        self.reward_config = RewardConfig() if reward_config is None else reward_config

        self.decisions = 0        # placements asked of the policy
        self.pieces = 0           # pieces locked (hard drops issued)
        self.actions = 0          # engine actions issued
        self.abandoned = 0        # plans that could not be carried out
        self.last_placement: Optional[Placement] = None
        self.last_action: Optional[Action] = None
        self.last_report: Optional[MoveReport] = None

        self._elapsed_ms = 0.0
        self._plan: Optional[Placement] = None
        self._moves = 0
        self._stuck = 0
        self._request: Optional[tuple] = None
        self._pending: Optional[Dict[str, object]] = None

    # ------------------------------------------------------------------ naming

    @property
    def name(self) -> str:
        """What to call this controller in a log line."""
        return f"PolicyController({getattr(self.agent, 'name', type(self.agent).__name__)})"

    def __repr__(self) -> str:
        return (f"<{self.name} decisions={self.decisions} pieces={self.pieces} "
                f"think_ms={self.think_ms:g}>")

    def describe(self) -> str:
        """A one-line summary for the end of a run."""
        return (f"decisions {self.decisions}  pieces {self.pieces}  "
                f"actions {self.actions}  abandoned plans {self.abandoned}")

    # ------------------------------------------------------------------ input

    def feed(self, events: Sequence[object], dt_ms: float) -> List[str]:
        """Advance the pacing clock and translate keys into app commands.

        Only the four commands a spectator of an AI game can still want -- pause,
        restart, ghost, quit. pygame is imported here rather than at module level
        so everything else in this file (and its tests) runs with no display and
        no pygame installed; this method is only ever called from the graphical
        loop, where pygame is already loaded.
        """
        import pygame

        self._elapsed_ms += max(float(dt_ms), 0.0)
        commands: List[str] = []
        mapping = {
            pygame.K_p: "pause",
            pygame.K_r: "restart",
            pygame.K_g: "ghost",
            pygame.K_q: "quit",
            pygame.K_ESCAPE: "quit",
        }
        for event in events:
            if getattr(event, "type", None) == pygame.KEYDOWN:
                command = mapping.get(getattr(event, "key", None))
                if command is not None:
                    commands.append(command)
        return commands

    def clear(self) -> None:
        """Forget the piece in flight (the app calls this on restart)."""
        self._plan = None
        self._moves = 0
        self._stuck = 0
        self._request = None
        self._elapsed_ms = 0.0
        self._pending = None

    # ------------------------------------------------------------------ report

    @staticmethod
    def _measure(game) -> Dict[str, object]:
        """The board facts a reward is allowed to use, via the env's own metrics."""
        board = occupancy(game.board.as_list())
        return {
            "holes": count_holes(board),
            "height": board_height(board),
            "score": game.score,
            "lines": game.lines_cleared,
            "piece": game.current.type.name if game.current is not None else "-",
        }

    @staticmethod
    def _evaluation(agent) -> "tuple":
        """The agent's own evaluation of the choice it just made, if it has one.

        The two agent families publish different things -- a DQN keeps a
        `Decision` with `best_q`, the heuristic keeps a `SearchResult` with
        `best_score` -- so this reads whichever exists and says what it is. The
        panel can then label the number honestly instead of calling a rounded
        heuristic score a Q-value.
        """
        decision = getattr(agent, "last_decision", None)
        if decision is None:
            return None, "value", None, None
        value = getattr(decision, "best_q", None)
        label = "Q-value"
        if value is None:
            value = getattr(decision, "best_score", None)
            label = "search score"
        if value is None:
            return None, "value", None, None
        epsilon = getattr(decision, "epsilon", None)
        legal = getattr(decision, "legal", None)
        if legal is None:
            scores = getattr(decision, "scores", None)
            legal = len(scores) if scores is not None else None
        return (float(value), label,
                None if epsilon is None else float(epsilon),
                None if legal is None else int(legal))

    def _settle(self, game) -> None:
        """Score the piece that just locked, if one did.

        Called at the start of the next decision -- and before the game-over
        early return, so the piece that ended the game still reports its reward
        (including the terminal penalty).
        """
        if self._pending is None:
            return
        before, self._pending = self._pending, None
        after = self._measure(game)
        outcome = StepOutcome(
            lines_cleared=int(after["lines"]) - int(before["lines"]),
            score_gained=int(after["score"]) - int(before["score"]),
            holes_before=int(before["holes"]),
            holes_after=int(after["holes"]),
            height_before=int(before["height"]),
            height_after=int(after["height"]),
            game_over=bool(game.is_game_over()),
            pieces_placed=self.pieces,
        )
        value, label, epsilon, legal = self._evaluation(self.agent)
        # A policy may answer with a `Placement` or a plain (rotation, column).
        if self.last_placement is None:
            rotation, column = -1, -1
        else:
            rotation, column = (int(part) for part in self.last_placement)
        self.last_report = MoveReport(
            piece=str(before["piece"]),
            rotation=rotation,
            column=column,
            lines_cleared=outcome.lines_cleared,
            score_gained=outcome.score_gained,
            holes=outcome.holes_after,
            holes_created=outcome.holes_created,
            height=outcome.height_after,
            height_added=outcome.height_added,
            reward=float(self.reward_config(outcome)),
            game_over=outcome.game_over,
            value=value,
            value_label=label,
            epsilon=epsilon,
            legal=legal,
        )
        if self.logger is not None:
            self.logger(
                f"piece {self.last_report.piece} -> rotation {self.last_report.rotation} "
                f"column {self.last_report.column}: reward {self.last_report.reward:+.2f} "
                f"(lines {outcome.lines_cleared}, holes {outcome.holes_created:+d}, "
                f"height {outcome.height_added:+d})"
            )

    def snapshot(self) -> Dict[str, object]:
        """Everything the viewer's panel shows about the AI, in one call.

        Plain data on purpose: the view is the only thing that knows about fonts,
        and the numbers stay checkable without a display.
        """
        report = self.last_report
        return {
            "agent": getattr(self.agent, "name", type(self.agent).__name__),
            "think_ms": self.think_ms,
            "action": None if self.last_action is None else str(self.last_action.value),
            "placement": None if self.last_placement is None else tuple(self.last_placement),
            "decisions": self.decisions,
            "pieces": self.pieces,
            "actions": self.actions,
            "abandoned": self.abandoned,
            "report": None if report is None else report._asdict(),
        }

    # ------------------------------------------------------------------ decide

    def choose_action(self, game) -> Optional[Action]:
        """One engine action toward the policy's placement, or None to wait.

        `None` is a real answer: the app then only advances gravity, which is how
        the pacing between two actions is expressed.
        """
        if game is None:
            return None
        self._settle(game)                        # score the piece that locked
        if game.current is None or game.is_game_over():
            return None
        if self.think_ms > 0.0 and self._elapsed_ms < self.think_ms:
            return None
        self._elapsed_ms = 0.0
        self.env.game = game                      # read the game on screen
        piece = game.current

        if self._plan is None:
            self._plan = self._ask(game)
            self._moves = 0
            self._stuck = 0
            self._request = None
            if self._plan is None:               # no legal placement at all
                return self._drop()

        rotation, column = self._plan
        if self._moves >= MOVE_LIMIT:
            self.abandoned += 1
            if self.logger is not None:
                self.logger(f"{piece!r}: plan {tuple(self._plan)} out of moves, dropping")
            return self._drop()

        if piece.rotation != rotation:
            return self._emit("rotate", piece.rotation, Action.ROTATE_CW)

        leftmost = min(col for _, col in piece.absolute_cells())
        if leftmost != column:
            direction = Action.RIGHT if leftmost < column else Action.LEFT
            return self._emit(direction, leftmost, direction)

        return self._drop()

    def _ask(self, game) -> Optional[Placement]:
        """Ask the policy for the current piece (one decision per piece)."""
        valid = self.env.get_valid_actions()
        if not valid:
            return None
        state = self.env.get_state()
        placement = self.agent.choose_action(state, valid)
        self.decisions += 1
        self.last_placement = placement
        if self.logger is not None:
            self.logger(f"{game.current!r}: policy chose rotation={placement.rotation} "
                        f"column={placement.column} of {len(valid)} legal placements")
        return placement

    def _emit(self, request: str, observed: int, action: Action) -> Action:
        """Issue `action`, counting a repeated no-op as a blocked move."""
        if self._request == (request, observed):
            self._stuck += 1
            if self._stuck >= STUCK_LIMIT:
                self.abandoned += 1
                if self.logger is not None:
                    self.logger(f"{request} blocked at {observed}; dropping where it stands")
                return self._drop()
        else:
            self._request = (request, observed)
            self._stuck = 0
        self._moves += 1
        self.actions += 1
        self.last_action = action
        return action

    def _drop(self) -> Action:
        """Commit the piece; the next decision belongs to the next piece.

        The board is measured *before* the lock so the next call can score what
        this piece did -- the same before/after `StepOutcome` the environment
        builds for the reward.
        """
        game = self.env.game
        if game is not None and game.current is not None:
            self._pending = self._measure(game)
        self._plan = None
        self._request = None
        self._stuck = 0
        self._moves = 0
        self.pieces += 1
        self.actions += 1
        self.last_action = Action.DROP
        return Action.DROP


__all__ = [
    "BestCheckpoint",
    "best_records",
    "DEFAULT_SPEED",
    "MOVE_LIMIT",
    "MoveReport",
    "PolicyController",
    "SPEED_LEVELS",
    "STUCK_LIMIT",
    "best_checkpoint",
    "build_agent",
    "campaign_of",
    "find_checkpoint",
    "reward_config_for",
    "speed_count",
    "speed_detail",
]
