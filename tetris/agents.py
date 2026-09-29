"""Agents that pick Tetris placements.

An agent is anything shaped like this:

    action = agent.choose_action(state, valid_actions)

`state` is a `tetris.env.TetrisState`, `valid_actions` is the list from
`env.get_valid_actions()`, and the return value must be one of those placements.
Nothing else is required -- an agent may be completely stateless.

An agent that carries its own randomness (or, later, trained weights) may also
define `reset(seed=None)`; the evaluation harness in `tetris.evaluate` calls it
before every game so a run is reproducible from a single seed.

Two agents live here:

    RandomAgent     picks uniformly at random            (the floor to beat)
    HeuristicAgent  scores every placement, keeps the best one

Neither one imports the environment, pygame or the terminal: they read the
`TetrisState` they are handed, so they run anywhere the environment does.
"""

from __future__ import annotations

import random
from typing import List, Optional, Protocol, Tuple, runtime_checkable  # noqa: UP035

import numpy as np

from .env import Placement, TetrisState
from .heuristic import DEFAULT_BEAM, HeuristicWeights, SearchResult, WeightMap, search
from .pieces import PieceType


def upcoming_pieces(state: TetrisState, lookahead: int = 0) -> Tuple[PieceType, ...]:
    """The piece in play, then the next `lookahead` pieces that will arrive.

    `state.queue[0]` *is* the next piece (the same one `state.next_piece_type`
    flags), so the queue tells the agent what is coming before it is dealt --
    which is exactly what a lookahead search needs to know.
    """
    current = PieceType(int(np.argmax(state.current_piece_type)) + 1)
    following = tuple(PieceType(int(piece)) for piece in state.queue[:lookahead])
    return (current,) + following


@runtime_checkable
class Agent(Protocol):
    """The agent interface: given a state and its options, pick one option.

    `reset(seed=None)` is optional -- implement it when the agent has state of
    its own (randomness, weights, statistics) so repeated evaluations with the
    same seed give the same numbers.
    """

    def choose_action(
        self, state: TetrisState, valid_actions: List[Placement]
    ) -> Placement:
        ...


class RandomAgent:
    """Picks uniformly at random from the legal placements. The baseline.

    It is the floor every other agent has to beat, and the control that shows
    what "no skill at all" scores on a given set of piece sequences.

    ::

        agent = RandomAgent(seed=0)
        action = agent.choose_action(env.get_state(), env.get_valid_actions())
    """

    name = "random"

    def __init__(self, seed: Optional[int] = None, rng: Optional[random.Random] = None):
        self.seed = seed
        self.rng = rng if rng is not None else random.Random(seed)

    def reset(self, seed: Optional[int] = None) -> None:
        """Restart the random stream.

        Called by `evaluate_agent` before each game, so evaluating the *same*
        agent object twice with the same seed gives identical results instead
        of continuing wherever the previous run left off.
        """
        self.seed = seed
        self.rng = random.Random(seed)

    def choose_action(
        self, state: TetrisState, valid_actions: List[Placement]
    ) -> Placement:
        """One legal placement, drawn uniformly at random.

        `state` is accepted for interface compatibility and deliberately unused.
        """
        if not valid_actions:
            raise ValueError(
                "no valid actions: the game is over, call env.reset() before choosing"
            )
        return self.rng.choice(valid_actions)

    def __repr__(self) -> str:
        return f"<RandomAgent seed={self.seed!r}>"


class HeuristicAgent:
    """Plans every legal placement and keeps the board that looks best.

    The pipeline, per decision::

        state.board (occupancy)
              -> every valid placement
              -> simulate each one           (tetris.heuristic.simulate_placement)
              -> extract_features(...)       (aggregate height, holes, ...)
              -> weighted score              (HeuristicWeights)
              -> highest score wins

    With `lookahead_depth` above zero the agent does not stop at the board a move
    leaves: it also plays out what the *next* pieces could do on it, and judges the
    move by the position it leads to::

        HeuristicAgent()                     # depth 0: score this placement
        HeuristicAgent(lookahead_depth=1)    # ... plus the best next placement
        HeuristicAgent(lookahead_depth=2)    # ... plus the one after that

    It needs no randomness and no training: the whole policy is the six weights
    in `HeuristicWeights` plus how far it looks. Ties go to the first action in
    the order `get_valid_actions` returns (rotation, then column), so it is fully
    reproducible. `self.last_scores` keeps the score of every legal move of the
    last decision and `self.last_decision` the whole search behind them, which is
    what to look at when a move seems odd.

    ::

        agent = HeuristicAgent()                        # defaults
        agent = HeuristicAgent({"holes": -10.0})        # one tweaked weight
        agent = HeuristicAgent(lookahead_depth=1)       # think one piece ahead
        agent = HeuristicAgent(lookahead_depth=1, debug=True)   # and explain each move

    `beam` caps how many continuations the search keeps per node below the
    current move (see `tetris.heuristic.search`), which is what keeps depth 2
    affordable. `debug=True` prints the chosen move, its score and its
    runners-up after every decision.
    """

    name = "heuristic"

    def __init__(
        self,
        weights: WeightMap = None,
        lookahead_depth: int = 0,
        beam: int = DEFAULT_BEAM,
        debug: bool = False,
    ) -> None:
        if lookahead_depth < 0:
            raise ValueError(
                f"lookahead_depth must be 0 or more, got {lookahead_depth}"
            )
        if beam < 1:
            raise ValueError(f"beam must be at least 1, got {beam}")
        self.weights = HeuristicWeights.from_any(weights)
        self.lookahead_depth = int(lookahead_depth)
        self.beam = int(beam)
        self.debug = bool(debug)
        #: `(action, score)` for every legal move of the last decision, in
        #: `valid_actions` order. Scores are the lookahead totals; `-inf` marks a
        #: move that ends the game.
        self.last_scores: List[Tuple[Placement, float]] = []
        #: The `SearchResult` behind the last decision (None before the first one).
        self.last_decision: Optional[SearchResult] = None

    def choose_action(
        self, state: TetrisState, valid_actions: List[Placement]
    ) -> Placement:
        """The placement with the best score, looking `lookahead_depth` pieces ahead.

        Reads the board, the current piece and the next pieces straight out of
        `state` -- it never touches the environment, so nothing it does can
        affect the real game.
        """
        self.last_scores = []
        self.last_decision = None
        if not valid_actions:
            raise ValueError(
                "no valid actions: the game is over, call env.reset() before choosing"
            )

        result = search(
            state.board,
            upcoming_pieces(state, self.lookahead_depth),
            self.weights,
            depth=self.lookahead_depth,
            beam=self.beam,
            actions=valid_actions,
        )
        self.last_scores = list(result.scores)
        self.last_decision = result
        if self.debug:
            print(result)
        return result.action

    def __repr__(self) -> str:
        return (
            f"<HeuristicAgent depth={self.lookahead_depth} beam={self.beam} "
            f"{self.weights.as_dict()}>"
        )


#: name -> a factory taking `(seed, weights, lookahead, beam, checkpoint)`.
#: Agents that have no use for an argument ignore it, so one registry can serve
#: every caller: the quick CLI (`evaluate.py`), the config-driven runner
#: (`tetris.experiment` / `experiments/run.py`) and the training script.
#:
#: `dqn` is built lazily, on purpose: it needs PyTorch, which is an optional
#: dependency, and importing it here would make every user of this module pay
#: for it (including the tests that assert nothing imports pygame or torch).
AGENTS = {
    "random": lambda seed=None, weights=None, lookahead=0, beam=DEFAULT_BEAM,
    checkpoint=None: RandomAgent(seed=seed),
    "heuristic": lambda seed=None, weights=None, lookahead=0, beam=DEFAULT_BEAM,
    checkpoint=None: HeuristicAgent(weights, lookahead_depth=lookahead, beam=beam),
    "dqn": lambda seed=None, weights=None, lookahead=0, beam=DEFAULT_BEAM,
    checkpoint=None: _build_dqn(checkpoint),
}


def _build_dqn(checkpoint: Optional[str]):
    """A saved DQN agent in playing mode (greedy, no exploration).

    Imported inside the function so that `tetris.agents` stays importable -- and
    torch-free -- for everything that does not ask for this agent by name.
    """
    if not checkpoint:
        raise ValueError(
            "the 'dqn' agent needs a checkpoint to load: pass checkpoint='<path>.pt' "
            "(in a config file) or --checkpoint <path>.pt on the command line"
        )
    try:
        from .rl.dqn import load_agent_for_evaluation
    except ImportError as problem:            # pragma: no cover - needs no torch
        raise ImportError(
            "the 'dqn' agent needs PyTorch: pip install -r requirements-rl.txt"
        ) from problem
    return load_agent_for_evaluation(checkpoint)


def build_agent(
    name: str,
    seed: Optional[int] = None,
    weights: WeightMap = None,
    lookahead: int = 0,
    beam: int = DEFAULT_BEAM,
    checkpoint: Optional[str] = None,
):
    """An agent instance for `name`, configured as asked.

    Raises `KeyError` for an unknown name (the caller usually turns that into a
    usage message). `weights` are `HeuristicWeights` overrides -- a dict, or a
    `HeuristicWeights` -- and reach the agents that have weights; `checkpoint`
    is the saved `.pt` file the RL agent loads.
    """
    if name not in AGENTS:
        raise KeyError(f"unknown agent {name!r}; known: {sorted(AGENTS)}")
    return AGENTS[name](
        seed=seed, weights=weights, lookahead=lookahead, beam=beam, checkpoint=checkpoint
    )
