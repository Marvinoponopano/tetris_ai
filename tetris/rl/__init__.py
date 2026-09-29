"""The reinforcement-learning parts: DQN over the placement action space.

This package is the *only* place PyTorch is imported. The engine, the
environment, the agents, the evaluation harness and the experiment framework all
work without it -- including the tests, which skip the modules here when torch
is absent. Install it with `pip install -r requirements-rl.txt`.

The pieces, in the order a reader should meet them:

    encoding.py   what the network sees: 3 binary planes, 40 action indices,
                  and the mask of legal placements for each piece
    network.py    the Q-network: 600 inputs -> 40 Q-values
    replay.py     the replay buffer: a ring of transitions, sampled at random
    config.py     TrainingConfig: the knobs, validated and JSON round-trippable
    dqn.py        DQNAgent: epsilon-greedy acting, the Bellman update, target
                  network, checkpoints
    training.py   the loop that ties it together, plus its log and checkpoints
"""

from __future__ import annotations

try:
    import torch  # noqa: F401
except ImportError as problem:      # pragma: no cover - exercised without torch
    raise ImportError(
        "the RL parts need PyTorch, which is an optional dependency: "
        "`pip install -r requirements-rl.txt` (or `pip install torch`)"
    ) from problem

from .config import DEVICES, TrainingConfig
from .dqn import DQNAgent, Decision, load_agent_for_evaluation
from .encoding import (
    ACTION_SIZE,
    STATE_SHAPE,
    action_index,
    encode_state,
    encode_states,
    index_for_placement,
    mask_for_piece,
    mask_for_state,
    placement_for_index,
    valid_indices,
)
from .network import DuelingQNetwork, QNetwork, build_dueling_network, build_network
from .nstep import NStepTransitions, Transition
from .prioritized import PrioritizedReplayBuffer
from .registry import load_agent, register
from .replay import Batch, ReplayBuffer
from .training import TrainingResult, train
# Imported for its side effect: every variant registers itself in
# `tetris.rl.registry` on import, which is what lets a checkpoint written by an
# arm be loaded by its name later (the comparison table, the diagnosis, the
# viewer). Last import on purpose -- it pulls in every other module above.
from . import variants

__all__ = [
    "ACTION_SIZE",
    "DEVICES",
    "STATE_SHAPE",
    "Batch",
    "DQNAgent",
    "Decision",
    "DuelingQNetwork",
    "NStepTransitions",
    "PrioritizedReplayBuffer",
    "QNetwork",
    "Transition",
    "ReplayBuffer",
    "TrainingConfig",
    "TrainingResult",
    "action_index",
    "build_dueling_network",
    "build_network",
    "encode_state",
    "encode_states",
    "index_for_placement",
    "load_agent",
    "load_agent_for_evaluation",
    "register",
    "mask_for_piece",
    "mask_for_state",
    "placement_for_index",
    "train",
    "valid_indices",
]
