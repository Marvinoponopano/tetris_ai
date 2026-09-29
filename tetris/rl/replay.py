"""The replay buffer: a fixed-size ring of past transitions, sampled at random.

Why a buffer at all: consecutive steps in Tetris are wildly correlated (the same
board, one piece later), and training on them in order makes the network chase
whatever it just saw. Sampling random *past* transitions breaks that correlation
and lets one experience be learned from many times -- the "experience replay" of
DQN.

One transition is everything the Bellman update needs::

    (state, action, reward, next_state, next_mask, done)

`next_mask` is stored rather than recomputed: the target update has to ignore
illegal actions in the next state, and the mask depends only on the next piece
type, so storing 40 booleans is cheaper and clearer than rebuilding it per batch.

The store is a ring: once full, the oldest transition is overwritten, so memory
is bounded by `capacity` and no unbounded history accumulates during a long run.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

from .encoding import ACTION_SIZE, STATE_SHAPE


class Batch(NamedTuple):
    """One training batch: every array has `batch_size` leading entries."""

    states: np.ndarray        # (B, 3, 20, 10) float32
    actions: np.ndarray       # (B,)          int64
    rewards: np.ndarray       # (B,)          float32
    next_states: np.ndarray   # (B, 3, 20, 10) float32
    next_masks: np.ndarray    # (B, 40)       bool
    dones: np.ndarray         # (B,)          float32  1.0 only on a real game over

    def __len__(self) -> int:
        return len(self.actions)


class ReplayBuffer:
    """A ring buffer of transitions, with uniform random sampling.

    ::

        buffer = ReplayBuffer(capacity=50_000)
        buffer.add(state, action, reward, next_state, next_mask, done)
        batch = buffer.sample(64, rng=np.random.default_rng(0))

    Sampling is uniform and *with replacement*, the standard DQN choice. Pass a
    seeded `rng` for reproducible sampling (the training loop does).
    """

    def __init__(
        self,
        capacity: int = 50_000,
        state_shape=STATE_SHAPE,
        action_size: int = ACTION_SIZE,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be 1 or more, got {capacity}")
        self.capacity = int(capacity)
        self.state_shape = tuple(state_shape)
        self.action_size = int(action_size)

        self.states = np.zeros((self.capacity, *self.state_shape), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, *self.state_shape), dtype=np.float32)
        self.next_masks = np.zeros((self.capacity, self.action_size), dtype=bool)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self._position = 0
        self._size = 0

    # ------------------------------------------------------------------ filling

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_mask: np.ndarray,
        done: bool,
    ) -> None:
        """Store one transition, overwriting the oldest when full."""
        position = self._position
        self.states[position] = state
        self.actions[position] = int(action)
        self.rewards[position] = float(reward)
        self.next_states[position] = next_state
        self.next_masks[position] = np.asarray(next_mask, dtype=bool)
        self.dones[position] = 1.0 if done else 0.0

        self._position = (position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    # ----------------------------------------------------------------- sampling

    def sample(self, batch_size: int, rng: Optional[np.random.Generator] = None) -> Batch:
        """`batch_size` transitions drawn uniformly at random (with replacement)."""
        if batch_size < 1:
            raise ValueError(f"batch_size must be 1 or more, got {batch_size}")
        if self._size == 0:
            raise ValueError("the buffer is empty: add transitions before sampling")
        generator = np.random.default_rng() if rng is None else rng
        picked = generator.integers(0, self._size, size=int(batch_size))

        return Batch(
            states=self.states[picked].copy(),
            actions=self.actions[picked].copy(),
            rewards=self.rewards[picked].copy(),
            next_states=self.next_states[picked].copy(),
            next_masks=self.next_masks[picked].copy(),
            dones=self.dones[picked].copy(),
        )

    # ------------------------------------------------------------------- lookups

    @property
    def size(self) -> int:
        """How many transitions are stored (never more than `capacity`)."""
        return self._size

    @property
    def full(self) -> bool:
        return self._size >= self.capacity

    @property
    def positions(self) -> np.ndarray:
        """The stored slots, oldest first -- what `sample` can ever return."""
        if not self.full:
            return np.arange(self._size)
        start = self._position
        return (start + np.arange(self.capacity)) % self.capacity

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return f"<ReplayBuffer {self._size}/{self.capacity}>"


__all__ = ["Batch", "ReplayBuffer"]
