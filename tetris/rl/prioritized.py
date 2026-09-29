"""Prioritised replay: sample the transitions the network is worst at.

Uniform sampling draws old and new, rare and common, informative and redundant
with the same probability. In this project that wastes most of the budget: the
agent makes ~86,000 decisions in an arm, each episode is ~30 transitions of
near-identical stacking, and a line clear -- the only event that carries the
signal we care about -- is 0.5% of them. Sampling uniformly, the network sees
about 430 line clears' worth of evidence over an entire run.

Prioritised experience replay (Schaul et al., 2016) draws a transition in
proportion to its last TD error, `p_i = (|delta_i| + epsilon)^alpha`: the
transitions whose value the network currently gets most wrong are the ones it
trains on most. Two details make that safe rather than a good way to overfit:

* **Importance sampling.** Drawing from a biased distribution biases the update,
  so each sample's loss is scaled by `w_i = (N * P(i))^(-beta)`, normalised by
  the batch maximum (which only rescales the gradient, not its direction). With
  `beta` annealed from `beta0` to 1.0 the bias is corrected by the end of the run
  while the early updates stay cheap.
* **A floor.** `epsilon` keeps every transition's probability non-zero, so a
  transition that once looked unimportant can still come back.

The sum-tree holds the priorities, so sampling a batch is `O(batch * log N)`
rather than a full pass over the buffer: leaves are the priorities, every parent
is the sum of its children, and a stratified sample descends from the root.

This module is storage only -- the priorities come from the agent's own TD
errors on each batch (`PrioritizedAgent.train_step`), because only the update
knows what they were.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .encoding import ACTION_SIZE, STATE_SHAPE
from .replay import Batch


class PrioritizedReplayBuffer:
    """A ring buffer of transitions, sampled in proportion to their TD error.

    ::

        buffer = PrioritizedReplayBuffer(capacity=50_000, alpha=0.6, beta=0.4)
        buffer.add(state, action, reward, next_state, next_mask, done)   # priority = max so far
        batch, indices, weights = buffer.sample(64, rng=rng, beta=0.7)
        buffer.update_priorities(indices, td_errors)

    Storage is identical to `ReplayBuffer` (same arrays, same ring), so a
    transition means the same thing in both and an arm can swap one for the
    other. The only additions are the sum-tree, the priorities and the weights.
    """

    def __init__(
        self,
        capacity: int = 50_000,
        state_shape: Sequence[int] = STATE_SHAPE,
        action_size: int = ACTION_SIZE,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be 1 or more, got {capacity}")
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha!r}")
        if not 0.0 <= float(beta) <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta!r}")
        if float(epsilon) <= 0:
            raise ValueError(f"epsilon must be > 0, got {epsilon!r}")

        self.capacity = int(capacity)
        self.state_shape = tuple(state_shape)
        self.action_size = int(action_size)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.epsilon = float(epsilon)

        self.states = np.zeros((self.capacity, *self.state_shape), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, *self.state_shape), dtype=np.float32)
        self.next_masks = np.zeros((self.capacity, self.action_size), dtype=bool)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self._position = 0
        self._size = 0

        # The sum-tree: a perfect binary tree with `_leaf_count` leaves (the next
        # power of two above capacity), where node i's value is the sum of its
        # children. Leaves 0.._leaf_count-1 are the slots' priorities.
        self._leaf_count = 1
        while self._leaf_count < self.capacity:
            self._leaf_count *= 2
        self._tree = np.zeros(2 * self._leaf_count, dtype=np.float64)

        #: The priority a brand-new transition gets: the largest seen so far, so
        #: nothing is starved before it has had a chance to be wrong.
        self.max_priority = 1.0

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
        """Store one transition with the current maximum priority."""
        position = self._position
        self.states[position] = state
        self.actions[position] = int(action)
        self.rewards[position] = float(reward)
        self.next_states[position] = next_state
        self.next_masks[position] = np.asarray(next_mask, dtype=bool)
        self.dones[position] = 1.0 if done else 0.0

        self._set_priority(position, self.max_priority)

        self._position = (position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    # ---------------------------------------------------------------- sampling

    def sample(
        self,
        batch_size: int,
        rng: Optional[np.random.Generator] = None,
        beta: Optional[float] = None,
    ) -> Tuple[Batch, np.ndarray, np.ndarray]:
        """`(batch, slot indices, importance weights)` drawn by priority.

        Stratified: the priority range is split into `batch_size` equal segments
        and one priority is drawn uniformly from each, which cannot miss the top
        of the range the way `batch_size` independent draws can. Weights are
        normalised by the batch maximum.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be 1 or more, got {batch_size}")
        if self._size == 0:
            raise ValueError("the buffer is empty: add transitions before sampling")
        generator = np.random.default_rng() if rng is None else rng
        beta = self.beta if beta is None else float(beta)

        total = float(self._tree[1])
        if total <= 0:
            raise RuntimeError("every priority is zero: the sum-tree is broken")

        segment = total / float(batch_size)
        boundaries = np.arange(batch_size, dtype=np.float64) * segment
        values = boundaries + generator.random(batch_size) * segment
        slots = np.array([self._find_slot(value) for value in values], dtype=np.int64)

        priorities = np.array([self._priority(slot) for slot in slots], dtype=np.float64)
        probabilities = priorities / total
        weights = np.power(self._size * probabilities, -beta)
        weights = weights / weights.max()
        return self._batch_at(slots), slots, weights.astype(np.float32)

    def update_priorities(self, slots: np.ndarray, errors: np.ndarray) -> None:
        """Set the priority of the sampled slots from their TD errors."""
        errors = np.asarray(errors, dtype=np.float64).reshape(-1)
        slots = np.asarray(slots, dtype=np.int64).reshape(-1)
        if len(errors) != len(slots):
            raise ValueError(
                f"got {len(errors)} error(s) for {len(slots)} slot(s): they must pair up"
            )
        for slot, error in zip(slots, errors):
            priority = (abs(float(error)) + self.epsilon) ** self.alpha
            self._set_priority(int(slot), priority)
            self.max_priority = max(self.max_priority, priority)

    def _batch_at(self, slots: np.ndarray) -> Batch:
        return Batch(
            states=self.states[slots].copy(),
            actions=self.actions[slots].copy(),
            rewards=self.rewards[slots].copy(),
            next_states=self.next_states[slots].copy(),
            next_masks=self.next_masks[slots].copy(),
            dones=self.dones[slots].copy(),
        )

    # --------------------------------------------------------------- sum-tree

    def _set_priority(self, slot: int, priority: float) -> None:
        """Write a leaf's priority and refresh the sums above it."""
        node = self._leaf_count + int(slot)
        self._tree[node] = float(priority)
        node //= 2
        while node >= 1:
            self._tree[node] = self._tree[2 * node] + self._tree[2 * node + 1]
            node //= 2

    def _priority(self, slot: int) -> float:
        return float(self._tree[self._leaf_count + int(slot)])

    def _find_slot(self, value: float) -> int:
        """The slot whose leaf covers `value` in the cumulative sum."""
        node = 1
        while node < self._leaf_count:
            left = 2 * node
            if value < self._tree[left]:
                node = left
            else:
                value -= self._tree[left]
                node = left + 1
        return node - self._leaf_count

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
        """The stored slots, oldest first -- what `sample` can ever return.

        Same meaning as on `ReplayBuffer`, so a test or a debug tool can walk
        either buffer the same way.
        """
        if not self.full:
            return np.arange(self._size)
        start = self._position
        return (start + np.arange(self.capacity)) % self.capacity

    @property
    def total_priority(self) -> float:
        """The sum of every stored priority (the root of the tree)."""
        return float(self._tree[1])

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return (
            f"<PrioritizedReplayBuffer {self._size}/{self.capacity} "
            f"alpha={self.alpha} beta={self.beta}>"
        )


__all__ = ["PrioritizedReplayBuffer"]
