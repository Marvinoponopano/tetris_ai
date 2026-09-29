"""n-step returns: let one transition see the next few steps' worth of reward.

The reference update is one step deep::

    Q(s, a)  ~=  r  +  gamma * max_a' Q_target(s', a')

and the reward `r` of a Tetris decision is mostly the small per-piece term. The
problem is what a *line clear* looks like from there: the four placements that
fill a row earn almost nothing, and then the fifth earns `per_line` plus the
height and hole relief. One-step bootstrapping has to propagate that back through
the value of `s'`, which it can only do once its own estimates are good -- and
those estimates come from the same sparse signal. This is the classic
delayed-reward case, and the classic cheap fix is to look further ahead in the
*return* instead of in the estimate::

    R^(n)_t = r_t + gamma r_{t+1} + ... + gamma^{n-1} r_{t+n-1}
    Q(s, a) ~= R^(n)_t + gamma^n * max_a' Q_target(s_{t+n}, a')

Four placements' worth of stack-building now sit inside the target of the move
that starts them, so the credit does not have to travel back through the network
one bootstrapped step at a time. Nothing about the network changes: this module
just rewrites the `reward` and `next_state` of the transitions that go into the
replay buffer.

The bookkeeping is where it is easy to be wrong, so this class is deliberately
dumb and total:

* a transition is emitted only once the window holds `n` of them, or as soon as
  the episode ends -- an episode's last `n-1` transitions still get learned
  from, with a shorter return and `done=1`;
* when a terminal transition is inside the window, the return stops there (the
  game is over: no reward after it) and `done` is set, so the terminal state's
  empty mask is never used to bootstrap;
* the states and masks stored are the *raw* ones from the environment, and the
  window is emptied at the episode boundary, so nothing leaks across episodes.

`gamma` is the run's own `gamma`, taken from the config by the agent: an n-step
return with a different discount than the bootstrap would be a different
objective, not a longer view of this one.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, NamedTuple, Optional

import numpy as np


class Transition(NamedTuple):
    """One transition, as the replay buffer stores it."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    next_mask: np.ndarray
    done: bool


class NStepTransitions:
    """Turns a stream of one-step transitions into n-step ones.

    ::

        window = NStepTransitions(n=3, gamma=0.99)
        for transition in window.push(state, action, reward, next_state, mask, done):
            buffer.add(*transition)

    `push` returns the transitions that are ready (usually none, sometimes one,
    and up to `n` when an episode ends). `flush` empties the window at an episode
    boundary; `push` already emits everything it has when `done` is set, so
    `flush` exists for the case where the caller stops mid-episode.
    """

    def __init__(self, n: int = 3, gamma: float = 0.99) -> None:
        if int(n) < 1:
            raise ValueError(f"n must be 1 or more, got {n!r}")
        if not 0.0 < float(gamma) <= 1.0:
            raise ValueError(f"gamma must be in (0, 1], got {gamma!r}")
        self.n = int(n)
        self.gamma = float(gamma)
        self._window: Deque[Transition] = deque(maxlen=self.n)

    # ------------------------------------------------------------------ filling

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_mask: np.ndarray,
        done: bool,
    ) -> List[Transition]:
        """Add one transition; return the n-step transitions that are now ready."""
        self._window.append(
            Transition(state, int(action), float(reward), next_state, next_mask, bool(done))
        )
        if done:
            # The episode is over: everything still in the window is emittable
            # now, with a shorter return, because no reward follows it.
            return self.flush()
        if len(self._window) < self.n:
            return []
        return [self._emit(self.n)]

    def flush(self) -> List[Transition]:
        """Emit what the window holds (shorter returns), oldest first, and empty it.

        Called at an episode boundary: the oldest transition is the one that has
        been waiting longest, so it is emitted first and its return covers
        whatever is left of the window.
        """
        ready: List[Transition] = []
        while self._window:
            ready.append(self._emit(len(self._window)))
            self._window.popleft()
        return ready

    def _emit(self, length: int) -> Transition:
        """The n-step transition for the oldest `length` entries of the window."""
        entries = list(self._window)[:length]
        reward = 0.0
        for index, entry in enumerate(entries):
            reward += (self.gamma ** index) * entry.reward
            if entry.done:
                # Terminal inside the window: the return stops at that reward
                # (nothing follows a finished game), and so does the offset.
                entries = entries[: index + 1]
                break
        last = entries[-1]
        return Transition(
            state=entries[0].state,
            action=entries[0].action,
            reward=float(reward),
            next_state=last.next_state,
            next_mask=last.next_mask,
            done=bool(last.done),
        )

    # ------------------------------------------------------------------- state

    def __len__(self) -> int:
        """How many transitions are waiting (never more than `n`)."""
        return len(self._window)

    @property
    def pending(self) -> Optional[Transition]:
        """The oldest waiting transition, or None (for tests and debugging)."""
        return self._window[0] if self._window else None

    def __repr__(self) -> str:
        return f"<NStepTransitions n={self.n} gamma={self.gamma} holding={len(self._window)}>"


__all__ = ["NStepTransitions", "Transition"]
