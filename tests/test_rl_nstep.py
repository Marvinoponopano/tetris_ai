"""Tests for n-step returns (`tetris/rl/nstep.py`).

The load-bearing test is `test_a_terminal_inside_the_window_stops_the_return`:
an n-step return that walks past the end of an episode does not just make a bad
target, it makes one that can never be right (`done` is what tells the update
"there is no future here"), and it is the mistake this class exists to not make.

The second is `test_nothing_is_built_across_an_episode_boundary`: a window left
partly full at the piece cap would build a target out of the *next* episode's
rewards, which is wrong in a way no amount of training fixes.
"""

from __future__ import annotations

import numpy as np
import pytest

from tetris.rl.nstep import NStepTransitions, Transition


def state(fill: float) -> np.ndarray:
    return np.full((3, 20, 10), fill, dtype=np.float32)


def mask(legal: bool = True) -> np.ndarray:
    grid = np.zeros(40, dtype=bool)
    grid[:9] = legal
    return grid


def steps(window: NStepTransitions, rewards, done_at=None, base: int = 0):
    """Push one transition per reward; return everything the window emitted.

    `base` shifts the state values and the action ids, which is how a test says
    "this is a different episode" without pretending the values changed.
    """
    emitted = []
    for index, reward in enumerate(rewards):
        done = done_at is not None and done_at == index
        emitted += window.push(state(base + index), base + index, reward,
                               state(base + index + 1), mask(), done=done)
    return emitted


# ------------------------------------------------------------------ the return


def test_a_full_window_emits_the_discounted_return():
    """The first emission happens on the n-th push -- the first moment the
    three-step return is knowable -- and it belongs to the *oldest* transition."""
    window = NStepTransitions(n=3, gamma=0.5)
    emitted = steps(window, [1.0, 2.0, 4.0])
    assert len(emitted) == 1
    transition = emitted[0]
    # the window held rewards 1, 2, 4 (the first three pushes)
    assert transition.reward == pytest.approx(1.0 + 0.5 * 2.0 + 0.25 * 4.0)
    assert transition.action == 0                        # the *oldest* action
    assert transition.state[0, 0, 0] == 0.0              # ... and its state
    assert transition.next_state[0, 0, 0] == 3.0         # three steps later
    assert transition.done is False

    # the next push slides the window: the same three-step shape, one step on
    later = steps(window, [8.0], base=3)
    assert len(later) == 1
    assert later[0].reward == pytest.approx(2.0 + 0.5 * 4.0 + 0.25 * 8.0)
    assert later[0].action == 1
    assert later[0].next_state[0, 0, 0] == 4.0


def test_the_emitted_next_state_is_n_steps_later():
    window = NStepTransitions(n=2, gamma=1.0)
    emitted = steps(window, [0.0, 0.0])
    assert len(emitted) == 1
    assert emitted[0].state[0, 0, 0] == 0.0
    assert emitted[0].next_state[0, 0, 0] == 2.0
    assert emitted[0].next_mask.any()                    # the raw next mask, kept


def test_gamma_is_the_one_it_was_built_with():
    window = NStepTransitions(n=3, gamma=0.99)
    emitted = steps(window, [1.0, 1.0, 1.0])
    assert emitted[0].reward == pytest.approx(1.0 + 0.99 + 0.99 ** 2)

    flat = NStepTransitions(n=3, gamma=1.0)
    assert steps(flat, [1.0, 1.0, 1.0])[0].reward == pytest.approx(3.0)


# ------------------------------------------------------------- the episode end


def test_a_terminal_inside_the_window_stops_the_return():
    """The rewards after the game over must not be counted, and done is set."""
    window = NStepTransitions(n=3, gamma=1.0)
    emitted = steps(window, [1.0, 2.0, 3.0, 4.0], done_at=2)
    # push 3 carried done: everything waiting is emitted, and every emitted
    # transition stops at that terminal (1+2+3, then 2+3, then 3 -- never 4).
    assert len(emitted) == 3
    assert [t.reward for t in emitted] == pytest.approx([6.0, 5.0, 3.0])
    assert [t.done for t in emitted] == [True, True, True]
    # the oldest transition's target ends at the terminal state
    assert emitted[0].next_state[0, 0, 0] == 3.0
    assert emitted[2].reward == 3.0                       # the terminal itself
    assert len(window) == 1                               # the 4th push, waiting


def test_a_terminal_seen_before_the_window_is_full_still_emits():
    window = NStepTransitions(n=5, gamma=0.9)
    emitted = steps(window, [1.0, 1.0], done_at=1)
    assert len(emitted) == 2
    assert emitted[0].reward == pytest.approx(1.0 + 0.9)
    assert emitted[0].done and emitted[1].done
    assert len(window) == 0


def test_the_tail_of_an_episode_is_emitted_with_shorter_returns():
    window = NStepTransitions(n=3, gamma=1.0)
    steps(window, [1.0, 1.0])                            # two pushes, no emission
    assert len(window) == 2
    emitted = window.flush()
    assert [t.reward for t in emitted] == pytest.approx([2.0, 1.0])
    assert emitted[0].state[0, 0, 0] == 0.0              # oldest first
    assert emitted[1].state[0, 0, 0] == 1.0
    assert all(t.done is False for t in emitted)         # a truncation, not a game over
    assert len(window) == 0


def test_nothing_is_built_across_an_episode_boundary():
    """Flushing at the boundary is what keeps the next episode's rewards out."""
    window = NStepTransitions(n=3, gamma=1.0)
    steps(window, [1.0, 1.0])
    window.flush()                                       # the piece cap / episode end
    emitted = steps(window, [100.0, 100.0, 100.0], base=50)
    assert len(emitted) == 1
    assert emitted[0].reward == pytest.approx(300.0)     # only *these* three
    assert emitted[0].state[0, 0, 0] == 50.0             # the new episode's states
    assert emitted[0].next_state[0, 0, 0] == 53.0


def test_an_empty_flush_is_harmless():
    window = NStepTransitions(n=2, gamma=0.99)
    assert window.flush() == []
    assert len(window) == 0


# -------------------------------------------------------------- shape and setup


def test_n_of_one_is_the_plain_transition():
    window = NStepTransitions(n=1, gamma=0.99)
    emitted = steps(window, [5.0])
    assert len(emitted) == 1
    assert emitted[0].reward == 5.0
    assert emitted[0].done is False


def test_the_window_never_holds_more_than_n():
    window = NStepTransitions(n=3, gamma=1.0)
    for index in range(10):
        window.push(state(index), index, 1.0, state(index + 1), mask(), done=False)
    assert len(window) <= 3


def test_the_pending_transition_is_the_oldest_one():
    window = NStepTransitions(n=3, gamma=1.0)
    assert window.pending is None
    steps(window, [1.0, 2.0])
    assert window.pending.action == 0
    assert window.pending.reward == 1.0


def test_a_push_returns_a_list_of_transitions():
    window = NStepTransitions(n=1, gamma=1.0)
    emitted = steps(window, [1.0])
    assert isinstance(emitted[0], Transition)
    assert isinstance(emitted[0].next_mask, np.ndarray)


@pytest.mark.parametrize("bad", [0, -1])
def test_a_window_shorter_than_one_step_is_rejected(bad):
    with pytest.raises(ValueError, match="n must be 1 or more"):
        NStepTransitions(n=bad)


@pytest.mark.parametrize("bad", [0.0, -0.5, 1.5])
def test_a_discount_outside_the_range_is_rejected(bad):
    with pytest.raises(ValueError, match="gamma must be in"):
        NStepTransitions(n=3, gamma=bad)


def test_it_says_what_it_is_holding():
    window = NStepTransitions(n=3, gamma=0.99)
    steps(window, [1.0])
    assert "n=3" in repr(window) and "holding=1" in repr(window)
