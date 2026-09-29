"""Tests for prioritised replay (`tetris/rl/prioritized.py`).

The load-bearing test is `test_the_sum_tree_stays_exact`: sampling a batch and
finding a slot both descend a tree of *sums*, and if a parent ever stops being
the sum of its children, the buffer silently samples the wrong transitions -- no
error, just a different distribution than the one the weights correct for. So the
tree is checked against the priorities it should hold, slot by slot, after
overwrites and priority updates.

The second is `test_a_transition_with_a_big_error_comes_back_often`: the whole
point of the change, measured rather than argued.
"""

from __future__ import annotations

import numpy as np
import pytest

from tetris.rl.encoding import ACTION_SIZE, STATE_SHAPE
from tetris.rl.prioritized import PrioritizedReplayBuffer
from tetris.rl.replay import ReplayBuffer


def transition(index: int, rng: np.random.Generator, done: bool = False):
    state = rng.integers(0, 2, size=STATE_SHAPE).astype(np.float32)
    next_state = rng.integers(0, 2, size=STATE_SHAPE).astype(np.float32)
    mask = np.zeros(ACTION_SIZE, dtype=bool)
    mask[:9] = True
    return state, index % ACTION_SIZE, float(index), next_state, mask, done


def filled(capacity: int = 32, count: int = 16, alpha: float = 1.0, beta: float = 0.4):
    buffer = PrioritizedReplayBuffer(capacity=capacity, alpha=alpha, beta=beta)
    rng = np.random.default_rng(0)
    for index in range(count):
        buffer.add(*transition(index, rng))
    return buffer


# ------------------------------------------------------------- storage basics


def test_it_stores_and_returns_what_was_given():
    buffer = PrioritizedReplayBuffer(capacity=8)
    rng = np.random.default_rng(1)
    stored = [transition(index, rng) for index in range(5)]
    for entry in stored:
        buffer.add(*entry)

    batch, slots, weights = buffer.sample(20, rng=np.random.default_rng(2))
    assert len(batch) == 20 and len(slots) == 20 and len(weights) == 20
    for slot in np.unique(slots):
        state, action, reward, next_state, mask, done = stored[slot]
        picked = np.flatnonzero((batch.states == state).all(axis=(1, 2, 3)))
        assert len(picked) > 0
        assert (batch.rewards[picked] == reward).all()
        assert (batch.actions[picked] == action).all()


def test_it_behaves_like_the_uniform_buffer_when_it_is_full():
    buffer = PrioritizedReplayBuffer(capacity=4)
    rng = np.random.default_rng(3)
    for index in range(10):                     # twice the capacity
        buffer.add(*transition(index, rng))
    assert len(buffer) == 4
    assert buffer.full
    assert buffer.size == 4                     # the ring kept the newest four


def test_a_new_transition_gets_the_highest_priority_seen_so_far():
    buffer = filled(capacity=8, count=4)
    assert buffer.max_priority == 1.0
    buffer.update_priorities(np.array([0]), np.array([5.0]))
    assert buffer.max_priority == pytest.approx(5.0)
    buffer.add(*transition(9, np.random.default_rng(4)))
    assert buffer._priority(buffer.size - 1) == pytest.approx(5.0)


def test_the_tree_total_is_the_sum_of_the_priorities():
    buffer = filled(capacity=8, count=6, alpha=1.0)
    assert buffer.total_priority == pytest.approx(6.0)      # six transitions at 1.0
    buffer.update_priorities(np.array([0, 1]), np.array([2.0, 3.0]))
    # (|2| + 1e-6)^1 + (|3| + 1e-6)^1 + the four that are still 1.0
    assert buffer.total_priority == pytest.approx(2.0 + 3.0 + 4.0, rel=1e-4)


def test_the_sum_tree_stays_exact():
    """Every slot's leaf must be its priority, and every parent the sum of its two."""
    buffer = filled(capacity=32, count=20, alpha=0.6)
    expected = [1.0] * 20                      # what `add` seeded: the max priority
    rng = np.random.default_rng(5)
    for round_ in range(4):
        slots = rng.integers(0, 20, size=6)
        errors = rng.normal(size=6) * (10.0 ** round_)
        buffer.update_priorities(slots, errors)
        for slot, error in zip(slots, errors):
            expected[slot] = (abs(float(error)) + buffer.epsilon) ** buffer.alpha

    leaves = buffer._tree[buffer._leaf_count:buffer._leaf_count + 32]
    for slot in range(20):
        # An added transition is seeded with the highest priority seen so far
        # (1.0 here, before anything has been updated); an updated one is
        # (|error| + epsilon) ** alpha. The tree must hold exactly that.
        assert leaves[slot] == pytest.approx(expected[slot], rel=1e-9), f"slot {slot}"
    assert (leaves[20:] == 0).all()                         # unused slots stay empty
    for node in range(1, buffer._leaf_count):
        assert buffer._tree[node] == pytest.approx(
            buffer._tree[2 * node] + buffer._tree[2 * node + 1], rel=1e-9
        ), f"node {node}"
    assert buffer.total_priority == pytest.approx(float(leaves.sum()), rel=1e-9)


def test_overwriting_a_slot_replaces_its_priority():
    buffer = PrioritizedReplayBuffer(capacity=3)
    rng = np.random.default_rng(6)
    buffer.add(*transition(0, rng))
    buffer.update_priorities(np.array([0]), np.array([9.0]))
    buffer.add(*transition(1, rng))
    buffer.add(*transition(2, rng))
    buffer.add(*transition(3, rng))                          # overwrites slot 0
    assert buffer._priority(0) == pytest.approx(buffer.max_priority)
    assert buffer.total_priority == pytest.approx(float(
        sum(buffer._priority(slot) for slot in range(3))
    ), rel=1e-9)


# ------------------------------------------------------------------ the point


def test_a_transition_with_a_big_error_comes_back_often():
    """Priority 100 vs 1 should be roughly 100x more likely (alpha = 1)."""
    buffer = PrioritizedReplayBuffer(capacity=8, alpha=1.0, beta=0.0)
    rng = np.random.default_rng(7)
    for index in range(8):
        buffer.add(*transition(index, rng))
    buffer.update_priorities(np.arange(8), np.full(8, 0.01))
    buffer.update_priorities(np.array([3]), np.array([10.0]))    # one loud slot

    _, slots, _ = buffer.sample(2000, rng=np.random.default_rng(8))
    share = float((slots == 3).mean())
    loud = 8 * 10.000001
    quiet = 7 * (8 * 0.01 + 1e-6)
    expected = loud / (loud + quiet)
    assert abs(share - expected) < 0.05, f"{share:.3f} against {expected:.3f}"
    assert share > 0.9, "the loud slot should dominate the batch"


def test_uniform_priorities_sample_uniformly():
    buffer = filled(capacity=64, count=64, alpha=1.0, beta=0.0)
    _, slots, _ = buffer.sample(4000, rng=np.random.default_rng(9))
    counts = np.bincount(slots, minlength=64)
    assert counts.min() > 0                                  # stratified: nothing missed
    assert abs(counts.mean() - 4000 / 64) < 20


def test_the_batch_is_stratified_so_the_top_of_the_range_is_never_missed():
    buffer = PrioritizedReplayBuffer(capacity=4, alpha=1.0, beta=0.0)
    rng = np.random.default_rng(10)
    for index in range(4):
        buffer.add(*transition(index, rng))
    buffer.update_priorities(np.arange(4), np.array([0.0, 0.0, 0.0, 1e6]))
    for seed in range(5):                                    # never misses it
        _, slots, _ = buffer.sample(4, rng=np.random.default_rng(seed))
        assert 3 in slots


# ------------------------------------------------------------------ the weights


def test_the_weights_normalise_to_one_at_the_batch_maximum():
    buffer = filled(capacity=64, count=64, beta=0.5)
    rng = np.random.default_rng(11)
    buffer.update_priorities(np.arange(64), rng.random(64) + 0.01)
    _, _, weights = buffer.sample(16, rng=np.random.default_rng(12))
    assert weights.max() == pytest.approx(1.0)
    assert (weights > 0).all()
    assert (weights <= 1.0 + 1e-6).all()


def test_beta_zero_means_no_correction_at_all():
    buffer = filled(capacity=32, count=32, beta=0.0)
    rng = np.random.default_rng(13)
    buffer.update_priorities(np.arange(32), rng.random(32) + 0.01)
    _, _, weights = buffer.sample(8, rng=np.random.default_rng(14), beta=0.0)
    assert np.allclose(weights, 1.0)


def test_a_rarer_transition_is_worth_more():
    buffer = PrioritizedReplayBuffer(capacity=4, alpha=1.0, beta=1.0)
    rng = np.random.default_rng(15)
    for index in range(4):
        buffer.add(*transition(index, rng))
    buffer.update_priorities(np.arange(4), np.array([1.0, 1.0, 1.0, 100.0]))
    _, slots, weights = buffer.sample(400, rng=np.random.default_rng(16))
    rare = weights[slots == 3]
    common = weights[slots == 0]
    assert len(rare) and len(common)
    assert rare.mean() < common.mean() / 10


def test_the_same_seed_gives_the_same_batch():
    buffer = filled(capacity=32, count=32)
    first, first_slots, first_weights = buffer.sample(8, rng=np.random.default_rng(17))
    second, second_slots, second_weights = buffer.sample(8, rng=np.random.default_rng(17))
    assert (first_slots == second_slots).all()
    assert (first_weights == second_weights).all()
    assert (first.rewards == second.rewards).all()


# ---------------------------------------------------------------- the errors


def test_sampling_an_empty_buffer_is_an_error():
    with pytest.raises(ValueError, match="the buffer is empty"):
        PrioritizedReplayBuffer(capacity=4).sample(2)


def test_mismatched_errors_and_slots_are_rejected():
    buffer = filled()
    with pytest.raises(ValueError, match="must pair up"):
        buffer.update_priorities(np.array([0, 1]), np.array([1.0]))


@pytest.mark.parametrize("kwargs, message", [
    ({"capacity": 0}, "capacity must be 1 or more"),
    ({"alpha": 1.5}, "alpha must be in"),
    ({"beta": -1.0}, "beta must be in"),
    ({"epsilon": 0.0}, "epsilon must be > 0"),
])
def test_bad_settings_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PrioritizedReplayBuffer(**kwargs)


def test_a_batch_size_below_one_is_rejected():
    buffer = filled()
    with pytest.raises(ValueError, match="batch_size must be 1 or more"):
        buffer.sample(0)


def test_it_says_what_it_is():
    buffer = filled(capacity=32, count=3)
    assert "3/32" in repr(buffer)
    assert isinstance(buffer.states, np.ndarray)
    assert buffer.states.shape == (32, *STATE_SHAPE)
    assert buffer.next_masks.shape == (32, ACTION_SIZE)
    assert buffer.dones.dtype == np.float32


def test_it_holds_the_same_transitions_the_uniform_buffer_would():
    """Same ring, same fields: an arm can swap one buffer for the other."""
    uniform = ReplayBuffer(capacity=5)
    prioritised = PrioritizedReplayBuffer(capacity=5)
    rng = np.random.default_rng(18)
    entries = [transition(index, rng) for index in range(9)]
    for entry in entries:
        uniform.add(*entry)
        prioritised.add(*entry)
    assert (uniform.states == prioritised.states).all()
    assert (uniform.actions == prioritised.actions).all()
    assert (uniform.rewards == prioritised.rewards).all()
    assert (uniform.next_states == prioritised.next_states).all()
    assert (uniform.next_masks == prioritised.next_masks).all()
    assert (uniform.dones == prioritised.dones).all()
    assert (uniform.positions == prioritised.positions).all()
