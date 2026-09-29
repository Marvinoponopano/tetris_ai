"""Tests for the replay buffer and the Q-network."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from tetris.rl.encoding import ACTION_SIZE, STATE_SHAPE  # noqa: E402
from tetris.rl.network import QNetwork, build_network  # noqa: E402
from tetris.rl.replay import ReplayBuffer  # noqa: E402


def a_transition(index: int = 0):
    """A transition whose numbers make it identifiable in a sample."""
    state = np.full(STATE_SHAPE, index, dtype=np.float32)
    next_state = np.full(STATE_SHAPE, index + 100, dtype=np.float32)
    mask = np.zeros(ACTION_SIZE, dtype=bool)
    mask[: 9 + (index % 3)] = True
    return state, index % ACTION_SIZE, float(index), next_state, mask, bool(index % 2)


# ------------------------------------------------------------------ the buffer


def test_a_new_buffer_is_empty():
    buffer = ReplayBuffer(capacity=4)
    assert len(buffer) == 0 and buffer.size == 0
    assert not buffer.full
    assert buffer.capacity == 4


def test_storing_keeps_the_arrays_at_the_documented_shape():
    buffer = ReplayBuffer(capacity=4)
    buffer.add(*a_transition(1))

    assert len(buffer) == 1
    assert buffer.states.shape == (4,) + STATE_SHAPE
    assert buffer.states.dtype == np.float32
    assert buffer.actions.dtype == np.int64
    assert buffer.next_masks.dtype == bool
    assert buffer.dones.dtype == np.float32
    assert buffer.states[0, 0, 0, 0] == 1.0
    assert buffer.rewards[0] == 1.0


def test_sampling_returns_the_batch_contract():
    buffer = ReplayBuffer(capacity=16)
    for index in range(6):
        buffer.add(*a_transition(index))

    batch = buffer.sample(5, rng=np.random.default_rng(0))

    assert batch.states.shape == (5,) + STATE_SHAPE
    assert batch.actions.shape == (5,)
    assert batch.rewards.shape == (5,)
    assert batch.next_states.shape == (5,) + STATE_SHAPE
    assert batch.next_masks.shape == (5, ACTION_SIZE)
    assert batch.dones.shape == (5,)
    assert len(batch) == 5


def test_a_sample_only_contains_what_was_stored():
    buffer = ReplayBuffer(capacity=8)
    for index in range(3):
        buffer.add(*a_transition(index))

    batch = buffer.sample(64, rng=np.random.default_rng(1))

    assert set(np.unique(batch.rewards)) <= {0.0, 1.0, 2.0}
    for row, (reward, action, done) in enumerate(zip(batch.rewards, batch.actions,
                                                     batch.dones)):
        _state, expected_action, _reward, _next, _mask, expected_done = a_transition(int(reward))
        assert action == expected_action and done == expected_done
        # state, action, reward and done came from the same stored transition
        assert batch.states[row][0, 0, 0] == reward
        assert batch.next_states[row][0, 0, 0] == reward + 100


def test_the_oldest_transition_is_overwritten_when_full():
    buffer = ReplayBuffer(capacity=3)
    for index in range(5):                       # 0 and 1 must be gone
        buffer.add(*a_transition(index))

    assert len(buffer) == 3 and buffer.full
    batch = buffer.sample(200, rng=np.random.default_rng(2))
    assert set(np.unique(batch.rewards)) <= {2.0, 3.0, 4.0}
    assert sorted(buffer.positions.tolist()) == [0, 1, 2]


def test_writing_past_capacity_keeps_everything_consistent():
    """After thousands of wraps the buffer still holds exactly its last capacity."""
    buffer = ReplayBuffer(capacity=5)
    for index in range(1_000):
        buffer.add(*a_transition(index))

    assert sorted(buffer.positions.tolist()) == [0, 1, 2, 3, 4]
    assert buffer.size == 5 and buffer.full
    rewards = {float(buffer.rewards[position]) for position in buffer.positions}
    assert rewards == {995.0, 996.0, 997.0, 998.0, 999.0}


def test_sampling_is_reproducible_with_a_seeded_generator():
    buffer = ReplayBuffer(capacity=10)
    for index in range(10):
        buffer.add(*a_transition(index))

    first = buffer.sample(6, rng=np.random.default_rng(7)).rewards
    second = buffer.sample(6, rng=np.random.default_rng(7)).rewards
    assert np.array_equal(first, second)


def test_sampling_an_empty_buffer_is_an_error():
    with pytest.raises(ValueError, match="empty"):
        ReplayBuffer(capacity=4).sample(2)
    with pytest.raises(ValueError, match="batch_size"):
        ReplayBuffer(capacity=4).sample(0)
    with pytest.raises(ValueError, match="capacity"):
        ReplayBuffer(capacity=0)


# ----------------------------------------------------------------- the network


def test_the_network_outputs_one_value_per_placement():
    """The network takes a batch (the agent adds the batch dim for one state)."""
    network = QNetwork(hidden_sizes=(32,))
    assert network(torch.zeros(4, *STATE_SHAPE)).shape == (4, ACTION_SIZE)
    assert network(torch.zeros(1, *STATE_SHAPE)).shape == (1, ACTION_SIZE)


def test_the_default_network_is_small():
    network = QNetwork()
    assert network.parameter_count == 229_928          # ~1 MB of weights
    assert network.parameter_count < 300_000
    assert "-> 40" in repr(network)


def test_a_seeded_network_is_reproducible():
    first = build_network(hidden_sizes=(16,), seed=3)
    second = build_network(hidden_sizes=(16,), seed=3)
    states = torch.rand(2, *STATE_SHAPE)
    assert torch.equal(first(states), second(states))


def test_a_fresh_network_starts_indifferent():
    """Zero-initialised output layer: no column is preferred before any learning."""
    values = build_network(hidden_sizes=(16,), seed=0)(torch.zeros(1, *STATE_SHAPE))
    assert torch.allclose(values, torch.zeros_like(values))


def test_the_network_trains():
    network = QNetwork(hidden_sizes=(16,))
    optimizer = torch.optim.SGD(network.parameters(), lr=0.1)
    states = torch.rand(3, *STATE_SHAPE)
    loss = nn.functional.mse_loss(network(states), torch.ones(3, ACTION_SIZE))
    optimizer.zero_grad()
    loss.backward()

    gradients = [p.grad for p in network.parameters() if p.grad is not None]
    assert gradients and any(float(g.abs().sum()) > 0 for g in gradients)

    optimizer.step()
    assert not torch.allclose(network(states), torch.zeros(3, ACTION_SIZE))


def test_hidden_sizes_must_not_be_empty():
    with pytest.raises(ValueError, match="hidden_sizes"):
        QNetwork(hidden_sizes=())
