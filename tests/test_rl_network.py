"""Tests for the network architectures (`tetris/rl/network.py`).

`QNetwork` is pinned elsewhere (through the agent); what is pinned here is the
dueling head: that it shares the same trunk as the reference network for the same
hidden sizes (so an arm compares architectures, not capacities by accident), that
`Q = V + A - mean(A)` holds exactly, and that it starts as indifferent as the
reference network does -- every action's Q equal -- because an arm that starts by
arbitrarily preferring one column is not measuring the arm.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tetris.rl.encoding import ACTION_SIZE, EXTENDED_SHAPE, STATE_SHAPE  # noqa: E402
from tetris.rl.network import (  # noqa: E402
    DuelingQNetwork,
    QNetwork,
    build_dueling_network,
    build_network,
    trunk_layers,
)


def states(count: int = 4, seed: int = 0, shape=STATE_SHAPE) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 2, size=(count, *shape)).astype(np.float32)
    return torch.as_tensor(data)


# ----------------------------------------------------------------- shared trunk


def test_the_dueling_trunk_is_the_reference_front_end():
    hidden = (16, 8)
    reference = QNetwork(hidden_sizes=hidden)
    dueling = DuelingQNetwork(hidden_sizes=hidden)
    reference_layers = [module for module in reference.layers]
    dueling_layers = [module for module in dueling.trunk]
    assert len(reference_layers) == len(dueling_layers) + 1     # the head differs
    for mine, theirs in zip(reference_layers[:-1], dueling_layers):
        assert type(mine) is type(theirs)
        if isinstance(mine, torch.nn.Linear):
            assert mine.weight.shape == theirs.weight.shape


def test_the_full_test_builds_the_same_trunk_layers():
    layers = trunk_layers((3, 20, 10), (256, 256))
    linears = [layer for layer in layers if isinstance(layer, torch.nn.Linear)]
    assert [layer.weight.shape for layer in linears] == [
        (256, 600), (256, 256)
    ]


# ------------------------------------------------------------- the decomposition


def test_q_is_value_plus_centred_advantage():
    network = DuelingQNetwork(hidden_sizes=(12,))
    batch = states(6)
    hidden = network.trunk(batch)
    value = network.value(hidden)
    advantage = network.advantage(hidden)
    expected = value + advantage - advantage.mean(dim=1, keepdim=True)
    assert torch.allclose(network(batch), expected, atol=1e-6)


def test_the_advantage_is_zero_mean_across_actions():
    """Which is what makes V and A identifiable rather than only their sum."""
    network = build_dueling_network(hidden_sizes=(12,), seed=1)
    batch = states(8)
    hidden = network.trunk(batch)
    advantage = network.advantage(hidden)
    assert torch.allclose(advantage.mean(dim=1), torch.zeros(8), atol=1e-6)


def test_the_dueling_network_returns_the_same_value_it_calls_the_state():
    network = build_dueling_network(hidden_sizes=(12,), seed=2)
    batch = states(3)
    values = network(batch)
    state_value = network.value(network.trunk(batch))
    # Q and V differ only by the (zero-mean) advantage, so the *mean* over
    # actions is the state value exactly.
    assert torch.allclose(values.mean(dim=1, keepdim=True), state_value, atol=1e-6)


# ------------------------------------------------------------------- the start


def test_both_networks_start_indifferent_between_actions():
    for builder in (build_network, build_dueling_network):
        network = builder(hidden_sizes=(16,), seed=0)
        values = network(states(5))
        spread = (values.max(dim=1).values - values.min(dim=1).values).abs().max()
        assert float(spread) < 1e-6, f"{builder.__name__} starts with a preference"


def test_a_dueling_network_costs_one_extra_scalar_per_hidden_unit():
    hidden = (256, 256)
    reference = QNetwork(hidden_sizes=hidden)
    dueling = DuelingQNetwork(hidden_sizes=hidden)
    # The reference head is Linear(last, 40); the dueling head is
    # Linear(last, 40) + Linear(last, 1) -- so exactly one bias and one row more.
    assert dueling.parameter_count - reference.parameter_count == hidden[-1] + 1


# ---------------------------------------------------------------- the shapes


def test_it_follows_the_state_shape_it_is_given():
    dueling = DuelingQNetwork(state_shape=EXTENDED_SHAPE, hidden_sizes=(8,))
    assert dueling.state_shape == EXTENDED_SHAPE
    assert dueling(states(2, shape=EXTENDED_SHAPE)).shape == (2, ACTION_SIZE)


def test_it_has_one_output_per_action():
    network = DuelingQNetwork(hidden_sizes=(8,))
    assert network(states(1)).shape == (1, ACTION_SIZE)
    assert network.action_size == ACTION_SIZE


def test_hidden_sizes_must_not_be_empty():
    with pytest.raises(ValueError, match="hidden_sizes must not be empty"):
        DuelingQNetwork(hidden_sizes=())


def test_it_describes_itself():
    dueling = DuelingQNetwork(hidden_sizes=(16, 8))
    text = repr(dueling)
    assert "DuelingQNetwork" in text and "V(1)" in text and "A(40)" in text
