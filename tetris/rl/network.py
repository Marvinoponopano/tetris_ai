"""The Q-network: a small fully-connected network over the encoded state.

Input  `(batch, 3, 20, 10)`  ->  flattened to 600 numbers
Output `(batch, 40)`         ->  one Q-value per `(rotation, column)`

Why a plain MLP rather than a CNN. A CNN would be the natural choice if the
network had to *score a board* (which is what the heuristic agent does), because
it can learn local patterns -- holes, wells, overhangs -- anywhere on the grid.
Here it does something easier: it sees the board and the two pieces and has to
rank 40 fixed actions, all of which it already knows the meaning of. A 600-input
MLP can express that, trains in seconds per thousand updates on a laptop CPU, and
gives the same Q-value for all 40 actions in a single forward pass (a per-action
board scorer needs 40). The layer stack is a one-line change to a `Conv2d`
version later if the MLP turns out to be the limit -- the input shape does not
move.

Size: `256 + 256` hidden units is ~230k parameters, about 1 MB, deliberately
small -- this stage is about a correct pipeline, not capacity.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import torch
from torch import nn

from .encoding import ACTION_SIZE, STATE_SHAPE


def trunk_layers(
    state_shape: Tuple[int, ...], hidden_sizes: Sequence[int]
) -> list:
    """`[Flatten, Linear, ReLU, ...]` up to the last hidden layer, as a list.

    Shared by `QNetwork` and `DuelingQNetwork` so both see the same front end
    for the same `hidden_sizes`: a dueling head splits *after* this, not instead
    of it, which is what makes the two architectures comparable parameter for
    parameter.
    """
    layers = [nn.Flatten()]
    previous = int(np.prod(tuple(state_shape)))
    for size in hidden_sizes:
        layers.append(nn.Linear(previous, int(size)))
        layers.append(nn.ReLU())
        previous = int(size)
    return layers


class QNetwork(nn.Module):
    """Maps state planes to a Q-value for every possible placement."""

    def __init__(
        self,
        state_shape: Tuple[int, ...] = STATE_SHAPE,
        action_size: int = ACTION_SIZE,
        hidden_sizes: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes must not be empty")
        self.state_shape = tuple(state_shape)
        self.action_size = int(action_size)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)

        layers = trunk_layers(self.state_shape, self.hidden_sizes)
        previous = self.hidden_sizes[-1]
        layers.append(nn.Linear(previous, self.action_size))
        self.layers = nn.Sequential(*layers)

        # Small init keeps early Q-values near zero, so early targets are not
        # dominated by noise before anything has been learned.
        for module in self.layers:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """`(batch, 3, 20, 10)` -> `(batch, 40)`."""
        return self.layers(states)

    @property
    def parameter_count(self) -> int:
        """How many numbers the network learns (for logging)."""
        return sum(parameter.numel() for parameter in self.parameters())

    def __repr__(self) -> str:
        return (
            f"<QNetwork {self.state_shape} -> {list(self.hidden_sizes)} -> "
            f"{self.action_size}, {self.parameter_count} params>"
        )


def build_network(
    hidden_sizes: Sequence[int] = (256, 256),
    state_shape: Tuple[int, ...] = STATE_SHAPE,
    action_size: int = ACTION_SIZE,
    seed: int = 0,
) -> QNetwork:
    """A new network with reproducible weights (`seed` reseeds torch globally)."""
    torch.manual_seed(seed)
    network = QNetwork(state_shape=state_shape, action_size=action_size,
                       hidden_sizes=hidden_sizes)
    # Zero the *last* layer's bias so the initial Q-values are all equal: the
    # agent is then truly indifferent at the start rather than arbitrarily
    # preferring one column.
    last = network.layers[-1]
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    return network


class DuelingQNetwork(nn.Module):
    """`Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))`, the dueling architecture.

    One network asked to produce 40 Q-values spends its capacity learning two
    different things at once: how good this position is at all, and which of the
    placements is best. Those separate cleanly --

        Q(s, a) = V(s) + A(s, a)

    -- and the dueling head makes that split explicit: a trunk shared by both
    heads, then one scalar output for `V(s)` and 40 outputs for `A(s, a)`, with
    the mean of `A` subtracted so the decomposition is identifiable (otherwise
    adding a constant to `A` and subtracting it from `V` gives the same `Q`, and
    the two heads drift).

    Why it might matter *here*, from the measurement rather than from the paper:
    the diagnosis of G's policy found a mean best Q of 3.63 with only 1.61
    between the best and the second-best action, so roughly 44% of the value the
    network reports is the same for every legal placement -- it is the value of
    the *position*, not of the move. The reference network has to represent that
    common part 40 times over, once per output, and it has no incentive to keep
    those 40 copies consistent, which is exactly the noise the subtract-the-mean
    head removes.

    Same trunk, same `hidden_sizes`, same input shape: only the last layer is
    replaced by two.
    """

    def __init__(
        self,
        state_shape: Tuple[int, ...] = STATE_SHAPE,
        action_size: int = ACTION_SIZE,
        hidden_sizes: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError("hidden_sizes must not be empty")
        self.state_shape = tuple(state_shape)
        self.action_size = int(action_size)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)

        self.trunk = nn.Sequential(*trunk_layers(self.state_shape, self.hidden_sizes))
        self.value = nn.Linear(self.hidden_sizes[-1], 1)
        self.advantage = nn.Linear(self.hidden_sizes[-1], self.action_size)

        for head in (self.value, self.advantage):
            nn.init.kaiming_uniform_(head.weight, nonlinearity="relu")
            nn.init.zeros_(head.bias)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """`(batch, 3, 20, 10)` -> `(batch, 40)`."""
        hidden = self.trunk(states)
        value = self.value(hidden)
        advantage = self.advantage(hidden)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    @property
    def parameter_count(self) -> int:
        """How many numbers the network learns (for logging)."""
        return sum(parameter.numel() for parameter in self.parameters())

    def __repr__(self) -> str:
        return (
            f"<DuelingQNetwork {self.state_shape} -> {list(self.hidden_sizes)} -> "
            f"V(1) + A({self.action_size}), {self.parameter_count} params>"
        )


def build_dueling_network(
    hidden_sizes: Sequence[int] = (256, 256),
    state_shape: Tuple[int, ...] = STATE_SHAPE,
    action_size: int = ACTION_SIZE,
    seed: int = 0,
) -> DuelingQNetwork:
    """A new dueling network, starting from all-equal Q-values (`build_network`).

    Both heads are zeroed, so `V = 0` and `A = 0` and every action starts with
    `Q = 0`: the same indifferent start the plain network gets from zeroing its
    output layer.
    """
    torch.manual_seed(seed)
    network = DuelingQNetwork(state_shape=state_shape, action_size=action_size,
                              hidden_sizes=hidden_sizes)
    for head in (network.value, network.advantage):
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
    return network


__all__ = [
    "DuelingQNetwork",
    "QNetwork",
    "build_dueling_network",
    "build_network",
    "trunk_layers",
]
