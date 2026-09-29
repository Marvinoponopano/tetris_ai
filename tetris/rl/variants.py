"""Agent variants -- a diagnosis turns into an experiment, not into a rewrite.

`DQNAgent` is the reference implementation (see `tetris.rl.dqn`). A variant is a
*subclass* that changes one thing about it, so the baseline stays untouched and
the two can be run side by side on the same seeds. Nothing in this module
modifies the classes it derives from; a variant is only ever used when a caller
asks for it by name, through `build_variant_agent`.

Variants
--------

``dqn`` (the default)
    `DQNAgent` itself: the target is ``r + gamma * max_a' Q_target(s', a')``.
    A plain, correct DQN -- which is the point. It is the arm everything else is
    compared against.

``double-dqn``
    `DoubleDQNAgent`: the *online* network picks the next action, the *target*
    network scores it::

        a* = argmax_a' Q_online(s', a')
        target = r + gamma * Q_target(s', a*)

    Vanilla DQN's max over the target network's own estimates is biased upward:
    the same noisy weights both choose and score, and the noise is optimised
    against. The bias grows with the action count (40 here, of which ~23 are
    legal per piece) and gets worse early in training, when the estimates are
    pure noise -- exactly when a bootstrap signal matters most. Double DQN
    removes the optimism at the cost of one extra forward pass, and the failure
    mode it addresses (learning to expect reward that never arrives) shows up as
    Q-values that keep drifting up while the score does not.

    Measured: this was the one arm that came out *worse* than its control (D:
    511.4 against C's 537.5), which is what ruled the estimator out as the
    bottleneck and moved the diagnosis to the objective.

``feature-planes``
    `FeaturePlanesAgent`: two more state planes -- each column's height, and the
    cells that are holes -- taken from `tetris.metrics`, which is where the
    heuristic agent gets them from too. Motivates itself: the heuristic is not
    better at arithmetic than a network, it is simply handed the geometry.

``queue-planes``
    `QueuePlanesAgent`: the five feature planes plus the piece after next, so the
    network can plan for a piece it has not been dealt yet -- the cheap form of
    lookahead (see `encode_state_with_queue`).

``n-step``
    `NStepAgent`: three-step returns, so a line clear's payoff is inside the
    target of the move that started building for it (see `tetris.rl.nstep`).

``dueling``
    `DuelingAgent`: `Q = V + A - mean(A)`, motivated by the measurement that
    44% of the value G reports is the same for every legal placement.

``prioritized``
    `PrioritizedAgent`: replay by TD error with importance weights, motivated by
    how few of the 86,000 decisions in a run are line clears (see
    `tetris.rl.prioritized`).

How far the "same weights, any loader" promise reaches
------------------------------------------------------

`double-dqn` only changes the Bellman target, so its checkpoint has the reference
network's exact shapes: it loads back into a plain `DQNAgent` and plays the same
moves, and the comparison against the baselines does not depend on which variant
produced the weights.

The four arms added for round two change *what the network is handed* or *what it
is*, so their checkpoints are not interchangeable with a plain `DQNAgent`: a
dueling head is not an MLP, and a five-plane input is not a three-plane one. Each
class registers itself in `tetris.rl.registry`, and `load_agent_for_evaluation`
looks the class up in the checkpoint's own `agent` field, so such a checkpoint
still loads and plays -- as the agent that wrote it -- in the comparison table,
the diagnosis and the viewer.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

from ..env import TetrisState
from .config import TrainingConfig
from .dqn import LARGE_NEGATIVE, MAX_GRADIENT_NORM, DQNAgent
from .encoding import (
    EXTENDED_SHAPE,
    QUEUE_SHAPE,
    encode_state_with_features,
    encode_state_with_queue,
)
from .network import build_dueling_network, build_network
from .nstep import NStepTransitions
from .prioritized import PrioritizedReplayBuffer
from .registry import register
from .replay import Batch


@register
class DoubleDQNAgent(DQNAgent):
    """DQN with Double Q-learning targets (see the module docstring)."""

    name = "double-dqn"

    def targets_for(self, batch: Batch) -> np.ndarray:
        """``r + gamma * Q_target(s', argmax_a' Q_online(s', a'))`` over legal actions.

        The argmax is taken over the *online* network's masked values, and the
        chosen action is then scored with the frozen target network. Masking
        happens before the argmax (as in the parent class), so the action that
        gets picked and the value that gets used are both legal ones.
        """
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(batch.next_masks, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            online_values = self.online(next_states).masked_fill(~next_masks, LARGE_NEGATIVE)
            chosen = online_values.argmax(dim=1, keepdim=True)
            target_values = self.target(next_states).gather(1, chosen).squeeze(1)
            targets = rewards + self.config.gamma * target_values * (1.0 - dones)
        return targets.cpu().numpy()


@register
class FeaturePlanesAgent(DQNAgent):
    """DQN over five planes instead of three: the board plus its geometry.

    `encode_state` gives the network the locked cells, the falling piece and the
    next piece -- and then asks it to work out, from those raw cells, how tall
    each column is and which empty cells are holes. Those are two of the four
    numbers the heuristic agent's whole score is made of, and it gets them from
    `tetris.metrics` rather than computing them.

    So this arm hands the network the same two measurements, as planes, on the
    grid it already reads (see `encode_state_with_features`). Same architecture,
    same 40 actions, same reward: 200 more input numbers and nothing else. If the
    DQN's problem is representational -- that 86,000 decisions is not enough to
    relearn arithmetic the heuristic is handed for free -- this is where it
    shows, and if it is not, this arm will look exactly like its control and say
    so.
    """

    name = "feature-planes"

    def __init__(self, config=None, network=None, **kwargs):
        if network is None:
            resolved = TrainingConfig() if config is None else TrainingConfig.from_any(config)
            network = build_network(
                hidden_sizes=resolved.hidden_sizes,
                state_shape=EXTENDED_SHAPE,
                seed=int(kwargs.get("seed", 0) or 0),
            )
        super().__init__(config=config, network=network, **kwargs)

    def encode(self, state: TetrisState) -> np.ndarray:
        """The five-plane encoding (`encode_state_with_features`)."""
        return encode_state_with_features(state)

    def variant_params(self) -> Dict[str, Any]:
        return {
            "state_shape": list(self.state_shape),
            "added_planes": ["column_heights", "holes"],
        }


@register
class QueuePlanesAgent(DQNAgent):
    """DQN that can see the piece after next (`TetrisState.queue`).

    The encoding has always carried the piece in hand and the piece that follows,
    and thrown the rest of the queue away (`TetrisState.queue[1:]`). That rules
    out the move a human makes near the end of a board: take a bad placement now
    because the piece two moves away turns it into a clear. The queue plane
    (`encode_state_with_queue`) shows that piece, so the network has the
    information a one-move lookahead would have used, at the cost of one plane
    instead of 40x40 network calls per decision.

    It builds on the feature encoding, so this arm is `feature-planes` plus the
    queue: read it against J, not against G, and the difference is one plane.
    """

    name = "queue-planes"

    def __init__(self, config=None, network=None, **kwargs):
        if network is None:
            resolved = TrainingConfig() if config is None else TrainingConfig.from_any(config)
            network = build_network(
                hidden_sizes=resolved.hidden_sizes,
                state_shape=QUEUE_SHAPE,
                seed=int(kwargs.get("seed", 0) or 0),
            )
        super().__init__(config=config, network=network, **kwargs)

    def encode(self, state: TetrisState) -> np.ndarray:
        """The six-plane encoding (`encode_state_with_queue`)."""
        return encode_state_with_queue(state)

    def variant_params(self) -> Dict[str, Any]:
        return {
            "state_shape": list(self.state_shape),
            "added_planes": ["column_heights", "holes", "piece_after_next"],
        }


@register
class DuelingFeaturePlanesAgent(DQNAgent):
    """The two changes round two found evidence for, together.

    `DuelingAgent` was the only arm of the five that improved the agent
    (+31.5 mean score over G, 4/5 seeds), and the five-plane encoding is the only
    one that made the heuristic's own ranking easier to learn -- the supervised
    capacity probe fits the same MLP to the heuristic's scores and agreement goes
    from 32.7% (three planes) to 45.0% (five). Neither arm is a big single win on
    its own; they attack different halves of the problem, so this arm runs them
    together, and it is explicitly a two-change arm like G was in round one.

    Written out rather than inherited from both parents because the network has
    to be built for the *five-plane* shape: a dueling net constructed at the
    default shape would not load these weights.
    """

    name = "dueling-feature-planes"

    def __init__(self, config: Optional[ConfigLike] = None, network=None, **kwargs):
        resolved = TrainingConfig() if config is None else TrainingConfig.from_any(config)
        if network is None:
            network = build_dueling_network(
                hidden_sizes=resolved.hidden_sizes,
                state_shape=EXTENDED_SHAPE,
                seed=int(kwargs.get("seed", 0)),
            )
        super().__init__(config=resolved, network=network, **kwargs)

    def encode(self, state) -> np.ndarray:
        """Five planes: the board, the pieces, column heights and holes."""
        return encode_state_with_features(state)

    def variant_params(self) -> Dict[str, Any]:
        return {
            "head": "dueling",
            "state_shape": list(EXTENDED_SHAPE),
            "extra_planes": ["column_heights", "holes"],
        }


@register
class NStepAgent(DQNAgent):
    """DQN with n-step returns (n = 3): one transition, three steps of reward.

    Everything a line clear is worth arrives on the fifth placement, and the four
    before it look like any other placement. Training one step at a time, that
    payoff has to travel back through the network's own value estimates, which is
    slow when the signal itself is that rare. This arm hands the update
    ``r_t + gamma r_{t+1} + gamma^2 r_{t+2}`` instead, so the clear is inside
    the target of the move that started the row.

    The window lives in the agent, and `end_episode` flushes it, so no transition
    is ever built across two games -- including across the piece cap, which ends
    an episode without a game over. See `tetris.rl.nstep` for the bookkeeping.
    """

    name = "n-step"
    n_step = 3

    def __init__(self, config=None, network=None, **kwargs):
        super().__init__(config=config, network=network, **kwargs)
        self.window = NStepTransitions(n=self.n_step, gamma=self.config.gamma)

    def remember(
        self,
        buffer,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_mask: np.ndarray,
        done: bool,
    ) -> None:
        """Feed the window; store whatever n-step transitions it now has ready."""
        for transition in self.window.push(state, action, reward, next_state, next_mask, done):
            buffer.add(*transition)

    def end_episode(self, buffer) -> None:
        """Emit the tail of the window (shorter returns) and empty it."""
        for transition in self.window.flush():
            buffer.add(*transition)

    def variant_params(self) -> Dict[str, Any]:
        return {"n_step": self.n_step, "gamma": self.config.gamma}


@register
class DuelingAgent(DQNAgent):
    """DQN with a dueling head: `Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))`.

    The diagnosis of G's policy found a mean best Q of 3.63 with 1.61 between the
    best and the second-best action: most of what the network reports is the
    value of the *position*, not of the placement. A plain network represents
    that shared part 40 times, once per output, with nothing keeping the copies
    consistent; the dueling head represents it once, in `V`, and lets the 40
    outputs be a zero-mean advantage. Same trunk, same hidden sizes.
    """

    name = "dueling"

    def __init__(self, config=None, network=None, **kwargs):
        if network is None:
            resolved = TrainingConfig() if config is None else TrainingConfig.from_any(config)
            network = build_dueling_network(
                hidden_sizes=resolved.hidden_sizes,
                seed=int(kwargs.get("seed", 0) or 0),
            )
        super().__init__(config=config, network=network, **kwargs)

    def variant_params(self) -> Dict[str, Any]:
        return {"head": "dueling", "hidden_sizes": list(self.hidden_sizes)}


@register
class PrioritizedAgent(DQNAgent):
    """DQN that replays the transitions it is most wrong about.

    A run makes ~86,000 decisions and clears a few hundred lines: line clears are
    roughly 0.5% of the evidence, and uniform sampling spends 99.5% of every
    batch on placements whose value the network already knows. This arm draws
    each transition with probability proportional to its last TD error
    (`|delta| + epsilon) ^ alpha`), with importance weights and `beta` annealed
    from 0.4 to 1.0 across the run, so the early updates are cheap and the late
    ones are unbiased. See `tetris.rl.prioritized`.

    `alpha` and `beta_start` are class attributes rather than config fields: the
    arm's identity is the variant name, which the campaign records, and keeping
    them here means the reference `TrainingConfig` does not grow a knob that only
    one arm reads.
    """

    name = "prioritized"
    alpha = 0.6
    beta_start = 0.4

    def __init__(self, config=None, network=None, **kwargs):
        super().__init__(config=config, network=network, **kwargs)
        self._buffer = None
        self._pending: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def build_buffer(self, config: TrainingConfig) -> PrioritizedReplayBuffer:
        """The prioritised buffer, sized and shaped exactly like the uniform one."""
        return PrioritizedReplayBuffer(
            capacity=config.buffer_size,
            state_shape=self.state_shape,
            action_size=self.action_size,
            alpha=self.alpha,
            beta=self.beta_start,
        )

    def beta(self) -> float:
        """The importance-sampling exponent: `beta_start` -> 1.0 over the run.

        Annealed by episode, because that is the clock the agent can read without
        being told how long episodes are.
        """
        episodes = max(1, int(self.config.episodes))
        progress = min(1.0, max(0.0, self.episode / episodes))
        return self.beta_start + (1.0 - self.beta_start) * progress

    def sample_minibatch(self, buffer, rng=None) -> Batch:
        """A priority sample, remembering its slots and weights for the update."""
        batch, slots, weights = buffer.sample(self.config.batch_size, rng, beta=self.beta())
        self._buffer = buffer
        self._pending = (slots, weights)
        return batch

    def train_step(self, batch: Batch) -> float:
        """A gradient step on the importance-weighted Huber loss.

        Then the priorities of the sampled slots are set from the TD errors this
        step measured -- which is the whole feedback loop: the update decides
        what to look at next.
        """
        slots, weights = self._pending if self._pending is not None else (None, None)
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=self.device)
        targets = torch.as_tensor(self.targets_for(batch), dtype=torch.float32,
                                  device=self.device)

        predicted = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        errors = (predicted - targets).detach()
        per_sample = nn.functional.smooth_l1_loss(
            predicted, targets, reduction="none"
        )
        if weights is not None:
            per_sample = per_sample * torch.as_tensor(
                weights, dtype=torch.float32, device=self.device
            )
        loss = per_sample.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), MAX_GRADIENT_NORM)
        self.optimizer.step()

        if slots is not None and self._buffer is not None:
            self._buffer.update_priorities(slots, errors.abs().cpu().numpy())

        self.steps += 1
        if self.steps % self.config.target_update_steps == 0:
            self.sync_target()
        return float(loss.item())

    def variant_params(self) -> Dict[str, Any]:
        return {"alpha": self.alpha, "beta_start": self.beta_start}


#: variant name -> a factory taking `(config, seed)` and returning an agent.
#: `None` means "let `train()` build the default agent", which is the baseline.
VARIANTS: Dict[str, Optional[Callable[[TrainingConfig, int], DQNAgent]]] = {
    "dqn": None,
    "double-dqn": lambda config, seed: DoubleDQNAgent(
        config=config, device=config.device if config.device != "auto" else "cpu", seed=seed
    ),
    "dueling-feature-planes": lambda config, seed: DuelingFeaturePlanesAgent(
        config=config, device=config.device if config.device != "auto" else "cpu", seed=seed
    ),
    "queue-planes": lambda config, seed: QueuePlanesAgent(
        config=config, device=config.device if config.device != "auto" else "cpu", seed=seed
    ),
    "feature-planes": lambda config, seed: FeaturePlanesAgent(
        config=config, device=config.device if config.device != "auto" else "cpu", seed=seed
    ),
    "n-step": lambda config, seed: NStepAgent(
        config=config, device=config.device if config.device != "auto" else "cpu", seed=seed
    ),
    "dueling": lambda config, seed: DuelingAgent(
        config=config, device=config.device if config.device != "auto" else "cpu", seed=seed
    ),
    "prioritized": lambda config, seed: PrioritizedAgent(
        config=config, device=config.device if config.device != "auto" else "cpu", seed=seed
    ),
}


def build_variant_agent(
    variant: Optional[str], config: TrainingConfig, seed: int
) -> Optional[DQNAgent]:
    """The agent a campaign should hand to `train()`, or None for the default.

    Returning `None` for ``dqn`` is not a detail: `train()` builds the reference
    agent (and seeds torch) itself, so the baseline arm goes through exactly the
    code path it always did.
    """
    name = "dqn" if variant is None else str(variant)
    if name not in VARIANTS:
        raise KeyError(f"unknown agent variant {name!r}; known: {sorted(VARIANTS)}")
    factory = VARIANTS[name]
    return None if factory is None else factory(config, seed)


__all__ = [
    "VARIANTS",
    "DoubleDQNAgent",
    "DuelingAgent",
    "DuelingFeaturePlanesAgent",
    "FeaturePlanesAgent",
    "NStepAgent",
    "PrioritizedAgent",
    "QueuePlanesAgent",
    "build_variant_agent",
]
