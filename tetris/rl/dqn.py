"""The DQN agent: what it does, and why it is shaped this way.

The problem being solved
------------------------

There is no labelled dataset saying which move is correct, so the agent has to
learn from *outcomes*. Reinforcement learning phrases that as: choose actions to
maximise the discounted sum of future rewards, ``G = r0 + gamma*r1 + gamma^2*r2
+ ...``. DQN learns an approximation of

    Q(s, a) = "how much total reward do I expect if I take action a in state s
               and then act sensibly for the rest of the game?"

and then plays by picking the legal action with the highest Q. Because the true
Q is unknown, the network is trained on its own estimates, bootstrapped one step
ahead -- the Bellman equation:

    Q(s, a)  ~=  r  +  gamma * max over legal a' of Q_target(s', a')

``r`` and ``s'`` come from the environment; the rest comes from a *target*
network: a frozen copy of the online network, refreshed every
`target_update_steps`. Without that copy the thing being predicted moves every
time the predictor moves, and training chases its own tail.

Why DQN fits this environment
-----------------------------

* The action space is discrete and small: 40 placements (`(rotation, column)`),
  of which 9-32 are legal for the piece in hand. That is exactly what a
  Q-network with one output per action needs.
* Actions are *legal or not*, not merely good or bad. The network outputs all 40
  Q-values and `action_mask` deletes the illegal ones, so a masked argmax can
  never select a move the environment would reject. The same mask appears in the
  Bellman target -- otherwise the agent would be told to expect reward it can
  never collect.
* One step is one piece, so credit assignment is short and episodes are only a
  few hundred steps: a laptop can run thousands of episodes.

What is deliberately left out (and would be the next steps, in order):
Double DQN (use the online net to pick a', the target net to score it),
n-step returns, prioritised replay, and a CNN trunk over the board. Each is a
small change to this file; none is needed to get a correct pipeline.

Terminal states
---------------

`done` is 1.0 only when the game is genuinely over: then the future is worth
nothing and the target is the reward itself. Hitting the `max_pieces` cap is a
*truncation*, not a terminal state -- the position at the cap really does have a
future, so it bootstraps normally and `done` stays 0. The one trap here is a
finished game: its mask is empty, and an empty mask would make the target
``gamma * max(nothing) = nan``, which poisons every weight in the network within
one batch. `LARGE_NEGATIVE` fills masked entries with a finite number and the
``(1 - done)`` factor removes the term, so nothing is ever NaN.
"""

from __future__ import annotations

import copy
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from ..board import WIDTH
from ..config import environment_info
from ..env import Placement, TetrisState
from .config import TrainingConfig
from .encoding import (
    ACTION_SIZE,
    STATE_SHAPE,
    encode_state,
    placement_for_index,
    index_for_placement,
)
from .network import QNetwork, build_network
from .registry import register
from .replay import Batch, ReplayBuffer

#: What a masked-out action's Q-value is replaced with. Finite on purpose: an
#: infinite fill times a zero `done` factor is a NaN.
LARGE_NEGATIVE = -1e9

#: Gradient clipping: DQN updates are bootstrapped, so one bad batch can produce
#: a huge gradient that undoes an hour of learning.
MAX_GRADIENT_NORM = 10.0

#: Bumped if the checkpoint layout ever changes.
CHECKPOINT_FORMAT = 1


@dataclass
class Decision:
    """What the agent did on one call, kept for the debug view and the meters.

    `nodes` is the work unit the experiment framework counts: for a DQN agent it
    is one network evaluation, which is what makes "boards per second" mean
    "network evaluations per second" rather than nothing at all.
    """

    action: int
    q_values: np.ndarray
    epsilon: float
    legal: int
    nodes: int = 1

    @property
    def best_q(self) -> float:
        return float(self.q_values[self.action])

    def __str__(self) -> str:
        return (
            f"AI selected:\n"
            f"  Rotation: {self.action // WIDTH}\n"
            f"  Column:   {self.action % WIDTH}\n"
            f"  Evaluation: {self.best_q:.4f}\n"
            f"  (Q-values from {self.legal} legal actions, epsilon {self.epsilon:.3f})"
        )


class DQNAgent:
    """Epsilon-greedy masked DQN with a target network and a replay buffer.

    ::

        agent = DQNAgent(config=TrainingConfig(episodes=200))
        action = agent.explore_action(state, mask, rng)     # during training
        loss = agent.train_step(buffer.sample(64, rng))     # one gradient step
        placement = agent.choose_action(state, valid)       # for evaluation

    `choose_action` is greedy -- it is the evaluation interface, so it never
    explores. Exploration belongs to `explore_action`, which the training loop
    drives with the schedule from the config.
    """

    name = "dqn"

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        network: Optional[QNetwork] = None,
        device: str = "cpu",
        seed: int = 0,
        learning_rate: Optional[float] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        self.config = TrainingConfig() if config is None else config
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)

        if network is None:
            network = build_network(hidden_sizes=self.config.hidden_sizes, seed=seed)
        self.online = network.to(self.device)
        self.target = copy.deepcopy(self.online).to(self.device)
        self.sync_target()

        rate = self.config.learning_rate if learning_rate is None else learning_rate
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=rate) \
            if optimizer is None else optimizer

        self.steps = 0            # gradient steps taken
        self.decisions = 0        # environment decisions taken
        self.episode = 0
        self.epsilon = float(self.config.epsilon_start)
        self.last_decision: Optional[Decision] = None

        #: What the network says about itself. Kept on the agent so that any
        #: network with a `forward` can stand in for the real one (tests do that).
        self.state_shape = tuple(getattr(self.online, "state_shape", STATE_SHAPE))
        self.action_size = int(getattr(self.online, "action_size", ACTION_SIZE))
        self.hidden_sizes = tuple(getattr(self.online, "hidden_sizes", ()))

    def adopt(self, config: TrainingConfig) -> None:
        """Follow a new config, keeping the weights, the optimizer state, the counters.

        This is what makes `--resume checkpoint.pt --episodes 300` work, and what
        keeps a resumed run honest when the hyper-parameters change: the agent's
        config-driven behaviour (gamma, target updates, the epsilon schedule) and
        the optimizer's learning rate come from the config you pass now, while
        the weights and the optimizer's own state come from the checkpoint.
        """
        self.config = config
        for group in self.optimizer.param_groups:
            group["lr"] = config.learning_rate
        self.epsilon = config.epsilon_for_step(self.decisions)

    # ------------------------------------------------------------ Q and acting

    def q_values(self, states: np.ndarray) -> np.ndarray:
        """Q-values for a state or a batch of states, as NumPy.

        Shapes: `(3, 20, 10)` -> `(40,)`, `(n, 3, 20, 10)` -> `(n, 40)`.
        No gradients are tracked: this is the acting path.
        """
        batch = np.asarray(states, dtype=np.float32)
        single = batch.ndim == len(self.state_shape)
        if single:
            batch = batch[None]
        with torch.no_grad():
            tensor = torch.as_tensor(batch, dtype=torch.float32, device=self.device)
            values = self.online(tensor).cpu().numpy()
        return values[0] if single else values

    @staticmethod
    def masked(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """`values` with everything outside `mask` set to `LARGE_NEGATIVE`."""
        masked = np.array(values, dtype=np.float32, copy=True)
        masked[~np.asarray(mask, dtype=bool)] = LARGE_NEGATIVE
        return masked

    def greedy_action(self, state: np.ndarray, mask: np.ndarray) -> int:
        """The legal action with the highest Q-value.

        Ties (including a brand-new network, where every Q-value is 0) go to the
        lowest index, so the choice is deterministic and testable.
        """
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            raise ValueError("no legal action: the game is over, call reset()")
        return int(np.argmax(self.masked(self.q_values(state), mask)))

    def explore_action(
        self,
        state: np.ndarray,
        mask: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> int:
        """Epsilon-greedy: with probability `epsilon` a random legal action.

        Exploration never picks an illegal action; that is the whole point of
        the mask, and it is why the agent spends its exploration on real moves.
        """
        generator = self.rng if rng is None else rng
        mask = np.asarray(mask, dtype=bool)
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            raise ValueError("no legal action: the game is over, call reset()")
        if float(self.epsilon) > 0 and generator.random() < float(self.epsilon):
            return int(generator.choice(legal))
        return self.greedy_action(state, mask)

    # ------------------------------------------------------------- the learning

    def targets_for(self, batch: Batch) -> np.ndarray:
        """The Bellman target `r + gamma * max_a' Q_target(s', a')` for a batch.

        The max runs over *legal* actions in `s'` only, and the whole future term
        vanishes when `done` is set -- so a terminal transition's target is just
        its reward.
        """
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(batch.next_masks, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            next_values = self.target(next_states).masked_fill(~next_masks, LARGE_NEGATIVE)
            best_next = next_values.max(dim=1).values
            targets = rewards + self.config.gamma * best_next * (1.0 - dones)
        return targets.cpu().numpy()

    def loss_for(self, batch: Batch) -> torch.Tensor:
        """The Huber loss between the network's Q(s, a) and the Bellman target.

        Huber (smooth L1) rather than squared error, as in the DQN paper: it
        behaves like L2 for small errors and like L1 for large ones, so a single
        wildly wrong target cannot dominate the batch.
        """
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=self.device)
        targets = torch.as_tensor(self.targets_for(batch), dtype=torch.float32,
                                  device=self.device)

        predicted = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        return nn.functional.smooth_l1_loss(predicted, targets)

    def train_step(self, batch: Batch) -> float:
        """One gradient step on one batch. Returns the loss (a float, for logging).

        Also advances the step counter and refreshes the target network on
        schedule -- the target is only ever synced here, never during acting.
        """
        loss = self.loss_for(batch)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), MAX_GRADIENT_NORM)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.config.target_update_steps == 0:
            self.sync_target()
        return float(loss.item())

    def sync_target(self) -> None:
        """Copy the online weights into the target network."""
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

    # --------------------------------------------------------------- the seams
    #
    # The training loop does not touch the encoder or the buffer directly: it
    # goes through these six methods. That is what lets an experiment arm change
    # *what a state is* (extra planes), *what a transition is* (n-step returns),
    # *which transitions come back* (prioritised replay) and *what a batch costs*
    # (importance weights) without a second copy of the loop that could drift
    # from this one.
    #
    # Every default is exactly the call the loop used to make, so the reference
    # agent trains and evaluates as it always did -- `tests/test_rl_seams.py`
    # asserts each one against the direct call it replaces, and the existing
    # determinism test covers the loop end to end.

    def encode(self, state: TetrisState) -> np.ndarray:
        """The planes this agent acts and trains on. Default: `encode_state`."""
        return encode_state(state)

    def build_buffer(self, config: TrainingConfig) -> ReplayBuffer:
        """The replay buffer to fill. Default: uniform, sized by the config."""
        return ReplayBuffer(
            capacity=config.buffer_size,
            state_shape=self.state_shape,
            action_size=self.action_size,
        )

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
        """Store one transition. Default: straight into the buffer, one step."""
        buffer.add(state, action, reward, next_state, next_mask, done)

    def end_episode(self, buffer) -> None:
        """The episode just ended (game over *or* the piece cap). Default: nothing.

        The default agent has no state that outlives a step, so there is nothing
        to do; an agent that accumulates across steps (n-step returns) flushes
        here. Both the game-over exit and the cap exit call it, because a
        transition must never be built across two episodes.
        """
        return None

    def sample_minibatch(self, buffer, rng: Optional[np.random.Generator] = None) -> Batch:
        """The batch for one gradient step. Default: a uniform sample."""
        return buffer.sample(self.config.batch_size, rng)

    def variant_params(self) -> Dict[str, Any]:
        """The knobs this agent's arm set, for the checkpoint's own record.

        Empty for the reference agent. A variant reports what makes it different
        (the n in n-step, alpha and beta, the plane count) so a checkpoint can be
        read years later without guessing which code produced it.
        """
        return {}

    # ------------------------------------------------------------- the schedule

    def note_decision(self) -> None:
        """Count one environment decision and re-derive epsilon from the config."""
        self.decisions += 1
        self.epsilon = self.config.epsilon_for_step(self.decisions)

    def set_epsilon(self, value: float) -> None:
        """Force a fixed exploration rate (0.0 for pure exploitation)."""
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {value!r}")
        self.epsilon = float(value)

    # ------------------------------------------------------- the agent protocol

    def choose_action(self, state, valid_actions: List[Placement]) -> Placement:
        """Greedily pick one of `valid_actions` -- the evaluation interface.

        `valid_actions` is what the environment offers (`env.get_valid_actions()`);
        building the mask from it rather than from the piece type keeps the agent
        honest about the environment's own list. The two agree -- there is a test
        for it -- so the arm that is short-circuited here is unreachable in
        practice.
        """
        mask = np.zeros(ACTION_SIZE, dtype=bool)
        for placement in valid_actions:
            mask[index_for_placement(placement)] = True
        if not mask.any():                       # no piece, or a finished game
            return valid_actions[0]

        encoded = self.encode(state)
        values = self.q_values(encoded)
        action = int(np.argmax(self.masked(values, mask)))
        self.last_decision = Decision(
            action=action, q_values=values, epsilon=self.epsilon, legal=int(mask.sum())
        )
        return placement_for_index(action)

    # ------------------------------------------------------------- checkpoints

    def save(self, path, **extra: Any) -> str:
        """Write a checkpoint that is enough to resume training exactly.

        Weights, optimizer state, step counters, epsilon, the config, the RNG
        state of replay sampling, and what produced it -- everything needed to
        continue as if the run had never stopped.
        """
        target = pathlib.Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "format": CHECKPOINT_FORMAT,
            "agent": self.name,
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "config_id": self.config.config_id,
            "state_shape": self.state_shape,
            "action_size": self.action_size,
            "steps": self.steps,
            "decisions": self.decisions,
            "episode": self.episode,
            "epsilon": self.epsilon,
            "rng_state": self.rng.bit_generator.state,
            "environment": dict(environment_info(), torch_version=str(torch.__version__)),
        }
        params = self.variant_params()
        if params:
            payload["variant_params"] = params
        payload.update(extra)
        torch.save(payload, target)
        return str(target)

    @classmethod
    def load(cls, path, device: str = "cpu") -> "DQNAgent":
        """Rebuild the agent that `save` wrote, ready to keep training or to play."""
        payload = torch.load(pathlib.Path(path), map_location="cpu")
        if payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(
                f"{path} is checkpoint format {payload.get('format')!r}, "
                f"this code writes {CHECKPOINT_FORMAT}"
            )
        config = TrainingConfig.from_dict(payload["config"])
        agent = cls(config=config, device=device, seed=int(config.seed or 0))
        agent.online.load_state_dict(payload["online"])
        agent.target.load_state_dict(payload.get("target", payload["online"]))
        agent.online.to(agent.device)
        agent.target.to(agent.device)
        if "optimizer" in payload:
            agent.optimizer.load_state_dict(payload["optimizer"])
        agent.steps = int(payload.get("steps", 0))
        agent.decisions = int(payload.get("decisions", 0))
        agent.episode = int(payload.get("episode", 0))
        agent.epsilon = float(payload.get("epsilon", config.epsilon_start))
        if payload.get("rng_state") is not None:
            agent.rng.bit_generator.state = payload["rng_state"]
        return agent

    # -------------------------------------------------------------- inspection

    @property
    def parameter_count(self) -> int:
        return self.online.parameter_count

    def __repr__(self) -> str:
        return (
            f"<DQNAgent {self.state_shape} -> {list(self.hidden_sizes)} -> "
            f"{self.action_size}, {self.parameter_count} params, "
            f"steps={self.steps}, epsilon={self.epsilon:.3f}>"
        )


def load_agent_for_evaluation(path, device: str = "cpu") -> DQNAgent:
    """A saved agent in playing mode: greedy, and told to stop exploring.

    The class comes from the checkpoint's own `agent` field (see
    `tetris.rl.registry`), not from this function: a checkpoint written by an
    experiment arm has that arm's network and encoder, and rebuilding it as a
    plain `DQNAgent` would either fail on the weights or -- worse, in the
    feature-plane case -- load a network whose input does not match the state it
    is handed.
    """
    from .registry import load_agent

    agent = load_agent(path, device=device)
    agent.set_epsilon(0.0)
    return agent


register(DQNAgent)

__all__ = [
    "CHECKPOINT_FORMAT",
    "LARGE_NEGATIVE",
    "MAX_GRADIENT_NORM",
    "Decision",
    "DQNAgent",
    "load_agent_for_evaluation",
]
