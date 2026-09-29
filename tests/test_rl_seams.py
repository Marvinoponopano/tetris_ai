"""Tests for the training loop's seams (`DQNAgent.encode` and friends).

The loop used to call `encode_state`, `ReplayBuffer(...)`, `buffer.add(...)` and
`buffer.sample(...)` directly. It now asks the agent for each of those, so that
an experiment arm can change what a state is, what a transition is, or how a batch
is chosen -- without a second copy of the loop that could drift from this one.

The whole point of that refactor is that it changes *nothing* for the reference
agent, so that is what is pinned here: each seam is asserted against the exact
call it replaced, and a short training run is asserted to be reproducible. The
campaign log from before the change is compared against in
`~/.hermes/cache/scratch/seam_regression.py`, which replays 250 episodes of G and
diffs every field of every episode row.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tetris.env import TetrisEnv  # noqa: E402
from tetris.rl.config import TrainingConfig  # noqa: E402
from tetris.rl.dqn import DQNAgent  # noqa: E402
from tetris.rl.encoding import encode_state  # noqa: E402
from tetris.rl.replay import ReplayBuffer  # noqa: E402
from tetris.rl.training import train  # noqa: E402


def config(**overrides) -> TrainingConfig:
    return TrainingConfig.from_dict(
        {"episodes": 2, "max_pieces": 6, "batch_size": 4, "hidden_sizes": (8,),
         "min_buffer_size": 2, "buffer_size": 32, "seed": 0, **overrides}
    )


def a_state(seed: int = 0):
    env = TetrisEnv(seed=seed)
    state = env.reset(seed=seed)
    return env, state


# ------------------------------------------------------------------- encoding


def test_encode_is_the_encoding_the_loop_used_to_call():
    agent = DQNAgent(config=config(), seed=0)
    _, state = a_state(3)
    assert np.array_equal(agent.encode(state), encode_state(state))


def test_the_acting_path_uses_the_agents_own_encoder():
    """A variant's `encode` must be what `choose_action` sees, not the default."""
    agent = DQNAgent(config=config(), seed=0)
    seen = {}

    def spy(state):
        seen["called"] = True
        return encode_state(state)

    agent.encode = spy
    env, state = a_state(4)
    agent.choose_action(state, env.get_valid_actions())
    assert seen.get("called"), "choose_action ignored the agent's encoder"


# -------------------------------------------------------------------- buffers


def test_build_buffer_is_the_buffer_the_loop_used_to_build():
    agent = DQNAgent(config=config(buffer_size=17), seed=0)
    buffer = agent.build_buffer(agent.config)
    assert isinstance(buffer, ReplayBuffer)
    assert buffer.capacity == 17
    assert buffer.states.shape == (17, 3, 20, 10)
    assert buffer.next_masks.shape == (17, 40)


def test_remember_stores_exactly_what_the_buffer_add_did():
    agent = DQNAgent(config=config(), seed=0)
    mine = agent.build_buffer(agent.config)
    theirs = ReplayBuffer(capacity=agent.config.buffer_size)
    rng = np.random.default_rng(0)
    for index in range(5):
        state = rng.integers(0, 2, size=(3, 20, 10)).astype(np.float32)
        mask = np.zeros(40, dtype=bool)
        mask[:9] = True
        agent.remember(mine, state, index, float(index), state, mask, index % 3 == 0)
        theirs.add(state, index, float(index), state, mask, index % 3 == 0)

    assert (mine.states == theirs.states).all()
    assert (mine.actions == theirs.actions).all()
    assert (mine.rewards == theirs.rewards).all()
    assert (mine.dones == theirs.dones).all()
    assert (mine.next_masks == theirs.next_masks).all()
    assert len(mine) == len(theirs) == 5


def test_sample_minibatch_is_the_uniform_sample_it_replaced():
    agent = DQNAgent(config=config(batch_size=6), seed=0)
    buffer = agent.build_buffer(agent.config)
    rng = np.random.default_rng(0)
    for index in range(12):
        state = rng.integers(0, 2, size=(3, 20, 10)).astype(np.float32)
        mask = np.zeros(40, dtype=bool)
        mask[:9] = True
        buffer.add(state, index % 40, float(index), state, mask, False)

    mine = agent.sample_minibatch(buffer, np.random.default_rng(1))
    theirs = buffer.sample(agent.config.batch_size, np.random.default_rng(1))
    assert len(mine) == len(theirs) == 6
    assert (mine.actions == theirs.actions).all()
    assert (mine.rewards == theirs.rewards).all()
    assert (mine.states == theirs.states).all()


def test_the_end_of_an_episode_is_a_no_op_for_the_reference_agent():
    agent = DQNAgent(config=config(), seed=0)
    buffer = agent.build_buffer(agent.config)
    rng = np.random.default_rng(0)
    state = rng.integers(0, 2, size=(3, 20, 10)).astype(np.float32)
    mask = np.zeros(40, dtype=bool)
    mask[:9] = True
    before = len(buffer)
    agent.end_episode(buffer)
    assert len(buffer) == before                     # nothing added, nothing dropped
    assert agent.window if hasattr(agent, "window") else True


def test_the_reference_agent_reports_no_variant_knobs():
    assert DQNAgent(config=config(), seed=0).variant_params() == {}


# -------------------------------------------------------------- the whole loop


def test_a_short_run_is_reproducible(tmp_path):
    """Same config twice -> the same episodes, through the whole loop."""
    first = train(config(), output_dir=tmp_path / "first", logger=None)
    second = train(config(), output_dir=tmp_path / "second", logger=None)
    fields = ("reward", "score", "lines", "pieces", "decisions", "learn_steps")
    for left, right in zip(first.episodes, second.episodes):
        for field in fields:
            assert left[field] == right[field], field


def test_the_loop_runs_every_kind_of_seam_shape(tmp_path):
    """A wide state and a non-uniform buffer both go through the same loop."""
    from tetris.rl.variants import build_variant_agent

    for arm in ("feature-planes", "n-step", "prioritized", "dueling"):
        agent = build_variant_agent(arm, config(), 0)
        result = train(config(), agent=agent, output_dir=tmp_path / arm, logger=None)
        assert result.episodes, arm
        assert all(row["pieces"] > 0 for row in result.episodes), arm
