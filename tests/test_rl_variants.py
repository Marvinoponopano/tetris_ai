"""Tests for the agent variants (`tetris/rl/variants.py`).

`DoubleDQNAgent` differs from `DQNAgent` in one line of the Bellman target, so
that is what is pinned here: the target is computed the Double-Q way, it really
is different from the plain max, and acting is untouched (which is what makes a
checkpoint from either variant loadable and comparable).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tetris.rl.config import TrainingConfig  # noqa: E402
from tetris.rl.dqn import DQNAgent  # noqa: E402
from tetris.rl.replay import ReplayBuffer  # noqa: E402
from tetris.rl.encoding import QUEUE_PLANE, encode_state_with_queue  # noqa: E402
from tetris.rl.variants import (  # noqa: E402
    VARIANTS,
    DoubleDQNAgent,
    build_variant_agent,
)


def tiny(**overrides) -> TrainingConfig:
    return TrainingConfig.from_dict(
        {"episodes": 1, "max_pieces": 5, "batch_size": 4, "hidden_sizes": (8,),
         "min_buffer_size": 2, "seed": 0, "gamma": 0.9, **overrides}
    )


def batch(size: int = 6, seed: int = 0):
    buffer = ReplayBuffer(capacity=32)
    rng = np.random.default_rng(seed)
    for index in range(16):
        state = rng.integers(0, 2, size=(3, 20, 10)).astype(np.float32)
        mask = np.zeros(40, dtype=bool)
        mask[rng.choice(40, size=6, replace=False)] = True
        buffer.add(state, int(rng.integers(0, 40)), float(rng.normal()),
                   rng.integers(0, 2, size=(3, 20, 10)).astype(np.float32), mask,
                   bool(index % 5 == 0))
    return buffer.sample(size, rng)


def pinned_target(agent, offset: float = 5.0) -> np.ndarray:
    """Replace the target network's head with a known, increasing value per action."""
    with torch.no_grad():
        last = agent.target.layers[-1]
        last.weight.zero_()
        last.bias.copy_(torch.arange(40, dtype=torch.float32) * offset / 40.0)
    return last.bias.detach().numpy().copy()


def test_the_double_target_is_the_value_of_the_action_the_online_net_picks():
    agent = DoubleDQNAgent(config=tiny(), seed=3)
    sample = batch()
    bias = pinned_target(agent)

    # the online network's choice (its own head is untouched, so use its Q-values)
    online = agent.online(torch.as_tensor(sample.next_states, dtype=torch.float32))
    masked = online.masked_fill(~torch.as_tensor(sample.next_masks, dtype=torch.bool), -1e9)
    chosen = masked.argmax(dim=1).numpy()
    expected = sample.rewards + 0.9 * bias[chosen] * (1.0 - sample.dones)

    assert np.allclose(agent.targets_for(sample), expected, atol=1e-6)


def test_the_double_target_is_not_the_plain_max_of_the_target_network():
    """If they were equal, the variant would be pointless -- so this must differ."""
    double = DoubleDQNAgent(config=tiny(), seed=3)
    vanilla = DQNAgent(config=tiny(), seed=3)
    sample = batch()
    bias = pinned_target(double)
    pinned_target(vanilla)

    legal = sample.next_masks
    plain_max = sample.rewards + 0.9 * np.where(legal, bias, -1e9).max(axis=1) * (1.0 - sample.dones)

    assert not np.allclose(double.targets_for(sample), plain_max)
    assert np.allclose(vanilla.targets_for(sample), plain_max, atol=1e-6)


def test_double_dqn_acts_exactly_like_the_reference_agent():
    """The variant changes the learning target, never the policy's interface."""
    config = tiny()
    double = DoubleDQNAgent(config=config, seed=4)
    vanilla = DQNAgent(config=config, seed=4)
    vanilla.online.load_state_dict(double.online.state_dict())
    vanilla.sync_target()

    rng = np.random.default_rng(0)
    for _ in range(20):
        state = rng.integers(0, 2, size=(3, 20, 10)).astype(np.float32)
        mask = rng.random(40) < 0.3
        assert double.greedy_action(state, mask) == vanilla.greedy_action(state, mask)


def test_a_double_dqn_checkpoint_reloads_as_a_plain_agent(tmp_path):
    """So the comparison against the baselines never depends on the variant."""
    agent = DoubleDQNAgent(config=tiny(), seed=1)
    path = agent.save(tmp_path / "checkpoint.pt")

    loaded = DQNAgent.load(path)
    assert loaded.config == agent.config
    assert isinstance(loaded, DQNAgent) and not isinstance(loaded, DoubleDQNAgent)


def test_the_registry_builds_the_agent_that_was_asked_for():
    config = tiny()

    assert build_variant_agent("dqn", config, 0) is None      # the default path
    assert build_variant_agent(None, config, 0) is None
    built = build_variant_agent("double-dqn", config, 7)
    assert isinstance(built, DoubleDQNAgent) and built.name == "double-dqn"


def test_an_unknown_variant_is_rejected():
    with pytest.raises(KeyError, match="unknown agent variant"):
        build_variant_agent("triple-dqn", tiny(), 0)


def test_the_declared_variants_are_the_ones_that_build():
    for name in VARIANTS:
        agent = build_variant_agent(name, tiny(), 0)
        assert agent is None or isinstance(agent, DQNAgent)


# ------------------------------------------------------- the round-2 arm agents


def test_every_arm_is_registered_under_its_own_name():
    from tetris.rl.registry import REGISTRY

    for name in VARIANTS:
        assert name in REGISTRY, f"{name} is not registered"
        assert REGISTRY[name].name == name


def test_the_feature_arm_encodes_five_planes():
    from tetris.env import TetrisEnv
    from tetris.rl.encoding import EXTENDED_SHAPE, encode_state_with_features

    agent = build_variant_agent("feature-planes", tiny(), 0)
    env = TetrisEnv(seed=5)
    state = env.reset(seed=5)
    assert agent.state_shape == EXTENDED_SHAPE
    assert np.array_equal(agent.encode(state), encode_state_with_features(state))
    # the network, the buffer and the agent all agree on the wider state
    assert agent.online.state_shape == EXTENDED_SHAPE
    assert agent.build_buffer(agent.config).states.shape[1:] == EXTENDED_SHAPE
    assert agent.variant_params()["state_shape"] == list(EXTENDED_SHAPE)


def test_the_dueling_arm_uses_a_dueling_network():
    from tetris.rl.network import DuelingQNetwork

    agent = build_variant_agent("dueling", tiny(), 0)
    assert isinstance(agent.online, DuelingQNetwork)
    assert isinstance(agent.target, DuelingQNetwork)
    assert agent.state_shape == (3, 20, 10)              # same input as the reference
    assert agent.variant_params()["head"] == "dueling"


def test_the_dueling_arm_acts_like_any_other_agent():
    from tetris.env import TetrisEnv

    agent = build_variant_agent("dueling", tiny(), 0)
    env = TetrisEnv(seed=6)
    state = env.reset(seed=6)
    for _ in range(5):
        placement = agent.choose_action(state, env.get_valid_actions())
        assert placement in env.get_valid_actions()
        state, _reward, done, _info = env.step(placement)
        if done:
            break


def test_the_nstep_arm_rewrites_the_transitions_it_stores():
    agent = build_variant_agent("n-step", tiny(), 0)
    buffer = agent.build_buffer(agent.config)
    state = np.zeros((3, 20, 10), dtype=np.float32)
    mask = np.zeros(40, dtype=bool)
    mask[:9] = True

    # fewer than n pushes: nothing is stored yet
    for index in range(agent.n_step - 1):
        agent.remember(buffer, state, index, 1.0, state, mask, done=False)
    assert len(buffer) == 0

    agent.remember(buffer, state, agent.n_step, 1.0, state, mask, done=False)
    assert len(buffer) == 1                              # the window filled up
    # the stored reward is the 3-step return with the run's own gamma
    expected = sum(agent.config.gamma ** step for step in range(agent.n_step))
    assert buffer.rewards[0] == pytest.approx(expected, rel=1e-5)


def test_the_nstep_arm_empties_its_window_at_the_end_of_an_episode():
    agent = build_variant_agent("n-step", tiny(), 0)
    buffer = agent.build_buffer(agent.config)
    state = np.zeros((3, 20, 10), dtype=np.float32)
    mask = np.zeros(40, dtype=bool)
    mask[:9] = True
    agent.remember(buffer, state, 1, 1.0, state, mask, done=False)
    assert len(agent.window) == 1

    agent.end_episode(buffer)                            # the piece cap, or a game over
    assert len(agent.window) == 0
    assert len(buffer) == 1                              # the tail was still learned from


def test_the_nstep_arm_stores_a_transition_when_the_game_ends():
    agent = build_variant_agent("n-step", tiny(), 0)
    buffer = agent.build_buffer(agent.config)
    state = np.zeros((3, 20, 10), dtype=np.float32)
    mask = np.zeros(40, dtype=bool)
    mask[:9] = True
    agent.remember(buffer, state, 1, 1.0, state, mask, done=True)
    assert len(buffer) == 1
    assert buffer.dones[0] == 1.0                        # and it is marked terminal
    assert len(agent.window) == 0


def test_the_prioritized_arm_builds_a_prioritized_buffer():
    from tetris.rl.prioritized import PrioritizedReplayBuffer

    agent = build_variant_agent("prioritized", tiny(), 0)
    buffer = agent.build_buffer(agent.config)
    assert isinstance(buffer, PrioritizedReplayBuffer)
    assert buffer.capacity == agent.config.buffer_size
    assert buffer.alpha == agent.alpha and buffer.beta == agent.beta_start


def test_the_prioritized_arms_beta_anneals_across_the_run():
    agent = build_variant_agent("prioritized", tiny(episodes=100), 0)
    agent.episode = 0
    start = agent.beta()
    agent.episode = 50
    middle = agent.beta()
    agent.episode = 100
    end = agent.beta()
    assert start == pytest.approx(agent.beta_start)
    assert start < middle < end
    assert end == pytest.approx(1.0)
    assert 0.0 <= start <= 1.0


def test_a_prioritized_step_moves_the_priorities_it_sampled():
    agent = build_variant_agent("prioritized", tiny(min_buffer_size=2, batch_size=4), 0)
    buffer = agent.build_buffer(agent.config)
    rng = np.random.default_rng(0)
    for index in range(8):
        state = rng.integers(0, 2, size=(3, 20, 10)).astype(np.float32)
        mask = np.zeros(40, dtype=bool)
        mask[:9] = True
        agent.remember(buffer, state, index, 1.0, state, mask, done=False)
    before = [buffer._priority(slot) for slot in range(8)]

    batch = agent.sample_minibatch(buffer, np.random.default_rng(1))
    loss = agent.train_step(batch)

    assert np.isfinite(loss)
    after = [buffer._priority(slot) for slot in range(8)]
    assert after != before, "the update did not feed back into the priorities"
    assert agent.steps == 1


def test_the_prioritized_arm_weights_its_loss_by_importance():
    """With every weight equal to 1 the loss is the plain mean Huber loss."""
    agent = build_variant_agent("prioritized", tiny(min_buffer_size=2, batch_size=3), 0)
    buffer = agent.build_buffer(agent.config)
    rng = np.random.default_rng(0)
    for index in range(6):
        state = rng.integers(0, 2, size=(3, 20, 10)).astype(np.float32)
        mask = np.zeros(40, dtype=bool)
        mask[:9] = True
        agent.remember(buffer, state, index, 1.0, state, mask, done=False)

    batch, slots, weights = buffer.sample(3, np.random.default_rng(2), beta=0.0)
    # measured before the step: `train_step` moves the weights it is scored on
    plain = float(agent.loss_for(batch).detach())
    agent._pending = (slots, np.ones(3, dtype=np.float32))
    weighted = agent.train_step(batch)
    assert weighted == pytest.approx(plain, rel=1e-5)


def test_a_variant_checkpoint_comes_back_as_the_same_arm(tmp_path):
    """The registry path the comparison table and the viewer use."""
    from tetris.env import TetrisEnv
    from tetris.rl.dqn import load_agent_for_evaluation
    from tetris.rl.registry import checkpoint_agent_name

    def play(agent, games=2):
        moves = []
        for offset in range(games):
            env = TetrisEnv(seed=77 + offset)
            state = env.reset(seed=77 + offset)
            while True:
                valid = env.get_valid_actions()
                if not valid:
                    break
                placement = agent.choose_action(state, valid)
                moves.append((int(placement.rotation), int(placement.column)))
                state, _reward, done, info = env.step(placement)
                if done or info["pieces_placed"] >= 30:
                    break
        return moves

    for arm in ("feature-planes", "n-step", "dueling", "prioritized"):
        agent = build_variant_agent(arm, tiny(episodes=1, max_pieces=6), 0)
        path = agent.save(tmp_path / f"{arm}.pt")
        assert checkpoint_agent_name(path) == arm        # the file names its arm
        reloaded = load_agent_for_evaluation(path)
        assert type(reloaded) is type(agent)
        assert reloaded.state_shape == agent.state_shape
        assert reloaded.epsilon == 0.0                   # it is in playing mode
        assert play(reloaded) == play(agent), arm


def test_the_summary_of_every_arm_says_what_it_changed():
    for name in ("feature-planes", "n-step", "dueling", "prioritized"):
        params = build_variant_agent(name, tiny(), 0).variant_params()
        assert params, f"{name} reports nothing about itself"
        assert all(isinstance(key, str) for key in params)


def test_the_queue_arm_sees_the_piece_after_next():
    from tetris.env import TetrisEnv
    from tetris.rl.encoding import QUEUE_SHAPE

    agent = build_variant_agent("queue-planes", tiny(), 0)
    env = TetrisEnv(seed=21)
    state = env.reset(seed=21)
    assert agent.state_shape == QUEUE_SHAPE
    assert np.array_equal(agent.encode(state),
                          encode_state_with_queue(state))
    # the plane holds exactly the tetromino the queue says comes second
    plane = agent.encode(state)[QUEUE_PLANE]
    assert plane.sum() == 4.0                       # every tetromino is four cells
    assert agent.variant_params()["state_shape"] == list(QUEUE_SHAPE)


def test_the_queue_arm_contains_the_feature_encoding():
    from tetris.env import TetrisEnv
    from tetris.rl.encoding import encode_state_with_features

    agent = build_variant_agent("queue-planes", tiny(), 0)
    env = TetrisEnv(seed=22)
    state = env.reset(seed=22)
    for _ in range(6):
        state, _reward, done, _info = env.step(env.get_valid_actions()[0])
        if done:
            break
    assert np.array_equal(agent.encode(state)[:5], encode_state_with_features(state))


def test_the_queue_plane_is_empty_when_there_is_no_queue():
    from tetris.rl.encoding import encode_state_with_queue

    class NoQueue:
        board = np.zeros((20, 10), dtype=np.int32)
        current_piece = np.zeros((0, 2), dtype=np.int64)
        current_piece_type = np.zeros(7, dtype=np.float32)
        next_piece = np.zeros((0, 2), dtype=np.int64)
        next_piece_type = np.zeros(7, dtype=np.float32)

    planes = encode_state_with_queue(NoQueue())
    assert planes.shape[0] == 6
    assert planes[QUEUE_PLANE].sum() == 0.0          # nothing known, nothing drawn


def test_the_combination_arm_is_both_changes_at_once():
    """M is dueling *and* five planes -- the config is G's, the agent is not."""
    from tetris.rl.encoding import EXTENDED_SHAPE
    from tetris.rl.network import DuelingQNetwork

    config = tiny()
    agent = build_variant_agent("dueling-feature-planes", config, seed=0)

    assert agent.name == "dueling-feature-planes"
    assert agent.state_shape == EXTENDED_SHAPE
    assert isinstance(agent.online, DuelingQNetwork)
    assert isinstance(agent.target, DuelingQNetwork)
    assert agent.variant_params()["head"] == "dueling"
    assert agent.variant_params()["state_shape"] == list(EXTENDED_SHAPE)
    # and it carries both parents' *function*, not a third architecture: the
    # dueling head's parameter count, on the five-plane input
    dueling = build_variant_agent("dueling", config, seed=0)
    planes = build_variant_agent("feature-planes", config, seed=0)
    reference = DQNAgent(config=config, seed=0)
    assert agent.online.parameter_count > dueling.online.parameter_count
    assert agent.online.parameter_count > planes.online.parameter_count
    # the dueling head adds the value head -- `hidden_last + 1` weights -- and it
    # adds the same number whatever the input planes are, which is what makes the
    # two changes independent rather than multiplicative
    assert agent.online.parameter_count - planes.online.parameter_count == (
        dueling.online.parameter_count - reference.online.parameter_count)
