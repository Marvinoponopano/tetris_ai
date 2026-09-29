"""Tests for the DQN agent: the Bellman target, the loss, the target network,
exploration, the evaluation interface and checkpoints.

These are the tests that decide whether the *learning* is right, as opposed to
merely running. The Bellman target and the loss are checked against values
computed by hand (`test_the_bellman_target_is_the_hand_computed_one`,
`test_the_loss_is_the_hand_computed_huber_loss`), not against a previous run of
the same code.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from tetris.env import Placement, TetrisEnv  # noqa: E402
from tetris.rl.config import TrainingConfig  # noqa: E402
from tetris.rl.dqn import (  # noqa: E402
    DQNAgent,
    LARGE_NEGATIVE,
    load_agent_for_evaluation,
)
from tetris.rl.encoding import (  # noqa: E402
    ACTION_SIZE,
    STATE_SHAPE,
    encode_state,
    mask_for_state,
    placement_for_index,
)
from tetris.rl.replay import Batch, ReplayBuffer  # noqa: E402

SMALL = dict(hidden_sizes=(16,), batch_size=4, min_buffer_size=4)


def a_batch(rewards=(1.0, 2.0, 3.0), dones=(0.0, 0.0, 1.0), actions=(0, 5, 39)):
    """A tiny batch with all-zero states, so only the numbers matter."""
    size = len(rewards)
    masks = np.zeros((size, ACTION_SIZE), dtype=bool)
    masks[:, :10] = True                     # columns 0-9 legal in the next state
    return Batch(
        states=np.zeros((size,) + STATE_SHAPE, dtype=np.float32),
        actions=np.array(actions, dtype=np.int64),
        rewards=np.array(rewards, dtype=np.float32),
        next_states=np.zeros((size,) + STATE_SHAPE, dtype=np.float32),
        next_masks=masks,
        dones=np.array(dones, dtype=np.float32),
    )


class FixedNetwork(nn.Module):
    """A stand-in whose `forward` returns a table of Q-values, row by row."""

    def __init__(self, table):
        super().__init__()
        self.table = torch.as_tensor(np.asarray(table, dtype=np.float32))
        self.register_buffer("bias", torch.zeros(1))

    def forward(self, states):
        return self.table[: len(states)].clone()


def agent_with_target(table, **config):
    agent = DQNAgent(config=TrainingConfig(**{**SMALL, **config}), seed=0)
    agent.target = FixedNetwork(table)
    return agent


# --------------------------------------------------------------- the Bellman step


def test_the_bellman_target_is_the_hand_computed_one():
    """r + gamma * max_a' Q_target(s', a'), and nothing else."""
    gamma = 0.5
    table = np.zeros((3, ACTION_SIZE), dtype=np.float32)
    table[0, 3] = 4.0                       # the best legal action of transition 0
    table[1, 7] = 8.0                       # ... of transition 1
    table[2, 1] = 100.0                     # ... of transition 2, which is terminal
    agent = agent_with_target(table, gamma=gamma)

    targets = agent.targets_for(a_batch(rewards=(1.0, 2.0, 3.0), dones=(0.0, 0.0, 1.0)))

    assert targets == pytest.approx([1.0 + gamma * 4.0, 2.0 + gamma * 8.0, 3.0], abs=1e-5)


def test_an_illegal_action_is_never_the_targets_max():
    """A huge Q-value on a placement that does not exist must be ignored."""
    table = np.zeros((1, ACTION_SIZE), dtype=np.float32)
    table[0, 3] = 4.0                        # legal (column 3)
    table[0, 35] = 1e6                       # illegal: rotation 3 is masked out
    agent = agent_with_target(table, gamma=0.99)

    batch = a_batch(rewards=(0.0,), dones=(0.0,), actions=(3,))
    batch.next_masks[:] = False
    batch.next_masks[0, :10] = True          # only columns 0-9 are legal

    assert agent.targets_for(batch)[0] == pytest.approx(0.99 * 4.0, abs=1e-4)


def test_a_terminal_transition_is_worth_its_reward_alone():
    table = np.full((1, ACTION_SIZE), 1000.0, dtype=np.float32)
    agent = agent_with_target(table, gamma=0.99)
    batch = a_batch(rewards=(7.5,), dones=(1.0,), actions=(0,))

    assert agent.targets_for(batch)[0] == pytest.approx(7.5, abs=1e-5)


def test_a_terminal_state_with_no_legal_actions_does_not_produce_nan():
    """The trap: an empty mask would make gamma * max(nothing) = nan and poison the run."""
    table = np.zeros((2, ACTION_SIZE), dtype=np.float32)
    table[:, :10] = 5.0
    agent = agent_with_target(table, gamma=0.99)

    batch = a_batch(rewards=(1.0, 2.0), dones=(1.0, 1.0), actions=(0, 1))
    batch.next_masks[:] = False                     # no legal action anywhere

    targets = agent.targets_for(batch)

    assert np.all(np.isfinite(targets))
    assert targets == pytest.approx([1.0, 2.0], abs=1e-5)


def test_the_target_network_is_the_one_that_scores_the_future():
    """Changing the online network must not move a target until a sync."""
    agent = DQNAgent(config=TrainingConfig(**SMALL), seed=1)
    batch = a_batch()
    before = agent.targets_for(batch)

    with torch.no_grad():
        for parameter in agent.online.parameters():
            parameter.add_(1.0)

    assert np.allclose(agent.targets_for(batch), before)


# -------------------------------------------------------------------- the loss


def test_the_loss_is_the_hand_computed_huber_loss():
    """Huber: 0.5*d^2 for |d| < 1, |d| - 0.5 beyond it, averaged over the batch."""
    agent = DQNAgent(config=TrainingConfig(**SMALL), seed=0)
    table = np.zeros((3, ACTION_SIZE), dtype=np.float32)     # every transition is terminal
    agent.target = FixedNetwork(table)
    predictions = np.zeros((3, ACTION_SIZE), dtype=np.float32)
    predictions[0, 0] = 0.5          # d = 0.5  -> 0.5 * 0.25 = 0.125
    predictions[1, 5] = 4.0          # d = 4.0  -> 4.0 - 0.5  = 3.5
    predictions[2, 39] = -1.0        # d = -1.0 -> 1.0 - 0.5  = 0.5
    agent.online = FixedNetwork(predictions)
    batch = a_batch(rewards=(0.0, 0.0, 0.0), dones=(1.0, 1.0, 1.0), actions=(0, 5, 39))

    expected = (0.125 + 3.5 + 0.5) / 3
    assert float(agent.loss_for(batch)) == pytest.approx(expected, abs=1e-6)


def test_the_loss_backpropagates_and_moves_the_weights():
    agent = DQNAgent(config=TrainingConfig(**SMALL, learning_rate=0.1), seed=2)
    before = [parameter.detach().clone() for parameter in agent.online.parameters()]

    loss = agent.train_step(a_batch())

    assert loss > 0
    assert any(not torch.equal(was, now) for was, now in
               zip(before, agent.online.parameters()))


def test_learning_reduces_the_loss_on_a_repeated_transition():
    """The pipeline learns *something*: fitting one transition should get easy."""
    agent = DQNAgent(config=TrainingConfig(**SMALL, learning_rate=0.05), seed=3)
    batch = a_batch(rewards=(5.0,), dones=(1.0,), actions=(4,))
    batch.next_masks[:] = True

    first = agent.train_step(batch)          # Huber of a 5.0 error = 5.0 - 0.5
    assert first == pytest.approx(4.5, abs=1e-5)
    for _ in range(299):
        last = agent.train_step(batch)

    assert last < first / 10, (first, last)
    assert last < 0.05, (first, last)
    # the Q-value that gets backed up is the reward it was told to expect
    assert agent.q_values(batch.states)[0, 4] == pytest.approx(5.0, abs=0.1)


def test_train_step_counts_and_syncs_the_target_on_schedule():
    agent = DQNAgent(config=TrainingConfig(**SMALL, target_update_steps=3), seed=4)
    batch = a_batch()

    with torch.no_grad():
        for parameter in agent.online.parameters():
            parameter.add_(0.5)                  # online now differs from target

    agent.train_step(batch)
    assert agent.steps == 1
    online = [p.detach().clone() for p in agent.online.parameters()]
    target = [p.detach().clone() for p in agent.target.parameters()]
    assert any(not torch.equal(a, b) for a, b in zip(online, target))   # no sync yet

    agent.train_step(batch)
    agent.train_step(batch)                      # the third step triggers the sync
    assert agent.steps == 3
    for a, b in zip(agent.online.parameters(), agent.target.parameters()):
        assert torch.equal(a.detach(), b.detach())


def test_the_optimizer_is_the_one_configured():
    agent = DQNAgent(config=TrainingConfig(**SMALL, learning_rate=0.007), seed=5)
    rate = agent.optimizer.param_groups[0]["lr"]
    assert rate == pytest.approx(0.007)


# -------------------------------------------------------------- acting policies


def test_q_values_accept_one_state_or_a_batch():
    agent = DQNAgent(config=TrainingConfig(**SMALL), seed=6)
    single = np.zeros(STATE_SHAPE, dtype=np.float32)
    assert agent.q_values(single).shape == (ACTION_SIZE,)
    assert agent.q_values(np.stack([single, single])).shape == (2, ACTION_SIZE)


def test_masking_removes_illegal_actions_from_the_choice():
    table = np.zeros((1, ACTION_SIZE), dtype=np.float32)
    table[0, 39] = 99.0                          # best overall, illegal for this mask
    table[0, 2] = 1.0
    agent = agent_with_target(table)
    agent.online = FixedNetwork(table)
    mask = np.zeros(ACTION_SIZE, dtype=bool)
    mask[0:10] = True

    assert agent.greedy_action(np.zeros(STATE_SHAPE, dtype=np.float32), mask) == 2


def test_a_mask_with_nothing_in_it_is_an_error_not_a_crash():
    agent = DQNAgent(config=TrainingConfig(**SMALL), seed=7)
    empty = np.zeros(ACTION_SIZE, dtype=bool)
    with pytest.raises(ValueError, match="no legal action"):
        agent.greedy_action(np.zeros(STATE_SHAPE, dtype=np.float32), empty)
    with pytest.raises(ValueError, match="no legal action"):
        agent.explore_action(np.zeros(STATE_SHAPE, dtype=np.float32), empty)


def test_exploration_only_ever_picks_legal_actions():
    agent = DQNAgent(config=TrainingConfig(**SMALL), seed=8)
    agent.set_epsilon(1.0)
    rng = np.random.default_rng(0)
    mask = np.zeros(ACTION_SIZE, dtype=bool)
    mask[[3, 17, 30]] = True

    picks = [agent.explore_action(np.zeros(STATE_SHAPE, dtype=np.float32), mask, rng)
             for _ in range(200)]

    assert set(picks) == {3, 17, 30}
    assert len(set(picks)) == 3                 # it really is exploring


def test_a_zero_epsilon_agent_always_plays_its_best_action():
    agent = DQNAgent(config=TrainingConfig(**SMALL), seed=9)
    agent.set_epsilon(0.0)
    table = np.zeros((1, ACTION_SIZE), dtype=np.float32)
    table[0, 4] = 1.0
    agent.online = FixedNetwork(table)
    mask = np.zeros(ACTION_SIZE, dtype=bool)
    mask[0:10] = True
    state = np.zeros(STATE_SHAPE, dtype=np.float32)

    assert {agent.explore_action(state, mask) for _ in range(50)} == {4}


def test_epsilon_must_be_a_probability():
    agent = DQNAgent(config=TrainingConfig(**SMALL))
    with pytest.raises(ValueError, match="epsilon"):
        agent.set_epsilon(2.0)


def test_the_schedule_is_driven_by_decisions():
    agent = DQNAgent(config=TrainingConfig(**SMALL, epsilon_start=1.0, epsilon_end=0.5,
                                           epsilon_decay_steps=4))
    assert agent.epsilon == 1.0
    seen = []
    for _ in range(4):
        agent.note_decision()
        seen.append(agent.epsilon)
    assert seen == pytest.approx([0.875, 0.75, 0.625, 0.5])
    assert agent.decisions == 4
    agent.note_decision()
    assert agent.epsilon == 0.5


# ------------------------------------------------------- the evaluation interface


def test_choose_action_returns_a_legal_placement_the_environment_accepts():
    agent = load_agent(seed=10)
    env = TetrisEnv(seed=21)
    state = env.reset()

    for _ in range(25):
        valid = env.get_valid_actions()
        placement = agent.choose_action(state, valid)

        assert isinstance(placement, Placement)
        assert placement in valid
        state, _reward, done, _info = env.step(placement)
        if done:
            break
    assert agent.last_decision is not None
    assert agent.last_decision.legal == len(valid)
    assert "AI selected" in str(agent.last_decision)


def test_the_decision_record_reports_what_the_framework_meters():
    agent = load_agent(seed=11)
    env = TetrisEnv(seed=22)
    state = env.reset()
    agent.choose_action(state, env.get_valid_actions())

    record = agent.last_decision
    assert record.nodes == 1                     # one network evaluation, one unit of work
    assert record.q_values.shape == (ACTION_SIZE,)
    assert record.best_q == pytest.approx(record.q_values[record.action])


def test_a_freshly_built_agent_starts_at_the_top_of_the_schedule():
    assert load_agent(seed=12).epsilon == pytest.approx(1.0)
    assert load_agent(seed=12, epsilon_start=0.3, epsilon_end=0.1).epsilon == pytest.approx(0.3)


def load_agent(seed=0, **config):
    return DQNAgent(config=TrainingConfig(**{**SMALL, **config}), seed=seed)


# ----------------------------------------------------------------- checkpoints


def test_a_checkpoint_round_trips_into_an_identical_agent(tmp_path):
    agent = load_agent(seed=13)
    agent.note_decision()
    agent.episode = 42
    batch = a_batch()
    agent.train_step(batch)
    path = agent.save(tmp_path / "checkpoint_42.pt")

    loaded = DQNAgent.load(path)

    assert loaded.steps == agent.steps
    assert loaded.decisions == agent.decisions
    assert loaded.episode == 42
    assert loaded.epsilon == pytest.approx(agent.epsilon)
    assert loaded.config == agent.config
    state = np.zeros(STATE_SHAPE, dtype=np.float32)
    assert np.array_equal(loaded.q_values(state), agent.q_values(state))


def test_a_checkpoint_carries_what_a_resume_needs(tmp_path):
    agent = load_agent(seed=14)
    agent.train_step(a_batch())                 # so the optimizer holds state
    path = agent.save(tmp_path / "checkpoint_1.pt")

    payload = torch.load(path)

    for key in ("online", "target", "optimizer", "config", "steps", "decisions",
                "episode", "epsilon", "rng_state", "environment", "format"):
        assert key in payload, key
    assert payload["agent"] == "dqn"
    assert payload["environment"]["torch_version"]
    assert payload["config"]["episodes"] == agent.config.episodes


def test_training_continues_after_a_reload(tmp_path):
    """Resuming is not just loading weights: the optimizer and counters come back."""
    agent = load_agent(seed=15)
    batch = a_batch()
    for _ in range(3):
        agent.train_step(batch)
    path = agent.save(tmp_path / "checkpoint_3.pt")

    resumed = DQNAgent.load(path)
    assert resumed.optimizer.state_dict()["state"]      # Adam's moments are there
    loss = resumed.train_step(batch)
    assert np.isfinite(loss)
    assert resumed.steps == agent.steps + 1


def test_the_replay_stream_continues_after_a_reload(tmp_path):
    agent = load_agent(seed=16)
    rng = agent.rng
    path = agent.save(tmp_path / "checkpoint.pt")
    resumed = DQNAgent.load(path)
    assert resumed.rng.random() == pytest.approx(rng.random())


def test_a_checkpoint_from_another_format_is_refused(tmp_path):
    path = tmp_path / "wrong.pt"
    torch.save({"format": 99}, path)
    with pytest.raises(ValueError, match="format"):
        DQNAgent.load(path)


def test_saving_creates_the_directory(tmp_path):
    agent = load_agent(seed=17)
    path = agent.save(tmp_path / "deep" / "nested" / "checkpoint_1.pt")
    assert (tmp_path / "deep" / "nested" / "checkpoint_1.pt").exists()
    assert path.endswith("checkpoint_1.pt")


def test_the_agent_can_be_played_from_a_checkpoint_file(tmp_path):
    agent = load_agent(seed=18)
    path = agent.save(tmp_path / "checkpoint_9.pt")

    loaded = load_agent_for_evaluation(path)

    assert loaded.epsilon == 0.0
    env = TetrisEnv(seed=23)
    state = env.reset()
    assert loaded.choose_action(state, env.get_valid_actions()) in env.get_valid_actions()


def test_the_agent_reports_its_size_and_answers_for_a_batch():
    agent = load_agent(seed=19)
    assert agent.parameter_count == agent.online.parameter_count
    batch = np.stack([np.zeros(STATE_SHAPE, dtype=np.float32)] * 3)
    assert agent.q_values(batch).shape == (3, ACTION_SIZE)
    assert agent.name == "dqn"
    assert "DQNAgent" in repr(agent) and str(agent.parameter_count) in repr(agent)
