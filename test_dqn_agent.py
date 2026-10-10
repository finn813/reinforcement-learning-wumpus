import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dqn_agent import DQNAgent, ReplayBuffer, Transition, parse_checkpoints, parse_maps


def test_replay_buffer_has_fixed_capacity_and_samples():
    replay = ReplayBuffer(capacity=2, seed=0)
    for action in range(3):
        replay.append(Transition(np.zeros(4), action, -1.0, np.ones(4), False))
    assert len(replay) == 2
    assert {item.action for item in replay.sample(2)} == {1, 2}


def test_dqn_learns_from_a_complete_batch():
    agent = DQNAgent(4, 4, batch_size=2, seed=0)
    agent.remember((0, 0, 0, 0), 3, -1.0, (1, 0, 0, 0), False)
    agent.remember((1, 0, 0, 0), 0, 100.0, (1, 1, 0, 0), True)
    loss = agent.learn()
    assert loss is not None
    assert np.isfinite(loss)


def test_greedy_action_is_valid():
    agent = DQNAgent(4, 4, seed=0)
    assert 0 <= agent.get_action((0, 0, 0, 0), explore=False) < 4


def test_map_and_checkpoint_parsing():
    assert parse_maps("3") == 3
    assert parse_maps("inf") == "inf"
    assert parse_checkpoints("100, 10,100", 100) == [10, 100]
