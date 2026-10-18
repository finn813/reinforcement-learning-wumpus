import numpy as np
import torch

from deep_q_agent import DeepWumpusAgent
from environment import WumpusEnv


def make_agent(**kwargs):
    env = WumpusEnv(seed=0)
    return DeepWumpusAgent(
        env=env,
        learning_rate=0.001,
        initial_epsilon=0.0,
        epsilon_decay=0.01,
        final_epsilon=0.0,
        **kwargs,
    )


def test_get_action_uses_network_when_not_exploring():
    agent = make_agent()
    obs, _ = agent.env.reset()

    for parameter in agent.q_network.parameters():
        torch.nn.init.constant_(parameter, 0.0)

    assert agent.get_action(obs) == 0


def test_update_trains_after_collecting_a_full_batch():
    agent = make_agent(batch_size=2)
    obs = np.zeros(4, dtype=np.float32)
    next_obs = np.ones(4, dtype=np.float32)

    agent.update(obs, 0, 1.0, False, next_obs)
    assert agent.training_error == []

    agent.update(obs, 0, 1.0, True, next_obs)
    assert len(agent.training_error) == 1
    assert agent.training_error[0] >= 0
