import numpy as np

from environment import WumpusEnv
from evaluation import evaluate_agent


class AlwaysUpAgent:
    def __init__(self):
        self.env = WumpusEnv()
        self.epsilon = 0.5

    def get_action(self, obs):
        return 0


def test_evaluation_is_reproducible_and_restores_epsilon():
    agent = AlwaysUpAgent()

    first = evaluate_agent(agent, num_test_maps=5, test_seed=7)
    second = evaluate_agent(agent, num_test_maps=5, test_seed=7)

    assert first == second
    assert agent.epsilon == 0.5
    assert np.isclose(
        first["win_rate"] + first["death_rate"] + first["timeout_rate"], 1.0
    )
