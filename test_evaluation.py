import numpy as np

from environment import WumpusEnv
import pytest

from evaluation import evaluate_agent, evaluate_train_test


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


def test_train_test_evaluation_reports_generalization_gap_without_mutating_env():
    training_env = WumpusEnv(num_training_maps=3, seed=11)
    agent = AlwaysUpAgent()
    original_pool = list(training_env._map_pool)

    results = evaluate_train_test(
        agent, training_env, num_test_maps=4, test_seed=12
    )

    assert results["training"]["episodes"] == 3
    assert results["test"]["episodes"] == 4
    assert np.isclose(
        results["generalization_gap"],
        results["training"]["win_rate"] - results["test"]["win_rate"],
    )
    assert training_env._map_pool == original_pool
    assert agent.epsilon == 0.5


def test_train_test_evaluation_rejects_infinite_training_stream():
    agent = AlwaysUpAgent()
    infinite_env = WumpusEnv(num_training_maps="inf", seed=1)

    with pytest.raises(ValueError, match="finite num_training_maps"):
        evaluate_train_test(agent, infinite_env)
