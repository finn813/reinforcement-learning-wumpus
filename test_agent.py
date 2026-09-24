import numpy as np

from agent import QLearningAgent, evaluate, observation_to_state, train
from environment import WumpusEnv


def test_observation_is_converted_to_integer_state():
    assert observation_to_state(np.array([2, 3, 1, 0], dtype=np.float32)) == (2, 3, 1, 0)


def test_terminal_update_does_not_bootstrap():
    agent = QLearningAgent(4, learning_rate=1.0, discount_factor=0.95, seed=0)
    agent.q_values[(1, 0, 0, 0)][0] = 500.0
    agent.update((0, 0, 0, 0), 3, 100.0, (1, 0, 0, 0), terminated=True)
    assert agent.q_values[(0, 0, 0, 0)][3] == 100.0


def test_truncation_style_update_bootstraps():
    agent = QLearningAgent(4, learning_rate=1.0, discount_factor=0.5, seed=0)
    agent.q_values[(1, 0, 0, 0)][0] = 10.0
    agent.update((0, 0, 0, 0), 3, -1.0, (1, 0, 0, 0), terminated=False)
    assert agent.q_values[(0, 0, 0, 0)][3] == 4.0


def test_training_learns_the_fixed_map():
    env = WumpusEnv(num_training_maps=1, seed=7)
    agent = QLearningAgent(
        4, learning_rate=0.1, initial_epsilon=1.0, final_epsilon=0.05,
        epsilon_decay=0.95 / 4_000, seed=7,
    )
    train(env, agent, 8_000, seed=7)
    metrics = evaluate(env, agent, 100, seed=8)
    assert metrics["goal_rate"] >= 0.95


def test_save_and_load_round_trip(tmp_path):
    agent = QLearningAgent(4, seed=1)
    agent.q_values[(0, 0, 0, 0)][3] = 12.5
    path = tmp_path / "table.pkl"
    agent.save(path)
    loaded = QLearningAgent.load(path, seed=1)
    assert loaded.epsilon == 0.0
    assert loaded.q_values[(0, 0, 0, 0)][3] == 12.5
