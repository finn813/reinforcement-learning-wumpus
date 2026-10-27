from environment import WumpusEnv
from evaluation import evaluate_agent
from map_sets import layout_key, master_pool, test_maps, training_env, training_pool


def keys(maps):
    return [layout_key(m) for m in maps]


def test_master_pool_is_distinct_and_pools_are_nested():
    master = keys(master_pool())
    assert len(master) == len(set(master)) == 1000
    for size in (1, 10, 100, 1000):
        assert keys(training_pool(size)) == master[:size]
        assert keys(training_env(size)._map_pool) == master[:size]


def test_small_pools_match_the_environment_pools():
    for size in (1, 10, 100):
        assert keys(training_pool(size)) == keys(WumpusEnv(num_training_maps=size, seed=42)._map_pool)


def test_test_maps_are_distinct_and_unseen():
    test = keys(test_maps())
    assert len(test) == len(set(test)) == 1000
    assert not set(test) & set(keys(master_pool()))
    assert keys(test_maps(100)) == test[:100]


class CountingAgent:
    """No epsilon; counts how often it is reset."""

    def __init__(self):
        self.resets = 0

    def reset(self):
        self.resets += 1

    def get_action(self, obs):
        return 0


def test_evaluation_resets_agents_without_hidden_state_and_epsilon():
    agent = CountingAgent()
    evaluate_agent(agent, num_test_maps=7)
    assert agent.resets == 7
