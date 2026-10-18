"""Shared, reproducible evaluation for all Wumpus agents."""

import random

import numpy as np

from environment import WumpusEnv


def evaluate_agent(
    agent,
    *,
    num_test_maps: int = 100,
    test_seed: int = 12345,
    policy_seed: int = 54321,
    label: str | None = None,
):
    """Evaluate an agent once on each map in one fixed unseen map pool.

    The same ``test_seed`` creates the same maps for every agent. Exploration is
    disabled during evaluation. Recurrent agents have their hidden state reset
    before every map.
    """
    if num_test_maps < 1:
        raise ValueError("num_test_maps must be at least 1")

    test_env = WumpusEnv(num_training_maps=num_test_maps, seed=test_seed)
    old_epsilon = agent.epsilon
    python_random_state = random.getstate()
    numpy_random_state = np.random.get_state()

    rewards = []
    episode_lengths = []
    wins = 0
    deaths = 0
    timeouts = 0

    try:
        agent.epsilon = 0.0
        random.seed(policy_seed)
        np.random.seed(policy_seed)

        for map_id in range(num_test_maps):
            obs, _ = test_env.reset(options={"map_id": map_id})
            if hasattr(agent, "reset_hidden"):
                agent.reset_hidden()

            episode_reward = 0.0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = agent.get_action(obs)
                obs, reward, terminated, truncated, info = test_env.step(action)
                episode_reward += reward

            rewards.append(episode_reward)
            episode_lengths.append(info["steps"])

            if terminated and reward == test_env.goal_reward:
                wins += 1
            elif terminated:
                deaths += 1
            else:
                timeouts += 1
    finally:
        agent.epsilon = old_epsilon
        random.setstate(python_random_state)
        np.random.set_state(numpy_random_state)

    results = {
        "episodes": num_test_maps,
        "win_rate": wins / num_test_maps,
        "death_rate": deaths / num_test_maps,
        "timeout_rate": timeouts / num_test_maps,
        "average_reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "average_steps": float(np.mean(episode_lengths)),
    }

    if label:
        print(label)
    print(f"Test maps:       {results['episodes']}")
    print(f"Win rate:        {results['win_rate']:.1%}")
    print(f"Death rate:      {results['death_rate']:.1%}")
    print(f"Timeout rate:    {results['timeout_rate']:.1%}")
    print(f"Average reward:  {results['average_reward']:.2f}")
    print(f"Reward std:      {results['reward_std']:.2f}")
    print(f"Average steps:   {results['average_steps']:.2f}")

    return results
