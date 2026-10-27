"""Shared, reproducible evaluation for all Wumpus agents."""

import random

import numpy as np

from environment import WumpusEnv


def _evaluate_on_maps(agent, env, map_ids, *, policy_seed: int):
    """Run a greedy policy once on every selected map."""
    map_ids = list(map_ids)
    old_epsilon = agent.epsilon
    python_random_state = random.getstate()
    numpy_random_state = np.random.get_state()

    rewards = []
    episode_lengths = []
    wins = deaths = timeouts = 0

    try:
        agent.epsilon = 0.0
        random.seed(policy_seed)
        np.random.seed(policy_seed)

        for map_id in map_ids:
            obs, _ = env.reset(options={"map_id": map_id})
            if hasattr(agent, "reset_hidden"):
                agent.reset_hidden()

            episode_reward = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                action = agent.get_action(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

            rewards.append(episode_reward)
            episode_lengths.append(info["steps"])
            if terminated and reward == env.goal_reward:
                wins += 1
            elif terminated:
                deaths += 1
            else:
                timeouts += 1
    finally:
        agent.epsilon = old_epsilon
        random.setstate(python_random_state)
        np.random.set_state(numpy_random_state)

    episodes = len(map_ids)
    return {
        "episodes": episodes,
        "win_rate": wins / episodes,
        "death_rate": deaths / episodes,
        "timeout_rate": timeouts / episodes,
        "average_reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "average_steps": float(np.mean(episode_lengths)),
    }


def _print_results(results, heading):
    print(heading)
    print(f"Maps:            {results['episodes']}")
    print(f"Win rate:        {results['win_rate']:.1%}")
    print(f"Death rate:      {results['death_rate']:.1%}")
    print(f"Timeout rate:    {results['timeout_rate']:.1%}")
    print(f"Average reward:  {results['average_reward']:.2f}")
    print(f"Reward std:      {results['reward_std']:.2f}")
    print(f"Average steps:   {results['average_steps']:.2f}")


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
    results = _evaluate_on_maps(
        agent, test_env, range(num_test_maps), policy_seed=policy_seed
    )
    _print_results(results, label or "Unseen test maps")

    return results


def evaluate_train_test(
    agent,
    training_env=None,
    *,
    num_test_maps: int = 100,
    test_seed: int = 12345,
    policy_seed: int = 54321,
    label: str | None = None,
):
    """Compare greedy performance on the training pool and unseen maps.

    ``generalization_gap`` is training win rate minus test win rate. A large
    positive value indicates that the policy memorised the training maps more
    successfully than it generalised to new ones.
    """
    if num_test_maps < 1:
        raise ValueError("num_test_maps must be at least 1")

    training_env = training_env or agent.env
    training_env = getattr(training_env, "unwrapped", training_env)
    if getattr(training_env, "_infinite_maps", True):
        raise ValueError(
            "training-map evaluation requires a finite num_training_maps pool"
        )

    # Evaluate a copy so the original training environment and its RNG/runtime
    # state remain untouched.
    train_eval_env = WumpusEnv(
        width=training_env.width,
        height=training_env.height,
        num_pits=training_env.num_pits,
        num_training_maps=len(training_env._map_pool),
        max_steps=training_env.max_steps,
        step_penalty=training_env.step_penalty,
        goal_reward=training_env.goal_reward,
        hazard_penalty=training_env.hazard_penalty,
        seed=0,
    )
    train_eval_env._map_pool = [
        (set(pits), wumpus, gold)
        for pits, wumpus, gold in training_env._map_pool
    ]
    test_env = WumpusEnv(
        width=training_env.width,
        height=training_env.height,
        num_pits=training_env.num_pits,
        num_training_maps=num_test_maps,
        max_steps=training_env.max_steps,
        step_penalty=training_env.step_penalty,
        goal_reward=training_env.goal_reward,
        hazard_penalty=training_env.hazard_penalty,
        seed=test_seed,
    )

    train = _evaluate_on_maps(
        agent, train_eval_env, range(len(train_eval_env._map_pool)),
        policy_seed=policy_seed,
    )
    test = _evaluate_on_maps(
        agent, test_env, range(num_test_maps), policy_seed=policy_seed,
    )
    comparison = {
        "training": train,
        "test": test,
        "generalization_gap": train["win_rate"] - test["win_rate"],
        "reward_gap": train["average_reward"] - test["average_reward"],
    }

    if label:
        print(label)
    _print_results(train, "Training maps")
    print()
    _print_results(test, "Unseen test maps")
    print()
    print(f"Generalization gap: {comparison['generalization_gap']:+.1%}")
    print(f"Reward gap:         {comparison['reward_gap']:+.2f}")
    return comparison
