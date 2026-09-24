"""Train and evaluate a tabular Q-learning agent on :class:`WumpusEnv`.

This follows Gymnasium's introductory Q-learning example, adapted to the
Wumpus observation, reward scale, and time-limit semantics.
"""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from environment import WumpusEnv

State = tuple[int, int, int, int]


def observation_to_state(observation: np.ndarray) -> State:
    """Convert ``[x, y, breeze, stench]`` into a stable table key."""
    values = np.asarray(observation)
    if values.shape != (4,):
        raise ValueError(f"expected observation shape (4,), got {values.shape}")
    return tuple(int(value) for value in values)  # type: ignore[return-value]


class QLearningAgent:
    """An epsilon-greedy, tabular Q-learning agent for discrete actions."""

    def __init__(
        self,
        number_of_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.95,
        initial_epsilon: float = 1.0,
        final_epsilon: float = 0.05,
        epsilon_decay: float = 0.0,
        seed: int | None = None,
    ) -> None:
        if number_of_actions < 1:
            raise ValueError("number_of_actions must be positive")
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0.0 <= discount_factor <= 1.0:
            raise ValueError("discount_factor must be in [0, 1]")
        if not 0.0 <= final_epsilon <= initial_epsilon <= 1.0:
            raise ValueError("epsilon values must satisfy 0 <= final <= initial <= 1")

        self.number_of_actions = number_of_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = max(0.0, epsilon_decay)
        self.rng = np.random.default_rng(seed)
        self.q_values: defaultdict[State, np.ndarray] = defaultdict(
            lambda: np.zeros(self.number_of_actions, dtype=np.float64)
        )
        self.training_errors: list[float] = []

    def get_action(self, state: State, *, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.number_of_actions))

        values = self.q_values[state]
        # Random tie-breaking prevents action 0 from being preferred merely
        # because all unseen actions initially have equal value.
        best_actions = np.flatnonzero(values == values.max())
        return int(self.rng.choice(best_actions))

    def update(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        terminated: bool,
    ) -> None:
        """Apply one Bellman update.

        A true terminal state has no future value. A Gymnasium truncation is a
        time limit rather than an MDP terminal state, so it deliberately still
        bootstraps (the distinction highlighted in Gymnasium's API).
        """
        future_value = 0.0 if terminated else float(self.q_values[next_state].max())
        target = reward + self.discount_factor * future_value
        error = target - self.q_values[state][action]
        self.q_values[state][action] += self.learning_rate * error
        self.training_errors.append(float(error))

    def decay_exploration(self) -> None:
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        payload = {
            "number_of_actions": self.number_of_actions,
            "q_values": dict(self.q_values),
        }
        with Path(path).open("wb") as file:
            pickle.dump(payload, file)

    @classmethod
    def load(cls, path: str | Path, *, seed: int | None = None) -> "QLearningAgent":
        with Path(path).open("rb") as file:
            payload = pickle.load(file)
        agent = cls(payload["number_of_actions"], initial_epsilon=0.0,
                    final_epsilon=0.0, seed=seed)
        for state, values in payload["q_values"].items():
            agent.q_values[state] = np.asarray(values, dtype=np.float64)
        return agent


def train(
    env: WumpusEnv,
    agent: QLearningAgent,
    episodes: int,
    *,
    seed: int | None = None,
) -> list[float]:
    """Train in-place and return the total reward from each episode."""
    returns: list[float] = []
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed if episode == 0 else None)
        state = observation_to_state(observation)
        episode_return = 0.0

        while True:
            action = agent.get_action(state)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            next_state = observation_to_state(next_observation)
            agent.update(state, action, reward, next_state, terminated)
            episode_return += reward
            state = next_state
            if terminated or truncated:
                break

        returns.append(episode_return)
        agent.decay_exploration()
    return returns


def evaluate(
    env: WumpusEnv,
    agent: QLearningAgent,
    episodes: int = 1_000,
    *,
    seed: int | None = None,
) -> dict[str, float]:
    """Evaluate greedily without exploration or learning."""
    returns: list[float] = []
    goals = hazards = truncations = 0
    for episode in range(episodes):
        observation, _ = env.reset(seed=seed if episode == 0 else None)
        state = observation_to_state(observation)
        episode_return = 0.0
        while True:
            action = agent.get_action(state, explore=False)
            observation, reward, terminated, truncated, _ = env.step(action)
            state = observation_to_state(observation)
            episode_return += reward
            if terminated or truncated:
                if truncated:
                    truncations += 1
                elif reward == env.goal_reward:
                    goals += 1
                else:
                    hazards += 1
                break
        returns.append(episode_return)

    return {
        "average_return": float(np.mean(returns)),
        "goal_rate": goals / episodes,
        "hazard_rate": hazards / episodes,
        "truncation_rate": truncations / episodes,
    }


def _tail_mean(values: Iterable[float], count: int = 1_000) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    return float(array[-count:].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=50_000)
    parser.add_argument("--evaluation-episodes", type=int, default=1_000)
    parser.add_argument("--maps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("wumpus_q_table.pkl"))
    args = parser.parse_args()
    if args.episodes < 1 or args.evaluation_episodes < 1:
        parser.error("episode counts must be positive")

    env = WumpusEnv(num_training_maps=args.maps, seed=args.seed)
    decay_episodes = max(1, args.episodes // 2)
    agent = QLearningAgent(
        number_of_actions=env.action_space.n,
        learning_rate=0.1,
        discount_factor=0.95,
        initial_epsilon=1.0,
        final_epsilon=0.05,
        epsilon_decay=(1.0 - 0.05) / decay_episodes,
        seed=args.seed,
    )
    returns = train(env, agent, args.episodes, seed=args.seed)
    metrics = evaluate(env, agent, args.evaluation_episodes, seed=args.seed + 1)
    agent.save(args.output)

    print(f"Training return (last 1000): {_tail_mean(returns):.2f}")
    print(f"Evaluation average return:    {metrics['average_return']:.2f}")
    print(f"Goal rate:                    {metrics['goal_rate']:.1%}")
    print(f"Hazard rate:                  {metrics['hazard_rate']:.1%}")
    print(f"Truncation rate:              {metrics['truncation_rate']:.1%}")
    print(f"Saved Q-table to {args.output}")


if __name__ == "__main__":
    main()
