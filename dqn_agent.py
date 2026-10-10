"""Train and evaluate a Deep Q-Network on the Wumpus environment.

The command-line interface supports learning-curve checkpoints and two fair
evaluation modes:

* ``training`` recreates the finite map pool used for training.
* ``new`` generates a fresh map for every evaluation episode.

Examples:
    python dqn_agent.py --maps 1 --episodes 5000
    python dqn_agent.py --maps 10 --eval-maps training --episodes 10000
    python dqn_agent.py --maps 10 --eval-maps new --episodes 10000
"""

from __future__ import annotations

import argparse
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn

from agent import evaluate, observation_to_state
from environment import WumpusEnv


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool


class ReplayBuffer:
    """Fixed-size replay memory with reproducible random sampling."""

    def __init__(self, capacity: int, seed: int | None = None) -> None:
        if capacity < 1:
            raise ValueError("replay capacity must be positive")
        self.transitions: deque[Transition] = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.transitions)

    def append(self, transition: Transition) -> None:
        self.transitions.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return self.rng.sample(self.transitions, batch_size)


class QNetwork(nn.Module):
    def __init__(self, observation_size: int, number_of_actions: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, number_of_actions),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.layers(observations)


class DQNAgent:
    """DQN with replay memory, a target network, and epsilon-greedy actions."""

    def __init__(
        self,
        observation_size: int,
        number_of_actions: int,
        *,
        learning_rate: float = 1e-3,
        discount_factor: float = 0.95,
        reward_scale: float = 100.0,
        initial_epsilon: float = 1.0,
        final_epsilon: float = 0.05,
        epsilon_decay: float = 0.0,
        replay_capacity: int = 20_000,
        batch_size: int = 64,
        target_update_steps: int = 250,
        seed: int | None = None,
        device: str = "cpu",
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if target_update_steps < 1:
            raise ValueError("target_update_steps must be positive")
        if reward_scale <= 0:
            raise ValueError("reward_scale must be positive")

        self.number_of_actions = number_of_actions
        self.discount_factor = discount_factor
        self.reward_scale = reward_scale
        self.epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay = max(0.0, epsilon_decay)
        self.batch_size = batch_size
        self.target_update_steps = target_update_steps
        self.device = torch.device(device)
        self.rng = np.random.default_rng(seed)
        self.replay = ReplayBuffer(replay_capacity, seed)
        self.optimization_steps = 0

        if seed is not None:
            torch.manual_seed(seed)
        self.online_network = QNetwork(observation_size, number_of_actions).to(self.device)
        self.target_network = QNetwork(observation_size, number_of_actions).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(), lr=learning_rate
        )
        self.loss = nn.SmoothL1Loss()

    def get_action(
        self, state: tuple[int, int, int, int] | np.ndarray, *, explore: bool = True
    ) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.number_of_actions))

        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_network(state_tensor).squeeze(0)
        best_actions = torch.nonzero(q_values == q_values.max()).flatten().cpu().numpy()
        return int(self.rng.choice(best_actions))

    def remember(
        self,
        state: tuple[int, int, int, int],
        action: int,
        reward: float,
        next_state: tuple[int, int, int, int],
        terminated: bool,
    ) -> None:
        self.replay.append(
            Transition(
                np.asarray(state, dtype=np.float32),
                action,
                reward,
                np.asarray(next_state, dtype=np.float32),
                terminated,
            )
        )

    def learn(self) -> float | None:
        if len(self.replay) < self.batch_size:
            return None

        batch = self.replay.sample(self.batch_size)
        states = torch.as_tensor(
            np.stack([item.state for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        actions = torch.as_tensor(
            [item.action for item in batch], dtype=torch.int64, device=self.device
        ).unsqueeze(1)
        rewards = torch.as_tensor(
            [item.reward for item in batch], dtype=torch.float32, device=self.device
        ) / self.reward_scale
        next_states = torch.as_tensor(
            np.stack([item.next_state for item in batch]),
            dtype=torch.float32,
            device=self.device,
        )
        terminated = torch.as_tensor(
            [item.terminated for item in batch], dtype=torch.float32, device=self.device
        )

        predicted = self.online_network(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_values = self.target_network(next_states).max(dim=1).values
            targets = rewards + self.discount_factor * (1.0 - terminated) * next_values

        loss = self.loss(predicted, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.optimization_steps += 1
        if self.optimization_steps % self.target_update_steps == 0:
            self.target_network.load_state_dict(self.online_network.state_dict())
        return float(loss.item())

    def decay_exploration(self) -> None:
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "number_of_actions": self.number_of_actions,
                "online_network": self.online_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )


def train_episode(env: WumpusEnv, agent: DQNAgent) -> float:
    observation, _ = env.reset()
    state = observation_to_state(observation)
    episode_return = 0.0

    while True:
        action = agent.get_action(state)
        next_observation, reward, terminated, truncated, _ = env.step(action)
        next_state = observation_to_state(next_observation)
        agent.remember(state, action, reward, next_state, terminated)
        agent.learn()
        episode_return += reward
        state = next_state
        if terminated or truncated:
            break

    agent.decay_exploration()
    return episode_return


def parse_maps(value: str) -> int | str:
    if value.lower() in {"inf", "infinite"}:
        return "inf"
    try:
        maps = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("maps must be a positive integer or 'inf'") from error
    if maps < 1:
        raise argparse.ArgumentTypeError("maps must be positive")
    return maps


def parse_checkpoints(value: str, maximum: int) -> list[int]:
    try:
        checkpoints = sorted({int(item.strip()) for item in value.split(",")})
    except ValueError as error:
        raise argparse.ArgumentTypeError("checkpoints must be comma-separated integers") from error
    if not checkpoints or checkpoints[0] < 1 or checkpoints[-1] > maximum:
        raise argparse.ArgumentTypeError(
            f"checkpoints must be between 1 and the episode count ({maximum})"
        )
    return checkpoints


def print_metrics(episode: int, metrics: dict[str, float]) -> None:
    print(
        f"{episode:>8} | {metrics['average_return']:>10.2f} | "
        f"{metrics['average_steps']:>9.2f} | "
        f"{metrics['average_goal_steps']:>10.2f} | "
        f"{metrics['goal_rate']:>7.1%} | {metrics['hazard_rate']:>7.1%} | "
        f"{metrics['truncation_rate']:>9.1%}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5_000)
    parser.add_argument("--evaluation-episodes", type=int, default=1_000)
    parser.add_argument("--maps", type=parse_maps, default=1)
    parser.add_argument(
        "--eval-maps",
        choices=("training", "new"),
        default="training",
        help="evaluate on a recreated training pool or a new map every episode",
    )
    parser.add_argument(
        "--checkpoints",
        default=None,
        help="comma-separated training episode counts to evaluate",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("wumpus_dqn.pt"))
    args = parser.parse_args()
    if args.episodes < 1 or args.evaluation_episodes < 1:
        parser.error("episode counts must be positive")
    if args.maps == "inf" and args.eval_maps == "training":
        parser.error("an infinite training stream cannot be recreated; use --eval-maps new")
    if args.checkpoints is None:
        checkpoints = [
            value for value in (100, 500, 1_000, 2_000, 5_000)
            if value <= args.episodes
        ]
    else:
        try:
            checkpoints = parse_checkpoints(args.checkpoints, args.episodes)
        except argparse.ArgumentTypeError as error:
            parser.error(str(error))
    if not checkpoints:
        checkpoints = [args.episodes]
    if checkpoints[-1] != args.episodes:
        checkpoints.append(args.episodes)

    training_env = WumpusEnv(num_training_maps=args.maps, seed=args.seed)
    evaluation_maps: int | str = args.maps if args.eval_maps == "training" else "inf"
    evaluation_seed = args.seed if args.eval_maps == "training" else args.seed + 10_000
    evaluation_env = WumpusEnv(
        num_training_maps=evaluation_maps, seed=evaluation_seed
    )
    agent = DQNAgent(
        observation_size=training_env.observation_space.shape[0],
        number_of_actions=training_env.action_space.n,
        epsilon_decay=(1.0 - 0.05) / max(1, args.episodes // 2),
        seed=args.seed,
    )

    print(" episodes | avg return | avg steps | goal steps |    goal |  hazard | truncated")
    print("----------+------------+-----------+------------+---------+---------+----------")
    checkpoint_set = set(checkpoints)
    returns: deque[float] = deque(maxlen=1_000)
    for episode in range(1, args.episodes + 1):
        returns.append(train_episode(training_env, agent))
        if episode in checkpoint_set:
            metrics = evaluate(
                evaluation_env,
                agent,
                args.evaluation_episodes,
                seed=evaluation_seed,
            )
            print_metrics(episode, metrics)

    agent.save(args.output)
    print(f"Training return (last {len(returns)}): {np.mean(returns):.2f}")
    print(f"Saved DQN to {args.output}")


if __name__ == "__main__":
    main()
