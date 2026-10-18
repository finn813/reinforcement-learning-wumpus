from collections import deque
import random

import gymnasium as gym
import numpy as np
import torch
from torch import nn


class QNetwork(nn.Module):
    """Small neural network that estimates one Q-value per action."""

    def __init__(self, observation_size: int, action_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


class DeepWumpusAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
        batch_size: int = 64,
        replay_buffer_size: int = 10_000,
        target_update_frequency: int = 100,
    ):
        """Initialize a simple Deep Q-Learning agent."""
        self.env = env
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency

        observation_size = int(np.prod(env.observation_space.shape))
        action_size = env.action_space.n

        self.q_network = QNetwork(observation_size, action_size)
        self.target_network = QNetwork(observation_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=learning_rate
        )
        self.replay_buffer = deque(maxlen=replay_buffer_size)

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []
        self.update_count = 0

    def get_action(self, obs: np.ndarray) -> int:
        """Choose an action using the epsilon-greedy strategy."""
        if np.random.random() < self.epsilon:
            return int(self.env.action_space.sample())

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).flatten()
        with torch.no_grad():
            q_values = self.q_network(obs_tensor)
        return int(torch.argmax(q_values).item())

    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: np.ndarray,
    ):
        """Store one experience and train the network on a random batch."""
        self.replay_buffer.append(
            (np.array(obs), action, reward, terminated, np.array(next_obs))
        )

        if len(self.replay_buffer) < self.batch_size:
            return

        batch = random.sample(self.replay_buffer, self.batch_size)
        observations, actions, rewards, terminated, next_observations = zip(*batch)

        observations = torch.as_tensor(np.array(observations), dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.long)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        terminated = torch.as_tensor(terminated, dtype=torch.float32)
        next_observations = torch.as_tensor(
            np.array(next_observations), dtype=torch.float32
        )

        current_q_values = self.q_network(observations)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            future_q_values = self.target_network(next_observations).max(dim=1).values
            target_q_values = rewards + (
                self.discount_factor * future_q_values * (1 - terminated)
            )

        loss = nn.functional.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.training_error.append(loss.item())
        self.update_count += 1

        if self.update_count % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)
