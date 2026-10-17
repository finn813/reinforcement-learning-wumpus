from collections import defaultdict
import gymnasium as gym
import numpy as np

class WumpusAgent:
    def __init__(
            self,
            env: gym.Env,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float = 0.95,
    ):
        """Initialize a Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        self.env = env

        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []

    def get_action(self, obs: np.ndarray) -> int:
        """Choose an action using epsilon-greedy strategy.
        Returns:
            action: 0 (up), 1 (left), 2 (down), 3 (right)
        """
        state = tuple(obs)
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # With probability (1-epsilon): exploit (best known action)
        q_values = self.q_values[state]
        best_actions = np.where(q_values == np.max(q_values))[0]
        return int(np.random.choice(best_actions))
        
    def update(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        terminated: bool,
        next_obs: np.ndarray,
    ):
        
        state = tuple(obs)
        next_state = tuple(next_obs)
        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)
        future_q_value = (not terminated) * np.max(self.q_values[next_state])

        # What should the Q-value be?
        target = reward + self.discount_factor * future_q_value

        # How wrong was our current estimate
        temporal_difference = target - self.q_values[state][action]

        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.q_values[state][action] = (
            self.q_values[state][action] + self.lr * temporal_difference
        )

        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


