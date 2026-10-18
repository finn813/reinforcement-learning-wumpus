import gymnasium as gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import collections
import copy
import numpy as np



# largely inspired by https://github.com/keep9oing/DRQN-Pytorch-CartPole-v1/blob/main/DRQN.py

class DRQNN(nn.Module):
    def __init__(self, state_space=None,
                 action_space=None,
                 obs_high=None):
        super(DRQNN, self).__init__()

        assert state_space is not None, "None state_space input: state_space should be selected."
        assert action_space is not None, "None action_space input: action_space should be selected."

        self.hidden_space = 64
        self.state_space = state_space
        self.action_space = action_space

        # Scale observations to [0, 1] inside the network, so acting and training
        # always see the same inputs. obs_high = env.observation_space.high
        if obs_high is None:
            obs_high = np.ones(self.state_space)
        scale = torch.as_tensor(obs_high, dtype=torch.float32).clamp(min=1.0)
        self.register_buffer("obs_scale", scale)

        self.Linear1 = nn.Linear(self.state_space, self.hidden_space)
        self.lstm = nn.LSTM(self.hidden_space, self.hidden_space, batch_first=True)
        self.Linear2 = nn.Linear(self.hidden_space, self.action_space)

    def forward(self, x, h, c):
        x = F.relu(self.Linear1(x / self.obs_scale))
        x, (new_h, new_c) = self.lstm(x, (h,c))
        x = self.Linear2(x)
        return x, new_h, new_c

    def init_hidden_state(self, batch_size, training=None):
    
        assert training is not None, "training step parameter should be dtermined"

        if training is True:
            return torch.zeros([1, batch_size, self.hidden_space]), torch.zeros([1, batch_size, self.hidden_space])
        else:
            return torch.zeros([1, 1, self.hidden_space]), torch.zeros([1, 1, self.hidden_space])

class EpisodeMemory():
    """Replay memory that stores whole episodes and samples padded batches of them."""

    def __init__(
            self,
            max_epi_num = 1000,
            batch_size = 8,):
        self.max_epi_num = max_epi_num
        self.batch_size = batch_size
        self.memory = collections.deque(maxlen=self.max_epi_num)

    def put(self, obs, actions, rewards, terminated):
        """Store one finished episode of T steps.

        Args:
            obs: T+1 observations o_0 ... o_T (the last one is the final next_obs)
            actions, rewards, terminated: T entries each
        """
        self.memory.append({
            "obs": torch.as_tensor(np.array(obs), dtype=torch.float32),
            "actions": torch.as_tensor(actions, dtype=torch.long),
            "rewards": torch.as_tensor(rewards, dtype=torch.float32),
            "terminated": torch.as_tensor(terminated, dtype=torch.float32),
        })

    def sample(self):
        """Sample batch_size random episodes, zero-padded to the longest one.

        Requires len(self) >= batch_size.

        Returns:
            obs:        (B, T_max+1, obs_dim)
            actions:    (B, T_max)  long
            rewards:    (B, T_max)
            terminated: (B, T_max)
            mask:       (B, T_max)  1.0 for real steps, 0.0 for padding
        """
        batch = random.sample(self.memory, self.batch_size)
        lengths = torch.tensor([len(ep["actions"]) for ep in batch])
        T_max = int(lengths.max())

        obs = pad_sequence([ep["obs"] for ep in batch], batch_first=True)
        actions = pad_sequence([ep["actions"] for ep in batch], batch_first=True)
        rewards = pad_sequence([ep["rewards"] for ep in batch], batch_first=True)
        terminated = pad_sequence([ep["terminated"] for ep in batch], batch_first=True)
        mask = (torch.arange(T_max)[None, :] < lengths[:, None]).float()

        return obs, actions, rewards, terminated, mask

    def __len__(self):
        return len(self.memory)

class DQRNWumpusAgent:
    def __init__(
            self,
            env: gym.Env,
            learning_rate: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            discount_factor: float = 0.95,
            memory_size: int = 1000,
            batch_size: int = 8,
            min_episodes: int = 20,
            target_update_period: int = 100,
            reward_scale: float = 100.0,
            ):
        """Initialize a DQRN-Learning agent.
        
        Args:
            env: The training environment
            learning_rate: Adam learning rate (e.g. 1e-3)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
            memory_size: How many episodes the replay memory keeps
            batch_size: How many episodes are sampled per training step
            min_episodes: Episodes to collect before training starts
            target_update_period: Gradient updates between target-network syncs
                (hard copy, as in the DRQN paper which uses 10,000)
            reward_scale: Rewards are divided by this before training (+-100 -> +-1)
        """

        
        self.env = env
        self.lr = learning_rate
        self.discount_factor = discount_factor

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []

        self.drqnn = DRQNN(state_space=self.env.observation_space.shape[0], 
                           action_space=self.env.action_space.n,
                           obs_high=self.env.observation_space.high)
        self.target_drqnn = copy.deepcopy(self.drqnn)  # target network, synced periodically
        self.optimizer = torch.optim.Adam(self.drqnn.parameters(), lr=self.lr)
        self.memory = EpisodeMemory(max_epi_num=memory_size, batch_size=batch_size)

        self.min_episodes = max(min_episodes, batch_size)  # warm-up before training
        self.target_update_period = target_update_period    # counted in gradient updates
        self.reward_scale = reward_scale
        self.train_steps = 0
        self.reset_hidden()

    def get_action(self, obs):
        x = torch.as_tensor(obs, dtype=torch.float32).view(1,1,-1)
        with torch.no_grad():
            q, self.h, self.c = self.drqnn(x, self.h, self.c)
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        return q.argmax().item()

    def reset_hidden(self):
       self.h, self.c = self.drqnn.init_hidden_state(batch_size=1, training=False)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def train_step(self):
        """One gradient update on a batch of whole episodes (bootstrapped sequential updates).

        Returns the loss, or None while the memory is still warming up.
        """
        if len(self.memory) < self.min_episodes:
            return None

        obs, actions, rewards, terminated, mask = self.memory.sample()
        rewards = rewards / self.reward_scale
        B = obs.shape[0]

        # Unroll both networks over the whole episodes, starting from a zero hidden state.
        # Index t of the output is Q(o_0..o_t), so [:, :-1] is "now" and [:, 1:] is "next".
        h, c = self.drqnn.init_hidden_state(batch_size=B, training=True)
        q_all, _, _ = self.drqnn(obs, h, c)                                 # (B, T+1, A)
        q_sa = q_all[:, :-1].gather(2, actions.unsqueeze(-1)).squeeze(-1)   # (B, T)

        with torch.no_grad():
            q_next, _, _ = self.target_drqnn(obs, h, c)                     # (B, T+1, A)
            max_q_next = q_next[:, 1:].max(dim=-1).values                   # (B, T)
            # no bootstrap after death/gold; truncated episodes still bootstrap
            target = rewards + self.discount_factor * (1.0 - terminated) * max_q_next

        # Huber loss, averaged over real steps only (padding is masked out)
        loss = (F.smooth_l1_loss(q_sa, target, reduction="none") * mask).sum() / mask.sum()

        self.optimizer.zero_grad()
        loss.backward()
        # DRQN paper: "LSTM's gradients were clipped to a value of ten"
        torch.nn.utils.clip_grad_value_(self.drqnn.lstm.parameters(), 10.0)
        self.optimizer.step()

        # DRQN paper (same as DQN): target network = stale copy of the online network,
        # refreshed every `target_update_period` gradient updates.
        self.train_steps += 1
        if self.train_steps % self.target_update_period == 0:
            self.update_target()

        self.training_error.append(loss.item())
        return loss.item()

    def update_target(self):
        """Hard copy of the online network's weights into the target network."""
        self.target_drqnn.load_state_dict(self.drqnn.state_dict())



    