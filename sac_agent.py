import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    def __init__(self, max_size=1_000_000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((
            state,
            action,
            reward,
            next_state,
            done,
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()

        y_t = torch.tanh(x_t)

        # convert from [-1, 1] to [0, 1]
        action = (y_t + 1.0) / 2.0

        log_prob = normal.log_prob(x_t)

        # correction for tanh squashing
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        return action, log_prob
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        mean = self.mean(x)
        log_std = self.log_std(x)

        log_std = torch.clamp(log_std, min=-20, max=2)

        return mean, log_std


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.q(x)

class SACAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        device=None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.actor = Actor(state_dim, action_dim, hidden_dim).to(self.device)

        self.critic_1 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic_2 = Critic(state_dim, action_dim, hidden_dim).to(self.device)

        self.target_critic_1 = Critic(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic_2 = Critic(state_dim, action_dim, hidden_dim).to(self.device)

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr,
        )

        self.critic_1_optimizer = torch.optim.Adam(
            self.critic_1.parameters(),
            lr=critic_lr,
        )

        self.critic_2_optimizer = torch.optim.Adam(
            self.critic_2.parameters(),
            lr=critic_lr,
        )

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if evaluate:
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
                action = (action + 1.0) / 2.0
            else:
                action, _ = self.actor.sample(state)

        return action.cpu().numpy()[0]

    def train(self, replay_buffer, batch_size=256):
        if len(replay_buffer) < batch_size:
            return None

        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)

            target_q1 = self.target_critic_1(next_state, next_action)
            target_q2 = self.target_critic_2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)

            target_q = reward + (1.0 - done) * self.gamma * (
                target_q - self.alpha * next_log_prob
            )

        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)

        critic_1_loss = F.mse_loss(current_q1, target_q)
        critic_2_loss = F.mse_loss(current_q2, target_q)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        new_action, log_prob = self.actor.sample(state)

        q1_new = self.critic_1(state, new_action)
        q2_new = self.critic_2(state, new_action)
        q_new = torch.min(q1_new, q2_new)

        actor_loss = (self.alpha * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

        return {
            "critic_1_loss": critic_1_loss.item(),
            "critic_2_loss": critic_2_loss.item(),
            "actor_loss": actor_loss.item(),
        }

    def soft_update(self, source, target):
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data
                + (1.0 - self.tau) * target_param.data
            )

