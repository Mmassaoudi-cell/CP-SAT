from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


class MaskedAttentionActorCritic(nn.Module):
    def __init__(self, task_dim: int = 12, action_dim: int = 10, hidden: int = 128, heads: int = 8):
        super().__init__()
        self.task_encoder = nn.Sequential(nn.Linear(task_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.actor_query = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh())
        self.actor_bias = nn.Linear(hidden, 1)
        self.critic = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, task: torch.Tensor, actions: torch.Tensor, mask: torch.Tensor):
        task_embedding = self.task_encoder(task)
        action_embedding = self.action_encoder(actions)
        attended, _ = self.attention(task_embedding[:, None, :], action_embedding, action_embedding, need_weights=False)
        joint = torch.cat([task_embedding, attended[:, 0, :]], dim=-1)
        query = self.actor_query(joint)
        logits = torch.einsum("bd,bad->ba", query, action_embedding) / (query.shape[-1] ** 0.5)
        logits = logits + self.actor_bias(action_embedding).squeeze(-1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        value = self.critic(joint).squeeze(-1)
        return logits, value

    def act(self, task, actions, mask, deterministic: bool = False):
        logits, value = self(task, actions, mask)
        distribution = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy(), value

    def evaluate_actions(self, task, actions, mask, selected):
        logits, value = self(task, actions, mask)
        distribution = Categorical(logits=logits)
        return distribution.log_prob(selected), distribution.entropy(), value
