from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from .topology import NeighborhoodEncoder


class TopologyActorCritic(nn.Module):
    """Source MaskedAttentionActorCritic + a NeighborhoodEncoder (Candidate 1: GAT-PPO)."""

    def __init__(self, task_dim: int = 12, action_dim: int = 10, hidden: int = 128, heads: int = 8):
        super().__init__()
        self.task_encoder = nn.Sequential(nn.Linear(task_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
        self.neighborhood = NeighborhoodEncoder(task_hidden=hidden, hidden=32, heads=4)
        self.task_fuse = nn.Sequential(nn.Linear(hidden + self.neighborhood.output_dim, hidden), nn.Tanh())
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.actor_query = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh())
        self.actor_bias = nn.Linear(hidden, 1)
        self.critic = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, task, actions, mask, neighbors, neighbor_mask):
        base_task_embedding = self.task_encoder(task)
        topo = self.neighborhood(base_task_embedding, neighbors, neighbor_mask)
        task_embedding = self.task_fuse(torch.cat([base_task_embedding, topo], dim=-1))
        action_embedding = self.action_encoder(actions)
        attended, _ = self.attention(task_embedding[:, None, :], action_embedding, action_embedding, need_weights=False)
        joint = torch.cat([task_embedding, attended[:, 0, :]], dim=-1)
        query = self.actor_query(joint)
        logits = torch.einsum("bd,bad->ba", query, action_embedding) / (query.shape[-1] ** 0.5)
        logits = logits + self.actor_bias(action_embedding).squeeze(-1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        value = self.critic(joint).squeeze(-1)
        return logits, value

    def act(self, task, actions, mask, neighbors, neighbor_mask, deterministic: bool = False):
        logits, value = self(task, actions, mask, neighbors, neighbor_mask)
        distribution = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy(), value

    def evaluate_actions(self, task, actions, mask, neighbors, neighbor_mask, selected):
        logits, value = self(task, actions, mask, neighbors, neighbor_mask)
        distribution = Categorical(logits=logits)
        return distribution.log_prob(selected), distribution.entropy(), value
