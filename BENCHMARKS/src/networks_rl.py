from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


class PoolingActorCritic(nn.Module):
    """Same actor-critic as the source reproduction, but masked mean-pooling
    over action embeddings replaces multi-head attention -- an architecture
    ablation isolating whether attention itself is doing any work (Candidate/
    benchmark: Plain-MLP-PPO, no-attention)."""

    def __init__(self, task_dim: int = 12, action_dim: int = 10, hidden: int = 128, heads: int = 8):
        super().__init__()
        del heads
        self.task_encoder = nn.Sequential(nn.Linear(task_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
        self.actor_query = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh())
        self.actor_bias = nn.Linear(hidden, 1)
        self.critic = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, task, actions, mask):
        task_embedding = self.task_encoder(task)
        action_embedding = self.action_encoder(actions)
        pooled = (action_embedding * mask[..., None].float()).sum(dim=1) / mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        joint = torch.cat([task_embedding, pooled], dim=-1)
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


class MaskedQNetwork(nn.Module):
    """Masked attention Q-network for a DQN baseline (value-based RL,
    contrasting with the source's and candidates' on-policy PPO family)."""

    def __init__(self, task_dim: int = 12, action_dim: int = 10, hidden: int = 128, heads: int = 8):
        super().__init__()
        self.task_encoder = nn.Sequential(nn.Linear(task_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
        self.action_encoder = nn.Sequential(nn.Linear(action_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.q_query = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh())
        self.q_bias = nn.Linear(hidden, 1)

    def forward(self, task, actions, mask):
        task_embedding = self.task_encoder(task)
        action_embedding = self.action_encoder(actions)
        attended, _ = self.attention(task_embedding[:, None, :], action_embedding, action_embedding, need_weights=False)
        joint = torch.cat([task_embedding, attended[:, 0, :]], dim=-1)
        query = self.q_query(joint)
        q = torch.einsum("bd,bad->ba", query, action_embedding) / (query.shape[-1] ** 0.5)
        q = q + self.q_bias(action_embedding).squeeze(-1)
        q = q.masked_fill(~mask, torch.finfo(q.dtype).min)
        return q
