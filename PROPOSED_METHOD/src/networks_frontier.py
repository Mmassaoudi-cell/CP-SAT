from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical

from .temporal import TemporalHorizonEncoder
from .topology import NeighborhoodEncoder


class FrontierMaskedAttentionActorCritic(nn.Module):
    """Batched-frontier actor-critic: scores every currently-ready subtask's
    own (DC, slot) options in ONE forward pass (Candidates 2, 4, 5). Optional
    NeighborhoodEncoder (Candidate 4/5: topology) and TemporalHorizonEncoder
    (Candidates 2, 4, 5: cached anticipatory horizon features) can be enabled
    independently, so this single module implements TAGS (both enabled) and
    its two single-mechanism ablations (either enabled alone) plus the plain
    HCBS-PPO baseline (both disabled)."""

    def __init__(
        self,
        task_dim: int = 12,
        action_dim: int = 10,
        hidden: int = 128,
        heads: int = 8,
        use_topology: bool = False,
        use_horizon: bool = False,
        n_dc: int = 3,
        horizon: int = 96,
    ):
        super().__init__()
        self.use_topology = use_topology
        self.use_horizon = use_horizon
        self.n_dc = n_dc
        self.horizon = horizon
        self.task_encoder = nn.Sequential(nn.Linear(task_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
        if use_topology:
            self.neighborhood = NeighborhoodEncoder(task_hidden=hidden, hidden=32, heads=4)
            self.task_fuse = nn.Sequential(nn.Linear(hidden + self.neighborhood.output_dim, hidden), nn.Tanh())
        action_in = action_dim
        if use_horizon:
            self.horizon_encoder = TemporalHorizonEncoder(in_channels=2, hidden=16, layers=3, kernel=3)
            action_in += self.horizon_encoder.output_dim
        self.action_encoder = nn.Sequential(nn.Linear(action_in, hidden), nn.Tanh(), nn.Linear(hidden, hidden))
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.actor_query = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh())
        self.actor_bias = nn.Linear(hidden, 1)
        self.critic = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def _augment_actions(self, actions: torch.Tensor, curves: torch.Tensor) -> torch.Tensor:
        if not self.use_horizon:
            return actions
        # curves: (B, n_dc, 2, horizon) -> horizon_embedding: (B, n_dc, horizon, H)
        B = curves.shape[0]
        flat_curves = curves.reshape(B * self.n_dc, 2, self.horizon)
        emb = self.horizon_encoder(flat_curves)  # (B*n_dc, horizon, H)
        emb = emb.reshape(B, self.n_dc * self.horizon, -1)  # aligns with action index dc*horizon+start
        F = actions.shape[1]
        emb = emb[:, None, :, :].expand(B, F, emb.shape[1], emb.shape[2])
        return torch.cat([actions, emb], dim=-1)

    def _task_embedding(self, tasks: torch.Tensor, neighbors: torch.Tensor, neighbor_mask: torch.Tensor) -> torch.Tensor:
        base = self.task_encoder(tasks)
        if not self.use_topology:
            return base
        topo = self.neighborhood(base, neighbors, neighbor_mask)
        return self.task_fuse(torch.cat([base, topo], dim=-1))

    def forward(self, tasks, actions, mask, valid, neighbors, neighbor_mask, curves):
        # tasks: (B,F,12) actions: (B,F,A,action_dim) mask: (B,F,A) valid: (B,F)
        B, F, A, _ = actions.shape
        task_embedding = self._task_embedding(tasks, neighbors, neighbor_mask)  # (B,F,H)
        actions_aug = self._augment_actions(actions, curves)
        H = task_embedding.shape[-1]
        act_flat = actions_aug.reshape(B * F, A, actions_aug.shape[-1])
        action_embedding = self.action_encoder(act_flat)  # (B*F,A,H)
        q = task_embedding.reshape(B * F, 1, H)
        row_has_valid = mask.reshape(B * F, A).any(dim=-1)
        safe_mask = mask.reshape(B * F, A).clone()
        safe_mask[~row_has_valid] = True
        key_padding_mask = ~safe_mask
        attended, _ = self.attention(q, action_embedding, action_embedding, key_padding_mask=key_padding_mask, need_weights=False)
        attended = attended.reshape(B, F, H)
        joint = torch.cat([task_embedding, attended], dim=-1)  # (B,F,2H)
        query = self.actor_query(joint)  # (B,F,H)
        query_flat = query.reshape(B * F, H)
        logits_flat = torch.einsum("nd,nad->na", query_flat, action_embedding) / (H**0.5)
        logits_flat = logits_flat + self.actor_bias(action_embedding).squeeze(-1)
        logits = logits_flat.reshape(B, F, A)
        neg_inf = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~mask, neg_inf)
        logits = logits.masked_fill(~valid[:, :, None], neg_inf)
        pooled = (joint * valid[:, :, None].float()).sum(dim=1) / valid.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        value = self.critic(pooled).squeeze(-1)
        return logits, value

    def act(self, tasks, actions, mask, valid, neighbors, neighbor_mask, curves, deterministic: bool = False):
        logits, value = self(tasks, actions, mask, valid, neighbors, neighbor_mask, curves)
        B, F, A = logits.shape
        flat_logits = logits.reshape(B * F, A)
        row_valid = valid.reshape(B * F)
        safe_logits = flat_logits.clone()
        safe_logits[~row_valid] = 0.0
        safe_logits[~row_valid, 0] = 1.0
        distribution = Categorical(logits=safe_logits)
        action = safe_logits.argmax(dim=-1) if deterministic else distribution.sample()
        log_prob = distribution.log_prob(action).reshape(B, F)
        entropy = distribution.entropy().reshape(B, F)
        log_prob = log_prob * valid.float()
        entropy = entropy * valid.float()
        action = action.reshape(B, F)
        return action, log_prob.sum(dim=1), entropy.sum(dim=1), value

    def evaluate_actions(self, tasks, actions, mask, valid, neighbors, neighbor_mask, curves, selected):
        logits, value = self(tasks, actions, mask, valid, neighbors, neighbor_mask, curves)
        B, F, A = logits.shape
        flat_logits = logits.reshape(B * F, A)
        row_valid = valid.reshape(B * F)
        safe_logits = flat_logits.clone()
        safe_logits[~row_valid] = 0.0
        safe_logits[~row_valid, 0] = 1.0
        distribution = Categorical(logits=safe_logits)
        selected_flat = selected.reshape(B * F).clamp(min=0, max=A - 1)
        log_prob = distribution.log_prob(selected_flat).reshape(B, F) * valid.float()
        entropy = distribution.entropy().reshape(B, F) * valid.float()
        return log_prob.sum(dim=1), entropy.sum(dim=1), value
