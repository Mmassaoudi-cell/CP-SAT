from __future__ import annotations

import numpy as np
import torch
from torch import nn

K_NEIGHBORS = 3
NEIGHBOR_RAW_DIM = 6


def build_neighbor_index(scenario, capacity_max: float, bandwidth_max: float, horizon: int):
    """Precompute padded predecessor/successor raw feature tensors for every task.

    Returns a dict of numpy arrays keyed by task_id -> (neighbors[2*K,6], mask[2*K]).
    Neighbor type flag (0=predecessor, 1=successor) is included in the raw feature.
    """
    successors: dict[int, list[int]] = {t.task_id: [] for t in scenario.tasks}
    by_id = {t.task_id: t for t in scenario.tasks}
    for t in scenario.tasks:
        for p in t.predecessors:
            successors[p].append(t.task_id)

    def raw(task_id: int, is_succ: float) -> np.ndarray:
        t = by_id[task_id]
        return np.asarray(
            [
                t.duration / 8.0,
                t.power_kw / capacity_max,
                t.bandwidth_mbps / bandwidth_max,
                t.arrival / horizon,
                t.deadline / horizon,
                is_succ,
            ],
            dtype=np.float32,
        )

    index = {}
    for t in scenario.tasks:
        preds = list(t.predecessors)[:K_NEIGHBORS]
        succs = successors[t.task_id][:K_NEIGHBORS]
        neigh = np.zeros((2 * K_NEIGHBORS, NEIGHBOR_RAW_DIM), dtype=np.float32)
        mask = np.zeros(2 * K_NEIGHBORS, dtype=bool)
        for i, p in enumerate(preds):
            neigh[i] = raw(p, 0.0)
            mask[i] = True
        for i, s in enumerate(succs):
            neigh[K_NEIGHBORS + i] = raw(s, 1.0)
            mask[K_NEIGHBORS + i] = True
        index[t.task_id] = (neigh, mask)
    return index


class NeighborhoodEncoder(nn.Module):
    """1-layer multi-head graph-attention neighbor encoder.

    Embeds a task's immediate predecessor/successor set (padded, masked) and
    attends over it using the task's own embedding as query, producing a
    fixed-size topology embedding. This is the mechanism a policy needs to see
    DAG structure beyond a scalar predecessor count (weakness #3).
    """

    def __init__(self, task_hidden: int, neighbor_dim: int = NEIGHBOR_RAW_DIM, hidden: int = 32, heads: int = 4):
        super().__init__()
        self.neighbor_encoder = nn.Sequential(nn.Linear(neighbor_dim, hidden), nn.Tanh())
        self.query_proj = nn.Linear(task_hidden, hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.Tanh())
        self.output_dim = hidden

    def forward(self, task_embedding: torch.Tensor, neighbors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # task_embedding: (..., task_hidden); neighbors: (..., 2K, neighbor_dim); mask: (..., 2K) bool (True=valid)
        shape = neighbors.shape[:-2]
        flat_n = neighbors.reshape(-1, neighbors.shape[-2], neighbors.shape[-1])
        flat_mask = mask.reshape(-1, mask.shape[-1])
        flat_q = self.query_proj(task_embedding.reshape(-1, task_embedding.shape[-1]))[:, None, :]
        neigh_emb = self.neighbor_encoder(flat_n)
        key_padding_mask = ~flat_mask
        all_invalid = key_padding_mask.all(dim=-1)
        safe_kpm = key_padding_mask.clone()
        safe_kpm[all_invalid] = False
        attended, _ = self.attention(flat_q, neigh_emb, neigh_emb, key_padding_mask=safe_kpm, need_weights=False)
        attended = attended[:, 0, :]
        attended = torch.where(all_invalid[:, None], torch.zeros_like(attended), attended)
        out = self.out(attended)
        return out.reshape(*shape, self.output_dim)
