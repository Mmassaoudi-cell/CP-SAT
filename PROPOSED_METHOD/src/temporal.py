from __future__ import annotations

import numpy as np
import torch
from torch import nn


class PrefixSumTable:
    """O(1) windowed-mean lookup for price/CEF curves, replacing the source
    reproduction's O(duration) slice-and-mean recomputed on every decision.

    Targets weakness #5 (unbatched, uncached per-step feature reconstruction).
    """

    def __init__(self, curves: np.ndarray):
        # curves: (n_dc, horizon)
        n_dc, horizon = curves.shape
        self.horizon = horizon
        cumsum = np.zeros((n_dc, horizon + 1), dtype=np.float64)
        cumsum[:, 1:] = np.cumsum(curves, axis=1)
        self.cumsum = cumsum

    def window_mean(self, dc: int, start: int, end: int) -> float:
        end = min(end, self.horizon)
        start = min(start, end)
        length = max(1, end - start)
        return float((self.cumsum[dc, end] - self.cumsum[dc, start]) / length)

    def window_mean_batch(self, dc: int, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        ends = np.minimum(ends, self.horizon)
        starts = np.minimum(starts, ends)
        length = np.maximum(1, ends - starts)
        return (self.cumsum[dc, ends] - self.cumsum[dc, starts]) / length


class TemporalHorizonEncoder(nn.Module):
    """Dilated causal Conv1D stack producing a per-(dc, slot) anticipatory
    embedding of the full price/CEF horizon, computed once per scenario and
    cached (looked up by slot index during decisions) instead of being
    recomputed from raw values at every subtask decision.
    """

    def __init__(self, in_channels: int = 2, hidden: int = 32, layers: int = 3, kernel: int = 3):
        super().__init__()
        blocks = []
        channels = in_channels
        for i in range(layers):
            dilation = 2**i
            pad = (kernel - 1) * dilation
            blocks.append(_CausalConvBlock(channels, hidden, kernel, dilation, pad))
            channels = hidden
        self.blocks = nn.ModuleList(blocks)
        self.output_dim = hidden

    def forward(self, curves: torch.Tensor) -> torch.Tensor:
        # curves: (n_dc, in_channels, horizon) -> (n_dc, horizon, hidden)
        x = curves
        for block in self.blocks:
            x = block(x)
        return x.transpose(1, 2)


class _CausalConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, dilation, pad):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation, padding=pad)
        self.pad = pad
        self.act = nn.Tanh()

    def forward(self, x):
        y = self.conv(x)
        if self.pad > 0:
            y = y[..., : -self.pad]
        return self.act(y)
