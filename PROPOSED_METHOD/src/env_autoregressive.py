from __future__ import annotations

import numpy as np

from SOURCE_METHOD_REPRODUCTION.src.environment import SchedulingEnvironment

from .topology import build_neighbor_index


class TopologyEnvironment(SchedulingEnvironment):
    """Source-style autoregressive env + per-task neighbor (predecessor/successor)
    raw feature tensors, for Candidate 1 (GAT-PPO)."""

    def __init__(self, scenario, config):
        env = config["environment"]
        horizon = int(config["experiment"]["horizon"])
        capacity_max = float(np.asarray(env["dc_capacity_kw"], dtype=np.float32).max())
        bandwidth_max = float(np.asarray(env["bandwidth_capacity_mbps"], dtype=np.float32).max())
        self._neighbor_index = build_neighbor_index(scenario, capacity_max, bandwidth_max, horizon)
        super().__init__(scenario, config)

    def observation(self):
        obs = super().observation()
        if self.done:
            neigh = np.zeros((6, 6), dtype=np.float32)
            mask = np.zeros(6, dtype=bool)
        else:
            neigh, mask = self._neighbor_index[self._task().task_id]
        obs = dict(obs)
        obs["neighbors"] = neigh
        obs["neighbor_mask"] = mask
        return obs


class LagrangianEnvironment(SchedulingEnvironment):
    """Source-style autoregressive env with a PID-Lagrangian carbon penalty
    replacing the fixed-weight reward, for Candidate 3 (LC-PPO)."""

    def __init__(self, scenario, config):
        super().__init__(scenario, config)
        self.carbon_lambda = 0.0

    def set_lambda(self, value: float) -> None:
        self.carbon_lambda = float(value)

    def step(self, action: int):
        task = self._task()
        obs = self.observation()
        if not obs["mask"][action]:
            raise ValueError(f"Masked action {action} selected")
        dc, start = divmod(int(action), self.horizon)
        end = start + task.duration
        if self._mask_is_fallback:
            self.fallback_actions += 1
        if np.any(self.base_power[dc] + self.power[dc, start:end] + task.power_kw > self.capacity[dc] + 1e-6):
            self.capacity_violations += 1
        self.power[dc, start:end] += task.power_kw
        self.bandwidth[dc, start:end] += task.bandwidth_mbps
        self.assignments[task.task_id] = (dc, start, end)
        energy = task.power_kw * task.duration * self.interval_hours
        mean_price = float(self.scenario.prices[dc, start:end].mean())
        mean_cef = float(self.scenario.cefs[dc, start:end].mean())
        cost_component = energy * mean_price
        emissions_component = energy * mean_cef
        reward = -(cost_component + self.carbon_lambda * emissions_component) / 100.0
        if self._mask_is_fallback:
            reward -= 100.0
        self.cursor += 1
        done = self.done
        next_obs = self.observation()
        return next_obs, float(reward), done, {}
