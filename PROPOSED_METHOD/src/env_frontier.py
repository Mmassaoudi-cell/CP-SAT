from __future__ import annotations

from dataclasses import asdict

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from SOURCE_METHOD_REPRODUCTION.src.scenario import Scenario, Subtask

from .temporal import PrefixSumTable
from .topology import build_neighbor_index

ACTION_DIM = 10
NEIGHBOR_SHAPE = (6, 6)


class FrontierSchedulingEnvironment:
    """Batched-frontier scheduling environment (Candidates 2, 4, 5).

    Instead of one autoregressive decision per subtask in a fixed list order,
    at each decision epoch every currently-ready subtask (predecessors done,
    arrived) is scored in ONE batched forward pass; each proposes its own best
    (DC, slot); proposals are then committed sequentially (task-id order) with
    a fast feasibility re-check, deferring one to the next epoch only if an
    earlier commit in the same epoch consumed the capacity it needed. This
    cuts network forward passes from one-per-subtask to one-per-epoch, and
    replaces the source reproduction's O(duration) recomputed price/CEF means
    with O(1) prefix-sum lookups and vectorized sliding-window capacity checks
    (weakness #5), while giving the policy visibility across all currently
    competing subtasks instead of a fixed processing order (weakness #2).
    """

    def __init__(self, scenario: Scenario, config: dict, frontier_max: int = 16, use_topology: bool = False):
        self.scenario = scenario
        self.config = config
        env = config["environment"]
        exp = config["experiment"]
        self.horizon = int(exp["horizon"])
        self.n_dc = int(exp["n_datacenters"])
        self.interval_hours = float(exp["interval_hours"])
        self.capacity = np.asarray(env["dc_capacity_kw"], dtype=np.float64)
        self.base_power = np.asarray(env["dc_base_kw"], dtype=np.float64)
        self.bandwidth_capacity = np.asarray(env["bandwidth_capacity_mbps"], dtype=np.float64)
        self.carbon_price = float(env["carbon_price_per_kg"])
        self.carbon_quota = float(env["carbon_quota_kg"])
        self.migration_slots = int(env["remote_migration_slots"])
        self.frontier_max = frontier_max
        self.use_topology = use_topology
        self.carbon_lambda = 0.0

        self.price_table = PrefixSumTable(scenario.prices.astype(np.float64))
        self.cef_table = PrefixSumTable(scenario.cefs.astype(np.float64))

        successors = {t.task_id: [] for t in scenario.tasks}
        for t in scenario.tasks:
            for p in t.predecessors:
                successors[p].append(t.task_id)
        self.successors = successors
        durations = {t.task_id: t.duration for t in scenario.tasks}
        tail_cache: dict[int, int] = {}

        def tail(task_id: int) -> int:
            if task_id not in tail_cache:
                tail_cache[task_id] = max(
                    (durations[c] + tail(c) for c in successors[task_id]), default=0
                )
            return tail_cache[task_id]

        self.tail_duration = {t.task_id: tail(t.task_id) for t in scenario.tasks}
        self.by_id = {t.task_id: t for t in scenario.tasks}
        if use_topology:
            self._neighbor_index = build_neighbor_index(
                scenario, float(self.capacity.max()), float(self.bandwidth_capacity.max()), self.horizon
            )
        self.curves = np.stack([scenario.prices.astype(np.float32), scenario.cefs.astype(np.float32)], axis=1)
        self.reset()

    def set_lambda(self, value: float) -> None:
        self.carbon_lambda = float(value)

    def reset(self):
        self.power = np.zeros((self.n_dc, self.horizon), dtype=np.float64)
        self.bandwidth = np.zeros((self.n_dc, self.horizon), dtype=np.float64)
        self.assignments: dict[int, tuple[int, int, int]] = {}
        self.scheduled: set[int] = set()
        self.fallback_actions = 0
        self.capacity_violations = 0
        self.epoch = 0
        return self.observation()

    @property
    def done(self) -> bool:
        return len(self.scheduled) >= len(self.scenario.tasks)

    def _earliest(self, task: Subtask, dc: int) -> int:
        predecessor_finish = max((self.assignments[p][2] for p in task.predecessors), default=task.arrival)
        migration = self.migration_slots if dc != task.home_dc else 0
        return max(task.arrival + migration, predecessor_finish)

    def _frontier(self) -> list[Subtask]:
        ready = []
        for t in self.scenario.tasks:
            if t.task_id in self.scheduled:
                continue
            if all(p in self.scheduled for p in t.predecessors):
                ready.append(t)
        return ready[: self.frontier_max]

    def _task_mask_and_features(self, task: Subtask) -> tuple[np.ndarray, np.ndarray, bool]:
        mask = np.zeros(self.n_dc * self.horizon, dtype=bool)
        precedence_only = np.zeros_like(mask)
        features = np.zeros((self.n_dc * self.horizon, ACTION_DIM), dtype=np.float32)
        duration = task.duration
        max_start = self.horizon - duration
        if max_start < 0:
            raise RuntimeError(f"Task duration exceeds horizon: {asdict(task)}")
        latest = min(task.deadline, self.horizon) - duration - self.tail_duration[task.task_id]
        starts = np.arange(0, max_start + 1)
        for dc in range(self.n_dc):
            earliest = self._earliest(task, dc)
            ends = starts + duration
            mean_price = self.price_table.window_mean_batch(dc, starts, ends)
            mean_cef = self.cef_table.window_mean_batch(dc, starts, ends)
            power_row = self.base_power[dc] + self.power[dc]
            bw_row = self.bandwidth[dc]
            power_windows = sliding_window_view(power_row, duration).max(axis=-1)
            bw_windows = sliding_window_view(bw_row, duration).max(axis=-1)
            remaining_power = (self.capacity[dc] - power_windows) / self.capacity[dc]
            remaining_bw = (self.bandwidth_capacity[dc] - bw_windows) / self.bandwidth_capacity[dc]
            power_ok = (power_windows + task.power_kw) <= self.capacity[dc] + 1e-6
            bw_ok = (bw_windows + task.bandwidth_mbps) <= self.bandwidth_capacity[dc] + 1e-6
            precedence_ok = (starts >= earliest) & (starts <= latest)
            feasible = precedence_ok & power_ok & bw_ok
            energy = task.power_kw * duration * self.interval_hours
            immediate_cost = energy * (mean_price + self.carbon_price * mean_cef)
            idx = dc * self.horizon + starts
            mask[idx] = feasible
            precedence_only[idx] = precedence_ok
            features[idx, 0] = mean_price
            features[idx, 1] = mean_cef
            features[idx, 2] = remaining_power
            features[idx, 3] = remaining_bw
            features[idx, 4] = starts / self.horizon
            features[idx, 5] = immediate_cost / 500.0
            features[idx, 6] = float(dc != task.home_dc)
            features[idx, 7] = float(dc == 0)
            features[idx, 8] = float(dc == 1)
            features[idx, 9] = float(dc == 2)
        is_fallback = False
        if not mask.any():
            mask = precedence_only
            is_fallback = True
            if not mask.any():
                raise RuntimeError(f"No deadline-feasible action for task {asdict(task)}")
        return mask, features, is_fallback

    def _task_features_vec(self, task: Subtask) -> np.ndarray:
        earliest_home = self._earliest(task, task.home_dc)
        return np.asarray(
            [
                task.arrival / self.horizon,
                task.deadline / self.horizon,
                task.duration / 8.0,
                task.power_kw / float(self.capacity.max()),
                task.bandwidth_mbps / float(self.bandwidth_capacity.max()),
                float(task.home_dc == 0),
                float(task.home_dc == 1),
                float(task.home_dc == 2),
                len(task.predecessors) / 3.0,
                earliest_home / self.horizon,
                max(0, task.deadline - earliest_home - task.duration) / self.horizon,
                task.workload_id / 10.0,
            ],
            dtype=np.float32,
        )

    def observation(self) -> dict[str, np.ndarray]:
        frontier = self._frontier()
        n_actions = self.n_dc * self.horizon
        tasks = np.zeros((self.frontier_max, 12), dtype=np.float32)
        actions = np.zeros((self.frontier_max, n_actions, ACTION_DIM), dtype=np.float32)
        mask = np.zeros((self.frontier_max, n_actions), dtype=bool)
        valid = np.zeros(self.frontier_max, dtype=bool)
        task_ids = -np.ones(self.frontier_max, dtype=np.int64)
        fallback_flags = np.zeros(self.frontier_max, dtype=bool)
        neighbors = np.zeros((self.frontier_max, *NEIGHBOR_SHAPE), dtype=np.float32)
        neighbor_mask = np.zeros((self.frontier_max, NEIGHBOR_SHAPE[0]), dtype=bool)
        for i, task in enumerate(frontier):
            tasks[i] = self._task_features_vec(task)
            m, f, is_fallback = self._task_mask_and_features(task)
            mask[i] = m
            actions[i] = f
            valid[i] = True
            task_ids[i] = task.task_id
            fallback_flags[i] = is_fallback
            if self.use_topology:
                neigh, nmask = self._neighbor_index[task.task_id]
                neighbors[i] = neigh
                neighbor_mask[i] = nmask
        return {
            "tasks": tasks,
            "actions": actions,
            "mask": mask,
            "valid": valid,
            "task_ids": task_ids,
            "fallback": fallback_flags,
            "neighbors": neighbors,
            "neighbor_mask": neighbor_mask,
            "curves": self.curves,
        }

    def step(self, slot_actions: np.ndarray, obs: dict) -> tuple[dict, float, bool, dict]:
        """slot_actions: (frontier_max,) int array, one action index per frontier slot
        (ignored for invalid/pad slots). Commits sequentially in task-id order,
        re-checking feasibility against the post-earlier-commit state."""
        order = np.argsort(np.where(obs["valid"], obs["task_ids"], np.iinfo(np.int64).max))
        total_reward = 0.0
        committed = 0
        for i in order:
            if not obs["valid"][i]:
                continue
            task_id = int(obs["task_ids"][i])
            task = self.by_id[task_id]
            action = int(slot_actions[i])
            dc, start = divmod(action, self.horizon)
            end = start + task.duration
            earliest = self._earliest(task, dc)
            latest = min(task.deadline, self.horizon) - task.duration - self.tail_duration[task_id]
            precedence_ok = earliest <= start <= latest
            power_ok = np.all(self.base_power[dc] + self.power[dc, start:end] + task.power_kw <= self.capacity[dc] + 1e-6)
            bw_ok = np.all(self.bandwidth[dc, start:end] + task.bandwidth_mbps <= self.bandwidth_capacity[dc] + 1e-6)
            if not precedence_ok:
                continue
            if not (power_ok and bw_ok):
                if obs["fallback"][i]:
                    self.fallback_actions += 1
                    self.capacity_violations += 1
                else:
                    continue
            self.power[dc, start:end] += task.power_kw
            self.bandwidth[dc, start:end] += task.bandwidth_mbps
            self.assignments[task_id] = (dc, start, end)
            self.scheduled.add(task_id)
            committed += 1
            energy = task.power_kw * task.duration * self.interval_hours
            mean_price = self.price_table.window_mean(dc, start, end)
            mean_cef = self.cef_table.window_mean(dc, start, end)
            cost_component = energy * mean_price
            emissions_component = energy * mean_cef
            reward = -(cost_component + self.carbon_lambda * emissions_component) / 100.0
            if obs["fallback"][i]:
                reward -= 100.0
            total_reward += reward
        self.epoch += 1
        if committed == 0 and not self.done:
            raise RuntimeError("Deadlock: no frontier commit made in an epoch")
        next_obs = self.observation()
        return next_obs, float(total_reward), self.done, {"committed": committed}

    def metrics(self) -> dict[str, float]:
        total_power = self.power + self.base_power[:, None]
        energy = total_power * self.interval_hours
        electricity_cost = float(np.sum(energy * self.scenario.prices))
        emissions = float(np.sum(energy * self.scenario.cefs))
        carbon_cost = self.carbon_price * (emissions - self.carbon_quota)
        system_cost = electricity_cost + carbon_cost
        remote = sum(int(dc != self.by_id[tid].home_dc) for tid, (dc, _, _) in self.assignments.items())
        delay = [start - self.by_id[tid].arrival for tid, (_, start, _) in self.assignments.items()]
        return {
            "electricity_cost": electricity_cost,
            "carbon_emissions_kg": emissions,
            "carbon_trading_cost": float(carbon_cost),
            "carbon_trading_revenue": float(max(0.0, -carbon_cost)),
            "system_cost": float(system_cost),
            "mean_start_delay_slots": float(np.mean(delay)) if delay else 0.0,
            "remote_subtask_fraction": float(remote / len(self.scenario.tasks)),
            "fallback_actions": float(self.fallback_actions),
            "capacity_violations": float(self.capacity_violations),
            "decision_epochs": float(self.epoch),
        }
