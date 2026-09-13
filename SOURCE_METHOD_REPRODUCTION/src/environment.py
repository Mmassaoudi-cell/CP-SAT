from __future__ import annotations

from dataclasses import asdict

import numpy as np

from .scenario import Scenario, Subtask


class SchedulingEnvironment:
    """Offline day-ahead, dependency-aware scheduling environment.

    One decision chooses a DC and start slot for the next topologically valid
    subtask. Dynamic masks enforce precedence, deadline, power, and bandwidth.
    """

    def __init__(self, scenario: Scenario, config: dict):
        self.scenario = scenario
        self.config = config
        env = config["environment"]
        exp = config["experiment"]
        self.horizon = int(exp["horizon"])
        self.n_dc = int(exp["n_datacenters"])
        self.interval_hours = float(exp["interval_hours"])
        self.capacity = np.asarray(env["dc_capacity_kw"], dtype=np.float32)
        self.base_power = np.asarray(env["dc_base_kw"], dtype=np.float32)
        self.bandwidth_capacity = np.asarray(env["bandwidth_capacity_mbps"], dtype=np.float32)
        self.carbon_price = float(env["carbon_price_per_kg"])
        self.carbon_quota = float(env["carbon_quota_kg"])
        self.migration_slots = int(env["remote_migration_slots"])
        self.action_size = self.n_dc * self.horizon
        successors = {task.task_id: [] for task in scenario.tasks}
        for task in scenario.tasks:
            for predecessor in task.predecessors:
                successors[predecessor].append(task.task_id)
        durations = {task.task_id: task.duration for task in scenario.tasks}
        tail_cache: dict[int, int] = {}

        def tail(task_id: int) -> int:
            if task_id not in tail_cache:
                tail_cache[task_id] = max(
                    (durations[child] + tail(child) for child in successors[task_id]),
                    default=0,
                )
            return tail_cache[task_id]

        self.tail_duration = {task.task_id: tail(task.task_id) for task in scenario.tasks}
        self.reset()

    def reset(self) -> dict[str, np.ndarray]:
        self.cursor = 0
        self.power = np.zeros((self.n_dc, self.horizon), dtype=np.float32)
        self.bandwidth = np.zeros((self.n_dc, self.horizon), dtype=np.float32)
        self.assignments: dict[int, tuple[int, int, int]] = {}
        self.fallback_actions = 0
        self.capacity_violations = 0
        return self.observation()

    @property
    def done(self) -> bool:
        return self.cursor >= len(self.scenario.tasks)

    def _task(self) -> Subtask:
        return self.scenario.tasks[self.cursor]

    def _earliest(self, task: Subtask, dc: int) -> int:
        predecessor_finish = max((self.assignments[p][2] for p in task.predecessors), default=task.arrival)
        migration = self.migration_slots if dc != task.home_dc else 0
        return max(task.arrival + migration, predecessor_finish)

    def _feasible(self, task: Subtask) -> np.ndarray:
        mask = np.zeros(self.action_size, dtype=bool)
        precedence_only = np.zeros_like(mask)
        for dc in range(self.n_dc):
            earliest = self._earliest(task, dc)
            latest = min(task.deadline, self.horizon) - task.duration - self.tail_duration[task.task_id]
            for start in range(max(0, earliest), latest + 1):
                index = dc * self.horizon + start
                precedence_only[index] = True
                end = start + task.duration
                power_ok = np.all(
                    self.base_power[dc] + self.power[dc, start:end] + task.power_kw <= self.capacity[dc] + 1e-6
                )
                bandwidth_ok = np.all(
                    self.bandwidth[dc, start:end] + task.bandwidth_mbps <= self.bandwidth_capacity[dc] + 1e-6
                )
                if power_ok and bandwidth_ok:
                    mask[index] = True
        if not mask.any():
            mask = precedence_only
            self._mask_is_fallback = True
        else:
            self._mask_is_fallback = False
        if not mask.any():
            raise RuntimeError(f"No deadline-feasible action for task {asdict(task)}")
        return mask

    def observation(self) -> dict[str, np.ndarray]:
        if self.done:
            task_features = np.zeros(12, dtype=np.float32)
            action_features = np.zeros((self.action_size, 10), dtype=np.float32)
            action_mask = np.ones(self.action_size, dtype=bool)
            return {"task": task_features, "actions": action_features, "mask": action_mask}
        task = self._task()
        mask = self._feasible(task)
        earliest_home = self._earliest(task, task.home_dc)
        task_features = np.asarray(
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
        features = np.zeros((self.action_size, 10), dtype=np.float32)
        for dc in range(self.n_dc):
            for start in range(self.horizon):
                end = min(self.horizon, start + task.duration)
                duration = max(1, end - start)
                mean_price = float(self.scenario.prices[dc, start:end].mean())
                mean_cef = float(self.scenario.cefs[dc, start:end].mean())
                remaining_power = float(
                    np.min(self.capacity[dc] - self.base_power[dc] - self.power[dc, start:end]) / self.capacity[dc]
                )
                remaining_bw = float(
                    np.min(self.bandwidth_capacity[dc] - self.bandwidth[dc, start:end]) / self.bandwidth_capacity[dc]
                )
                energy = task.power_kw * duration * self.interval_hours
                immediate_cost = energy * (mean_price + self.carbon_price * mean_cef)
                features[dc * self.horizon + start] = np.asarray(
                    [
                        mean_price,
                        mean_cef,
                        remaining_power,
                        remaining_bw,
                        start / self.horizon,
                        immediate_cost / 500.0,
                        float(dc != task.home_dc),
                        float(dc == 0),
                        float(dc == 1),
                        float(dc == 2),
                    ],
                    dtype=np.float32,
                )
        return {"task": task_features, "actions": features, "mask": mask}

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, dict]:
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
        incremental_system_cost = energy * (mean_price + self.carbon_price * mean_cef)
        reward = -incremental_system_cost / 100.0
        if self._mask_is_fallback:
            reward -= 100.0
        self.cursor += 1
        done = self.done
        next_obs = self.observation()
        return next_obs, float(reward), done, {}

    def metrics(self) -> dict[str, float]:
        total_power = self.power + self.base_power[:, None]
        energy = total_power * self.interval_hours
        electricity_cost = float(np.sum(energy * self.scenario.prices))
        emissions = float(np.sum(energy * self.scenario.cefs))
        carbon_cost = self.carbon_price * (emissions - self.carbon_quota)
        system_cost = electricity_cost + carbon_cost
        remote = sum(int(dc != self.scenario.tasks[task_id].home_dc) for task_id, (dc, _, _) in self.assignments.items())
        delay = []
        for task in self.scenario.tasks:
            _, start, _ = self.assignments[task.task_id]
            delay.append(start - task.arrival)
        return {
            "electricity_cost": electricity_cost,
            "carbon_emissions_kg": emissions,
            "carbon_trading_cost": float(carbon_cost),
            "carbon_trading_revenue": float(max(0.0, -carbon_cost)),
            "system_cost": float(system_cost),
            "mean_start_delay_slots": float(np.mean(delay)),
            "remote_subtask_fraction": float(remote / len(self.scenario.tasks)),
            "fallback_actions": float(self.fallback_actions),
            "capacity_violations": float(self.capacity_violations),
        }


def run_heuristic(scenario: Scenario, config: dict, method: str) -> tuple[dict[str, float], list[dict]]:
    env = SchedulingEnvironment(scenario, config)
    schedule = []
    while not env.done:
        obs = env.observation()
        task = env._task()
        feasible = np.flatnonzero(obs["mask"])
        if method == "s1_zero_delay":
            home = feasible[feasible // env.horizon == task.home_dc]
            pool = home if len(home) else feasible
            action = int(pool[np.argmin(pool % env.horizon)])
        elif method == "s2_single_dc_temporal":
            home = feasible[feasible // env.horizon == task.home_dc]
            pool = home if len(home) else feasible
            scores = obs["actions"][pool, 5]
            action = int(pool[np.argmin(scores)])
        elif method == "greedy_spatiotemporal":
            scores = obs["actions"][feasible, 5]
            action = int(feasible[np.argmin(scores)])
        else:
            raise ValueError(method)
        _, _, _, _ = env.step(action)
        dc, start = divmod(action, env.horizon)
        schedule.append({"task_id": task.task_id, "workload_id": task.workload_id, "dc": dc, "start": start, "end": start + task.duration})
    return env.metrics(), schedule
