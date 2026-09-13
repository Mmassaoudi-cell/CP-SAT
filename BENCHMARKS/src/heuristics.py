from __future__ import annotations

import numpy as np

from SOURCE_METHOD_REPRODUCTION.src.environment import SchedulingEnvironment


def run_edf(scenario, config) -> tuple[dict, list[dict]]:
    """Earliest-Deadline-First: among feasible (dc, start), always pick the
    task's own feasible slot that starts soonest, preferring the home DC only
    when it ties (a standard, dependency-agnostic scheduling heuristic)."""
    env = SchedulingEnvironment(scenario, config)
    schedule = []
    while not env.done:
        obs = env.observation()
        task = env._task()
        feasible = np.flatnonzero(obs["mask"])
        starts = feasible % env.horizon
        action = int(feasible[np.argmin(starts)])
        env.step(action)
        dc, start = divmod(action, env.horizon)
        schedule.append({"task_id": task.task_id, "workload_id": task.workload_id, "dc": dc, "start": start, "end": start + task.duration})
    return env.metrics(), schedule


def run_carbon_greedy(scenario, config) -> tuple[dict, list[dict]]:
    """Always pick the feasible (dc, start) with the lowest mean CEF, ignoring
    price -- a carbon-only heuristic used to bound how much emissions reduction
    is achievable by carbon-mindedness alone, without cost awareness."""
    env = SchedulingEnvironment(scenario, config)
    schedule = []
    while not env.done:
        obs = env.observation()
        task = env._task()
        feasible = np.flatnonzero(obs["mask"])
        cef = obs["actions"][feasible, 1]
        action = int(feasible[np.argmin(cef)])
        env.step(action)
        dc, start = divmod(action, env.horizon)
        schedule.append({"task_id": task.task_id, "workload_id": task.workload_id, "dc": dc, "start": start, "end": start + task.duration})
    return env.metrics(), schedule


def run_random_feasible(scenario, config, rng: np.random.Generator) -> tuple[dict, list[dict]]:
    """Uniformly random feasible (dc, start) per task -- a floor baseline."""
    env = SchedulingEnvironment(scenario, config)
    schedule = []
    while not env.done:
        obs = env.observation()
        task = env._task()
        feasible = np.flatnonzero(obs["mask"])
        action = int(rng.choice(feasible))
        env.step(action)
        dc, start = divmod(action, env.horizon)
        schedule.append({"task_id": task.task_id, "workload_id": task.workload_id, "dc": dc, "start": start, "end": start + task.duration})
    return env.metrics(), schedule


def _upward_rank(scenario) -> dict[int, float]:
    """HEFT-style upward rank: expected remaining critical-path cost from each
    task to a sink, using mean cost-per-slot as the per-task weight. Tasks with
    higher rank are scheduled first (standard HEFT priority ordering)."""
    successors = {t.task_id: [] for t in scenario.tasks}
    by_id = {t.task_id: t for t in scenario.tasks}
    for t in scenario.tasks:
        for p in t.predecessors:
            successors[p].append(t.task_id)
    mean_weight = {t.task_id: t.duration * t.power_kw for t in scenario.tasks}
    rank_cache: dict[int, float] = {}

    def rank(task_id: int) -> float:
        if task_id not in rank_cache:
            rank_cache[task_id] = mean_weight[task_id] + max(
                (rank(c) for c in successors[task_id]), default=0.0
            )
        return rank_cache[task_id]

    return {t.task_id: rank(t.task_id) for t in scenario.tasks}


def run_heft(scenario, config) -> tuple[dict, list[dict]]:
    """HEFT-style list scheduler: at every decision point, among all currently
    dependency-ready tasks, always schedule the one with the highest upward
    rank (standard dependency-aware DAG-scheduling priority) to its cheapest
    feasible (dc, start). Unlike the source reproduction's fixed task-id
    processing order, this heuristic can freely reorder ready tasks by
    criticality -- the strongest classical, non-learned, explicitly
    dependency-aware baseline. Implemented directly on
    FrontierSchedulingEnvironment's vectorized feasibility computation."""
    from PROPOSED_METHOD.src.env_frontier import FrontierSchedulingEnvironment

    env = FrontierSchedulingEnvironment(scenario, config, frontier_max=len(scenario.tasks), use_topology=False)
    rank = _upward_rank(scenario)
    schedule = []
    while not env.done:
        frontier = env._frontier()
        task = max(frontier, key=lambda t: rank[t.task_id])
        mask, features, is_fallback = env._task_mask_and_features(task)
        feasible = np.flatnonzero(mask)
        scores = features[feasible, 5]
        action = int(feasible[np.argmin(scores)])
        dc, start = divmod(action, env.horizon)
        end = start + task.duration
        if is_fallback:
            env.fallback_actions += 1
            if np.any(env.base_power[dc] + env.power[dc, start:end] + task.power_kw > env.capacity[dc] + 1e-6):
                env.capacity_violations += 1
        env.power[dc, start:end] += task.power_kw
        env.bandwidth[dc, start:end] += task.bandwidth_mbps
        env.assignments[task.task_id] = (dc, start, end)
        env.scheduled.add(task.task_id)
        env.epoch += 1
        schedule.append({"task_id": task.task_id, "workload_id": task.workload_id, "dc": dc, "start": start, "end": end})
    return env.metrics(), schedule


HEURISTICS = {
    "edf": run_edf,
    "carbon_greedy": run_carbon_greedy,
    "heft": run_heft,
}
