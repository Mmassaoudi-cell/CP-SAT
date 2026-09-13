from __future__ import annotations

import copy

import numpy as np

from SOURCE_METHOD_REPRODUCTION.src.scenario import (
    Scenario,
    Subtask,
    _edges,
    _paper_like_market,
    _topological_order,
)


def with_market_noise(scenario: Scenario, price_noise_std: float = 0.0, cef_noise_std: float = 0.0, seed: int = 0) -> Scenario:
    """Additive Gaussian sensor/forecast noise on price and CEF curves, on top
    of the frozen scenario's own generation noise -- tests robustness to
    market/forecast uncertainty without touching workload structure."""
    rng = np.random.default_rng(seed)
    prices = scenario.prices + rng.normal(0.0, price_noise_std, scenario.prices.shape).astype(np.float32)
    cefs = scenario.cefs + rng.normal(0.0, cef_noise_std, scenario.cefs.shape).astype(np.float32)
    prices = np.clip(prices, 0.05, 1.5).astype(np.float32)
    cefs = np.clip(cefs, 0.05, 1.2).astype(np.float32)
    return Scenario(scenario_id=f"{scenario.scenario_id}_noise_p{price_noise_std}_c{cef_noise_std}", seed=scenario.seed, prices=prices, cefs=cefs, tasks=scenario.tasks)


def with_missing_bandwidth_signal(scenario: Scenario, dropout_prob: float = 0.0, seed: int = 0) -> Scenario:
    """Simulates missing/corrupted per-task bandwidth demand readings by
    replacing a random subset of subtasks' bandwidth demand with the fleet
    mean (a common, deployable degradation: a sensor/telemetry dropout), while
    leaving the true environment capacity constraints untouched (so infeasible
    corrupted plans still get penalized) -- tests policy robustness to
    corrupted observations, a realistic operational condition not covered by
    the source paper's clean-data evaluation."""
    rng = np.random.default_rng(seed)
    mean_bw = float(np.mean([t.bandwidth_mbps for t in scenario.tasks]))
    tasks = []
    for t in scenario.tasks:
        if rng.random() < dropout_prob:
            tasks.append(Subtask(**{**t.__dict__, "bandwidth_mbps": mean_bw}))
        else:
            tasks.append(t)
    return Scenario(scenario_id=f"{scenario.scenario_id}_bwdrop{dropout_prob}", seed=scenario.seed, prices=scenario.prices, cefs=scenario.cefs, tasks=tasks)


def generate_scenario_variant(seed: int, n_workloads: int = 10, slack_scale: float = 1.0, horizon: int = 96) -> Scenario:
    """Same generative process as generate_scenario, but with a configurable
    workload count (replicates the source paper's 6/7/8/9/10-workload
    sensitivity experiment) and a deadline-slack scale factor (tighter/looser
    deadlines) for robustness testing."""
    rng = np.random.default_rng(seed)
    prices, cefs = _paper_like_market(seed + 97, horizon)
    base_sizes = [5, 4, 6, 5, 5, 4, 6, 5, 5, 5]
    sizes = base_sizes[:n_workloads] if n_workloads <= len(base_sizes) else base_sizes + [5] * (n_workloads - len(base_sizes))
    kinds = ["pipeline", "fork_join", "tree", "shared"]
    arrivals = np.sort(rng.integers(0, 60, size=len(sizes)))
    tasks: list[Subtask] = []
    global_id = 0
    for workload_id, (size, arrival) in enumerate(zip(sizes, arrivals, strict=True)):
        kind = kinds[workload_id % len(kinds)]
        edges = _edges(kind, size)
        order = _topological_order(size, edges)
        pred_local: dict[int, list[int]] = {i: [] for i in range(size)}
        for source, target in edges:
            pred_local[target].append(source)
        durations = rng.integers(1, 5, size=size)
        critical_path = int(durations.sum()) if kind == "pipeline" else int(durations.sum() * 0.72)
        slack = max(4, int(rng.integers(10, 24) * slack_scale))
        deadline = min(horizon, int(arrival) + critical_path + slack)
        deadline = max(deadline, int(arrival) + max(12, int(12 * slack_scale)))
        home_dc = int(rng.integers(0, 3))
        local_to_global: dict[int, int] = {}
        for local_id in order:
            task_id = global_id
            global_id += 1
            local_to_global[local_id] = task_id
            predecessors = tuple(local_to_global[p] for p in pred_local[local_id])
            tasks.append(
                Subtask(
                    task_id=task_id, workload_id=workload_id, local_id=local_id, home_dc=home_dc,
                    arrival=int(arrival), deadline=int(deadline), duration=int(durations[local_id]),
                    power_kw=float(rng.uniform(90.0, 260.0)), bandwidth_mbps=float(rng.uniform(250.0, 950.0)),
                    predecessors=predecessors,
                )
            )
    return Scenario(scenario_id=f"variant_{seed}_w{n_workloads}_s{slack_scale}", seed=seed, prices=prices, cefs=cefs, tasks=tasks)
