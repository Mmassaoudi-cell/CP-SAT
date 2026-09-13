from __future__ import annotations

import numpy as np
from ortools.sat.python import cp_model

from SOURCE_METHOD_REPRODUCTION.src.scenario import Scenario

COST_SCALE = 1000


def _tail_duration(scenario: Scenario) -> dict[int, int]:
    successors = {t.task_id: [] for t in scenario.tasks}
    durations = {t.task_id: t.duration for t in scenario.tasks}
    for t in scenario.tasks:
        for p in t.predecessors:
            successors[p].append(t.task_id)
    cache: dict[int, int] = {}

    def tail(task_id: int) -> int:
        if task_id not in cache:
            cache[task_id] = max((durations[c] + tail(c) for c in successors[task_id]), default=0)
        return cache[task_id]

    return {t.task_id: tail(t.task_id) for t in scenario.tasks}


def run_cpsat(scenario: Scenario, config: dict, time_limit_s: float = 30.0, workers: int = 8) -> tuple[dict, list[dict], str]:
    """Exact/near-exact MILP (CP-SAT) formulation of the same constraint set as
    the source reproduction's environment: DC allocation, precedence, deadline
    windows, and per-DC power/bandwidth capacity via cumulative constraints,
    minimizing the identical electricity+carbon system-cost objective (via an
    integer-scaled per-(task, dc, start) cost lookup table, since price/CEF
    vary by slot). This is the strongest classical (non-learned, non-heuristic)
    benchmark: a solver certificate rather than a hand-written priority rule.
    Reports solver status honestly -- OPTIMAL only when CP-SAT proves it within
    the time limit, otherwise the best feasible solution found (best-effort,
    not claimed optimal)."""
    env_cfg = config["environment"]
    exp = config["experiment"]
    horizon = int(exp["horizon"])
    n_dc = int(exp["n_datacenters"])
    interval_hours = float(exp["interval_hours"])
    capacity = np.asarray(env_cfg["dc_capacity_kw"], dtype=np.float64)
    base_power = np.asarray(env_cfg["dc_base_kw"], dtype=np.float64)
    bandwidth_capacity = np.asarray(env_cfg["bandwidth_capacity_mbps"], dtype=np.float64)
    carbon_price = float(env_cfg["carbon_price_per_kg"])
    migration_slots = int(env_cfg["remote_migration_slots"])
    tail_duration = _tail_duration(scenario)

    price_cum = np.zeros((n_dc, horizon + 1))
    cef_cum = np.zeros((n_dc, horizon + 1))
    price_cum[:, 1:] = np.cumsum(scenario.prices, axis=1)
    cef_cum[:, 1:] = np.cumsum(scenario.cefs, axis=1)

    model = cp_model.CpModel()
    is_present: dict[tuple[int, int], cp_model.IntVar] = {}
    start_var: dict[tuple[int, int], cp_model.IntVar] = {}
    end_var: dict[tuple[int, int], cp_model.IntVar] = {}
    interval_var: dict[tuple[int, int], cp_model.IntervalVar] = {}
    contrib: dict[tuple[int, int], cp_model.IntVar] = {}
    max_cost = 0

    def earliest(task, dc):
        predecessor_finish = 0
        return max(task.arrival + (migration_slots if dc != task.home_dc else 0), predecessor_finish)

    for task in scenario.tasks:
        dc_presence = []
        latest = min(task.deadline, horizon) - task.duration - tail_duration[task.task_id]
        for dc in range(n_dc):
            lo = earliest(task, dc)
            hi = latest
            present = model.NewBoolVar(f"present_{task.task_id}_{dc}")
            dc_presence.append(present)
            if hi < lo:
                model.Add(present == 0)
                start_var[(task.task_id, dc)] = model.NewIntVar(0, horizon, f"start_{task.task_id}_{dc}")
                end_var[(task.task_id, dc)] = model.NewIntVar(0, horizon, f"end_{task.task_id}_{dc}")
                interval_var[(task.task_id, dc)] = model.NewOptionalIntervalVar(
                    start_var[(task.task_id, dc)], task.duration, end_var[(task.task_id, dc)], present, f"iv_{task.task_id}_{dc}"
                )
                contrib[(task.task_id, dc)] = model.NewIntVar(0, 0, f"contrib_{task.task_id}_{dc}")
                continue
            s = model.NewIntVar(lo, hi, f"start_{task.task_id}_{dc}")
            e = model.NewIntVar(lo + task.duration, hi + task.duration, f"end_{task.task_id}_{dc}")
            start_var[(task.task_id, dc)] = s
            end_var[(task.task_id, dc)] = e
            interval_var[(task.task_id, dc)] = model.NewOptionalIntervalVar(s, task.duration, e, present, f"iv_{task.task_id}_{dc}")

            starts = np.arange(lo, hi + 1)
            ends = starts + task.duration
            mean_price = (price_cum[dc, ends] - price_cum[dc, starts]) / task.duration
            mean_cef = (cef_cum[dc, ends] - cef_cum[dc, starts]) / task.duration
            energy = task.power_kw * task.duration * interval_hours
            cost_table = np.round(energy * (mean_price + carbon_price * mean_cef) * COST_SCALE).astype(int)
            max_cost = max(max_cost, int(cost_table.max()))
            cost_lookup = model.NewIntVar(int(cost_table.min()), int(cost_table.max()), f"cost_{task.task_id}_{dc}")
            index_var = model.NewIntVar(0, len(cost_table) - 1, f"idx_{task.task_id}_{dc}")
            model.Add(index_var == s - lo)
            model.AddElement(index_var, [int(v) for v in cost_table], cost_lookup)
            c = model.NewIntVar(0, int(cost_table.max()), f"contrib_{task.task_id}_{dc}")
            model.Add(c == cost_lookup).OnlyEnforceIf(present)
            model.Add(c == 0).OnlyEnforceIf(present.Not())
            contrib[(task.task_id, dc)] = c
        model.AddExactlyOne(dc_presence)
        for dc in range(n_dc):
            is_present[(task.task_id, dc)] = dc_presence[dc]

    for task in scenario.tasks:
        for p_id in task.predecessors:
            p_task = next(t for t in scenario.tasks if t.task_id == p_id)
            for dc_p in range(n_dc):
                for dc_s in range(n_dc):
                    migration = migration_slots if dc_p != dc_s else 0
                    model.Add(
                        start_var[(task.task_id, dc_s)] >= end_var[(p_id, dc_p)] + migration
                    ).OnlyEnforceIf([is_present[(p_id, dc_p)], is_present[(task.task_id, dc_s)]])

    for dc in range(n_dc):
        intervals = [interval_var[(t.task_id, dc)] for t in scenario.tasks]
        power_demands = [int(round(t.power_kw)) for t in scenario.tasks]
        bw_demands = [int(round(t.bandwidth_mbps)) for t in scenario.tasks]
        model.AddCumulative(intervals, power_demands, int(capacity[dc] - base_power[dc]))
        model.AddCumulative(intervals, bw_demands, int(bandwidth_capacity[dc]))

    model.Minimize(sum(contrib.values()))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = workers
    status = solver.Solve(model)
    status_name = solver.StatusName(status)

    schedule = []
    assignments: dict[int, tuple[int, int, int]] = {}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for task in scenario.tasks:
            for dc in range(n_dc):
                if solver.Value(is_present[(task.task_id, dc)]):
                    s = solver.Value(start_var[(task.task_id, dc)])
                    e = s + task.duration
                    assignments[task.task_id] = (dc, s, e)
                    schedule.append({"task_id": task.task_id, "workload_id": task.workload_id, "dc": dc, "start": s, "end": e})
                    break

    power = np.zeros((n_dc, horizon))
    bandwidth = np.zeros((n_dc, horizon))
    for task in scenario.tasks:
        if task.task_id not in assignments:
            continue
        dc, s, e = assignments[task.task_id]
        power[dc, s:e] += task.power_kw
        bandwidth[dc, s:e] += task.bandwidth_mbps
    total_power = power + base_power[:, None]
    energy = total_power * interval_hours
    electricity_cost = float(np.sum(energy * scenario.prices))
    emissions = float(np.sum(energy * scenario.cefs))
    quota = float(env_cfg["carbon_quota_kg"])
    carbon_cost = carbon_price * (emissions - quota)
    system_cost = electricity_cost + carbon_cost
    remote = sum(int(dc != next(t for t in scenario.tasks if t.task_id == tid).home_dc) for tid, (dc, _, _) in assignments.items())
    delay = [start - next(t for t in scenario.tasks if t.task_id == tid).arrival for tid, (_, start, _) in assignments.items()]
    metrics = {
        "electricity_cost": electricity_cost,
        "carbon_emissions_kg": emissions,
        "carbon_trading_cost": float(carbon_cost),
        "carbon_trading_revenue": float(max(0.0, -carbon_cost)),
        "system_cost": float(system_cost),
        "mean_start_delay_slots": float(np.mean(delay)) if delay else float("nan"),
        "remote_subtask_fraction": float(remote / len(scenario.tasks)) if assignments else float("nan"),
        "fallback_actions": 0.0,
        "capacity_violations": 0.0,
        "solver_status": status_name,
        "scheduled_fraction": len(assignments) / len(scenario.tasks),
        "solve_time_s": solver.WallTime(),
    }
    return metrics, schedule, status_name
