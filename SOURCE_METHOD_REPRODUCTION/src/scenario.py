from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class Subtask:
    task_id: int
    workload_id: int
    local_id: int
    home_dc: int
    arrival: int
    deadline: int
    duration: int
    power_kw: float
    bandwidth_mbps: float
    predecessors: tuple[int, ...]


@dataclass
class Scenario:
    scenario_id: str
    seed: int
    prices: np.ndarray
    cefs: np.ndarray
    tasks: list[Subtask]


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _paper_like_market(seed: int, horizon: int = 96) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the qualitative ranges/patterns in Figs. 5-6.

    These are synthetic traces, not digitized or author-provided observations.
    """
    rng = np.random.default_rng(seed)
    hour = np.arange(horizon) / 4.0
    daytime = (np.sin(2 * np.pi * (hour - 6.0) / 24.0) + 1.0) / 2.0
    evening = np.exp(-0.5 * ((hour - 19.0) / 2.5) ** 2)

    prices = np.vstack(
        [
            0.20 + 0.70 * daytime + 0.08 * evening,
            0.82 - 0.54 * daytime + 0.10 * np.exp(-0.5 * ((hour - 2.0) / 2.7) ** 2),
            0.48 + 0.10 * np.sin(2 * np.pi * (hour - 4.0) / 24.0),
        ]
    )
    prices += rng.normal(0.0, 0.025, prices.shape)
    prices = np.clip(prices, 0.12, 1.02)

    cefs = np.vstack(
        [
            0.42 - 0.28 * daytime,
            0.66 + 0.18 * np.sin(2 * np.pi * (hour + 2.0) / 24.0),
            0.44 + 0.16 * np.sin(2 * np.pi * (hour - 7.0) / 24.0),
        ]
    )
    cefs += rng.normal(0.0, 0.018, cefs.shape)
    cefs[0] = np.clip(cefs[0], 0.10, 0.50)
    cefs[1] = np.clip(cefs[1], 0.50, 0.90)
    cefs[2] = np.clip(cefs[2], 0.20, 0.70)
    return prices.astype(np.float32), cefs.astype(np.float32)


def _edges(kind: str, n: int) -> list[tuple[int, int]]:
    if kind == "pipeline":
        return [(i, i + 1) for i in range(n - 1)]
    if kind == "fork_join":
        edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
        edges.extend((i, i + 1) for i in range(3, n - 1))
        return edges
    if kind == "tree":
        edges = [(0, 2), (1, 2), (0, 3), (1, 3)]
        if n >= 5:
            edges += [(2, 4), (3, 4)]
        edges.extend((i, i + 1) for i in range(4, n - 1))
        return edges
    if kind == "shared":
        edges = [(0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
        edges.extend((i, i + 1) for i in range(3, n - 1))
        return edges
    raise ValueError(kind)


def _topological_order(n: int, edges: Iterable[tuple[int, int]]) -> list[int]:
    successors = {i: [] for i in range(n)}
    indegree = np.zeros(n, dtype=int)
    for source, target in edges:
        successors[source].append(target)
        indegree[target] += 1
    queue = [i for i in range(n) if indegree[i] == 0]
    order: list[int] = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for target in successors[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if len(order) != n:
        raise ValueError("Generated workload is not acyclic")
    return order


def generate_scenario(seed: int, scenario_id: str | None = None, horizon: int = 96) -> Scenario:
    rng = np.random.default_rng(seed)
    prices, cefs = _paper_like_market(seed + 97, horizon)
    sizes = [5, 4, 6, 5, 5, 4, 6, 5, 5, 5]
    kinds = ["pipeline", "fork_join", "tree", "shared"]
    arrivals = np.sort(rng.integers(0, 60, size=len(sizes)))
    tasks: list[Subtask] = []
    global_id = 0
    for workload_id, (size, arrival) in enumerate(zip(sizes, arrivals, strict=True)):
        kind = kinds[workload_id % len(kinds)]
        edges = _edges(kind, size)
        order = _topological_order(size, edges)
        pred_local = {i: [] for i in range(size)}
        for source, target in edges:
            pred_local[target].append(source)
        durations = rng.integers(1, 5, size=size)
        critical_path = int(durations.sum()) if kind == "pipeline" else int(durations.sum() * 0.72)
        slack = int(rng.integers(10, 24))
        deadline = min(horizon, int(arrival) + critical_path + slack)
        deadline = max(deadline, int(arrival) + 12)
        home_dc = int(rng.integers(0, 3))
        local_to_global: dict[int, int] = {}
        for local_id in order:
            task_id = global_id
            global_id += 1
            local_to_global[local_id] = task_id
            predecessors = tuple(local_to_global[p] for p in pred_local[local_id])
            tasks.append(
                Subtask(
                    task_id=task_id,
                    workload_id=workload_id,
                    local_id=local_id,
                    home_dc=home_dc,
                    arrival=int(arrival),
                    deadline=int(deadline),
                    duration=int(durations[local_id]),
                    power_kw=float(rng.uniform(90.0, 260.0)),
                    bandwidth_mbps=float(rng.uniform(250.0, 950.0)),
                    predecessors=predecessors,
                )
            )
    return Scenario(
        scenario_id=scenario_id or f"scenario_{seed}",
        seed=seed,
        prices=prices,
        cefs=cefs,
        tasks=tasks,
    )


def write_split_manifest(config: dict, output_path: str | Path) -> None:
    exp = config["experiment"]
    rows = []
    for split, count_key, seed_key in (
        ("train", "train_scenarios", "train_seed_start"),
        ("validation", "validation_scenarios", "validation_seed_start"),
        ("test", "test_scenarios", "test_seed_start"),
    ):
        for offset in range(int(exp[count_key])):
            seed = int(exp[seed_key]) + offset
            rows.append(
                {
                    "scenario_id": f"reconstructed_{split}_{offset:04d}",
                    "split": split,
                    "seed": seed,
                    "source": "synthetic_paper_like_reconstruction",
                    "selection_status": "frozen_before_model_screening",
                }
            )
    pd.DataFrame(rows).to_csv(output_path, index=False)
