from __future__ import annotations

import numpy as np

from SOURCE_METHOD_REPRODUCTION.src.scenario import Scenario, Subtask, _edges, _topological_order
from REALDATA.src.real_traces import (
    cef_from_renewable_fraction,
    extract_day,
    load_real_price_series,
    load_real_wind_series,
    rescale_to_range,
    resample_to_15min,
)

# Same per-DC CEF operating ranges as the synthetic generator (scenario.py
# _paper_like_market), preserved for comparability; only the temporal SHAPE
# is replaced with real data.
CEF_RANGES = [(0.10, 0.50), (0.50, 0.90), (0.20, 0.70)]
PRICE_TARGET_MEAN_STD = [(0.45, 0.28), (0.62, 0.22), (0.48, 0.10)]  # DC1/2/3, matches
# the qualitative source-paper pattern (DC1 high-volatility day-peaking, DC2
# high-volatility night-peaking via the real day chosen, DC3 low-volatility)
# approximately -- exact values are not claimed to match unavailable author data.

_PRICE_15MIN = None
_WIND_15MIN = {}


def _get_price_series():
    global _PRICE_15MIN
    if _PRICE_15MIN is None:
        _PRICE_15MIN = resample_to_15min(load_real_price_series()).dropna()
    return _PRICE_15MIN


def _get_wind_series(zone: int):
    if zone not in _WIND_15MIN:
        _WIND_15MIN[zone] = resample_to_15min(load_real_wind_series(zone)).dropna()
    return _WIND_15MIN[zone]


def _random_valid_day(series, rng: np.random.Generator, horizon: int = 96, max_tries: int = 200):
    days = series.index.normalize().unique()
    for _ in range(max_tries):
        day = days[rng.integers(0, len(days))]
        window = extract_day(series, day, horizon)
        if window is not None:
            return window
    raise RuntimeError("Could not find a valid real-data day after max_tries")


def generate_real_calibrated_scenario(seed: int, scenario_id: str | None = None, horizon: int = 96) -> Scenario:
    """Same workload/DAG generative process as SOURCE_METHOD_REPRODUCTION's
    synthetic generate_scenario, but price and CEF traces are drawn from real
    GEFCom2014 electricity-price and wind-power data (resampled to 15-minute
    resolution) instead of a hand-authored sinusoid model. Documented,
    disclosed rescaling matches each DC's operating range to the rest of this
    study for comparability (see REALDATA/src/real_traces.py); the temporal
    SHAPE (volatility, day-to-day predictability) is genuinely real, not
    synthetic. No workload/subtask data is available publicly for this
    problem, so that part of the generative process is unchanged."""
    rng = np.random.default_rng(seed)
    price_series = _get_price_series()
    wind_zones = [1, 5, 9]

    prices = np.zeros((3, horizon), dtype=np.float32)
    cefs = np.zeros((3, horizon), dtype=np.float32)
    for dc in range(3):
        raw_price = _random_valid_day(price_series, rng, horizon)
        target_mean, target_std = PRICE_TARGET_MEAN_STD[dc]
        prices[dc] = np.clip(rescale_to_range(raw_price, target_mean, target_std), 0.05, 1.5)

        wind_series = _get_wind_series(wind_zones[dc])
        raw_wind = _random_valid_day(wind_series, rng, horizon)
        cef_min, cef_max = CEF_RANGES[dc]
        cefs[dc] = cef_from_renewable_fraction(raw_wind, cef_min, cef_max)

    sizes = [5, 4, 6, 5, 5, 4, 6, 5, 5, 5]
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
                    task_id=task_id, workload_id=workload_id, local_id=local_id, home_dc=home_dc,
                    arrival=int(arrival), deadline=int(deadline), duration=int(durations[local_id]),
                    power_kw=float(rng.uniform(90.0, 260.0)), bandwidth_mbps=float(rng.uniform(250.0, 950.0)),
                    predecessors=predecessors,
                )
            )
    return Scenario(scenario_id=scenario_id or f"real_calibrated_{seed}", seed=seed, prices=prices, cefs=cefs, tasks=tasks)
