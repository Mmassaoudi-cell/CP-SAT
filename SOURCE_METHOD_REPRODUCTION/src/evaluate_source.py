from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from SOURCE_METHOD_REPRODUCTION.src.environment import SchedulingEnvironment, run_heuristic
from SOURCE_METHOD_REPRODUCTION.src.model import MaskedAttentionActorCritic
from SOURCE_METHOD_REPRODUCTION.src.scenario import generate_scenario, load_config


def run_policy(scenario, config, model, device):
    env = SchedulingEnvironment(scenario, config)
    schedule = []
    latencies = []
    while not env.done:
        obs = env.observation()
        task = env._task()
        task_t = torch.from_numpy(obs["task"])[None].to(device)
        actions_t = torch.from_numpy(obs["actions"])[None].to(device)
        mask_t = torch.from_numpy(obs["mask"])[None].to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            action, _, _, _ = model.act(task_t, actions_t, mask_t, deterministic=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - started) * 1000.0)
        action_int = int(action.item())
        env.step(action_int)
        dc, start = divmod(action_int, env.horizon)
        schedule.append({"task_id": task.task_id, "workload_id": task.workload_id, "dc": dc, "start": start, "end": start + task.duration})
    metrics = env.metrics()
    metrics["mean_decision_latency_ms"] = float(np.mean(latencies))
    metrics["p95_decision_latency_ms"] = float(np.quantile(latencies, 0.95))
    return metrics, schedule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="SOURCE_METHOD_REPRODUCTION/config/source_reproduction.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["reproduction", "validation", "test"], default="reproduction")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = MaskedAttentionActorCritic(
        hidden=int(config["ppo"]["hidden_size"]), heads=int(config["ppo"]["attention_heads"])
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    if args.split == "reproduction":
        seeds = [3691718]
    else:
        start = int(config["experiment"][f"{args.split}_seed_start"])
        count = int(config["experiment"][f"{args.split}_scenarios"])
        seeds = list(range(start, start + count))
    result_rows = []
    schedules = []
    for seed in seeds:
        scenario = generate_scenario(seed, f"{args.split}_{seed}")
        for method in ("s1_zero_delay", "s2_single_dc_temporal", "greedy_spatiotemporal"):
            metrics, schedule = run_heuristic(scenario, config, method)
            result_rows.append({"scenario_id": scenario.scenario_id, "seed": seed, "method": method, **metrics})
            if args.split == "reproduction":
                schedules.extend({"method": method, **row} for row in schedule)
        metrics, schedule = run_policy(scenario, config, model, device)
        result_rows.append({"scenario_id": scenario.scenario_id, "seed": seed, "method": "source_ppo_attention", **metrics})
        if args.split == "reproduction":
            schedules.extend({"method": "source_ppo_attention", **row} for row in schedule)
    output = Path("SOURCE_METHOD_REPRODUCTION/results")
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / f"{args.split}_results.csv"
    pd.DataFrame(result_rows).to_csv(result_path, index=False)
    if schedules:
        pd.DataFrame(schedules).to_csv(output / "reproduction_schedules.csv", index=False)
    print(json.dumps({"results": str(result_path), "rows": len(result_rows), "summary": pd.DataFrame(result_rows).groupby("method")["system_cost"].mean().to_dict()}))


if __name__ == "__main__":
    main()
