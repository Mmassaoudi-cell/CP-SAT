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

from SOURCE_METHOD_REPRODUCTION.src.scenario import generate_scenario, load_config

from PROPOSED_METHOD.src.train import CANDIDATES, build_model, make_env, tensorize_autoregressive, tensorize_frontier


def run_policy_autoregressive(scenario, config, candidate, model, device):
    spec = CANDIDATES[candidate]
    env = make_env(candidate, scenario, config, 0)
    latencies = []
    use_topology = spec["topology"]
    while not env.done:
        obs = env.observation()
        task_t, actions_t, mask_t, neigh_t, neigh_mask_t = tensorize_autoregressive([obs], device, use_topology)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            if use_topology:
                action, _, _, _ = model.act(task_t, actions_t, mask_t, neigh_t, neigh_mask_t, deterministic=True)
            else:
                action, _, _, _ = model.act(task_t, actions_t, mask_t, deterministic=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - started) * 1000.0)
        env.step(int(action.item()))
    metrics = env.metrics()
    metrics["mean_decision_latency_ms"] = float(np.mean(latencies))
    metrics["p95_decision_latency_ms"] = float(np.quantile(latencies, 0.95))
    metrics["decision_calls"] = float(len(latencies))
    return metrics


def run_policy_frontier(scenario, config, candidate, model, device, frontier_max):
    env = make_env(candidate, scenario, config, frontier_max)
    obs = env.reset()
    latencies = []
    while not env.done:
        tensors = tensorize_frontier([obs], device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            action, _, _, _ = model.act(*tensors, deterministic=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - started) * 1000.0)
        action_np = action.cpu().numpy()[0]
        obs, _, _, _ = env.step(action_np, obs)
    metrics = env.metrics()
    metrics["mean_decision_latency_ms"] = float(np.mean(latencies))
    metrics["p95_decision_latency_ms"] = float(np.quantile(latencies, 0.95))
    metrics["decision_calls"] = float(len(latencies))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="PROPOSED_METHOD/config/proposed_method.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--candidate", required=True, choices=list(CANDIDATES))
    parser.add_argument("--split", choices=["validation", "test", "reproduction"], default="validation")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    spec = CANDIDATES[args.candidate]
    model = build_model(args.candidate, config, device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()

    if args.split == "reproduction":
        seeds = [3691718]
    else:
        start = int(config["experiment"][f"{args.split}_seed_start"])
        count = int(config["experiment"][f"{args.split}_scenarios"])
        seeds = list(range(start, start + count))

    frontier_max = int(config["frontier"]["frontier_max"])
    rows = []
    for seed in seeds:
        scenario = generate_scenario(seed, f"{args.split}_{seed}")
        if spec["family"] == "autoregressive":
            metrics = run_policy_autoregressive(scenario, config, args.candidate, model, device)
        else:
            metrics = run_policy_frontier(scenario, config, args.candidate, model, device, frontier_max)
        rows.append({"scenario_id": scenario.scenario_id, "seed": seed, "candidate": args.candidate, "checkpoint": Path(args.checkpoint).stem, **metrics})
    df = pd.DataFrame(rows)
    out_path = Path(args.out) if args.out else Path("PROPOSED_METHOD/results") / f"{Path(args.checkpoint).stem}_{args.split}_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(json.dumps({
        "results": str(out_path),
        "rows": len(df),
        "mean_system_cost": float(df["system_cost"].mean()),
        "std_system_cost": float(df["system_cost"].std()),
        "mean_carbon_emissions_kg": float(df["carbon_emissions_kg"].mean()),
        "mean_decision_latency_ms": float(df["mean_decision_latency_ms"].mean()),
    }))


if __name__ == "__main__":
    main()
