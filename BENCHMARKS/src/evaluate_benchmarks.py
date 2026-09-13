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

from SOURCE_METHOD_REPRODUCTION.src.environment import SchedulingEnvironment
from SOURCE_METHOD_REPRODUCTION.src.model import MaskedAttentionActorCritic
from SOURCE_METHOD_REPRODUCTION.src.scenario import generate_scenario, load_config

from BENCHMARKS.src.heuristics import HEURISTICS, run_random_feasible
from BENCHMARKS.src.milp_cpsat import run_cpsat
from BENCHMARKS.src.networks_rl import MaskedQNetwork, PoolingActorCritic

RL_MODEL_BUILDERS = {
    "a2c": lambda hidden, heads: MaskedAttentionActorCritic(hidden=hidden, heads=heads),
    "mlp_ppo": lambda hidden, heads: PoolingActorCritic(hidden=hidden, heads=heads),
    "dqn": lambda hidden, heads: MaskedQNetwork(hidden=hidden, heads=heads),
}


def run_rl_policy(scenario, config, algorithm, model, device):
    env = SchedulingEnvironment(scenario, config)
    latencies = []
    while not env.done:
        obs = env.observation()
        task_t = torch.as_tensor(obs["task"], device=device)[None]
        actions_t = torch.as_tensor(obs["actions"], device=device)[None]
        mask_t = torch.as_tensor(obs["mask"], device=device, dtype=torch.bool)[None]
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            if algorithm == "dqn":
                q = model(task_t, actions_t, mask_t)
                action = int(q.argmax(dim=-1).item())
            else:
                action_t, _, _, _ = model.act(task_t, actions_t, mask_t, deterministic=True)
                action = int(action_t.item())
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - started) * 1000.0)
        env.step(action)
    metrics = env.metrics()
    metrics["mean_decision_latency_ms"] = float(np.mean(latencies))
    metrics["p95_decision_latency_ms"] = float(np.quantile(latencies, 0.95))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="SOURCE_METHOD_REPRODUCTION/config/source_reproduction.yaml")
    parser.add_argument("--split", choices=["reproduction", "validation", "test"], default="validation")
    parser.add_argument("--methods", nargs="*", default=["edf", "carbon_greedy", "heft", "random"])
    parser.add_argument("--rl_checkpoints", nargs="*", default=[], help="algorithm:path pairs, e.g. a2c:BENCHMARKS/results/a2c_seed2026_3000ep.pt")
    parser.add_argument("--cpsat_time_limit", type=float, default=30.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None)
    parser.add_argument("--max_scenarios", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)

    if args.split == "reproduction":
        seeds = [3691718]
    else:
        start = int(config["experiment"][f"{args.split}_seed_start"])
        count = int(config["experiment"][f"{args.split}_scenarios"])
        seeds = list(range(start, start + count))
    if args.max_scenarios:
        seeds = seeds[: args.max_scenarios]

    rl_models = {}
    hidden = int(config["ppo"]["hidden_size"])
    heads = int(config["ppo"]["attention_heads"])
    for pair in args.rl_checkpoints:
        algo, path = pair.split(":", 1)
        model = RL_MODEL_BUILDERS[algo](hidden, heads).to(device)
        payload = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        model.eval()
        rl_models[algo] = model

    rows = []
    rng = np.random.default_rng(0)
    for seed in seeds:
        scenario = generate_scenario(seed, f"{args.split}_{seed}")
        for method in args.methods:
            if method == "random":
                metrics, _ = run_random_feasible(scenario, config, rng)
            elif method == "cpsat":
                metrics, _, status = run_cpsat(scenario, config, time_limit_s=args.cpsat_time_limit)
                metrics = dict(metrics)
            else:
                metrics, _ = HEURISTICS[method](scenario, config)
            rows.append({"scenario_id": scenario.scenario_id, "seed": seed, "method": method, **metrics})
        for algo, model in rl_models.items():
            metrics = run_rl_policy(scenario, config, algo, model, device)
            rows.append({"scenario_id": scenario.scenario_id, "seed": seed, "method": algo, **metrics})

    df = pd.DataFrame(rows)
    out_path = Path(args.out) if args.out else Path("BENCHMARKS/results") / f"benchmarks_{args.split}_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    summary = df.groupby("method")["system_cost"].agg(["mean", "std", "count"]).to_dict("index")
    print(json.dumps({"results": str(out_path), "rows": len(df), "summary": summary}, default=str))


if __name__ == "__main__":
    main()
