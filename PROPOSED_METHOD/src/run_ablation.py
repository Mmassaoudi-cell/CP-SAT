from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from SOURCE_METHOD_REPRODUCTION.src.scenario import generate_scenario, load_config

from PROPOSED_METHOD.src.evaluate import run_policy_autoregressive, run_policy_frontier
from PROPOSED_METHOD.src.train import CANDIDATES, run_autoregressive, run_frontier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--config", default="PROPOSED_METHOD/config/proposed_method.yaml")
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 7, 42, 123, 777])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval_split", default="validation")
    parser.add_argument("--out", default="PROPOSED_METHOD/results/ablation_results.csv")
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    rows = []
    for candidate in args.candidates:
        spec = CANDIDATES[candidate]
        for seed in args.seeds:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            runner = run_autoregressive if spec["family"] == "autoregressive" else run_frontier
            tag = f"ablation_{candidate}_seed{seed}_{args.episodes}ep"
            model, episode_rows, _ = runner(candidate, config, args.episodes, device, seed, tag, log_every=10**9)
            model.eval()
            n_params = sum(p.numel() for p in model.parameters())
            exp = config["experiment"]
            start = int(exp[f"{args.eval_split}_seed_start"])
            count = int(exp[f"{args.eval_split}_scenarios"])
            frontier_max = int(config["frontier"]["frontier_max"])
            costs, emissions, latencies = [], [], []
            for s in range(start, start + count):
                scenario = generate_scenario(s, f"{args.eval_split}_{s}")
                if spec["family"] == "autoregressive":
                    metrics = run_policy_autoregressive(scenario, config, candidate, model, device)
                else:
                    metrics = run_policy_frontier(scenario, config, candidate, model, device, frontier_max)
                costs.append(metrics["system_cost"])
                emissions.append(metrics["carbon_emissions_kg"])
                latencies.append(metrics["mean_decision_latency_ms"])
            row = {
                "candidate": candidate, "seed": seed, "episodes": args.episodes, "n_params": n_params,
                "mean_system_cost": float(np.mean(costs)), "std_system_cost": float(np.std(costs)),
                "mean_carbon_emissions_kg": float(np.mean(emissions)), "mean_decision_latency_ms": float(np.mean(latencies)),
                "final_train_system_cost_100": float(pd.DataFrame(episode_rows[-100:])["system_cost"].mean()) if episode_rows else None,
            }
            rows.append(row)
            print(json.dumps(row))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(json.dumps({"out": str(out_path), "rows": len(rows)}))


if __name__ == "__main__":
    main()
