from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from SOURCE_METHOD_REPRODUCTION.src.scenario import generate_scenario, load_config

from ANALYSIS.src.robustness import generate_scenario_variant, with_market_noise, with_missing_bandwidth_signal
from PROPOSED_METHOD.src.evaluate import run_policy_autoregressive, run_policy_frontier
from PROPOSED_METHOD.src.train import CANDIDATES, build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="PROPOSED_METHOD/config/proposed_method.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--candidate", required=True, choices=list(CANDIDATES))
    parser.add_argument("--n_scenarios", type=int, default=50)
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
    frontier_max = int(config["frontier"]["frontier_max"])

    start = int(config["experiment"]["validation_seed_start"])
    seeds = list(range(start, start + args.n_scenarios))

    conditions = {
        "clean": lambda s, seed: s,
        "price_noise_low": lambda s, seed: with_market_noise(s, price_noise_std=0.03, cef_noise_std=0.0, seed=seed),
        "price_noise_high": lambda s, seed: with_market_noise(s, price_noise_std=0.10, cef_noise_std=0.0, seed=seed),
        "cef_noise_high": lambda s, seed: with_market_noise(s, price_noise_std=0.0, cef_noise_std=0.08, seed=seed),
        "bandwidth_dropout_20pct": lambda s, seed: with_missing_bandwidth_signal(s, dropout_prob=0.2, seed=seed),
        "bandwidth_dropout_50pct": lambda s, seed: with_missing_bandwidth_signal(s, dropout_prob=0.5, seed=seed),
    }
    rows = []
    for seed in seeds:
        base = generate_scenario(seed, f"robust_{seed}")
        for cond_name, fn in conditions.items():
            scenario = fn(base, seed)
            if spec["family"] == "autoregressive":
                metrics = run_policy_autoregressive(scenario, config, args.candidate, model, device)
            else:
                metrics = run_policy_frontier(scenario, config, args.candidate, model, device, frontier_max)
            rows.append({"seed": seed, "condition": cond_name, **metrics})

    tight_rows = []
    for n_workloads in (6, 7, 8, 9, 10):
        for seed in seeds[:20]:
            scenario = generate_scenario_variant(seed, n_workloads=n_workloads)
            if spec["family"] == "autoregressive":
                metrics = run_policy_autoregressive(scenario, config, args.candidate, model, device)
            else:
                metrics = run_policy_frontier(scenario, config, args.candidate, model, device, frontier_max)
            tight_rows.append({"seed": seed, "n_workloads": n_workloads, **metrics})
    for slack_scale in (0.5, 0.75, 1.0, 1.5):
        for seed in seeds[:20]:
            scenario = generate_scenario_variant(seed, n_workloads=10, slack_scale=slack_scale)
            if spec["family"] == "autoregressive":
                metrics = run_policy_autoregressive(scenario, config, args.candidate, model, device)
            else:
                metrics = run_policy_frontier(scenario, config, args.candidate, model, device, frontier_max)
            tight_rows.append({"seed": seed, "slack_scale": slack_scale, **metrics})

    df = pd.DataFrame(rows)
    df_variant = pd.DataFrame(tight_rows)
    out_path = Path(args.out) if args.out else Path("ANALYSIS/results") / f"robustness_{Path(args.checkpoint).stem}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    df_variant.to_csv(out_path.with_name(out_path.stem + "_variants.csv"), index=False)
    summary = df.groupby("condition")["system_cost"].agg(["mean", "std"]).to_dict("index")
    print(json.dumps({"out": str(out_path), "rows": len(df), "summary": summary}, default=str))


if __name__ == "__main__":
    main()
