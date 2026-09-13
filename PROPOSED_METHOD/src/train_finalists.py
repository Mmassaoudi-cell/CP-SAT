from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from SOURCE_METHOD_REPRODUCTION.src.scenario import load_config

from PROPOSED_METHOD.src.train import CANDIDATES, run_autoregressive, run_frontier


def apply_best_params(config: dict, candidate: str, best_params: dict) -> dict:
    config = copy.deepcopy(config)
    if "lr" in best_params:
        config["ppo"]["actor_learning_rate"] = best_params["lr"]
        config["ppo"]["critic_learning_rate"] = best_params["lr"]
    if "entropy_coef" in best_params:
        config["ppo"]["entropy_coefficient"] = best_params["entropy_coef"]
    if "clip_range" in best_params:
        config["ppo"]["clip_range"] = best_params["clip_range"]
    if "hidden_size" in best_params:
        config["ppo"]["hidden_size"] = best_params["hidden_size"]
    if "pid_kp" in best_params:
        config["lagrangian"]["kp"] = best_params["pid_kp"]
    if "pid_ki" in best_params:
        config["lagrangian"]["ki"] = best_params["pid_ki"]
    if "pid_lambda_max" in best_params:
        config["lagrangian"]["lambda_max"] = best_params["pid_lambda_max"]
    if "frontier_max" in best_params:
        config["frontier"]["frontier_max"] = best_params["frontier_max"]
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--config", default="PROPOSED_METHOD/config/proposed_method.yaml")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 7, 42, 123, 777])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    base_config = load_config(args.config)
    device = torch.device(args.device)
    output = Path("PROPOSED_METHOD/results")
    output.mkdir(parents=True, exist_ok=True)

    for candidate in args.candidates:
        best_path = output / f"tuning_{candidate}_best.json"
        if best_path.exists():
            with open(best_path) as f:
                best = json.load(f)["best_params"]
        else:
            best = {}
        config = apply_best_params(base_config, candidate, best)
        with open(output / f"final_config_{candidate}.json", "w") as f:
            json.dump(config, f, indent=2)
        spec = CANDIDATES[candidate]
        runner = run_autoregressive if spec["family"] == "autoregressive" else run_frontier
        for seed in args.seeds:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            tag = f"final_{candidate}_seed{seed}_{args.episodes}ep"
            model, episode_rows, updates = runner(candidate, config, args.episodes, device, seed, tag, log_every=50)
            checkpoint = output / f"{tag}.pt"
            torch.save({"model": model.state_dict(), "config": config, "candidate": candidate, "episodes": args.episodes, "seed": seed}, checkpoint)
            pd.DataFrame(episode_rows).to_csv(output / f"{tag}_training_log.csv", index=False)
            n_params = sum(p.numel() for p in model.parameters())
            print(json.dumps({"candidate": candidate, "seed": seed, "checkpoint": str(checkpoint), "n_params": n_params, "updates": updates}))


if __name__ == "__main__":
    main()
