from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import optuna
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from SOURCE_METHOD_REPRODUCTION.src.scenario import generate_scenario, load_config

from PROPOSED_METHOD.src.evaluate import run_policy_autoregressive, run_policy_frontier
from PROPOSED_METHOD.src.train import CANDIDATES, run_autoregressive, run_frontier

# Validation-only tuning (Stage 3): objective is mean system_cost on a fixed
# validation subset. TEST scenarios are never touched here or anywhere in
# candidate development, per the master protocol's validation-only screening
# rule.


def objective_factory(candidate: str, base_config: dict, tune_episodes: int, eval_scenarios: int, device):
    spec = CANDIDATES[candidate]

    def objective(trial: optuna.Trial) -> float:
        config = copy.deepcopy(base_config)
        config["ppo"]["actor_learning_rate"] = trial.suggest_float("lr", 3e-5, 5e-4, log=True)
        config["ppo"]["entropy_coefficient"] = trial.suggest_float("entropy_coef", 0.02, 0.3, log=True)
        config["ppo"]["clip_range"] = trial.suggest_float("clip_range", 0.1, 0.3)
        config["ppo"]["hidden_size"] = trial.suggest_categorical("hidden_size", [64, 128, 192])
        if spec["lagrangian"]:
            config["lagrangian"]["kp"] = trial.suggest_float("pid_kp", 0.3, 15.0, log=True)
            config["lagrangian"]["ki"] = trial.suggest_float("pid_ki", 0.03, 1.5, log=True)
            config["lagrangian"]["lambda_max"] = trial.suggest_float("pid_lambda_max", 0.2, 3.0, log=True)
        if spec["family"] == "frontier":
            config["frontier"]["frontier_max"] = trial.suggest_categorical("frontier_max", [10, 16, 24])
        seed = 2026
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        runner = run_autoregressive if spec["family"] == "autoregressive" else run_frontier
        model, _, _ = runner(candidate, config, tune_episodes, device, seed, f"tune_{candidate}_{trial.number}", log_every=10**9)
        model.eval()
        exp = config["experiment"]
        start = int(exp["validation_seed_start"])
        seeds = list(range(start, start + eval_scenarios))
        costs = []
        for s in seeds:
            scenario = generate_scenario(s, f"tune_eval_{s}")
            if spec["family"] == "autoregressive":
                metrics = run_policy_autoregressive(scenario, config, candidate, model, device)
            else:
                metrics = run_policy_frontier(scenario, config, candidate, model, device, int(config["frontier"]["frontier_max"]))
            costs.append(metrics["system_cost"])
        return float(np.mean(costs))

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, choices=list(CANDIDATES))
    parser.add_argument("--config", default="PROPOSED_METHOD/config/proposed_method.yaml")
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--tune_episodes", type=int, default=400)
    parser.add_argument("--eval_scenarios", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    base_config = load_config(args.config)
    device = torch.device(args.device)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=2026))
    study.optimize(objective_factory(args.candidate, base_config, args.tune_episodes, args.eval_scenarios, device), n_trials=args.trials)
    output = Path("PROPOSED_METHOD/results")
    output.mkdir(parents=True, exist_ok=True)
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output / f"tuning_{args.candidate}_trials.csv", index=False)
    with open(output / f"tuning_{args.candidate}_best.json", "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params}, f, indent=2)
    print(json.dumps({"candidate": args.candidate, "best_value": study.best_value, "best_params": study.best_params}))


if __name__ == "__main__":
    main()
