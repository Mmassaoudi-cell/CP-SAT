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

from PROPOSED_METHOD.src.evaluate import run_policy_frontier
from PROPOSED_METHOD.src.train import CANDIDATES, run_frontier

# Compliance-aware re-tuning of TAGS, added in response to review: the original
# Stage-3 tuning (tune.py) optimized mean system_cost alone, with no penalty
# for exceeding the carbon quota, so it had no reason to select gains that
# trade a little extra cost for compliance. This script instead optimizes
# cost + a heavy penalty per kg of quota overshoot, searching explicitly for a
# better cost/compliance operating point on the same architecture family.

PENALTY_PER_KG = 5.0  # $ per kg over quota; >> carbon_price (0.10) so the
# search must find configurations that seriously chase compliance, not just
# nudge emissions down marginally.


def objective_factory(candidate: str, base_config: dict, tune_episodes: int, eval_scenarios: int, device, quota: float):
    def objective(trial: optuna.Trial) -> float:
        config = copy.deepcopy(base_config)
        config["ppo"]["actor_learning_rate"] = trial.suggest_float("lr", 3e-5, 5e-4, log=True)
        config["ppo"]["entropy_coefficient"] = trial.suggest_float("entropy_coef", 0.02, 0.3, log=True)
        config["ppo"]["clip_range"] = trial.suggest_float("clip_range", 0.1, 0.3)
        config["ppo"]["hidden_size"] = trial.suggest_categorical("hidden_size", [64, 128, 192])
        config["lagrangian"]["kp"] = trial.suggest_float("pid_kp", 0.5, 40.0, log=True)
        config["lagrangian"]["ki"] = trial.suggest_float("pid_ki", 0.05, 4.0, log=True)
        config["lagrangian"]["lambda_max"] = trial.suggest_float("pid_lambda_max", 0.3, 8.0, log=True)
        config["frontier"]["frontier_max"] = trial.suggest_categorical("frontier_max", [10, 16, 24])
        seed = 2026
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model, _, _ = run_frontier(candidate, config, tune_episodes, device, seed, f"tunecompl_{candidate}_{trial.number}", log_every=10**9)
        model.eval()
        exp = config["experiment"]
        start = int(exp["validation_seed_start"])
        seeds = list(range(start, start + eval_scenarios))
        costs, emissions = [], []
        frontier_max = int(config["frontier"]["frontier_max"])
        for s in seeds:
            scenario = generate_scenario(s, f"tunecompl_eval_{s}")
            metrics = run_policy_frontier(scenario, config, candidate, model, device, frontier_max)
            costs.append(metrics["system_cost"])
            emissions.append(metrics["carbon_emissions_kg"])
        mean_cost = float(np.mean(costs))
        mean_emissions = float(np.mean(emissions))
        overshoot = max(0.0, mean_emissions - quota)
        objective_value = mean_cost + PENALTY_PER_KG * overshoot
        trial.set_user_attr("mean_cost", mean_cost)
        trial.set_user_attr("mean_emissions", mean_emissions)
        return objective_value

    return objective


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", default="tags")
    parser.add_argument("--config", default="PROPOSED_METHOD/config/proposed_method.yaml")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--tune_episodes", type=int, default=800)
    parser.add_argument("--eval_scenarios", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    base_config = load_config(args.config)
    quota = float(base_config["environment"]["carbon_quota_kg"])
    device = torch.device(args.device)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=2026))
    study.optimize(objective_factory(args.candidate, base_config, args.tune_episodes, args.eval_scenarios, device, quota), n_trials=args.trials)
    output = Path("PROPOSED_METHOD/results")
    output.mkdir(parents=True, exist_ok=True)
    trials_df = study.trials_dataframe()
    trials_df.to_csv(output / f"tuning_compliance_{args.candidate}_trials.csv", index=False)
    best = study.best_trial
    with open(output / f"tuning_compliance_{args.candidate}_best.json", "w") as f:
        json.dump({"best_value": study.best_value, "best_params": best.params, "mean_cost": best.user_attrs.get("mean_cost"), "mean_emissions": best.user_attrs.get("mean_emissions"), "quota": quota}, f, indent=2)
    print(json.dumps({"candidate": args.candidate, "best_value": study.best_value, "best_params": best.params, "mean_cost": best.user_attrs.get("mean_cost"), "mean_emissions": best.user_attrs.get("mean_emissions")}))


if __name__ == "__main__":
    main()
