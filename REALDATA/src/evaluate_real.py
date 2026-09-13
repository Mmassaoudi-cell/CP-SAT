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

from SOURCE_METHOD_REPRODUCTION.src.environment import SchedulingEnvironment, run_heuristic
from SOURCE_METHOD_REPRODUCTION.src.model import MaskedAttentionActorCritic
from SOURCE_METHOD_REPRODUCTION.src.scenario import load_config

from BENCHMARKS.src.heuristics import HEURISTICS, run_random_feasible
from BENCHMARKS.src.milp_cpsat import run_cpsat
from BENCHMARKS.src.networks_rl import MaskedQNetwork, PoolingActorCritic
from BENCHMARKS.src.evaluate_benchmarks import run_rl_policy

from PROPOSED_METHOD.src.evaluate import run_policy_autoregressive, run_policy_frontier
from PROPOSED_METHOD.src.train import CANDIDATES, build_model

from REALDATA.src.real_scenario import generate_real_calibrated_scenario

RL_MODEL_BUILDERS = {
    "a2c": lambda hidden, heads: MaskedAttentionActorCritic(hidden=hidden, heads=heads),
    "mlp_ppo": lambda hidden, heads: PoolingActorCritic(hidden=hidden, heads=heads),
    "dqn": lambda hidden, heads: MaskedQNetwork(hidden=hidden, heads=heads),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_scenarios", type=int, default=50)
    parser.add_argument("--seed_start", type=int, default=5000)
    parser.add_argument("--cpsat_time_limit", type=float, default=30.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="REALDATA/results/real_calibrated_results.csv")
    args = parser.parse_args()

    device = torch.device(args.device)
    source_config = load_config("SOURCE_METHOD_REPRODUCTION/config/source_reproduction.yaml")
    proposed_config = load_config("PROPOSED_METHOD/config/proposed_method.yaml")
    tags_config = load_config("PROPOSED_METHOD/results/final_config_tags.yaml")
    hcbs_config = load_config("PROPOSED_METHOD/results/final_config_hcbs_ppo.yaml")
    frontier_max_tags = int(tags_config["frontier"]["frontier_max"])
    frontier_max_hcbs = int(hcbs_config["frontier"]["frontier_max"])
    frontier_max_dfps = int(proposed_config["frontier"]["frontier_max"])

    seeds = list(range(args.seed_start, args.seed_start + args.n_scenarios))
    scenarios = [generate_real_calibrated_scenario(s, f"real_{s}") for s in seeds]

    rows = []

    # --- Heuristics + CP-SAT (source-style environment/config) ---
    rng = np.random.default_rng(0)
    for scenario in scenarios:
        for method in ("s1_zero_delay", "s2_single_dc_temporal", "greedy_spatiotemporal"):
            metrics, _ = run_heuristic(scenario, source_config, method)
            rows.append({"scenario_id": scenario.scenario_id, "method": method, **metrics})
        for method in ("edf", "carbon_greedy", "heft"):
            metrics, _ = HEURISTICS[method](scenario, source_config)
            rows.append({"scenario_id": scenario.scenario_id, "method": method, **metrics})
        metrics, _ = run_random_feasible(scenario, source_config, rng)
        rows.append({"scenario_id": scenario.scenario_id, "method": "random", **metrics})
        cp_metrics, _, status = run_cpsat(scenario, source_config, time_limit_s=args.cpsat_time_limit)
        rows.append({"scenario_id": scenario.scenario_id, "method": "cpsat", **cp_metrics})
        print(json.dumps({"progress": "heuristics+cpsat", "scenario": scenario.scenario_id}))

    # --- Source PPO (5 seeds not required for mandatory benchmark; use the frozen 3000ep checkpoint) ---
    hidden = int(source_config["ppo"]["hidden_size"]); heads = int(source_config["ppo"]["attention_heads"])
    src_model = MaskedAttentionActorCritic(hidden=hidden, heads=heads).to(device)
    payload = torch.load("SOURCE_METHOD_REPRODUCTION/results/ppo_attention_3000_episodes.pt", map_location=device, weights_only=False)
    src_model.load_state_dict(payload["model"]); src_model.eval()
    for scenario in scenarios:
        metrics = run_rl_policy(scenario, source_config, "a2c", src_model, device)  # a2c/source share MaskedAttentionActorCritic .act() signature
        rows.append({"scenario_id": scenario.scenario_id, "method": "source_ppo", **metrics})

    # --- A2C, MLP-PPO, DQN (1 seed each, frozen checkpoints) ---
    for algo, ckpt in [("a2c", "BENCHMARKS/results/a2c_seed2026_3000ep.pt"), ("mlp_ppo", "BENCHMARKS/results/mlp_ppo_seed2026_3000ep.pt"), ("dqn", "BENCHMARKS/results/dqn_seed2026_1500ep.pt")]:
        model = RL_MODEL_BUILDERS[algo](hidden, heads).to(device)
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"]); model.eval()
        for scenario in scenarios:
            metrics = run_rl_policy(scenario, source_config, algo, model, device)
            rows.append({"scenario_id": scenario.scenario_id, "method": algo, **metrics})
        print(json.dumps({"progress": "done", "algo": algo}))

    # --- DFPS, TAGS, HCBS-PPO (frontier family, 5 seeds each) ---
    for name, config, frontier_max, candidate, ckpt_pattern in [
        ("dfps", proposed_config, frontier_max_dfps, "tags", "PROPOSED_METHOD/results/final_distilled_seed{seed}.pt"),
        ("tags", tags_config, frontier_max_tags, "tags", "PROPOSED_METHOD/results/final_tags_seed{seed}_3000ep.pt"),
        ("hcbs_ppo", hcbs_config, frontier_max_hcbs, "hcbs_ppo", "PROPOSED_METHOD/results/final_hcbs_ppo_seed{seed}_3000ep.pt"),
    ]:
        for seed in (2026, 7, 42, 123, 777):
            ckpt_path = ckpt_pattern.format(seed=seed)
            model = build_model(candidate, config, device)
            payload = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(payload["model"]); model.eval()
            for scenario in scenarios:
                metrics = run_policy_frontier(scenario, config, candidate, model, device, frontier_max)
                rows.append({"scenario_id": scenario.scenario_id, "method": name, "seed": seed, **metrics})
        print(json.dumps({"progress": "done", "family": name}))

    df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(json.dumps({"out": str(out_path), "rows": len(df)}))


if __name__ == "__main__":
    main()
