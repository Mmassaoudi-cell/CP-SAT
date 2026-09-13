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

from SOURCE_METHOD_REPRODUCTION.src.model import MaskedAttentionActorCritic
from SOURCE_METHOD_REPRODUCTION.src.scenario import generate_scenario, load_config

from PROPOSED_METHOD.src.env_autoregressive import LagrangianEnvironment, TopologyEnvironment
from PROPOSED_METHOD.src.env_frontier import FrontierSchedulingEnvironment
from PROPOSED_METHOD.src.lagrangian import PIDLagrangian
from PROPOSED_METHOD.src.networks_autoregressive import TopologyActorCritic
from PROPOSED_METHOD.src.networks_frontier import FrontierMaskedAttentionActorCritic

CANDIDATES = {
    "gat_ppo": {"family": "autoregressive", "topology": True, "lagrangian": False},
    "lc_ppo": {"family": "autoregressive", "topology": False, "lagrangian": True},
    "hcbs_ppo": {"family": "frontier", "topology": False, "lagrangian": False},
    "tags": {"family": "frontier", "topology": True, "lagrangian": True},
    # Ablation-only variants (TAGS minus one mechanism), not part of the 5
    # screened candidates: complete the topology x lagrangian 2x2 grid within
    # the frontier family so TAGS's two combined mechanisms can each be
    # removed independently once TAGS is selected (Section 19 requirement).
    "tags_no_lagrangian": {"family": "frontier", "topology": True, "lagrangian": False},
    "tags_no_topology": {"family": "frontier", "topology": False, "lagrangian": True},
}


def build_model(candidate: str, config: dict, device: torch.device):
    ppo = config["ppo"]
    hidden = int(ppo["hidden_size"])
    heads = int(ppo["attention_heads"])
    spec = CANDIDATES[candidate]
    if spec["family"] == "autoregressive":
        if spec["topology"]:
            return TopologyActorCritic(hidden=hidden, heads=heads).to(device)
        return MaskedAttentionActorCritic(hidden=hidden, heads=heads).to(device)
    horizon = int(config["experiment"]["horizon"])
    n_dc = int(config["experiment"]["n_datacenters"])
    return FrontierMaskedAttentionActorCritic(
        hidden=hidden, heads=heads, use_topology=spec["topology"], use_horizon=True, n_dc=n_dc, horizon=horizon
    ).to(device)


def make_env(candidate: str, scenario, config, frontier_max: int):
    spec = CANDIDATES[candidate]
    if spec["family"] == "autoregressive":
        if spec["topology"]:
            return TopologyEnvironment(scenario, config)
        if spec["lagrangian"]:
            return LagrangianEnvironment(scenario, config)
        raise ValueError(candidate)
    return FrontierSchedulingEnvironment(scenario, config, frontier_max=frontier_max, use_topology=spec["topology"])


def tensorize_autoregressive(states, device, use_topology: bool):
    task = torch.as_tensor(np.stack([s["task"] for s in states]), device=device)
    actions = torch.as_tensor(np.stack([s["actions"] for s in states]), device=device)
    mask = torch.as_tensor(np.stack([s["mask"] for s in states]), device=device, dtype=torch.bool)
    if not use_topology:
        return task, actions, mask, None, None
    neighbors = torch.as_tensor(np.stack([s["neighbors"] for s in states]), device=device)
    neighbor_mask = torch.as_tensor(np.stack([s["neighbor_mask"] for s in states]), device=device, dtype=torch.bool)
    return task, actions, mask, neighbors, neighbor_mask


def tensorize_frontier(states, device):
    tasks = torch.as_tensor(np.stack([s["tasks"] for s in states]), device=device)
    actions = torch.as_tensor(np.stack([s["actions"] for s in states]), device=device)
    mask = torch.as_tensor(np.stack([s["mask"] for s in states]), device=device, dtype=torch.bool)
    valid = torch.as_tensor(np.stack([s["valid"] for s in states]), device=device, dtype=torch.bool)
    neighbors = torch.as_tensor(np.stack([s["neighbors"] for s in states]), device=device)
    neighbor_mask = torch.as_tensor(np.stack([s["neighbor_mask"] for s in states]), device=device, dtype=torch.bool)
    curves = torch.as_tensor(np.stack([s["curves"] for s in states]), device=device)
    return tasks, actions, mask, valid, neighbors, neighbor_mask, curves


def run_autoregressive(candidate, config, target_episodes, device, seed, out_prefix, log_every=10):
    spec = CANDIDATES[candidate]
    ppo = config["ppo"]
    exp = config["experiment"]
    model = build_model(candidate, config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(ppo["actor_learning_rate"]))
    n_envs = int(ppo["parallel_envs"])
    rollout_steps = int(ppo["rollout_steps"])
    rng = np.random.default_rng(seed)
    pid = PIDLagrangian(**config["lagrangian"]) if spec["lagrangian"] else None
    quota = float(config["environment"]["carbon_quota_kg"])

    def new_env():
        offset = int(rng.integers(0, int(exp["train_scenarios"])))
        scenario_seed = int(exp["train_seed_start"]) + offset
        env = make_env(candidate, generate_scenario(scenario_seed, f"train_{offset:04d}"), config, 0)
        if spec["lagrangian"]:
            env.set_lambda(pid.value)
        return env

    envs = [new_env() for _ in range(n_envs)]
    states = [env.reset() for env in envs]
    completed = 0
    update = 0
    episode_rows = []
    use_topology = spec["topology"]

    while completed < target_episodes:
        storage = {k: [] for k in ("task", "actions_feat", "mask", "neighbors", "neighbor_mask", "action", "logp", "value", "reward", "done")}
        for _ in range(rollout_steps):
            task_t, actions_t, mask_t, neigh_t, neigh_mask_t = tensorize_autoregressive(states, device, use_topology)
            with torch.no_grad():
                if use_topology:
                    selected, logp, _, values = model.act(task_t, actions_t, mask_t, neigh_t, neigh_mask_t)
                else:
                    selected, logp, _, values = model.act(task_t, actions_t, mask_t)
            selected_np = selected.cpu().numpy()
            next_states, rewards, dones = [], [], []
            for index, env in enumerate(envs):
                next_state, reward, done, _ = env.step(int(selected_np[index]))
                rewards.append(reward)
                dones.append(done)
                if done:
                    metrics = env.metrics()
                    if spec["lagrangian"]:
                        error = (metrics["carbon_emissions_kg"] - quota) / quota
                        pid.update(error)
                    row = {"episode": completed, "update": update, "scenario_id": env.scenario.scenario_id, **metrics}
                    if spec["lagrangian"]:
                        row["carbon_lambda"] = pid.value
                    episode_rows.append(row)
                    completed += 1
                    envs[index] = new_env()
                    if spec["lagrangian"]:
                        envs[index].set_lambda(pid.value)
                    next_state = envs[index].reset()
                else:
                    if spec["lagrangian"]:
                        env.set_lambda(pid.value)
                next_states.append(next_state)
            storage["task"].append(task_t.cpu())
            storage["actions_feat"].append(actions_t.cpu())
            storage["mask"].append(mask_t.cpu())
            if use_topology:
                storage["neighbors"].append(neigh_t.cpu())
                storage["neighbor_mask"].append(neigh_mask_t.cpu())
            storage["action"].append(selected.cpu())
            storage["logp"].append(logp.cpu())
            storage["value"].append(values.cpu())
            storage["reward"].append(torch.tensor(rewards, dtype=torch.float32))
            storage["done"].append(torch.tensor(dones, dtype=torch.float32))
            states = next_states
            if completed >= target_episodes:
                break

        with torch.no_grad():
            nt, na, nm, nn_, nnm = tensorize_autoregressive(states, device, use_topology)
            if use_topology:
                _, next_value = model(nt, na, nm, nn_, nnm)
            else:
                _, next_value = model(nt, na, nm)
            next_value = next_value.cpu()
        rewards = torch.stack(storage["reward"])
        dones = torch.stack(storage["done"])
        values = torch.stack(storage["value"])
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(n_envs)
        for step in reversed(range(len(rewards))):
            bootstrap = next_value if step == len(rewards) - 1 else values[step + 1]
            nonterminal = 1.0 - dones[step]
            delta = rewards[step] + float(ppo["discount_factor"]) * bootstrap * nonterminal - values[step]
            gae = delta + float(ppo["discount_factor"]) * float(ppo["gae_lambda"]) * nonterminal * gae
            advantages[step] = gae
        returns = advantages + values

        flat = {
            "task": torch.cat(storage["task"]),
            "actions_feat": torch.cat(storage["actions_feat"]),
            "mask": torch.cat(storage["mask"]),
            "action": torch.cat(storage["action"]),
            "logp": torch.cat(storage["logp"]),
            "advantages": advantages.flatten(),
            "returns": returns.flatten(),
        }
        if use_topology:
            flat["neighbors"] = torch.cat(storage["neighbors"])
            flat["neighbor_mask"] = torch.cat(storage["neighbor_mask"])
        flat["advantages"] = (flat["advantages"] - flat["advantages"].mean()) / (flat["advantages"].std() + 1e-8)
        count = len(flat["action"])
        for _ in range(int(ppo["update_epochs"])):
            permutation = torch.randperm(count)
            for start in range(0, count, int(ppo["minibatch_size"])):
                idx = permutation[start : start + int(ppo["minibatch_size"])]
                if use_topology:
                    logp_new, entropy, value_new = model.evaluate_actions(
                        flat["task"][idx].to(device), flat["actions_feat"][idx].to(device), flat["mask"][idx].to(device),
                        flat["neighbors"][idx].to(device), flat["neighbor_mask"][idx].to(device), flat["action"][idx].to(device),
                    )
                else:
                    logp_new, entropy, value_new = model.evaluate_actions(
                        flat["task"][idx].to(device), flat["actions_feat"][idx].to(device), flat["mask"][idx].to(device),
                        flat["action"][idx].to(device),
                    )
                ratio = (logp_new - flat["logp"][idx].to(device)).exp()
                advantage = flat["advantages"][idx].to(device)
                unclipped = ratio * advantage
                clipped = ratio.clamp(1.0 - float(ppo["clip_range"]), 1.0 + float(ppo["clip_range"])) * advantage
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = 0.5 * (value_new - flat["returns"][idx].to(device)).pow(2).mean()
                loss = policy_loss + float(ppo["value_coefficient"]) * value_loss - float(ppo["entropy_coefficient"]) * entropy.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(ppo["gradient_clip"]))
                optimizer.step()
        update += 1
        if update % log_every == 0:
            recent = pd.DataFrame(episode_rows[-100:])
            print(json.dumps({"candidate": candidate, "episodes": completed, "update": update, "mean_system_cost_100": float(recent["system_cost"].mean()) if len(recent) else None}))

    return model, episode_rows, update


def run_frontier(candidate, config, target_episodes, device, seed, out_prefix, log_every=10, init_model=None):
    spec = CANDIDATES[candidate]
    ppo = config["ppo"]
    exp = config["experiment"]
    frontier_max = int(config["frontier"]["frontier_max"])
    model = init_model if init_model is not None else build_model(candidate, config, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(ppo["actor_learning_rate"]))
    n_envs = int(ppo["parallel_envs"])
    rollout_steps = int(ppo["rollout_steps"])
    rng = np.random.default_rng(seed)
    pid = PIDLagrangian(**config["lagrangian"]) if spec["lagrangian"] else None
    quota = float(config["environment"]["carbon_quota_kg"])

    def new_env():
        offset = int(rng.integers(0, int(exp["train_scenarios"])))
        scenario_seed = int(exp["train_seed_start"]) + offset
        env = make_env(candidate, generate_scenario(scenario_seed, f"train_{offset:04d}"), config, frontier_max)
        if spec["lagrangian"]:
            env.set_lambda(pid.value)
        return env

    envs = [new_env() for _ in range(n_envs)]
    states = [env.reset() for env in envs]
    completed = 0
    update = 0
    episode_rows = []

    while completed < target_episodes:
        storage = {k: [] for k in ("tasks", "actions_feat", "mask", "valid", "neighbors", "neighbor_mask", "curves", "action", "logp", "value", "reward", "done")}
        for _ in range(rollout_steps):
            tensors = tensorize_frontier(states, device)
            with torch.no_grad():
                selected, logp, _, values = model.act(*tensors)
            selected_np = selected.cpu().numpy()
            next_states, rewards, dones = [], [], []
            for index, env in enumerate(envs):
                next_state, reward, done, _ = env.step(selected_np[index], states[index])
                rewards.append(reward)
                dones.append(done)
                if done:
                    metrics = env.metrics()
                    if spec["lagrangian"]:
                        error = (metrics["carbon_emissions_kg"] - quota) / quota
                        pid.update(error)
                    row = {"episode": completed, "update": update, "scenario_id": env.scenario.scenario_id, **metrics}
                    if spec["lagrangian"]:
                        row["carbon_lambda"] = pid.value
                    episode_rows.append(row)
                    completed += 1
                    envs[index] = new_env()
                    if spec["lagrangian"]:
                        envs[index].set_lambda(pid.value)
                    next_state = envs[index].reset()
                else:
                    if spec["lagrangian"]:
                        env.set_lambda(pid.value)
                next_states.append(next_state)
            tasks_t, actions_t, mask_t, valid_t, neigh_t, neigh_mask_t, curves_t = tensors
            storage["tasks"].append(tasks_t.cpu())
            storage["actions_feat"].append(actions_t.cpu())
            storage["mask"].append(mask_t.cpu())
            storage["valid"].append(valid_t.cpu())
            storage["neighbors"].append(neigh_t.cpu())
            storage["neighbor_mask"].append(neigh_mask_t.cpu())
            storage["curves"].append(curves_t.cpu())
            storage["action"].append(selected.cpu())
            storage["logp"].append(logp.cpu())
            storage["value"].append(values.cpu())
            storage["reward"].append(torch.tensor(rewards, dtype=torch.float32))
            storage["done"].append(torch.tensor(dones, dtype=torch.float32))
            states = next_states
            if completed >= target_episodes:
                break

        with torch.no_grad():
            next_tensors = tensorize_frontier(states, device)
            _, next_value = model(*next_tensors)
            next_value = next_value.cpu()
        rewards = torch.stack(storage["reward"])
        dones = torch.stack(storage["done"])
        values = torch.stack(storage["value"])
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(n_envs)
        for step in reversed(range(len(rewards))):
            bootstrap = next_value if step == len(rewards) - 1 else values[step + 1]
            nonterminal = 1.0 - dones[step]
            delta = rewards[step] + float(ppo["discount_factor"]) * bootstrap * nonterminal - values[step]
            gae = delta + float(ppo["discount_factor"]) * float(ppo["gae_lambda"]) * nonterminal * gae
            advantages[step] = gae
        returns = advantages + values

        flat = {key: torch.cat(storage[key]) for key in ("tasks", "actions_feat", "mask", "valid", "neighbors", "neighbor_mask", "curves", "action", "logp")}
        flat["advantages"] = advantages.flatten()
        flat["returns"] = returns.flatten()
        flat["advantages"] = (flat["advantages"] - flat["advantages"].mean()) / (flat["advantages"].std() + 1e-8)
        count = len(flat["action"])
        for _ in range(int(ppo["update_epochs"])):
            permutation = torch.randperm(count)
            for start in range(0, count, int(ppo["minibatch_size"])):
                idx = permutation[start : start + int(ppo["minibatch_size"])]
                logp_new, entropy, value_new = model.evaluate_actions(
                    flat["tasks"][idx].to(device), flat["actions_feat"][idx].to(device), flat["mask"][idx].to(device),
                    flat["valid"][idx].to(device), flat["neighbors"][idx].to(device), flat["neighbor_mask"][idx].to(device),
                    flat["curves"][idx].to(device), flat["action"][idx].to(device),
                )
                ratio = (logp_new - flat["logp"][idx].to(device)).exp()
                advantage = flat["advantages"][idx].to(device)
                unclipped = ratio * advantage
                clipped = ratio.clamp(1.0 - float(ppo["clip_range"]), 1.0 + float(ppo["clip_range"])) * advantage
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = 0.5 * (value_new - flat["returns"][idx].to(device)).pow(2).mean()
                loss = policy_loss + float(ppo["value_coefficient"]) * value_loss - float(ppo["entropy_coefficient"]) * entropy.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(ppo["gradient_clip"]))
                optimizer.step()
        update += 1
        if update % log_every == 0:
            recent = pd.DataFrame(episode_rows[-100:])
            print(json.dumps({"candidate": candidate, "episodes": completed, "update": update, "mean_system_cost_100": float(recent["system_cost"].mean()) if len(recent) else None}))

    return model, episode_rows, update


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True, choices=list(CANDIDATES))
    parser.add_argument("--config", default="PROPOSED_METHOD/config/proposed_method.yaml")
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    spec = CANDIDATES[args.candidate]
    runner = run_autoregressive if spec["family"] == "autoregressive" else run_frontier
    tag = args.tag or f"{args.candidate}_seed{args.seed}_{args.episodes}ep"
    model, episode_rows, updates = runner(args.candidate, config, args.episodes, device, args.seed, tag)
    output = Path("PROPOSED_METHOD/results")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / f"{tag}.pt"
    torch.save({"model": model.state_dict(), "config": config, "candidate": args.candidate, "episodes": args.episodes, "seed": args.seed}, checkpoint)
    pd.DataFrame(episode_rows).to_csv(output / f"{tag}_training_log.csv", index=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(json.dumps({"checkpoint": str(checkpoint), "episodes": args.episodes, "updates": updates, "n_params": n_params, "device": str(device)}))


if __name__ == "__main__":
    main()
