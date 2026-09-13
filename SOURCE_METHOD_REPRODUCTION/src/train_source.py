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

from SOURCE_METHOD_REPRODUCTION.src.environment import SchedulingEnvironment
from SOURCE_METHOD_REPRODUCTION.src.model import MaskedAttentionActorCritic
from SOURCE_METHOD_REPRODUCTION.src.scenario import generate_scenario, load_config, write_split_manifest


def tensorize(states, device):
    return (
        torch.as_tensor(np.stack([state["task"] for state in states]), device=device),
        torch.as_tensor(np.stack([state["actions"] for state in states]), device=device),
        torch.as_tensor(np.stack([state["mask"] for state in states]), device=device, dtype=torch.bool),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="SOURCE_METHOD_REPRODUCTION/config/source_reproduction.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    config = load_config(args.config)
    ppo = config["ppo"]
    exp = config["experiment"]
    target_episodes = args.episodes or int(ppo["training_episodes"])
    seed = int(ppo["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)

    output = Path("SOURCE_METHOD_REPRODUCTION/results")
    output.mkdir(parents=True, exist_ok=True)
    write_split_manifest(config, "DATA_SPLIT_MANIFEST.csv")

    model = MaskedAttentionActorCritic(hidden=int(ppo["hidden_size"]), heads=int(ppo["attention_heads"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(ppo["actor_learning_rate"]))
    n_envs = int(ppo["parallel_envs"])
    rollout_steps = int(ppo["rollout_steps"])
    rng = np.random.default_rng(seed)

    def new_env() -> SchedulingEnvironment:
        offset = int(rng.integers(0, int(exp["train_scenarios"])))
        scenario_seed = int(exp["train_seed_start"]) + offset
        return SchedulingEnvironment(generate_scenario(scenario_seed, f"train_{offset:04d}"), config)

    envs = [new_env() for _ in range(n_envs)]
    states = [env.reset() for env in envs]
    completed = 0
    update = 0
    episode_rows = []

    while completed < target_episodes:
        storage = {key: [] for key in ("task", "actions_feat", "mask", "action", "logp", "value", "reward", "done")}
        for _ in range(rollout_steps):
            task_t, actions_t, mask_t = tensorize(states, device)
            with torch.no_grad():
                selected, logp, _, values = model.act(task_t, actions_t, mask_t)
            selected_np = selected.cpu().numpy()
            next_states = []
            rewards = []
            dones = []
            for index, env in enumerate(envs):
                next_state, reward, done, _ = env.step(int(selected_np[index]))
                rewards.append(reward)
                dones.append(done)
                if done:
                    row = {"episode": completed, "update": update, "scenario_id": env.scenario.scenario_id, **env.metrics()}
                    episode_rows.append(row)
                    completed += 1
                    envs[index] = new_env()
                    next_state = envs[index].reset()
                next_states.append(next_state)
            storage["task"].append(task_t.cpu())
            storage["actions_feat"].append(actions_t.cpu())
            storage["mask"].append(mask_t.cpu())
            storage["action"].append(selected.cpu())
            storage["logp"].append(logp.cpu())
            storage["value"].append(values.cpu())
            storage["reward"].append(torch.tensor(rewards, dtype=torch.float32))
            storage["done"].append(torch.tensor(dones, dtype=torch.float32))
            states = next_states
            if completed >= target_episodes:
                break

        with torch.no_grad():
            next_task, next_actions, next_mask = tensorize(states, device)
            _, next_value = model(next_task, next_actions, next_mask)
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
        flat["advantages"] = (flat["advantages"] - flat["advantages"].mean()) / (flat["advantages"].std() + 1e-8)
        count = len(flat["action"])
        for _ in range(int(ppo["update_epochs"])):
            permutation = torch.randperm(count)
            for start in range(0, count, int(ppo["minibatch_size"])):
                idx = permutation[start : start + int(ppo["minibatch_size"])]
                logp_new, entropy, value_new = model.evaluate_actions(
                    flat["task"][idx].to(device),
                    flat["actions_feat"][idx].to(device),
                    flat["mask"][idx].to(device),
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
        if update % 10 == 0:
            recent = pd.DataFrame(episode_rows[-100:])
            print(json.dumps({"episodes": completed, "update": update, "mean_system_cost_100": float(recent["system_cost"].mean()) if len(recent) else None}))

    checkpoint = output / f"ppo_attention_{target_episodes}_episodes.pt"
    torch.save({"model": model.state_dict(), "config": config, "episodes": target_episodes, "seed": seed}, checkpoint)
    pd.DataFrame(episode_rows).to_csv(output / f"training_log_{target_episodes}_episodes.csv", index=False)
    print(json.dumps({"checkpoint": str(checkpoint), "episodes": target_episodes, "device": str(device)}))


if __name__ == "__main__":
    main()
