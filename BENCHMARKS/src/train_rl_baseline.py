from __future__ import annotations

import argparse
import json
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from SOURCE_METHOD_REPRODUCTION.src.environment import SchedulingEnvironment
from SOURCE_METHOD_REPRODUCTION.src.model import MaskedAttentionActorCritic
from SOURCE_METHOD_REPRODUCTION.src.scenario import generate_scenario, load_config

from BENCHMARKS.src.networks_rl import MaskedQNetwork, PoolingActorCritic


def tensorize(states, device):
    return (
        torch.as_tensor(np.stack([s["task"] for s in states]), device=device),
        torch.as_tensor(np.stack([s["actions"] for s in states]), device=device),
        torch.as_tensor(np.stack([s["mask"] for s in states]), device=device, dtype=torch.bool),
    )


def train_actor_critic(algorithm: str, config: dict, target_episodes: int, device, seed: int, tag: str, log_every=10):
    """algorithm: 'a2c' (single-epoch, no clip, MaskedAttentionActorCritic) or
    'mlp_ppo' (full PPO update, PoolingActorCritic / no attention)."""
    ppo = config["ppo"]
    exp = config["experiment"]
    hidden, heads = int(ppo["hidden_size"]), int(ppo["attention_heads"])
    model = (MaskedAttentionActorCritic(hidden=hidden, heads=heads) if algorithm == "a2c" else PoolingActorCritic(hidden=hidden, heads=heads)).to(device)
    lr = float(ppo["actor_learning_rate"]) if algorithm == "mlp_ppo" else float(ppo["actor_learning_rate"]) * 3.0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_envs = int(ppo["parallel_envs"])
    rollout_steps = int(ppo["rollout_steps"])
    rng = np.random.default_rng(seed)

    def new_env():
        offset = int(rng.integers(0, int(exp["train_scenarios"])))
        scenario_seed = int(exp["train_seed_start"]) + offset
        return SchedulingEnvironment(generate_scenario(scenario_seed, f"train_{offset:04d}"), config)

    envs = [new_env() for _ in range(n_envs)]
    states = [env.reset() for env in envs]
    completed, update = 0, 0
    episode_rows = []

    while completed < target_episodes:
        storage = {k: [] for k in ("task", "actions_feat", "mask", "action", "logp", "value", "reward", "done")}
        for _ in range(rollout_steps):
            task_t, actions_t, mask_t = tensorize(states, device)
            with torch.no_grad():
                selected, logp, _, values = model.act(task_t, actions_t, mask_t)
            selected_np = selected.cpu().numpy()
            next_states, rewards, dones = [], [], []
            for index, env in enumerate(envs):
                next_state, reward, done, _ = env.step(int(selected_np[index]))
                rewards.append(reward)
                dones.append(done)
                if done:
                    episode_rows.append({"episode": completed, "update": update, "scenario_id": env.scenario.scenario_id, **env.metrics()})
                    completed += 1
                    envs[index] = new_env()
                    next_state = envs[index].reset()
                next_states.append(next_state)
            for key, val in zip(("task", "actions_feat", "mask", "action", "logp", "value"), (task_t.cpu(), actions_t.cpu(), mask_t.cpu(), selected.cpu(), logp.cpu(), values.cpu())):
                storage[key].append(val)
            storage["reward"].append(torch.tensor(rewards, dtype=torch.float32))
            storage["done"].append(torch.tensor(dones, dtype=torch.float32))
            states = next_states
            if completed >= target_episodes:
                break

        with torch.no_grad():
            nt, na, nm = tensorize(states, device)
            _, next_value = model(nt, na, nm)
            next_value = next_value.cpu()
        rewards, dones, values = torch.stack(storage["reward"]), torch.stack(storage["done"]), torch.stack(storage["value"])
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
            "task": torch.cat(storage["task"]), "actions_feat": torch.cat(storage["actions_feat"]), "mask": torch.cat(storage["mask"]),
            "action": torch.cat(storage["action"]), "logp": torch.cat(storage["logp"]),
            "advantages": advantages.flatten(), "returns": returns.flatten(),
        }
        flat["advantages"] = (flat["advantages"] - flat["advantages"].mean()) / (flat["advantages"].std() + 1e-8)
        count = len(flat["action"])
        epochs = 1 if algorithm == "a2c" else int(ppo["update_epochs"])
        batch_size = count if algorithm == "a2c" else int(ppo["minibatch_size"])
        for _ in range(epochs):
            permutation = torch.randperm(count)
            for start in range(0, count, batch_size):
                idx = permutation[start : start + batch_size]
                logp_new, entropy, value_new = model.evaluate_actions(
                    flat["task"][idx].to(device), flat["actions_feat"][idx].to(device), flat["mask"][idx].to(device), flat["action"][idx].to(device)
                )
                advantage = flat["advantages"][idx].to(device)
                if algorithm == "a2c":
                    policy_loss = -(logp_new * advantage).mean()
                else:
                    ratio = (logp_new - flat["logp"][idx].to(device)).exp()
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
            print(json.dumps({"algorithm": algorithm, "episodes": completed, "update": update, "mean_system_cost_100": float(recent["system_cost"].mean()) if len(recent) else None}))
    return model, episode_rows


def train_dqn(config: dict, target_episodes: int, device, seed: int, tag: str, log_every=200,
              buffer_size=50000, batch_size=256, gamma=0.99, lr=1e-4, eps_start=1.0, eps_end=0.05, eps_decay_episodes=1500,
              target_update_every=500, train_freq=4):
    exp = config["experiment"]
    ppo = config["ppo"]
    hidden, heads = int(ppo["hidden_size"]), int(ppo["attention_heads"])
    q_net = MaskedQNetwork(hidden=hidden, heads=heads).to(device)
    target_net = MaskedQNetwork(hidden=hidden, heads=heads).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    buffer = deque(maxlen=buffer_size)

    def new_env():
        offset = int(rng.integers(0, int(exp["train_scenarios"])))
        scenario_seed = int(exp["train_seed_start"]) + offset
        return SchedulingEnvironment(generate_scenario(scenario_seed, f"train_{offset:04d}"), config)

    env = new_env()
    state = env.reset()
    completed, step_count = 0, 0
    episode_rows = []
    while completed < target_episodes:
        eps = max(eps_end, eps_start - (eps_start - eps_end) * completed / eps_decay_episodes)
        mask = state["mask"]
        feasible = np.flatnonzero(mask)
        if rng.random() < eps:
            action = int(rng.choice(feasible))
        else:
            with torch.no_grad():
                task_t = torch.as_tensor(state["task"], device=device)[None]
                actions_t = torch.as_tensor(state["actions"], device=device)[None]
                mask_t = torch.as_tensor(mask, device=device, dtype=torch.bool)[None]
                q = q_net(task_t, actions_t, mask_t)
                action = int(q.argmax(dim=-1).item())
        next_state, reward, done, _ = env.step(action)
        buffer.append((state["task"], state["actions"], state["mask"], action, reward, next_state["task"], next_state["actions"], next_state["mask"], float(done)))
        state = next_state
        step_count += 1
        if done:
            episode_rows.append({"episode": completed, "scenario_id": env.scenario.scenario_id, "epsilon": eps, **env.metrics()})
            completed += 1
            env = new_env()
            state = env.reset()
        if len(buffer) >= batch_size and step_count % train_freq == 0:
            batch = [buffer[i] for i in rng.integers(0, len(buffer), size=batch_size)]
            task_b = torch.as_tensor(np.stack([b[0] for b in batch]), device=device)
            actions_b = torch.as_tensor(np.stack([b[1] for b in batch]), device=device)
            mask_b = torch.as_tensor(np.stack([b[2] for b in batch]), device=device, dtype=torch.bool)
            action_b = torch.as_tensor([b[3] for b in batch], device=device, dtype=torch.int64)
            reward_b = torch.as_tensor([b[4] for b in batch], device=device, dtype=torch.float32)
            next_task_b = torch.as_tensor(np.stack([b[5] for b in batch]), device=device)
            next_actions_b = torch.as_tensor(np.stack([b[6] for b in batch]), device=device)
            next_mask_b = torch.as_tensor(np.stack([b[7] for b in batch]), device=device, dtype=torch.bool)
            done_b = torch.as_tensor([b[8] for b in batch], device=device, dtype=torch.float32)
            q_values = q_net(task_b, actions_b, mask_b).gather(1, action_b[:, None]).squeeze(-1)
            with torch.no_grad():
                next_q = target_net(next_task_b, next_actions_b, next_mask_b)
                next_q = torch.where(torch.isfinite(next_q), next_q, torch.full_like(next_q, -1e4))
                next_q_max = next_q.max(dim=-1).values
                target = reward_b + gamma * next_q_max * (1.0 - done_b)
            loss = torch.nn.functional.smooth_l1_loss(q_values, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
            optimizer.step()
            if step_count % target_update_every == 0:
                target_net.load_state_dict(q_net.state_dict())
        if completed % log_every == 0 and done:
            recent = pd.DataFrame(episode_rows[-100:])
            print(json.dumps({"algorithm": "dqn", "episodes": completed, "mean_system_cost_100": float(recent["system_cost"].mean()) if len(recent) else None}))
    return q_net, episode_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", required=True, choices=["a2c", "mlp_ppo", "dqn"])
    parser.add_argument("--config", default="SOURCE_METHOD_REPRODUCTION/config/source_reproduction.yaml")
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
    tag = args.tag or f"{args.algorithm}_seed{args.seed}_{args.episodes}ep"
    if args.algorithm == "dqn":
        model, episode_rows = train_dqn(config, args.episodes, device, args.seed, tag)
    else:
        model, episode_rows = train_actor_critic(args.algorithm, config, args.episodes, device, args.seed, tag)
    output = Path("BENCHMARKS/results")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / f"{tag}.pt"
    torch.save({"model": model.state_dict(), "config": config, "algorithm": args.algorithm, "episodes": args.episodes, "seed": args.seed}, checkpoint)
    pd.DataFrame(episode_rows).to_csv(output / f"{tag}_training_log.csv", index=False)
    n_params = sum(p.numel() for p in model.parameters())
    print(json.dumps({"checkpoint": str(checkpoint), "episodes": args.episodes, "n_params": n_params, "device": str(device)}))


if __name__ == "__main__":
    main()
