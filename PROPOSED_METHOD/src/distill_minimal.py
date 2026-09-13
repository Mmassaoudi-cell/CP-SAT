from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from SOURCE_METHOD_REPRODUCTION.src.environment import SchedulingEnvironment
from SOURCE_METHOD_REPRODUCTION.src.model import MaskedAttentionActorCritic
from SOURCE_METHOD_REPRODUCTION.src.scenario import generate_scenario, load_config

from PROPOSED_METHOD.src.train import run_autoregressive

# Minimal-Distilled baseline: the SAME architecture as the source-paper
# reproduction (autoregressive, fixed task-order, no topology encoder, no
# batched-frontier environment) trained via CP-SAT-teacher imitation instead
# of from-scratch PPO. Isolates whether DFPS's frontier-batching + topology
# machinery adds value over distillation alone, per the Devil's Advocate's
# MAJOR #2 finding and R1's ablation-completeness concern.


def collect_dataset(config, teacher_dir: Path, max_scenarios: int | None = None):
    n_tasks_expected = int(config["experiment"]["subtasks"])
    files = sorted(teacher_dir.glob("train_*.json"))
    if max_scenarios:
        files = files[:max_scenarios]
    samples = []
    used, skipped = 0, 0
    for path in files:
        with open(path) as f:
            record = json.load(f)
        if record["status"] not in ("OPTIMAL", "FEASIBLE") or len(record["schedule"]) < n_tasks_expected:
            skipped += 1
            continue
        assignment = {s["task_id"]: (s["dc"], s["start"]) for s in record["schedule"]}
        scenario = generate_scenario(int(record["seed"]), record["scenario_id"])
        env = SchedulingEnvironment(scenario, config)
        obs = env.reset()
        scenario_samples = []
        try:
            while not env.done:
                task = env._task()
                dc, start = assignment[task.task_id]
                action = dc * env.horizon + start
                if not obs["mask"][action]:
                    raise ValueError("teacher action infeasible on replay")
                scenario_samples.append({"task": obs["task"].copy(), "actions": obs["actions"].copy(), "mask": obs["mask"].copy(), "target": action})
                obs, _, _, _ = env.step(action)
        except (ValueError, KeyError):
            # CP-SAT rounds power/bandwidth to integers for the solver, which can rarely make
            # its schedule infeasible-by-a-hair against the environment's exact float capacity
            # check (same phenomenon documented for the frontier-environment teacher replay in
            # distill.py). Skip the scenario's replay rather than crash the whole run.
            skipped += 1
            continue
        used += 1
        samples.extend(scenario_samples)
    print(json.dumps({"scenarios_used": used, "scenarios_skipped": skipped, "n_samples": len(samples)}))
    return samples


def train_imitation(samples, config, device, epochs=15, batch_size=64, lr=3e-4, seed=2026):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    hidden = int(config["ppo"]["hidden_size"])
    heads = int(config["ppo"]["attention_heads"])
    model = MaskedAttentionActorCritic(hidden=hidden, heads=heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(samples)
    for epoch in range(epochs):
        perm = np.random.permutation(n)
        total_loss, total_correct = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch = [samples[i] for i in idx]
            task = torch.as_tensor(np.stack([b["task"] for b in batch]), device=device)
            actions = torch.as_tensor(np.stack([b["actions"] for b in batch]), device=device)
            mask = torch.as_tensor(np.stack([b["mask"] for b in batch]), device=device, dtype=torch.bool)
            target = torch.as_tensor([b["target"] for b in batch], device=device, dtype=torch.int64)
            logits, _ = model(task, actions, mask)
            logp = torch.log_softmax(logits, dim=-1)
            loss = -logp.gather(-1, target[:, None]).squeeze(-1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                total_correct += (logits.argmax(dim=-1) == target).float().sum().item()
            total_loss += loss.item() * len(idx)
        print(json.dumps({"epoch": epoch, "loss": total_loss / n, "top1_acc": total_correct / n}))
    return model


def finetune(model, config, episodes, device, seed, tag):
    # Reuse the source-style autoregressive PPO loop with a MaskedAttentionActorCritic
    # init_model injection is not in run_autoregressive's signature (unlike run_frontier);
    # implement a thin local loop instead to keep this file self-contained.
    from PROPOSED_METHOD.src.train import tensorize_autoregressive
    import pandas as pd
    ppo = config["ppo"]; exp = config["experiment"]
    optimizer = torch.optim.Adam(model.parameters(), lr=float(ppo["actor_learning_rate"]))
    n_envs = int(ppo["parallel_envs"]); rollout_steps = int(ppo["rollout_steps"])
    rng = np.random.default_rng(seed)

    def new_env():
        offset = int(rng.integers(0, int(exp["train_scenarios"])))
        scenario_seed = int(exp["train_seed_start"]) + offset
        return SchedulingEnvironment(generate_scenario(scenario_seed, f"train_{offset:04d}"), config)

    envs = [new_env() for _ in range(n_envs)]
    states = [env.reset() for env in envs]
    completed, update, episode_rows = 0, 0, []
    while completed < episodes:
        storage = {k: [] for k in ("task", "actions_feat", "mask", "action", "logp", "value", "reward", "done")}
        for _ in range(rollout_steps):
            task_t, actions_t, mask_t, _, _ = tensorize_autoregressive(states, device, False)
            with torch.no_grad():
                selected, logp, _, values = model.act(task_t, actions_t, mask_t)
            selected_np = selected.cpu().numpy()
            next_states, rewards, dones = [], [], []
            for index, env in enumerate(envs):
                next_state, reward, done, _ = env.step(int(selected_np[index]))
                rewards.append(reward); dones.append(done)
                if done:
                    episode_rows.append({"episode": completed, **env.metrics()})
                    completed += 1
                    envs[index] = new_env()
                    next_state = envs[index].reset()
                next_states.append(next_state)
            storage["task"].append(task_t.cpu()); storage["actions_feat"].append(actions_t.cpu()); storage["mask"].append(mask_t.cpu())
            storage["action"].append(selected.cpu()); storage["logp"].append(logp.cpu()); storage["value"].append(values.cpu())
            storage["reward"].append(torch.tensor(rewards, dtype=torch.float32)); storage["done"].append(torch.tensor(dones, dtype=torch.float32))
            states = next_states
            if completed >= episodes:
                break
        with torch.no_grad():
            nt, na, nm, _, _ = tensorize_autoregressive(states, device, False)
            _, next_value = model(nt, na, nm)
            next_value = next_value.cpu()
        rewards = torch.stack(storage["reward"]); dones = torch.stack(storage["done"]); values = torch.stack(storage["value"])
        advantages = torch.zeros_like(rewards); gae = torch.zeros(n_envs)
        for step in reversed(range(len(rewards))):
            bootstrap = next_value if step == len(rewards) - 1 else values[step + 1]
            nonterminal = 1.0 - dones[step]
            delta = rewards[step] + float(ppo["discount_factor"]) * bootstrap * nonterminal - values[step]
            gae = delta + float(ppo["discount_factor"]) * float(ppo["gae_lambda"]) * nonterminal * gae
            advantages[step] = gae
        returns = advantages + values
        flat = {"task": torch.cat(storage["task"]), "actions_feat": torch.cat(storage["actions_feat"]), "mask": torch.cat(storage["mask"]),
                "action": torch.cat(storage["action"]), "logp": torch.cat(storage["logp"]), "advantages": advantages.flatten(), "returns": returns.flatten()}
        flat["advantages"] = (flat["advantages"] - flat["advantages"].mean()) / (flat["advantages"].std() + 1e-8)
        count = len(flat["action"])
        for _ in range(int(ppo["update_epochs"])):
            permutation = torch.randperm(count)
            for start in range(0, count, int(ppo["minibatch_size"])):
                idx = permutation[start : start + int(ppo["minibatch_size"])]
                logp_new, entropy, value_new = model.evaluate_actions(flat["task"][idx].to(device), flat["actions_feat"][idx].to(device), flat["mask"][idx].to(device), flat["action"][idx].to(device))
                ratio = (logp_new - flat["logp"][idx].to(device)).exp()
                advantage = flat["advantages"][idx].to(device)
                unclipped = ratio * advantage
                clipped = ratio.clamp(1.0 - float(ppo["clip_range"]), 1.0 + float(ppo["clip_range"])) * advantage
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = 0.5 * (value_new - flat["returns"][idx].to(device)).pow(2).mean()
                loss = policy_loss + float(ppo["value_coefficient"]) * value_loss - float(ppo["entropy_coefficient"]) * entropy.mean()
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(ppo["gradient_clip"])); optimizer.step()
        update += 1
    return model, episode_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="PROPOSED_METHOD/config/proposed_method.yaml")
    parser.add_argument("--teacher_dir", default="PROPOSED_METHOD/results/teacher_labels")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--finetune_episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default="minimal_distilled")
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    samples = collect_dataset(config, Path(args.teacher_dir))
    model = train_imitation(samples, config, device, epochs=args.epochs, seed=args.seed)
    if args.finetune_episodes > 0:
        model, _ = finetune(model, config, args.finetune_episodes, device, args.seed, args.tag)
    output = Path("PROPOSED_METHOD/results")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / f"{args.tag}.pt"
    torch.save({"model": model.state_dict(), "config": config, "seed": args.seed}, checkpoint)
    n_params = sum(p.numel() for p in model.parameters())
    print(json.dumps({"checkpoint": str(checkpoint), "n_params": n_params, "n_samples": len(samples)}))


if __name__ == "__main__":
    main()
