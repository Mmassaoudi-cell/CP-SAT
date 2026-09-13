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

from SOURCE_METHOD_REPRODUCTION.src.scenario import generate_scenario, load_config

from PROPOSED_METHOD.src.env_frontier import FrontierSchedulingEnvironment
from PROPOSED_METHOD.src.networks_frontier import FrontierMaskedAttentionActorCritic
from PROPOSED_METHOD.src.train import run_frontier, tensorize_frontier

DISTILL_CANDIDATE = "tags"  # reuse TAGS's featurization (topology + horizon) for a fair architecture comparison


def collect_dataset(config, teacher_dir: Path, device, max_scenarios: int | None = None, use_topology: bool = True):
    n_tasks_expected = int(config["experiment"]["subtasks"])
    files = sorted(teacher_dir.glob("train_*.json"))
    if max_scenarios:
        files = files[:max_scenarios]
    frontier_max = int(config["frontier"]["frontier_max"])
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
        env = FrontierSchedulingEnvironment(scenario, config, frontier_max=frontier_max, use_topology=use_topology)
        obs = env.reset()
        horizon = env.horizon
        scenario_samples = []
        try:
            while not env.done:
                teacher_actions = np.zeros(frontier_max, dtype=np.int64)
                for i, (tid, valid) in enumerate(zip(obs["task_ids"], obs["valid"])):
                    if not valid:
                        continue
                    dc, start = assignment[int(tid)]
                    teacher_actions[i] = dc * horizon + start
                scenario_samples.append({k: obs[k].copy() for k in ("tasks", "actions", "mask", "valid", "neighbors", "neighbor_mask", "curves")} | {"target": teacher_actions})
                obs, _, _, info = env.step(teacher_actions, obs)
                if info["committed"] == 0:
                    raise RuntimeError("no commit")
        except (RuntimeError, KeyError):
            # CP-SAT rounds power/bandwidth to integers for the solver, which can rarely make its
            # schedule infeasible-by-a-hair against the environment's exact float capacity check.
            # Skip the scenario's replay rather than crash the whole label-collection run.
            skipped += 1
            continue
        used += 1
        samples.extend(scenario_samples)
    print(json.dumps({"scenarios_used": used, "scenarios_skipped": skipped, "n_samples": len(samples)}))
    return samples


def train_imitation(samples, config, device, epochs=15, batch_size=64, lr=3e-4, seed=2026, use_topology=True):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    hidden = int(config["ppo"]["hidden_size"])
    heads = int(config["ppo"]["attention_heads"])
    horizon = int(config["experiment"]["horizon"])
    n_dc = int(config["experiment"]["n_datacenters"])
    model = FrontierMaskedAttentionActorCritic(hidden=hidden, heads=heads, use_topology=use_topology, use_horizon=True, n_dc=n_dc, horizon=horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(samples)
    for epoch in range(epochs):
        perm = np.random.permutation(n)
        total_loss, total_correct, total_count = 0.0, 0, 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            batch = [samples[i] for i in idx]
            tasks = torch.as_tensor(np.stack([b["tasks"] for b in batch]), device=device)
            actions = torch.as_tensor(np.stack([b["actions"] for b in batch]), device=device)
            mask = torch.as_tensor(np.stack([b["mask"] for b in batch]), device=device, dtype=torch.bool)
            valid = torch.as_tensor(np.stack([b["valid"] for b in batch]), device=device, dtype=torch.bool)
            neighbors = torch.as_tensor(np.stack([b["neighbors"] for b in batch]), device=device)
            neighbor_mask = torch.as_tensor(np.stack([b["neighbor_mask"] for b in batch]), device=device, dtype=torch.bool)
            curves = torch.as_tensor(np.stack([b["curves"] for b in batch]), device=device)
            target = torch.as_tensor(np.stack([b["target"] for b in batch]), device=device, dtype=torch.int64)
            logits, _ = model(tasks, actions, mask, valid, neighbors, neighbor_mask, curves)
            B, F, A = logits.shape
            logp = torch.log_softmax(logits, dim=-1)
            loss_per_slot = -logp.gather(-1, target[..., None]).squeeze(-1)
            loss = (loss_per_slot * valid.float()).sum() / valid.float().sum().clamp(min=1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                correct = ((pred == target).float() * valid.float()).sum().item()
            total_loss += loss.item() * valid.float().sum().item()
            total_correct += correct
            total_count += valid.float().sum().item()
        print(json.dumps({"epoch": epoch, "loss": total_loss / max(1, total_count), "top1_acc": total_correct / max(1, total_count)}))
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="PROPOSED_METHOD/config/proposed_method.yaml")
    parser.add_argument("--teacher_dir", default="PROPOSED_METHOD/results/teacher_labels")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--finetune_episodes", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default="distilled_student")
    parser.add_argument("--no_topology", action="store_true", help="ablation: HCBS-style (no GAT) featurization for both teacher-replay and student")
    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(args.device)
    use_topology = not args.no_topology
    finetune_candidate = "tags" if use_topology else "tags_no_topology"
    samples = collect_dataset(config, Path(args.teacher_dir), device, use_topology=use_topology)
    print(json.dumps({"n_samples": len(samples)}))
    model = train_imitation(samples, config, device, epochs=args.epochs, seed=args.seed, use_topology=use_topology)
    if args.finetune_episodes > 0:
        model, _, _ = run_frontier(finetune_candidate, config, args.finetune_episodes, device, args.seed, f"{args.tag}_finetune", init_model=model)
    output = Path("PROPOSED_METHOD/results")
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / f"{args.tag}.pt"
    torch.save({"model": model.state_dict(), "config": config, "candidate": finetune_candidate, "seed": args.seed}, checkpoint)
    n_params = sum(p.numel() for p in model.parameters())
    print(json.dumps({"checkpoint": str(checkpoint), "n_params": n_params, "n_samples": len(samples)}))


if __name__ == "__main__":
    main()
