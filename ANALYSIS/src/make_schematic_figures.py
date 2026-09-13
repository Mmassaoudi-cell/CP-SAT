from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

OUT = "ANALYSIS/figures"


def box(ax, xy, w, h, text, fc="#eef2f7", ec="#333333", fontsize=9):
    rect = mpatches.FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                                    linewidth=1.2, edgecolor=ec, facecolor=fc)
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)
    return rect


def arrow(ax, p1, p2, color="#333333"):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14, color=color, linewidth=1.3)
    ax.add_patch(a)


def fig1_architecture():
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    ax.set_title("Fig. 1 — DFPS architecture and training pipeline", fontsize=12)

    box(ax, (0.3, 4.6), 2.2, 1.0, "Training scenarios\n(200 CP-SAT\nteacher solves)", fc="#fde0dd")
    box(ax, (0.3, 3.0), 2.2, 1.0, "Ready-task frontier\n(FrontierSchedulingEnvironment)", fc="#e5f5e0")
    box(ax, (0.3, 1.2), 2.2, 1.2, "Neighborhood\nattention encoder\n(topology, 1-hop GAT-style)", fc="#deebf7")

    box(ax, (2.9, 3.0), 2.3, 1.0, "Horizon-cached\ntemporal encoder\n(dilated causal conv +\nprefix-sum action features)", fc="#deebf7")

    box(ax, (5.6, 2.5), 2.4, 2.0, "Shared bilinear scorer\n(batched over the\nentire ready frontier\nin ONE forward pass)", fc="#fff7bc")

    box(ax, (8.4, 4.0), 1.4, 1.0, "Stage 2:\nimitation\n(cross-entropy\nvs. CP-SAT)", fc="#fde0dd")
    box(ax, (8.4, 2.6), 1.4, 1.0, "Stage 3:\nPID-Lagrangian\nPPO fine-tune\n(300 episodes)", fc="#c7e9c0")
    box(ax, (8.4, 1.2), 1.4, 1.0, "DFPS\n(deployed\npolicy)", fc="#2c7fb8", ec="#08306b", fontsize=10)

    arrow(ax, (2.5, 5.1), (2.5, 4.0))
    arrow(ax, (2.5, 3.0), (1.4, 2.4))
    arrow(ax, (2.5, 3.5), (2.9, 3.5))
    arrow(ax, (1.4, 1.2), (5.6, 3.2))
    arrow(ax, (5.2, 3.5), (5.6, 3.5))
    arrow(ax, (8.0, 3.5), (8.4, 4.5))
    arrow(ax, (9.1, 4.0), (9.1, 3.6))
    arrow(ax, (9.1, 2.6), (9.1, 2.2))

    ax.text(5.05, 0.6, "One batched forward pass over the ready frontier replaces one\n"
                        "sequential network call per subtask (weakness #5); training\n"
                        "methodology (imitation of a near-optimal solver) is the\n"
                        "load-bearing mechanism, per the ablation in Fig. 5(a).",
            fontsize=8, ha="center", style="italic")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_architecture.png")
    plt.close(fig)


def fig2_workflow_comparison():
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.4))

    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 2); ax.axis("off")
    ax.set_title("(a) Source paper: from-scratch on-policy PPO", fontsize=11, loc="left")
    steps = ["Random\ninit", "3,000 episodes\non-policy rollouts\n+ PPO updates", "Deployed\npolicy"]
    xs = [0.3, 3.2, 7.6]
    ws = [1.6, 4.0, 1.6]
    for x, w, s in zip(xs, ws, steps):
        box(ax, (x, 0.4), w, 1.1, s, fc="#fee8c8")
    arrow(ax, (1.9, 0.95), (3.2, 0.95))
    arrow(ax, (7.2, 0.95), (7.6, 0.95))
    ax.text(5.2, 0.15, "Full training cost paid independently for every seed (~40 min/seed, isolated)", fontsize=8, ha="center", style="italic")

    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 2); ax.axis("off")
    ax.set_title("(b) Proposed: solver-distillation + light RL adapter (DFPS)", fontsize=11, loc="left")
    box(ax, (0.3, 0.4), 2.0, 1.1, "CP-SAT teacher\n(one-time, 200\nscenarios, reused\nacross all seeds)", fc="#c7e9c0")
    box(ax, (2.7, 0.4), 2.2, 1.1, "Imitation\n(20 epochs,\ncross-entropy)", fc="#deebf7")
    box(ax, (5.3, 0.4), 2.2, 1.1, "Light RL\nfine-tune\n(300 episodes)", fc="#fde0dd")
    box(ax, (7.9, 0.4), 1.8, 1.1, "DFPS\n(deployed\npolicy)", fc="#2c7fb8", ec="#08306b")
    arrow(ax, (2.3, 0.95), (2.7, 0.95))
    arrow(ax, (4.9, 0.95), (5.3, 0.95))
    arrow(ax, (7.5, 0.95), (7.9, 0.95))
    ax.text(5.0, 0.15, "One-time teacher cost amortized across seeds: ~51 min (seed 1) then ~1 min/seed thereafter", fontsize=8, ha="center", style="italic")

    fig.suptitle("Fig. 2 — Source vs. proposed training workflow", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_workflow_comparison.png")
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    fig1_architecture()
    fig2_workflow_comparison()
    print("schematic figures written")
