from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
OUT = "ANALYSIS/figures"


def fig3_main_benchmark():
    df = pd.read_csv("ANALYSIS/results/test_comparison_combined.csv")
    order = df.groupby("method")["system_cost"].mean().sort_values().index.tolist()
    means = df.groupby("method")["system_cost"].mean().reindex(order)
    stds = df.groupby("method")["system_cost"].std().reindex(order)
    colors = ["#2c7fb8" if m == "DFPS" else ("#fdae61" if m == "Minimal_Distilled" else ("#31a354" if m == "TAGS_Compliant" else ("#7fcdbb" if m in ("TAGS", "HCBS-PPO") else "#636363"))) for m in order]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(order))
    ax.barh(y, means.values, xerr=stds.values, color=colors, capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("System cost, $ (test split, 100 scenarios; lower is better)")
    ax.set_title("Main benchmark comparison — DFPS vs. 15 comparisons + TAGS-Compliant (test split)")
    ax.invert_yaxis()
    for i, (m, s) in enumerate(zip(means.values, stds.values)):
        ax.text(m + s + 30, i, f"{m:,.0f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_main_benchmark.png")
    plt.close(fig)


def fig4_generalization():
    df = pd.read_csv("ANALYSIS/results/robustness_final_distilled_seed2026_variants.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    wl = df.dropna(subset=["n_workloads"]).groupby("n_workloads")["system_cost"].agg(["mean", "std"])
    axes[0].errorbar(wl.index, wl["mean"], yerr=wl["std"], marker="o", color="#2c7fb8")
    axes[0].set_xlabel("Workload count (training used 10; 6-9 are out-of-distribution)")
    axes[0].set_ylabel("System cost, $")
    axes[0].set_title("(a) Generalization to unseen workload counts")
    axes[0].axvline(10, color="gray", linestyle="--", linewidth=1, label="training distribution")
    axes[0].legend(fontsize=8)

    sl = df.dropna(subset=["slack_scale"]).groupby("slack_scale")["system_cost"].agg(["mean", "std"])
    axes[1].errorbar(sl.index, sl["mean"], yerr=sl["std"], marker="s", color="#e34a33")
    axes[1].set_xlabel("Deadline-slack scale (1.0 = training distribution)")
    axes[1].set_title("(b) Sensitivity to deadline tightness")
    axes[1].axvline(1.0, color="gray", linestyle="--", linewidth=1)
    fig.suptitle("DFPS generalization / robustness to problem-structure shift (validation-derived scenarios)")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_generalization.png")
    plt.close(fig)


def fig5_ablation_robustness():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    labels = ["DFPS\n(full)", "- RL\nfinetune", "- topology\n(GAT)", "TAGS\n(- distillation)"]
    means = [11229.65, 11228.78, 11230.44, 11854.01]
    stds = [195.99, 194.86, 195.68, 574.55]
    colors = ["#2c7fb8", "#7fcdbb", "#7fcdbb", "#e34a33"]
    axes[0].bar(labels, means, yerr=stds, color=colors, capsize=3)
    axes[0].set_ylabel("System cost, $ (validation)")
    axes[0].set_title("(a) Ablation: removing one mechanism at a time")
    axes[0].set_ylim(10800, 12700)

    rob = pd.read_csv("ANALYSIS/results/robustness_final_distilled_seed2026.csv")
    cond_order = ["clean", "price_noise_low", "price_noise_high", "cef_noise_high", "bandwidth_dropout_20pct", "bandwidth_dropout_50pct"]
    cond_labels = ["Clean", "Price\nnoise\n(low)", "Price\nnoise\n(high)", "CEF\nnoise\n(high)", "BW\ndropout\n20%", "BW\ndropout\n50%"]
    means_r = [rob[rob["condition"] == c]["system_cost"].mean() for c in cond_order]
    stds_r = [rob[rob["condition"] == c]["system_cost"].std() for c in cond_order]
    axes[1].bar(cond_labels, means_r, yerr=stds_r, color="#2c7fb8", capsize=3)
    axes[1].set_title("(b) DFPS robustness to input perturbation")
    axes[1].set_ylabel("System cost, $")
    axes[1].set_ylim(10800, 11700)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_ablation_robustness.png")
    plt.close(fig)


def fig6_pareto():
    eff = pd.read_csv("ANALYSIS/results/efficiency_table.csv")
    cost = pd.read_csv("ANALYSIS/results/test_comparison_combined.csv").groupby("method")["system_cost"].mean()
    name_map = {
        "Source_PPO": "Source_PPO", "A2C": "A2C", "MLP-PPO_no_attn": "MLP-PPO_no_attn",
        "HCBS-PPO": "HCBS-PPO", "TAGS": "TAGS", "DFPS": "DFPS_final", "CP-SAT": "CP-SAT",
        "Minimal_Distilled": "Minimal_Distilled",
    }
    points = []
    for disp, effrow in name_map.items():
        row = eff[eff["method"] == effrow]
        if row.empty or disp not in cost.index:
            continue
        mins = row.iloc[0]["total_cost_5_seeds_isolated_min"]
        if pd.isna(mins):
            continue
        points.append((disp, cost[disp], float(mins)))

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for disp, c, mins in points:
        is_dfps = disp == "DFPS"
        is_minimal = disp == "Minimal_Distilled"
        color = "#2c7fb8" if is_dfps else ("#fdae61" if is_minimal else "#636363")
        marker = "*" if is_dfps else ("D" if is_minimal else "o")
        size = 260 if is_dfps else (150 if is_minimal else 90)
        ax.scatter(mins, c, color=color, marker=marker, s=size, zorder=3)
        offset = (6, -14) if is_minimal else (6, 4)
        ax.annotate(disp, (mins, c), textcoords="offset points", xytext=offset, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Total training compute, 5 seeds, isolated wall-clock (minutes, log scale)")
    ax.set_ylabel("Test-split system cost, $ (lower is better)")
    ax.set_title("Performance-efficiency Pareto: cost vs. total training compute")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig6_pareto.png")
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    fig3_main_benchmark()
    fig4_generalization()
    fig5_ablation_robustness()
    fig6_pareto()
    print("figures written to", OUT)
