from __future__ import annotations

import pandas as pd

OUT = "ANALYSIS/tables"


def main_comparison_table():
    df = pd.read_csv("ANALYSIS/results/test_comparison_combined.csv")
    agg = df.groupby("method").agg(cost_mean=("system_cost", "mean"), cost_std=("system_cost", "std"),
                                    carbon_mean=("carbon_emissions_kg", "mean")).reset_index()
    lat = df.groupby("method")["mean_decision_latency_ms"].mean()
    agg["latency"] = agg["method"].map(lat)
    agg = agg.sort_values("cost_mean")
    display_names = {
        "CP-SAT": "CP-SAT (MILP, near-optimal)", "Greedy": "Greedy spatiotemporal", "HEFT": "HEFT",
        "Source_PPO": "Source PPO (reproduced)", "A2C": "A2C", "MLP-PPO_no_attn": "MLP-PPO (no attn.)",
        "DQN": "DQN", "DFPS": "\\textbf{DFPS (proposed, final)}", "HCBS-PPO": "HCBS-PPO",
        "S2_single_dc": "S2 single-DC temporal", "TAGS": "TAGS", "S1_zero_delay": "S1 zero-delay",
        "Random": "Random feasible", "EDF": "EDF", "Carbon_Greedy": "Carbon-greedy",
        "Minimal_Distilled": "Minimal-distilled (168K params)",
        "TAGS_Compliant": "TAGS-Compliant (quota-compliant)",
    }
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Test-split comparison (100 scenarios). System cost in \\$; mean $\\pm$ std. Lower cost is better. DFPS is the final selected model.}",
        "\\label{tab:main_comparison}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Method & System cost (\\$) & Carbon (kg) & Latency (ms) \\\\",
        "\\midrule",
    ]
    for _, r in agg.iterrows():
        name = display_names.get(r["method"], r["method"])
        lat_str = f"{r['latency']:.2f}" if pd.notna(r["latency"]) else "--"
        lines.append(f"{name} & {r['cost_mean']:,.1f} $\\pm$ {r['cost_std']:,.1f} & {r['carbon_mean']:,.0f} & {lat_str} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(f"{OUT}/main_comparison.tex", "w") as f:
        f.write("\n".join(lines))


def wtl_table():
    df = pd.read_csv("BENCHMARK_WTL.csv").sort_values("mean_diff", ascending=False)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{DFPS win/tie/loss vs. all 15 comparisons (test split, paired $t$-test, Holm-corrected, $\\alpha$=0.05). Positive \\% improvement favors DFPS. The Minimal-distilled comparison is Holm-significant only because of the paired design's very low variance (Cohen's $d=-0.20$, the smallest magnitude in the table by an order of magnitude); at 0.04\\% it is a practical tie, discussed in the text.}",
        "\\label{tab:wtl}",
        "\\begin{tabular}{lrrl}",
        "\\toprule",
        "Benchmark & \\% improvement & Cohen's $d$ & Verdict \\\\",
        "\\midrule",
    ]
    display_names = {
        "Carbon_Greedy": "Carbon-greedy", "EDF": "EDF", "Random": "Random feasible",
        "S1_zero_delay": "S1 zero-delay", "TAGS": "TAGS", "S2_single_dc": "S2 single-DC temporal",
        "HCBS-PPO": "HCBS-PPO", "DQN": "DQN", "MLP-PPO_no_attn": "MLP-PPO (no attn.)", "A2C": "A2C",
        "Source_PPO": "Source PPO (reproduced)", "HEFT": "HEFT", "Greedy": "Greedy spatiotemporal", "CP-SAT": "CP-SAT",
        "Minimal_Distilled": "Minimal-distilled (practical tie)",
    }
    for _, r in df.iterrows():
        name = display_names.get(r["benchmark"], r["benchmark"]).replace("_", "\\_")
        lines.append(f"{name} & {r['pct_improvement']:.2f} & {r['cohens_d']:.2f} & {r['verdict']} \\\\")
    n_win = (df["verdict"] == "win").sum()
    n_loss = (df["verdict"] == "loss").sum()
    n_tie = (df["verdict"] == "tie").sum()
    lines += [
        "\\midrule",
        f"\\multicolumn{{4}}{{l}}{{\\textbf{{DFPS: {n_win} wins / {n_tie} ties / {n_loss} losses}}}} \\\\",
        "\\bottomrule", "\\end{tabular}", "\\end{table}",
    ]
    with open(f"{OUT}/wtl.tex", "w") as f:
        f.write("\n".join(lines))


def ablation_table():
    rows = [
        ("DFPS (full)", 11229.65, 195.99, 10851.45, 202162, "---"),
        ("$-$ RL fine-tune (imitation only)", 11228.78, 194.86, 10874.21, 202162, "PID-Lagrangian PPO adapter"),
        ("$-$ topology (HCBS-style)", 11230.44, 195.68, 10848.24, 202162, "Neighborhood-attention (GAT) encoder"),
        ("Minimal-distilled (source arch.)", 11224.70, 192.32, 10863.93, 168194, "Topology \\& batched-frontier env.\\ (both)"),
        ("TAGS ($-$ distillation, from-scratch RL)", 11854.01, 574.55, 10365.06, 202162, "CP-SAT-teacher distillation"),
    ]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Ablation study (validation split, 5 seeds, paired significance vs.\\ DFPS across the 100 validation scenarios). Removing the topology encoder, the RL fine-tune phase, or both plus the batched-frontier environment (Minimal-distilled, the source paper's own unmodified architecture) all tie DFPS (Holm-corrected $p>0.05$). Removing the distillation mechanism (TAGS) costs 5.6\\% and 3x the seed variance ($p<10^{-88}$).}",
        "\\label{tab:ablation}",
        "\\begin{tabular}{lrrrl}",
        "\\toprule",
        "Variant & System cost (\\$) & Carbon (kg) & Params & Mechanism removed \\\\",
        "\\midrule",
    ]
    for name, cost, std, carbon, params, removed in rows:
        lines.append(f"{name} & {cost:,.2f} $\\pm$ {std:,.2f} & {carbon:,.2f} & {params:,} & {removed} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(f"{OUT}/ablation.tex", "w") as f:
        f.write("\n".join(lines))


def efficiency_table():
    df = pd.read_csv("ANALYSIS/results/efficiency_table.csv")
    keep = ["Source_PPO", "A2C", "MLP-PPO_no_attn", "HCBS-PPO", "TAGS", "DFPS_final", "Minimal_Distilled", "CP-SAT"]
    df = df[df["method"].isin(keep)]
    display_names = {"Source_PPO": "Source PPO", "A2C": "A2C", "MLP-PPO_no_attn": "MLP-PPO (no attn.)",
                      "HCBS-PPO": "HCBS-PPO", "TAGS": "TAGS", "DFPS_final": "\\textbf{DFPS (proposed)}",
                      "Minimal_Distilled": "Minimal-distilled", "CP-SAT": "CP-SAT"}
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Training-compute efficiency (isolated wall-clock measurements). DFPS's one-time CP-SAT teacher cost is amortized across all 5 seeds.}",
        "\\label{tab:efficiency}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Method & Params & Total cost, 5 seeds (min) & Decision latency (ms) \\\\",
        "\\midrule",
    ]
    for _, r in df.iterrows():
        name = display_names.get(r["method"], r["method"])
        total = r["total_cost_5_seeds_isolated_min"]
        total_str = f"{total:,.1f}" if pd.notna(total) else "--"
        lines.append(f"{name} & {int(r['n_params']):,} & {total_str} & {r['mean_decision_latency_ms']:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    with open(f"{OUT}/efficiency.tex", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    main_comparison_table()
    wtl_table()
    ablation_table()
    efficiency_table()
    print("tables written to", OUT)
