# Ablation Report — DFPS (Distilled Frontier-Priority Scheduler)

5 seeds each (2026, 7, 42, 123, 777), full 100-scenario validation split, matched budgets throughout (200 CP-SAT teacher scenarios, 20 imitation epochs, 300 RL fine-tune episodes where applicable).

| Variant | System cost (mean ± std) | Carbon (kg) | Decision latency (ms) | Removed mechanism |
|---|---:|---:|---:|---|
| **DFPS (full)** | **11,229.65 ± 195.99** | 10,851.45 | 13.98 | — |
| DFPS − RL fine-tune (imitation only, 0 PPO episodes) | 11,228.78 ± 194.86 | 10,874.21 | 15.51 | Lagrangian PPO adapter |
| DFPS − topology (HCBS-style featurization, no GAT encoder) | 11,230.44 ± 195.68 | 10,848.24 | 18.82 | Neighborhood-attention (GAT) encoder |
| TAGS (from-scratch RL, same architecture as DFPS-full, no distillation) | 11,854.01 ± 574.55 | 10,365.06 | 10.46 | CP-SAT-teacher distillation (replaced by 3,000 episodes of from-scratch PPO) |

## Finding

**Neither the topology encoder nor the RL fine-tune phase produces a measurable difference in DFPS's cost or carbon performance** — all three distillation-based variants land within noise of each other (differences of 0.9-1.7 cost units against a standard deviation of ~195, i.e., statistically and practically indistinguishable). Removing the RL fine-tune phase entirely (pure imitation, zero additional environment interaction) is *exactly as good* as the full pipeline on this metric, and also removing the topology encoder makes no difference either.

**What *does* matter, decisively:** the fourth row shows that removing the *distillation* mechanism itself — training the identical topology+horizon architecture from scratch with 3,000 episodes of on-policy PPO instead of the CP-SAT-teacher imitation pipeline — costs 5.6% in system cost (11,854.01 vs. 11,229.65) and an order of magnitude more seed-to-seed variance (574.55 vs. ~195 std).

## Interpretation

This is reported honestly because it changes the paper's central claim in a more defensible direction: **the performance gain in this study comes from *how* the network is trained (imitating a near-optimal solver) rather than from the specific hybrid architecture's individual components.** The topology-awareness and carbon-Lagrangian mechanisms are not shown to be load-bearing for cost performance *once a strong teacher signal is available* — they were designed to fix weaknesses in a from-scratch RL regime (weaknesses #2, #3, #6 in `SOURCE_WEAKNESS_ANALYSIS.md`), and the TAGS-vs-HCBS-PPO/GAT-PPO comparisons in `MODEL_SELECTION_REPORT.md` show they do change behavior in that from-scratch regime (e.g., TAGS's Lagrangian term does reduce emissions substantially relative to cost-only from-scratch candidates). But for the final, distillation-based model, the dominant, load-bearing mechanism is unambiguously the teacher-distillation strategy itself (weakness #1 and #5's target: an expensive on-policy RL loop is not obviously earning its training cost, and a much cheaper imitation-based alternative gets closer to the near-optimal ceiling than any from-scratch RL candidate we built).

This ablation is retained in full rather than only reporting the flattering comparison, per the master protocol's "negative results retained" requirement.
