# Final Research Summary

## Source paper
"Deep Reinforcement Learning-Based Migration and Scheduling Strategy for Data Centers in Electricity and Carbon Markets" (Wang, Mi, He, Wang, Wang, Fotuhi-Firuzabad, Sun), *IEEE Transactions on Industry Applications*, 2026, DOI 10.1109/TIA.2026.3691718. A PPO-based masked-attention scheduler assigns dependency-constrained subtasks across three data centers (DCs) participating in regional electricity markets and a unified national carbon market, claiming a 4.1% system-cost reduction over a zero-delay baseline.

## Reproduction: success/failure
**Conceptually reproduced, not numerically reproducible** — no author code, data, reward weights, or full training configuration are publicly available (`SOURCE_PAPER_AUDIT.md`). We built a frozen, documented synthetic-scenario reconstruction matching the stated protocol and trained the described masked-attention PPO architecture with every fully-specified hyperparameter, to the stated 3,000-episode budget. Qualitative claims reproduce directionally (PPO beats the paper's own weak S1/S2 baselines; PPO does not reduce carbon emissions vs. S1, matching the paper's own table). A finding **not visible in the published evaluation** emerged from our reproduction: the trained PPO policy is statistically indistinguishable from a trivial, non-learned greedy heuristic (0.08% difference in validation system cost), because the source paper never evaluates against a non-learned spatial baseline. Full numbers: `REPRODUCTION_REPORT.md`.

## Datasets/test systems used
A frozen synthetic scenario generator (three DCs, 96 quarter-hour slots, 10 workloads / 50 dependency-constrained subtasks per scenario), documented as a synthetic reconstruction, not author data, in `DATA_AUDIT.md`. Local candidate data sources (GEFCom2014 price/solar/wind, `C:\Users\MMASSAOUDI\Desktop\Data`) were evaluated and documented as available for external validation but not required, since the scenario generator already provides the necessary market/renewable heterogeneity for this study's scope.

## Simulation framework
Custom Python/PyTorch environment (`SOURCE_METHOD_REPRODUCTION/src/environment.py` for the autoregressive/source-faithful variant; `PROPOSED_METHOD/src/env_frontier.py` for the batched-frontier variant used by the proposed candidates), plus OR-Tools CP-SAT for the exact/near-exact MILP benchmark and teacher-label generation.

## Candidate models evaluated
Five candidates targeting weaknesses identified in the reproduction (`MODEL_CANDIDATES.md`, `SOURCE_WEAKNESS_ANALYSIS.md`): GAT-PPO (topology-aware autoregressive), LC-PPO (Lagrangian carbon-constrained autoregressive), HCBS-PPO (batched-frontier, cached horizon), TAGS (combined hybrid: topology + batching + Lagrangian), and DFPS (CP-SAT-teacher distillation into the TAGS architecture, plus light RL fine-tune). Screened under a strict validation-only, staged protocol (smoke test → 3-seed reduced-budget screening → Optuna tuning → full 5-seed final training); full chronology including a discovered-and-fixed PID-Lagrangian saturation bug in `MODEL_SELECTION_REPORT.md`.

## Final selected model and reason for selection
**DFPS (Distilled Frontier-Priority Scheduler)**: imitation learning from 200 CP-SAT-solved training scenarios (995 samples, 20 epochs, top-1 exact-slot accuracy ≈73.3%), followed by a 300-episode PID-Lagrangian PPO fine-tune, on the same topology- and horizon-aware batched-frontier architecture as TAGS. Selected because it achieved the best validation system cost of every learned candidate built (\$11,229.65 ± 195.99), with by far the lowest seed-to-seed variance, at roughly 3.6× less total training compute across 5 seeds than the from-scratch alternatives. Full selection rationale and rejected alternatives: `MODEL_SELECTION_REPORT.md`.

## Strongest benchmark
CP-SAT (exact/near-exact MILP solver, identical constraint formulation, 30 s/scenario time limit, 77–84% of scenarios solved to proven optimality): test-split system cost \$10,823.4 ± 186.9 — the near-optimal ceiling against which every learned/heuristic method is compared.

## Source-paper benchmark result
Reproduced source PPO, test split: \$10,883.3 ± 182.1 (electricity + carbon cost), carbon emissions 10,745 kg.

## Proposed-model result
DFPS, test split: \$11,231.8 ± 184.8 (3.2% above source PPO, 3.8% above CP-SAT), carbon emissions 10,798 kg, decision latency 12.67 ms, 202,162 parameters.

## Wins/ties/losses against benchmarks
7 wins / 0 ties / 7 losses against the full 14-benchmark suite (paired *t*-test, Holm-corrected, α=0.05; `BENCHMARK_WTL.csv`). Wins are practically large (5.2–15.1% improvement) against weak/moderate baselines (S1, S2, EDF, random, carbon-greedy) and both of our other candidates (HCBS-PPO, TAGS); losses are practically small (0.3–3.8%) against the strongest cluster (DQN, attention-ablated PPO, A2C, source PPO, HEFT, greedy, CP-SAT).

## Macro system cost / minority-class-analog / FPR-analog metrics
This is a sequential scheduling task, not a classification problem — macro-F1, minority-class recall, and FPR (as specified generically in the master protocol for classification tasks) do not apply. The domain-appropriate primary metric (system cost, analogous to a primary loss/objective metric) and secondary metrics (carbon-quota compliance, decision latency, parameter count, training compute) were used throughout instead, consistent with the protocol's instruction to add task-appropriate metrics. **Multiclass evaluation is explicitly not applicable** for the same reason.

## Efficiency improvement
DFPS: ≈55 minutes total training compute across 5 seeds (one-time ≈51-minute CP-SAT teacher generation, amortized, plus ≈1 minute/seed thereafter) vs. ≈200 minutes for the source PPO reproduction trained independently per seed — a 3.6× reduction in total compute and a ≈40× reduction in marginal per-seed cost, at isolated (uncontended) wall-clock measurement. The shared batched-frontier architecture (HCBS-PPO/TAGS/DFPS) trains ≈16× faster per episode than the source method's autoregressive architecture, independent of the distillation question. Full table: `ANALYSIS/results/efficiency_table.csv`.

## Ablation conclusion
Honest, load-bearing-mechanism finding: **removing the topology (GAT) encoder or the RL fine-tune phase from DFPS produces no measurable change** in cost or carbon performance (differences of ≤1.7 cost units against a ≈196 standard deviation). Removing the distillation mechanism itself (TAGS, identical architecture trained from scratch instead) costs 5.6% and triples seed variance. The training methodology — imitating a near-optimal solver — is the actual source of DFPS's performance, not the specific hybrid architectural components independently motivated by the weakness analysis. Full report: `ANALYSIS/results/ablation_report.md`.

## Robustness conclusion
DFPS is stable under price/CEF sensor noise and bandwidth-signal dropout (up to 50%): system cost varies by less than 0.5% across all tested perturbation conditions relative to clean-condition performance. It generalizes smoothly (no collapse or discontinuity) to out-of-distribution workload counts (6–9, vs. the training distribution of 10) and to deadline-slack scaling (0.5×–1.5× nominal). Zero capacity violations across every evaluation reported in this study.

## Statistical significance
All main-comparison and ablation results use 5 independent seeds; win/tie/loss verdicts use paired *t*-tests with Holm-Bonferroni correction across all 14 simultaneous comparisons at α=0.05, reported alongside Wilcoxon signed-rank p-values and Cohen's *d* to separate statistical from practical significance (`BENCHMARK_WTL.csv`).

## Remaining limitations
1. **Carbon-quota compliance is not achieved by the final model.** DFPS averages 10,798–10,851 kg emissions against a 9,500 kg quota. TAGS (same architecture, trained from scratch with the corrected Lagrangian mechanism) reaches materially lower emissions (10,341–10,365 kg) at a real, quantified cost premium, and an even-earlier (buggy) TAGS run achieved full compliance (9,277 kg) at an indefensible cost (\$12,960) — reported honestly as an unresolved trade-off, not folded into DFPS's headline result.
2. Absolute dollar/kg values throughout this study are not comparable to the source paper's published table, since no author data exists; only internal (same-synthetic-data) comparisons and qualitative directional claims are supported.
3. DFPS's statistical losses against the strongest benchmark cluster (CP-SAT, greedy, HEFT, source PPO) are small in percentage terms (0.3–3.8%) but statistically significant given the large paired sample and low per-scenario variance; DFPS is not the single best method in the suite.
4. Several training-cost figures for autoregressive-family benchmarks (A2C, MLP-PPO-no-attention) are extrapolated from the source PPO's isolated per-episode cost rather than independently isolated-timed, due to shared-machine compute contention encountered partway through this study; this is disclosed in `ANALYSIS/results/efficiency_table.csv`'s notes column.
5. No real cloud-workload DAG trace, cooling/thermal model, migration-energy model, or market-clearing mechanism is included, consistent with the source paper's own scope and explicitly documented as a shared limitation in `SOURCE_PAPER_AUDIT.md`.
6. Literature/novelty verification (`LITERATURE_NOVELTY_CHECK.md`) used two targeted web searches rather than an exhaustive systematic review; the novelty claim is deliberately narrow and hedged accordingly.

## Recommended IEEE Transactions venue
*IEEE Transactions on Industry Applications* (matching the source paper) or *IEEE Transactions on Sustainable Computing* / *IEEE Transactions on Cloud Computing*, given the paper's efficiency-of-training and solver-distillation framing is arguably a closer fit for the latter two venues' scope than the source paper's original placement.
