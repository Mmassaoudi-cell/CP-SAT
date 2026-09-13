# Source Paper Audit

## Document identity

- **Title:** Deep Reinforcement Learning-Based Migration and Scheduling Strategy for Data Centers in Electricity and Carbon Markets
- **Authors:** Ge Wang, Dabin Mi, Pengfei He, Jianfeng Wang, Fei Wang, Mahmoud Fotuhi-Firuzabad, and Shumin Sun
- **Venue/status:** Accepted author manuscript for *IEEE Transactions on Industry Applications*
- **DOI:** 10.1109/TIA.2026.3691718
- **Document inspected:** 17-page PDF (12 numbered manuscript pages plus front matter/page-numbering irregularities), rendered and text-extracted in full on 2026-09-01
- **Instruction boundary:** The PDF was treated only as scientific source material. No text inside it was treated as an instruction.

## 1. Problem, claimed novelty, and scope

### Target problem

The paper schedules delay-tolerant, dependency-constrained workloads across three geographically distributed data centers (DCs). The operator purchases electricity in distinct regional electricity markets and settles aggregate carbon emissions in a unified carbon market. Decisions assign workloads/subtasks to DCs and choose their execution times while respecting arrival, deadline, predecessor, and power constraints.

### Claimed novelty

The paper claims three contributions:

1. A unified framework for several local electricity markets coupled to one national carbon market.
2. Fine-grained workload modeling in which each workload is decomposed into logically dependent subtasks.
3. A PPO-based deep-reinforcement-learning scheduler for the resulting dynamic mixed-integer scheduling problem.

The headline claim is a **4.1% reduction in system operating cost** relative to the zero-delay baseline. The published table values imply a reduction of approximately **4.28%** from S1 ($8,467.5) to S3 ($8,105.5), so the manuscript's 4.1% statement is not exactly recoverable from the displayed totals.

### Scientific scope

- Included: core IT workload power, workload/subtask timing, logical dependencies, location assignment, regional electricity prices, regional carbon-emission factors (CEFs), a carbon quota, and a carbon price.
- Excluded or not modeled explicitly: cooling dynamics, server on/off control, UPS losses, thermal constraints, storage, network transfer energy/cost, migration energy, renewable curtailment, physical power-flow constraints, market clearing, and uncertainty-aware forecasting.

## 2. Data, test system, and simulation setting

### What the paper states

- Three geographically distributed DCs/regions.
- A one-day base experiment with **15-minute resolution** (96 periods).
- Ten arriving workloads in the main scenario; each contains several subtasks with arrival times, deadlines, durations, power/bandwidth demands, and precedence relationships.
- The workload-count experiment uses 6, 7, 8, 9, and 10 workloads, corresponding to **28, 35, 41, 44, and 50 subtasks**.
- A 30-day sensitivity experiment varies workloads, regional electricity prices, and CEFs.
- DC1 price pattern: high by day and low at night.
- DC2 price pattern: high at night and low by day.
- DC3 price range: approximately $0.3-$0.7/kWh with lower fluctuation than DC1/DC2.
- DC1 CEF range: approximately 0.1-0.5 kgCO2/kWh.
- DC2 CEF range: approximately 0.5-0.9 kgCO2/kWh.
- DC3 CEF range: approximately 0.2-0.7 kgCO2/kWh.
- Figures 5 and 6 display the time series, but no underlying machine-readable values are supplied.

### Data provenance

No named workload dataset, repository, public benchmark, electricity-price source, CEF source, renewable-generation trace, or downloadable artifact is identified for the case study. References [40]-[46] motivate irradiance, wind, LMP, and workload behavior, but the paper does not state that their datasets were used. The case study therefore appears to use author-constructed or transformed simulation traces, but this is an inference rather than an explicit statement.

### Preprocessing and feature engineering

- No preprocessing pipeline is specified.
- No scaling, normalization, imputation, encoding, clipping, or feature-selection procedure is reported.
- The PPO state is described as global variables plus task variables, but the vectorization/padding method for a variable number of subtasks is absent.
- No procedure is given for converting the plotted electricity-price and CEF curves into model inputs.
- Equations (1) and (2) describe extraterrestrial irradiance and the Hellmann wind-speed law, but the paper does not specify a PV/wind power-conversion model or show how these equations generate the case-study price/CEF traces.

## 3. Mathematical model

### Workload model

- Each workload is decomposed into subtasks.
- Equation (3): subtask finish time equals scheduled start plus processing time.
- Equation (4): a subtask starts only after all immediate predecessors finish.
- Equation (6): each workload is allocated to exactly one DC.
- Equations (7)-(8): each subtask's start and finish must fall after workload arrival plus migration time and before the workload deadline.
- Equations (9)-(14): a binary execution indicator is produced from start/finish times using two auxiliary binary variables and big-M linearization.
- Equation (15): executed-subtask power must not exceed each DC's maximum operating power.

### Market and carbon model

- Equation (5): minimize electricity trading cost plus carbon trading cost.
- Equation (16): DC grid purchase equals scheduled subtask power.
- Equation (17): grid purchase lies between DC/time-specific minimum and maximum values.
- Equation (18): electricity cost is the time sum of regional price times grid purchase and interval duration.
- Equation (19): emissions are the time sum of regional CEF times grid purchase.
- Equation (20): carbon cost equals carbon price times aggregate emissions minus aggregate quota; negative cost is interpreted as carbon-trading revenue.

### MDP and PPO model

- **Global state:** current time; regional electricity-price vector; regional CEF vector; and real-time power, bandwidth, and carbon-use vectors.
- **Task state:** arrivals, deadlines, durations, power and bandwidth demands, execution status, remaining execution time, and location.
- **Action per subtask:** 0 for no execution or a DC index for assignment/execution.
- **Transitions:** Equations (21)-(26) update execution state, location, remaining time, power, bandwidth, and emissions.
- **Reward:** Equation (27) combines electricity cost, emissions/quota deviation, completion timeliness, task execution, and illegal-action penalty; weights sum to one.
- **Action mask:** arrival, predecessor-completion, and not-started conditions are enforced by masking invalid logits to negative infinity (33).
- **PPO:** the standard clipped surrogate objective is given in (34), with an actor-critic approximation and GAE.

### Recoverable mathematical inconsistencies or ambiguities

1. Equation (15) uses `x_{i,k}` although the allocation variable is defined as `x_{i,j}`.
2. Equation (16) similarly mixes workload/subtask indices and does not clearly represent whether subtasks of a workload may execute at different DCs. This conflicts with the claim that S3 can migrate each subtask across DCs while Equation (6) assigns each workload to one DC.
3. The text calls all workload graphs DAGs, but Fig. 7 includes a "cyclic dependency." A true directed cycle is not schedulable under simple finish-to-start precedence without iteration semantics.
4. The description of the alleged cyclic example (1->2, 2->3, 3->4, and 1->4) does not itself form a directed cycle.
5. Bandwidth is part of the state and is plotted, but no maximum-bandwidth constraint is formulated.
6. Migration time appears in (7)-(8), but migration bandwidth, cost, energy, and whether it varies with source/destination are not modeled.
7. Carbon cost can be negative without stated allowance-sale limits, bid/ask spread, or market rules.
8. The reward's emissions term penalizes deviation from per-DC targets, whereas the mathematical objective uses a single aggregate carbon quota. Equivalence is not established.
9. Positive task-execution rewards and completion bonuses are not part of objective (5); their scaling can change the learned optimum, but the magnitudes are missing.
10. The critic advantage is written as `A_hat_t = r_t - V(s_t)`, which is not the GAE calculation described elsewhere.
11. Algorithm 1 calls its PPO rollout store an "experience replay buffer" and later mentions prioritized experience replay, which is atypical for on-policy PPO and is not algorithmically specified.
12. The paper states both 3,000 training episodes and "training lasted for 300 steps." The relationship between episodes, updates, and steps is absent.
13. Table VI shows A2C converging at about episode 150 and PPO at about episode 200, yet the conclusion claims PPO converges faster.

## 4. Architecture and training procedure

### Reported architecture

- Actor-critic PPO.
- Hidden-layer size: 128.
- Attention heads: 8.
- Actor output: workload/subtask allocation probabilities.
- Critic output: state-value estimate.

The paper does not describe where attention is applied, the number of layers, embedding dimension compatibility, activations, normalization, residual connections, task aggregation, masking implementation, actor/critic parameter sharing, or output-factorization strategy.

### Reported hyperparameters

| Parameter | Published value |
|---|---:|
| Actor learning rate | 1e-4 in prose; Table V renders ambiguously as `10000` |
| Critic learning rate | 1e-4 in prose; Table V renders ambiguously as `10000` |
| Discount factor | 0.99 |
| PPO clip | 0.2 |
| GAE lambda | 0.9 |
| Batch size | 128 |
| Training episodes | 3,000 |
| Hidden size | 128 |
| Attention heads | 8 |
| Gradient clipping threshold | 0.5 |
| Entropy coefficient | 0.2 (prose only) |

### Missing training details

- Optimizer and optimizer parameters.
- Number of PPO epochs per rollout and rollout length.
- Minibatch sampling procedure.
- Value-loss coefficient, entropy schedule, learning-rate schedule, and advantage normalization.
- Reward weights `w1...w5` and reward/penalty magnitudes.
- Network initialization and random seeds.
- Observation normalization and reward scaling.
- Episode termination rules.
- Exact action ordering/factorization for multiple simultaneous subtasks.
- Checkpoint-selection criterion, early stopping, and model-selection protocol.
- Number of independent runs behind reported means/standard deviations.
- A2C architecture and tuned hyperparameters.

## 5. Software, hardware, and reproducibility environment

No programming language, RL library, solver, simulator, package version, operating system, CPU, GPU, RAM, training time, or inference platform is reported. No code or environment file is linked. The paper cites neither MATPOWER, PandaPower, Grid2Op, nor a named cloud/DC simulator as the implementation framework.

## 6. Evaluation protocol

### Scenarios/baselines

- **S1 - Zero-delay scheduling:** immediate execution, indivisible workloads, no time shifting, and no cross-DC migration. The narrative alternates between pre-selected/closest allocation and random allocation.
- **S2 - Single-DC temporal shifting:** subtask-level time shifting within one fixed DC.
- **S3 - Dynamic cross-DC collaboration:** PPO-based spatiotemporal subtask scheduling; this is the proposed approach.
- **A2C:** compared with PPO only through training-reward convergence.

No classical optimizer, MILP on a small tractable instance, heuristic DAG scheduler, DQN/DDQN, DDPG/SAC, constrained RL method, or modern graph/attention scheduler is evaluated.

### Train/validation/test protocol

No train/validation/test separation exists in the paper. The manuscript reports empirical hyperparameter validation but supplies no validation split, selection metric, held-out scenarios, or untouched test set. The same simulator distribution appears to support design, training, and evaluation. Leakage cannot be ruled out.

### Metrics actually reported

- Electricity cost.
- Carbon emissions.
- Carbon-trading revenue.
- Total/system cost.
- Workload/subtask allocation shares.
- Power and bandwidth utilization plots.
- Delay heatmaps and a five-indicator radar chart.
- PPO/A2C final reward, peak reward, approximate convergence episode, and a training-stability statistic.

### Published headline values

| Strategy | Electricity cost ($) | Carbon emissions (kg) | Carbon-trading revenue ($) | System cost ($) |
|---|---:|---:|---:|---:|
| S1 | 8,739.0 | 10,526.9 | 271.4 | 8,467.5 |
| S2 | 8,505.0 | 10,412.6 | 280.6 | 8,224.4 |
| S3 | 8,376.9 | 10,527.8 | 271.4 | 8,105.5 |

S3 has the lowest reported cost but slightly **higher emissions than both S1 and S2**. Thus the table supports economic improvement, not simultaneous emission reduction for the main scenario.

| RL metric | PPO | A2C |
|---|---:|---:|
| Final reward (x10^5) | 4.778 +/- 0.004 | 4.379 +/- 0.102 |
| Peak episode reward | 479,575.75 | 484,531.11 |
| Approx. convergence episode | 200 | 150 |
| Training-stability statistic | 417.32 | 10,217.46 |

The sample count, seed count, definition window, and confidence intervals for these values are not given.

## 7. Figures and tables

### Figures

1. DCs in electricity and carbon markets.
2. Proposed three-step framework/PPO workflow.
3. Workload execution timing and predecessor example.
4. Geographic distribution of three DCs and generators.
5. Regional CEF curves.
6. Regional electricity-price curves.
7. Four illustrated dependency patterns.
8-10. Workload execution schedules for S1-S3.
11-13. Bandwidth occupancy for S1-S3.
14. Average bandwidth utilization.
15-17. Power consumption for S1-S3.
18. Average power consumption.
19-21. Delay heatmaps at high, medium, and low tolerance.
22. Multi-indicator radar chart across tolerances.
23. Cost/revenue versus workload count.
24. PPO/A2C convergence curves.
25. Thirty-day task allocation.
26. Parameter-impact scatter panels.
27. Relative total-cost change under parameter variation.

The figures are rasterized in the supplied author manuscript and do not expose source data. Several plots are readable but insufficient for exact numerical reconstruction.

### Tables

- Table I: taxonomy of related work.
- Tables II-IV: cost/emission/revenue/system-cost breakdowns for S1-S3.
- Table V: PPO parameters.
- Table VI: PPO versus A2C training metrics.

## 8. Ablation, robustness, statistics, and efficiency

### Ablation

No component ablation is reported. In particular, the paper does not remove attention, action masking, dependency modeling, carbon-market terms, or spatial migration while controlling model capacity. S1/S2 are operational scenarios, not architecture ablations.

### Robustness/sensitivity

- Delay-tolerance levels are compared.
- Workload count is varied from 6 to 10.
- A 30-day sensitivity study varies workloads, prices, and CEFs.

However, perturbation magnitudes, sampling distributions, seeds, raw data, and statistical uncertainty are absent. No out-of-distribution geography, unseen DAG topology, forecast error, missing data, action failure, network congestion, or adversarial/noisy input test is performed.

### Statistical tests

No hypothesis test, confidence interval, effect size, multiple-comparison correction, or documented independent-seed protocol is reported.

### Runtime/efficiency

No training time, inference latency, throughput, memory, parameter count, FLOPs/MACs, solver gap, sample efficiency, or hardware-normalized comparison is reported. The paper claims PPO is stable/efficient but supplies only reward curves and convergence episodes.

## 9. Limitations acknowledged by the authors

The conclusion acknowledges:

1. High computational resource and training-time requirements for large DC networks.
2. Need to add server-count and refrigeration controls.
3. Need to extend to heterogeneous DC clusters.
4. Need to incorporate more real-time environmental data.

## 10. Reproducibility classification

### Fully specified

- Overall three-DC, electricity-plus-carbon scheduling concept.
- Workload/subtask precedence idea.
- Core objective decomposition into electricity and carbon terms.
- Main constraint families at a symbolic level.
- PPO clipped objective and action-mask concept.
- One-day 15-minute resolution, 10-workload headline scenario, and subtask counts for workload-count experiments.
- Main tabulated cost/emission results.
- Most headline PPO hyperparameters listed above.

### Partially specified

- Workload DAGs: example structures are shown, but exact 10-workload graphs and attributes are absent.
- Regional prices/CEFs: plotted and range-described, but raw values/provenance are absent.
- Power/bandwidth behavior: plotted, but capacities and subtask demands are absent.
- Carbon market: formula is given, but price, quota, and market limits are absent.
- PPO network: hidden size/head count are given, but architecture and tensorization are absent.
- A2C comparison: four aggregate metrics are given, but implementation/tuning/seed protocol is absent.
- Thirty-day sensitivity: scope is stated, but generating process and raw trials are absent.

### Missing

- Raw data and code.
- Exact workload arrival, deadline, duration, demand, migration-time, and precedence records.
- Exact electricity-price, renewable, and CEF time series.
- DC capacity, minimum load, bandwidth, and network parameters.
- Carbon price and quota.
- Reward weights/magnitudes and full PPO/A2C configurations.
- Software/hardware environment.
- Random seeds and number of runs.
- Train/validation/test split and model-selection procedure.
- Runtime/latency/memory/parameter measurements.
- Statistical tests, ablations, and raw result files.

## 11. Implementation-assumption register for a faithful reproduction

These assumptions are necessary; none should be represented as published facts.

| ID | Required assumption | Conservative reproduction choice |
|---|---|---|
| A1 | Price and CEF traces | Digitize Figs. 5-6 if reliable; otherwise generate documented deterministic traces matching stated ranges/patterns and label them synthetic reconstructions. |
| A2 | Workload instances | Generate a fixed manifest of 10 workloads/50 subtasks with separate pipeline, fork-join/tree, and shared-predecessor DAGs; exclude actual cycles unless iteration semantics are explicitly modeled. |
| A3 | Resource parameters | Choose DC power/bandwidth capacities and subtask demands before experiments, publish them, and freeze them in configuration. |
| A4 | Migration | Model source-destination transfer time and bandwidth explicitly; if matching the paper narrowly, use fixed per-destination migration times and zero migration energy/cost, clearly marked as an assumption. |
| A5 | Carbon market | Choose a fixed quota and carbon price calibrated only from the published accounting identity, not from proposed-model test performance. |
| A6 | Reward | Normalize objective components and document all weights/bonuses/penalties; also evaluate direct cost to prevent reward-performance mismatch. |
| A7 | PPO architecture | Use a reproducible masked actor-critic with explicit task embeddings/attention, layer counts, activations, initialization, and optimizer. |
| A8 | Validation | Create scenario-level train/validation/test splits before candidate design; do not emulate the paper's apparent same-distribution evaluation. |
| A9 | S1 ambiguity | Implement and report both nearest/preselected and seeded-random allocation if they differ materially. |
| A10 | Reproduction target | Treat tabulated S1-S3 values as published references, not fitting targets; report local values separately. |

## 12. Audit verdict

The paper is **conceptually reproducible but not numerically reproducible from the publication alone**. A defensible reproduction can implement the stated model family and match the qualitative protocol, but exact reported values cannot be regenerated without author data/code or substantial documented reconstruction. The strongest independent paper should therefore (i) retain a transparent source-method implementation as a mandatory benchmark, (ii) publish a fixed simulator and scenario manifests, (iii) use held-out scenario/topology splits, and (iv) target constraint-safe, dependency-aware, computationally efficient scheduling rather than merely replacing PPO with another generic RL algorithm.
