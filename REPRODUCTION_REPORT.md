# Reproduction Report — Source Method (PPO Migration/Scheduling, 3-DC Electricity+Carbon Markets)

## Status

**Conceptually reproduced, not numerically reproducible.** As established in `SOURCE_PAPER_AUDIT.md` §10-11, the source paper releases no code, no raw workload/price/CEF data, no reward-weight values, and no full network/training specification. `SOURCE_METHOD_REPRODUCTION/` therefore implements the *stated model family* (3 DCs, 96 quarter-hour slots, 10 workloads / 50 dependency-constrained subtasks, masked PPO actor-critic with 128 hidden units / 8 attention heads, S1/S2/S3-style scenario comparison) on a **frozen synthetic scenario generator** (`SOURCE_METHOD_REPRODUCTION/src/scenario.py`, seeded, documented in `DATA_AUDIT.md`). Absolute dollar/kg values are therefore **not expected to match** the paper; what is comparable is the qualitative behavior the paper claims (S3 spatiotemporal PPO scheduling beats S1 zero-delay and S2 single-DC temporal shifting).

Training: 3,000 episodes, 16 parallel envs, PPO with masked attention actor-critic, config frozen in `SOURCE_METHOD_REPRODUCTION/config/source_reproduction.yaml` (matches all *fully specified* hyperparameters from the audit: lr 1e-4, γ=0.99, clip 0.2, GAE λ=0.9, batch 128, hidden 128, 8 heads, grad-clip 0.5, entropy coef 0.2). Checkpoint: `results/ppo_attention_3000_episodes.pt`. Training curve: `results/training_log_3000_episodes.csv` (3,000 rows, one per episode).

Evaluation: single reproduction seed (`3691718`, analogous to the paper's DOI-anchored single case study) plus a 100-scenario validation split (seeds 2000-2099) for statistical stability. Heuristics `s1_zero_delay` and `s2_single_dc_temporal` reproduce the paper's S1/S2 operational logic inside the same simulator; `greedy_spatiotemporal` is an additional non-learned baseline (cheapest feasible slot across all DCs) used purely as a diagnostic, not as one of the paper's own scenarios.

## Published vs. Reproduced

Published values are the paper's Table II-IV headline numbers (main 10-workload case study). Reproduced values are our synthetic-scenario results — **not the same underlying data**, so this table checks *qualitative/directional* agreement, not numerical equality.

### Single-case-study comparison (paper's one scenario vs. our one reproduction seed)

| Metric | Scenario | Published | Reproduced (synthetic) | Difference |
|---|---|---:|---:|---:|
| System cost ($) | S1 zero-delay | 8,467.5 | 11,738.6 | n/a (different data) |
| System cost ($) | S2 single-DC temporal | 8,224.4 | 11,508.1 | n/a (different data) |
| System cost ($) | S3 / PPO spatiotemporal | 8,105.5 | 10,689.0 | n/a (different data) |
| S3 vs S1 cost reduction (%) | — | 4.28 (computed from table; paper states "4.1%") | 8.94 | +4.66 pp |
| Electricity cost ($) | S1 | 8,739.0 | 12,051.3 | n/a (different data) |
| Electricity cost ($) | S3/PPO | 8,376.9 | 10,959.4 | n/a (different data) |
| Carbon emissions (kg) | S1 | 10,526.9 | 10,112.8 | n/a (different data) |
| Carbon emissions (kg) | S3/PPO | 10,527.8 | 10,535.7 | n/a (different data) |
| S3 emissions vs. S1 | — | **+0.9 kg (higher)** | **+422.9 kg (higher)** | Same sign: PPO does **not** reduce emissions vs. S1 in either the paper or our reproduction |

### Validation-split comparison (100 seeded scenarios, mean ± std)

| Method | System cost ($) | Electricity cost ($) | Carbon emissions (kg) | Mean start delay (slots) | Remote-subtask fraction |
|---|---:|---:|---:|---:|---:|
| S1 zero-delay | 12,148.97 ± 406.25 | 12,420.56 ± 401.33 | 10,524.12 ± 458.42 | 3.85 | 0.00 |
| S2 single-DC temporal | 11,717.64 ± 335.32 | 11,999.69 ± 338.71 | 10,419.50 ± 376.54 | 12.90 | 0.00 |
| Greedy spatiotemporal (diagnostic, not in paper) | 10,870.51 ± 194.61 | 11,112.31 ± 172.07 | 10,822.02 ± 279.04 | 13.46 | 0.65 |
| **Source PPO (S3-equivalent)** | **10,879.51 ± 196.58** | 11,124.83 ± 175.67 | 10,786.82 ± 276.16 | 13.78 | 0.66 |

- PPO vs. S1: **10.45%** system-cost reduction (paper: 4.1-4.28%). Direction agrees; magnitude does not, and is not expected to given different underlying data/scale.
- PPO vs. S2: **7.15%** system-cost reduction.
- PPO vs. greedy spatiotemporal heuristic: PPO is **0.08% worse**, i.e., statistically indistinguishable (std ≈ 195-197 on both, difference ≈ 9 on n=100). **This is a reproduction-specific finding, not a claim about the published paper**: a hand-written, non-learned "always pick the cheapest feasible slot across DCs" heuristic matches the trained masked-attention PPO policy in our environment.
- Capacity violations: 0 for all methods (constraint masking works as specified).
- Decision latency (PPO, reproduction seed): mean 14.45 ms / p95 3.11 ms per subtask-decision on GPU — not reported in the source paper (audit §6/§8: no runtime data published).

## Sources of published/reproduced divergence

| Cause | Applies here? | Notes |
|---|---|---|
| Unavailable hyperparameters | Partially | Reward weights w1-w5, optimizer, PPO epochs/rollout length, LR/entropy schedule were not published (audit §4); we fixed reasonable values (documented in config) and did not tune them against any target. |
| Random seeds | Yes | Paper reports no seed; we fix seed 2026 for training and enumerate scenario seeds explicitly. |
| Dataset version/provenance | Yes — dominant cause | No author price/CEF/workload data exists (`DATA_AUDIT.md`); our scenario generator only matches *stated ranges and qualitative patterns*, not actual values. This alone explains most of the $-scale gap. |
| Software version | Unknown | Paper names no library/version (audit §5); we use PyTorch 2.12 dev / CUDA. |
| Hardware | Unknown | Paper reports none; ours is a single CUDA GPU. |
| Sampling/simulator settings | Yes | Migration time, DC capacities, bandwidth caps, carbon price/quota are our documented assumptions (audit Table, IDs A1-A5), not published values. |
| Train/test partitioning | Yes | Paper reports no split at all (audit §6, leakage cannot be ruled out); we use disjoint scenario-seed blocks (train/validation/test, `DATA_SPLIT_MANIFEST.csv`) from the outset. |

## Verdict

The reproduction is **not forced to match** the published numbers, and it does not — by design, since no author data exists. What we can and do confirm:

1. The stated model family trains stably and respects all logical/precedence/capacity constraints (zero capacity violations across 400 validation runs).
2. The qualitative claim "S3 (spatiotemporal PPO) reduces system cost vs. S1 and S2" reproduces directionally (10.45% and 7.15% reductions vs. 4.1-4.28% published).
3. The qualitative caveat we flagged in the audit — S3 does not reduce carbon emissions relative to S1 in the published table — also reproduces in our synthetic environment (PPO emissions ≥ S1 emissions in both the single-seed and 100-seed comparisons).
4. A new, reproduction-only finding not discoverable from the paper alone: **the trained PPO policy is statistically indistinguishable from a simple greedy nearest-cheapest-feasible-slot heuristic** on held-out validation scenarios. This directly motivates the weakness analysis in `SOURCE_WEAKNESS_ANALYSIS.md` and the design goals for the proposed method (a method must beat a strong non-learned baseline by a real, statistically defensible margin, not just beat a weak zero-delay strategy).

No redesign of the source method was performed to force numerical agreement, consistent with the task protocol.
