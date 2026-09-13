# Source-Method Weakness Analysis

Basis: `SOURCE_PAPER_AUDIT.md` (what the paper claims/omits) + `REPRODUCTION_REPORT.md` (what our faithful reconstruction actually does). Weaknesses are ranked by expected impact on a follow-on method's ability to win on validation macro-metrics and survive scrutiny as an IEEE Transactions contribution.

## Ranked weaknesses

### 1. [HIGH] Learned policy does not beat a trivial non-learned heuristic
Reproduction finding: `source_ppo_attention` (system cost 10,879.51 ± 196.58 over 100 validation scenarios) is **statistically indistinguishable** from `greedy_spatiotemporal` (10,870.51 ± 194.61) — a one-line "assign to the cheapest feasible (DC, slot) by immediate marginal cost" rule with no learning, no memory, no training cost. PPO is nominally 0.08% *worse*. The paper's own S1/S2 baselines are weak by construction (no spatial migration, no temporal shift), so beating them (paper: 4.1-4.28%; our reproduction: 10.45%/7.15%) is not evidence that reinforcement learning is earning its training cost. This is the single most decision-relevant finding: **any new method must be benchmarked against a strong non-learned heuristic**, not only against S1/S2-style weak baselines, or its contribution is unsupported.

### 2. [HIGH] Single-step greedy reward shaping induces myopic, non-anticipatory decisions
The environment's reward is the negative *immediate* marginal cost of the just-scheduled subtask (`environment.py` `step()`), discounted only through PPO's γ and GAE — there is no explicit lookahead over future price/CEF troughs, no explicit representation of opportunity cost from consuming capacity now, and no coordination mechanism across the sequential per-subtask decisions besides the shared value function. This architecturally caps the ceiling above a greedy immediate-cost heuristic: a policy trained to approximate a myopic reward signal converges toward myopic behavior, which is consistent with finding #1. The paper never states a non-myopic reward design either (audit §3, item 8-9: reward weights/magnitudes for completion/execution bonuses are unpublished), so this limitation is inherited rather than something we introduced.

### 3. [HIGH] No representation of dependency structure beyond a scalar predecessor count
The paper's headline contribution is fine-grained subtask/DAG modeling with explicit logical constraints (audit §1, contribution 2). But the actual state fed to the policy per the reported architecture is a flat per-task feature vector plus per-action features (paper: "task state" = arrivals/deadlines/durations/... ; our reproduction's `task_features` includes only `len(predecessors)/3.0` as the sole dependency signal, since the paper specifies no explicit graph/sequence encoder for the DAG). Neither the paper nor a literal reading of its architecture description (128 hidden units, 8 attention heads, no stated graph layer) shows the policy actually consuming DAG topology (which tasks are siblings, joint critical-path position, how many descendants still need to run). This is a scientific gap between the claimed contribution ("dependency-aware" scheduling) and the demonstrated mechanism (attention over a same-workload token set of unstated composition — audit §4, "paper does not describe where attention is applied").

### 4. [MEDIUM-HIGH] No proof that PPO — specifically — is required
Only one RL algorithm (PPO) is evaluated against one other (A2C), compared solely via training-reward convergence curves (audit §6), never via task metrics (cost/emissions) on held-out scenarios. Given finding #1, it is not established that *any* form of sequential learned control is needed at all for this problem class, let alone that PPO's on-policy clipped-objective machinery in particular is the right computational investment. A defensible new method must show that its added machinery (whatever it is) earns its keep against both a no-learning baseline and simpler learned baselines.

### 5. [MEDIUM] Per-subtask sequential decision-making creates O(J·K) unrewarded sequential forward passes with no batching in the decision loop
Each of the up to 50 subtasks is a **separate autoregressive policy call with an O(n_dc · horizon) = O(3·96)=288-way masked softmax recomputed from scratch each step** (`environment.observation()` reconstructs all 288 action features, including a fresh `prices[dc, start:end].mean()` reduction, every single step). Measured: mean 14.45 ms / p95 3.11 ms per subtask-decision on a CUDA GPU for a 128-hidden/8-head network — for a full 50-subtask episode this is ~0.7s of pure decision latency for a same-day scheduling problem, dominated by Python-level feature reconstruction, not the neural network. Nothing in the paper reports or addresses this (audit §8: no runtime, latency, memory, or FLOPs numbers at all). This is a concrete efficiency target: a candidate method that precomputes/caches action features once per horizon change, or that scores multiple independent subtasks in one batched forward pass, is a measurable, honest efficiency improvement over the source design — not just the source paper's un-measured claim of "efficiency."

### 6. [MEDIUM] Carbon and cost objectives are not jointly improved — S3/PPO does not reduce emissions
Published Table II-IV: S3 has the *lowest* cost but *slightly higher* emissions than S1 (10,527.8 kg vs. 10,526.9 kg) and than S2 (10,527.8 vs. 10,412.6 kg). Our reproduction shows the same qualitative pattern at larger magnitude (PPO emissions 10,535.7 kg reproduction-seed / 10,786.82 kg validation-mean vs. S1's 10,112.8 kg / 10,524.12 kg — PPO emissions are *higher* than S1's in both cases). Despite the framework's "unified cost-carbon objective" framing (audit contribution/gap analysis, Table I row A3/gap 2), the demonstrated behavior optimizes cost and effectively treats carbon as a pass-through of the (potentially negative, i.e., revenue-generating) `carbon_trading_cost` term rather than a controlled quantity. A method claiming joint cost-carbon improvement needs an explicit mechanism (e.g., constrained optimization, a carbon-shadow-price-aware term, or multi-objective Pareto reporting) and must show emission parity or improvement, not merely note it in passing.

### 7. [MEDIUM] No train/validation/test protocol; leakage cannot be ruled out
Audit §6: the paper reports no data split at all. Whatever hyperparameter/architecture decisions were made appear to have been checked against the same scenario distribution used for headline reporting. This is a methodological weakness independent of the specific architecture, and it is why the present study's `DATA_SPLIT_MANIFEST.csv` and validation-only candidate-screening rule exist.

### 8. [MEDIUM] No statistical protocol
Audit §6/§8: PPO's reported final reward has a ± spread (4.778 ± 0.004 ×10⁵) but no stated seed count, no significance test against A2C, and no test at all against S1/S2/S3 cost differences. A 4.1% headline improvement with no confidence interval or seed count cannot be certified as more than a point estimate. Our reproduction addresses this directly by using 100 validation scenarios per method with reported std, and will use ≥5 seeds with paired tests for the final model.

### 9. [LOW-MEDIUM] Unmodeled operational realities inflate apparent flexibility
No cooling, migration energy/cost, network congestion, renewable curtailment, storage, or market-clearing constraints are modeled (audit §1 "Scientific scope"). This is shared by our reproduction (by design, to isolate the scheduling/migration decision) and is not something the new method is obligated to fully solve, but it bounds how strongly either paper's cost claims generalize to real DC operation. We note it as a limitation to be stated plainly in the new manuscript rather than a gap to close.

### 10. [LOW] Internal inconsistencies reduce confidence in the reported numbers
Audit §3 catalogs index mismatches (Eq. 15-16 `x_{i,k}` vs. `x_{i,j}`), an unschedulable "cyclic DAG" example, an advantage-estimator formula (`Â_t = r_t - V(s_t)`) inconsistent with the stated GAE, and a stated PPO convergence-episode table (Table VI: PPO ~200, A2C ~150) that contradicts the prose claim that PPO converges faster. These do not change our reproduction (which follows the *intended* model, documented via the assumption register, audit §11) but they justify treating exact numeric replication as out of reach and support the audit's "conceptually reproducible, not numerically reproducible" verdict.

## Categorized summary

**Predictive limitations:** #1 (ties trivial heuristic), #6 (no emissions co-improvement), #8 (no significance testing to support the headline claim).

**Architectural limitations:** #3 (dependency structure not actually consumed by the policy despite being the headline contribution), #4 (algorithm choice unjustified), #2 (myopic reward caps achievable policy quality).

**Computational limitations:** #5 (unbatched, uncached per-subtask decision loop; no runtime figures published to compare against).

**Scientific-protocol limitations:** #7 (no train/val/test discipline), #8 (no statistics), #10 (internal inconsistencies), #9 (scope caveats to state explicitly).

## Implication for candidate design (feeds `MODEL_CANDIDATES.md`)

A credible successor method must, at minimum:
- **Beat the greedy spatiotemporal heuristic**, not just S1/S2, by a margin that survives a paired significance test (targets weakness #1, #8).
- **Explicitly encode DAG/dependency structure** (e.g., a lightweight graph or sequence encoder over sibling/successor subtasks) so "dependency-aware" is a demonstrated property, not an assumed one (targets #3).
- **Reduce or amortize the per-decision cost**, e.g., by scoring an entire workload's ready subtasks in one batched pass instead of one autoregressive call per subtask, and/or caching horizon-level price/CEF features instead of recomputing them every step (targets #5).
- **Treat carbon as a controlled objective**, not a pass-through, with an explicit mechanism and reported emissions parity/improvement alongside cost (targets #6).
- **Justify its own extra machinery** with a component-level ablation against a parameter-matched, no-learning, and simpler-learned baseline (targets #4, and the mandatory ablation requirement).
