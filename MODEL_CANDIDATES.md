# Model Candidates

Five candidates, each targeting specific ranked weaknesses from `SOURCE_WEAKNESS_ANALYSIS.md`. Candidates 1-3 isolate one fix apiece; Candidate 4 (TAGS) combines all three into the primary hybrid; Candidate 5 tests whether the fixes even require on-policy RL at all. This structure is deliberate: candidates 1-3 double as leave-one-component-out ablations for TAGS once TAGS is selected (Section 19 of the master protocol), so no extra ablation implementations are needed later.

All candidates share the source method's environment contract (`SchedulingEnvironment`: masked feasible (DC, slot) actions, same constraints, same cost/carbon accounting) so comparisons isolate the policy/representation, not the simulator.

---

## Candidate 1 — GAT-PPO (Graph-Topology Backbone)

**Problem it fixes:** Weakness #3 — the source policy's only dependency signal is a scalar `len(predecessors)/3.0`; the paper's headline "dependency-aware" claim is not actually consumed by the network.

**Architecture:** Replace the scalar predecessor count with a small graph-attention encoder (2-layer GATv2, 4 heads, hidden 64) run once per scenario over each workload's subtask DAG. Node features: duration, power, bandwidth, remaining descendants, critical-path remaining work (sum of duration along the longest remaining path from this node to a sink). Node embeddings for a task and its immediate predecessors/successors are pooled (mean + max) into a `topology_embedding` (dim 32) that augments the existing 12-dim task feature vector. Everything else (action encoder, attention over ready-slot actions, PPO objective) is unchanged from the source reproduction.

**Core formulation:** For workload DAG $G_j=(V_j,E_j)$, node embedding update $h_v^{(l+1)} = \text{GATv2}(h_v^{(l)}, \{h_u^{(l)}: u \in \mathcal{N}(v)\})$; task feature becomes $\tilde{s}_{j,k} = [s_{j,k} \,\|\, \text{pool}(h_k^{(2)}, h_{\phi_k}^{(2)}, h_{\text{succ}(k)}^{(2)})]$.

**Why it should outperform the source method:** Gives the policy visibility into sibling/critical-path position it structurally cannot see today; directly tests whether weakness #3 alone explains the tie with the greedy heuristic (weakness #1).

**Expected efficiency benefit:** None by itself (small added GAT cost); it is a representation-quality candidate, not an efficiency candidate.

**Expected weakness:** Still myopic (reward unchanged) and still one autoregressive decision per subtask, so it may still tie the greedy heuristic if myopia (#2) rather than topology-blindness (#3) is the binding constraint.

**Required ablation:** GAT backbone vs. scalar-count backbone, everything else held fixed (this candidate *is* that ablation, run against Candidate 4).

**Estimated computational cost:** +15-20% training time vs. source PPO (small GAT forward/backward per scenario reset); comparable inference latency.

---

## Candidate 2 — HCBS-PPO (Horizon-Cached Batched Frontier Scorer)

**Problem it fixes:** Weakness #5 (unbatched, uncached, recomputed-every-step action features) and, as a side effect, weakness #2 (myopia), by giving the scorer a precomputed view of the *entire* horizon rather than only the task's own feasible window.

**Architecture:** (a) **Caching:** compute a per-DC horizon embedding once per scenario reset via a lightweight temporal convolution (3-layer causal Conv1D, kernel 5, channels 32) over the 96-slot `[price, cef]` curves per DC; this replaces the source's per-decision `prices[dc,start:end].mean()` recomputation with a single cached lookup per (dc, slot) pair reused across all subtasks. (b) **Batched non-autoregressive frontier decisions:** instead of one environment step per subtask in a fixed task-list order, at each decision epoch the environment exposes the *entire currently-ready frontier* (all subtasks whose predecessors are complete and that have arrived) and the policy scores all (frontier task × feasible DC × feasible slot) triples in one batched forward pass using the shared attention scorer; the highest-scoring feasible triple is committed, ties broken by earliest deadline. This does not change the action semantics (still one (DC, slot) commit per step) but removes the fixed, arbitrary task-processing order and amortizes horizon-feature computation.

**Core formulation:** Cached horizon embedding $H_{dc} = \text{TCN}([\text{price}_{dc,0:96}, \text{cef}_{dc,0:96}])\in\mathbb{R}^{96\times32}$, computed once; per-decision action feature becomes $a_{j,k,dc,t} = [H_{dc,t} \,\|\, \text{remaining\_power}_{dc,t} \,\|\, \text{remaining\_bw}_{dc,t}]$ (lookup, not recomputation). Frontier scoring: $\text{logit}(j,k,dc,t) = f_\theta(\tilde s_{j,k}, a_{j,k,dc,t})$ over the batched ready set $\mathcal{R}_\tau$ at decision epoch $\tau$.

**Why it should outperform the source method:** Removing the fixed task-order autoregression lets the policy trade off *across* currently-competing ready subtasks (e.g., defer a flexible one in favor of an urgent one) instead of committing to whichever subtask happens to be next in the list — this is a real behavioral change, not just a speed-up.

**Expected efficiency benefit:** Primary target. Horizon-feature caching removes the dominant cost identified in weakness #5 (per-step `mean()` reductions over up to 288 action slots); batched frontier scoring reduces the number of sequential network calls per episode from 50 (one per subtask) to the number of decision epochs (≤50, typically fewer once multiple ready subtasks are batched together).

**Expected weakness:** No explicit dependency-topology embedding (inherits weakness #3) and no explicit carbon control mechanism (inherits weakness #6).

**Required ablation:** Cached+batched vs. source's uncached+autoregressive loop, network capacity held fixed.

**Estimated computational cost:** Lower wall-clock than source PPO per training episode despite the added TCN (fewer, larger batched forward passes typically amortize better on GPU than many small ones); target ≥30% training-time reduction, to be measured empirically, not assumed.

---

## Candidate 3 — LC-PPO (Lagrangian Carbon-Constrained PPO)

**Problem it fixes:** Weakness #6 — carbon is a reward pass-through, not a controlled quantity; S3/PPO does not reduce emissions vs. S1 in either the paper or our reproduction.

**Architecture:** Same network as the source reproduction (masked attention actor-critic). Training objective changes from a fixed weighted-sum reward to a **PID-Lagrangian constrained formulation**: the task reward is pure negative electricity cost; carbon is enforced as an explicit per-episode budget constraint via a dual variable $\lambda\geq0$ updated by a PID controller on the emissions-budget violation, added as a penalty $-\lambda \cdot (\text{emissions} - \text{quota})^+$ to the per-step reward. $\lambda$ adapts online instead of being a fixed hand-tuned weight (unlike the source paper's unpublished $w_1,\dots,w_5$).

**Core formulation:** $r_t = -\text{cost}_t - \lambda_t \cdot [\text{cef}_{dc,t}\cdot\text{energy}_t]$, with $\lambda_{t+1} = \text{clip}(\lambda_t + K_P e_t + K_I \textstyle\sum e_t, 0, \lambda_{\max})$ where $e_t = \text{cumulative emissions so far} - \text{prorated quota}$.

**Why it should outperform the source method:** Directly targets the demonstrated cost-carbon non-improvement; a constraint-driven dual price should push the policy toward emissions parity or reduction relative to S1, something no candidate without this mechanism can promise by construction.

**Expected efficiency benefit:** None (same network, same decision loop); this is a training-objective candidate, not an efficiency candidate.

**Expected weakness:** Still myopic and topology-blind (inherits #2, #3, #5); PID-Lagrangian dual can oscillate or be slow to adapt within only 3,000 training episodes, which is itself a measurable risk to report honestly if it occurs.

**Required ablation:** Fixed-weight reward vs. PID-Lagrangian carbon term, network and decision loop held fixed.

**Estimated computational cost:** Negligible overhead (~1-2% — one scalar PID update per step).

---

## Candidate 4 — TAGS: Topology-Aware Anticipatory Graph Scheduler (primary hybrid)

**Problem it fixes:** Combines Candidates 1-3 into one method: dependency-blindness (#3), myopic/unbatched decisions (#2, #5), and carbon pass-through (#6) simultaneously, because the weakness analysis shows these are not independent — a policy that cannot see topology or the horizon cannot make an anticipatory carbon-cost trade-off no matter how its reward is shaped.

**Architecture (backbone + innovation + adaptation + head, per the master protocol's 2-4-mechanism principle):**
1. **Backbone:** GATv2 topology encoder (Candidate 1) producing per-subtask topology embeddings.
2. **Task-specific innovation:** horizon-cached batched frontier scoring (Candidate 2) — the anticipatory, efficiency-bearing mechanism.
3. **Efficient adaptation/optimization:** PID-Lagrangian carbon-budget constraint (Candidate 3) replacing fixed reward weights.
4. **Compact prediction head:** a single shared bilinear scorer (unchanged in parameter count from the source's actor-query mechanism) reused across the batched frontier — no growth in head complexity despite the richer inputs.

**Core mathematical formulation:** $\text{logit}(j,k,dc,t) = \frac{1}{\sqrt d}\, q(\tilde s_{j,k})^\top a(H_{dc,t}, \text{remaining}_{dc,t}) + b(a)$, with $\tilde s_{j,k}$ from the GAT backbone and $H_{dc,t}$ the cached TCN horizon embedding; trained with clipped PPO on $r_t = -\text{cost}_t - \lambda_t[\cdot]^+$ over the batched-frontier decision process.

**Why it should outperform the source method:** It is the only candidate that can simultaneously (a) beat the greedy heuristic by exploiting topology + horizon information no heuristic uses, and (b) demonstrably control emissions rather than merely inheriting whatever emissions a cost-only greedy policy produces.

**Expected efficiency benefit:** Inherits Candidate 2's caching/batching benefit; GAT and PID overhead are small relative to the removed per-step recomputation, so TAGS is expected to be *both* more accurate and cheaper to train/run than the source PPO — to be confirmed empirically, not assumed, per the efficiency-analysis requirement.

**Expected weakness:** Highest implementation complexity of the five candidates; most hyperparameters (GAT layers/heads, TCN channels, PID gains) to tune, raising overfitting-to-validation risk if tuning budget is not controlled (mitigated by the validation-only screening protocol and a capped Optuna budget in Stage 3).

**Required ablation:** Full TAGS vs. TAGS-minus-GAT (=Candidate 2+3 essentially), TAGS-minus-caching/batching (=Candidate 1+3), TAGS-minus-Lagrangian (=Candidate 1+2), each parameter-matched where feasible.

**Estimated computational cost:** Expected net reduction in training wall-clock vs. source PPO (batching dominates over added GAT/TCN/PID cost) but higher parameter count (~+40-60% vs. source's ~330K, exact count to be reported from the implementation, not estimated).

---

## Candidate 5 — Distilled Frontier-Priority Student (non-RL amortized baseline)

**Problem it fixes:** Weakness #1/#4 directly — since PPO ties a hand-written greedy heuristic in the source reproduction, this candidate asks whether a much cheaper *imitation-learned* scorer can match TAGS/PPO-family candidates without any on-policy RL training loop at all, testing whether the RL machinery itself is the right computational investment (efficiency mandate, Section 8: "large teacher → distilled student", "repeated retraining → adapter/hypernetwork").

**Architecture:** Same batched-frontier + topology-embedding input representation as TAGS (reusing Candidates 1-2's featurization for a fair comparison), but the scorer is trained by **supervised imitation** against a computed teacher: a short-horizon (8-slot) exhaustive/branch-and-bound lookahead search run offline on training scenarios only, producing a "which of the currently-ready (task, DC, slot) triples has the lowest 8-slot-lookahead marginal cost" label at each decision epoch. The student network (same architecture as TAGS's head) is trained with a cross-entropy loss against these labels, then optionally fine-tuned with a small number of PPO updates (an "adapter" phase) directly on task reward, capped at 10% of TAGS's training-episode budget.

**Core mathematical formulation:** $\mathcal{L} = \mathbb{E}_{\tau}\left[-\log \pi_\theta(a^\star_\tau \mid \mathcal R_\tau)\right] + \beta\,\mathcal{L}_{\text{PPO-finetune}}$, where $a^\star_\tau$ is the lookahead-teacher's chosen triple and $\beta$ activates only in the short fine-tuning phase.

**Why it should outperform the source method:** If it matches or beats source PPO at a fraction of the training cost, it directly demonstrates that the source paper's costly full on-policy RL loop (3,000 episodes) is not necessary for this problem class — a strong, honest efficiency finding either way (a genuine miss is also reportable and useful, per the negative-results-retained requirement).

**Expected efficiency benefit:** Primary target — expected order-of-magnitude reduction in training compute (no environment rollouts needed for the imitation phase beyond generating teacher labels once).

**Expected weakness:** Bounded by teacher quality (an 8-slot lookahead is itself a heuristic, not globally optimal); may underperform TAGS on scenarios where longer-horizon anticipation matters; offline label generation adds a one-time preprocessing cost that must be reported honestly as part of total pipeline cost, not hidden.

**Required ablation:** Imitation-only vs. imitation+PPO-finetune, to isolate whether any RL fine-tuning is adding value at all.

**Estimated computational cost:** Teacher-label generation: O(ready-set size × 8-slot lookahead) per decision epoch on training scenarios only (one-time). Student training: comparable to a handful of supervised epochs over the collected label set — expected far below 3,000 PPO episodes.

---

## Ranking (initial, pre-screening)

| Rank | Candidate | Expected predictive performance | Novelty | Efficiency | Data compatibility | Feasibility | Transactions-level contribution |
|---|---|---|---|---|---|---|---|
| 1 | TAGS (4) | Highest (combines all fixes) | High (novel combination + PID-Lagrangian scheduling head) | Likely net-positive | Full | Moderate-high effort | Strong — primary candidate |
| 2 | HCBS-PPO (2) | Medium-high | Medium | High (primary efficiency candidate) | Full | Moderate | Strong efficiency contribution alone |
| 3 | Distilled Student (5) | Uncertain (bounded by teacher) | Medium-high (distillation angle) | Very high | Full | Moderate | Strong if it holds up — efficiency headline |
| 4 | LC-PPO (3) | Medium (carbon-specific) | Medium | Neutral | Full | Low effort | Useful ablation / secondary finding |
| 5 | GAT-PPO (1) | Low-medium alone | Low-medium | Neutral | Full | Low effort | Useful ablation only |

This ranking is a prior, not a decision — Stage 1-3 validation-only screening (`MODEL_SELECTION_REPORT.md`) determines the actual outcome.
