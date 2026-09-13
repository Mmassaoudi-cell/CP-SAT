# Literature and Novelty Verification

Conducted before finalizing the manuscript's contribution claims, per the master protocol's requirement to search 2024-2026 literature and avoid unsupported "first-ever" claims.

## Searches performed

1. "distill MILP solver imitation learning combinatorial scheduling policy 2025 2026"
2. "deep reinforcement learning data center workload migration scheduling carbon electricity market 2025"

## Relevant related work found

- **Imitation learning for MILP branching policies** (multiple 2025-2026 papers, e.g., AAAI/arXiv work on PPO- and imitation-based branching policies, and general ML4CO surveys via the `awesome-ml4co` community list). This is a related but **distinct** target: these methods imitate a solver's internal *branching decisions* to speed up the solver itself, whereas our approach imitates a solver's *final solution* (a completed near-optimal schedule) to train a standalone, solver-free deployment-time policy. We did not find prior work applying full-solution CP-SAT distillation specifically to the DC migration/scheduling-under-electricity-and-carbon-markets problem.
- **"On the Role of DAG topology in Energy-Aware Cloud Scheduling: A GNN-Based Deep Reinforcement Learning Approach"** (arXiv, 2026) — directly relevant prior art for the topology-aware (GNN/GAT) direction explored in our GAT-PPO/TAGS candidates. This confirms graph-topology-aware cloud scheduling is an active, non-novel-in-isolation research direction; our contribution does not claim topology-awareness itself as novel, and our own ablation (`ANALYSIS/results/ablation_report.md`) shows topology-awareness is not the load-bearing mechanism in our final selected model regardless.
- **Hierarchical multi-agent RL for carbon-aware AI data centers in power distribution systems** (arXiv, 2026) and multiple ScienceDirect/Springer 2025-2026 papers on carbon- and market-aware DC/HPC scheduling (mixture-of-experts multi-critic RL, uncertainty-aware RL for joint operations+bidding, DDPG-based multi-objective cost/carbon scheduling). These confirm carbon- and electricity-market-aware DRL scheduling is an active, populated research area; our contribution is not "the first" to jointly consider cost and carbon in DC scheduling.
- No prior work was found that specifically reproduces or extends *this* source paper (Wang et al., PPO-based multi-DC migration/scheduling in electricity and carbon markets, IEEE TIA 2026), consistent with its recent (2026) publication.

## Resulting novelty framing for the manuscript

We do **not** claim: topology-aware scheduling is new; carbon-aware DC scheduling is new; or that PPO for DC scheduling is new — all are established, active research directions with multiple 2025-2026 papers.

We **do** claim, framed narrowly and defensibly: to our knowledge, existing DC/cloud migration-scheduling studies (including the source paper and the related work identified above) train their RL policies entirely from scratch via on-policy interaction, and none evaluate whether a much cheaper alternative — distilling a classical near-optimal solver's (CP-SAT) full solutions into a deployable neural scheduler via supervised imitation, optionally followed by a light RL fine-tune — can match or approach from-scratch RL performance at a fraction of the training cost, for this class of dependency-aware, multi-market DC scheduling problem. Our reproduction-stage finding that the source paper's own from-scratch PPO ties a trivial non-learned heuristic (`REPRODUCTION_REPORT.md`, `SOURCE_WEAKNESS_ANALYSIS.md` weakness #1) is the direct motivation: if expensive on-policy RL is not clearly earning its training cost even in the source paper's own problem class, a solver-distillation alternative is a natural, underexplored question, and our ablation study (`ANALYSIS/results/ablation_report.md`) isolates that the training *methodology* (distillation from a strong teacher), not the network's topology/carbon-Lagrangian components, is what drives the result.

This is a "to our knowledge, existing studies have not jointly addressed..." framing rather than a "first-ever" claim, consistent with the master protocol's guidance.

Sources consulted:
- [Learning Branching Policies for MILPs with Proximal Policy Optimization](https://ojs.aaai.org/index.php/AAAI/article/download/39619/43580)
- [Learning Branching Policies for MILPs with Proximal Policy Optimization (arXiv)](https://arxiv.org/html/2511.12986)
- [awesome-ml4co (GitHub)](https://github.com/Thinklab-SJTU/awesome-ml4co)
- [awesome-fm4co (GitHub)](https://github.com/ai4co/awesome-fm4co)
- [Speeding Up Mixed-Integer Programming Solvers with Sparse Learning for Branching](https://arxiv.org/pdf/2604.00094)
- [Imitation Learning in the Deep Learning Era: A Novel Taxonomy and Recent Advances](https://arxiv.org/pdf/2511.03565)
- [On-Policy Distillation - Thinking Machines Lab](https://thinkingmachines.ai/blog/on-policy-distillation/)
- [corl: Reinforcement Learning of MILP Policies Solved via Branch-and-Bound](https://arxiv.org/html/2512.11169)
- [Toward Sustainable and Cost-Efficient HPC Systems: A DRL Job Scheduling Approach](https://link.springer.com/chapter/10.1007/978-981-95-4960-3_33)
- [Mixture-of-experts based multi-critic DRL for sustainable management of data center microgrids](https://www.sciencedirect.com/science/article/pii/S0306261926002138)
- [Leveraging DRL within Optimal Renewable Energy Strategies for Sustainable AI Data Centers](https://pubs.acs.org/doi/10.1021/acs.est.5c09990)
- [Uncertainty-aware RL for joint optimization of data center operations and electricity market bidding](https://www.sciencedirect.com/science/article/abs/pii/S0306261926008603)
- [A survey of DRL techniques for Energy-efficient green cloud computing](https://link.springer.com/article/10.1007/s10586-025-05727-w)
- [GreenDRL: managing green datacenters using deep RL](https://dl.acm.org/doi/10.1145/3542929.3563501)
- [Hierarchical Multi-Agent RL for Carbon-Aware AI Data Centers in Power Distribution Systems](https://arxiv.org/pdf/2607.03324)
- [On the Role of DAG topology in Energy-Aware Cloud Scheduling: A GNN-Based DRL Approach](https://arxiv.org/pdf/2604.09202)
