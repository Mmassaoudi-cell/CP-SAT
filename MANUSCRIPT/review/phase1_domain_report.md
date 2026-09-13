## Domain Review Report (Peer Reviewer 2)

### Reviewer Identity
Senior researcher in data-center energy management and carbon-aware computing, with publications on electricity-market-integrated workload scheduling and grid-interactive computing; deep familiarity with the DC-scheduling DRL literature (2023–2026).

### Overall Recommendation
Major Revision

### Confidence Score
4

### Summary Assessment
This is a candid, methodologically disciplined reproduction-and-methodology study whose central empirical claim — that a from-scratch PPO scheduler for dependency-aware, multi-market DC migration ties a zero-training greedy heuristic (p.2, §III-B: $10,870.51 vs. $10,879.51, 0.08%) — is a genuinely useful diagnostic for the DC-scheduling DRL literature, and the paper's transparency about its own bug (PID-Lagrangian saturation, p.4 §V-B) and mixed 7-win/7-loss result (Table II) is commendable domain citizenship rarely seen in this subfield. However, as a *domain* contribution to DC-scheduling and carbon-aware computing specifically, the paper is narrower than its framing suggests: it explicitly disclaims novelty for carbon-aware scheduling, topology-aware scheduling, and PPO-for-DC-scheduling (p.2, §II "Positioning"), leaving "solver-distillation applied to this specific problem class" as the residual claim. Two issues weigh most heavily on domain validity: the entire empirical edifice sits on a synthetic environment reconstructed from a publication that provides no author data (p.2, §III-A), so no downstream percentage gap can be validated against real DC market dynamics; and the paper's own recommended model (DFPS) fails the primary domain-relevant axis of the very problem it reproduces — carbon-market compliance (p.5, §VI-F) — while the compliant variant (TAGS) loses on the headline metric. The literature review, while structurally reasonable, is thin on named, verifiable citations and omits at least one directly load-bearing methodological source.

### Strengths
1. **Empirically grounded critique of an existing DC-scheduling paradigm**: The reproduction-stage finding (p.2, §III-B, 0.08% difference) is a specific, falsifiable, field-relevant result — the kind of negative finding the DC-scheduling DRL literature under-reports.
2. **Honest chronology of a real Lagrangian-tuning failure**: Reporting the PID-Lagrangian saturation bug and its cost consequence in full (p.4 §V-B) rather than silently correcting it is a domain-appropriate standard of evidence for constrained-RL carbon control.
3. **Explicit, non-overclaiming positioning statement**: Section II's "Positioning" paragraph (p.2) reduces overclaiming risk common in this subfield.
4. **Multi-axis efficiency accounting appropriate to DC operations**: Table IV's separation of one-time solver cost from per-seed marginal cost reflects a realistic operational framing.

### Weaknesses
1. **The synthetic-environment dependency undermines the domain validity of every headline percentage.** Section III-A states plainly that no author data is available and that the reconstruction "is not expected to reproduce the paper's exact reported dollar values." Every subsequent number — the 0.08% parity, the 3.2%/3.8% DFPS gaps, the win margins, the carbon-quota shortfall — is generated inside a self-authored scenario generator whose price/CEF volatility, capacity tightness, and deadline-slack distributions match only a qualitative protocol description. **Suggested improvement**: calibrate/validate the generator's price and CEF volatility statistics against public real-world traces (e.g., day-ahead LMP series from PJM/CAISO/Nord Pool, and marginal/average CEF data from WattTime or ElectricityMaps). **Recommended references**: Radovanović et al., "Carbon-Aware Computing for Datacenters," IEEE Trans. Power Systems, 2023; Wiesner et al., "Let's Wait Awhile: How Temporal Workload Shifting Can Reduce Carbon Emissions in the Cloud," ACM Middleware 2021.
2. **The paper's own recommended model fails compliance on the domain metric that motivates the source problem.** Section VI-F reports DFPS at 10,851 kg against a 9,500 kg quota (14% over), while the pre-bug-fix TAGS run *did* reach compliance (9,277 kg) at "an indefensible cost" ($12,960). No systematically-tuned compliance-oriented variant (PID gain sweep, longer fine-tune with decaying cost-weight schedule) is shown before declaring the trade-off "unresolved." **Recommended reference**: Stooke, Achiam, and Abbeel, "Responsive Safety in Reinforcement Learning by PID Lagrangian Methods," ICML 2020 — the direct methodological origin of the "PID-Lagrangian" mechanism used in §IV-C, currently uncited.
3. **Literature review citation practice is substandard for the venue and leaves at least one directly relevant, load-bearing source uncited.** References [2]–[11] (p.5) carry no author names — only title/venue/year — while reference [1] is fully attributed. This asymmetry is not acceptable citation practice for IEEE TIA.
4. **The domain-specific novelty claim narrows under scrutiny of the broader learning-for-combinatorial-optimization literature.** The claim that full-solution distillation "has not been applied to dependency-aware, multi-market DC migration scheduling specifically" is true only at the narrowest level of specificity; the underlying imitate-then-fine-tune recipe is established more broadly in neural combinatorial optimization (e.g., job-shop/vehicle-routing policy learning). **Recommended reference (category)**: Zhang, Song, Cao, Zhang, Tan, Chi, "Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning," NeurIPS 2020 — flagged as a category recommendation for the authors to verify.

### Detailed Comments

#### Literature Review
- **Coverage**: The related-work taxonomy is structurally sound but omits foundational empirical carbon-aware-computing papers and the more skeptical strand of the carbon-shifting literature questioning real-world benefit of temporal/spatial shifting — directly analogous in spirit to this paper's own greedy-ties-PPO finding.
- **Integration quality**: The related-work section explicitly relates cited clusters back to the paper's own ablation findings — a genuine critical-synthesis move above the norm for this literature.
- **Research gap argument**: Reasonably persuasive at the narrow level claimed, overstated slightly by absence of the broader ML4CO precedent.

#### Theoretical Framework
- **Appropriateness**: The constrained-MDP/Lagrangian-dual framing for carbon compliance is appropriate and current.
- **Application depth**: Applied with real depth — the mechanism is debugged in public, evidence of genuine engagement with its practical failure modes.
- **Alternative frameworks**: No discussion of why a hard-constraint/safe-RL formulation or a market-based multi-agent bidding formulation was not considered, given the source problem's explicit "shared national carbon market" framing.

#### Academic Argument Quality
- **Factual accuracy**: No domain factual errors identified.
- **Argument logic**: Internally consistent; the chain from weakness-diagnosis to mechanism to ablation-based attribution is well constructed.
- **Terminology precision**: "Carbon market" is used loosely — the modeled mechanism (fixed price, fixed quota, PID-controlled penalty) is closer to an internal shadow-price compliance budget than to a true market with price discovery.

#### Contribution to the Field
- **Incremental contribution**: Genuine but narrow: (1) a documented negative reproduction finding, and (2) an efficiency-oriented training-methodology substitution, with honest attribution of the gain to methodology rather than architecture.
- **Positioning**: Well-positioned relative to immediate related work; less well-positioned relative to the broader ML4CO imitation-learning literature.
- **Overclaiming**: Low overall, but the abstract could be read as a deployment recommendation when the recommended model does not meet the compliance bar the source problem is built around.

#### Missing Key References
- Stooke, Achiam, Abbeel, "Responsive Safety in Reinforcement Learning by PID Lagrangian Methods," ICML 2020 — high confidence, directly on-point.
- Radovanović et al., "Carbon-Aware Computing for Datacenters," IEEE Trans. Power Systems, 2023 — high confidence.
- Wiesner et al., "Let's Wait Awhile...," ACM Middleware 2021 — high confidence.
- A 2023–2024 paper questioning the real-world magnitude of carbon-aware shifting benefits — category recommendation, authors should verify.
- Zhang et al., "Learning to Dispatch for Job Shop Scheduling via DRL," NeurIPS 2020 — category recommendation.

### Questions for Authors
1. Has the synthetic scenario generator's price/CEF volatility been validated against any real regional market or public CEF dataset?
2. Was any systematic tuning attempted to find a compliant-but-not-indefensible point on the carbon/cost Pareto frontier before declaring the trade-off "unresolved"?
3. Does the environment model any migration-specific cost/penalty (data-transfer volume, live-migration downtime) distinct from destination-DC execution cost, given "migration" is central to the title?
4. Why are references [2]–[11] listed without author names while [1] is fully attributed?

### Minor Issues
- Citation formatting inconsistency between [1] and [2]–[11] should be corrected regardless of content concerns.
- "Carbon market" terminology should be clarified as a fixed-price, fixed-quota compliance budget rather than a true trading market.
- Table II's very large Cohen's d values could be briefly glossed for domain readers so magnitude is not misread as large practical effect.
