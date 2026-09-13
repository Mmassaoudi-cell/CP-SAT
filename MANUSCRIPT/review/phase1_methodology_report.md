## Methodology Review Report (Peer Reviewer 1)

### Reviewer Identity
Research scientist specializing in deep reinforcement learning for combinatorial scheduling and learning-to-imitate-solvers (ML4CO), with a statistics/experimental-design background. Regularly reviews for NeurIPS/ICML systems tracks and IEEE Transactions venues on RL-for-systems applications.

### Overall Recommendation
Major Revision

### Confidence Score
4

### Summary Assessment
This is a methodologically unusual and, in several respects, exemplary submission for its genre: a pre-registered, validation-only candidate-search protocol with a frozen test split (§V-A–B), a documented and corrected mid-study bug (the PID-Lagrangian saturation error, §IV-C, §V-B), Holm-corrected paired significance testing with effect sizes for the main 14-benchmark comparison (Table II), and an honest 7-win/7-loss report rather than a cherry-picked one. These are exactly the practices the field under-rewards and should be credited. However, the paper's most novel and rhetorically load-bearing claim — that "the training methodology, not the network's specific hybrid components, is the load-bearing mechanism" (Abstract; Contribution 4; §VI-C) — is supported by a component ablation (Table III, Fig. 5a) that does not meet the statistical standard the paper itself applies everywhere else: no paired test, no p-value, no Cohen's d, no discussion of Type II error risk, and only 5 seeds with no scenario-level pairing. Given that the paper's own §III-B finding shows on-policy PPO needs thousands of episodes to move performance at all, a 300-episode "light" RL fine-tune producing "no measurable change" is at least as consistent with an underpowered/under-budgeted probe as with a genuine null result. Additional concerns include an unaddressed same-team circularity risk (the CP-SAT teacher uses "the identical constraint set as the environment," §IV-D), inconsistent statistical resolution between the n=100-paired main table and the n=5 ablation table, and reliance on several unincluded companion artifacts (reproduction report, model-selection report) for key verifiability details. None of these are fatal, but the central causal claim needs firmer statistical footing before publication.

### Strengths
1. **Pre-registered, contamination-resistant development protocol**: The paper establishes a disjoint 600/100/100 train/validation/test scenario split "before any candidate development and never altered based on results" (§V-A), screens and selects among five candidates using validation data only, and explicitly states "all architecture, hyperparameter, and training-procedure decisions were frozen at this point, before any test-split evaluation" (§V-B, p.4). This is a textbook defense against the optimizer's-bias / overfitting-to-test-set fallacy and should be highlighted as a model for the subfield.
2. **Transparent disclosure of a mid-study bug and its correction**: §IV-C and §V-B document, with a specific mechanism and quantified consequence, that an early PID-Lagrangian carbon controller "was found... to saturate at its maximum penalty from the first training episode onward," driving system cost to $12,960.35, and that this was caught and corrected via training-log inspection *before any test-set use*. This is a rare and valuable degree of process transparency; it substantially reduces (though does not eliminate — see Weakness 2) the risk of undisclosed post-hoc HARKing.
3. **Honest, non-cherry-picked benchmark reporting with multiplicity correction**: Table II reports all 14 pairwise comparisons, including 7 statistically significant losses to the strongest cluster (DQN, MLP-PPO-no-attn, A2C, Source_PPO, HEFT, Greedy, CP-SAT), using Holm-Bonferroni correction across all 14 tests at α=0.05 (§V-D), rather than reporting only favorable comparisons. This directly satisfies the "no selective reporting" and "multiple-comparisons handled" criteria in the Methodological Fallacies Checklist.
4. **A genuinely novel and useful reproduction-stage finding, reported with appropriate epistemic humility**: §III-B's finding that the reproduced PPO policy is statistically indistinguishable from a non-learned greedy heuristic (0.08% difference, p.2) is explicitly caveated ("this reconstruction is not expected to reproduce the paper's exact reported dollar values, since no author data exists; what is comparable is the qualitative behavior of the method," §III-A) — an appropriately scoped claim rather than an overclaimed refutation of [1].
5. **Robustness and out-of-distribution testing beyond the primary metric**: Fig. 4 tests generalization to unseen workload counts (6–9 vs. training's 10) and deadline-slack perturbation, and Fig. 5(b) tests input-noise robustness (price/CEF sensor noise, up to 50% bandwidth-signal dropout), both showing smooth, non-catastrophic degradation. This is good practice not required by the stated protocol and strengthens external-validity claims within the synthetic environment.

### Weaknesses
1. **The headline causal claim (ablation) is not held to the paper's own statistical standard**: Table III and Fig. 5(a) report only means ± SD across 5 seeds for the ablation, with no paired significance test, no p-value, no confidence interval, and no Cohen's d — in direct contrast to Table II's paired t-test + Holm correction + Cohen's d for the main comparison. The prose (§VI-C) nonetheless asserts a strong causal reading ("statistically and practically indistinguishable," "not measurably load-bearing"). With n=5 seeds and reported SDs of ~195, the standard error of each mean is ≈87, giving roughly ±$240 (95% CI half-width, t-multiplier for df=4) around a mean of ~$11,230 — i.e., a true effect of up to roughly 2% could not be statistically distinguished from zero at this sample size, yet 2.12% is exactly the magnitude of DFPS's *significant* win margin over HCBS-PPO in Table II. The paper never discusses this Type II error risk. *Improvement*: report a paired test on the 5 seeds (or bootstrap CI) for each ablation contrast, and explicitly discuss the minimum detectable effect size given n=5, alongside the raw point-estimate argument (which is separately reasonably strong, see Detailed Comments).
2. **The RL fine-tune ablation may be confounded with an underpowered fine-tune budget, using the paper's own evidence**: §III-B establishes that on-policy PPO required the *full* 3,000-episode budget merely to tie a non-learned heuristic in this environment — i.e., this method class needs a large episode budget to move performance at all. Yet the ablation's "RL fine-tune adds nothing" conclusion (§VI-C) is based on removing only 300 episodes of "light" fine-tuning (§IV-D, Fig. 1). No learning curve for the fine-tune phase is shown, so it is not established whether 300 episodes represents a plateaued/converged phase or simply too few episodes to detect any effect either way. *Improvement*: report a fine-tune-episode scaling curve (e.g., 300/900/1500/3000 episodes) with variance bars to distinguish a plateau from an under-trained probe.
3. **Same-team circularity in the teacher/environment relationship is not discussed as a limitation**: §IV-D states the CP-SAT teacher solves "an formulation of the identical constraint set as the environment" — i.e., the same team that authored the synthetic scenario generator (§III-A, entirely built in-house) also authored the evaluation environment and the teacher solver against a matching constraint model. This means CP-SAT's "near-optimal ceiling" status and DFPS's ability to imitate it well are, to some extent, guaranteed by construction rather than by any external ground truth. *Improvement*: add an explicit limitations paragraph acknowledging that all components were built by the same team without independent replication or real-world data validation.
4. **Cohen's d values are inconsistent with the "practically small" framing used to explain away the losses**: Table II reports Cohen's d as large as −17.57 (CP-SAT), −12.26 (HEFT), −11.41 (Greedy), and −9.92 (Source PPO) for DFPS's losses, which are simultaneously described as "practically small (0.3–3.8%)" (§VI-B, Abstract). Conventionally, |d| > 0.8 is "large"; values in the 10–18 range are enormous by any standard interpretation, almost certainly an artifact of computing d on the paired-difference SD (which becomes very small precisely because DFPS and the deterministic comparators are evaluated on the *same* 100 scenarios). *Improvement*: state explicitly which formula was used for d, and if paired, note the resulting d is not comparable to conventional between-groups benchmarks.
5. **Asymmetric hyperparameter-tuning budgets across compared methods are not fully specified, and no CI/power analysis is reported**: §V-B states only "the two most promising on-policy variants (HCBS-PPO, TAGS)" received a capped Optuna TPE search; it is not stated whether DFPS's own hyperparameters or the five external RL baselines received any tuning at all. No a priori power analysis is reported anywhere, and no explicit 95% CIs are given (only ± SD). *Improvement*: clarify the tuning budget applied to every method in a methods appendix, and report 95% CIs for the primary cost metric.

### Detailed Comments

#### Research Questions & Hypotheses
The central research question — "is expensive from-scratch reinforcement learning training necessary, or can a cheaper alternative reach comparable quality?" — is clear, falsifiable, and directly motivated by a concrete empirical finding (§III-B). No formal statistical hypotheses are stated for the ablation, which becomes a liability once ablation results are used to support a strong causal claim in the Abstract and Conclusion.

#### Research Design
The overall design — reproduce → diagnose → screen on validation → freeze → evaluate once on test — is appropriate and the freeze-before-test discipline is a genuine strength. External validity is limited by the entirely synthetic, single-team-authored environment with no real electricity/carbon market data.

#### Sampling Strategy
The 600/100/100 scenario split is adequately sized for the paired t-tests over 100 test scenarios. The 5-seed protocol is modest for detecting anything but large effects when used as the sole basis for a null-result claim (ablation) rather than as a variance-characterization device.

#### Data Collection
The environment is described with commendable specificity. The CP-SAT teacher's 15-second per-scenario time limit for training labels versus a longer 30-second limit for the benchmark ceiling (§V-C) is an inconsistency not discussed — up to ~49% of imitation-learning target labels may be sub-optimal, bearing on the 73.3% top-1 imitation accuracy figure.

#### Analysis Methods
The primary analysis (paired t-test, Holm-Bonferroni, α=0.05, % improvement + Cohen's d) is well-chosen and well-executed for the main table. Gaps: no a priori power analysis; no 95% CI reporting; no Type II error discussion for the ablation; no normality/independence checks reported (minor via CLT at n=100, but should be mentioned).

#### Results Presentation
Results are presented completely, including unfavorable ones. Table III's ablation presents no uncertainty quantification beyond ± SD, and the error bars in Fig. 5(a) appear to be of individual variants, not of the paired differences between DFPS and each ablated variant — the relevant quantity for a causal "no effect" claim.

#### Reproducibility
The paper references several companion artifacts (reproduction report, model-selection report, literature-check artifact) not included in the 10-page manuscript. I recommend the editor confirm these are attached/accessible, since several of my concerns hinge on details asserted to live in them.

#### Methodological Fallacies Detected
- Overfitting / insufficient power for a null claim (ablation).
- Multiple comparisons correctly handled for the main table (strength) but the ablation's implicit multiple "no difference" verdicts have no correction or single-test evidence.
- Confirmation bias risk (partially mitigated by freeze-before-test; residual risk in the ablation's causal narrative and the unaddressed teacher/environment circularity).
- Endogeneity / omitted-variable risk in the "training methodology is the mechanism" causal claim (component removal vs. hyperparameter-transfer effects not separated).

### Questions for Authors
1. For the ablation study, can you report a formal paired test or bootstrap CI for each component-removal contrast, plus the minimum detectable effect size at n=5?
2. Was the 300-episode RL fine-tune assessed at any other episode budget, given §III-B independently establishes PPO needs the full 3,000-episode budget to move performance at all?
3. Which Cohen's d formula was used in Table II? Please reconcile the large d values with the "practically small" framing.
4. Were DFPS's own hyperparameters subjected to the same Optuna search as HCBS-PPO/TAGS? Were the five external RL baselines tuned at all?
5. For win/loss verdicts against deterministic, 1-seed benchmarks, how is the paired t-test constructed given DFPS has 5 seeds per scenario — averaged first, or pseudo-replicated?
6. What independent check would help rule out constraint-model circularity between the CP-SAT teacher and the evaluation environment?
7. Why do the training-teacher CP-SAT run (15s limit) and the benchmark-ceiling CP-SAT run (30s limit) use different time budgets, and does this affect the 73.3% imitation-accuracy interpretation?

### Minor Issues
- §V-D states RL baselines "not central to the primary comparison" use "1 seed," yet Table I reports ± SD for A2C, DQN, MLP-PPO-no-attn; clarify these SDs are across scenarios, not seeds.
- A single consolidated "Limitations" subsection would help (currently scattered).
- Table II's "verdict" column would be easier to audit with per-comparison post-Holm p-values shown, not only the binary label.
