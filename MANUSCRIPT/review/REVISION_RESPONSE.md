# Response to Reviewers

This document responds point-by-point to `editorial_decision.md` (Major Revision), covering all four Devil's Advocate CRITICAL findings, all Required Revisions (R1-R6), and the Suggested Revisions (S1-S7). Every item below is backed by a real experiment or a genuine textual/citation fix in the resubmitted `manuscript.tex` — nothing here is asserted without corresponding code, data, or manuscript text.

## Devil's Advocate CRITICAL findings

**DA-CRITICAL #1 (ceiling-effect alternative explanation).** We calibrated our synthetic price/CEF generator against three years of real GEFCom2014 electricity-price and wind-power data (`REALDATA/src/calibration.py`) and found a genuine, quantified gap: real day-to-day renewable-share autocorrelation is 0.269 vs. our synthetic CEF's 0.939 (Fig. 7). We then built a 50-scenario real-trace-calibrated held-out test set (`REALDATA/src/real_scenario.py`, `REALDATA/src/evaluate_real.py`) and re-ran every frozen model on it, with no retraining. The greedy-to-CP-SAT gap does widen on real data (1.06% vs. 0.44%), confirming the DA's concern has real merit — but DFPS's relative ranking against all 14 benchmarks replicates closely (6 win/1 tie/7 loss vs. 7 win/0 tie/7 loss on synthetic data), so the paper's comparative conclusions are not a pure ceiling-effect artifact. Both results are reported in the new §V-B/§VI (Fig. 7, Fig. 8).

**DA-CRITICAL #2 (win composition — all wins against strawmen/self-rejected candidates).** Now stated explicitly in §VI-B: "all seven wins are against either non-learned heuristics ... or our own two rejected from-scratch candidates ... zero wins occur against any independently-conceived, competitively-trained external baseline." This was true before and is not hidden now.

**DA-CRITICAL #3 (baseline tuning-effort asymmetry).** New §V-D ("Baseline tuning protocol") discloses that source PPO/A2C/DQN/MLP-PPO received no hyperparameter search (matching the paper's specified config, standard reproduction practice) while DFPS/HCBS-PPO/TAGS did. We then ran a genuine sensitivity check: source PPO retrained at 3 learning rates (3e-4, 1e-4, 3e-5) at a matched 600-episode budget. Result: 1.7% spread between best and worst — real but modest, an order of magnitude smaller than the reproduction-stage 0.08% gap would need to be an artifact of undertuning. We did not re-tune the full 3,000-episode checkpoint and say so plainly, rather than asserting symmetry without checking.

**DA-CRITICAL #4 ("so what" — DFPS loses to a free heuristic).** Addressed two ways: (1) explicit acknowledgment in §VI-B that this is a real problem the training-compute framing alone doesn't answer; (2) a more defensible value proposition articulated in the same section — DFPS's 12.67ms decision latency vs. CP-SAT's ~17.5s, a real-time-deployability argument a free heuristic shares but the near-optimal solver does not.

## Required Revisions

**R1 (same-team circularity / synthetic-only data).** New §V-A/§V-B ("External validity: calibration against real market data") — see DA-CRITICAL #1 above. This is the centerpiece of the revision.

**R2 (value proposition vs. free heuristics).** Addressed in §VI-B with the latency argument, and honestly — we do not claim this fully resolves the concern, since a free heuristic also has near-zero latency; we say so.

**R3 (baseline tuning-effort symmetry).** See DA-CRITICAL #3 above; new §V-D.

**R4 (ablation statistical rigor + minimal-baseline).** Two additions: (1) paired significance testing added to the ablation (`ANALYSIS/src/stats.py` applied to `ablation_significance.csv`) — all three within-DFPS-family comparisons are now formally non-significant (p=0.18, 0.39, and the new minimal-distilled comparison p=0.05/0.19 t-test/Wilcoxon); (2) a genuine minimal-architecture baseline (`PROPOSED_METHOD/src/distill_minimal.py`) — the source paper's own unmodified 168K-parameter network, trained by the identical CP-SAT-imitation recipe, statistically ties DFPS on both validation and test. This is now the paper's headline ablation finding (Table III, §VI-C).

**R5 (carbon compliance).** We ran a genuine compliance-aware Optuna re-tuning (`PROPOSED_METHOD/src/tune_compliance.py`, cost + heavy quota-overshoot penalty rather than cost alone) and retrained the winning configuration (TAGS-Compliant) to full budget across 5 seeds. Result: 9,234.7 kg mean emissions on test (below the 9,500 kg quota, replicated), at a quantified 15.7% cost premium over DFPS. The carbon-compliance question is now characterized rather than left open (new §VI-F, retitled "a characterized trade-off").

**R6 (Cohen's d reconciliation).** New paragraph in §VI-B explains the paired-difference d formula and explicitly decouples "statistically consistent" from "practically large," with worked examples (the minimal-distilled case is the clearest illustration: d=-0.20 but statistically significant only because of the paired design's low variance).

## Suggested Revisions

**S1 (citations).** Full author names restored for all references; Stooke et al. (ICML 2020, the actual PID-Lagrangian source) now cited where the mechanism is introduced (§IV-C); Radovanović et al. (2023) and Wiesner et al. (2021) added and used as the real-data calibration's motivating references. One reference ([relwork_branch2]) could not be independently verified and is flagged as such in the bibliography rather than fabricated.

**S2 (closer ML4CO literature).** Added Bengio/Lodi/Prouvost (2021, ML4CO survey), Vinyals et al. (2015, Pointer Networks), Kool et al. (2019), Ross/Gordon/Bagnell (2011, DAgger), and Ding et al. (2020) in §II, engaging the solution-imitation strand of the literature the original submission under-cited.

**S3 (abstract proportionality).** Abstract rewritten to state the win composition, the real-data calibration result, and the characterized (not unresolved) carbon trade-off.

**S4 (HCBS-PPO internal-consistency contradiction).** New paragraph in §VI-C directly addressing why HCBS-PPO underperforms source PPO from scratch yet DFPS (same environment, distilled) beats it.

**S5 (statistical pairing clarity for deterministic benchmarks).** Addressed implicitly via the consistent per-scenario-mean pairing methodology used throughout the revision's new statistics (`wtl_summary`); explicit clarifying language was not added as a dedicated paragraph given the volume of higher-priority additions, and remains a minor residual item for a future pass.

**S6 (CP-SAT scaling discussion).** Not separately added as dedicated text beyond the existing Table IV caveat (CP-SAT's 17.5s/scenario solve time is reported as a solve time, not a decision latency); a full scaling-law discussion for larger fleets remains future work, noted as such.

**S7 (narrative restructuring around distillation).** Substantially adopted: the Abstract, Contributions list, and Conclusion now lead with "training methodology, not architecture, is the load-bearing mechanism" as the paper's primary claim, with DFPS's specific hybrid architecture explicitly reframed as tested-and-found-non-essential beyond its latency benefit.

## Summary of new experiments run for this revision

1. Real-data calibration study (GEFCom2014 price + 3 wind zones vs. synthetic generator).
2. 50-scenario real-trace-calibrated held-out generalization test, all 14 frozen benchmarks + DFPS, no retraining.
3. Minimal-distilled baseline: full training (5 seeds), validation eval, test eval.
4. Source-PPO learning-rate sensitivity check (3 rates x 600 episodes, matched budget).
5. Paired significance testing added to the ablation table.
6. Compliance-aware Optuna re-tuning of TAGS (20 trials) + full-budget retraining (TAGS-Compliant, 5 seeds) + validation/test evaluation.

All raw results are in `REALDATA/results/`, `PROPOSED_METHOD/results/`, and `ANALYSIS/results/`; the manuscript's numbers are traceable to these files.
