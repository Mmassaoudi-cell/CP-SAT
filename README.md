# Solver-Distilled Reinforcement Learning for Multi-Data-Center Migration Scheduling

Independent reproduction, weakness analysis, and follow-on methodology study built from the IEEE Transactions on Industry Applications paper *"Deep Reinforcement Learning-Based Migration and Scheduling Strategy for Data Centers in Electricity and Carbon Markets"* (Wang et al., 2026, DOI: 10.1109/TIA.2026.3691718).

## Start here

Read these in order for the full narrative: `SOURCE_PAPER_AUDIT.md` → `REPRODUCTION_REPORT.md` → `SOURCE_WEAKNESS_ANALYSIS.md` → `MODEL_CANDIDATES.md` → `MODEL_SELECTION_REPORT.md` → `FINAL_MODEL_CONFIG.yaml` → `ANALYSIS/results/ablation_report.md` → `BENCHMARK_WTL.csv` → `MANUSCRIPT/manuscript.pdf` → `FINAL_RESEARCH_SUMMARY.md`.

## Repository layout

```
SOURCE_METHOD_REPRODUCTION/   Frozen reproduction of the source paper's PPO scheduler (mandatory benchmark)
PROPOSED_METHOD/               5 candidate hybrid models, screening, tuning, final training, distillation, ablation
BENCHMARKS/                    10+ benchmark implementations (heuristics, CP-SAT, A2C, DQN, MLP-PPO)
ANALYSIS/                      Statistics, robustness, figures, LaTeX tables
MANUSCRIPT/                    IEEE Transactions manuscript (manuscript.tex / manuscript.pdf)
DATA_SPLIT_MANIFEST.csv        Frozen 600/100/100 train/validation/test scenario split
FINAL_MODEL_CONFIG.yaml        Frozen configuration for the selected final model (DFPS)
BENCHMARK_WTL.csv              Win/tie/loss statistical results, DFPS vs. all 14 benchmarks
```

## Setup

```bash
pip install -r requirements.txt
```

CUDA is used automatically if available (`torch.cuda.is_available()`); all scripts fall back to CPU otherwise (slower, particularly for the autoregressive-family candidates).

All commands below are run from the repository root.

## 1. Reproduce the source method

```bash
python -m SOURCE_METHOD_REPRODUCTION.src.train_source --episodes 3000
python -m SOURCE_METHOD_REPRODUCTION.src.evaluate_source --checkpoint SOURCE_METHOD_REPRODUCTION/results/ppo_attention_3000_episodes.pt --split validation
python -m SOURCE_METHOD_REPRODUCTION.src.evaluate_source --checkpoint SOURCE_METHOD_REPRODUCTION/results/ppo_attention_3000_episodes.pt --split test
```

## 2. Candidate screening (validation only)

```bash
# Stage 2: reduced-budget, 3-seed screening for each of the 4 PPO-trainable candidates
for cand in gat_ppo lc_ppo hcbs_ppo tags; do
  for seed in 2026 7 42; do
    python -m PROPOSED_METHOD.src.train --candidate $cand --episodes 600 --seed $seed --tag screen_${cand}_seed${seed}
  done
done

# Stage 3: Optuna tuning (validation-only, capped budget)
python -m PROPOSED_METHOD.src.tune --candidate hcbs_ppo --trials 15 --tune_episodes 400 --eval_scenarios 30
python -m PROPOSED_METHOD.src.tune --candidate tags --trials 15 --tune_episodes 800 --eval_scenarios 30
```

## 3. Train the final model (DFPS)

```bash
# One-time CP-SAT teacher-label generation (200 training scenarios)
python -m PROPOSED_METHOD.src.generate_teacher_labels --n_scenarios 200 --time_limit_s 15

# Distillation + light RL fine-tune, 5 seeds
for seed in 2026 7 42 123 777; do
  python -m PROPOSED_METHOD.src.distill --epochs 20 --finetune_episodes 300 --seed $seed --tag final_distilled_seed${seed}
done

# Evaluate on validation, then on the untouched test split
for seed in 2026 7 42 123 777; do
  python -m PROPOSED_METHOD.src.evaluate --checkpoint PROPOSED_METHOD/results/final_distilled_seed${seed}.pt --candidate tags --split validation
  python -m PROPOSED_METHOD.src.evaluate --checkpoint PROPOSED_METHOD/results/final_distilled_seed${seed}.pt --candidate tags --split test
done
```

## 4. Ablation

```bash
for seed in 2026 7 42 123 777; do
  python -m PROPOSED_METHOD.src.distill --epochs 20 --finetune_episodes 0 --seed $seed --tag ablation_no_finetune_seed${seed}
  python -m PROPOSED_METHOD.src.distill --epochs 20 --finetune_episodes 300 --no_topology --seed $seed --tag ablation_no_topology_seed${seed}
done
```

## 5. Robustness

```bash
python -m ANALYSIS.src.run_robustness --checkpoint PROPOSED_METHOD/results/final_distilled_seed2026.pt --candidate tags --n_scenarios 50
```

## 6. Benchmark suite

```bash
python -m BENCHMARKS.src.evaluate_benchmarks --split test --methods edf carbon_greedy heft random --out BENCHMARKS/results/heuristics_test_results.csv
python -m BENCHMARKS.src.evaluate_benchmarks --split test --methods cpsat --cpsat_time_limit 30 --out BENCHMARKS/results/cpsat_test_results.csv
python -m BENCHMARKS.src.train_rl_baseline --algorithm a2c --episodes 3000 --seed 2026 --tag a2c_seed2026_3000ep
python -m BENCHMARKS.src.train_rl_baseline --algorithm mlp_ppo --episodes 3000 --seed 2026 --tag mlp_ppo_seed2026_3000ep
python -m BENCHMARKS.src.train_rl_baseline --algorithm dqn --episodes 1500 --seed 2026 --tag dqn_seed2026_1500ep
python -m BENCHMARKS.src.evaluate_benchmarks --split test --methods --rl_checkpoints a2c:BENCHMARKS/results/a2c_seed2026_3000ep.pt mlp_ppo:BENCHMARKS/results/mlp_ppo_seed2026_3000ep.pt dqn:BENCHMARKS/results/dqn_seed2026_1500ep.pt --out BENCHMARKS/results/rl_baselines_test_results.csv
```

## 7. Statistics, tables, figures

```bash
python -m ANALYSIS.src.make_tables
python -m ANALYSIS.src.make_figures
python -m ANALYSIS.src.make_schematic_figures
```

The win/tie/loss statistical analysis (`BENCHMARK_WTL.csv`) is produced by `ANALYSIS.src.stats.wtl_summary`; see `MODEL_SELECTION_REPORT.md` for the exact invocation used.

## 8. Compile the manuscript

```bash
cd MANUSCRIPT
pdflatex -interaction=nonstopmode manuscript.tex
pdflatex -interaction=nonstopmode manuscript.tex   # second pass for cross-references
```

Requires a LaTeX distribution with the `IEEEtran` document class (MiKTeX or TeX Live).

## Notes on reproducibility

- All scenario generation is seeded and deterministic (`SOURCE_METHOD_REPRODUCTION/src/scenario.py`); the train/validation/test split (`DATA_SPLIT_MANIFEST.csv`) is frozen and was never altered based on results.
- `FINAL_MODEL_CONFIG.yaml` is frozen after Stage 4 model selection; no test-split result influenced any config value in that file.
- Training-compute figures reported in the manuscript were measured in isolated (uncontended) conditions on a shared 24-core machine; later runs in this repository's own logs may show longer wall-clock times due to concurrent unrelated processes on that machine, but relative comparisons (episode counts, decision-epoch counts, parameter counts) are contention-independent and are the figures relied upon in the paper's efficiency claims.
- The data used throughout is a documented **synthetic reconstruction** (`DATA_AUDIT.md`) of the source paper's stated protocol, not author-released data (none exists publicly). Absolute dollar values are not expected to match the source paper's published table; qualitative comparisons and all of this repository's own internal comparisons (which all use the same synthetic data) are valid.
