from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sstats


def paired_comparison(proposed: np.ndarray, baseline: np.ndarray, alpha: float = 0.05) -> dict:
    """Paired comparison on matched scenarios: paired t-test + Wilcoxon
    signed-rank + Cohen's d effect size. Lower is better (cost/emissions)."""
    proposed = np.asarray(proposed, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    assert len(proposed) == len(baseline)
    diff = baseline - proposed  # positive => proposed is better (lower)
    t_stat, t_p = sstats.ttest_rel(baseline, proposed)
    try:
        w_stat, w_p = sstats.wilcoxon(baseline, proposed)
    except ValueError:
        w_stat, w_p = float("nan"), float("nan")
    pooled_std = diff.std(ddof=1)
    cohens_d = float(diff.mean() / pooled_std) if pooled_std > 0 else float("nan")
    pct_improvement = float(diff.mean() / baseline.mean() * 100.0)
    return {
        "n": len(proposed),
        "mean_proposed": float(proposed.mean()),
        "mean_baseline": float(baseline.mean()),
        "mean_diff": float(diff.mean()),
        "pct_improvement": pct_improvement,
        "t_stat": float(t_stat),
        "t_pvalue": float(t_p),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_pvalue": float(w_p),
        "cohens_d": cohens_d,
        "significant_alpha_0.05_uncorrected": bool(t_p < alpha),
    }


def holm_correction(pvalues: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni step-down correction. Returns per-comparison reject flags."""
    order = np.argsort(pvalues)
    m = len(pvalues)
    reject = [False] * m
    for rank, idx in enumerate(order):
        threshold = alpha / (m - rank)
        if pvalues[idx] <= threshold:
            reject[idx] = True
        else:
            break
    return reject


def wtl_summary(proposed_col: str, comparison_df: pd.DataFrame, methods: list[str], metric: str = "system_cost", alpha: float = 0.05) -> pd.DataFrame:
    """comparison_df must have columns [scenario_id, method, <metric>] with one
    row per (scenario, method), matched scenario_ids across methods."""
    rows = []
    proposed = comparison_df[comparison_df["method"] == proposed_col].set_index("scenario_id")[metric]
    pvalues = {}
    stats_by_method = {}
    for method in methods:
        baseline = comparison_df[comparison_df["method"] == method].set_index("scenario_id")[metric]
        common = proposed.index.intersection(baseline.index)
        result = paired_comparison(proposed.loc[common].values, baseline.loc[common].values, alpha=alpha)
        pvalues[method] = result["t_pvalue"]
        stats_by_method[method] = result
    reject = holm_correction(list(pvalues.values()), alpha=alpha)
    for method, rejected in zip(pvalues.keys(), reject):
        result = stats_by_method[method]
        if rejected and result["mean_diff"] > 0:
            verdict = "win"
        elif rejected and result["mean_diff"] < 0:
            verdict = "loss"
        else:
            verdict = "tie"
        rows.append({"benchmark": method, "verdict": verdict, "holm_significant": rejected, **result})
    return pd.DataFrame(rows)
