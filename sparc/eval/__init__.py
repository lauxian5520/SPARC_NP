"""评价 (§12)。"""

from sparc.eval.safe_coverage import SafeCoverageResult, risk_coverage_curve, safe_coverage
from sparc.eval.metrics import (
    HypothesisResult,
    bootstrap_ci,
    cohens_d,
    evaluate_h1,
    evaluate_h2,
    evaluate_h3,
    harm_rate,
    negative_transfer_rate,
    paired_bootstrap_auroc_diff,
    roc_auc,
    rmse,
    spearman,
)
from sparc.eval.baselines import MeanPredictor, evaluate_baselines
from sparc.eval.tables import render_table0, render_table1, render_table2, render_table3
from sparc.eval.figures import plot_gate_coefficients, plot_risk_coverage

__all__ = [
    "SafeCoverageResult", "risk_coverage_curve", "safe_coverage",
    "HypothesisResult", "bootstrap_ci", "cohens_d", "evaluate_h1", "evaluate_h2", "evaluate_h3",
    "harm_rate", "negative_transfer_rate", "paired_bootstrap_auroc_diff", "roc_auc", "rmse", "spearman",
    "MeanPredictor", "evaluate_baselines",
    "render_table0", "render_table1", "render_table2", "render_table3",
    "plot_gate_coefficients", "plot_risk_coverage",
]
