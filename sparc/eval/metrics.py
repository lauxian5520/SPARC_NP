"""指标与三个假设的判据评估 (§1.3, §12.2)。

**每一张表都必须带均值预测器基线**（事实 A）：在 NPASS 上按 NaFM 口径
复原其 8 个靶点时，已发表 SOTA 在其中 4 个跑不过"预测训练集均值"。
PTP-1B 上 NaFM 的 RMSE 比均值基线高 31%（隐含 R² = −0.73）。
这就是主标不能是 RMSE 的直接理由。

统计工具全部用 numpy 自实现 —— 服务器上未必装 scipy，而这些量
（bootstrap CI、Cohen's d、AUROC、Spearman）都不复杂，
自实现比多一个依赖更可控。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sparc.common.logging_utils import get_logger

_LOGGER = get_logger(__name__)


# ======================================================================
# 基础统计
# ======================================================================
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方根误差。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2))) if y_true.size else float("nan")


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """决定系数 R²（可为负 —— 事实 A 里 NaFM 在 PTP-1B 上就是 −0.73）。"""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _rankdata(values: np.ndarray) -> np.ndarray:
    """平均秩（处理并列）。"""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < sorted_values.size:
        j = i
        while j + 1 < sorted_values.size and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return ranks


def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman 秩相关。"""
    a = np.asarray(y_true, dtype=np.float64)
    b = np.asarray(y_pred, dtype=np.float64)
    if a.size < 2:
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """AUROC（用秩公式，等价于 Mann–Whitney U）。"""
    labels = np.asarray(labels, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = float((labels == 1).sum())
    n_neg = float((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(scores)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def cohens_d(values: np.ndarray, mu0: float = 0.0) -> float:
    """单样本 Cohen's d（H1 判据 (c)：d ≥ 0.2）。"""
    v = np.asarray(values, dtype=np.float64)
    if v.size < 2:
        return float("nan")
    sd = v.std(ddof=1)
    return float((v.mean() - mu0) / sd) if sd > 0 else float("inf" if v.mean() > mu0 else 0.0)


def bootstrap_ci(
    values: np.ndarray,
    statistic: str = "mean",
    ci_level: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """bootstrap 置信区间。

    Args:
        values: 一维样本。
        statistic: ``"mean"`` / ``"median"``。
        ci_level: 置信水平。
        n_bootstrap: 重抽样次数。
        seed: 随机种子。

    Returns:
        ``(点估计, CI 下界, CI 上界)``。
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    func = np.mean if statistic == "mean" else np.median
    point = float(func(v))
    rng = np.random.default_rng(seed)
    samples = v[rng.integers(0, v.size, size=(n_bootstrap, v.size))]
    stats = func(samples, axis=1)
    alpha = (1.0 - ci_level) / 2.0
    return point, float(np.quantile(stats, alpha)), float(np.quantile(stats, 1.0 - alpha))


def paired_bootstrap_auroc_diff(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    ci_level: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """配对 bootstrap 比较两个打分器的 AUROC（H2 判据 (a)）。

    Args:
        labels: ``(N,)`` 0/1。
        scores_a: ``(N,)`` 待检打分器（28 维门控）。
        scores_b: ``(N,)`` 基线打分器（Tanimoto-Top1 标量）。
        ci_level: 置信水平。
        n_bootstrap: 重抽样次数。
        seed: 随机种子。

    Returns:
        含 ``auroc_a`` / ``auroc_b`` / ``diff`` / ``ci_lower`` / ``ci_upper``
        与 ``criterion_a_passed``（CI 下界 > 0）的字典。
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap, dtype=np.float64)
    n = labels.size
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(labels[idx])) < 2:
            diffs[i] = np.nan
            continue
        diffs[i] = roc_auc(labels[idx], scores_a[idx]) - roc_auc(labels[idx], scores_b[idx])
    diffs = diffs[np.isfinite(diffs)]
    alpha = (1.0 - ci_level) / 2.0
    lower = float(np.quantile(diffs, alpha)) if diffs.size else float("nan")
    upper = float(np.quantile(diffs, 1.0 - alpha)) if diffs.size else float("nan")
    return {
        "auroc_a": roc_auc(labels, scores_a),
        "auroc_b": roc_auc(labels, scores_b),
        "diff": float(diffs.mean()) if diffs.size else float("nan"),
        "ci_lower": lower,
        "ci_upper": upper,
        "criterion_a_passed": bool(np.isfinite(lower) and lower > 0),
    }


# ======================================================================
# 辅助指标 (§12.2)
# ======================================================================
def negative_transfer_rate(loss_retrieval: np.ndarray, loss_base: np.ndarray) -> float:
    """``P(ℓ_retr > ℓ_base)``。"""
    return float((np.asarray(loss_retrieval) > np.asarray(loss_base)).mean())


def harm_rate(loss_retrieval: np.ndarray, loss_base: np.ndarray, epsilon: float = 0.01) -> float:
    """``P(ℓ_retr > ℓ_base + ε)``。"""
    return float((np.asarray(loss_retrieval) > np.asarray(loss_base) + epsilon).mean())


def retrieval_coverage(gate_values: np.ndarray, lam: float) -> float:
    """``P(g ≥ λ)``。"""
    return float((np.asarray(gate_values) >= lam).mean())


def risk_coverage_auc(curve: Sequence[Dict[str, float]]) -> float:
    """Risk–Coverage 曲线下面积（梯形法）。"""
    if len(curve) < 2:
        return float("nan")
    coverage = np.array([row["coverage"] for row in curve])
    risk = np.array([row["harm_rate"] for row in curve])
    return float(np.trapezoid(risk, coverage)) if hasattr(np, "trapezoid") else float(np.trapz(risk, coverage))


# ======================================================================
# 三个假设的判据评估
# ======================================================================
@dataclass
class HypothesisResult:
    """一个假设的判定结果。"""

    hypothesis: str
    passed: bool
    criteria: Dict[str, Any] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转成可写报告的字典。"""
        return {
            "hypothesis": self.hypothesis, "passed": self.passed,
            "criteria": self.criteria, "details": self.details, "note": self.note,
        }


def evaluate_h1(
    delta_loss_low_support: np.ndarray,
    per_target_delta_mean: Dict[str, float],
    ci_level: float = 0.95,
    n_bootstrap: int = 10000,
    min_target_agreement: float = 0.60,
    min_cohens_d: float = 0.20,
    seed: int = 42,
) -> HypothesisResult:
    """H1：朴素跨域检索在低支持区造成负迁移。

    判据（三条**全部**满足才算成立）：
      (a) ``Δℓ = ℓ(naive) − ℓ(base)`` 在低支持子集上均值的 95% CI 下界 > 0；
      (b) 在 ≥ 60% 的入选靶点上方向一致；
      (c) 效应量 Cohen's d ≥ 0.2。

    Args:
        delta_loss_low_support: ``(N,)`` 低支持子集（Top-1 Tanimoto < 0.35）上的 Δℓ。
        per_target_delta_mean: ``{target_id: Δℓ 均值}``。
        ci_level: 置信水平。
        n_bootstrap: bootstrap 次数。
        min_target_agreement: 判据 (b) 阈值。
        min_cohens_d: 判据 (c) 阈值。
        seed: 随机种子。

    Returns:
        :class:`HypothesisResult`。**不成立时应立即触发 §13.5 止损**，
        转为分析/基准论文（Plan-B）—— 那不是失败，是换定位。
    """
    point, lower, upper = bootstrap_ci(delta_loss_low_support, "mean", ci_level, n_bootstrap, seed)
    agreement = (
        sum(1 for v in per_target_delta_mean.values() if v > 0) / len(per_target_delta_mean)
        if per_target_delta_mean else 0.0
    )
    d = cohens_d(delta_loss_low_support, 0.0)

    criteria = {
        "a_ci_lower_gt_zero": {"value": lower, "threshold": 0.0, "passed": bool(lower > 0)},
        "b_target_agreement": {"value": agreement, "threshold": min_target_agreement,
                               "passed": bool(agreement >= min_target_agreement)},
        "c_cohens_d": {"value": d, "threshold": min_cohens_d, "passed": bool(d >= min_cohens_d)},
    }
    passed = all(c["passed"] for c in criteria.values())
    return HypothesisResult(
        hypothesis="H1", passed=passed, criteria=criteria,
        details={"delta_loss_mean": point, "ci": [lower, upper], "n_samples": int(np.size(delta_loss_low_support)),
                 "n_targets": len(per_target_delta_mean)},
        note=("H1 成立：低支持区存在可测的负迁移" if passed else
              "H1 不成立 ⇒ 无负迁移可控 ⇒ 立即执行 §13.5 Plan-B（基准完整性 + 分析论文）"),
    )


def evaluate_h2(
    labels: np.ndarray,
    gate_scores_oof: np.ndarray,
    tanimoto_scores: np.ndarray,
    gate_scores_test: Optional[np.ndarray] = None,
    labels_test: Optional[np.ndarray] = None,
    cliff_mask: Optional[np.ndarray] = None,
    sign_flips: Optional[Sequence[str]] = None,
    min_cliff_auroc: float = 0.65,
    max_generalization_gap: float = 0.10,
    max_sign_flips: int = 2,
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> HypothesisResult:
    """H2：多维支持度可识别有害检索。

    判据（**全部**满足）：
      (a) OOF AUROC > Tanimoto-Top1 标量基线，配对 bootstrap 95% CI 下界 > 0；
      (b) 在 activity-cliff 子集上 AUROC ≥ 0.65；
      (c) 泛化间隙 ``|AUROC_OOF − AUROC_test| ≤ 0.10``；
      (d) 系数符号与化学先验一致，符号翻转数 ≤ 2/28。

    Args:
        labels: ``(N,)`` 效用标签 ``u_i``。
        gate_scores_oof: ``(N,)`` 门控 OOF 分数。
        tanimoto_scores: ``(N,)`` Tanimoto-Top1 基线分数。
        gate_scores_test: ``(M,)`` 测试集门控分数（判据 c）。
        labels_test: ``(M,)`` 测试集标签。
        cliff_mask: ``(N,)`` bool，activity-cliff 子集（判据 b）。
        sign_flips: 符号翻转的特征名列表（判据 d）。
        min_cliff_auroc: 判据 (b) 阈值。
        max_generalization_gap: 判据 (c) 阈值。
        max_sign_flips: 判据 (d) 阈值。
        n_bootstrap: bootstrap 次数。
        seed: 随机种子。

    Returns:
        :class:`HypothesisResult`。
    """
    comparison = paired_bootstrap_auroc_diff(labels, gate_scores_oof, tanimoto_scores,
                                             n_bootstrap=n_bootstrap, seed=seed)
    auroc_oof = comparison["auroc_a"]

    cliff_auroc = (
        roc_auc(labels[cliff_mask], gate_scores_oof[cliff_mask])
        if cliff_mask is not None and cliff_mask.any() else float("nan")
    )
    auroc_test = (
        roc_auc(labels_test, gate_scores_test)
        if gate_scores_test is not None and labels_test is not None else float("nan")
    )
    gap = abs(auroc_oof - auroc_test) if np.isfinite(auroc_test) else float("nan")
    n_flips = len(sign_flips) if sign_flips is not None else -1

    criteria = {
        "a_beats_tanimoto": {"value": comparison["diff"], "ci_lower": comparison["ci_lower"],
                             "passed": comparison["criterion_a_passed"]},
        "b_cliff_auroc": {"value": cliff_auroc, "threshold": min_cliff_auroc,
                          "passed": bool(np.isfinite(cliff_auroc) and cliff_auroc >= min_cliff_auroc)},
        "c_generalization_gap": {"value": gap, "threshold": max_generalization_gap,
                                 "passed": bool(np.isfinite(gap) and gap <= max_generalization_gap)},
        "d_sign_flips": {"value": n_flips, "threshold": max_sign_flips, "flips": list(sign_flips or []),
                         "passed": bool(0 <= n_flips <= max_sign_flips)},
    }
    passed = all(c["passed"] for c in criteria.values())
    return HypothesisResult(
        hypothesis="H2", passed=passed, criteria=criteria,
        details={"auroc_oof": auroc_oof, "auroc_tanimoto": comparison["auroc_b"], "auroc_test": auroc_test},
        note=("H2 成立" if passed else
              "H2 不成立。若失败在判据 (d)（符号翻转）：§16 R5 —— 符号翻转常常是"
              "未识别泄漏通道的指示器，应先排查泄漏而不是直接判失败"),
    )


def evaluate_h3(
    safe_coverage_value: float,
    per_target_safe_coverage: Dict[str, float],
    test_risk: Optional[float] = None,
    alpha: float = 0.10,
    min_safe_coverage: float = 0.30,
    min_target_positive_frac: float = 0.60,
) -> HypothesisResult:
    """H3：门控残差在保证不劣化的前提下获得非平凡覆盖率。

    判据：
      (a) ``SafeCoverage ≥ 0.30``（不可通过降覆盖率平凡满足）；
      (b) 在 ≥ 60% 的入选靶点上 ``SafeCoverage > 0``；
      (c) LTT 给出的 λ 在测试集上实际风险 ≤ α。

    Args:
        safe_coverage_value: 全局 ``c*``。
        per_target_safe_coverage: ``{target_id: c*}``。
        test_risk: LTT 的 ``λ*`` 在测试集上的实际风险。
        alpha: 风险上界。
        min_safe_coverage: 判据 (a) 阈值。
        min_target_positive_frac: 判据 (b) 阈值。

    Returns:
        :class:`HypothesisResult`。
    """
    positive_frac = (
        sum(1 for v in per_target_safe_coverage.values() if v > 0) / len(per_target_safe_coverage)
        if per_target_safe_coverage else 0.0
    )
    criteria = {
        "a_safe_coverage": {"value": safe_coverage_value, "threshold": min_safe_coverage,
                            "passed": bool(safe_coverage_value >= min_safe_coverage)},
        "b_target_positive_frac": {"value": positive_frac, "threshold": min_target_positive_frac,
                                   "passed": bool(positive_frac >= min_target_positive_frac)},
        "c_test_risk_le_alpha": {"value": test_risk, "threshold": alpha,
                                 "passed": bool(test_risk is not None and test_risk <= alpha)},
    }
    passed = all(c["passed"] for c in criteria.values())
    return HypothesisResult(
        hypothesis="H3", passed=passed, criteria=criteria,
        details={"n_targets": len(per_target_safe_coverage)},
        note=("H3 成立" if passed else "H3 不成立；若 λ* 不存在，见 §16 R4："
              "报告『在 α=0.10 下无安全工作点』本身是有效结论，并给出 α=0.20 的曲线"),
    )
