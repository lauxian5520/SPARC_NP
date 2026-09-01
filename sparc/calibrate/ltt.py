"""Learn-then-Test 阈值标定 (§11) —— 本项目的核心理论贡献。

**问题** (§11.1)：``talk.md`` 的做法是"枚举候选 τ_g，挑验证集上伤害率
最低的"。这是调参：报告出来的成绩带**后选择偏倚**，而 H3 的判据
建立在这个被污染的数字上。

**v1.0 方案** (§11.2)::

    风险函数  R(λ) = P( ℓ_SPARC(q) > ℓ_base(q) + ε | g(q) ≥ λ )
    即"在接受检索的样本中，被检索害到的比例"。

    1. 候选阈值网格  Λ = {0.00, 0.01, ..., 0.99}
    2. 对每个 λ ∈ Λ，在标定折上对 H_λ: R(λ) > α 做 Hoeffding–Bentkus p 值
    3. 用固定序列检验（λ 从大到小）或 Bonferroni 做 FWER ≤ δ 的多重校正
    4. λ* = 通过检验的最小 λ            # 风险受控前提下的最大覆盖率
    5. 输出保证：P( R(λ*) ≤ α ) ≥ 1 − δ    有限样本、分布无关

参数冻结在 ``preregistration.yaml``：α = 0.10、δ = 0.05、ε = 0.01。

**标定折的独立性**：§11.2 规定标定折不参与 Θ_B/Θ_R 训练，且该折的
骨架家族同时从其自身的记忆库视图中剔除。若标定折参与过 Base 训练，
H3 全部三条判据同时失效。本模块假定调用方已经保证了这一点 ——
:func:`learn_then_test` 会检查并要求显式传入 ``calibration_is_held_out=True``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sparc.common.logging_utils import get_logger

_LOGGER = get_logger(__name__)


# ======================================================================
# p 值
# ======================================================================
def _binomial_cdf(k: int, n: int, p: float) -> float:
    """二项分布 CDF ``P(X ≤ k)``，用对数域求和避免大 n 溢出。"""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    total = 0.0
    log_p, log_q = math.log(p), math.log1p(-p)
    for i in range(0, min(k, n) + 1):
        log_term = (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                    + i * log_p + (n - i) * log_q)
        total += math.exp(log_term)
    return min(total, 1.0)


def hoeffding_pvalue(empirical_risk: float, n: int, alpha: float) -> float:
    """Hoeffding 界给出的 p 值：``exp(−2n·max(0, α−R̂)²)``。

    Args:
        empirical_risk: 标定折上的经验风险 ``R̂``。
        n: 标定折中"被接受检索"的样本数。
        alpha: 风险上界。

    Returns:
        ``H_λ: R(λ) > α`` 的 p 值。
    """
    if n <= 0:
        return 1.0
    gap = alpha - empirical_risk
    if gap <= 0:
        return 1.0
    return math.exp(-2.0 * n * gap * gap)


def bentkus_pvalue(empirical_risk: float, n: int, alpha: float) -> float:
    """Bentkus 界给出的 p 值：``e · P(Bin(n, α) ≤ ⌈n·R̂⌉)``。

    Args:
        empirical_risk: 经验风险。
        n: 样本数。
        alpha: 风险上界。

    Returns:
        p 值（截断到 1）。
    """
    if n <= 0:
        return 1.0
    k = int(math.ceil(n * empirical_risk))
    return min(1.0, math.e * _binomial_cdf(k, n, alpha))


def hoeffding_bentkus_pvalue(empirical_risk: float, n: int, alpha: float) -> float:
    """HB p 值：取 Hoeffding 与 Bentkus 两个界的**较小者**。

    Angelopoulos et al. 的标准做法 —— 两个界在不同区域各有优势
    （Hoeffding 在 α 远离 0/1 时紧，Bentkus 在小 α、小样本时紧），
    取小者仍是有效 p 值。

    Args:
        empirical_risk: ``R̂(λ)``。
        n: 被接受检索的样本数。
        alpha: 风险上界（冻结为 0.10）。

    Returns:
        有效 p 值。
    """
    return min(hoeffding_pvalue(empirical_risk, n, alpha), bentkus_pvalue(empirical_risk, n, alpha))


# ======================================================================
# 风险函数
# ======================================================================
def risk_at_lambda(
    gate_values: np.ndarray,
    loss_sparc: np.ndarray,
    loss_base: np.ndarray,
    lam: float,
    epsilon: float = 0.01,
) -> Tuple[float, int, float]:
    """计算 ``R(λ) = P(ℓ_SPARC > ℓ_base + ε | g ≥ λ)``。

    Args:
        gate_values: ``(N,)`` 门控值 ``g``。
        loss_sparc: ``(N,)`` SPARC-NP 的逐样本损失。
        loss_base: ``(N,)`` Base 的逐样本损失。
        lam: 阈值。
        epsilon: 伤害裕度（冻结为 0.01）。

    Returns:
        ``(经验风险, 被接受样本数, 覆盖率)``。被接受样本数为 0 时
        风险定义为 0.0 —— 空集上没有伤害，但覆盖率也是 0，
        这正是 ``SafeCoverage`` 不可被"少检索"平凡满足的原因。
    """
    accepted = gate_values >= lam
    n_accepted = int(accepted.sum())
    coverage = n_accepted / len(gate_values) if len(gate_values) else 0.0
    if n_accepted == 0:
        return 0.0, 0, 0.0
    harmed = (loss_sparc[accepted] > loss_base[accepted] + epsilon)
    return float(harmed.mean()), n_accepted, coverage


# ======================================================================
# LTT 主流程
# ======================================================================
@dataclass
class LTTResult:
    """Learn-then-Test 的完整输出。"""

    lambda_star: Optional[float]
    alpha: float
    delta: float
    epsilon: float
    multiple_testing: str
    n_calibration: int
    per_lambda: List[Dict[str, float]] = field(default_factory=list)
    guarantee: str = ""

    @property
    def found(self) -> bool:
        """是否存在满足风险约束的 λ。"""
        return self.lambda_star is not None

    def coverage_at_star(self) -> float:
        """``λ*`` 处的覆盖率。"""
        if not self.found:
            return 0.0
        for row in self.per_lambda:
            if abs(row["lambda"] - self.lambda_star) < 1e-12:
                return row["coverage"]
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """转成可写报告的字典。"""
        return {
            "lambda_star": self.lambda_star,
            "found": self.found,
            "coverage_at_lambda_star": self.coverage_at_star(),
            "alpha": self.alpha, "delta": self.delta, "epsilon": self.epsilon,
            "multiple_testing": self.multiple_testing,
            "n_calibration": self.n_calibration,
            "guarantee": self.guarantee,
            "per_lambda": self.per_lambda,
        }


def learn_then_test(
    gate_values: np.ndarray,
    loss_sparc: np.ndarray,
    loss_base: np.ndarray,
    alpha: float = 0.10,
    delta: float = 0.05,
    epsilon: float = 0.01,
    lambda_grid: Optional[Sequence[float]] = None,
    multiple_testing: str = "fixed_sequence",
    lambda_order: Optional[Sequence[float]] = None,
    calibration_is_held_out: bool = False,
) -> LTTResult:
    """在标定折上执行 Learn-then-Test，返回 ``λ*``。

    Args:
        gate_values: ``(N,)`` 标定折上的门控值。
        loss_sparc: ``(N,)`` SPARC-NP 逐样本损失。
        loss_base: ``(N,)`` Base 逐样本损失。
        alpha: 风险上界（0.10）。
        delta: FWER 水平（0.05）。
        epsilon: 伤害裕度（0.01）。
        lambda_grid: 候选阈值；``None`` 时用 ``{0.00, 0.01, ..., 0.99}``。
        multiple_testing: ``"fixed_sequence"``（一旦失败即停）或
            ``"bonferroni"``（全网格，阈值 δ/|Λ|）。
            **本项目默认 bonferroni**，理由见 :func:`diagnose_fixed_sequence_power`：
            固定序列从 λ=0.99 起测，而标定折在最大 λ 处往往只有个位数样本，
            此时即便经验风险为 0 也无功效（HB p 值 ≈ 0.9），序列在第一步就中断。
            规范 §11.2 允许两者择一，因此选 bonferroni 是规范内的选择而非偏离。
        lambda_order: 固定序列的**预先指定**顺序。为保持有限样本有效性，
            该顺序不得依赖标定折 —— 应由**内层模型选择折**（dev 的 20%）
            计算，见 :func:`fixed_sequence_order_from_inner_fold`。
            ``None`` 时退化为 §11.2 字面的"λ 从大到小"。
        calibration_is_held_out: 调用方确认标定折未参与 Θ_B/Θ_R 训练、
            且其骨架家族已从自身记忆库视图中剔除 (§11.2)。

    Returns:
        :class:`LTTResult`。``λ*`` 为 ``None`` 表示在 α 下不存在安全工作点 ——
        §16 的 R4 说明：**这本身是有效结论**，应报告并同时给出 α=0.20 的曲线。

    Raises:
        ValueError: 未确认标定折独立性，或输入长度不一致。
    """
    if not calibration_is_held_out:
        raise ValueError(
            "必须显式确认 calibration_is_held_out=True。§11.2：标定折不得参与 Θ_B/Θ_R 训练，"
            "且该折的骨架家族必须从其自身的记忆库视图中剔除。若标定折参与过 Base 训练，"
            "H3 的三条判据同时失效 —— 这个参数存在的目的就是让绕过它成为一个显式动作。"
        )
    gate_values = np.asarray(gate_values, dtype=np.float64)
    loss_sparc = np.asarray(loss_sparc, dtype=np.float64)
    loss_base = np.asarray(loss_base, dtype=np.float64)
    if not (gate_values.shape == loss_sparc.shape == loss_base.shape):
        raise ValueError("gate_values / loss_sparc / loss_base 长度必须一致")

    grid = list(lambda_grid) if lambda_grid is not None else [round(i * 0.01, 2) for i in range(100)]
    n_total = len(gate_values)

    # 固定序列的检验顺序：优先用调用方（在内层折上）预先指定的顺序；
    # 否则退化为 §11.2 字面的"λ 从大到小"。顺序不得依赖标定折，否则保证失效。
    if lambda_order is not None:
        ordered = [float(x) for x in lambda_order]
        unknown = set(ordered) - set(float(g) for g in grid)
        if unknown:
            raise ValueError(f"lambda_order 含不在网格中的值：{sorted(unknown)}")
    else:
        ordered = sorted(grid, reverse=True)
    threshold = delta if multiple_testing == "fixed_sequence" else delta / max(len(grid), 1)

    per_lambda: List[Dict[str, float]] = []
    lambda_star: Optional[float] = None
    sequence_broken = False

    for lam in ordered:
        risk, n_accepted, coverage = risk_at_lambda(gate_values, loss_sparc, loss_base, lam, epsilon)
        pvalue = hoeffding_bentkus_pvalue(risk, n_accepted, alpha)
        passed = (pvalue <= threshold) and (n_accepted > 0)

        per_lambda.append({
            "lambda": float(lam), "empirical_risk": float(risk), "n_accepted": float(n_accepted),
            "coverage": float(coverage), "pvalue": float(pvalue), "passed": float(passed),
        })

        if multiple_testing == "fixed_sequence":
            if sequence_broken:
                continue
            if passed:
                lambda_star = float(lam)      # 继续往小走，取通过检验的最小 λ
            else:
                sequence_broken = True        # 固定序列检验：一旦失败即停止
        elif passed:
            lambda_star = float(lam) if lambda_star is None else min(lambda_star, float(lam))

    per_lambda.sort(key=lambda row: row["lambda"])
    guarantee = (
        f"P( R(λ*={lambda_star}) ≤ α={alpha} ) ≥ 1 − δ = {1 - delta}（有限样本、分布无关）"
        if lambda_star is not None else
        f"在 α={alpha}、δ={delta} 下不存在满足风险约束的 λ —— 这是有效结论（§16 R4），"
        "应报告并同时给出 α=0.20 的 Risk–Coverage 曲线"
    )
    if lambda_star is None:
        _LOGGER.warning("LTT 未找到 λ*：%s", guarantee)
    else:
        row = next(r for r in per_lambda if abs(r["lambda"] - lambda_star) < 1e-12)
        _LOGGER.info(
            "LTT 完成：λ* = %.2f，覆盖率 %.3f，经验风险 %.3f，p 值 %.4g（n_cal=%d）",
            lambda_star, row["coverage"], row["empirical_risk"], row["pvalue"], n_total,
        )

    return LTTResult(
        lambda_star=lambda_star, alpha=alpha, delta=delta, epsilon=epsilon,
        multiple_testing=multiple_testing, n_calibration=n_total,
        per_lambda=per_lambda, guarantee=guarantee,
    )


def validate_on_test(
    result: LTTResult,
    gate_values: np.ndarray,
    loss_sparc: np.ndarray,
    loss_base: np.ndarray,
) -> Dict[str, Any]:
    """H3 判据 (c)：验证 ``λ*`` 在测试集上的实际风险 ≤ α。

    **只能在 S5 调用一次** (§10.4：S4 与 S5 之间不允许任何回溯修改)。

    Args:
        result: LTT 结果。
        gate_values: ``(N,)`` 测试集门控值。
        loss_sparc: ``(N,)``。
        loss_base: ``(N,)``。

    Returns:
        ``{"test_risk": .., "alpha": .., "criterion_c_passed": bool, ...}``。
    """
    if not result.found:
        return {"criterion_c_passed": False, "reason": "λ* 不存在，无可验证对象"}
    risk, n_accepted, coverage = risk_at_lambda(
        gate_values, loss_sparc, loss_base, result.lambda_star, result.epsilon
    )
    return {
        "lambda_star": result.lambda_star,
        "test_risk": risk,
        "test_coverage": coverage,
        "n_accepted": n_accepted,
        "alpha": result.alpha,
        "criterion_c_passed": bool(risk <= result.alpha),
        "note": "H3 判据 (c)：LTT 给出的 λ 在测试集上实际风险 ≤ α，验证保证有效",
    }


# ======================================================================
# 固定序列检验的功效诊断
# ======================================================================
def diagnose_fixed_sequence_power(
    gate_values: np.ndarray,
    alpha: float = 0.10,
    delta: float = 0.05,
    lambda_grid: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """诊断固定序列检验会不会在第一步就因样本不足而中断。

    **为什么需要这个函数**：§11.2 字面写的固定序列是"λ 从大到小"。
    但在最大的 λ 上，标定折里"被接受检索"的样本往往只有个位数 ——
    此时即便经验风险为 0，Hoeffding–Bentkus p 值仍约 0.9（``exp(−2n α²)``
    在 n=5、α=0.1 时是 0.90），远大于 δ=0.05。固定序列一旦第一步失败
    就整体中断，于是 λ* 恒为 ``None``，H3 无法被检验 —— 而这与
    "真的不存在安全工作点"是两回事，必须区分开。

    本项目标定折约占 dev 的 20%，规模在 10² 量级，正落在这个区间里。

    Args:
        gate_values: ``(N,)`` 标定折门控值。
        alpha: 风险上界。
        delta: FWER 水平。
        lambda_grid: 候选网格。

    Returns:
        含 ``min_n_to_reject``（即便风险为 0 也需要的最小样本数）、
        各 λ 的可达性，以及推荐的多重校正方式。
    """
    grid = list(lambda_grid) if lambda_grid is not None else [round(i * 0.01, 2) for i in range(100)]
    gate_values = np.asarray(gate_values, dtype=np.float64)

    # 风险为 0 时，Hoeffding p 值 = exp(−2nα²) ≤ δ 需要 n ≥ ln(1/δ)/(2α²)
    min_n = int(math.ceil(math.log(1.0 / delta) / (2.0 * alpha ** 2)))

    rows: List[Dict[str, float]] = []
    for lam in sorted(grid, reverse=True):
        n_accepted = int((gate_values >= lam).sum())
        rows.append({
            "lambda": float(lam),
            "n_accepted": float(n_accepted),
            "can_reject_even_at_zero_risk": float(
                hoeffding_bentkus_pvalue(0.0, n_accepted, alpha) <= delta
            ),
        })

    first = rows[0] if rows else {"can_reject_even_at_zero_risk": 0.0, "lambda": float("nan")}
    first_blocked = not bool(first["can_reject_even_at_zero_risk"])
    n_reachable = sum(1 for r in rows if r["can_reject_even_at_zero_risk"])

    result = {
        "min_n_to_reject_at_zero_risk": min_n,
        "first_lambda": first["lambda"],
        "first_lambda_n_accepted": first.get("n_accepted", 0.0),
        "fixed_sequence_blocked_at_first_step": first_blocked,
        "n_lambda_with_power": n_reachable,
        "n_lambda_total": len(rows),
        "recommendation": (
            "改用 bonferroni（§11.2 允许）或用内层折预先指定 lambda_order"
            if first_blocked else "固定序列可用"
        ),
        "per_lambda": rows,
    }
    if first_blocked:
        _LOGGER.warning(
            "固定序列检验会在第一步（λ=%.2f，n=%d）因样本不足中断：风险为 0 时也需 n ≥ %d。%s",
            first["lambda"], int(first.get("n_accepted", 0)), min_n, result["recommendation"],
        )
    return result


def fixed_sequence_order_from_inner_fold(
    inner_gate_values: np.ndarray,
    inner_loss_sparc: np.ndarray,
    inner_loss_base: np.ndarray,
    alpha: float = 0.10,
    epsilon: float = 0.01,
    lambda_grid: Optional[Sequence[float]] = None,
) -> List[float]:
    """在**内层模型选择折**上给出固定序列的检验顺序。

    固定序列检验只要求顺序是**预先指定**的 —— 不要求它是 λ 的大小顺序。
    因此用一个与标定折互斥的折（§11.2 的 dev 内层 20%）来排序，
    既保留了有限样本、分布无关的保证，又把最可能通过检验的 λ 排在前面，
    解决 :func:`diagnose_fixed_sequence_power` 指出的功效问题。

    排序依据：内层折上的 HB p 值从小到大（最有把握的先测）。

    Args:
        inner_gate_values: ``(N_inner,)`` 内层折门控值。
        inner_loss_sparc: ``(N_inner,)``。
        inner_loss_base: ``(N_inner,)``。
        alpha: 风险上界。
        epsilon: 伤害裕度。
        lambda_grid: 候选网格。

    Returns:
        λ 的检验顺序。**必须在看标定折之前算好并冻结**。
    """
    grid = list(lambda_grid) if lambda_grid is not None else [round(i * 0.01, 2) for i in range(100)]
    scored: List[Tuple[float, float]] = []
    for lam in grid:
        risk, n_accepted, _ = risk_at_lambda(
            inner_gate_values, inner_loss_sparc, inner_loss_base, lam, epsilon
        )
        scored.append((hoeffding_bentkus_pvalue(risk, n_accepted, alpha), float(lam)))
    # p 值升序；平手时取较小的 λ（覆盖率更高）
    scored.sort(key=lambda t: (t[0], t[1]))
    order = [lam for _, lam in scored]
    _LOGGER.info("固定序列顺序已由内层折确定，前 5 个 λ：%s", order[:5])
    return order
