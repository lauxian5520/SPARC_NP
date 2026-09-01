"""SafeCoverage —— 主标指标 (§12.1)。

```
c* = max { c ∈ [0,1] : UCB_95%( E[ ℓ_SPARC − ℓ_base | g ≥ Q_{1−c}(g) ] ) ≤ 0 }
```

**在不造成任何统计上可检出的平均伤害的前提下，模型能安全接受检索的
最大样本比例。**

设计要点：
* 一个标量，不需要挑工作点；
* **覆盖率趋 0 时 c* 趋 0 ⇒ 不可通过"少检索"平凡满足**。
  旧方案的"伤害率下降 30% + 全样本非劣"两条都能靠"几乎不检索"满足；
* 朴素检索的 c* 通常为 0，Tanimoto 门控为小正数，Oracle 门控给出上界；
* 在 Risk–Coverage 图上就是曲线与零伤害线的交点。

**关于 c → 0 的极限**：覆盖率极小时，条件均值的 bootstrap UCB 会因
样本太少而变宽、超过 0，于是 c* 不会被"少检索"推高。本实现额外要求
覆盖区间内至少有 ``min_samples`` 个样本，把这一点写死而不是依赖运气。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from sparc.common.logging_utils import get_logger

_LOGGER = get_logger(__name__)


@dataclass
class SafeCoverageResult:
    """SafeCoverage 及其 Risk–Coverage 曲线。"""

    c_star: float
    curve: List[Dict[str, float]] = field(default_factory=list)
    ci_level: float = 0.95
    bootstrap_n: int = 10000
    min_samples: int = 20
    method: str = "bootstrap_percentile"

    def to_dict(self) -> Dict[str, Any]:
        """转成可写报告的字典。"""
        return {
            "safe_coverage": round(self.c_star, 4),
            "ci_level": self.ci_level,
            "bootstrap_n": self.bootstrap_n,
            "min_samples": self.min_samples,
            "method": self.method,
            "curve": self.curve,
        }

    def curve_arrays(self) -> Dict[str, np.ndarray]:
        """便于画 Figure 1 的数组视图。"""
        return {
            key: np.array([row[key] for row in self.curve])
            for key in ("coverage", "mean_delta_loss", "ucb", "n_covered", "gate_threshold")
        }


def _bootstrap_upper_bound(
    values: np.ndarray,
    ci_level: float,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> float:
    """条件均值的 bootstrap 上置信界。

    Args:
        values: ``(n,)`` 覆盖区间内的 ``ℓ_SPARC − ℓ_base``。
        ci_level: 置信水平（0.95）。
        n_bootstrap: 重抽样次数。
        rng: 随机源。

    Returns:
        单侧上界（percentile 法）。样本数为 0 时返回 ``+inf``，
        使空覆盖不可能被判为"安全" —— 这是 c* 不可被平凡满足的实现基础。
    """
    n = values.size
    if n == 0:
        return float("inf")
    if n == 1:
        return float(values[0]) if values[0] > 0 else float("inf")
    indices = rng.integers(0, n, size=(n_bootstrap, n))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, ci_level))


def risk_coverage_curve(
    gate_values: np.ndarray,
    loss_sparc: np.ndarray,
    loss_base: np.ndarray,
    coverage_grid_step: float = 0.01,
    ci_level: float = 0.95,
    bootstrap_n: int = 10000,
    min_samples: int = 20,
    seed: int = 42,
) -> List[Dict[str, float]]:
    """计算 Risk–Coverage 曲线（Figure 1 的数据）。

    对每个覆盖率 ``c``，取门控值最高的 ``c`` 比例样本，计算
    ``E[ℓ_SPARC − ℓ_base]`` 及其 95% 上置信界。

    Args:
        gate_values: ``(N,)``。
        loss_sparc: ``(N,)``。
        loss_base: ``(N,)``。
        coverage_grid_step: 覆盖率网格步长。
        ci_level: 置信水平。
        bootstrap_n: bootstrap 次数。
        min_samples: 覆盖区间内的最小样本数；低于此值一律判为不安全。
        seed: 随机种子。

    Returns:
        逐覆盖率的记录列表。
    """
    gate_values = np.asarray(gate_values, dtype=np.float64)
    delta_loss = np.asarray(loss_sparc, dtype=np.float64) - np.asarray(loss_base, dtype=np.float64)
    n_total = gate_values.size
    if n_total == 0:
        return []

    rng = np.random.default_rng(seed)
    order = np.argsort(-gate_values)          # 门控值从高到低
    sorted_delta = delta_loss[order]
    sorted_gate = gate_values[order]

    curve: List[Dict[str, float]] = []
    n_steps = int(round(1.0 / coverage_grid_step))
    for step in range(1, n_steps + 1):
        coverage = step * coverage_grid_step
        n_covered = int(round(coverage * n_total))
        if n_covered == 0:
            continue
        covered = sorted_delta[:n_covered]
        ucb = (
            _bootstrap_upper_bound(covered, ci_level, bootstrap_n, rng)
            if n_covered >= min_samples else float("inf")
        )
        curve.append({
            "coverage": round(coverage, 4),
            "n_covered": float(n_covered),
            "gate_threshold": float(sorted_gate[n_covered - 1]),
            "mean_delta_loss": float(covered.mean()),
            "ucb": ucb,
            "harm_rate": float((covered > 0).mean()),
        })
    return curve


def safe_coverage(
    gate_values: np.ndarray,
    loss_sparc: np.ndarray,
    loss_base: np.ndarray,
    coverage_grid_step: float = 0.01,
    ci_level: float = 0.95,
    bootstrap_n: int = 10000,
    min_samples: int = 20,
    seed: int = 42,
) -> SafeCoverageResult:
    """计算 ``c*`` (§12.1)。

    Args:
        gate_values: ``(N,)`` 门控值 ``g``。
        loss_sparc: ``(N,)`` SPARC-NP 逐样本损失。
        loss_base: ``(N,)`` Base 逐样本损失。
        coverage_grid_step: 覆盖率网格步长。
        ci_level: 置信水平（0.95）。
        bootstrap_n: bootstrap 次数。
        min_samples: 最小样本数守卫。
        seed: 随机种子。

    Returns:
        :class:`SafeCoverageResult`。``c_star = 0`` 表示不存在
        统计上无伤害的正覆盖率 —— 朴素检索通常就是这个结果。
    """
    curve = risk_coverage_curve(
        gate_values, loss_sparc, loss_base, coverage_grid_step,
        ci_level, bootstrap_n, min_samples, seed,
    )
    safe = [row["coverage"] for row in curve if row["ucb"] <= 0.0]
    c_star = max(safe) if safe else 0.0
    _LOGGER.info(
        "SafeCoverage = %.3f（%d/%d 个覆盖率点满足 UCB ≤ 0）",
        c_star, len(safe), len(curve),
    )
    return SafeCoverageResult(
        c_star=c_star, curve=curve, ci_level=ci_level,
        bootstrap_n=bootstrap_n, min_samples=min_samples,
    )


def oracle_gate_values(loss_sparc: np.ndarray, loss_base: np.ndarray) -> np.ndarray:
    """Oracle 门控 —— **不可部署的上界**（Table 1 与 Figure 1 的第四条曲线）。

    Args:
        loss_sparc: ``(N,)``。
        loss_base: ``(N,)``。

    Returns:
        ``(N,)`` 用 ``−(ℓ_SPARC − ℓ_base)`` 排序的伪门控值：
        检索帮助越大排越前。它给出 SafeCoverage 的理论上界，
        任何可部署门控都不应超过它。
    """
    return -(np.asarray(loss_sparc, dtype=np.float64) - np.asarray(loss_base, dtype=np.float64))


def tanimoto_gate_values(top1_tanimoto: np.ndarray) -> np.ndarray:
    """Tanimoto 门控基线 —— Table 1 与 H2 判据 (a) 的对照。

    Args:
        top1_tanimoto: ``(N,)`` 每个查询的 Top-1 Tanimoto。

    Returns:
        ``(N,)``，直接用 Top-1 Tanimoto 当门控值。
    """
    return np.asarray(top1_tanimoto, dtype=np.float64)
