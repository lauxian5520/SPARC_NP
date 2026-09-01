"""效用/伤害/校准损失与 L_R 合成 (§10.2)。

```
L_R = L_task^final
    + λ_rank    · L_rank
    + λ_utility · L_utility-gate
    + λ_cal     · L_cal
    + 1e-5 · ||Θ_R||²

L_utility-gate = L_utility + 0.5 · L_harm
L_harm = (1/B) Σ_i  g_i · max(0, −d_i)
```

**``d_i`` 是四个组件唯一的耦合量** (§11.3)::

    d_i = ℓ(base) − ℓ(base + g̃Δ)     # 正号 = 检索有帮助

* 拿掉冻结的 Base ⇒ ``d_i`` 无定义 ⇒ 效用标签不存在 ⇒ 门控无法训练；
* 把残差换成第二个完整预测器 ⇒ ``g=0`` 不再退化回 Base ⇒
  ``d_i`` 不再度量"检索的边际效应"；
* 拿掉门控 ⇒ 没有逐样本风险可控 ⇒ LTT 无对象。

**``L_harm`` 的方向**：``g_i · max(0, −d_i)`` 只在"检索有害且门控开着"时
产生梯度，把 ``g`` 往下压。它不奖励"少检索" —— 那由 ``L_task`` 与
``SafeCoverage`` 的定义共同约束。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
from torch import Tensor


def marginal_effect(loss_base: Tensor, loss_retrieval: Tensor) -> Tensor:
    """计算 ``d_i = ℓ(base) − ℓ(base + g̃Δ)``。

    Args:
        loss_base: ``(B,)`` 只用 Base 的逐样本损失。
        loss_retrieval: ``(B,)`` 施加门控残差后的逐样本损失。

    Returns:
        ``(B,)``；**正值表示检索有帮助**。
    """
    return loss_base - loss_retrieval


def utility_labels(
    loss_base: Tensor,
    loss_retrieval: Tensor,
    epsilon: float = 0.01,
) -> Tensor:
    """二元效用标签 ``u_i = 1[ℓ(retr) < ℓ(base) − ε]``（H2 的判别目标）。

    Args:
        loss_base: ``(B,)``。
        loss_retrieval: ``(B,)``。
        epsilon: 判定裕度（冻结为 0.01）。

    Returns:
        ``(B,)`` float，0/1。
    """
    return (loss_retrieval < loss_base - epsilon).to(loss_base.dtype)


def utility_loss(gate: Tensor, labels: Tensor, weights: Optional[Tensor] = None) -> Tensor:
    """门控对效用标签的二元交叉熵。

    Args:
        gate: ``(B,)`` 连续 ``g``（**训练期必须是连续的**，§8.3.5）。
        labels: ``(B,)`` 效用标签。
        weights: ``(B,)`` 样本权重（类别不平衡时用）。

    Returns:
        标量损失。
    """
    g = gate.clamp(min=1e-6, max=1.0 - 1e-6)
    bce = -(labels * torch.log(g) + (1.0 - labels) * torch.log(1.0 - g))
    return (bce * weights).mean() if weights is not None else bce.mean()


def harm_loss(gate: Tensor, marginal: Tensor) -> Tensor:
    """``L_harm = (1/B) Σ_i g_i · max(0, −d_i)``。

    Args:
        gate: ``(B,)`` 连续 ``g``。
        marginal: ``(B,)`` ``d_i``。

    Returns:
        标量损失。只在"检索有害（d_i < 0）且门控开着"时非零。
    """
    return (gate * torch.clamp(-marginal, min=0.0)).mean()


def calibration_loss(
    mu: Tensor,
    log_var: Tensor,
    y: Tensor,
    n_bins: int = 10,
) -> Tensor:
    """校准损失 ``L_cal``：预测区间的经验覆盖率应匹配名义覆盖率。

    对每个名义分位 ``p``，计算 ``|经验覆盖率 − p|`` 并平均。用可微的
    sigmoid 软指示替代硬指示，使梯度能回传到 ``log_var``。

    Args:
        mu: ``(B,)``。
        log_var: ``(B,)``。
        y: ``(B,)``。
        n_bins: 分位数个数。

    Returns:
        标量损失。
    """
    sigma = torch.exp(0.5 * log_var).clamp(min=1e-6)
    z = ((y - mu) / sigma).abs()
    device = z.device
    # 标准正态的双侧分位点（对应覆盖率 p）
    levels = torch.linspace(1.0 / (n_bins + 1), n_bins / (n_bins + 1), n_bins, device=device)
    quantiles = torch.sqrt(torch.tensor(2.0, device=device)) * torch.erfinv(levels)
    # 软指示：sigmoid((q − z)/温度)，温度随 batch 规模缩放
    empirical = torch.sigmoid((quantiles.unsqueeze(-1) - z.unsqueeze(0)) * 10.0).mean(dim=-1)
    return (empirical - levels).abs().mean()


def retrieval_total_loss(
    task_loss: Tensor,
    rank_loss: Tensor,
    gate: Tensor,
    loss_base: Tensor,
    loss_retrieval: Tensor,
    mu: Tensor,
    log_var: Tensor,
    y: Tensor,
    lambda_rank: float = 0.20,
    lambda_utility: float = 0.50,
    lambda_cal: float = 0.10,
    harm_weight: float = 0.5,
    epsilon: float = 0.01,
    l2_penalty: Optional[Tensor] = None,
    l2_weight: float = 1e-5,
) -> Dict[str, Tensor]:
    """合成 ``L_R`` (§10.2)。

    Args:
        task_loss: 标量，``L_task^final``（Tobit）。
        rank_loss: 标量，``L_rank``。
        gate: ``(B,)`` 连续 ``g``。
        loss_base: ``(B,)`` Base 逐样本损失。
        loss_retrieval: ``(B,)`` 检索后逐样本损失。
        mu: ``(B,)`` 最终预测均值（校准损失用）。
        log_var: ``(B,)`` 最终对数方差。
        y: ``(B,)`` 观测值。
        lambda_rank: 冻结默认 0.20（网格 {0.05,0.10,0.20,0.40}）。
        lambda_utility: 冻结默认 0.50（网格 {0.25,0.50,1.00}）。
        lambda_cal: 冻结默认 0.10（网格 {0.05,0.10,0.20}）。
        harm_weight: ``L_utility-gate = L_utility + 0.5·L_harm``。
        epsilon: 效用标签裕度。
        l2_penalty: ``||Θ_R||²``；``None`` 时跳过（改由优化器 weight_decay 承担）。
        l2_weight: L2 系数（1e-5）。

    Returns:
        ``{"total": .., "task": .., "rank": .., "utility": .., "harm": .., "cal": ..}``，
        各分项均为标量张量，便于逐项写进 MetricMonitor。
    """
    marginal = marginal_effect(loss_base, loss_retrieval)
    labels = utility_labels(loss_base, loss_retrieval, epsilon)
    l_utility = utility_loss(gate, labels)
    l_harm = harm_loss(gate, marginal)
    l_cal = calibration_loss(mu, log_var, y)

    total = (task_loss
             + lambda_rank * rank_loss
             + lambda_utility * (l_utility + harm_weight * l_harm)
             + lambda_cal * l_cal)
    if l2_penalty is not None:
        total = total + l2_weight * l2_penalty

    return {
        "total": total,
        "task": task_loss.detach(),
        "rank": rank_loss.detach(),
        "utility": l_utility.detach(),
        "harm": l_harm.detach(),
        "cal": l_cal.detach(),
        "utility_positive_rate": labels.mean().detach(),
        "marginal_mean": marginal.mean().detach(),
    }


def utility_labels_numpy(loss_base: np.ndarray, loss_retrieval: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
    """numpy 版效用标签 —— Stage 3 的交叉拟合门控用 numpy 拟合 logistic。"""
    return (np.asarray(loss_retrieval) < np.asarray(loss_base) - epsilon).astype(np.float64)
