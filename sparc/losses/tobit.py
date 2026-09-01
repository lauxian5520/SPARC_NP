"""Tobit（审查高斯）似然 (§10.1)。

```
none  : ℓ = ½[(y−μ)²/σ² + log σ²]
right : ℓ = −log Φ((μ−y)/σ)      # 实测 IC50 > y，真值更大
left  : ℓ = −log Φ((y−μ)/σ)
```

**为什么这是必须的**：``talk.md`` 与 NaFM 都把 censored 记录当点值处理
（NaFM 的样本量与本项目"含 censored"口径吻合，见事实 A 表）。
PTP-1B 有 17.5% 的记录是 censored，AChE 11.7%。这是一个必须修正的
方法学缺陷，也是本项目相对 NaFM 的一个**干净的技术改进**。

数值稳定性：``log Φ(x)`` 在 x 很负时会下溢到 -inf。这里用
``torch.special.log_ndtr``，它在整个实轴上数值稳定（内部对左尾
用渐近展开），比 ``log(0.5*erfc(-x/√2))`` 可靠得多。
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

# 与 CensorFlag 对应的整数编码（数据管线里统一用它进张量）
CENSOR_NONE = 0
CENSOR_LEFT = 1
CENSOR_RIGHT = 2


def _log_ndtr(x: Tensor) -> Tensor:
    """数值稳定的 ``log Φ(x)``。"""
    if hasattr(torch.special, "log_ndtr"):
        return torch.special.log_ndtr(x)
    # 退路：老版本 torch 没有 log_ndtr
    return torch.log(torch.clamp(0.5 * torch.erfc(-x / torch.sqrt(torch.tensor(2.0, device=x.device))), min=1e-30))


def per_sample_loss(
    mu: Tensor,
    log_var: Tensor,
    y: Tensor,
    censor_code: Optional[Tensor] = None,
    min_log_var: float = -10.0,
    max_log_var: float = 10.0,
) -> Tensor:
    """逐样本 Tobit 负对数似然。

    **逐样本**是关键：``d_i = ℓ(base + g̃Δ) − ℓ(base)`` 是本方法四个组件
    的唯一耦合量 (§11.3)，它要求损失在样本级可比，不能只有 batch 平均。

    Args:
        mu: ``(B,)`` 预测均值。
        log_var: ``(B,)`` 预测对数方差。
        y: ``(B,)`` 观测值（pIC50）。
        censor_code: ``(B,)`` long，取值 0/1/2（none/left/right）；
            ``None`` 时全部按 none 处理。
        min_log_var: ``log_var`` 下界。
        max_log_var: 上界。

    Returns:
        ``(B,)`` 逐样本损失。
    """
    log_var = log_var.clamp(min=min_log_var, max=max_log_var)
    sigma = torch.exp(0.5 * log_var)

    # 未审查：高斯 NLL（去掉常数项 ½log 2π，对优化与 d_i 都无影响）
    nll_none = 0.5 * (((y - mu) ** 2) / torch.exp(log_var) + log_var)

    if censor_code is None:
        return nll_none

    z_right = (mu - y) / sigma      # P(真值 > y)
    z_left = (y - mu) / sigma       # P(真值 < y)
    nll_right = -_log_ndtr(z_right)
    nll_left = -_log_ndtr(z_left)

    loss = torch.where(censor_code == CENSOR_RIGHT, nll_right, nll_none)
    loss = torch.where(censor_code == CENSOR_LEFT, nll_left, loss)
    return loss


def censored_gaussian_nll(
    mu: Tensor,
    log_var: Tensor,
    y: Tensor,
    censor_code: Optional[Tensor] = None,
    reduction: str = "mean",
    weights: Optional[Tensor] = None,
) -> Tensor:
    """Tobit 损失（带归约）。

    Args:
        mu: ``(B,)``。
        log_var: ``(B,)``。
        y: ``(B,)``。
        censor_code: ``(B,)`` long。
        reduction: ``"mean"`` / ``"sum"`` / ``"none"``。
        weights: ``(B,)`` 样本权重（如靶点级再平衡）。

    Returns:
        标量或 ``(B,)``。

    Raises:
        ValueError: 未知的 reduction。
    """
    loss = per_sample_loss(mu, log_var, y, censor_code)
    if weights is not None:
        loss = loss * weights
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"未知 reduction '{reduction}'")


def squared_error_loss(mu: Tensor, y: Tensor, censor_code: Optional[Tensor] = None) -> Tensor:
    """逐样本平方误差 —— 敏感性分析用（"丢弃 censored"口径，§10.1 的 P1）。

    Args:
        mu: ``(B,)``。
        y: ``(B,)``。
        censor_code: 提供时，审查样本的损失置 0（等价于从平均里剔除）。

    Returns:
        ``(B,)``。
    """
    err = (y - mu) ** 2
    if censor_code is not None:
        err = err * (censor_code == CENSOR_NONE).to(err.dtype)
    return err
