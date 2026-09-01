"""排序损失 L_rank (§8.3.2, §10.2)。

目标 ``ρ_i = −|y_i − y_q|``：邻居标签与查询标签越接近，排序应越靠前。

**已知副作用（§8.3.2 明确记录）**：这个目标会教重排器把活性悬崖邻居
排到后面，从而抹平门控赖以工作的冲突特征。v1.0 的对策不是改目标，
而是把**冲突类门控特征改到 K₀=256 候选池上计算**（见
:mod:`sparc.retrieval.features`），并新增第 25 维
``Var_K(y)/Var_{K₀}(y)`` 显式度量"重排器抹平了多少标签方差"。

**分数尺度不可辨识**：``L_rank`` 是平移不变的（listwise softmax 对
分数整体加常数不变），因此 raw 分数的 max/mean/std/gap 跨 fold/seed
不可比 —— 这就是 §8.3.5 删掉旧第 1–4 维的原因。门控只用 z-score 分位数。
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def rank_targets(neighbor_labels: Tensor, query_label: Tensor) -> Tensor:
    """计算排序目标 ``ρ_i = −|y_i − y_q|``。

    Args:
        neighbor_labels: ``(B, K)`` 邻居标签。
        query_label: ``(B,)`` 查询标签。

    Returns:
        ``(B, K)``。
    """
    return -(neighbor_labels - query_label.unsqueeze(-1)).abs()


def listwise_rank_loss(
    scores: Tensor,
    targets: Tensor,
    mask: Optional[Tensor] = None,
    temperature: float = 1.0,
) -> Tensor:
    """ListNet 风格的 listwise 排序损失。

    ``L = −Σ_i softmax(ρ/τ)_i · log softmax(s)_i``（目标分布与预测分布的交叉熵）。

    Args:
        scores: ``(B, K)`` 重排分数。
        targets: ``(B, K)`` 排序目标 ρ。
        mask: ``(B, K)`` 有效候选掩码。
        temperature: 目标分布温度。

    Returns:
        标量损失。
    """
    neg_inf = torch.finfo(scores.dtype).min
    if mask is not None:
        scores = scores.masked_fill(~mask, neg_inf)
        targets = targets.masked_fill(~mask, neg_inf)

    target_dist = torch.softmax(targets / temperature, dim=-1)
    log_pred = torch.log_softmax(scores, dim=-1)
    per_query = -(target_dist * log_pred).sum(dim=-1)

    if mask is not None:
        valid = mask.any(dim=-1)
        if not valid.any():
            return scores.sum() * 0.0        # 保持计算图连通，梯度为 0
        return per_query[valid].mean()
    return per_query.mean()


def pairwise_rank_loss(
    scores: Tensor,
    targets: Tensor,
    mask: Optional[Tensor] = None,
    margin: float = 0.0,
) -> Tensor:
    """RankNet 风格的成对损失 —— 消融对照 (Table 2 的 "−Reranker" 行需要)。

    Args:
        scores: ``(B, K)``。
        targets: ``(B, K)``。
        mask: ``(B, K)``。
        margin: 铰链间隔。

    Returns:
        标量损失。
    """
    score_diff = scores.unsqueeze(-1) - scores.unsqueeze(-2)         # (B, K, K)
    target_diff = targets.unsqueeze(-1) - targets.unsqueeze(-2)
    pair_mask = target_diff > 0
    if mask is not None:
        pair_mask = pair_mask & mask.unsqueeze(-1) & mask.unsqueeze(-2)
    if not pair_mask.any():
        return scores.sum() * 0.0
    losses = torch.nn.functional.softplus(-(score_diff - margin))
    return losses[pair_mask].mean()
