"""Reranker-lite (§8.3.2)。**参数量 15,151（原 1,056,783，压缩 70×）。**

**为什么从 1927 维降到 103 维**：在"同靶点 + 同 endpoint"硬过滤下，
原 ``x^rank`` 里 ``c_t``(256) + ``c_i``(256) + ``|c_t−c_i|``(256)
+ ``s_context``(1) + ``c_assay``(1) + ``c_endpoint``(1) = **771 维恒为常数（40.0%）**，
``W_r1`` 有 512×771 ≈ 394,752 个权重连在常数输入上。

v1.0 把上下文块塌缩为 ``s_ctx`` 一个标量（多靶点时携带 ortholog 距离，
单靶点时恒 1 并在 manifest 中标记为 dead）。

**排序目标的副作用修正**（§8.3.2）：``L_rank`` 用 ``ρ_i = −|y_i − y_q|``
会教重排器把活性悬崖邻居排到后面，从而抹平门控赖以工作的冲突特征。
因此冲突类门控特征在 **K₀=256 候选池** 上算，支持类在 Top-K=16 上算 ——
这个分工由 :mod:`sparc.retrieval.features` 实现，本模块只负责保证
两个池的分数都被返回。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass
class RerankOutput:
    """重排器输出。"""

    scores: Tensor          # (B, K0) 候选池内全部分数
    topk_scores: Tensor     # (B, K)  Top-K 分数
    topk_index: Tensor      # (B, K)  Top-K 在候选池中的下标
    ligand_proj_query: Tensor   # (B, 32)    u_q
    ligand_proj_cand: Tensor    # (B, K0, 32) u_i


class RerankerLite(nn.Module):
    """剪枝后的重排器。

    ``x^rank = [u_q ⊙ u_i (32); |u_q − u_i| (32); a_i (32);
                s_tan, s_pharm, s_emb, m_i, s_ctx, c_assay, c_endpoint (7)] ∈ R^103``
    """

    N_SCALAR_FEATURES = 7   # s_tan, s_pharm, s_emb, m_i, s_ctx, c_assay, c_endpoint

    def __init__(
        self,
        d_query_repr: int = 256,
        d_proj: int = 32,
        d_align: int = 32,
        d_hidden: int = 64,
    ) -> None:
        """
        Args:
            d_query_repr: ``v_q = [z_q'; h_q]`` 的维度（256）。
            d_proj: ``W_z`` 输出维（32），query 与候选**共享**该投影。
            d_align: 对齐摘要维（32）。
            d_hidden: ``W_r1`` 隐藏维（64）。
        """
        super().__init__()
        self.ligand_proj = nn.Linear(d_query_repr, d_proj)      # W_z: 256→32   8,224
        d_rank = 3 * d_proj + self.N_SCALAR_FEATURES            # 103
        self.input_norm = nn.LayerNorm(d_rank)                  #                  206
        self.hidden = nn.Linear(d_rank, d_hidden)               # W_r1          6,656
        self.score = nn.Linear(d_hidden, 1)                     # w_r2             65
        self.d_rank = d_rank

    # ------------------------------------------------------------------
    def build_rank_features(
        self,
        query_repr: Tensor,
        cand_repr: Tensor,
        align_summary: Tensor,
        scalar_features: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """拼出 ``x^rank``。

        Args:
            query_repr: ``(B, 256)`` ``v_q``。
            cand_repr: ``(B, K0, 256)`` ``v_i``。
            align_summary: ``(B, K0, 32)`` ``a_i``。
            scalar_features: ``(B, K0, 7)`` 七个标量特征，顺序固定为
                ``[s_tan, s_pharm, s_emb, m_i, s_ctx, c_assay, c_endpoint]``。

        Returns:
            ``(x_rank, u_q, u_i)``。
        """
        u_q = self.ligand_proj(query_repr)                       # (B, 32)
        u_i = self.ligand_proj(cand_repr)                        # (B, K0, 32)
        u_q_expanded = u_q.unsqueeze(1).expand_as(u_i)
        x_rank = torch.cat([
            u_q_expanded * u_i,
            (u_q_expanded - u_i).abs(),
            align_summary,
            scalar_features,
        ], dim=-1)                                               # (B, K0, 103)
        return x_rank, u_q, u_i

    def forward(
        self,
        query_repr: Tensor,
        cand_repr: Tensor,
        align_summary: Tensor,
        scalar_features: Tensor,
        candidate_mask: Optional[Tensor] = None,
        k_top: int = 16,
    ) -> RerankOutput:
        """前向：给候选池打分并取 Top-K。

        Args:
            query_repr: ``(B, 256)``。
            cand_repr: ``(B, K0, 256)``。
            align_summary: ``(B, K0, 32)``。
            scalar_features: ``(B, K0, 7)``。
            candidate_mask: ``(B, K0)`` 有效候选掩码（候选不足 K₀ 时补齐用）。
            k_top: Top-K（冻结为 16）。

        Returns:
            :class:`RerankOutput`。**同时返回 K₀ 全池分数与 Top-K**，
            因为门控的冲突类特征必须在 K₀ 上计算 (§8.3.2)。
        """
        x_rank, u_q, u_i = self.build_rank_features(query_repr, cand_repr, align_summary, scalar_features)
        hidden = torch.nn.functional.gelu(self.hidden(self.input_norm(x_rank)))
        scores = self.score(hidden).squeeze(-1)                  # (B, K0)

        if candidate_mask is not None:
            scores = scores.masked_fill(~candidate_mask, torch.finfo(scores.dtype).min)

        k = min(k_top, scores.shape[-1])
        topk_scores, topk_index = torch.topk(scores, k=k, dim=-1)
        return RerankOutput(
            scores=scores, topk_scores=topk_scores, topk_index=topk_index,
            ligand_proj_query=u_q, ligand_proj_cand=u_i,
        )

    def n_parameters(self) -> int:
        """可训练参数量（预算核对：15,151）。"""
        return sum(p.numel() for p in self.parameters())

    def parameter_breakdown(self) -> Dict[str, int]:
        """分模块参数量。"""
        return {
            "ligand_proj": sum(p.numel() for p in self.ligand_proj.parameters()),
            "input_norm": sum(p.numel() for p in self.input_norm.parameters()),
            "hidden": sum(p.numel() for p in self.hidden.parameters()),
            "score": sum(p.numel() for p in self.score.parameters()),
            "total": self.n_parameters(),
        }
