"""Evidence-lite (§8.3.3)。**参数量 6,688（原 180,128，压缩 27×）。**

```
φ(y_i) = [y_i^std, 1[label present], 1[left cens], 1[right cens], 1[domain=drug]] ∈ R^5
e_i^raw = [u_i(32); a_i(32); e_y(16); e_a(8); t_i(8)] ∈ R^96
c_R = Σ_i α_i e_i,   α = softmax(s_i / τ_rank)
```

两个设计要点：

1. ``y_i^std`` 用 **NP fold-train 的均值/标准差**归一化，**不用记忆库统计量**。
   用记忆库统计量会让残差的尺度随记忆库组成漂移，而记忆库视图是逐 fold
   变化的 —— 那样 ``Δ`` 在不同 fold 之间不可比。
2. 第 5 维 ``1[domain=drug]`` 是 v1.0 新增：让残差能区分"这条证据来自药物库"
   与"来自天然产物"。``C_FORCED_RETRIEVAL`` 诊断 (§13.4) 与消融中
   会同时出现两种来源。

``t_i`` 是 8 维**无参数**的候选元特征（其 8 维预算已含在 ``d_evidence_raw=96``
里，且 §8.3.3 的参数表未给它任何投影矩阵）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn


# t_i 的 8 个分量（顺序固定，写进维度契约测试）
CANDIDATE_META_FIELDS = (
    "tanimoto", "emb_similarity", "match_confidence", "rank_norm",
    "score_zscore", "assay_compatible", "is_censored", "n_records_log",
)


@dataclass
class EvidenceOutput:
    """证据聚合输出。"""

    context: Tensor          # (B, 64)  c_R
    attention: Tensor        # (B, K)   α_i
    raw_evidence: Tensor     # (B, K, 96) e_i^raw
    pooled_raw: Tensor       # (B, 96)  Σ α_i e_i^raw —— 不确定性头的输入之一


class EvidenceLite(nn.Module):
    """证据编码与注意力聚合。**参数量 6,688。**"""

    N_LABEL_FEATURES = 5     # φ(y_i)

    def __init__(
        self,
        d_proj: int = 32,
        d_align: int = 32,
        d_label_emb: int = 16,
        n_assay_family: int = 32,
        d_assay_emb: int = 8,
        d_meta: int = 8,
        d_context: int = 64,
        tau_rank: float = 1.0,
    ) -> None:
        """
        Args:
            d_proj: ``u_i`` 维（32）。
            d_align: ``a_i`` 维（32）。
            d_label_emb: ``e_y`` 维（16）。
            n_assay_family: assay family 数（32）。
            d_assay_emb: ``e_a`` 维（8）。
            d_meta: ``t_i`` 维（8，无参数）。
            d_context: ``c_R`` 维（64）。
            tau_rank: 注意力温度。
        """
        super().__init__()
        self.label_proj = nn.Linear(self.N_LABEL_FEATURES, d_label_emb)      # W_y      96
        self.assay_embedding = nn.Embedding(n_assay_family, d_assay_emb)     #         256
        d_raw = d_proj + d_align + d_label_emb + d_assay_emb + d_meta        # 96
        self.evidence_proj = nn.Linear(d_raw, d_context)                     # W_e   6,208
        self.evidence_norm = nn.LayerNorm(d_context)                         #         128
        self.tau_rank = tau_rank
        self.d_raw = d_raw
        self.d_context = d_context

    # ------------------------------------------------------------------
    @staticmethod
    def build_label_features(
        pactivity_std: Tensor,
        label_present: Tensor,
        left_censored: Tensor,
        right_censored: Tensor,
        is_drug_domain: Tensor,
    ) -> Tensor:
        """拼出 ``φ(y_i) ∈ R^5``。

        Args:
            pactivity_std: ``(B, K)`` 用 **NP fold-train** 统计量标准化后的标签。
            label_present: ``(B, K)`` 是否有标签。
            left_censored: ``(B, K)``。
            right_censored: ``(B, K)``。
            is_drug_domain: ``(B, K)`` 第 5 维 ``1[domain=drug]``。

        Returns:
            ``(B, K, 5)``。
        """
        return torch.stack([
            pactivity_std * label_present,     # 无标签时置 0，避免噪声进入
            label_present,
            left_censored,
            right_censored,
            is_drug_domain,
        ], dim=-1)

    def forward(
        self,
        ligand_proj_cand: Tensor,
        align_summary: Tensor,
        label_features: Tensor,
        assay_family: Tensor,
        candidate_meta: Tensor,
        rank_scores: Tensor,
        candidate_mask: Optional[Tensor] = None,
    ) -> EvidenceOutput:
        """前向。

        Args:
            ligand_proj_cand: ``(B, K, 32)`` ``u_i``。
            align_summary: ``(B, K, 32)`` ``a_i``。
            label_features: ``(B, K, 5)`` ``φ(y_i)``。
            assay_family: ``(B, K)`` long，assay family 索引。
            candidate_meta: ``(B, K, 8)`` ``t_i``（无参数元特征）。
            rank_scores: ``(B, K)`` 重排分数，用于注意力。
            candidate_mask: ``(B, K)`` 有效候选掩码。

        Returns:
            :class:`EvidenceOutput`。
        """
        e_y = self.label_proj(label_features)                    # (B, K, 16)
        e_a = self.assay_embedding(assay_family)                 # (B, K, 8)
        raw = torch.cat([ligand_proj_cand, align_summary, e_y, e_a, candidate_meta], dim=-1)   # (B, K, 96)
        encoded = self.evidence_norm(self.evidence_proj(raw))    # (B, K, 64)

        logits = rank_scores / self.tau_rank
        if candidate_mask is not None:
            logits = logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)
        attention = torch.softmax(logits, dim=-1)                # (B, K)

        context = torch.einsum("bk,bkd->bd", attention, encoded)     # c_R  (B, 64)
        pooled_raw = torch.einsum("bk,bkd->bd", attention, raw)      #      (B, 96)
        return EvidenceOutput(context=context, attention=attention, raw_evidence=raw, pooled_raw=pooled_raw)

    def n_parameters(self) -> int:
        """可训练参数量（预算核对：6,688）。"""
        return sum(p.numel() for p in self.parameters())

    def parameter_breakdown(self) -> Dict[str, int]:
        """分模块参数量。"""
        return {
            "label_proj": sum(p.numel() for p in self.label_proj.parameters()),
            "assay_embedding": sum(p.numel() for p in self.assay_embedding.parameters()),
            "evidence_proj+norm": sum(
                p.numel() for p in [*self.evidence_proj.parameters(), *self.evidence_norm.parameters()]
            ),
            "total": self.n_parameters(),
        }
