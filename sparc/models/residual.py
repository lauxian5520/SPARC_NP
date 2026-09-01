"""Residual 与不确定性头 (§8.3.4)。**参数量 2,633 + 322。**

```
U ∈ R^{256×8},  V ∈ R^{64×8}                      2,560
Δ = w_Δᵀ(Uᵀv_q ⊙ Vᵀc_R) + w_cᵀc_R + b_Δ              73
```

**``h_q`` 不再冻结。** 原方案冻结 ``h_q^B`` 是为了保证 ``g̃=0`` 时精确退化回 Base
—— 但那个保证是由"乘以 ``g̃``"给出的，与冻不冻结无关。放开后残差可以微调
查询表示以适配证据空间。

实现上的调和：S2 阶段 Θ_B 的**参数**仍 ``requires_grad=False``（§10.4 的
阶段表是权威），但梯度**允许穿过** ``h_q``（不 detach）。这由
``frozen_hparams.yaml`` 的 ``train.retrieval.detach_base_hidden: false`` 控制。

回退等价性由下列断言保证，且强制 fp32（fallback 路径不允许混合精度容差）::

    assert torch.allclose(predict(q, g_override=0.0), base_predict(q), atol=1e-6)
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn


class ResidualHead(nn.Module):
    """低秩双线性残差 ``Δ(q, t, R)``。**参数量 2,633。**"""

    def __init__(self, d_query_repr: int = 256, d_context: int = 64, rank: int = 8) -> None:
        """
        Args:
            d_query_repr: ``v_q`` 维（256）。
            d_context: ``c_R`` 维（64）。
            rank: 低秩秩（8；原方案 32，消融中作为对照，见 Table 2）。
        """
        super().__init__()
        self.u = nn.Parameter(torch.empty(d_query_repr, rank))     # 2,048
        self.v = nn.Parameter(torch.empty(d_context, rank))        #   512
        self.w_delta = nn.Parameter(torch.zeros(rank))             #     8
        self.w_context = nn.Parameter(torch.zeros(d_context))      #    64
        self.bias = nn.Parameter(torch.zeros(1))                   #     1
        nn.init.xavier_uniform_(self.u)
        nn.init.xavier_uniform_(self.v)
        # w_delta / w_context / bias 初始化为 0 ⇒ 训练起点 Δ ≡ 0，
        # 即"起手就等于 Base"。这让 S2 早期的 d_i 不被随机残差污染。

    def forward(self, query_repr: Tensor, evidence_context: Tensor) -> Tensor:
        """计算残差 ``Δ``。

        Args:
            query_repr: ``(B, 256)`` ``v_q``。
            evidence_context: ``(B, 64)`` ``c_R``。

        Returns:
            ``(B,)`` 残差标量。
        """
        a = query_repr @ self.u                                     # (B, 8)
        b = evidence_context @ self.v                               # (B, 8)
        interaction = (a * b) @ self.w_delta                        # (B,)
        linear = evidence_context @ self.w_context                  # (B,)
        return interaction + linear + self.bias

    def n_parameters(self) -> int:
        """可训练参数量（预算核对：2,633）。"""
        return sum(p.numel() for p in self.parameters())

    def parameter_breakdown(self) -> Dict[str, int]:
        """分模块参数量。"""
        return {
            "u": self.u.numel(),
            "v": self.v.numel(),
            "w_delta": self.w_delta.numel(),
            "w_context": self.w_context.numel(),
            "bias": self.bias.numel(),
            "total": self.n_parameters(),
        }


class UncertaintyHead(nn.Module):
    """检索侧不确定性头。**参数量 322。**

    输入 ``[c_R (64); pooled_raw_evidence (96)] ∈ R^160``，输出两个标量：
    * ``delta_log_var`` —— 施加残差后对 ``logσ²`` 的修正（Tobit 似然用）；
    * ``temperature_logit`` —— 校准损失 ``L_cal`` 的温度参数。

    160 → 2 的 Linear 恰好 322 个参数，与 §8.3.6 的"不确定性头 322"一致。
    """

    def __init__(self, d_context: int = 64, d_raw: int = 96) -> None:
        """
        Args:
            d_context: ``c_R`` 维（64）。
            d_raw: 注意力加权后的原始证据维（96）。
        """
        super().__init__()
        self.head = nn.Linear(d_context + d_raw, 2)        # 160 → 2 = 322
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # 同样零初始化：起点不修正 Base 的方差估计

    def forward(self, evidence_context: Tensor, pooled_raw: Tensor) -> tuple[Tensor, Tensor]:
        """前向。

        Args:
            evidence_context: ``(B, 64)``。
            pooled_raw: ``(B, 96)``。

        Returns:
            ``(delta_log_var, temperature_logit)``，各为 ``(B,)``。
        """
        out = self.head(torch.cat([evidence_context, pooled_raw], dim=-1))
        return out[:, 0].clamp(min=-5.0, max=5.0), out[:, 1]

    def n_parameters(self) -> int:
        """可训练参数量（预算核对：322）。"""
        return sum(p.numel() for p in self.parameters())
