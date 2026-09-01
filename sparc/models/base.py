"""Θ_B —— Base 预测器 (§8.2)。**参数量 57,218，零容差。**

结构::

    u_q      = ModelA.encode(smiles)              ∈ R^{d_A}   冻结
    z_q      = PCA_whiten_128(u_q)                ∈ R^128     冻结（fold-train 拟合）
    z_q'     = LN(W_AB z_q + b_AB)                ∈ R^128     16,768
    p_raw    = ESM2.mean_pool(seq)                ∈ R^640     冻结
    p_pca    = PCA_64(p_raw)                      ∈ R^64      冻结
    c_t      = LN(W_p p_pca + b_p)                ∈ R^64       4,288
    s_bil    = 低秩双线性 4 分量 → 外积展开        ∈ R^16         768
    x_q      = [z_q'; c_t; s_bil]                 ∈ R^208
    h_q      = LN(W_Bf x_q + b_Bf)                ∈ R^128     27,008
    r        = Dropout_0.1(GELU(W_B1 h_q + b_B1)) ∈ R^64       8,256
    [μ, logσ²] = W_B2 r + b_B2                    ∈ R^2          130

关于 ``s_bil ⊗ broadcast``：§8.2 给的预算是 768 = 128×4 + 64×4，
即 ``U_b``、``V_b`` 两个无偏置投影，展开步骤本身**不含参数**。
因此这里取 ``a = U_bᵀz'`` 与 ``b = V_bᵀc`` 的外积 ``a ⊗ b ∈ R^{4×4}``
展平成 16 维 —— 其对角线正是 §8.2 所说的"4 个低秩分量" ``a⊙b``，
非对角线是交叉项，全程无额外参数。

关于蛋白编码降到 4,288 参数（原 328,448）：**68 个蛋白序列养不起
32.8 万参数**。原设计在单靶点下等于用 32.8 万参数计算一个常数。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass
class BaseOutput:
    """Base 预测器的输出。"""

    mu: Tensor          # (B,) 预测均值 —— 回归任务的自然参数 η_base
    log_var: Tensor     # (B,) 预测对数方差 —— Tobit 似然的 σ²
    hidden: Tensor      # (B, 128) h_q，检索分支与残差要用
    ligand: Tensor      # (B, 128) z_q'，构成 v_q 的前半

    @property
    def sigma(self) -> Tensor:
        """aleatoric 标准差 σ_a —— 门控特征第 28 维。"""
        return torch.exp(0.5 * self.log_var)

    def query_repr(self) -> Tensor:
        """``v_q = [z_q'; h_q] ∈ R^256`` —— 重排器与残差共用的查询表示。"""
        return torch.cat([self.ligand, self.hidden], dim=-1)


class BasePredictor(nn.Module):
    """Θ_B（§8.2）。仅用查询与靶点，**不访问记忆库**。

    这一点是整个方法的锚：``d_i = ℓ(base + g̃Δ) − ℓ(base)`` 的可解释性
    完全依赖于 base 是一个纯查询预测器 (§11.3)。ActFound 的
    ``ŷ_k = Σ_j a_kj(y_j + Δ̂_kj)`` 完全由邻居标签构成，
    没有邻居就什么都产生不了，因此它结构上无法拒检。
    """

    def __init__(
        self,
        d_ligand_in: int = 128,
        d_protein_in: int = 64,
        d_ligand_hidden: int = 128,
        d_protein_ctx: int = 64,
        bilinear_rank: int = 4,
        d_hidden: int = 128,
        d_head: int = 64,
        dropout: float = 0.10,
    ) -> None:
        """
        Args:
            d_ligand_in: PCA 白化后的配体维度（冻结为 128）。
            d_protein_in: PCA 后的蛋白维度（冻结为 64）。
            d_ligand_hidden: ``z_q'`` 维度。
            d_protein_ctx: ``c_t`` 维度。
            bilinear_rank: 低秩双线性的秩（4）。
            d_hidden: ``h_q`` 维度。
            d_head: 头部隐藏维度。
            dropout: Dropout 概率（冻结为 0.1）。
        """
        super().__init__()
        self.bilinear_rank = bilinear_rank

        # z_q' = LN(W_AB z_q + b_AB)                       16,768
        self.ligand_proj = nn.Linear(d_ligand_in, d_ligand_hidden)
        self.ligand_norm = nn.LayerNorm(d_ligand_hidden)

        # c_t = LN(W_p p_pca + b_p)                         4,288
        self.protein_proj = nn.Linear(d_protein_in, d_protein_ctx)
        self.protein_norm = nn.LayerNorm(d_protein_ctx)

        # 低秩双线性（无偏置）                                 768
        self.bilinear_u = nn.Parameter(torch.empty(d_ligand_hidden, bilinear_rank))
        self.bilinear_v = nn.Parameter(torch.empty(d_protein_ctx, bilinear_rank))
        nn.init.xavier_uniform_(self.bilinear_u)
        nn.init.xavier_uniform_(self.bilinear_v)

        d_fusion_in = d_ligand_hidden + d_protein_ctx + bilinear_rank ** 2   # 128+64+16 = 208
        self.fusion = nn.Linear(d_fusion_in, d_hidden)                        # 27,008（含 LN）
        self.fusion_norm = nn.LayerNorm(d_hidden)

        self.head_hidden = nn.Linear(d_hidden, d_head)                        #  8,256
        self.dropout = nn.Dropout(dropout)
        self.head_out = nn.Linear(d_head, 2)                                  #    130

        self.d_fusion_in = d_fusion_in

    # ------------------------------------------------------------------
    def forward(self, ligand_pca: Tensor, protein_pca: Tensor) -> BaseOutput:
        """前向。

        Args:
            ligand_pca: ``(B, 128)`` 冻结 PCA 白化后的 Model A embedding。
            protein_pca: ``(B, 64)`` 冻结 PCA 后的 ESM-2 mean-pool。

        Returns:
            :class:`BaseOutput`。
        """
        z = self.ligand_norm(self.ligand_proj(ligand_pca))       # (B, 128)
        c = self.protein_norm(self.protein_proj(protein_pca))    # (B, 64)

        a = z @ self.bilinear_u                                  # (B, 4)
        b = c @ self.bilinear_v                                  # (B, 4)
        # 外积展开：对角线即 a⊙b 的 4 个低秩分量，非对角线为交叉项；无参数
        s_bil = torch.einsum("bi,bj->bij", a, b).flatten(start_dim=1)   # (B, 16)

        x = torch.cat([z, c, s_bil], dim=-1)                     # (B, 208)
        h = self.fusion_norm(self.fusion(x))                     # (B, 128)
        r = self.dropout(nn.functional.gelu(self.head_hidden(h)))  # (B, 64)
        out = self.head_out(r)                                   # (B, 2)

        return BaseOutput(
            mu=out[:, 0],
            log_var=out[:, 1].clamp(min=-10.0, max=10.0),        # 防止 σ² 数值下溢/爆炸
            hidden=h,
            ligand=z,
        )

    # ------------------------------------------------------------------
    def freeze(self) -> "BasePredictor":
        """S1-F：冻结 Θ_B 并切到 eval mode。

        Returns:
            ``self``。冻结后 Dropout 关闭 —— 否则 S2 里 base 的输出
            带随机性，``d_i = ℓ(base+g̃Δ) − ℓ(base)`` 会混入噪声。
        """
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()
        return self

    def n_parameters(self) -> int:
        """可训练参数量（用于 §8.3.6 的预算核对）。"""
        return sum(p.numel() for p in self.parameters())

    def parameter_breakdown(self) -> Dict[str, int]:
        """分模块参数量，供 ``tests/test_param_budget.py`` 逐项比对。"""
        return {
            "ligand_proj+norm": sum(p.numel() for p in [*self.ligand_proj.parameters(), *self.ligand_norm.parameters()]),
            "protein_proj+norm": sum(p.numel() for p in [*self.protein_proj.parameters(), *self.protein_norm.parameters()]),
            "bilinear": self.bilinear_u.numel() + self.bilinear_v.numel(),
            "fusion+norm": sum(p.numel() for p in [*self.fusion.parameters(), *self.fusion_norm.parameters()]),
            "head_hidden": sum(p.numel() for p in self.head_hidden.parameters()),
            "head_out": sum(p.numel() for p in self.head_out.parameters()),
            "total": self.n_parameters(),
        }
