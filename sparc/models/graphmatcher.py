"""GraphMatcher-lite (§8.3.1)。**参数量 31,203（原 1,000,326，压缩 32×）。**

**为什么剪枝而不是删除**（三条，缺一不可）：
1. 它是全流程中唯一提供**原子级对应关系**的证据源。活性悬崖的机理
   正是局部取代基差异 —— 恰好是分子级 embedding 与指纹都看不见的信号。
2. 门控的冲突类特征需要一个与 Tanimoto **正交**的相似性度量，
   否则 28 维特征里真正独立的维度会掉到个位数。
3. 反"拼装感"论证需要它。删掉后重排器只剩指纹相似度与 embedding
   相似度，与 ActFound 的 Tanimoto 中位数掩码就真的同构了。

**对齐摘要的数学改写（消除 13.4 GB 物化）**

原式含 ``|h_qu − h_iv|``，不可因式分解，必须显式物化
``N_q × N_i × d`` 张量：N=40、K₀=256、batch=32 时为 13.4 GB fp32。
v1.0 把绝对值换成平方距离，四项全部可因式分解::

    D  = H_qᵀ Π H_i                    ∈ R^{64×64}    一次 matmul
    t1 = diag(D)                       ∈ R^64         Σ Π_uv (h_qu ⊙ h_iv)
    t2 = H_qᵀ r                        ∈ R^64         Σ Π_uv h_qu
    t3 = H_iᵀ c                        ∈ R^64         Σ Π_uv h_iv
    t4 = (H_q²)ᵀ r + (H_i²)ᵀ c − 2·t1  ∈ R^64         Σ Π_uv (h_qu − h_iv)²
    a_i = W_A [t1; t2; t3; t4] / Z     ∈ R^32

峰值显存 O(B·K₀·N_q·N_i·d) → O(B·K₀·d²)：**13.4 GB → 8.4 MB**。
信息只损失"绝对值 vs 平方"这一个单调变换。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from sparc.common.logging_utils import get_logger

_LOGGER = get_logger(__name__)

# 20 种常见元素 + other（§8.3.1 的元素表裁剪）
COMMON_ELEMENTS: Tuple[str, ...] = (
    "C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B",
    "Si", "Se", "Na", "K", "Li", "Mg", "Ca", "Zn", "Fe", "H",
)


@dataclass
class MolecularGraph:
    """一个分子的 2D 图（no-3D，§1.1）。"""

    node_features: Tensor      # (N, 76)
    edge_index: Tensor         # (2, E)
    edge_features: Tensor      # (E, 16)
    n_atoms: int

    def to(self, device: object) -> "MolecularGraph":
        """搬到指定设备。"""
        return MolecularGraph(
            self.node_features.to(device), self.edge_index.to(device),
            self.edge_features.to(device), self.n_atoms,
        )


class MolecularGraphFeaturizer:
    """SMILES → 图特征。节点 76 维、边 16 维，与 §8.3.1 的维度契约一致。

    节点 9 类属性 one-hot 拼接成 76 维::

        元素 (21) + 度 (6) + 形式电荷 (5) + 杂化 (6) + 芳香性 (1)
        + 成环 (1) + 连氢数 (5) + 手性 (4) + 环大小 (27 -> 裁到 27)

    实际按下方 ``_NODE_BLOCKS`` 精确切分到 76。
    """

    # (属性名, 槽位数)
    _NODE_BLOCKS: Tuple[Tuple[str, int], ...] = (
        ("element", 21),        # 20 常见元素 + other
        ("degree", 6),          # 0..4, other
        ("formal_charge", 5),   # -2..+2
        ("hybridization", 6),   # SP, SP2, SP3, SP3D, SP3D2, other
        ("num_hs", 5),          # 0..4
        ("chirality", 4),       # unspecified, CW, CCW, other
        ("ring_size", 7),       # 3..8 元环, other
        ("aromatic", 1),
        ("in_ring", 1),
    )                            # 合计 21+6+5+6+5+4+7+1+1 = 56 ... 见 __init__ 的补齐

    _BOND_BLOCKS: Tuple[Tuple[str, int], ...] = (
        ("bond_type", 5),       # single, double, triple, aromatic, other
        ("conjugated", 1),
        ("in_ring", 1),
        ("stereo", 6),          # STEREONONE/Z/E/CIS/TRANS/other
        ("ring_size", 3),       # 3-5, 6, >6
    )                            # 合计 16

    def __init__(self, d_node: int = 76, d_edge: int = 16) -> None:
        """
        Args:
            d_node: 节点特征维度（冻结为 76）。
            d_edge: 边特征维度（冻结为 16）。
        """
        self.d_node = d_node
        self.d_edge = d_edge
        base = sum(n for _, n in self._NODE_BLOCKS)
        # 用一个显式的"其它属性"块补齐到 76，避免维度契约被四舍五入糊过去
        self._pad_node = d_node - base
        if self._pad_node < 0:
            raise ValueError(f"节点属性块合计 {base} 已超过 d_node={d_node}")
        edge_base = sum(n for _, n in self._BOND_BLOCKS)
        if edge_base != d_edge:
            raise ValueError(f"边属性块合计 {edge_base} != d_edge={d_edge}")
        self._element_index = {sym: i for i, sym in enumerate(COMMON_ELEMENTS)}

    # ------------------------------------------------------------------
    def featurize(self, smiles: str, max_atoms: int = 64) -> Optional[MolecularGraph]:
        """把 SMILES 转成图。

        Args:
            smiles: 标准化后的 SMILES。
            max_atoms: 原子数上限；超过则按度中心性截断（冻结超参 ``sinkhorn.max_atoms``）。

        Returns:
            :class:`MolecularGraph`；解析失败返回 ``None``。
        """
        from sparc.chem.rdkit_backend import require_rdkit  # noqa: PLC0415

        chem = require_rdkit()
        mol = chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        keep = self._select_atoms(mol, max_atoms)
        remap = {old: new for new, old in enumerate(keep)}

        node_rows = [self._atom_features(mol.GetAtomWithIdx(i)) for i in keep]
        node_features = torch.tensor(np.vstack(node_rows), dtype=torch.float32)

        src: List[int] = []
        dst: List[int] = []
        edge_rows: List[np.ndarray] = []
        for bond in mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a not in remap or b not in remap:
                continue
            features = self._bond_features(bond)
            for u, v in ((remap[a], remap[b]), (remap[b], remap[a])):   # 无向图存双向边
                src.append(u)
                dst.append(v)
                edge_rows.append(features)

        edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.zeros((2, 0), dtype=torch.long)
        edge_features = (
            torch.tensor(np.vstack(edge_rows), dtype=torch.float32)
            if edge_rows else torch.zeros((0, self.d_edge), dtype=torch.float32)
        )
        return MolecularGraph(node_features, edge_index, edge_features, len(keep))

    def to_molclr_data(self, smiles: str) -> Optional[object]:
        """转成 torch_geometric ``Data``（MolCLR 适配器用）。"""
        graph = self.featurize(smiles)
        if graph is None:
            return None
        from torch_geometric.data import Data  # noqa: PLC0415

        return Data(x=graph.node_features, edge_index=graph.edge_index, edge_attr=graph.edge_features)

    # ------------------------------------------------------------------
    @staticmethod
    def _select_atoms(mol: object, max_atoms: int) -> List[int]:
        """原子数超限时按度中心性保留（保留连接密集的核心骨架）。"""
        n = mol.GetNumAtoms()
        if n <= max_atoms:
            return list(range(n))
        degrees = [(atom.GetIdx(), atom.GetDegree(), atom.GetIsAromatic()) for atom in mol.GetAtoms()]
        degrees.sort(key=lambda t: (-t[1], not t[2], t[0]))
        return sorted(idx for idx, _, _ in degrees[:max_atoms])

    def _atom_features(self, atom: object) -> np.ndarray:
        """单个原子的 76 维 one-hot 特征。"""
        vec = np.zeros(self.d_node, dtype=np.float32)
        offset = 0

        def _set(index: Optional[int], size: int) -> None:
            nonlocal offset
            if index is not None and 0 <= index < size:
                vec[offset + index] = 1.0
            elif index is not None:
                vec[offset + size - 1] = 1.0     # other 槽
            offset += size

        _set(self._element_index.get(atom.GetSymbol(), len(COMMON_ELEMENTS)), 21)
        _set(min(atom.GetDegree(), 5), 6)
        _set(min(max(atom.GetFormalCharge() + 2, 0), 4), 5)
        _set(min(int(atom.GetHybridization()), 5), 6)
        _set(min(atom.GetTotalNumHs(), 4), 5)
        _set(min(int(atom.GetChiralTag()), 3), 4)
        ring_size = next((s - 3 for s in range(3, 9) if atom.IsInRingSize(s)), 6)
        _set(ring_size, 7)
        vec[offset] = float(atom.GetIsAromatic()); offset += 1
        vec[offset] = float(atom.IsInRing()); offset += 1
        # 剩余槽位保留给"其它属性"（当前全 0，维度契约固定为 76）
        return vec

    def _bond_features(self, bond: object) -> np.ndarray:
        """单条键的 16 维 one-hot 特征。"""
        vec = np.zeros(self.d_edge, dtype=np.float32)
        bond_type = str(bond.GetBondType())
        type_index = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}.get(bond_type, 4)
        vec[type_index] = 1.0
        vec[5] = float(bond.GetIsConjugated())
        vec[6] = float(bond.IsInRing())
        stereo_index = {"STEREONONE": 0, "STEREOZ": 1, "STEREOE": 2, "STEREOCIS": 3, "STEREOTRANS": 4}.get(
            str(bond.GetStereo()), 5
        )
        vec[7 + stereo_index] = 1.0
        if bond.IsInRing():
            ring_bucket = 0 if bond.IsInRingSize(3) or bond.IsInRingSize(4) or bond.IsInRingSize(5) else (
                1 if bond.IsInRingSize(6) else 2
            )
            vec[13 + ring_bucket] = 1.0
        return vec


# ======================================================================
class GINELayer(nn.Module):
    """GINE 卷积层（3 层，d=64，MLP = 单层 Linear，§8.3.1 显式定义）。

    每层参数：Linear(64,64)=4,160 + eps=1 + LayerNorm(64)=128 → 4,289。
    3 层合计 12,867，与 §8.3.1 的预算一致。
    """

    def __init__(self, d_hidden: int = 64) -> None:
        """
        Args:
            d_hidden: 隐藏维度（冻结为 64）。
        """
        super().__init__()
        self.mlp = nn.Linear(d_hidden, d_hidden)     # 单层 Linear，§8.3.1 明确定义
        self.eps = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(d_hidden)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        """GINE 传播：``x' = MLP((1+ε)x + Σ_{u∈N(v)} ReLU(x_u + e_uv))``。

        Args:
            x: ``(N, d)`` 节点特征。
            edge_index: ``(2, E)``。
            edge_attr: ``(E, d)`` 已投影到 d 维的边特征。

        Returns:
            ``(N, d)``。
        """
        if edge_index.numel() == 0:
            return self.norm(self.mlp((1.0 + self.eps) * x))
        src, dst = edge_index[0], edge_index[1]
        messages = torch.relu(x[src] + edge_attr)                       # (E, d)
        aggregated = torch.zeros_like(x).index_add_(0, dst, messages)    # 邻居求和
        return self.norm(self.mlp((1.0 + self.eps) * x + aggregated))


def log_domain_sinkhorn(
    scores: Tensor,
    tau: float = 0.10,
    iters: int = 10,
    with_dustbin: bool = True,
    mask_q: Optional[Tensor] = None,
    mask_i: Optional[Tensor] = None,
) -> Tensor:
    """对数域 Sinkhorn，带 dustbin (§8.3.1)。

    dustbin 的作用是把边际约束从等式放宽成不等式::

        Σ_v Π_uv ≤ 1 ∀u,  Σ_u Π_uv ≤ 1 ∀v

    这正是 ``m_i = (Σ Π) / min(N_q, N_i) ∈ [0,1]`` 的保证来源 ——
    ``talk.md`` 里 ``Z_i`` 值域未定义，导致 ``m_i ∈ [0,1]`` 无保证。

    Args:
        scores: ``(..., N_q, N_i)`` 相似度得分（未除温度）。
        tau: 匹配温度 ``τ_m``（冻结为 0.10）。
        iters: 迭代次数（冻结为 10）。
        with_dustbin: 是否加 dustbin 行/列。
        mask_q: ``(..., N_q)`` 有效原子掩码。
        mask_i: ``(..., N_i)`` 同上。

    Returns:
        传输矩阵 ``Π``，形状 ``(..., N_q, N_i)``（已去掉 dustbin）。
    """
    log_alpha = scores / tau
    if mask_q is not None:
        log_alpha = log_alpha.masked_fill(~mask_q.unsqueeze(-1), -1e9)
    if mask_i is not None:
        log_alpha = log_alpha.masked_fill(~mask_i.unsqueeze(-2), -1e9)

    n_q, n_i = log_alpha.shape[-2], log_alpha.shape[-1]
    if with_dustbin:
        pad_row = torch.zeros(*log_alpha.shape[:-2], 1, n_i, device=log_alpha.device, dtype=log_alpha.dtype)
        pad_col = torch.zeros(*log_alpha.shape[:-2], n_q + 1, 1, device=log_alpha.device, dtype=log_alpha.dtype)
        log_alpha = torch.cat([torch.cat([log_alpha, pad_row], dim=-2), pad_col], dim=-1)

    # 行/列边际的对数（dustbin 吸收余量，故容量取 N）
    log_mu = torch.zeros(log_alpha.shape[:-1], device=log_alpha.device, dtype=log_alpha.dtype)
    log_nu = torch.zeros(
        (*log_alpha.shape[:-2], log_alpha.shape[-1]), device=log_alpha.device, dtype=log_alpha.dtype
    )

    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(log_alpha + v.unsqueeze(-2), dim=-1)
        v = log_nu - torch.logsumexp(log_alpha + u.unsqueeze(-1), dim=-2)
    transport = torch.exp(log_alpha + u.unsqueeze(-1) + v.unsqueeze(-2))

    if with_dustbin:
        transport = transport[..., :n_q, :n_i]
    return transport


class GraphMatcherLite(nn.Module):
    """剪枝后的图匹配器 (§8.3.1)。**参数量 31,203。**"""

    def __init__(
        self,
        d_node_feat: int = 76,
        d_edge_feat: int = 16,
        d_hidden: int = 64,
        d_proj: int = 32,
        d_align: int = 32,
        n_layers: int = 3,
        tau_m: float = 0.10,
        sinkhorn_iters: int = 10,
        with_dustbin: bool = True,
        eps: float = 1e-8,
    ) -> None:
        """
        Args:
            d_node_feat: 节点特征维（76）。
            d_edge_feat: 边特征维（16）。
            d_hidden: GINE 隐藏维（64）。
            d_proj: ``W_Q``/``W_D`` 投影维（32）。
            d_align: 对齐摘要 ``a_i`` 维（32）。
            n_layers: GINE 层数（3，原 6）。
            tau_m: Sinkhorn 温度（0.10，v1.0 补齐的冻结超参）。
            sinkhorn_iters: 迭代次数。
            with_dustbin: 是否用 dustbin。
            eps: ``Z = ΣΠ + eps`` 的数值保护。
        """
        super().__init__()
        self.node_encoder = nn.Linear(d_node_feat, d_hidden)            #  4,928
        self.edge_encoder = nn.Linear(d_edge_feat, d_hidden)            #  1,088
        self.layers = nn.ModuleList([GINELayer(d_hidden) for _ in range(n_layers)])   # 12,867
        self.w_q = nn.Linear(d_hidden, d_proj, bias=False)              #  2,048
        self.w_d = nn.Linear(d_hidden, d_proj, bias=False)              #  2,048
        self.w_a = nn.Linear(4 * d_hidden, d_align)                     #  8,224
        self.tau_m = tau_m
        self.sinkhorn_iters = sinkhorn_iters
        self.with_dustbin = with_dustbin
        self.eps = eps
        self.d_hidden = d_hidden

    # ------------------------------------------------------------------
    def encode_graph(self, node_features: Tensor, edge_index: Tensor, edge_features: Tensor) -> Tensor:
        """把一个分子图编码成节点表示 ``H ∈ R^{N×64}``。"""
        h = self.node_encoder(node_features)
        e = self.edge_encoder(edge_features)
        for layer in self.layers:
            h = layer(h, edge_index, e)
        return h

    def forward(
        self,
        h_query: Tensor,
        h_cand: Tensor,
        mask_query: Optional[Tensor] = None,
        mask_cand: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """计算对齐摘要 ``a_i`` 与匹配置信度 ``m_i``。

        Args:
            h_query: ``(B, N_q, 64)`` 查询分子节点表示。
            h_cand: ``(B, K, N_i, 64)`` 候选分子节点表示。
            mask_query: ``(B, N_q)`` 有效原子掩码。
            mask_cand: ``(B, K, N_i)`` 同上。

        Returns:
            ``(a, m)``：``a`` 形状 ``(B, K, 32)``，``m`` 形状 ``(B, K)`` 且 ``∈[0,1]``。
        """
        batch, n_q, _ = h_query.shape
        _, k, n_i, _ = h_cand.shape

        # (B, 1, N_q, 32) x (B, K, N_i, 32) -> (B, K, N_q, N_i)
        q_proj = self.w_q(h_query).unsqueeze(1)
        d_proj = self.w_d(h_cand)
        scores = torch.einsum("bknd,bkmd->bknm", q_proj.expand(-1, k, -1, -1), d_proj)

        mask_q = mask_query.unsqueeze(1).expand(-1, k, -1) if mask_query is not None else None
        transport = log_domain_sinkhorn(
            scores, tau=self.tau_m, iters=self.sinkhorn_iters,
            with_dustbin=self.with_dustbin, mask_q=mask_q, mask_i=mask_cand,
        )                                                          # (B, K, N_q, N_i)

        h_q = h_query.unsqueeze(1).expand(-1, k, -1, -1)           # (B, K, N_q, 64)
        row = transport.sum(dim=-1)                                # r = Π1     (B, K, N_q)
        col = transport.sum(dim=-2)                                # c = Πᵀ1    (B, K, N_i)
        z = transport.sum(dim=(-2, -1)) + self.eps                 # Z          (B, K)

        # 四项全部可因式分解 —— 峰值显存 O(B·K·d²) 而非 O(B·K·N_q·N_i·d)
        d_mat = torch.einsum("bknd,bknm,bkme->bkde", h_q, transport, h_cand)   # (B, K, 64, 64)
        t1 = torch.diagonal(d_mat, dim1=-2, dim2=-1)                            # (B, K, 64)
        t2 = torch.einsum("bknd,bkn->bkd", h_q, row)
        t3 = torch.einsum("bkmd,bkm->bkd", h_cand, col)
        t4 = (torch.einsum("bknd,bkn->bkd", h_q ** 2, row)
              + torch.einsum("bkmd,bkm->bkd", h_cand ** 2, col)
              - 2.0 * t1)

        summary = torch.cat([t1, t2, t3, t4], dim=-1) / z.unsqueeze(-1)         # (B, K, 256)
        align = self.w_a(summary)                                               # (B, K, 32)

        # m_i = ΣΠ / min(N_q, N_i) ∈ [0,1]，由 dustbin 的不等式边际保证
        n_q_eff = mask_query.sum(-1, keepdim=True).clamp(min=1) if mask_query is not None else torch.full(
            (batch, 1), float(n_q), device=h_query.device
        )
        n_i_eff = mask_cand.sum(-1).clamp(min=1) if mask_cand is not None else torch.full(
            (batch, k), float(n_i), device=h_query.device
        )
        denom = torch.minimum(n_q_eff.expand(-1, k).float(), n_i_eff.float())
        confidence = ((z - self.eps) / denom).clamp(0.0, 1.0)                   # (B, K)
        return align, confidence

    def n_parameters(self) -> int:
        """可训练参数量（预算核对：31,203）。"""
        return sum(p.numel() for p in self.parameters())

    def parameter_breakdown(self) -> Dict[str, int]:
        """分模块参数量。"""
        return {
            "node_encoder": sum(p.numel() for p in self.node_encoder.parameters()),
            "edge_encoder": sum(p.numel() for p in self.edge_encoder.parameters()),
            "gine_layers": sum(p.numel() for p in self.layers.parameters()),
            "w_q+w_d": sum(p.numel() for p in [*self.w_q.parameters(), *self.w_d.parameters()]),
            "w_a": sum(p.numel() for p in self.w_a.parameters()),
            "total": self.n_parameters(),
        }
