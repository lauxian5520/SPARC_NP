"""检索管线编排 (§9.1)。

把 ``M_view`` → 三源召回 → K₀ 候选池 → 图匹配张量包 → Reranker
串成一条可批处理的路径，并输出 :class:`RetrievalBatch` 供
:class:`~sparc.models.sparc_model.SparcNP` 消费。

**顺序纪律**：视图 → 索引 → 召回 → 断言 → 图匹配 → 重排。
绝不允许"先召回再过滤" —— 兼容记录占比 ~10⁻³ 时，512 个里平均
只剩 0.5 个可用，候选集会静默为空并永久走 fallback 而不报错。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

from sparc.common.logging_utils import get_logger
from sparc.data.schema import CensorFlag, Domain, MemoryRecord
from sparc.models.graphmatcher import GraphMatcherLite, MolecularGraph, MolecularGraphFeaturizer
from sparc.retrieval.index import CandidatePool, RetrievalIndex, assert_compatible_ratio
from sparc.retrieval.memory import MemoryView

_LOGGER = get_logger(__name__)


@dataclass
class RetrievalBatch:
    """喂给 :meth:`SparcNP.forward` 的检索张量包。"""

    query_node_repr: Tensor          # (B, N_q, 64)
    cand_node_repr: Tensor           # (B, K0, N_i, 64)
    query_atom_mask: Tensor          # (B, N_q) bool
    cand_atom_mask: Tensor           # (B, K0, N_i) bool
    cand_repr: Tensor                # (B, K0, 256) v_i
    rank_scalar_features: Tensor     # (B, K0, 7)
    label_features: Tensor           # (B, K0, 5)
    assay_family: Tensor             # (B, K0) long
    candidate_meta: Tensor           # (B, K0, 8) t_i
    candidate_mask: Tensor           # (B, K0) bool
    pool_tanimoto: Tensor            # (B, K0)
    pool_labels: Tensor              # (B, K0)
    k_top: int = 16
    pools: List[CandidatePool] = field(default_factory=list)
    insufficient: Tensor = None      # (B,) bool，|M_view| < 200 的查询

    def to(self, device: object) -> "RetrievalBatch":
        """把全部张量搬到指定设备。"""
        moved = {}
        for key, value in self.__dict__.items():
            moved[key] = value.to(device) if isinstance(value, Tensor) else value
        return RetrievalBatch(**moved)

    def as_model_input(self) -> Dict[str, Any]:
        """转成 :meth:`SparcNP.forward` 的 ``retrieval_batch`` 参数。"""
        return {
            "query_node_repr": self.query_node_repr,
            "cand_node_repr": self.cand_node_repr,
            "query_atom_mask": self.query_atom_mask,
            "cand_atom_mask": self.cand_atom_mask,
            "cand_repr": self.cand_repr,
            "rank_scalar_features": self.rank_scalar_features,
            "label_features": self.label_features,
            "assay_family": self.assay_family,
            "candidate_meta": self.candidate_meta,
            "candidate_mask": self.candidate_mask,
            "k_top": self.k_top,
        }


class RetrievalPipeline:
    """按 §9.1 的强制顺序执行检索。"""

    def __init__(
        self,
        graph_matcher: GraphMatcherLite,
        featurizer: Optional[MolecularGraphFeaturizer] = None,
        k_src: int = 512,
        k0: int = 256,
        k_top: int = 16,
        max_atoms: int = 64,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            graph_matcher: 提供 ``encode_graph``（节点表示由它算，
                这样图编码器的梯度能从 ``L_rank`` 流回来）。
            featurizer: SMILES → 图特征；``None`` 时新建。
            k_src: 每源召回上限。
            k0: 候选池大小。
            k_top: Top-K。
            max_atoms: 原子数上限。
            device: 张量设备。
        """
        self.graph_matcher = graph_matcher
        self.featurizer = featurizer or MolecularGraphFeaturizer()
        self.k_src = k_src
        self.k0 = k0
        self.k_top = k_top
        self.max_atoms = max_atoms
        self.device = device
        self._graph_cache: Dict[str, Optional[MolecularGraph]] = {}
        self._index_cache: Dict[Tuple[str, ...], RetrievalIndex] = {}

    # ------------------------------------------------------------------
    @classmethod
    def for_model(cls, model: Any, **kwargs: Any) -> "RetrievalPipeline":
        """从 :class:`~sparc.models.sparc_model.SparcNP` 构造管线。

        **组装 Stage 2 时请一律用这个入口，不要自己 new 一个 GraphMatcherLite。**
        原因见 :func:`assert_graph_matcher_shared`。

        Args:
            model: 已构造的 ``SparcNP``。
            **kwargs: 其余管线参数（``k_src`` / ``k0`` / ``k_top`` / ``max_atoms`` / ``device``）。

        Returns:
            与 ``model`` 共享图编码器的管线。
        """
        return cls(graph_matcher=model.graph_matcher, **kwargs)

    # ------------------------------------------------------------------
    def build_index(
        self,
        view: MemoryView,
        ecfp: np.ndarray,
        embeddings: np.ndarray,
        pharmacophore: Optional[np.ndarray] = None,
        seed: int = 42,
    ) -> RetrievalIndex:
        """在**已过滤的**视图上建三源索引（§9.1 步骤 2）。

        Args:
            view: 记忆库视图。
            ecfp: ``(|M_view|, 2048)``。
            embeddings: ``(|M_view|, d)``。
            pharmacophore: ``(|M_view|, n_bits)`` 或 ``None``。
            seed: LSH 平面种子（固定，保证跨 fold 召回可复现）。

        Returns:
            :class:`RetrievalIndex`（按视图键缓存）。
        """
        cache_key = view.key()
        if cache_key not in self._index_cache:
            self._index_cache[cache_key] = RetrievalIndex(
                ecfp, embeddings, pharmacophore, k_src=self.k_src, k0=self.k0, seed=seed,
            )
        return self._index_cache[cache_key]

    def graph_of(self, smiles: str) -> Optional[MolecularGraph]:
        """带缓存的图特征化（同一分子在多个查询的候选池里反复出现）。"""
        if smiles not in self._graph_cache:
            self._graph_cache[smiles] = self.featurizer.featurize(smiles, self.max_atoms)
        return self._graph_cache[smiles]

    # ------------------------------------------------------------------
    def retrieve_batch(
        self,
        query_smiles: Sequence[str],
        query_ecfp: np.ndarray,
        query_embeddings: np.ndarray,
        query_pharmacophore: Optional[np.ndarray],
        views: Sequence[MemoryView],
        indexes: Sequence[RetrievalIndex],
        query_reprs: Tensor,
        memory_reprs: Sequence[Tensor],
        label_mean: float,
        label_std: float,
    ) -> RetrievalBatch:
        """执行一个 batch 的完整检索。

        Args:
            query_smiles: ``(B,)`` 查询 SMILES。
            query_ecfp: ``(B, 2048)``。
            query_embeddings: ``(B, d)``。
            query_pharmacophore: ``(B, n_bits)`` 或 ``None``。
            views: ``(B,)`` 每个查询对应的记忆库视图。
            indexes: ``(B,)`` 对应的索引。
            query_reprs: ``(B, 256)`` ``v_q``（仅用于形状对齐，实际由模型算）。
            memory_reprs: ``(B,)`` 每个视图的 ``(|M_view|, 256)`` 记忆表示。
            label_mean: **NP fold-train** 的标签均值（§8.3.3：不用记忆库统计量）。
            label_std: 同上的标准差。

        Returns:
            :class:`RetrievalBatch`。
        """
        batch_size = len(query_smiles)
        pools: List[CandidatePool] = []
        insufficient: List[bool] = []

        for i in range(batch_size):
            if views[i].insufficient or indexes[i].size == 0:
                pools.append(CandidatePool(i, np.zeros(0, np.int64), *(np.zeros(0, np.float32) for _ in range(4))))
                insufficient.append(True)
                continue
            pool = indexes[i].retrieve(
                i, query_ecfp[i], query_embeddings[i],
                query_pharmacophore[i] if query_pharmacophore is not None else None,
            )
            assert_compatible_ratio(pool)          # §9.1 步骤 5，必须恒真
            pools.append(pool)
            insufficient.append(False)

        k0_actual = max((len(p) for p in pools), default=0) or 1
        return self._assemble(
            query_smiles, views, pools, memory_reprs, k0_actual,
            label_mean, label_std, insufficient,
        )

    # ------------------------------------------------------------------
    def _assemble(
        self,
        query_smiles: Sequence[str],
        views: Sequence[MemoryView],
        pools: Sequence[CandidatePool],
        memory_reprs: Sequence[Tensor],
        k0: int,
        label_mean: float,
        label_std: float,
        insufficient: Sequence[bool],
    ) -> RetrievalBatch:
        """把候选池组装成对齐的张量包（含 padding 与掩码）。"""
        batch_size = len(query_smiles)
        device = self.device

        # --- 图编码 ---
        query_graphs = [self.graph_of(s) for s in query_smiles]
        n_q = max((g.n_atoms for g in query_graphs if g), default=1)
        query_nodes = torch.zeros(batch_size, n_q, self.graph_matcher.d_hidden, device=device)
        query_mask = torch.zeros(batch_size, n_q, dtype=torch.bool, device=device)
        for i, graph in enumerate(query_graphs):
            if graph is None:
                continue
            g = graph.to(device)
            query_nodes[i, :g.n_atoms] = self.graph_matcher.encode_graph(
                g.node_features, g.edge_index, g.edge_features
            )
            query_mask[i, :g.n_atoms] = True

        cand_graphs: List[List[Optional[MolecularGraph]]] = []
        for i, pool in enumerate(pools):
            records = [views[i].records[j] for j in pool.indices.tolist()]
            cand_graphs.append([self.graph_of(r.smiles) for r in records])
        n_i = max((g.n_atoms for row in cand_graphs for g in row if g), default=1)

        d_hidden = self.graph_matcher.d_hidden
        cand_nodes = torch.zeros(batch_size, k0, n_i, d_hidden, device=device)
        cand_atom_mask = torch.zeros(batch_size, k0, n_i, dtype=torch.bool, device=device)
        cand_repr = torch.zeros(batch_size, k0, memory_reprs[0].shape[-1] if len(memory_reprs) else 256, device=device)
        rank_scalars = torch.zeros(batch_size, k0, 7, device=device)
        label_features = torch.zeros(batch_size, k0, 5, device=device)
        assay_family = torch.zeros(batch_size, k0, dtype=torch.long, device=device)
        candidate_meta = torch.zeros(batch_size, k0, 8, device=device)
        candidate_mask = torch.zeros(batch_size, k0, dtype=torch.bool, device=device)
        pool_tanimoto = torch.zeros(batch_size, k0, device=device)
        pool_labels = torch.zeros(batch_size, k0, device=device)

        for i, pool in enumerate(pools):
            n_cand = len(pool)
            if n_cand == 0:
                continue
            records = [views[i].records[j] for j in pool.indices.tolist()]
            candidate_mask[i, :n_cand] = True
            cand_repr[i, :n_cand] = memory_reprs[i][torch.as_tensor(pool.indices, device=device)]

            tan = torch.as_tensor(pool.tanimoto, device=device, dtype=torch.float32)
            emb = torch.as_tensor(pool.embedding_similarity, device=device, dtype=torch.float32)
            pharm = torch.as_tensor(pool.pharmacophore_similarity, device=device, dtype=torch.float32)
            pool_tanimoto[i, :n_cand] = tan

            labels = torch.tensor([r.pactivity for r in records], device=device, dtype=torch.float32)
            pool_labels[i, :n_cand] = labels
            # y_i^std 用 NP fold-train 的统计量，不用记忆库统计量（§8.3.3）
            labels_std = (labels - label_mean) / max(label_std, 1e-6)

            left = torch.tensor([r.censor_flag is CensorFlag.LEFT for r in records], device=device, dtype=torch.float32)
            right = torch.tensor([r.censor_flag is CensorFlag.RIGHT for r in records], device=device, dtype=torch.float32)
            is_drug = torch.tensor([r.domain is Domain.DRUG for r in records], device=device, dtype=torch.float32)
            present = torch.ones(n_cand, device=device)

            label_features[i, :n_cand] = torch.stack([labels_std, present, left, right, is_drug], dim=-1)
            assay_family[i, :n_cand] = torch.tensor([r.assay_family for r in records], device=device, dtype=torch.long)

            # x^rank 的 7 个标量：[s_tan, s_pharm, s_emb, m_i(占位), s_ctx, c_assay, c_endpoint]
            # m_i 由 SparcNP.forward 用图匹配结果就地覆盖第 4 列
            ranks = torch.arange(n_cand, device=device, dtype=torch.float32) / max(n_cand - 1, 1)
            rank_scalars[i, :n_cand, 0] = tan
            rank_scalars[i, :n_cand, 1] = pharm
            rank_scalars[i, :n_cand, 2] = emb
            rank_scalars[i, :n_cand, 4] = 1.0      # s_ctx：单靶点恒 1，manifest 标 dead
            rank_scalars[i, :n_cand, 5] = 1.0      # c_assay：硬过滤后恒 1
            rank_scalars[i, :n_cand, 6] = 1.0      # c_endpoint：硬过滤后恒 1

            # t_i 的 8 个无参数元特征（顺序见 evidence.CANDIDATE_META_FIELDS）
            stage1 = torch.as_tensor(pool.stage1_scores, device=device, dtype=torch.float32)
            stage1_z = (stage1 - stage1.mean()) / max(float(stage1.std()), 1e-6) if n_cand > 1 else stage1 * 0
            candidate_meta[i, :n_cand] = torch.stack([
                tan, emb, torch.zeros_like(tan), ranks, stage1_z,
                torch.ones_like(tan),                                   # assay_compatible（恒 1）
                (left + right).clamp(max=1.0),                          # is_censored
                torch.log1p(torch.tensor([r.n_source_records for r in records], device=device, dtype=torch.float32)),
            ], dim=-1)

            for j, graph in enumerate(cand_graphs[i]):
                if graph is None:
                    continue
                g = graph.to(device)
                cand_nodes[i, j, :g.n_atoms] = self.graph_matcher.encode_graph(
                    g.node_features, g.edge_index, g.edge_features
                )
                cand_atom_mask[i, j, :g.n_atoms] = True

        return RetrievalBatch(
            query_node_repr=query_nodes, cand_node_repr=cand_nodes,
            query_atom_mask=query_mask, cand_atom_mask=cand_atom_mask,
            cand_repr=cand_repr, rank_scalar_features=rank_scalars,
            label_features=label_features, assay_family=assay_family,
            candidate_meta=candidate_meta, candidate_mask=candidate_mask,
            pool_tanimoto=pool_tanimoto, pool_labels=pool_labels,
            k_top=self.k_top, pools=list(pools),
            insufficient=torch.tensor(list(insufficient), dtype=torch.bool, device=device),
        )


class GraphEncoderNotSharedError(RuntimeError):
    """检索管线与模型用了不同的 ``GraphMatcherLite`` 实例。

    单独定义异常类型的目的和 :class:`~sparc.data.purge.LeakageAssertionError`
    一样：让"绕过它"成为一个刻意动作。
    """


def assert_graph_matcher_shared(model: Any, pipeline: "RetrievalPipeline") -> None:
    """断言管线与模型共享同一个图编码器实例。

    **为什么这条必须硬断言。** ``GraphMatcherLite.forward`` 只用到
    ``w_q`` / ``w_d`` / ``w_a``；``node_encoder`` / ``edge_encoder`` /
    3 层 GINE 只在 :meth:`~sparc.models.graphmatcher.GraphMatcherLite.encode_graph`
    里被用到，而 ``encode_graph`` 是**管线**在装配候选张量时调用的
    （见 :meth:`RetrievalPipeline._assemble`）。

    于是两个实例时会发生这件事：管线那一份拿到梯度但不在优化器的参数表里，
    模型这一份在参数表里却永远拿不到梯度 —— **18,883 个参数
    （占全部可训练参数的 16.7%、GraphMatcher 的 60.5%）停在随机初始化，一步都不动。**

    最坏的地方在于它**完全静默**：损失照常下降（``w_q``/``w_d``/``w_a``、
    重排器、证据、残差都还在训），参数量仍然是 113,244，没有任何断言会响。
    你只是在训练一个原子级图编码器是冻结噪声的模型 —— 而 §8.3.1 的三条
    保留理由里，第 3 条正是"GraphMatcher 是本方法与 ActFound 之间唯一的
    结构性差异"。编码器不训练，这条论证就空了。

    Args:
        model: ``SparcNP``。
        pipeline: 检索管线。

    Raises:
        GraphEncoderNotSharedError: 两者不是同一个对象。
    """
    if pipeline.graph_matcher is not getattr(model, "graph_matcher", None):
        raise GraphEncoderNotSharedError(
            "RetrievalPipeline 与 SparcNP 使用了不同的 GraphMatcherLite 实例。\n"
            "后果：node_encoder / edge_encoder / 3 层 GINE 共 18,883 个参数"
            "（全部可训练参数的 16.7%）永远拿不到梯度，停在随机初始化，"
            "且损失曲线一切正常、参数量核对也照样通过 —— 这个失败是静默的。\n"
            "正确做法：RetrievalPipeline.for_model(model, k_src=..., k0=..., k_top=...)"
        )


def naive_knn_residual(
    pool_tanimoto: np.ndarray,
    pool_labels: np.ndarray,
    base_prediction: float,
    k: int = 16,
    temperature: float = 0.1,
) -> float:
    """朴素 kNN 残差 —— H1 的对照组 (§13.3)。

    这是 Stage 1 唯一需要的检索逻辑：**不需要门控、不需要重排器、
    不需要图匹配、不需要交叉拟合**，一天能跑完，却决定整个项目
    有没有立足点。

    Args:
        pool_tanimoto: ``(K0,)``。
        pool_labels: ``(K0,)``。
        base_prediction: Base 预测值。
        k: 取前 k 个邻居。
        temperature: softmax 温度。

    Returns:
        残差（加到 base 上）：``Σ w_j y_j − base``，权重 ``w ∝ exp(tan/τ)``。
    """
    tan = np.asarray(pool_tanimoto, dtype=np.float64)
    labels = np.asarray(pool_labels, dtype=np.float64)
    if tan.size == 0:
        return 0.0
    top = np.argsort(-tan)[:min(k, tan.size)]
    weights = np.exp(tan[top] / temperature)
    weights = weights / max(weights.sum(), 1e-12)
    return float((weights * labels[top]).sum() - base_prediction)
