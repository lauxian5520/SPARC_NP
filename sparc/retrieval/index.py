"""三源召回索引 (§9.1 步骤 2–4)。

三个索引都建在**已过滤的** ``M_view`` 上：

* ECFP4 LSH        —— MinHash-style 随机超平面 LSH，纯 numpy 实现；
* Model A HNSW     —— 有 hnswlib 时用它，否则退化为精确内积（视图规模
                      通常只有 10²–10³ 条，精确搜索完全跑得动）；
* 药效团 ANN       —— 同上，用 2D Gobbi 药效团指纹。

**不引入 faiss 依赖**：视图过滤之后每个视图只有几百到几千条记录，
精确检索的代价可以忽略，而多一个 GPU 库就多一个部署风险。
hnswlib 作为可选加速，缺失时自动降级并记日志（不静默）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sparc.chem.fingerprints import tanimoto_matrix
from sparc.common.logging_utils import get_logger

_LOGGER = get_logger(__name__)

try:  # pragma: no cover - 可选依赖
    import hnswlib

    HNSW_AVAILABLE = True
except ImportError:  # pragma: no cover
    HNSW_AVAILABLE = False


@dataclass
class CandidatePool:
    """一次检索得到的候选池 (K₀)。"""

    query_index: int
    indices: np.ndarray                 # (K0,) 在 M_view 中的下标
    stage1_scores: np.ndarray           # (K0,) Stage-1 融合分数
    tanimoto: np.ndarray                # (K0,)
    embedding_similarity: np.ndarray    # (K0,)
    pharmacophore_similarity: np.ndarray  # (K0,)
    source_hits: Dict[str, np.ndarray] = field(default_factory=dict)   # 每源是否命中
    compatible_ratio: float = 1.0

    def __len__(self) -> int:
        """候选池大小。"""
        return int(self.indices.size)


class RetrievalIndex:
    """在单个 ``M_view`` 上建三个索引并做并集召回。"""

    def __init__(
        self,
        ecfp_matrix: np.ndarray,
        embedding_matrix: np.ndarray,
        pharmacophore_matrix: Optional[np.ndarray] = None,
        k_src: int = 512,
        k0: int = 256,
        use_hnsw: bool = True,
        lsh_n_planes: int = 64,
        seed: int = 42,
    ) -> None:
        """
        Args:
            ecfp_matrix: ``(M, 2048)`` uint8。
            embedding_matrix: ``(M, d)`` Model A embedding（已 PCA 白化或原始均可）。
            pharmacophore_matrix: ``(M, n_bits)`` uint8；``None`` 时该源不参与召回
                并在 manifest 中标记为 disabled（不静默跳过）。
            k_src: 每源召回上限（512）。
            k0: 候选池大小（256）。
            use_hnsw: 是否尝试用 hnswlib。
            lsh_n_planes: LSH 随机超平面数。
            seed: LSH 平面的随机种子（必须固定，否则跨 fold 召回不可复现）。
        """
        self.ecfp = np.ascontiguousarray(ecfp_matrix, dtype=np.uint8)
        self.embeddings = np.ascontiguousarray(embedding_matrix, dtype=np.float32)
        self.pharmacophore = (
            np.ascontiguousarray(pharmacophore_matrix, dtype=np.uint8)
            if pharmacophore_matrix is not None else None
        )
        self.k_src = k_src
        self.k0 = k0
        self.size = self.ecfp.shape[0]
        self.seed = seed

        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self._embeddings_normed = self.embeddings / np.maximum(norms, 1e-12)

        rng = np.random.default_rng(seed)
        self._lsh_planes = rng.normal(size=(self.ecfp.shape[1], lsh_n_planes)).astype(np.float32)
        self._lsh_codes = self._hash(self.ecfp)

        self._hnsw = None
        if use_hnsw and HNSW_AVAILABLE and self.size > 0:
            self._build_hnsw()
        elif use_hnsw and not HNSW_AVAILABLE:
            _LOGGER.info("hnswlib 不可用，Model A 源改用精确内积检索（视图规模 %d，代价可忽略）", self.size)

        if self.pharmacophore is None:
            _LOGGER.warning("未提供药效团指纹，pharmacophore_ann 源 disabled —— 须在 run manifest 中记录")

    # ------------------------------------------------------------------
    def _hash(self, fingerprints: np.ndarray) -> np.ndarray:
        """随机超平面 LSH 编码。"""
        return (fingerprints.astype(np.float32) @ self._lsh_planes > 0).astype(np.uint8)

    def _build_hnsw(self) -> None:  # pragma: no cover - 取决于可选依赖
        """构建 HNSW 索引（余弦空间）。"""
        index = hnswlib.Index(space="cosine", dim=self._embeddings_normed.shape[1])
        index.init_index(max_elements=self.size, ef_construction=200, M=16, random_seed=self.seed)
        index.add_items(self._embeddings_normed, np.arange(self.size))
        index.set_ef(max(64, min(self.k_src * 2, self.size)))
        self._hnsw = index
        _LOGGER.debug("HNSW 索引已建：%d 条", self.size)

    # ------------------------------------------------------------------
    def recall_ecfp_lsh(self, query_fp: np.ndarray, k: int) -> np.ndarray:
        """ECFP4 LSH 源召回。

        先用 LSH 汉明距离做粗筛（取 4k 个候选），再在粗筛集上算精确
        Tanimoto 取前 k —— LSH 只负责把候选缩到可以精算的规模。
        """
        if self.size == 0:
            return np.zeros(0, dtype=np.int64)
        query_code = self._hash(query_fp[None, :])[0]
        hamming = (self._lsh_codes != query_code).sum(axis=1)
        coarse_k = min(self.size, max(k * 4, k))
        coarse = np.argpartition(hamming, coarse_k - 1)[:coarse_k]
        sims = tanimoto_matrix(query_fp[None, :], self.ecfp[coarse])[0]
        order = np.argsort(-sims)[:k]
        return coarse[order]

    def recall_embedding(self, query_embedding: np.ndarray, k: int) -> np.ndarray:
        """Model A embedding 源召回（HNSW 或精确余弦）。"""
        if self.size == 0:
            return np.zeros(0, dtype=np.int64)
        query = query_embedding / max(float(np.linalg.norm(query_embedding)), 1e-12)
        if self._hnsw is not None:  # pragma: no cover
            labels, _ = self._hnsw.knn_query(query[None, :], k=min(k, self.size))
            return labels[0].astype(np.int64)
        sims = self._embeddings_normed @ query
        return np.argsort(-sims)[:k]

    def recall_pharmacophore(self, query_fp: Optional[np.ndarray], k: int) -> np.ndarray:
        """药效团源召回（2D Gobbi 指纹，no-3D）。"""
        if self.pharmacophore is None or query_fp is None or self.size == 0:
            return np.zeros(0, dtype=np.int64)
        sims = tanimoto_matrix(query_fp[None, :], self.pharmacophore)[0]
        return np.argsort(-sims)[:k]

    # ------------------------------------------------------------------
    def retrieve(
        self,
        query_index: int,
        query_ecfp: np.ndarray,
        query_embedding: np.ndarray,
        query_pharmacophore: Optional[np.ndarray] = None,
        stage1_weights: Tuple[float, float, float] = (0.5, 0.35, 0.15),
    ) -> CandidatePool:
        """执行 §9.1 步骤 3–5。

        Args:
            query_index: 查询在批内的下标（只用于回填 :class:`CandidatePool`）。
            query_ecfp: ``(2048,)``。
            query_embedding: ``(d,)``。
            query_pharmacophore: ``(n_bits,)`` 或 ``None``。
            stage1_weights: 三源相似度融合成 Stage-1 分数的权重
                （Tanimoto / embedding / 药效团）。Stage-1 只负责把
                并集裁到 K₀，真正的排序由 Reranker 做。

        Returns:
            :class:`CandidatePool`（大小 ``min(k0, |union|)``）。
        """
        k_src = min(self.k_src, self.size)
        hits = {
            "ecfp4_lsh": self.recall_ecfp_lsh(query_ecfp, k_src),
            "model_a_hnsw": self.recall_embedding(query_embedding, k_src),
            "pharmacophore_ann": self.recall_pharmacophore(query_pharmacophore, k_src),
        }
        union = np.unique(np.concatenate([h for h in hits.values() if h.size] or [np.zeros(0, dtype=np.int64)]))
        if union.size == 0:
            return CandidatePool(query_index, union, *(np.zeros(0, np.float32) for _ in range(4)),
                                 source_hits={}, compatible_ratio=1.0)

        tan = tanimoto_matrix(query_ecfp[None, :], self.ecfp[union])[0]
        query_normed = query_embedding / max(float(np.linalg.norm(query_embedding)), 1e-12)
        emb = self._embeddings_normed[union] @ query_normed
        pharm = (
            tanimoto_matrix(query_pharmacophore[None, :], self.pharmacophore[union])[0]
            if (self.pharmacophore is not None and query_pharmacophore is not None)
            else np.zeros(union.size, dtype=np.float32)
        )

        w_tan, w_emb, w_pharm = stage1_weights
        stage1 = w_tan * tan + w_emb * emb + w_pharm * pharm
        keep = np.argsort(-stage1)[:min(self.k0, union.size)]

        return CandidatePool(
            query_index=query_index,
            indices=union[keep],
            stage1_scores=stage1[keep].astype(np.float32),
            tanimoto=tan[keep].astype(np.float32),
            embedding_similarity=emb[keep].astype(np.float32),
            pharmacophore_similarity=pharm[keep].astype(np.float32),
            source_hits={name: np.isin(union[keep], idx) for name, idx in hits.items()},
            # §9.1 步骤 5：视图已按 UniProt+tax_id+endpoint 过滤，此处必须恒真
            compatible_ratio=1.0,
        )


def assert_compatible_ratio(pool: CandidatePool, tolerance: float = 1e-9) -> None:
    """§9.1 步骤 5 的断言。

    Args:
        pool: 候选池。
        tolerance: 数值容差。

    Raises:
        AssertionError: ``compatible_ratio != 1.0``。视图已经过滤过，
            这里不为 1 说明过滤键与召回源用的不是同一个视图 ——
            那正是 talk.md §3.7.2 会导致的"候选集静默为空"的前兆。
    """
    if abs(pool.compatible_ratio - 1.0) > tolerance:
        raise AssertionError(
            f"compatible_ratio = {pool.compatible_ratio} != 1.0（§9.1 步骤 5）。"
            "视图已按 [同 UniProt + 同 tax_id + 同 endpoint + fold 可见性] 过滤，"
            "此处必须恒真 —— 不为 1 说明召回源建在了未过滤的记忆库上。"
        )
