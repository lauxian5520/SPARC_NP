"""28 维支持度特征 x_g (§8.3.5)。

**计算池的分工是本模块的要点** (§8.3.2 的"排序目标副作用修正")：

    冲突类特征（17–20, 24–26 维）在 K₀=256 候选池上计算
    支持类特征（1–16, 21–23 维）在 Top-K=16 上计算
    第 25 维 Var_K(y)/Var_{K₀}(y) 跨两池

理由：``L_rank`` 用 ``ρ_i = −|y_i − y_q|`` 会教重排器把活性悬崖邻居
排到后面，从而**抹平门控赖以工作的冲突特征**。若冲突类也在 Top-K 上算，
门控就永远看不到"这个查询周围有活性悬崖"这件事。

已删除的旧维度（不要加回来，§8.3.5）：
* rerank 分数的 raw max/mean/std/gap —— ``L_rank`` 平移不变 ⇒ 分数尺度
  不可辨识 ⇒ 跨 fold/seed 不可比。本模块只用 **z-score 分位数**；
* ``σ(s)`` 超过 0.3/0.5/0.7 的比例 —— 同上；
* context similarity 的 mean/min、endpoint-compatible ratio —— 硬过滤后恒为 1；
* 3D disagreement —— no-3D 模式下恒为 0。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from sparc.chem.fingerprints import tanimoto_matrix
from sparc.common.logging_utils import get_logger

_LOGGER = get_logger(__name__)

# 顺序必须与 gate_feature_manifest.yaml 完全一致
FEATURE_ORDER: tuple[str, ...] = (
    "alpha_entropy", "alpha_ess", "score_margin_norm",
    "tanimoto_max", "tanimoto_mean", "tanimoto_std", "tanimoto_top1_top2_gap",
    "emb_sim_max", "emb_sim_mean", "emb_sim_std",
    "rank_zscore_q50", "rank_zscore_q75", "rank_zscore_q90",
    "match_conf_max", "match_conf_mean", "match_conf_std",
    "neighbor_label_variance", "neighbor_label_mad", "neighbor_label_range",
    "unique_scaffold_ratio",
    "pairwise_tanimoto_mean", "pairwise_tanimoto_std",
    "assay_compatible_ratio", "activity_cliff_score",
    "var_ratio_k_over_k0", "dropped_high_tanimoto_cnt",
    "base_ensemble_variance", "base_aleatoric_sigma",
)

HIGH_TANIMOTO_THRESHOLD = 0.70    # 第 26 维：被重排器移除的高 Tanimoto 邻居阈值


@dataclass
class GateFeatureBundle:
    """一个查询的 28 维特征及其可追溯来源。"""

    values: np.ndarray                      # (28,)
    names: Sequence[str] = FEATURE_ORDER
    blocked: List[str] = field(default_factory=list)   # 预算不足被标 blocked 的维度

    def as_dict(self) -> Dict[str, float]:
        """名字 → 值。"""
        return {name: float(v) for name, v in zip(self.names, self.values)}


class GateFeatureExtractor:
    """按 manifest 顺序计算 28 维支持度特征。"""

    def __init__(
        self,
        n_features: int = 28,
        feature_names: Optional[Sequence[str]] = None,
        base_ensemble_enabled: bool = True,
    ) -> None:
        """
        Args:
            n_features: 特征维数（28）。
            feature_names: 来自 manifest 的特征名；用于顺序校验。
            base_ensemble_enabled: 第 27 维（Base 5-seed 集成方差）是否可用。
                §8.3.5：预算不足时在 ``pilot.yaml`` 中标 ``blocked``，
                **不得静默改语义** —— 因此这里置 0 并记入 ``blocked`` 列表，
                而不是偷偷换成别的量。
        """
        self.n_features = n_features
        self.names = list(feature_names or FEATURE_ORDER)
        if len(self.names) != n_features:
            raise ValueError(f"特征名数量 {len(self.names)} != n_features {n_features}")
        if feature_names is not None and list(feature_names) != list(FEATURE_ORDER):
            raise ValueError(
                "gate_feature_manifest.yaml 的特征顺序与 features.py 的 FEATURE_ORDER 不一致。"
                "顺序即 w_g 的分量顺序，Figure 2 直接按此顺序标注 —— 必须同步修改两处。"
            )
        self.base_ensemble_enabled = base_ensemble_enabled

    # ------------------------------------------------------------------
    def extract(
        self,
        attention: np.ndarray,
        topk_scores: np.ndarray,
        pool_scores: np.ndarray,
        topk_tanimoto: np.ndarray,
        pool_tanimoto: np.ndarray,
        topk_emb_sim: np.ndarray,
        topk_match_conf: np.ndarray,
        topk_labels: np.ndarray,
        pool_labels: np.ndarray,
        topk_scaffolds: Sequence[str],
        topk_assay_compatible: np.ndarray,
        topk_fingerprints: Optional[np.ndarray] = None,
        pool_fingerprints: Optional[np.ndarray] = None,
        base_ensemble_variance: float = 0.0,
        base_aleatoric_sigma: float = 0.0,
    ) -> GateFeatureBundle:
        """计算一个查询的 28 维特征。

        Args:
            attention: ``(K,)`` 证据注意力 α。
            topk_scores: ``(K,)`` Top-K 重排分数。
            pool_scores: ``(K0,)`` 候选池全部分数。
            topk_tanimoto: ``(K,)``。
            pool_tanimoto: ``(K0,)``。
            topk_emb_sim: ``(K,)`` Model A embedding 相似度。
            topk_match_conf: ``(K,)`` 图匹配置信度 ``m_i``。
            topk_labels: ``(K,)`` 邻居 pIC50。
            pool_labels: ``(K0,)`` 候选池邻居 pIC50。
            topk_scaffolds: ``(K,)`` 邻居骨架家族 ID。
            topk_assay_compatible: ``(K,)`` 0/1。
            topk_fingerprints: ``(K, n_bits)``，算邻居两两 Tanimoto 用。
            pool_fingerprints: ``(K0, n_bits)``，算活性悬崖分数用。
            base_ensemble_variance: 第 27 维。
            base_aleatoric_sigma: 第 28 维（Base 的 σ_a）。

        Returns:
            :class:`GateFeatureBundle`。
        """
        blocked: List[str] = []
        values = np.zeros(self.n_features, dtype=np.float32)

        # --- 1–3：注意力形状（Top-K） ---
        alpha = np.asarray(attention, dtype=np.float64)
        alpha = alpha / max(alpha.sum(), 1e-12)
        values[0] = float(-(alpha * np.log(alpha + 1e-12)).sum())            # 熵
        values[1] = float(1.0 / max((alpha ** 2).sum(), 1e-12))              # ESS
        sorted_scores = np.sort(np.asarray(topk_scores, dtype=np.float64))[::-1]
        if sorted_scores.size >= 2:
            spread = sorted_scores[0] - sorted_scores[-1] + 1e-6
            values[2] = float((sorted_scores[0] - sorted_scores[1]) / spread)

        # --- 4–7：Tanimoto（Top-K） ---
        tan = np.asarray(topk_tanimoto, dtype=np.float64)
        values[3] = float(tan.max()) if tan.size else 0.0
        values[4] = float(tan.mean()) if tan.size else 0.0
        values[5] = float(tan.std()) if tan.size > 1 else 0.0
        tan_sorted = np.sort(tan)[::-1]
        values[6] = float(tan_sorted[0] - tan_sorted[1]) if tan.size >= 2 else 0.0

        # --- 8–10：Model A embedding 相似度（Top-K） ---
        emb = np.asarray(topk_emb_sim, dtype=np.float64)
        values[7] = float(emb.max()) if emb.size else 0.0
        values[8] = float(emb.mean()) if emb.size else 0.0
        values[9] = float(emb.std()) if emb.size > 1 else 0.0

        # --- 11–13：rerank 分数在候选池内的 z-score 分位数（K₀） ---
        # 只用 z-score 分位数，不用 raw 分数：L_rank 平移不变 ⇒ 尺度不可辨识
        pool = np.asarray(pool_scores, dtype=np.float64)
        if pool.size > 1:
            zscores = (pool - pool.mean()) / max(pool.std(), 1e-9)
            values[10] = float(np.quantile(zscores, 0.50))
            values[11] = float(np.quantile(zscores, 0.75))
            values[12] = float(np.quantile(zscores, 0.90))

        # --- 14–16：图匹配置信度（Top-K） ---
        conf = np.asarray(topk_match_conf, dtype=np.float64)
        values[13] = float(conf.max()) if conf.size else 0.0
        values[14] = float(conf.mean()) if conf.size else 0.0
        values[15] = float(conf.std()) if conf.size > 1 else 0.0

        # --- 17–19：邻居标签离散度（K₀ —— 冲突类，避开重排器抹平） ---
        pool_y = np.asarray(pool_labels, dtype=np.float64)
        if pool_y.size:
            values[16] = float(pool_y.var())
            values[17] = float(np.median(np.abs(pool_y - np.median(pool_y))))
            values[18] = float(pool_y.max() - pool_y.min())

        # --- 20：唯一骨架比（Top-K） ---
        if len(topk_scaffolds):
            values[19] = float(len(set(topk_scaffolds)) / len(topk_scaffolds))

        # --- 21–22：邻居两两 Tanimoto（Top-K，内聚性） ---
        if topk_fingerprints is not None and len(topk_fingerprints) > 1:
            pairwise = tanimoto_matrix(topk_fingerprints, topk_fingerprints)
            upper = pairwise[np.triu_indices_from(pairwise, k=1)]
            values[20] = float(upper.mean())
            values[21] = float(upper.std())

        # --- 23：assay 兼容比（硬过滤下应恒 1） ---
        compat = np.asarray(topk_assay_compatible, dtype=np.float64)
        values[22] = float(compat.mean()) if compat.size else 1.0

        # --- 24：活性悬崖分数（K₀ —— 冲突类） ---
        values[23] = self._activity_cliff_score(pool_tanimoto, pool_labels, pool_fingerprints)

        # --- 25：Var_K(y) / Var_{K₀}(y)（两池 —— 重排器抹平了多少方差） ---
        topk_y = np.asarray(topk_labels, dtype=np.float64)
        pool_var = float(pool_y.var()) if pool_y.size else 0.0
        values[24] = float(topk_y.var() / pool_var) if pool_var > 1e-12 and topk_y.size else 0.0

        # --- 26：被重排器移除的高 Tanimoto 邻居计数（两池） ---
        pool_tan = np.asarray(pool_tanimoto, dtype=np.float64)
        n_high_pool = int((pool_tan >= HIGH_TANIMOTO_THRESHOLD).sum())
        n_high_topk = int((tan >= HIGH_TANIMOTO_THRESHOLD).sum())
        values[25] = float(max(n_high_pool - n_high_topk, 0))

        # --- 27：Base 集成方差（5 seed） ---
        if self.base_ensemble_enabled:
            values[26] = float(base_ensemble_variance)
        else:
            values[26] = 0.0
            blocked.append("base_ensemble_variance")

        # --- 28：Base aleatoric 不确定性 σ_a ---
        values[27] = float(base_aleatoric_sigma)

        return GateFeatureBundle(values=values, names=tuple(self.names), blocked=blocked)

    # ------------------------------------------------------------------
    @staticmethod
    def _activity_cliff_score(
        pool_tanimoto: np.ndarray,
        pool_labels: np.ndarray,
        pool_fingerprints: Optional[np.ndarray],
        similarity_threshold: float = 0.60,
    ) -> float:
        """活性悬崖分数：结构高度相似但标签差异大的邻居对占比 × 平均标签差。

        与 MoleculeACE 的 ``cliff_mol`` 定义同源（相似度高 + 活性差 ≥ 1 log）。
        在 **K₀ 候选池** 上计算 —— 这正是重排器会抹平的信号。

        Args:
            pool_tanimoto: ``(K0,)`` 候选与查询的 Tanimoto。
            pool_labels: ``(K0,)`` 候选标签。
            pool_fingerprints: ``(K0, n_bits)``；提供时用邻居两两相似度，
                否则退化为"与查询的相似度"版本。
            similarity_threshold: 判定"结构相似"的阈值。

        Returns:
            非负分数；无悬崖时为 0。
        """
        labels = np.asarray(pool_labels, dtype=np.float64)
        if labels.size < 2:
            return 0.0

        if pool_fingerprints is not None and len(pool_fingerprints) == labels.size:
            sims = tanimoto_matrix(pool_fingerprints, pool_fingerprints)
            iu = np.triu_indices_from(sims, k=1)
            pair_sim = sims[iu]
            pair_diff = np.abs(labels[iu[0]] - labels[iu[1]])
        else:
            tan = np.asarray(pool_tanimoto, dtype=np.float64)
            if tan.size != labels.size:
                return 0.0
            # 退化版本：以查询为中心，用"与查询相似度的最小值"近似邻居间相似度
            iu = np.triu_indices(labels.size, k=1)
            pair_sim = np.minimum(tan[iu[0]], tan[iu[1]])
            pair_diff = np.abs(labels[iu[0]] - labels[iu[1]])

        cliff_mask = (pair_sim >= similarity_threshold) & (pair_diff >= 1.0)
        if not cliff_mask.any():
            return 0.0
        return float(cliff_mask.mean() * pair_diff[cliff_mask].mean())

    # ------------------------------------------------------------------
    def batch_matrix(self, bundles: Sequence[GateFeatureBundle]) -> np.ndarray:
        """把多个 bundle 堆成 ``(N, 28)`` 矩阵。"""
        if not bundles:
            return np.zeros((0, self.n_features), dtype=np.float32)
        return np.vstack([b.values for b in bundles]).astype(np.float32)


def tanimoto_top1_baseline(pool_tanimoto: np.ndarray) -> float:
    """H2 判据 (a) 的对照：单一 Tanimoto-Top1 标量基线。

    Args:
        pool_tanimoto: ``(K0,)``。

    Returns:
        Top-1 Tanimoto。28 维门控必须在 OOF AUROC 上**严格优于**它，
        且配对 bootstrap 95% CI 下界 > 0。
    """
    tan = np.asarray(pool_tanimoto, dtype=np.float64)
    return float(tan.max()) if tan.size else 0.0
