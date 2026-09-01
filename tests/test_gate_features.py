"""28 维支持度特征的行为 (§8.3.5, §8.3.2)。

重点检验那条最容易被实现错的规则：**冲突类特征在 K₀ 候选池上算，
支持类在 Top-K 上算**。若冲突类也在 Top-K 上算，重排器
（目标 ``ρ_i = −|y_i − y_q|``）会把活性悬崖邻居排到后面，
门控就永远看不到"这个查询周围有活性悬崖"这件事。
"""

from __future__ import annotations

import numpy as np
import pytest

from sparc.retrieval.features import (
    FEATURE_ORDER,
    GateFeatureExtractor,
    HIGH_TANIMOTO_THRESHOLD,
    tanimoto_top1_baseline,
)


@pytest.fixture
def extractor(config):
    """按 manifest 顺序构造提取器。"""
    return GateFeatureExtractor(28, config.gate_manifest.names)


def _extract(extractor, *, k=16, k0=256, seed=0, pool_labels=None, topk_labels=None,
             topk_tanimoto=None, pool_tanimoto=None, base_ensemble_variance=0.05,
             base_aleatoric_sigma=0.4):
    """跑一次提取，允许覆写关键输入。"""
    rng = np.random.default_rng(seed)
    return extractor.extract(
        attention=rng.random(k),
        topk_scores=rng.normal(size=k),
        pool_scores=rng.normal(size=k0),
        topk_tanimoto=rng.random(k) if topk_tanimoto is None else topk_tanimoto,
        pool_tanimoto=rng.random(k0) if pool_tanimoto is None else pool_tanimoto,
        topk_emb_sim=rng.random(k),
        topk_match_conf=rng.random(k),
        topk_labels=rng.normal(6, 1, k) if topk_labels is None else topk_labels,
        pool_labels=rng.normal(6, 1, k0) if pool_labels is None else pool_labels,
        topk_scaffolds=[f"FAM{i % 4}" for i in range(k)],
        topk_assay_compatible=np.ones(k),
        topk_fingerprints=(rng.random((k, 64)) > 0.5).astype(np.uint8),
        pool_fingerprints=(rng.random((k0, 64)) > 0.5).astype(np.uint8),
        base_ensemble_variance=base_ensemble_variance,
        base_aleatoric_sigma=base_aleatoric_sigma,
    )


class TestFeatureContract:
    """特征清单契约。"""

    def test_order_matches_manifest(self, config):
        """FEATURE_ORDER 必须与 manifest 完全一致 —— 顺序即 w_g 分量顺序。"""
        assert list(FEATURE_ORDER) == list(config.gate_manifest.names)

    def test_mismatched_order_rejected(self):
        """顺序不一致必须报错，而不是悄悄按名字对齐。"""
        wrong = list(FEATURE_ORDER)
        wrong[0], wrong[1] = wrong[1], wrong[0]
        with pytest.raises(ValueError, match="顺序"):
            GateFeatureExtractor(28, wrong)

    def test_output_shape_and_finiteness(self, extractor):
        """输出必须是 28 维且全部有限（NaN 会让 logistic 直接失效）。"""
        bundle = _extract(extractor)
        assert bundle.values.shape == (28,)
        assert np.isfinite(bundle.values).all()


class TestPoolSeparation:
    """§8.3.2 的两池分工。"""

    def test_conflict_features_track_pool_not_topk(self, extractor):
        """邻居标签方差必须跟随 K₀ 池变化，而不是 Top-K。"""
        k, k0 = 16, 256
        rng = np.random.default_rng(1)
        tight_topk = np.full(k, 6.0)                       # Top-K 标签完全一致
        wide_pool = rng.normal(6, 3, k0)                   # 候选池标签高度分散

        bundle = _extract(extractor, topk_labels=tight_topk, pool_labels=wide_pool)
        idx_var = FEATURE_ORDER.index("neighbor_label_variance")
        idx_range = FEATURE_ORDER.index("neighbor_label_range")

        assert bundle.values[idx_var] > 1.0, (
            "邻居标签方差被算在了 Top-K 上 —— 重排器抹平后它恒为 0，"
            "门控将看不到冲突信号（§8.3.2）"
        )
        assert bundle.values[idx_range] > 3.0

    def test_var_ratio_detects_reranker_flattening(self, extractor):
        """第 25 维 ``Var_K/Var_K0`` 应在重排器抹平方差时接近 0。"""
        rng = np.random.default_rng(2)
        bundle = _extract(extractor, topk_labels=np.full(16, 6.0), pool_labels=rng.normal(6, 3, 256))
        idx = FEATURE_ORDER.index("var_ratio_k_over_k0")
        assert bundle.values[idx] < 0.05

    def test_dropped_high_tanimoto_counts_reranker_removals(self, extractor):
        """第 26 维应统计被重排器移除的高 Tanimoto 邻居。"""
        k, k0 = 16, 256
        pool_tan = np.concatenate([np.full(50, 0.9), np.full(k0 - 50, 0.1)])
        topk_tan = np.full(k, 0.2)                          # Top-K 里一个高相似的都没有
        bundle = _extract(extractor, topk_tanimoto=topk_tan, pool_tanimoto=pool_tan)
        idx = FEATURE_ORDER.index("dropped_high_tanimoto_cnt")
        assert bundle.values[idx] == 50
        assert HIGH_TANIMOTO_THRESHOLD == 0.70


class TestFeatureSemantics:
    """个别特征的语义正确性。"""

    def test_alpha_ess_bounds(self, extractor):
        """ESS ∈ [1, K]：均匀注意力时为 K，退化到单点时为 1。"""
        k = 16
        rng = np.random.default_rng(3)
        uniform = extractor.extract(
            attention=np.full(k, 1.0 / k), topk_scores=rng.normal(size=k),
            pool_scores=rng.normal(size=256), topk_tanimoto=rng.random(k),
            pool_tanimoto=rng.random(256), topk_emb_sim=rng.random(k),
            topk_match_conf=rng.random(k), topk_labels=rng.normal(6, 1, k),
            pool_labels=rng.normal(6, 1, 256), topk_scaffolds=["A"] * k,
            topk_assay_compatible=np.ones(k),
        )
        idx = FEATURE_ORDER.index("alpha_ess")
        assert abs(uniform.values[idx] - k) < 1e-6

        peaked = np.zeros(k)
        peaked[0] = 1.0
        one_point = extractor.extract(
            attention=peaked, topk_scores=rng.normal(size=k), pool_scores=rng.normal(size=256),
            topk_tanimoto=rng.random(k), pool_tanimoto=rng.random(256), topk_emb_sim=rng.random(k),
            topk_match_conf=rng.random(k), topk_labels=rng.normal(6, 1, k),
            pool_labels=rng.normal(6, 1, 256), topk_scaffolds=["A"] * k,
            topk_assay_compatible=np.ones(k),
        )
        assert abs(one_point.values[idx] - 1.0) < 1e-3

    def test_rank_zscore_is_scale_invariant(self, extractor):
        """§8.3.5 删旧第 1–4 维的理由：分数尺度不可辨识。

        z-score 分位数在分数整体平移/缩放后必须不变。
        """
        rng = np.random.default_rng(4)
        pool = rng.normal(size=256)
        common = dict(
            attention=rng.random(16), topk_scores=rng.normal(size=16),
            topk_tanimoto=rng.random(16), pool_tanimoto=rng.random(256),
            topk_emb_sim=rng.random(16), topk_match_conf=rng.random(16),
            topk_labels=rng.normal(6, 1, 16), pool_labels=rng.normal(6, 1, 256),
            topk_scaffolds=["A"] * 16, topk_assay_compatible=np.ones(16),
        )
        a = extractor.extract(pool_scores=pool, **common)
        b = extractor.extract(pool_scores=pool * 3.0 + 7.0, **common)
        for name in ("rank_zscore_q50", "rank_zscore_q75", "rank_zscore_q90"):
            i = FEATURE_ORDER.index(name)
            assert abs(a.values[i] - b.values[i]) < 1e-6, f"{name} 不是尺度不变的"

    def test_unique_scaffold_ratio(self, extractor):
        """唯一骨架比 = 唯一家族数 / K。"""
        bundle = _extract(extractor)         # topk_scaffolds 为 4 个家族循环
        idx = FEATURE_ORDER.index("unique_scaffold_ratio")
        assert abs(bundle.values[idx] - 4 / 16) < 1e-9

    def test_activity_cliff_score_positive_when_cliffs_present(self, extractor):
        """构造明显的活性悬崖时，第 24 维必须 > 0。"""
        k0 = 64
        rng = np.random.default_rng(5)
        # 全部分子指纹几乎相同（高相似），但标签分成相差 3 log 的两群
        fingerprint = (rng.random(64) > 0.5).astype(np.uint8)
        pool_fp = np.tile(fingerprint, (k0, 1))
        labels = np.concatenate([np.full(k0 // 2, 4.0), np.full(k0 // 2, 8.0)])
        score = extractor._activity_cliff_score(np.full(k0, 0.95), labels, pool_fp)
        assert score > 0.5

    def test_activity_cliff_score_zero_without_cliffs(self, extractor):
        """标签一致时不应报出悬崖。"""
        k0 = 64
        rng = np.random.default_rng(6)
        pool_fp = (rng.random((k0, 64)) > 0.5).astype(np.uint8)
        assert extractor._activity_cliff_score(np.full(k0, 0.9), np.full(k0, 6.0), pool_fp) == 0.0

    def test_blocked_feature_is_recorded_not_repurposed(self, config):
        """§8.3.5：第 27 维预算不足时标 blocked 并置 0，**不得静默改语义**。"""
        blocked_extractor = GateFeatureExtractor(28, config.gate_manifest.names, base_ensemble_enabled=False)
        bundle = _extract(blocked_extractor, base_ensemble_variance=0.9)
        idx = FEATURE_ORDER.index("base_ensemble_variance")
        assert bundle.values[idx] == 0.0
        assert "base_ensemble_variance" in bundle.blocked

    def test_tanimoto_top1_baseline(self):
        """H2 判据 (a) 的对照基线。"""
        assert tanimoto_top1_baseline(np.array([0.1, 0.9, 0.4])) == 0.9
        assert tanimoto_top1_baseline(np.zeros(0)) == 0.0


class TestBatchMatrix:
    """批量堆叠。"""

    def test_batch_matrix_shape(self, extractor):
        """N 个 bundle 堆成 (N, 28)。"""
        bundles = [_extract(extractor, seed=s) for s in range(5)]
        assert extractor.batch_matrix(bundles).shape == (5, 28)

    def test_empty_batch(self, extractor):
        """空输入返回 (0, 28)，不报错。"""
        assert extractor.batch_matrix([]).shape == (0, 28)
