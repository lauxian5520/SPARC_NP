"""维度契约与参数预算 (§8.2, §8.3.6)。

**参数预算是硬约束，零容差。** §8.3.6 的 113,244 不是"大约"——
它是"27× 压缩"这个论证的全部内容，也是 §14 的"1 张 GPU 可完成全部
主实验"的直接依据。

本文件做三方比对：

    解析计算（param_budget.py）  ==  frozen_hparams.yaml  ==  实际 nn.Module

无 torch 时做两方比对（树莓派上仍必须绿）；有 torch 时三方全比。
"""

from __future__ import annotations

import pytest

from sparc.models.param_budget import (
    evidence_breakdown,
    full_budget,
    gate_breakdown,
    graphmatcher_breakdown,
    reranker_breakdown,
    residual_breakdown,
    supervision_ratio,
    theta_b_breakdown,
    uncertainty_breakdown,
)
from sparc.retrieval.features import FEATURE_ORDER

from conftest import requires_torch


class TestDimensionContract:
    """§8.2/§8.3 的维度恒等式。"""

    def test_dims_self_consistent(self, dims):
        """维度契约自洽（DimConfig.validate 已在载入时跑过，这里显式再验一次）。"""
        dims.validate()

    def test_bilinear_block(self, dims):
        """``s_bil ⊗`` 展开必须是 rank²（4×4=16），且不引入参数。"""
        assert dims.d_bilinear_block == dims.bilinear_rank ** 2 == 16

    def test_base_fusion_input(self, dims):
        """``x_q = [z_q'(128); c_t(64); s_bil(16)] = 208``。"""
        assert dims.d_base_fusion_in == 208
        assert dims.d_ligand_hidden + dims.d_protein_ctx + dims.d_bilinear_block == 208

    def test_query_repr(self, dims):
        """``v_q = [z_q'(128); h_q(128)] = 256`` —— 重排器 W_z 与残差 U 的共同输入。"""
        assert dims.d_query_repr == 256

    def test_rank_feature(self, dims):
        """``x^rank = 3×32 + 7 = 103``（原 1927，其中 771 维恒为常数）。"""
        assert dims.d_rank_feature == 103

    def test_evidence_raw(self, dims):
        """``e_i^raw = 32+32+16+8+8 = 96``（原 696）。"""
        assert dims.d_evidence_raw == 96

    def test_uncertainty_head_input(self, dims):
        """不确定性头输入 ``[c_R(64); pooled_raw(96)] = 160`` ⇒ 160×2+2 = 322。"""
        assert dims.d_evidence_ctx + dims.d_evidence_raw == 160


class TestGateFeatureManifest:
    """28 维门控特征清单 (§8.3.5)。"""

    def test_count_and_order(self, config):
        """条目数、顺序、参数数必须与 manifest 声明一致。"""
        manifest = config.gate_manifest
        assert manifest.n_features == 28
        assert manifest.raw["n_params"] == 29
        assert list(manifest.names) == list(FEATURE_ORDER)

    def test_sign_priors_are_pm_one(self, config):
        """化学先验符号只能是 ±1（H2 判据 (d) 的判定基础）。"""
        assert set(config.gate_manifest.sign_priors) <= {1, -1}

    def test_conflict_features_use_k0_pool(self, config):
        """§8.3.2：冲突类特征必须在 K₀ 池上计算，否则会被重排器抹平。"""
        manifest = config.gate_manifest
        pools = dict(zip(manifest.names, manifest.pools))
        for name in ("neighbor_label_variance", "neighbor_label_mad",
                     "neighbor_label_range", "activity_cliff_score"):
            assert pools[name] == "k0", f"{name} 必须在 K₀ 候选池上计算（§8.3.2）"

    def test_removed_features_not_reintroduced(self, config):
        """已删除的旧维度不得被加回来（§8.3.5 的删除理由表）。"""
        removed_names = {
            "rerank_score_max", "rerank_score_mean", "rerank_score_std", "rerank_score_gap",
            "context_similarity_mean", "context_similarity_min",
            "endpoint_compatible_ratio", "disagreement_3d",
        }
        assert not (set(config.gate_manifest.names) & removed_names)


class TestParameterBudget:
    """§8.3.6 的参数总账，零容差。"""

    EXPECTED = {
        "theta_b": 57218, "graphmatcher": 31203, "reranker": 15151,
        "evidence": 6688, "residual": 2633, "gate": 29, "uncertainty_head": 322,
        "theta_r": 56026, "total_trainable": 113244,
    }

    def test_analytic_matches_spec(self, dims):
        """解析计算必须精确等于 method.md §8.3.6 的表。"""
        budget = full_budget(dims)
        for key, expected in self.EXPECTED.items():
            assert budget[key] == expected, f"{key}: {budget[key]} != {expected}（§8.3.6 零容差）"

    def test_analytic_matches_frozen_yaml(self, config, dims):
        """解析计算必须精确等于 frozen_hparams.yaml 的 param_budget。"""
        budget = full_budget(dims)
        frozen = config.hparams.param_budget
        assert frozen["tolerance"] == 0, "参数预算容差必须为 0"
        for key in self.EXPECTED:
            assert budget[key] == frozen[key], f"{key}: 解析 {budget[key]} != YAML {frozen[key]}"

    def test_module_breakdowns(self, dims):
        """逐模块分项也要对得上（防止两处错误互相抵消）。"""
        assert theta_b_breakdown(dims)["ligand_proj+norm"] == 16768
        assert theta_b_breakdown(dims)["protein_proj+norm"] == 4288
        assert theta_b_breakdown(dims)["bilinear"] == 768
        assert theta_b_breakdown(dims)["fusion+norm"] == 27008
        assert graphmatcher_breakdown(dims)["gine_layers"] == 12867
        assert graphmatcher_breakdown(dims)["w_a"] == 8224
        assert reranker_breakdown(dims)["ligand_proj"] == 8224
        assert evidence_breakdown(dims)["assay_embedding"] == 256
        assert residual_breakdown(dims)["u"] == 2048
        assert uncertainty_breakdown(dims)["head"] == 322
        assert gate_breakdown(dims)["linear"] == 29

    def test_supervision_ratio_near_target(self, dims):
        """§8.3.6 的样本/参数平衡：70 靶点池下 Θ_R : 独立监督 ≈ 10 : 1。"""
        ratio = supervision_ratio(full_budget(dims)["theta_r"], n_train_queries=3510, n_memory_labels=2000)
        assert 9.0 <= ratio["theta_r_per_supervision"] <= 11.0

    def test_gate_is_29_parameters(self, dims):
        """门控必须正好 29 个参数 —— Figure 2 的存在前提。"""
        assert gate_breakdown(dims)["total"] == 29

    @requires_torch
    def test_torch_modules_match_analytic(self, dims):
        """有 torch 时，实际 nn.Module 的参数量必须等于解析计算。"""
        from sparc.models.base import BasePredictor
        from sparc.models.evidence import EvidenceLite
        from sparc.models.gate import SupportGate
        from sparc.models.graphmatcher import GraphMatcherLite
        from sparc.models.reranker import RerankerLite
        from sparc.models.residual import ResidualHead, UncertaintyHead

        actual = {
            "theta_b": BasePredictor().n_parameters(),
            "graphmatcher": GraphMatcherLite().n_parameters(),
            "reranker": RerankerLite().n_parameters(),
            "evidence": EvidenceLite().n_parameters(),
            "residual": ResidualHead().n_parameters(),
            "gate": SupportGate().n_parameters(),
            "uncertainty_head": UncertaintyHead().n_parameters(),
        }
        for key, value in actual.items():
            assert value == self.EXPECTED[key], f"{key}: 实际模块 {value} != 规范 {self.EXPECTED[key]}"
        assert sum(actual.values()) == self.EXPECTED["total_trainable"]

    @requires_torch
    def test_whitening_has_no_trainable_parameters(self):
        """冻结 PCA 白化必须是纯 buffer —— 否则会污染参数预算。"""
        import numpy as np

        from sparc.models.whitening import FrozenPCAWhitening

        rng = np.random.default_rng(0)
        whitening = FrozenPCAWhitening(16).fit(rng.normal(size=(200, 64)))
        module = whitening.as_torch_module()
        assert sum(p.numel() for p in module.parameters()) == 0
