"""解析式参数预算计算 (§8.3.6)。

**为什么需要它**：本项目在两台机器上跑 —— 树莓派上没有 torch，
但参数预算是规范的硬约束（113,244，零容差），必须能在写规范的那台
机器上验证。本模块用纯 Python 从维度契约算出每个模块的参数量，
``tests/test_param_budget.py`` 再把三个来源对齐：

    解析计算  ==  frozen_hparams.yaml 的 param_budget  ==  实际 nn.Module

三者任一不一致即测试失败。有 torch 时三方比对，无 torch 时两方比对。
"""

from __future__ import annotations

from typing import Any, Dict


def _linear(d_in: int, d_out: int, bias: bool = True) -> int:
    """``nn.Linear`` 的参数量。"""
    return d_in * d_out + (d_out if bias else 0)


def _layer_norm(d: int) -> int:
    """``nn.LayerNorm`` 的参数量（weight + bias）。"""
    return 2 * d


def theta_b_breakdown(dims: Any) -> Dict[str, int]:
    """Θ_B 的分项参数量 (§8.2)。

    Args:
        dims: :class:`~sparc.common.config.DimConfig`。

    Returns:
        分项字典，含 ``total``。
    """
    items = {
        # z_q' = LN(W_AB z_q + b_AB)
        "ligand_proj+norm": _linear(dims.d_pca_ligand, dims.d_ligand_hidden) + _layer_norm(dims.d_ligand_hidden),
        # c_t = LN(W_p p_pca + b_p)
        "protein_proj+norm": _linear(dims.d_pca_protein, dims.d_protein_ctx) + _layer_norm(dims.d_protein_ctx),
        # 低秩双线性 U_b, V_b（无偏置）
        "bilinear": dims.d_ligand_hidden * dims.bilinear_rank + dims.d_protein_ctx * dims.bilinear_rank,
        # h_q = LN(W_Bf x_q + b_Bf)
        "fusion+norm": _linear(dims.d_base_fusion_in, dims.d_base_hidden) + _layer_norm(dims.d_base_hidden),
        # r = Dropout(GELU(W_B1 h_q + b_B1))
        "head_hidden": _linear(dims.d_base_hidden, dims.d_base_head),
        # [μ, logσ²]
        "head_out": _linear(dims.d_base_head, 2),
    }
    items["total"] = sum(items.values())
    return items


def graphmatcher_breakdown(dims: Any, n_gine_layers: int = 3) -> Dict[str, int]:
    """GraphMatcher-lite 的分项参数量 (§8.3.1)。

    每个 GINE 层 = Linear(d,d) + eps(1) + LayerNorm(d)。
    §8.3.1 给的 12,867 = 3 × (4,160 + 1 + 128)。
    """
    gine_layer = _linear(dims.d_gm_hidden, dims.d_gm_hidden) + 1 + _layer_norm(dims.d_gm_hidden)
    items = {
        "node_encoder": _linear(dims.d_gm_node_feat, dims.d_gm_hidden),
        "edge_encoder": _linear(dims.d_gm_edge_feat, dims.d_gm_hidden),
        "gine_layers": n_gine_layers * gine_layer,
        # W_Q, W_D ∈ R^{64×32}，无偏置
        "w_q+w_d": 2 * _linear(dims.d_gm_hidden, dims.d_gm_proj, bias=False),
        # W_A: 4*64 = 256 → 32（对齐摘要的四项拼接）
        "w_a": _linear(4 * dims.d_gm_hidden, dims.d_align_summary),
    }
    items["total"] = sum(items.values())
    return items


def reranker_breakdown(dims: Any) -> Dict[str, int]:
    """Reranker-lite 的分项参数量 (§8.3.2)。"""
    items = {
        "ligand_proj": _linear(dims.d_query_repr, dims.d_gm_proj),      # W_z: 256→32
        "input_norm": _layer_norm(dims.d_rank_feature),                 # LN(103)
        "hidden": _linear(dims.d_rank_feature, dims.d_rank_hidden),     # W_r1: 103→64
        "score": _linear(dims.d_rank_hidden, 1),                        # w_r2: 64→1
    }
    items["total"] = sum(items.values())
    return items


def evidence_breakdown(dims: Any, n_label_features: int = 5) -> Dict[str, int]:
    """Evidence-lite 的分项参数量 (§8.3.3)。"""
    items = {
        "label_proj": _linear(n_label_features, dims.d_evidence_y),                    # W_y: 5→16
        "assay_embedding": dims.n_assay_family * dims.d_assay_family_emb,              # 32×8
        "evidence_proj+norm": _linear(dims.d_evidence_raw, dims.d_evidence_ctx)
                              + _layer_norm(dims.d_evidence_ctx),                      # W_e: 96→64 + LN
    }
    items["total"] = sum(items.values())
    return items


def residual_breakdown(dims: Any) -> Dict[str, int]:
    """Residual 的分项参数量 (§8.3.4)。"""
    items = {
        "u": dims.d_query_repr * dims.residual_rank,        # 256×8
        "v": dims.d_evidence_ctx * dims.residual_rank,      # 64×8
        "w_delta": dims.residual_rank,
        "w_context": dims.d_evidence_ctx,
        "bias": 1,
    }
    items["total"] = sum(items.values())
    return items


def uncertainty_breakdown(dims: Any) -> Dict[str, int]:
    """不确定性头的参数量 (§8.3.6 的 322)。

    输入 ``[c_R (64); pooled_raw (96)] = 160``，输出 2 个标量。
    """
    d_in = dims.d_evidence_ctx + dims.d_evidence_raw     # 64 + 96 = 160
    items = {"head": _linear(d_in, 2)}                   # 160*2 + 2 = 322
    items["total"] = sum(items.values())
    return items


def gate_breakdown(dims: Any) -> Dict[str, int]:
    """门控参数量 (§8.3.5)：28 权重 + 1 截距。"""
    items = {"linear": _linear(dims.n_gate_features, 1)}
    items["total"] = sum(items.values())
    return items


def full_budget(dims: Any, n_gine_layers: int = 3) -> Dict[str, Any]:
    """完整参数预算。

    Args:
        dims: :class:`~sparc.common.config.DimConfig`。
        n_gine_layers: GINE 层数（3，§8.3.1 从 6 层剪到 3 层）。

    Returns:
        含 ``theta_b`` / ``theta_r`` / ``total_trainable`` 与各模块分项的字典。
    """
    theta_b = theta_b_breakdown(dims)
    graphmatcher = graphmatcher_breakdown(dims, n_gine_layers)
    reranker = reranker_breakdown(dims)
    evidence = evidence_breakdown(dims)
    residual = residual_breakdown(dims)
    uncertainty = uncertainty_breakdown(dims)
    gate = gate_breakdown(dims)

    theta_r_total = (graphmatcher["total"] + reranker["total"] + evidence["total"]
                     + residual["total"] + gate["total"] + uncertainty["total"])

    return {
        "theta_b": theta_b["total"],
        "graphmatcher": graphmatcher["total"],
        "reranker": reranker["total"],
        "evidence": evidence["total"],
        "residual": residual["total"],
        "gate": gate["total"],
        "uncertainty_head": uncertainty["total"],
        "theta_r": theta_r_total,
        "total_trainable": theta_b["total"] + theta_r_total,
        "breakdown": {
            "theta_b": theta_b, "graphmatcher": graphmatcher, "reranker": reranker,
            "evidence": evidence, "residual": residual, "uncertainty_head": uncertainty, "gate": gate,
        },
    }


def supervision_ratio(theta_r: int, n_train_queries: int, n_memory_labels: int = 2000) -> Dict[str, float]:
    """样本 / 参数 平衡核算 (§8.3.6)。

    "独立监督" = 训练查询数 + 兼容记忆库标签数。理由：``L_rank`` 的
    3,510×256 ≈ 90 万个排序目标全部是 3,510 + |M| 个底层标签的确定性函数，
    信息自由度以后者为上界。

    Args:
        theta_r: Θ_R 参数量。
        n_train_queries: Θ_R 的训练查询数。
        n_memory_labels: 兼容记忆库标签数的保守估计。

    Returns:
        ``{"independent_supervision": .., "theta_r_per_supervision": ..}``。
        目标比值 ≈ 10 : 1。
    """
    independent = n_train_queries + n_memory_labels
    return {
        "independent_supervision": float(independent),
        "theta_r_per_supervision": round(theta_r / independent, 2) if independent else float("inf"),
        "target_ratio": 10.0,
    }
