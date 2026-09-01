"""SPARC-NP 组合模型 (§4.1)。

**路由的自然参数，不是两个模型的混合**::

    η(q, t) = η_base(q, t) + 1[ρ(t)=B] · g̃(x_g) · Δ(q, t, R)
    ŷ = ψ_t(η)        ψ_t = identity（回归）/ sigmoid（二分类）
    g̃ = g · 1[g ≥ λ],   g = σ(w_gᵀ x_g + b_g)

四个组件通过 ``d_i = ℓ(f₀ + g̃Δ) − ℓ(f₀)`` 这一个量真实耦合 (§11.3)：
* 拿掉冻结的 Base ⇒ ``d_i`` 无定义 ⇒ 效用标签不存在 ⇒ 门控无法训练；
* 把残差换成第二个完整预测器 ⇒ ``g=0`` 不再退化回 Base ⇒
  ``d_i`` 不再度量"检索的边际效应"；
* 拿掉门控 ⇒ 没有逐样本风险可控 ⇒ LTT 无对象。

**任务路由 ρ(t)**：B 类（共有任务）走完整路径；C 类（天然产物专属）
部署默认走 ``INV-C`` 硬隔离（``g̃ ≡ 0``，不访问记忆库），
诊断分支 ``C_FORCED_RETRIEVAL`` 例外 (§13.4)。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn

from sparc.common.logging_utils import get_logger
from sparc.models.base import BaseOutput, BasePredictor
from sparc.models.evidence import EvidenceLite
from sparc.models.gate import SupportGate
from sparc.models.graphmatcher import GraphMatcherLite
from sparc.models.reranker import RerankerLite
from sparc.models.residual import ResidualHead, UncertaintyHead

_LOGGER = get_logger(__name__)

TASK_TYPE_B = "B"      # 共有任务（药物/天然产物共享的靶点活性）
TASK_TYPE_C = "C"      # 天然产物专属任务


@dataclass
class SparcOutput:
    """完整前向的输出。"""

    eta: Tensor                 # (B,) 路由后的自然参数
    eta_base: Tensor            # (B,) η_base
    delta: Tensor               # (B,) 残差 Δ
    gate: Tensor                # (B,) 连续 g
    gate_effective: Tensor      # (B,) g̃（训练期 = g；S4 之后 = g·1[g≥λ]）
    log_var: Tensor             # (B,) 修正后的 logσ²
    base_log_var: Tensor        # (B,) Base 的 logσ²
    attention: Optional[Tensor] = None       # (B, K) 证据注意力
    rank_scores: Optional[Tensor] = None     # (B, K0) 候选池分数
    match_confidence: Optional[Tensor] = None  # (B, K) m_i

    @property
    def prediction(self) -> Tensor:
        """回归任务的预测值 ``ŷ = ψ_t(η) = η``。"""
        return self.eta


class SparcNP(nn.Module):
    """SPARC-NP 完整模型。"""

    def __init__(
        self,
        base: BasePredictor,
        graph_matcher: GraphMatcherLite,
        reranker: RerankerLite,
        evidence: EvidenceLite,
        residual: ResidualHead,
        uncertainty: UncertaintyHead,
        gate: SupportGate,
        detach_base_hidden: bool = False,
    ) -> None:
        """
        Args:
            base: Θ_B（S2 起冻结）。
            graph_matcher: GraphMatcher-lite。
            reranker: Reranker-lite。
            evidence: Evidence-lite。
            residual: 残差头。
            uncertainty: 不确定性头。
            gate: 29 参数门控。
            detach_base_hidden: 是否切断到 ``h_q`` 的梯度。
                §8.3.4 规定 ``h_q`` 不再冻结，故默认 ``False``；
                Θ_B 的**参数**冻结由 :meth:`BasePredictor.freeze` 负责。
        """
        super().__init__()
        self.base = base
        self.graph_matcher = graph_matcher
        self.reranker = reranker
        self.evidence = evidence
        self.residual = residual
        self.uncertainty = uncertainty
        self.gate = gate
        self.detach_base_hidden = detach_base_hidden

    # ------------------------------------------------------------------
    def base_predict(self, ligand_pca: Tensor, protein_pca: Tensor) -> BaseOutput:
        """只跑 Base 预测器 —— ``d_i`` 的锚点。"""
        return self.base(ligand_pca, protein_pca)

    def forward(
        self,
        ligand_pca: Tensor,
        protein_pca: Tensor,
        retrieval_batch: Optional[Dict[str, Tensor]] = None,
        gate_features: Optional[Tensor] = None,
        task_type: str = TASK_TYPE_B,
        g_override: Optional[float] = None,
        apply_threshold: bool = False,
        lam: Optional[float] = None,
    ) -> SparcOutput:
        """完整前向。

        Args:
            ligand_pca: ``(B, 128)``。
            protein_pca: ``(B, 64)``。
            retrieval_batch: 检索管线产出的张量包；``None`` 或 C 类任务时
                走 ``INV-C`` 硬隔离（``g̃ ≡ 0``，不访问记忆库）。
            gate_features: ``(B, 28)`` 支持度特征。
            task_type: ``"B"`` 或 ``"C"``；``1[ρ(t)=B]`` 的实现。
            g_override: 强制指定 ``g̃``（``0.0`` 用于 §8.3.4 的退化断言）。
            apply_threshold: 是否施加硬阈值。**训练期必须 ``False``**（§8.3.5）。
            lam: LTT 标定出的 ``λ``。

        Returns:
            :class:`SparcOutput`。
        """
        base_out = self.base(ligand_pca, protein_pca)
        eta_base = base_out.mu
        batch_size = eta_base.shape[0]
        zeros = torch.zeros_like(eta_base)

        route_b = (task_type == TASK_TYPE_B)
        if not route_b or retrieval_batch is None:
            # INV-C 硬隔离：不访问记忆库，g̃ ≡ 0，精确退化回 Base
            return SparcOutput(
                eta=eta_base, eta_base=eta_base, delta=zeros,
                gate=zeros, gate_effective=zeros,
                log_var=base_out.log_var, base_log_var=base_out.log_var,
            )

        query_repr = base_out.query_repr()
        if self.detach_base_hidden:
            query_repr = query_repr.detach()

        # --- 图匹配 → 对齐摘要 a_i 与匹配置信度 m_i ---
        align, match_conf = self.graph_matcher(
            retrieval_batch["query_node_repr"],
            retrieval_batch["cand_node_repr"],
            retrieval_batch.get("query_atom_mask"),
            retrieval_batch.get("cand_atom_mask"),
        )

        # --- 重排 ---
        scalar_features = retrieval_batch["rank_scalar_features"].clone()
        scalar_features[..., 3] = match_conf          # 第 4 个标量是 m_i
        rerank = self.reranker(
            query_repr, retrieval_batch["cand_repr"], align, scalar_features,
            retrieval_batch.get("candidate_mask"), k_top=retrieval_batch.get("k_top", 16),
        )
        topk = rerank.topk_index

        # --- 证据聚合（只在 Top-K 上） ---
        def _gather(tensor: Tensor) -> Tensor:
            """按 Top-K 下标收集 ``(B, K0, ...)`` → ``(B, K, ...)``。"""
            index = topk
            while index.dim() < tensor.dim():
                index = index.unsqueeze(-1)
            return torch.gather(tensor, 1, index.expand(-1, -1, *tensor.shape[2:]))

        evidence_out = self.evidence(
            _gather(rerank.ligand_proj_cand),
            _gather(align),
            _gather(retrieval_batch["label_features"]),
            torch.gather(retrieval_batch["assay_family"], 1, topk),
            _gather(retrieval_batch["candidate_meta"]),
            rerank.topk_scores,
            torch.gather(retrieval_batch["candidate_mask"], 1, topk) if "candidate_mask" in retrieval_batch else None,
        )

        # --- 残差与门控 ---
        delta = self.residual(query_repr, evidence_out.context)
        delta_log_var, _ = self.uncertainty(evidence_out.context, evidence_out.pooled_raw)

        if g_override is not None:
            g = torch.full_like(eta_base, float(g_override))
            g_tilde = g
        else:
            if gate_features is None:
                raise ValueError("B 类任务需要 gate_features（28 维支持度特征）")
            g = self.gate(gate_features, apply_threshold=False)
            g_tilde = self.gate(gate_features, apply_threshold=True, lam=lam) if apply_threshold else g

        eta = eta_base + g_tilde * delta
        # 不确定性修正同样按 g̃ 缩放 ⇒ g̃=0 时 logσ² 也精确退化回 Base
        log_var = base_out.log_var + g_tilde * delta_log_var

        return SparcOutput(
            eta=eta, eta_base=eta_base, delta=delta, gate=g, gate_effective=g_tilde,
            log_var=log_var, base_log_var=base_out.log_var,
            attention=evidence_out.attention, rank_scores=rerank.scores,
            match_confidence=match_conf,
        )

    # ------------------------------------------------------------------
    def assert_fallback_identity(
        self,
        ligand_pca: Tensor,
        protein_pca: Tensor,
        retrieval_batch: Dict[str, Tensor],
        gate_features: Tensor,
        atol: float = 1e-6,
    ) -> None:
        """§8.3.4 的退化等价性断言。

        ``predict(q, g=0) == base_predict(q)`` 必须精确成立。

        **强制 fp32**：fallback 路径不允许混合精度容差。这不是保守，
        而是因为 ``g̃=0`` 的退化是"拒检"这个概念的定义本身 ——
        它在 bf16 下"差不多相等"，那么被拒检的样本就不是真的走了 Base。

        Args:
            ligand_pca: ``(B, 128)``。
            protein_pca: ``(B, 64)``。
            retrieval_batch: 检索张量包。
            gate_features: ``(B, 28)``。
            atol: 绝对容差（1e-6）。

        Raises:
            AssertionError: 退化不精确。
            RuntimeError: 输入不是 fp32。
        """
        for name, tensor in (("ligand_pca", ligand_pca), ("protein_pca", protein_pca)):
            if tensor.dtype != torch.float32:
                raise RuntimeError(
                    f"退化断言要求 fp32，但 {name} 是 {tensor.dtype}。"
                    "§8.3.4：fallback 路径不允许混合精度容差。"
                )

        was_training = self.training
        self.eval()
        with torch.no_grad():
            base_out = self.base_predict(ligand_pca, protein_pca)
            routed = self.forward(
                ligand_pca, protein_pca, retrieval_batch, gate_features,
                task_type=TASK_TYPE_B, g_override=0.0,
            )
            isolated = self.forward(ligand_pca, protein_pca, None, None, task_type=TASK_TYPE_C)
        if was_training:
            self.train()

        max_diff = (routed.eta - base_out.mu).abs().max().item()
        if not torch.allclose(routed.eta, base_out.mu, atol=atol, rtol=0.0):
            raise AssertionError(
                f"g̃=0 时未精确退化回 Base：最大差 {max_diff:.3e} > atol {atol:.1e}（§8.3.4）"
            )
        var_diff = (routed.log_var - base_out.log_var).abs().max().item()
        if not torch.allclose(routed.log_var, base_out.log_var, atol=atol, rtol=0.0):
            raise AssertionError(f"g̃=0 时 logσ² 未精确退化：最大差 {var_diff:.3e}（§8.3.4）")
        if not torch.allclose(isolated.eta, base_out.mu, atol=atol, rtol=0.0):
            raise AssertionError("INV-C 硬隔离路径未精确退化回 Base（§4.1）")

        _LOGGER.info("退化等价性断言通过（fp32，max_diff=%.3e / %.3e）", max_diff, var_diff)

    # ------------------------------------------------------------------
    def parameter_budget(self) -> Dict[str, int]:
        """按 §8.3.6 的口径统计参数量。

        Returns:
            各模块参数量与 ``theta_b`` / ``theta_r`` / ``total_trainable`` 合计。
        """
        theta_b = self.base.n_parameters()
        modules = {
            "graphmatcher": self.graph_matcher.n_parameters(),
            "reranker": self.reranker.n_parameters(),
            "evidence": self.evidence.n_parameters(),
            "residual": self.residual.n_parameters(),
            "gate": self.gate.n_parameters(),
            "uncertainty_head": self.uncertainty.n_parameters(),
        }
        theta_r = sum(modules.values())
        return {"theta_b": theta_b, **modules, "theta_r": theta_r, "total_trainable": theta_b + theta_r}

    def freeze_base(self) -> "SparcNP":
        """S1-F：冻结 Θ_B。"""
        self.base.freeze()
        return self

    def freeze_retrieval(self) -> "SparcNP":
        """S3 之前：冻结 Θ_R \\ gate（门控单独交叉拟合训练）。"""
        for module in (self.graph_matcher, self.reranker, self.evidence, self.residual, self.uncertainty):
            for param in module.parameters():
                param.requires_grad_(False)
            module.eval()
        return self

    def trainable_parameters(self, stage: str) -> list:
        """按阶段返回应当训练的参数 (§10.4)。

        Args:
            stage: ``"s1"``（Θ_B）/ ``"s2"``（Θ_R \\ gate）/ ``"s3"``（gate）。

        Returns:
            参数列表。

        Raises:
            ValueError: 未知阶段。
        """
        if stage == "s1":
            return list(self.base.parameters())
        if stage == "s2":
            params = []
            for module in (self.graph_matcher, self.reranker, self.evidence, self.residual, self.uncertainty):
                params.extend(module.parameters())
            return params
        if stage == "s3":
            return list(self.gate.parameters())
        raise ValueError(f"未知阶段 '{stage}'，可选 s1 / s2 / s3")
