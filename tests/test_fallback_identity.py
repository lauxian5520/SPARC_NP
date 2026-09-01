"""§8.3.4 的退化等价性：``predict(q, g=0) == base_predict(q)``，fp32 精确。

**为什么强制 fp32**：``g̃=0`` 的退化不是一个数值细节，它是"拒检"这个
概念的定义本身。如果它在 bf16 下只是"差不多相等"，那么被拒检的样本
就不是真的走了 Base，H3 的保证也就不再指向它声称的对象。

需要 torch，因此这些测试只在服务器上跑。树莓派上会 skip —— 但
:class:`TestFallbackContract` 里的契约测试不需要 torch，始终执行。
"""

from __future__ import annotations

import pytest

from conftest import requires_torch


class TestFallbackContract:
    """不依赖 torch 的契约检查。"""

    def test_config_forbids_mixed_precision_in_fallback(self, config):
        """``frozen_hparams.yaml`` 必须要求 fallback 断言走 fp32。"""
        amp = config.hparams.train.amp
        assert amp["fallback_assert_fp32"] is True, (
            "§8.3.4：fallback 路径不允许混合精度容差。"
            "这一项被关掉时，退化断言会在 bf16 下'通过'，而拒检不再精确。"
        )

    def test_detach_base_hidden_default_false(self, config):
        """§8.3.4：``h_q`` 不再冻结（梯度可穿过）。"""
        assert config.hparams.train.retrieval["detach_base_hidden"] is False


@requires_torch
class TestFallbackIdentity:
    """需要 torch 的数值断言。"""

    @staticmethod
    def _build_model():
        """构造一个完整 SparcNP。"""
        from sparc.models.base import BasePredictor
        from sparc.models.evidence import EvidenceLite
        from sparc.models.gate import SupportGate
        from sparc.models.graphmatcher import GraphMatcherLite
        from sparc.models.reranker import RerankerLite
        from sparc.models.residual import ResidualHead, UncertaintyHead
        from sparc.models.sparc_model import SparcNP

        return SparcNP(
            base=BasePredictor(), graph_matcher=GraphMatcherLite(), reranker=RerankerLite(),
            evidence=EvidenceLite(), residual=ResidualHead(), uncertainty=UncertaintyHead(),
            gate=SupportGate(),
        )

    @staticmethod
    def _fake_batch(batch_size: int = 4, k0: int = 8, n_atoms: int = 6, d_hidden: int = 64):
        """构造一个形状正确的假检索批。"""
        import torch

        return {
            "query_node_repr": torch.randn(batch_size, n_atoms, d_hidden),
            "cand_node_repr": torch.randn(batch_size, k0, n_atoms, d_hidden),
            "query_atom_mask": torch.ones(batch_size, n_atoms, dtype=torch.bool),
            "cand_atom_mask": torch.ones(batch_size, k0, n_atoms, dtype=torch.bool),
            "cand_repr": torch.randn(batch_size, k0, 256),
            "rank_scalar_features": torch.rand(batch_size, k0, 7),
            "label_features": torch.randn(batch_size, k0, 5),
            "assay_family": torch.randint(0, 32, (batch_size, k0)),
            "candidate_meta": torch.rand(batch_size, k0, 8),
            "candidate_mask": torch.ones(batch_size, k0, dtype=torch.bool),
            "k_top": 4,
        }

    def test_g_zero_recovers_base_exactly(self):
        """``g̃=0`` 时 η 与 logσ² 都必须精确等于 Base。"""
        import torch

        torch.manual_seed(0)
        model = self._build_model().eval()
        ligand = torch.randn(4, 128)
        protein = torch.randn(4, 64)
        batch = self._fake_batch()
        gate_features = torch.randn(4, 28)

        # 先把残差训成非零，否则断言会因为"Δ 恰好为 0"而平凡通过
        with torch.no_grad():
            model.residual.w_delta.fill_(0.7)
            model.residual.w_context.fill_(0.3)
            model.residual.bias.fill_(0.5)
            model.uncertainty.head.weight.fill_(0.01)
            model.uncertainty.head.bias.fill_(0.02)

        with torch.no_grad():
            base = model.base_predict(ligand, protein)
            routed = model(ligand, protein, batch, gate_features, g_override=0.0)
            active = model(ligand, protein, batch, gate_features, g_override=1.0)

        assert not torch.allclose(active.eta, base.mu), "残差恒为 0，断言会平凡通过 —— 测试无效"
        assert torch.allclose(routed.eta, base.mu, atol=1e-6, rtol=0.0)
        assert torch.allclose(routed.log_var, base.log_var, atol=1e-6, rtol=0.0)

    def test_assert_fallback_identity_helper(self):
        """模型自带的断言方法必须通过。"""
        import torch

        torch.manual_seed(1)
        model = self._build_model().eval()
        model.assert_fallback_identity(
            torch.randn(4, 128), torch.randn(4, 64), self._fake_batch(), torch.randn(4, 28)
        )

    def test_assert_rejects_non_fp32(self):
        """非 fp32 输入必须被拒绝，而不是"放宽容差通过"。"""
        import torch

        model = self._build_model().eval()
        with pytest.raises(RuntimeError, match="fp32"):
            model.assert_fallback_identity(
                torch.randn(4, 128, dtype=torch.float64), torch.randn(4, 64),
                self._fake_batch(), torch.randn(4, 28),
            )

    def test_task_type_c_isolates_memory(self):
        """C 类任务走 INV-C 硬隔离：``g̃ ≡ 0``，不访问记忆库。"""
        import torch

        torch.manual_seed(2)
        model = self._build_model().eval()
        ligand, protein = torch.randn(4, 128), torch.randn(4, 64)
        with torch.no_grad():
            base = model.base_predict(ligand, protein)
            isolated = model(ligand, protein, self._fake_batch(), torch.randn(4, 28), task_type="C")
        assert torch.allclose(isolated.eta, base.mu, atol=1e-6, rtol=0.0)
        assert float(isolated.gate_effective.abs().max()) == 0.0

    def test_gate_receives_gradient_with_continuous_g(self):
        """§8.3.5：连续 ``g`` 下门控参数必须拿到梯度。

        断言对象是 **``w_g`` 自己的梯度**，不是对输入 ``x_g`` 的梯度。
        后者在初始化时恒为 0 —— ``SupportGate.__init__`` 用
        ``nn.init.zeros_(weight)``（起点 σ(b_g)=σ(2.0)≈0.88，倾向接受检索），
        于是 ``∂g/∂x = σ'·w = σ'·0 = 0``。那是设计如此，且无害：
        ``x_g`` 由 :mod:`sparc.retrieval.features` 在 numpy 里算出，是叶子输入，
        它上游没有可训练的东西。真正关乎 §8.3.5 的是 ``∂L/∂w_g``。
        """
        import torch

        from sparc.models.gate import SupportGate

        gate = SupportGate()
        gate(torch.randn(16, 28), apply_threshold=False).sum().backward()
        assert gate.linear.weight.grad is not None
        assert float(gate.linear.weight.grad.abs().sum()) > 0, "连续 g 下 w_g 必须有梯度"
        assert float(gate.linear.bias.grad.abs().sum()) > 0

    def test_hard_threshold_zeroes_gradient_of_rejected_samples(self):
        """§8.3.5 的实质：硬阈值让被拒样本的梯度归零，被拒集成为吸收态。

        训练期若按字面实现 ``g̃ = g·1[g≥λ]``，``g < λ`` 的样本对 ``w_g``
        贡献恒为 0 —— 门控再也学不会把错拒的样本捞回来。这就是
        ``apply_threshold`` 默认 ``False``、硬阈值只在 Stage 4 之后生效的原因。
        """
        import torch

        from sparc.models.gate import SupportGate

        def weight_grad(apply_threshold, lam=None):
            torch.manual_seed(0)
            gate = SupportGate()
            torch.nn.init.normal_(gate.linear.weight, std=0.5)   # 模拟训练中期
            torch.manual_seed(1)
            out = gate(torch.randn(64, 28), apply_threshold=apply_threshold, lam=lam)
            out.sum().backward()
            return float(gate.linear.weight.grad.abs().sum()), int((out > 0).sum())

        continuous, n_all = weight_grad(False)
        thresholded, n_accepted = weight_grad(True, lam=0.90)

        assert n_accepted < n_all, "夹具无效：λ=0.90 下应当有样本被拒"
        assert thresholded < continuous, (
            f"硬阈值下的梯度({thresholded:.4f})必须小于连续 g({continuous:.4f})——"
            f"只有 {n_accepted}/{n_all} 个被接受样本还在贡献梯度"
        )

    def test_gate_requires_lambda_when_thresholding(self):
        """施加硬阈值却不给 λ 必须报错 —— λ 只能来自 LTT。"""
        import torch

        from sparc.models.gate import SupportGate

        with pytest.raises(ValueError, match="Learn-then-Test"):
            SupportGate()(torch.randn(4, 28), apply_threshold=True)

    def test_match_confidence_in_unit_interval(self):
        """§8.3.1：dustbin 的不等式边际保证 ``m_i ∈ [0,1]``。"""
        import torch

        from sparc.models.graphmatcher import GraphMatcherLite

        torch.manual_seed(3)
        matcher = GraphMatcherLite()
        h_q = torch.randn(3, 10, 64)
        h_c = torch.randn(3, 5, 12, 64)
        mask_q = torch.ones(3, 10, dtype=torch.bool)
        mask_c = torch.ones(3, 5, 12, dtype=torch.bool)
        align, confidence = matcher(h_q, h_c, mask_q, mask_c)
        assert align.shape == (3, 5, 32)
        assert confidence.shape == (3, 5)
        assert float(confidence.min()) >= 0.0 and float(confidence.max()) <= 1.0

    def test_alignment_summary_memory_is_bounded(self):
        """§8.3.1 的因式分解改写：显存不随 N_q×N_i 增长。

        直接验证不现实，这里验证形状与可微性 —— 若实现退回原式，
        ``d_mat`` 的构造会变成 ``(B,K,N_q,N_i,d)``，此处会 OOM 或形状不符。
        """
        import torch

        from sparc.models.graphmatcher import GraphMatcherLite

        matcher = GraphMatcherLite()
        h_q = torch.randn(2, 40, 64, requires_grad=True)
        h_c = torch.randn(2, 32, 40, 64)
        align, _ = matcher(h_q, h_c)
        align.sum().backward()
        assert h_q.grad is not None
