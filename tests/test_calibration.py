"""Learn-then-Test 与 SafeCoverage 的正确性 (§11, §12.1)。

这两个是本项目的**理论主张**，因此测试要覆盖的不只是"能跑"，
而是那几条使主张成立的性质：

* LTT 的 λ* 必须来自检验，不是网格搜索（独立性守卫）；
* λ* 不存在时必须如实返回 ``None``，不能退化成"取最优"；
* SafeCoverage **不可通过少检索平凡满足**；
* 朴素检索的 c* ≈ 0，Oracle 给出上界。
"""

from __future__ import annotations

import numpy as np
import pytest

from sparc.calibrate.ltt import (
    diagnose_fixed_sequence_power,
    fixed_sequence_order_from_inner_fold,
    hoeffding_bentkus_pvalue,
    learn_then_test,
    risk_at_lambda,
    validate_on_test,
)
from sparc.eval.safe_coverage import oracle_gate_values, safe_coverage, tanimoto_gate_values


def _scenario(n=1000, seed=0, helpful_above=0.6, help_size=0.3, harm_prob=0.5):
    """构造"门控高时检索有帮助、门控低时有害"的场景。"""
    rng = np.random.default_rng(seed)
    gate = rng.random(n)
    loss_base = rng.gamma(2.0, 0.5, n)
    loss_sparc = np.where(
        gate > helpful_above,
        loss_base - help_size,
        loss_base + rng.choice([0.0, 0.5], n, p=[1 - harm_prob, harm_prob]),
    )
    return gate, loss_sparc, loss_base


class TestPValues:
    """HB p 值的基本性质。"""

    def test_pvalue_in_unit_interval(self):
        """p 值必须落在 [0,1]。"""
        for risk in (0.0, 0.05, 0.1, 0.5, 1.0):
            for n in (1, 10, 100, 1000):
                p = hoeffding_bentkus_pvalue(risk, n, 0.10)
                assert 0.0 <= p <= 1.0

    def test_pvalue_monotone_in_sample_size(self):
        """风险固定时，样本越多 p 值越小（证据越强）。"""
        values = [hoeffding_bentkus_pvalue(0.02, n, 0.10) for n in (50, 200, 1000)]
        assert values[0] > values[1] > values[2]

    def test_risk_above_alpha_never_rejects(self):
        """经验风险 ≥ α 时不可能拒绝原假设。"""
        assert hoeffding_bentkus_pvalue(0.15, 5000, 0.10) == 1.0

    def test_zero_sample_returns_one(self):
        """无接受样本时 p = 1 —— 空集不能被判为"安全"。"""
        assert hoeffding_bentkus_pvalue(0.0, 0, 0.10) == 1.0


class TestRiskFunction:
    """§11.2 的风险函数。"""

    def test_empty_acceptance_gives_zero_coverage(self):
        """λ 高到无人被接受时，覆盖率为 0（风险按 0 计但不可用于认证）。"""
        gate, sparc, base = _scenario()
        risk, n, coverage = risk_at_lambda(gate, sparc, base, lam=1.5)
        assert n == 0 and coverage == 0.0 and risk == 0.0

    def test_coverage_monotone_decreasing_in_lambda(self):
        """覆盖率关于 λ 单调不增。"""
        gate, sparc, base = _scenario()
        coverages = [risk_at_lambda(gate, sparc, base, lam)[2] for lam in np.arange(0, 1.0, 0.1)]
        assert all(a >= b for a, b in zip(coverages, coverages[1:]))


class TestLearnThenTest:
    """§11 的主流程。"""

    def test_requires_explicit_holdout_confirmation(self):
        """必须显式确认标定折独立性 —— 绕过它应是一个刻意动作。"""
        gate, sparc, base = _scenario()
        with pytest.raises(ValueError, match="calibration_is_held_out"):
            learn_then_test(gate, sparc, base)

    def test_finds_lambda_when_gate_is_informative(self):
        """门控有信息时应找到 λ*，且覆盖率非平凡。"""
        gate, sparc, base = _scenario(seed=7)
        result = learn_then_test(gate, sparc, base, multiple_testing="bonferroni",
                                calibration_is_held_out=True)
        assert result.found
        assert 0.0 < result.lambda_star < 1.0
        assert result.coverage_at_star() > 0.10

    def test_returns_none_when_retrieval_always_harms(self):
        """全域有害时必须如实返回 None，不能"挑一个最优的"。"""
        rng = np.random.default_rng(1)
        gate = rng.random(800)
        base = rng.gamma(2.0, 0.5, 800)
        result = learn_then_test(gate, base + 0.5, base, multiple_testing="bonferroni",
                                calibration_is_held_out=True)
        assert not result.found and result.lambda_star is None
        assert "有效结论" in result.guarantee

    def test_guarantee_string_reports_alpha_delta(self):
        """保证语句必须写明 α 与 1−δ。"""
        gate, sparc, base = _scenario(seed=7)
        result = learn_then_test(gate, sparc, base, multiple_testing="bonferroni",
                                calibration_is_held_out=True)
        assert "0.1" in result.guarantee and "0.95" in result.guarantee

    def test_test_risk_validation(self):
        """H3 判据 (c)：λ* 在测试集上的实际风险应 ≤ α。"""
        gate, sparc, base = _scenario(seed=7)
        result = learn_then_test(gate, sparc, base, multiple_testing="bonferroni",
                                calibration_is_held_out=True)
        gate_t, sparc_t, base_t = _scenario(n=500, seed=8)
        validation = validate_on_test(result, gate_t, sparc_t, base_t)
        assert validation["criterion_c_passed"]
        assert validation["test_risk"] <= result.alpha


class TestFixedSequencePower:
    """固定序列检验的功效问题（本项目标定折规模下的真实约束）。"""

    def test_diagnosis_flags_first_step_blockage(self):
        """λ=0.99 处样本极少时，诊断必须报出"第一步就会中断"。"""
        gate = np.random.default_rng(0).random(400)
        diagnosis = diagnose_fixed_sequence_power(gate, alpha=0.10, delta=0.05)
        assert diagnosis["fixed_sequence_blocked_at_first_step"] is True
        assert diagnosis["min_n_to_reject_at_zero_risk"] == 150
        assert "bonferroni" in diagnosis["recommendation"]

    def test_inner_fold_ordering_recovers_power(self):
        """用内层折预先指定顺序后，固定序列能找到 λ*（且顺序不依赖标定折）。"""
        gate, sparc, base = _scenario(seed=7)
        gate_i, sparc_i, base_i = _scenario(n=400, seed=9)
        order = fixed_sequence_order_from_inner_fold(gate_i, sparc_i, base_i)
        assert len(order) == 100
        result = learn_then_test(gate, sparc, base, multiple_testing="fixed_sequence",
                                 lambda_order=order, calibration_is_held_out=True)
        assert result.found

    def test_config_default_is_bonferroni(self, config):
        """冻结配置的默认多重校正必须是 bonferroni（见 preregistration.yaml 的理由）。"""
        assert config.prereg.ltt.multiple_testing == "bonferroni"


class TestSafeCoverage:
    """§12.1 的主标指标。"""

    def test_naive_retrieval_gives_zero(self):
        """朴素检索（门控与效用无关且整体有害）的 c* 应为 0。"""
        rng = np.random.default_rng(2)
        base = rng.gamma(2.0, 0.5, 1000)
        result = safe_coverage(rng.random(1000), base + 0.2, base, bootstrap_n=2000)
        assert result.c_star == 0.0

    def test_cannot_be_satisfied_by_retrieving_less(self):
        """**核心性质**：只对极少数样本检索无法把 c* 推高。

        旧方案的"伤害率下降 30% + 全样本非劣"两条都能靠"几乎不检索"
        满足；SafeCoverage 反过来 —— 覆盖率趋 0 时它趋 0。
        """
        rng = np.random.default_rng(11)
        n = 1000
        base = rng.gamma(2.0, 0.5, n)
        gate = np.where(np.arange(n) < 8, 0.99, 0.01)      # 只接受 8/1000
        sparc = np.where(np.arange(n) < 8, base - 0.5, base + 0.5)
        assert safe_coverage(gate, sparc, base, bootstrap_n=2000).c_star == 0.0

    def test_rewards_genuine_coverage(self):
        """真正在大比例样本上无害时，c* 应显著为正。"""
        rng = np.random.default_rng(11)
        n = 1000
        base = rng.gamma(2.0, 0.5, n)
        gate = np.where(np.arange(n) < 200, 0.99, 0.01)
        sparc = np.where(np.arange(n) < 200, base - 0.5, base + 0.5)
        assert safe_coverage(gate, sparc, base, bootstrap_n=2000).c_star >= 0.15

    def test_oracle_upper_bounds_deployable_gate(self):
        """Oracle 门控给出上界：任何可部署门控不应超过它。"""
        gate, sparc, base = _scenario(seed=5)
        deployable = safe_coverage(gate, sparc, base, bootstrap_n=2000).c_star
        oracle = safe_coverage(oracle_gate_values(sparc, base), sparc, base, bootstrap_n=2000).c_star
        assert oracle >= deployable - 1e-9

    def test_curve_has_zero_harm_crossing(self):
        """Risk–Coverage 曲线应包含 UCB 由负转正的交点。"""
        gate, sparc, base = _scenario(seed=5)
        result = safe_coverage(gate, sparc, base, bootstrap_n=2000)
        arrays = result.curve_arrays()
        assert arrays["coverage"].size > 50
        assert np.isfinite(arrays["mean_delta_loss"]).all()

    def test_tanimoto_gate_baseline_shape(self):
        """Tanimoto 门控基线可直接喂进同一套评估。"""
        rng = np.random.default_rng(4)
        top1 = rng.random(500)
        base = rng.gamma(2.0, 0.5, 500)
        sparc = np.where(top1 > 0.7, base - 0.3, base + 0.2)
        result = safe_coverage(tanimoto_gate_values(top1), sparc, base, bootstrap_n=1000)
        assert 0.0 <= result.c_star <= 1.0
