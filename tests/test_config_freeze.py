"""冻结配置与规范一致性 (§7.2, §10.2, §11.1, §12.2)。

§10.2 原文："未冻结的搜索网格是数据泄漏最常见的入口。"
本文件检查那些一旦被"顺手改一下"就会静默破坏结论的配置项。
"""

from __future__ import annotations

import copy

import pytest

from sparc.common.config import ModelAEligibilityError, resolve_model_a_entry
from sparc.data.units import to_pactivity, UnitConversionReport, aggregate_pactivity


class TestFreezeManifest:
    """冻结配置必须可 hash。"""

    def test_all_configs_hashable(self, config):
        """五份配置文件都必须存在且能算出 sha256。"""
        manifest = config.freeze_manifest()
        assert set(manifest) == {"preregistration", "frozen_hparams", "gate_feature_manifest",
                                 "paths", "model_registry"}
        assert all(v != "MISSING" and len(v) == 64 for v in manifest.values())

    def test_run_manifest_is_self_hashed(self, config):
        """run manifest 自带 hash，便于审计。"""
        manifest = config.run_manifest(device_info={"kind": "cpu"})
        assert len(manifest["manifest_sha256"]) == 64
        assert manifest["seed"] == config.hparams.train.seed


class TestPreregistration:
    """§1.3 与 §11.2 的预注册参数。"""

    def test_ltt_parameters(self, config):
        """α = 0.10、δ = 0.05、ε = 0.01（§11.2 冻结值）。"""
        ltt = config.prereg.ltt
        assert ltt.alpha == 0.10 and ltt.delta == 0.05 and ltt.epsilon == 0.01

    def test_lambda_grid_is_full(self, config):
        """Λ = {0.00, 0.01, ..., 0.99}，100 个候选。"""
        grid = config.prereg.ltt.lambda_grid()
        assert len(grid) == 100 and grid[0] == 0.0 and grid[-1] == 0.99

    def test_h1_criteria(self, config):
        """H1 的三条判据阈值。"""
        h1 = config.prereg.h1
        assert h1["low_support_tanimoto_threshold"] == 0.35
        assert h1["criteria"]["b_min_target_agreement_frac"] == 0.60
        assert h1["criteria"]["c_min_cohens_d"] == 0.20

    def test_h2_criteria(self, config):
        """H2 的四条判据阈值（含 v1.0 加严的泛化间隙与符号一致性）。"""
        criteria = config.prereg.h2["criteria"]
        assert criteria["b_min_auroc_on_activity_cliff_subset"] == 0.65
        assert criteria["c_max_generalization_gap"] == 0.10
        assert criteria["d_max_sign_flips"] == 2

    def test_h3_criteria(self, config):
        """H3：SafeCoverage ≥ 0.30，且 ≥60% 靶点为正。"""
        criteria = config.prereg.h3["criteria"]
        assert criteria["a_min_safe_coverage"] == 0.30
        assert criteria["b_min_target_positive_frac"] == 0.60

    def test_mean_baseline_is_mandatory(self, config):
        """§12.2 与事实 A：均值预测器必须是强制基线。"""
        assert "mean_predictor" in config.prereg.mandatory_baselines
        assert "ecfp4_rf" in config.prereg.mandatory_baselines

    def test_stage0_gate(self, config):
        """§13.1 Stage 0 闸门：Top-1 Tanimoto 中位数上限 0.6，|M_t| 下限 200。"""
        gate = config.prereg.stage0_gate
        assert gate["max_median_top1_tanimoto"] == 0.60
        assert gate["min_memory_per_target"] == 200


class TestFrozenHparams:
    """§8–§10 的冻结超参。"""

    def test_loss_weight_defaults_and_grids(self, config):
        """§10.2 从 structure5.6.md 恢复的默认值与网格。"""
        loss = config.hparams.loss
        assert (loss.lambda_rank, loss.lambda_utility, loss.lambda_cal) == (0.20, 0.50, 0.10)
        assert loss.grid_rank == [0.05, 0.10, 0.20, 0.40]
        assert loss.grid_utility == [0.25, 0.50, 1.00]
        assert loss.grid_cal == [0.05, 0.10, 0.20]
        assert loss.harm_weight == 0.5

    def test_inner_cv_grid_size(self, config):
        """内层 CV 网格 = 4 × 3 × 3 = 36 组。"""
        assert len(config.hparams.loss.inner_cv_grid()) == 36

    def test_sinkhorn_tau_m_present(self, config):
        """§8.3.1：``τ_m = 0.10`` 是 v1.0 补齐的冻结超参（talk.md 中缺失）。"""
        assert config.hparams.sinkhorn.tau_m == 0.10
        assert config.hparams.sinkhorn.with_dustbin is True

    def test_retrieval_constants(self, config):
        """§9.1：K_src=512、K₀=256、K=16、|M_view| 下限 200。"""
        retrieval = config.hparams.retrieval
        assert (retrieval.k_src, retrieval.k0, retrieval.k_top) == (512, 256, 16)
        assert retrieval.min_memory_view == 200

    def test_view_filter_includes_tax_id(self, config):
        """事实 F / R8：跨物种靶点必须按 tax_id 硬过滤。"""
        assert "organism_tax_id" in config.hparams.retrieval.view_filter_keys

    def test_split_fractions(self, config):
        """§11.2：外层 test 20%，dev 内 60:20:20。"""
        split = config.hparams.split
        assert split.outer_test_frac == 0.20
        assert (split.dev_train_frac, split.dev_inner_frac, split.dev_calib_frac) == (0.60, 0.20, 0.20)

    def test_scaffold_family_all_four_keys_enabled(self, config):
        """§6.1 的四把钥匙默认全开 —— 关掉任一把都会打开一条泄漏通道。"""
        split = config.hparams.split
        assert split.murcko_tanimoto_threshold == 0.50
        assert split.use_inchikey_skeleton and split.use_deglyco_core and split.use_tautomer_family

    def test_lambda_d_default_zero(self, config):
        """§6.4：默认 ``λ_D = 0``，药物数据不进入 Θ_B 的梯度。"""
        assert config.hparams.data.lambda_d == 0.0

    def test_crossfit_j_and_ks_threshold(self, config):
        """§10.3：J = 5，KS > 0.15 时需缩小 J。"""
        crossfit = config.hparams.train.crossfit
        assert crossfit["n_folds"] == 5 and crossfit["max_ks_shift"] == 0.15


class TestModelRegistry:
    """§7.2 的资格判据与硬阻断。"""

    @pytest.mark.parametrize("name", ["nafm", "graphmvp", "unimol", "unimol2", "gem", "3d_infomax"])
    def test_blocked_models_cannot_be_resolved(self, config, name):
        """判据 8/9 的硬阻断不可被配置绕过。"""
        with pytest.raises(ValueError, match="硬阻断"):
            resolve_model_a_entry(config.model_registry, name)

    def test_nafm_block_reason_mentions_domain_gate(self, config):
        """NaFM 的阻断理由必须点明它会抹掉跨域前提。"""
        with pytest.raises(ValueError) as exc:
            resolve_model_a_entry(config.model_registry, "nafm")
        assert "COCONUT" in str(exc.value) and "跨域" in str(exc.value)

    def test_candidates_have_pinned_revision_field(self, config):
        """判据 3：每个候选都必须有 revision 字段（值可以是待钉死的占位符）。"""
        for entry in config.model_registry["candidates"]:
            assert "revision" in entry, f"{entry['name']} 缺 revision（§7.2 判据 3）"

    def test_r_np_threshold(self, config):
        """判据 9：``r_NP ≤ 0.05``。"""
        assert config.model_registry["eligibility_thresholds"]["c9_max_r_np"] == 0.05

    def test_mean_predictor_is_mandatory_reference(self, config):
        """均值预测器在注册表里也标记为 mandatory。"""
        references = {r["name"]: r for r in config.model_registry["references"]}
        assert references["mean_predictor"]["mandatory"] is True
        assert references["ecfp4_rf"]["mandatory"] is True

    def test_nafm_reference_role_is_upper_bound_only(self, config):
        """NaFM 只能作天然产物侧的上界参照，绝不作 Model A。"""
        references = {r["name"]: r for r in config.model_registry["references"]}
        assert references["nafm"]["role"] == "np_side_upper_reference"


class TestP0EligibilityGate:
    """§7.2 的 P0 判据闸门 —— "unverified 不等于 pass" 这句话的执行者。

    名单阻断挡住的是**已经想到的**那几个（NaFM / GraphMVP / Uni-Mol / GEM）；
    这道闸门挡住的是**还没想到的**。在它存在之前，四个候选的 ``c9``
    全是 ``unverified``，而 ``resolve_model_a_entry`` 一路畅通。
    """

    @staticmethod
    def _with_eligibility(config, name: str, **overrides):
        """复制注册表并改写某个候选的 eligibility。"""
        registry = copy.deepcopy(config.model_registry)
        for entry in registry["candidates"]:
            if entry["name"] == name:
                entry["eligibility"].update(overrides)
        return registry

    def test_unverified_c9_is_blocked(self, config):
        """判据 9 规定 unknown ⇒ blocked，不得按"大概是药物库"放行。"""
        with pytest.raises(ModelAEligibilityError, match="c9"):
            resolve_model_a_entry(config.model_registry, "molformer_xl")

    def test_all_shipped_candidates_currently_fail_the_gate(self, config):
        """当前注册表里没有任何候选通过 P0 —— c9/c10 都还没实测。

        这条测试是**状态快照**，不是永久约束：把某个候选的 c9/c10 实测
        回填成 pass 之后，它会失败，届时按实际情况放宽即可。它存在的目的
        是防止有人在没有回填的情况下把 eligibility 悄悄改成 pass。
        """
        for entry in config.model_registry["candidates"]:
            with pytest.raises(ModelAEligibilityError):
                resolve_model_a_entry(config.model_registry, entry["name"])

    @pytest.mark.parametrize("code", ["c8", "c9", "c10"])
    def test_each_p0_criterion_blocks_independently(self, config, code):
        """三条 P0 判据各自都能单独拦下。"""
        statuses = {"c8": "pass", "c9": "pass", "c10": "pass"}
        statuses[code] = "fail"
        registry = self._with_eligibility(config, "molformer_xl", **statuses)
        with pytest.raises(ModelAEligibilityError, match=code):
            resolve_model_a_entry(registry, "molformer_xl")

    def test_missing_criterion_is_treated_as_unverified(self, config):
        """判据字段缺失 ⇒ 与 unverified 同等对待，不是"没写就当过"。"""
        registry = copy.deepcopy(config.model_registry)
        for entry in registry["candidates"]:
            if entry["name"] == "molformer_xl":
                entry["eligibility"] = {"c8": "pass", "c9": "pass"}      # 缺 c10
        with pytest.raises(ModelAEligibilityError, match="c10"):
            resolve_model_a_entry(registry, "molformer_xl")

    def test_stage0b_may_waive_c10_only(self, config):
        """Stage 0-B 可以放行 c10（它本来就要由 S0-B 产出），但 c9 仍然拦。"""
        registry = self._with_eligibility(config, "molformer_xl", c8="pass", c9="pass")
        entry = resolve_model_a_entry(registry, "molformer_xl", allow_unverified=["c10"])
        assert entry["name"] == "molformer_xl"

        still_blocked = self._with_eligibility(config, "molformer_xl", c8="pass")
        with pytest.raises(ModelAEligibilityError, match="c9"):
            resolve_model_a_entry(still_blocked, "molformer_xl", allow_unverified=["c10"])

    def test_all_pass_resolves(self, config):
        """三条 P0 全部 pass 后正常返回。"""
        registry = self._with_eligibility(config, "kpgt", c8="pass", c9="pass", c10="pass")
        assert resolve_model_a_entry(registry, "kpgt")["name"] == "kpgt"

    def test_waiver_cannot_unblock_the_blocked_list(self, config):
        """放行清单**不能**绕过名单阻断 —— NaFM 在任何配置下都进不来。"""
        with pytest.raises(ValueError, match="硬阻断"):
            resolve_model_a_entry(
                config.model_registry, "nafm", allow_unverified=["c8", "c9", "c10"],
            )

    def test_error_message_names_the_legal_way_forward(self, config):
        """报错必须告诉人合法的推进方式，否则只会被 enforce_eligibility=False 绕过。"""
        with pytest.raises(ModelAEligibilityError) as exc:
            resolve_model_a_entry(config.model_registry, "molclr_gin")
        message = str(exc.value)
        assert "allow_unverified" in message and "r_NP" in message


class TestUnitConversion:
    """§5.2 步骤 3 的单位统一。"""

    def test_nm_to_pactivity(self):
        """1 nM IC50 → pIC50 = 9。"""
        assert abs(to_pactivity(1.0, "nM") - 9.0) < 1e-9

    def test_um_to_pactivity(self):
        """1 µM → pIC50 = 6。"""
        assert abs(to_pactivity(1.0, "uM") - 6.0) < 1e-9

    def test_mass_concentration_needs_mw(self):
        """``ug.mL-1`` 缺 MW 必须丢弃并计数，不能沉默。"""
        report = UnitConversionReport()
        assert to_pactivity(10.0, "ug.mL-1", molecular_weight=None, report=report) is None
        assert report.n_dropped_no_mw == 1
        value = to_pactivity(10.0, "ug.mL-1", molecular_weight=500.0, report=report)
        assert abs(value - 4.69897) < 1e-4       # 10 mg/L / 500 g/mol = 2e-5 M

    def test_ambiguous_mm_is_dropped_by_default(self):
        """``mm`` 在 NPASS 中语义歧义（毫摩尔 vs 毫米），默认丢弃并计数。"""
        report = UnitConversionReport()
        assert to_pactivity(1.0, "mm", report=report) is None
        assert report.n_dropped_ambiguous_unit == 1

    def test_non_convertible_units_counted(self):
        """``%``、``cells.uL-1`` 等不可换算单位必须计数。"""
        report = UnitConversionReport()
        for unit in ("%", "cells.uL-1", "IU.L-1"):
            assert to_pactivity(50.0, unit, report=report) is None
        assert report.n_dropped_bad_unit == 3

    def test_high_spread_records_dropped(self):
        """§5.2 步骤 4：极差 > 1.5 log 单位的整条丢弃。"""
        assert aggregate_pactivity([5.0, 7.0], max_spread=1.5) == (None, "high_spread")
        value, status = aggregate_pactivity([6.0, 6.5, 7.0], max_spread=1.5)
        assert status == "ok" and abs(value - 6.5) < 1e-9

    def test_censor_flag_mapping(self, config):
        """§5.2 步骤 5：关系符必须映射到 left/none/right，不转点值。"""
        data = config.hparams.data
        assert data.censor_flag(">") == "right"
        assert data.censor_flag(">=") == "right"
        assert data.censor_flag("<") == "left"
        assert data.censor_flag("=") == "none"
        assert data.censor_flag("n.a.") == "none"
