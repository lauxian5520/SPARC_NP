"""§5.4 的四条零重叠断言 + 划分协议的泄漏防护。

这些测试覆盖的是**规范正确性**，不依赖 torch/RDKit，因此在树莓派上
也必须全绿。它们保护的是本项目最容易被"顺手放宽"的地方：
事实 B 说查询池 79.0% 自带 ChEMBL ID，若断言退化成 warning，
整条管线看起来照常跑，但 H1/H2/H3 全部失效。
"""

from __future__ import annotations

import pytest

from conftest import requires_torch

from sparc.data.purge import LeakageAssertionError, NPPurger, assert_zero_overlap
from sparc.data.blacklist import NaturalProductBlacklist
from sparc.data.schema import CensorFlag, Domain, MemoryRecord, QueryRecord
from sparc.data.splits import SplitBuilder


def _query(qid: str, key: str, skel: str, deglyco: str, taut: str, family: str = "FAM0",
           target_id: str = "T1") -> QueryRecord:
    """构造一条查询记录。"""
    return QueryRecord(
        query_id=qid, compound_id=qid, inchikey=key, smiles="CCO",
        target_id=target_id, uniprot_id="P1", organism_tax_id="9606",
        pactivity=6.0, censor_flag=CensorFlag.NONE,
        skeleton14=skel, deglyco_core_hash=deglyco, tautomer_family_id=taut,
        scaffold_family_id=family, ortholog_group_id="OG1",
        source_organism_family="Rosaceae", reference_year=2020,
    )


def _memory(mid: str, key: str, skel: str, deglyco: str, taut: str, family: str = "FAM9",
            target_id: str = "T1") -> MemoryRecord:
    """构造一条记忆库记录。"""
    return MemoryRecord(
        memory_id=mid, inchikey=key, smiles="CCC",
        target_id=target_id, uniprot_id="P1", organism_tax_id="9606",
        pactivity=7.0, censor_flag=CensorFlag.NONE,
        skeleton14=skel, deglyco_core_hash=deglyco, tautomer_family_id=taut,
        scaffold_family_id=family, domain=Domain.DRUG,
    )


class TestZeroOverlapAssertions:
    """§5.4 的四条断言必须 fail hard。"""

    def test_clean_sets_pass(self):
        """完全不重叠时四条断言全部通过。"""
        queries = [_query("q1", "AAAAAAAAAAAAAA-X-N", "AAAAAAAAAAAAAA", "DG_A", "TF_A")]
        memory = [_memory("m1", "ZZZZZZZZZZZZZZ-X-N", "ZZZZZZZZZZZZZZ", "DG_Z", "TF_Z")]
        overlaps = assert_zero_overlap(queries, memory)
        assert overlaps == {"inchikey": 0, "skeleton14": 0, "deglyco_core_hash": 0, "tautomer_family_id": 0}

    @pytest.mark.parametrize(
        ("field", "shared"),
        [
            ("inchikey", "AAAAAAAAAAAAAA-X-N"),
            ("skeleton14", "AAAAAAAAAAAAAA"),
            ("deglyco_core_hash", "DG_A"),
            ("tautomer_family_id", "TF_A"),
        ],
    )
    def test_each_key_fails_hard(self, field, shared):
        """四把钥匙中任一重叠都必须抛 LeakageAssertionError，而不是告警。"""
        queries = [_query("q1", "AAAAAAAAAAAAAA-X-N", "AAAAAAAAAAAAAA", "DG_A", "TF_A")]
        values = {"inchikey": "ZZZZZZZZZZZZZZ-Y-N", "skeleton14": "ZZZZZZZZZZZZZZ",
                  "deglyco_core_hash": "DG_Z", "tautomer_family_id": "TF_Z"}
        values[field] = shared
        memory = [_memory("m1", values["inchikey"], values["skeleton14"],
                          values["deglyco_core_hash"], values["tautomer_family_id"])]
        with pytest.raises(LeakageAssertionError) as exc:
            assert_zero_overlap(queries, memory)
        assert field in str(exc.value)

    def test_glycoside_channel_is_caught_by_deglyco_key(self):
        """糖苷通道（事实 C）：InChIKey 与骨架块都不同，只有脱糖母核相同。

        这正是旧方案（Murcko + 精确 InChIKey）会漏掉的那 15.8%。
        """
        queries = [_query("rutin", "IKGXIBQEEMLURG-X-N", "IKGXIBQEEMLURG", "REFJWTPEDVJJIY", "TF_RUTIN")]
        memory = [_memory("quercetin", "REFJWTPEDVJJIY-Y-N", "REFJWTPEDVJJIY", "REFJWTPEDVJJIY", "TF_QUER")]
        with pytest.raises(LeakageAssertionError) as exc:
            assert_zero_overlap(queries, memory)
        # 骨架块也会命中（因为苷元的 skeleton14 == 母核 hash），但脱糖键必须在报错里出现
        assert "deglyco_core_hash" in str(exc.value)


class TestNPPurge:
    """§5.3 的 NP-purge。"""

    def _blacklist(self) -> NaturalProductBlacklist:
        """构造一个小黑名单。"""
        bl = NaturalProductBlacklist()
        bl.coconut_keys = {"AAAAAAAAAAAAAA-X-N"}
        bl.lotus_keys = {"BBBBBBBBBBBBBB-X-N"}
        bl.full_keys = bl.coconut_keys | bl.lotus_keys
        bl.skeleton_keys = {k[:14] for k in bl.full_keys}
        return bl

    def test_exact_and_skeleton_hits_counted_separately(self):
        """精确命中与骨架块命中必须分开计数（§5.3 要求分别报告）。"""
        purger = NPPurger(self._blacklist(), min_memory_per_target=1)
        candidates = [
            _memory("m1", "AAAAAAAAAAAAAA-X-N", "AAAAAAAAAAAAAA", "d1", "t1"),   # 精确
            _memory("m2", "BBBBBBBBBBBBBB-Y-N", "BBBBBBBBBBBBBB", "d2", "t2"),   # 骨架块
            _memory("m3", "CCCCCCCCCCCCCC-X-N", "CCCCCCCCCCCCCC", "d3", "t3"),   # 保留
        ]
        kept, report = purger.purge(candidates)
        assert len(kept) == 1 and kept[0].memory_id == "m3"
        assert report.n_dropped_exact == 1
        assert report.n_dropped_skeleton == 1
        assert report.np_purge_count == 2
        assert abs(report.np_purge_rate - 2 / 3) < 1e-9

    def test_min_memory_gate_flags_targets(self):
        """§5.3 硬闸门：|M_t| < 200 的靶点必须被标记，不允许静默继续。"""
        purger = NPPurger(self._blacklist(), min_memory_per_target=200)
        kept, report = purger.purge([_memory("m3", "CCCCCCCCCCCCCC-X-N", "CCCCCCCCCCCCCC", "d3", "t3")])
        assert "T1" in report.downgraded_targets
        assert report.to_dict()["downgraded_targets"] == ["T1"]


class TestSplitLeakage:
    """§6.1 / §11.2 的划分与记忆库可见性。"""

    def _queries(self, n: int = 600):
        """构造一批查询，每 6 个共享一个骨架家族。"""
        return [
            _query(f"q{i}", f"K{i:012d}-X-N", f"S{i:013d}", f"D{i}", f"T{i}", family=f"FAM{i // 6}")
            for i in range(n)
        ]

    def test_scaffold_family_is_atomic(self):
        """同一骨架家族的分子绝不允许跨 split。"""
        queries = self._queries()
        assignment = SplitBuilder(seed=7).build(queries, "S")
        by_family = {}
        for q in queries:
            by_family.setdefault(q.scaffold_family_id, set()).add(assignment.split_of[q.query_id])
        assert all(len(splits) == 1 for splits in by_family.values())

    def test_split_proportions(self):
        """外层 20% test，dev 内 60:20:20（允许 ±5 个百分点的粒度误差）。"""
        queries = self._queries()
        counts = SplitBuilder(seed=7).build(queries, "S").counts()
        total = sum(counts.values())
        assert abs(counts["test"] / total - 0.20) < 0.05
        assert abs(counts["train"] / total - 0.48) < 0.05      # 0.8 * 0.6
        assert abs(counts["calib"] / total - 0.16) < 0.05      # 0.8 * 0.2

    def test_calibration_fold_excluded_from_its_own_memory(self):
        """§11.2：标定折的骨架家族必须从其自身的记忆库视图中剔除。

        这是 LTT 保证成立的前提 —— 若标定折的家族还在它自己的记忆库里，
        H3 的三条判据同时失效。
        """
        queries = self._queries()
        assignment = SplitBuilder(seed=7).build(queries, "S")
        visible = SplitBuilder.memory_visibility_filter(assignment, "calib")
        calib_families = [u for u, s in assignment.unit_split.items() if s == "calib"]
        train_families = [u for u, s in assignment.unit_split.items() if s == "train"]
        assert calib_families and train_families
        assert all(not visible(f) for f in calib_families)
        assert all(visible(f) for f in train_families)

    def test_ortholog_group_stays_together_under_protocol_t(self):
        """§6.3：同一 ortholog_group 在协议 T 下必须整组同侧。"""
        queries = [
            _query(f"q{i}", f"K{i}", f"S{i}", f"D{i}", f"T{i}", family=f"FAM{i}")
            for i in range(200)
        ]
        queries = [
            QueryRecord(**{**q.__dict__, "ortholog_group_id": f"OG{i % 5}"})
            for i, q in enumerate(queries)
        ]
        assignment = SplitBuilder(seed=3).build(queries, "T")
        by_group = {}
        for q in queries:
            by_group.setdefault(q.ortholog_group_id, set()).add(assignment.split_of[q.query_id])
        assert all(len(splits) == 1 for splits in by_group.values())

    def test_missing_split_key_is_an_error_not_silent(self):
        """缺划分键必须报错 —— 静默会让这些样本无声地落到某一侧。"""
        queries = [_query("q1", "K1", "S1", "D1", "T1", family="")]
        with pytest.raises(ValueError, match="缺少划分键"):
            SplitBuilder().build(queries, "S")


class TestGraphEncoderSharing:
    """§8.3.1：检索管线与模型必须共享同一个 GraphMatcherLite 实例。

    这不是风格问题。``GraphMatcherLite.forward`` 只用 ``w_q``/``w_d``/``w_a``；
    ``node_encoder``/``edge_encoder``/3 层 GINE 只在 ``encode_graph`` 里被用到，
    而 ``encode_graph`` 是**管线**调用的。两个实例 ⇒ 模型这一份 18,883 个参数
    （全部可训练参数的 16.7%）永远拿不到梯度，停在随机初始化 ——
    而损失照常下降、参数量核对照样通过，**失败完全静默**。
    """

    @staticmethod
    def _model():
        from sparc.models.base import BasePredictor
        from sparc.models.evidence import EvidenceLite
        from sparc.models.gate import SupportGate
        from sparc.models.graphmatcher import GraphMatcherLite
        from sparc.models.reranker import RerankerLite
        from sparc.models.residual import ResidualHead, UncertaintyHead
        from sparc.models.sparc_model import SparcNP

        return SparcNP(base=BasePredictor(), graph_matcher=GraphMatcherLite(),
                       reranker=RerankerLite(), evidence=EvidenceLite(),
                       residual=ResidualHead(), uncertainty=UncertaintyHead(),
                       gate=SupportGate())

    @requires_torch
    def test_for_model_shares_the_instance(self):
        """``RetrievalPipeline.for_model`` 是唯一推荐的组装入口。"""
        from sparc.retrieval.pipeline import RetrievalPipeline, assert_graph_matcher_shared

        model = self._model()
        pipeline = RetrievalPipeline.for_model(model, k_src=8, k0=4, k_top=2)
        assert pipeline.graph_matcher is model.graph_matcher
        assert_graph_matcher_shared(model, pipeline)          # 不抛异常即通过

    @requires_torch
    def test_separate_instances_are_rejected(self):
        """自己 new 一个 GraphMatcherLite 必须被硬拦。"""
        from sparc.models.graphmatcher import GraphMatcherLite
        from sparc.retrieval.pipeline import (
            GraphEncoderNotSharedError, RetrievalPipeline, assert_graph_matcher_shared,
        )

        model = self._model()
        stray = RetrievalPipeline(graph_matcher=GraphMatcherLite(), k_src=8, k0=4, k_top=2)
        with pytest.raises(GraphEncoderNotSharedError, match="18,883"):
            assert_graph_matcher_shared(model, stray)

    @requires_torch
    def test_unshared_encoder_never_trains(self):
        """回归夹具：不共享时图编码器权重一步都不动。

        这条测试是这个 bug 的"活证据"—— 它直接测量后果，
        而不是测量"我们记得写了断言"。
        """
        import torch

        from sparc.models.graphmatcher import GraphMatcherLite

        prefixes = ("graph_matcher.node_encoder", "graph_matcher.edge_encoder", "graph_matcher.layers")
        node_feat = torch.randn(6, 76)
        edge_index = torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]])
        edge_feat = torch.randn(5, 16)

        def moved_params(share: bool) -> int:
            torch.manual_seed(0)
            model = self._model()
            encoder = model.graph_matcher if share else GraphMatcherLite()
            opt = torch.optim.Adam(model.parameters(), lr=1e-2)
            before = [p.detach().clone() for n, p in model.named_parameters() if n.startswith(prefixes)]
            lig, pro, gf, y = torch.randn(2, 128), torch.randn(2, 64), torch.randn(2, 28), torch.randn(2)
            for _ in range(3):        # 前 2 步：残差零初始化，Δ=0，梯度还没打通
                opt.zero_grad()
                enc = encoder.encode_graph(node_feat, edge_index, edge_feat)
                batch = {
                    "query_node_repr": enc.unsqueeze(0).expand(2, 6, 64),
                    "cand_node_repr": enc.unsqueeze(0).unsqueeze(0).expand(2, 4, 6, 64),
                    "query_atom_mask": torch.ones(2, 6, dtype=torch.bool),
                    "cand_atom_mask": torch.ones(2, 4, 6, dtype=torch.bool),
                    "cand_repr": torch.randn(2, 4, 256),
                    "rank_scalar_features": torch.rand(2, 4, 7),
                    "label_features": torch.randn(2, 4, 5),
                    "assay_family": torch.randint(0, 32, (2, 4)),
                    "candidate_meta": torch.rand(2, 4, 8),
                    "candidate_mask": torch.ones(2, 4, dtype=torch.bool),
                    "k_top": 2,
                }
                out = model(lig, pro, batch, gf, task_type="B")
                ((out.eta - y) ** 2).mean().backward()
                opt.step()
            after = [p for n, p in model.named_parameters() if n.startswith(prefixes)]
            return sum(1 for a, b in zip(after, before) if float((a.detach() - b).abs().sum()) > 1e-9)

        n_total = sum(1 for n, _ in self._model().named_parameters() if n.startswith(prefixes))
        assert moved_params(share=True) == n_total, "共享实例时图编码器应当全部被训练"
        assert moved_params(share=False) == 0, "不共享时图编码器一步都不该动（这正是 bug 的后果）"


class TestScaffoldPercolation:
    """条款 1 是单连接聚类 —— 阈值偏低会渗流出吞掉一切的巨型家族。

    实测（NPASS 真实分子，Murcko + ECFP4，2026-09-02）::

        阈值    N=7,455   N=16,735   N=129,328(服务器)
        0.50     51.4%     63.3%       99.4%   ❌
        0.65       —        6.0%         —
        0.70      4.2%      4.0%         —     ✅ 平台区

    坍缩不会报错：§5.4 断言查的是四把钥匙的重叠、不是家族大小，
    参数量核对也照样通过。所以必须有主动断言。
    """

    @staticmethod
    def _chain(n: int):
        """构造一条"每个分子只跟下一个同骨架"的链 —— 单连接下会并成一个家族。"""
        from sparc.chem.scaffold import ScaffoldKeys
        # 两把钥匙各自错开一位配对：skeleton14 连 (0,1)(2,3)…，deglyco 连 (1,2)(3,4)…
        # 两者叠加把所有分子串成一条链 ⇒ 单连接下并成一个巨型家族。
        return [
            ScaffoldKeys(
                molecule_id=f"M{i}", inchikey=f"IK{i}",
                skeleton14=f"P{i // 2}",
                deglyco_core_hash=f"Q{(i + 1) // 2}",
                tautomer_family_id=f"T{i}",
            )
            for i in range(n)
        ]

    def test_frozen_threshold_is_070(self, config):
        """v1.0.3 起冻结为 0.70；0.50 会在真实规模下坍缩。"""
        assert config.hparams.split.murcko_tanimoto_threshold == 0.70

    def test_percolation_is_rejected(self):
        """最大家族超过上限 ⇒ 抛 ScaffoldPercolationError，不允许静默继续。"""
        from sparc.chem.scaffold import ScaffoldFamilyBuilder, ScaffoldPercolationError

        builder = ScaffoldFamilyBuilder(max_largest_family_frac=0.20,
                                        percolation_min_molecules=10)
        with pytest.raises(ScaffoldPercolationError, match="渗流坍缩"):
            builder.build(self._chain(100))

    def test_healthy_families_pass(self):
        """家族分布正常时不该误报。"""
        from sparc.chem.scaffold import ScaffoldFamilyBuilder, ScaffoldKeys

        keys = [
            ScaffoldKeys(molecule_id=f"M{i}", inchikey=f"IK{i}", skeleton14=f"S{i}",
                         deglyco_core_hash=f"D{i}", tautomer_family_id=f"T{i}")
            for i in range(50)
        ]
        result = ScaffoldFamilyBuilder(max_largest_family_frac=0.20).build(keys)
        assert len(set(result.family_of.values())) == 50

    def test_guard_can_be_disabled_only_explicitly(self):
        """把上限设为 1.0 才关得掉 —— 必须是刻意动作。"""
        from sparc.chem.scaffold import ScaffoldFamilyBuilder

        result = ScaffoldFamilyBuilder(max_largest_family_frac=1.0,
                                       percolation_min_molecules=10).build(self._chain(100))
        assert len(set(result.family_of.values())) == 1      # 确实并成了一个巨型家族


class TestDegenerateSplitGuard:
    """协议 A 的年份缺失会让划分退化成一个桶，而且不报错 (§6.1)。"""

    @staticmethod
    def _queries(n: int, year):
        from dataclasses import replace
        return [replace(_query(f"q{i}", f"K{i}", f"S{i}", f"D{i}", f"T{i}", family=f"FAM{i}"),
                        reference_year=year)
                for i in range(n)]

    def test_protocol_a_without_years_fails_hard(self):
        """全部 reference_year 为 None ⇒ 所有查询进 year_unknown ⇒ 必须报错。

        修复前这里会静默通过，Table 3 的协议 A 行因此是假的。
        """
        from sparc.data.splits import DegenerateSplitError

        builder = SplitBuilder(0.2, 0.6, 0.2, 0.2, seed=0)
        with pytest.raises(DegenerateSplitError, match="year_unknown"):
            builder.build(self._queries(50, None), "A")

    def test_protocol_a_with_years_succeeds(self):
        """年份齐备时协议 A 正常划分。"""
        from dataclasses import replace

        queries = [replace(q, reference_year=2000 + i % 12)
                   for i, q in enumerate(self._queries(60, 2020))]
        assignment = SplitBuilder(0.2, 0.6, 0.2, 0.2, seed=0).build(queries, "A")
        assert len(set(assignment.unit_of.values())) == 12

    def test_guard_applies_to_every_protocol(self):
        """不只协议 A：任何协议下单一单位吃掉 >80% 都是退化。"""
        from sparc.data.splits import DegenerateSplitError

        queries = [_query(f"q{i}", f"K{i}", f"S{i}", f"D{i}", f"T{i}", family="FAM_ALL")
                   for i in range(40)]
        with pytest.raises(DegenerateSplitError, match="退化"):
            SplitBuilder(0.2, 0.6, 0.2, 0.2, seed=0).build(queries, "S")

    def test_small_sets_are_not_blocked(self):
        """样本太少时占比没有意义，不应误伤（与渗流护栏同一考量）。"""
        queries = [_query(f"q{i}", f"K{i}", f"S{i}", f"D{i}", f"T{i}", family=f"FAM{i % 5}")
                   for i in range(20)]
        assert SplitBuilder(0.2, 0.6, 0.2, 0.2, seed=0).build(queries, "S") is not None



class TestMemoryGate:
    """§5.3 硬闸门：|M_t| < 200 的靶点必须降级或排除，不允许静默继续。"""

    @staticmethod
    def _target(tid: str, tier: str = "tier1"):
        from sparc.data.schema import TargetRecord

        return TargetRecord(
            target_id=tid, target_name=tid, target_type="SINGLE PROTEIN",
            uniprot_id=f"P{tid}", organism_tax_id="9606", organism="Homo sapiens",
            ortholog_group_id=f"OG_{tid}", n_unique_compounds=120,
            censored_frac=0.1, pactivity_std=1.0, tier=tier, has_sequence=True,
        )

    def _purge_report(self, kept_per_target, known):
        from sparc.data.purge import PurgeReport

        report = PurgeReport()
        for tid, n in kept_per_target.items():
            report.per_target_kept[tid] = n
        report.known_target_ids = sorted(known)
        universe = set(report.per_target_kept) | set(report.known_target_ids)
        report.downgraded_targets = sorted(t for t in universe
                                           if report.per_target_kept.get(t, 0) < 200)
        return report

    def test_zero_memory_target_is_caught(self):
        """回归：|M_t| = 0 的靶点此前躲过闸门。

        ``per_target_kept`` 只在有记录被保留时才出现某个 target_id，旧代码
        只遍历它，于是记忆库最空的靶点反而不会被标记 —— 闸门对它恰好失效。
        """
        from sparc.data.blacklist import NaturalProductBlacklist
        from sparc.data.purge import NPPurger

        blacklist = NaturalProductBlacklist(set(), set())
        # T_EMPTY 在候选池里一条记录都没有
        records = [_memory(f"m{i}", f"K{i}", f"S{i}", f"D{i}", f"T{i}", target_id="T_OK")
                   for i in range(5)]
        _, report = NPPurger(blacklist, min_memory_per_target=3).purge(
            records, all_target_ids=["T_OK", "T_EMPTY"])
        assert "T_EMPTY" in report.downgraded_targets
        assert "T_OK" not in report.downgraded_targets
        assert report.to_dict()["min_memory_size"] == 0
        assert report.to_dict()["n_targets_zero_memory"] == 1

    def test_exclude_policy_drops_target_and_its_queries(self):
        from sparc.data.purge import apply_memory_gate

        targets = {"T_OK": self._target("T_OK"), "T_SMALL": self._target("T_SMALL")}
        report = self._purge_report({"T_OK": 500, "T_SMALL": 12}, targets)
        queries = [_query(f"q{i}", f"K{i}", f"S{i}", f"D{i}", f"TA{i}", family=f"F{i}",
                          target_id="T_OK" if i % 2 else "T_SMALL") for i in range(6)]
        memory = [_memory(f"m{i}", f"MK{i}", f"MS{i}", f"MD{i}", f"MT{i}",
                          target_id="T_OK" if i % 2 else "T_SMALL") for i in range(6)]

        new_targets, kept_q, kept_m, summary = apply_memory_gate(targets, report, queries, memory)
        assert new_targets["T_SMALL"].tier == "tier_x"
        assert new_targets["T_OK"].tier == "tier1"
        assert "insufficient_memory(|M_t|=12)" in new_targets["T_SMALL"].exclusion_reasons
        assert {q.target_id for q in kept_q} == {"T_OK"}
        assert {m.target_id for m in kept_m} == {"T_OK"}
        assert summary["n_excluded"] == 1 and summary["n_queries_dropped"] == 3

    def test_downgrade_policy_moves_tier1_to_tier2(self):
        from sparc.data.purge import MemoryGatePolicy, apply_memory_gate

        targets = {"T1": self._target("T1", "tier1"), "T2": self._target("T2", "tier2")}
        report = self._purge_report({"T1": 5, "T2": 5}, targets)
        new_targets, kept_q, _, summary = apply_memory_gate(
            targets, report, [], [], policy=MemoryGatePolicy.DOWNGRADE)
        # tier1 降到 tier2；tier2 没有下一档，只能排除
        assert new_targets["T1"].tier == "tier2"
        assert new_targets["T2"].tier == "tier_x"
        assert summary["n_downgraded"] == 1 and summary["n_excluded"] == 1

    def test_unknown_policy_raises(self):
        import pytest as _pytest

        from sparc.data.purge import apply_memory_gate

        with _pytest.raises(ValueError, match="未知的 policy"):
            apply_memory_gate({}, self._purge_report({}, []), [], [], policy="warn")

    def test_no_flagged_targets_is_a_noop(self):
        from sparc.data.purge import apply_memory_gate

        targets = {"T_OK": self._target("T_OK")}
        report = self._purge_report({"T_OK": 900}, targets)
        queries = [_query("q0", "K0", "S0", "D0", "TA0", family="F0", target_id="T_OK")]
        new_targets, kept_q, kept_m, summary = apply_memory_gate(targets, report, queries, [])
        assert new_targets["T_OK"].tier == "tier1"
        assert len(kept_q) == 1 and summary["n_flagged"] == 0
