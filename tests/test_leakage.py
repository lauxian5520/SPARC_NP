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


def _query(qid: str, key: str, skel: str, deglyco: str, taut: str, family: str = "FAM0") -> QueryRecord:
    """构造一条查询记录。"""
    return QueryRecord(
        query_id=qid, compound_id=qid, inchikey=key, smiles="CCO",
        target_id="T1", uniprot_id="P1", organism_tax_id="9606",
        pactivity=6.0, censor_flag=CensorFlag.NONE,
        skeleton14=skel, deglyco_core_hash=deglyco, tautomer_family_id=taut,
        scaffold_family_id=family, ortholog_group_id="OG1",
        source_organism_family="Rosaceae", reference_year=2020,
    )


def _memory(mid: str, key: str, skel: str, deglyco: str, taut: str, family: str = "FAM9") -> MemoryRecord:
    """构造一条记忆库记录。"""
    return MemoryRecord(
        memory_id=mid, inchikey=key, smiles="CCC",
        target_id="T1", uniprot_id="P1", organism_tax_id="9606",
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
