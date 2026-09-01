"""划分协议 S / T / A / X 与外层三分 (§6.1, §11.2)。

外层结构（硬性，§11.2）::

    外层 test 20%   完全隔离，只在 S5 触碰一次
    外层 dev  80%   内部再分 60 : 20 : 20
      ├── 60%  Θ_B 与 Θ_R 训练
      ├── 20%  内层模型选择（λ_rank 等网格）
      └── 20%  λ 标定折  ← 该折的骨架家族同时从其自身的记忆库视图中剔除

最后那半句是整个 LTT 保证成立的前提：若标定折参与过 Base 训练，
或它的骨架家族还留在它自己的记忆库里，H3 的三条判据同时失效 (§11.1)。
:meth:`SplitBuilder.memory_visibility_filter` 就是它的实现。

四个协议的划分单位::

    S（主协议）  骨架家族          —— 全部主结果
    T            靶点 / ortholog_group —— cold-target 泛化
    A            assay / 文献年份   —— 时序外推
    X            生物来源（属/科）  —— 天然产物特有的分布偏移
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from sparc.common.logging_utils import get_logger
from sparc.data.schema import QueryRecord

_LOGGER = get_logger(__name__)

SPLIT_TRAIN = "train"
SPLIT_INNER = "inner"
SPLIT_CALIB = "calib"
SPLIT_TEST = "test"
ALL_SPLITS = (SPLIT_TRAIN, SPLIT_INNER, SPLIT_CALIB, SPLIT_TEST)


@dataclass
class SplitAssignment:
    """一次划分的结果。"""

    protocol: str
    split_of: Dict[str, str]                  # query_id -> split
    unit_of: Dict[str, str]                   # query_id -> 划分单位 id
    unit_split: Dict[str, str]                # 划分单位 id -> split
    seed: int
    fingerprint: str = ""                     # 划分内容的 sha256，写入 splits/ 目录

    def ids(self, split: str) -> List[str]:
        """取某一 split 的 query_id 列表。"""
        return [q for q, s in self.split_of.items() if s == split]

    def counts(self) -> Dict[str, int]:
        """各 split 的样本数。"""
        out: Dict[str, int] = {s: 0 for s in ALL_SPLITS}
        for s in self.split_of.values():
            out[s] = out.get(s, 0) + 1
        return out

    def unit_counts(self) -> Dict[str, int]:
        """各 split 的划分单位数。"""
        out: Dict[str, int] = {s: 0 for s in ALL_SPLITS}
        for s in self.unit_split.values():
            out[s] = out.get(s, 0) + 1
        return out

    def to_dict(self) -> Dict[str, Any]:
        """转成可写 JSON 的字典（落盘到 ``data/processed/splits/``）。"""
        return {
            "protocol": self.protocol,
            "seed": self.seed,
            "fingerprint": self.fingerprint,
            "counts": self.counts(),
            "unit_counts": self.unit_counts(),
            "split_of": self.split_of,
            "unit_split": self.unit_split,
        }

    def compute_fingerprint(self) -> str:
        """计算划分指纹 —— 划分一旦落盘就不允许改（§10.4 的 S4→S5 纪律）。"""
        payload = "|".join(f"{k}={self.split_of[k]}" for k in sorted(self.split_of))
        self.fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return self.fingerprint


class SplitBuilder:
    """按协议构造划分，划分单位为原子（不可被拆散）。"""

    def __init__(
        self,
        outer_test_frac: float = 0.20,
        dev_train_frac: float = 0.60,
        dev_inner_frac: float = 0.20,
        dev_calib_frac: float = 0.20,
        seed: int = 42,
    ) -> None:
        """
        Args:
            outer_test_frac: 外层 test 比例。
            dev_train_frac: dev 内训练比例。
            dev_inner_frac: dev 内模型选择比例。
            dev_calib_frac: dev 内 λ 标定比例。
            seed: 随机种子。
        """
        total = dev_train_frac + dev_inner_frac + dev_calib_frac
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"dev 三分之和必须为 1.0，当前 {total}")
        self.outer_test_frac = outer_test_frac
        self.dev_fracs = (dev_train_frac, dev_inner_frac, dev_calib_frac)
        self.seed = seed

    # ------------------------------------------------------------------
    def build(self, queries: Sequence[QueryRecord], protocol: str = "S") -> SplitAssignment:
        """按协议构造划分。

        Args:
            queries: 查询集。
            protocol: ``"S"`` / ``"T"`` / ``"A"`` / ``"X"``。

        Returns:
            :class:`SplitAssignment`。

        Raises:
            ValueError: 协议未知，或某些查询缺少该协议所需的划分键。
        """
        unit_fn = self._unit_function(protocol)
        unit_of: Dict[str, str] = {}
        missing: List[str] = []
        for q in queries:
            unit = unit_fn(q)
            if not unit:
                missing.append(q.query_id)
                continue
            unit_of[q.query_id] = unit
        if missing:
            raise ValueError(
                f"协议 {protocol} 下有 {len(missing)} 条查询缺少划分键（例如 {missing[:5]}）。"
                "缺键会让这些样本无声地落到某一侧 —— 必须先补齐或显式剔除。"
            )

        units = sorted(set(unit_of.values()))
        unit_sizes = {u: 0 for u in units}
        for unit in unit_of.values():
            unit_sizes[unit] += 1

        unit_split = self._assign_units(units, unit_sizes)
        assignment = SplitAssignment(
            protocol=protocol,
            split_of={q: unit_split[u] for q, u in unit_of.items()},
            unit_of=unit_of,
            unit_split=unit_split,
            seed=self.seed,
        )
        assignment.compute_fingerprint()
        _LOGGER.info(
            "协议 %s 划分完成：样本 %s；划分单位 %s；指纹 %s",
            protocol, assignment.counts(), assignment.unit_counts(), assignment.fingerprint[:12],
        )
        return assignment

    # ------------------------------------------------------------------
    @staticmethod
    def _unit_function(protocol: str) -> Callable[[QueryRecord], str]:
        """返回协议对应的划分单位提取函数。"""
        if protocol == "S":
            # 骨架家族 —— 四把钥匙的连通分量，见 sparc.chem.scaffold
            return lambda q: q.scaffold_family_id
        if protocol == "T":
            # 同一 ortholog_group 必须整组同侧 (§6.3)
            return lambda q: q.ortholog_group_id or q.target_id
        if protocol == "A":
            # assay / 文献年份；无年份的落到 "year_unknown" 单独一桶
            return lambda q: f"year_{q.reference_year}" if q.reference_year else "year_unknown"
        if protocol == "X":
            return lambda q: q.source_organism_family
        raise ValueError(f"未知协议 '{protocol}'，可选 S / T / A / X")

    def _assign_units(self, units: Sequence[str], unit_sizes: Dict[str, int]) -> Dict[str, str]:
        """把划分单位分配到四个 split，尽量贴近目标样本比例。

        用"最大剩余需求优先"的贪心：把单位按样本数从大到小排，
        每次投给"当前缺额最大"的 split。相比纯随机，它在单位大小
        极不均衡时（骨架家族天然如此：少数大家族 + 大量单例）
        能显著减少比例偏移。随机性由 seed 控制的打散提供。
        """
        rng = np.random.default_rng(self.seed)
        order = list(units)
        rng.shuffle(order)
        order.sort(key=lambda u: unit_sizes[u], reverse=True)

        n_total = sum(unit_sizes.values())
        dev_frac = 1.0 - self.outer_test_frac
        targets = {
            SPLIT_TEST: self.outer_test_frac * n_total,
            SPLIT_TRAIN: dev_frac * self.dev_fracs[0] * n_total,
            SPLIT_INNER: dev_frac * self.dev_fracs[1] * n_total,
            SPLIT_CALIB: dev_frac * self.dev_fracs[2] * n_total,
        }
        current = {s: 0.0 for s in targets}
        assignment: Dict[str, str] = {}
        for unit in order:
            deficits = {s: targets[s] - current[s] for s in targets}
            chosen = max(deficits, key=lambda s: (deficits[s], s))
            assignment[unit] = chosen
            current[chosen] += unit_sizes[unit]
        return assignment

    # ------------------------------------------------------------------
    @staticmethod
    def memory_visibility_filter(
        assignment: SplitAssignment,
        eval_split: str,
    ) -> Callable[[str], bool]:
        """构造记忆库视图的 fold 可见性过滤器 (§9.1 步骤 1, §11.2)。

        规则：评估某个 split 时，**该 split 自身的划分单位**必须从
        它的记忆库视图中剔除。对标定折尤其关键 —— §11.2 原话：
        "该折的骨架家族同时从其自身的记忆库视图中剔除"。

        Args:
            assignment: 划分结果。
            eval_split: 正在评估的 split。

        Returns:
            ``fn(unit_id) -> bool``，``True`` 表示该单位的记忆记录可见。

        Note:
            药物记忆库里的分子经过 NP-purge 后不会出现在查询集中，
            但它们仍可能与查询共享骨架家族（Murcko Tanimoto ≥ 0.50 连边）。
            因此这个过滤器按**划分单位**而不是按分子 ID 工作。
        """
        forbidden = {u for u, s in assignment.unit_split.items() if s == eval_split}

        def _visible(unit_id: str) -> bool:
            return unit_id not in forbidden

        return _visible


def summarize_split(assignment: SplitAssignment, queries: Sequence[QueryRecord]) -> Dict[str, Any]:
    """给出划分的分靶点摘要，用于 Stage 0 报告与 Table 3。

    Args:
        assignment: 划分结果。
        queries: 查询集。

    Returns:
        含每个 split 的样本数、靶点数、以及分靶点样本数的字典。
    """
    by_split: Dict[str, List[QueryRecord]] = defaultdict(list)
    for q in queries:
        split = assignment.split_of.get(q.query_id)
        if split:
            by_split[split].append(q)

    summary: Dict[str, Any] = {"protocol": assignment.protocol, "fingerprint": assignment.fingerprint}
    for split in ALL_SPLITS:
        records = by_split.get(split, [])
        per_target: Dict[str, int] = defaultdict(int)
        for r in records:
            per_target[r.target_id] += 1
        summary[split] = {
            "n_queries": len(records),
            "n_targets": len(per_target),
            "n_units": sum(1 for s in assignment.unit_split.values() if s == split),
            "per_target": dict(sorted(per_target.items())),
        }
    return summary
