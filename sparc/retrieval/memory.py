"""fold-scoped 记忆库视图 (§9.1 步骤 1, §11.2)。

**先建过滤视图，再建索引。** 过滤键（冻结在 ``frozen_hparams.yaml``）::

    同 UniProt  +  同 tax_id  +  同 endpoint  +  fold 可见性

其中 tax_id 是事实 F 的直接后果：NaFM 的"人源靶点"里酪氨酸酶是
双孢蘑菇、COX-1 是绵羊。跨物种靶点的药物证据必须按同一物种酶匹配，
否则"靶点同一性"假设不成立。

fold 可见性 (§11.2)：评估某个 split 时，该 split 自身的划分单位
必须从它的记忆库视图中剔除。对标定折尤其关键 —— 若标定折的骨架家族
还留在它自己的记忆库里，H3 的三条判据同时失效。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from sparc.common.logging_utils import get_logger
from sparc.data.schema import MemoryRecord

_LOGGER = get_logger(__name__)


@dataclass
class MemoryView:
    """一个 (fold, 靶点) 组合下的记忆库视图。"""

    target_id: str
    uniprot_id: str
    organism_tax_id: str
    endpoint: str
    split: str
    records: List[MemoryRecord] = field(default_factory=list)
    insufficient: bool = False        # |M_view| < 200 ⇒ 标记并计入报告，g̃ ≡ 0

    @property
    def size(self) -> int:
        """``|M_view|``。"""
        return len(self.records)

    def key(self) -> Tuple[str, str, str, str]:
        """视图的唯一键。"""
        return (self.target_id, self.organism_tax_id, self.endpoint, self.split)

    def to_dict(self) -> Dict[str, Any]:
        """写进报告的摘要（不含记录本体）。"""
        return {
            "target_id": self.target_id,
            "uniprot_id": self.uniprot_id,
            "organism_tax_id": self.organism_tax_id,
            "endpoint": self.endpoint,
            "split": self.split,
            "size": self.size,
            "insufficient_memory": self.insufficient,
        }


class MemoryViewBuilder:
    """按冻结的过滤键构建记忆库视图。"""

    def __init__(
        self,
        memory: Sequence[MemoryRecord],
        min_view_size: int = 200,
        view_filter_keys: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Args:
            memory: NP-purge 之后的全量药物记忆库。
            min_view_size: §9.1 闸门（冻结为 200）。
            view_filter_keys: 过滤键；默认 ``["uniprot_id", "organism_tax_id", "endpoint"]``。
        """
        self.memory = list(memory)
        self.min_view_size = min_view_size
        self.filter_keys = list(view_filter_keys or ["uniprot_id", "organism_tax_id", "endpoint"])
        self._by_filter: Dict[Tuple[str, ...], List[MemoryRecord]] = defaultdict(list)
        for record in self.memory:
            self._by_filter[self._filter_key(record)].append(record)
        _LOGGER.info(
            "记忆库索引就绪：%d 条记录 → %d 个 (%s) 分组",
            len(self.memory), len(self._by_filter), "+".join(self.filter_keys),
        )

    def _filter_key(self, record: MemoryRecord) -> Tuple[str, ...]:
        """按冻结的过滤键提取分组键。"""
        return tuple(str(getattr(record, key, "")) for key in self.filter_keys)

    # ------------------------------------------------------------------
    def build(
        self,
        target_id: str,
        uniprot_id: str,
        organism_tax_id: str,
        endpoint: str,
        split: str,
        visibility: Optional[Callable[[str], bool]] = None,
        query_keys: Optional[Dict[str, Set[str]]] = None,
    ) -> MemoryView:
        """构建一个视图。

        Args:
            target_id: 靶点。
            uniprot_id: UniProt accession（硬过滤键）。
            organism_tax_id: 物种 tax_id（硬过滤键，事实 F/R8）。
            endpoint: 活性类型（硬过滤键，冻结为 IC50）。
            split: 正在评估的 split，决定 fold 可见性。
            visibility: ``fn(scaffold_family_id) -> bool``，来自
                :meth:`~sparc.data.splits.SplitBuilder.memory_visibility_filter`。
            query_keys: 该 split 查询集的四把钥匙集合；提供时做**逐视图**
                的零重叠自检，把 §5.4 的全局断言下推到每个视图。

        Returns:
            :class:`MemoryView`。``size < min_view_size`` 时 ``insufficient=True``
            —— §9.1 规定标记并计入报告，不静默 fallback。
        """
        key = tuple(str(v) for v in (uniprot_id, organism_tax_id, endpoint))
        candidates = self._by_filter.get(key, [])

        if visibility is not None:
            candidates = [r for r in candidates if visibility(r.scaffold_family_id)]

        if query_keys:
            candidates = self._drop_leaking(candidates, query_keys, target_id, split)

        view = MemoryView(
            target_id=target_id, uniprot_id=uniprot_id, organism_tax_id=organism_tax_id,
            endpoint=endpoint, split=split, records=candidates,
            insufficient=len(candidates) < self.min_view_size,
        )
        if view.insufficient:
            _LOGGER.warning(
                "视图不足：target=%s split=%s |M_view|=%d < %d ⇒ insufficient_memory，g̃≡0，计入报告",
                target_id, split, view.size, self.min_view_size,
            )
        return view

    @staticmethod
    def _drop_leaking(
        candidates: Sequence[MemoryRecord],
        query_keys: Dict[str, Set[str]],
        target_id: str,
        split: str,
    ) -> List[MemoryRecord]:
        """逐视图剔除与查询集共享任一把钥匙的记录。

        全局 NP-purge 之后本不该有命中；有命中说明黑名单或钥匙计算
        出了问题，因此这里 **计数并告警**，让它在日志里留下痕迹，
        而不是悄悄修好。
        """
        kept: List[MemoryRecord] = []
        n_dropped = 0
        for record in candidates:
            leaking = any(
                getattr(record, key, "") and getattr(record, key) in values
                for key, values in query_keys.items()
            )
            if leaking:
                n_dropped += 1
            else:
                kept.append(record)
        if n_dropped:
            _LOGGER.error(
                "视图 target=%s split=%s 中发现 %d 条与查询集共享泄漏键的记录并已剔除。"
                "全局 NP-purge 后不应出现这种情况 —— 请检查黑名单覆盖与四把钥匙的计算。",
                target_id, split, n_dropped,
            )
        return kept

    # ------------------------------------------------------------------
    def build_all(
        self,
        targets: Sequence[Tuple[str, str, str]],
        split: str,
        endpoint: str = "IC50",
        visibility: Optional[Callable[[str], bool]] = None,
    ) -> Dict[str, MemoryView]:
        """为一组靶点批量构建视图。

        Args:
            targets: ``[(target_id, uniprot_id, organism_tax_id)]``。
            split: 评估的 split。
            endpoint: 活性类型。
            visibility: fold 可见性过滤器。

        Returns:
            ``{target_id: MemoryView}``。
        """
        views = {
            target_id: self.build(target_id, uniprot, tax_id, endpoint, split, visibility)
            for target_id, uniprot, tax_id in targets
        }
        n_insufficient = sum(1 for v in views.values() if v.insufficient)
        sizes = sorted(v.size for v in views.values())
        _LOGGER.info(
            "split=%s 视图构建完成：%d 个靶点，|M_view| 中位数 %d，不足 %d 个",
            split, len(views), sizes[len(sizes) // 2] if sizes else 0, n_insufficient,
        )
        return views


def memory_view_report(views: Dict[str, MemoryView]) -> Dict[str, Any]:
    """记忆库视图的规模报告（Table 3 的 ``|M_view|`` 中位数列）。"""
    sizes = sorted(v.size for v in views.values())
    return {
        "n_views": len(views),
        "median_size": sizes[len(sizes) // 2] if sizes else 0,
        "min_size": sizes[0] if sizes else 0,
        "max_size": sizes[-1] if sizes else 0,
        "n_insufficient": sum(1 for v in views.values() if v.insufficient),
        "insufficient_targets": sorted(t for t, v in views.items() if v.insufficient),
        "per_target": {t: v.size for t, v in sorted(views.items())},
    }
