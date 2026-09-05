"""NP-purge 与四条零重叠断言 (§5.3, §5.4)。

**本项目的核心清洗步骤。** 事实 B：查询池 79.0% 自带 ChEMBL ID。
若不做这一步，Top-1 邻居很可能就是查询分子本身。

§5.4 的四条断言必须 **fail hard**，绝不允许 warning 后继续::

    assert Q.inchikey_set        ∩ M.inchikey_set        == ∅
    assert Q.skeleton14_set      ∩ M.skeleton14_set      == ∅
    assert Q.deglyco_core_set    ∩ M.deglyco_core_set    == ∅
    assert Q.tautomer_family_set ∩ M.tautomer_family_set == ∅
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from sparc.common.logging_utils import get_logger
from sparc.data.blacklist import NaturalProductBlacklist
from sparc.data.schema import MemoryRecord, QueryRecord, TargetRecord

_LOGGER = get_logger(__name__)


class LeakageAssertionError(AssertionError):
    """泄漏断言失败。

    §5.4：任一非空 ⇒ 构建失败，报错退出，**不允许 warning 后继续**。
    这个异常类型存在的唯一目的，是让"降级为警告"这件事在代码里
    显得刻意而不是顺手。
    """


@dataclass
class PurgeReport:
    """NP-purge 的计数报告 —— §5.3 要求这些数字进论文。"""

    n_input: int = 0
    n_kept: int = 0
    n_dropped_exact: int = 0
    n_dropped_skeleton: int = 0
    per_target_kept: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    per_target_dropped: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    downgraded_targets: List[str] = field(default_factory=list)
    # 本轮参与实验的靶点全集。**必须显式传入**：`per_target_kept` 只在有记录被保留时
    # 才出现某个 target_id，于是 |M_t| = 0 的靶点（记忆被全部清洗，或候选池里压根
    # 没出现过）不会成为它的键 —— 记忆库最空的靶点反而躲过为它设的闸门。
    known_target_ids: List[str] = field(default_factory=list)

    @property
    def np_purge_count(self) -> int:
        """被清洗掉的记录总数。"""
        return self.n_dropped_exact + self.n_dropped_skeleton

    @property
    def np_purge_rate(self) -> float:
        """清洗率。"""
        return self.np_purge_count / self.n_input if self.n_input else 0.0

    def per_target_purge_rate(self) -> Dict[str, float]:
        """分靶点清洗率 (§5.3 明确要求分靶点报告)。"""
        rates: Dict[str, float] = {}
        for tid in set(self.per_target_kept) | set(self.per_target_dropped):
            kept = self.per_target_kept.get(tid, 0)
            dropped = self.per_target_dropped.get(tid, 0)
            total = kept + dropped
            rates[tid] = dropped / total if total else 0.0
        return rates

    def to_dict(self) -> Dict[str, Any]:
        """转成可写报告的字典。"""
        # 用全集补零，否则 min_memory_size 会漏掉 |M_t| = 0 的靶点、报出虚高的最小值
        memory_sizes = {tid: 0 for tid in self.known_target_ids}
        memory_sizes.update(self.per_target_kept)
        return {
            "n_input": self.n_input,
            "n_kept": self.n_kept,
            "np_purge_count": self.np_purge_count,
            "np_purge_rate": round(self.np_purge_rate, 4),
            "n_dropped_exact": self.n_dropped_exact,
            "n_dropped_skeleton": self.n_dropped_skeleton,
            "memory_size_per_target": memory_sizes,
            "min_memory_size": min(memory_sizes.values()) if memory_sizes else 0,
            "median_memory_size": sorted(memory_sizes.values())[len(memory_sizes) // 2] if memory_sizes else 0,
            "per_target_purge_rate": {k: round(v, 4) for k, v in self.per_target_purge_rate().items()},
            "n_targets_zero_memory": sum(1 for n in memory_sizes.values() if n == 0),
            "downgraded_targets": sorted(self.downgraded_targets),
        }


class NPPurger:
    """把 COCONUT ∪ LOTUS 从候选记忆池中清洗出去。"""

    def __init__(self, blacklist: NaturalProductBlacklist, min_memory_per_target: int = 200) -> None:
        """
        Args:
            blacklist: COCONUT ∪ LOTUS 黑名单。
            min_memory_per_target: §5.3 的硬闸门。清洗后 ``|M_t|`` 低于此值的
                靶点自动降级到 Tier-2 或排除，**不允许静默继续**。
        """
        self.blacklist = blacklist
        self.min_memory_per_target = min_memory_per_target

    def purge(
        self,
        candidates: Iterable[MemoryRecord],
        all_target_ids: Optional[Iterable[str]] = None,
    ) -> Tuple[List[MemoryRecord], PurgeReport]:
        """执行 §5.3 的清洗。

        Args:
            candidates: 候选记忆池（ChEMBL + BindingDB 合并去重后）。
            all_target_ids: 本轮参与实验的靶点全集。**强烈建议传入** ——
                不传的话闸门只能看见"至少保住了一条记录"的靶点，
                而 ``|M_t| = 0`` 的靶点（记忆被全部清洗，或其 accession
                在 ChEMBL/BindingDB 里压根没有 IC50 记录）不会出现在
                ``per_target_kept`` 中，于是**恰恰逃过为它设的闸门**。

        Returns:
            ``(清洗后的记忆库, PurgeReport)``。
        """
        report = PurgeReport()
        if all_target_ids is not None:
            report.known_target_ids = sorted(set(all_target_ids))
        kept: List[MemoryRecord] = []
        for rec in candidates:
            report.n_input += 1
            hit = self.blacklist.contains(rec.inchikey)
            if hit == "np_exact":
                report.n_dropped_exact += 1
                report.per_target_dropped[rec.target_id] += 1
            elif hit == "np_skeleton":
                report.n_dropped_skeleton += 1
                report.per_target_dropped[rec.target_id] += 1
            else:
                kept.append(rec)
                report.n_kept += 1
                report.per_target_kept[rec.target_id] += 1

        # 三种都要覆盖：保留数不足 / 记忆被全部清洗 / 候选池里从未出现
        universe = set(report.per_target_kept) | set(report.per_target_dropped)
        universe |= set(report.known_target_ids)
        report.downgraded_targets = sorted(
            tid for tid in universe
            if report.per_target_kept.get(tid, 0) < self.min_memory_per_target
        )
        _LOGGER.info(
            "NP-purge：输入 %d → 保留 %d（清洗率 %.1f%%；精确 %d / 骨架块 %d）",
            report.n_input, report.n_kept, 100 * report.np_purge_rate,
            report.n_dropped_exact, report.n_dropped_skeleton,
        )
        if report.downgraded_targets:
            zero = [t for t in report.downgraded_targets
                    if report.per_target_kept.get(t, 0) == 0]
            _LOGGER.warning(
                "以下 %d 个靶点清洗后 |M_t| < %d，按 §5.3 硬闸门必须降级或排除：%s",
                len(report.downgraded_targets), self.min_memory_per_target,
                report.downgraded_targets[:20],
            )
            if zero:
                _LOGGER.warning(
                    "其中 %d 个靶点 |M_t| = 0（记忆被全部清洗，或其 accession 在 "
                    "ChEMBL/BindingDB 里没有 IC50 记录）：%s —— 这类靶点的检索分支"
                    "恒为 insufficient_memory（g̃ ≡ 0），留在实验里只会稀释 "
                    "H1(b)/H3(b) 的靶点级判据分母", len(zero), zero[:20])
        if not report.known_target_ids:
            _LOGGER.warning(
                "purge() 未收到靶点全集，|M_t| = 0 的靶点无法被闸门看见。"
                "请传 all_target_ids —— 见 §5.3。")
        return kept, report


# ======================================================================
# §5.3 硬闸门：|M_t| < 200 的靶点必须降级或排除
# ======================================================================
class MemoryGatePolicy:
    """``|M_t| < 200`` 时对靶点的处置方式（§5.3 给了两个选项）。"""

    EXCLUDE = "exclude"      # 移出实验，记为 tier_x（默认）
    DOWNGRADE = "downgrade"  # tier1 → tier2；已是 tier2 的仍然移出


def apply_memory_gate(
    targets: Dict[str, TargetRecord],
    report: PurgeReport,
    queries: Sequence[QueryRecord],
    memory: Sequence[MemoryRecord],
    policy: str = MemoryGatePolicy.EXCLUDE,
) -> Tuple[Dict[str, TargetRecord], List[QueryRecord], List[MemoryRecord], Dict[str, Any]]:
    """执行 §5.3 的硬闸门：**清洗后 ``|M_t| < 200`` 的靶点不允许静默继续**。

    此前这条闸门只打了一条 warning，靶点照常留在实验里。那不是无害的：
    ``|M_view| < 200`` 会让该靶点的每个查询都落进 ``insufficient_memory``，
    于是 ``g̃ ≡ 0``、检索分支恒等于零，模型在这些靶点上**退化成纯 Base**。
    它们仍然计入 H1 判据 (b)（≥ 2/3 的 Tier-1 靶点）与 H3 判据 (b)
    （≥ 60% 的入选靶点 SafeCoverage > 0）的**分母**，纯粹稀释信号 ——
    一个结构上不可能产生检索证据的靶点，不该被拿去检验"检索证据有没有用"。

    **默认取 ``EXCLUDE``。** §5.3 的原文是"降级到 Tier-2 或排除"，两者都合规；
    选排除是因为降级并不能改变 ``|M_t|``，被降级的靶点在 Tier-2 里照样是
    ``g̃ ≡ 0``，只是换了个分母继续稀释。被排除者进 Tier-X ——
    §3.2 给 Tier-X 的定位正是"作为失败边界写进论文"，这类靶点属于那一类：
    NPASS 有活性数据、而药物侧没有可用证据。

    Args:
        targets: ``{target_id: TargetRecord}``，闸门前的靶点表。
        report: :meth:`NPPurger.purge` 的报告，提供 ``downgraded_targets``。
        queries: 查询集。
        memory: 清洗后的记忆库。
        policy: :class:`MemoryGatePolicy` 之一。

    Returns:
        ``(处置后的靶点表, 过滤后的查询集, 过滤后的记忆库, 闸门报告)``。

    Raises:
        ValueError: ``policy`` 不是合法取值。
    """
    if policy not in (MemoryGatePolicy.EXCLUDE, MemoryGatePolicy.DOWNGRADE):
        raise ValueError(f"未知的 policy：{policy!r}，只接受 "
                         f"{MemoryGatePolicy.EXCLUDE!r} / {MemoryGatePolicy.DOWNGRADE!r}")

    flagged = set(report.downgraded_targets)
    sizes = report.per_target_kept
    new_targets: Dict[str, TargetRecord] = {}
    excluded: List[str] = []
    downgraded: List[str] = []

    for tid, rec in targets.items():
        if tid not in flagged:
            new_targets[tid] = rec
            continue
        reason = f"insufficient_memory(|M_t|={sizes.get(tid, 0)})"
        # 降级只对 tier1 有意义；tier2 再降就没有下一档，只能排除
        if policy == MemoryGatePolicy.DOWNGRADE and rec.tier == "tier1":
            new_targets[tid] = replace(rec, tier="tier2",
                                       exclusion_reasons=rec.exclusion_reasons + (reason,))
            downgraded.append(tid)
        else:
            new_targets[tid] = replace(rec, tier="tier_x",
                                       exclusion_reasons=rec.exclusion_reasons + (reason,))
            excluded.append(tid)

    dropped = set(excluded)
    kept_queries = [q for q in queries if q.target_id not in dropped]
    kept_memory = [m for m in memory if m.target_id not in dropped]

    summary = {
        "policy": policy,
        "n_flagged": len(flagged),
        "n_excluded": len(excluded),
        "n_downgraded": len(downgraded),
        "excluded_targets": sorted(excluded),
        "downgraded_targets": sorted(downgraded),
        "n_queries_dropped": len(queries) - len(kept_queries),
        "n_memory_dropped": len(memory) - len(kept_memory),
        "memory_size_of_flagged": {tid: sizes.get(tid, 0) for tid in sorted(flagged)},
    }
    if flagged:
        _LOGGER.warning(
            "§5.3 硬闸门执行（policy=%s）：%d 个靶点 |M_t| 不足 —— 排除 %d 个、降级 %d 个；"
            "同时丢弃 %d 条查询与 %d 条记忆记录",
            policy, len(flagged), len(excluded), len(downgraded),
            summary["n_queries_dropped"], summary["n_memory_dropped"])
    else:
        _LOGGER.info("§5.3 硬闸门：所有靶点的 |M_t| 均达标，无需降级或排除")
    return new_targets, kept_queries, kept_memory, summary


# ======================================================================
# §5.4 四条断言
# ======================================================================
_ASSERTION_KEYS: Tuple[Tuple[str, str], ...] = (
    ("inchikey", "精确 InChIKey"),
    ("skeleton14", "InChIKey 骨架块（前 14 位）"),
    ("deglyco_core_hash", "脱糖母核（糖苷–苷元通道，§6.2）"),
    ("tautomer_family_id", "互变体家族"),
)


def assert_zero_overlap(
    queries: Sequence[QueryRecord],
    memory: Sequence[MemoryRecord],
    max_examples: int = 10,
    scope: str = "global",
) -> Dict[str, int]:
    """执行 §5.4 的四条零重叠断言。

    Args:
        queries: 查询集 Q。
        memory: 药物记忆库 M。
        max_examples: 报错信息中展示的冲突样例数上限。
        scope: 断言范围描述（如 ``"global"`` 或 ``"fold=3/target=NPT204"``），
            只用于错误信息。

    Returns:
        ``{键名: 重叠数}``（全为 0 时才返回；否则抛异常）。

    Raises:
        LeakageAssertionError: 任一交集非空。
            §5.4 规定"任一非空 ⇒ 构建失败，报错退出"。
    """
    overlaps: Dict[str, int] = {}
    failures: List[str] = []

    for key, description in _ASSERTION_KEYS:
        q_set = {getattr(q, key) for q in queries if getattr(q, key)}
        m_set = {getattr(m, key) for m in memory if getattr(m, key)}
        intersection = q_set & m_set
        overlaps[key] = len(intersection)
        if intersection:
            examples = sorted(intersection)[:max_examples]
            failures.append(
                f"  · {description}（{key}）：{len(intersection)} 个重叠，例如 {examples}"
            )

    if failures:
        raise LeakageAssertionError(
            f"§5.4 零重叠断言失败（scope={scope}）—— 构建终止。\n"
            + "\n".join(failures)
            + "\n\n可能原因：\n"
            "  1. NP-purge 未执行或黑名单未覆盖（事实 B：79.0% 查询自带 ChEMBL ID）；\n"
            "  2. deglyco_core_hash 未计算（事实 C：15.8% 的 COCONUT 是糖苷，"
            "Murcko 保留糖环会让苷元合法留在其糖苷的记忆库里）；\n"
            "  3. 记忆库视图构建时漏了 fold 可见性过滤（§9.1 步骤 1）。\n"
            "§5.4 明确规定：不允许 warning 后继续。"
        )

    _LOGGER.info("§5.4 四条零重叠断言全部通过（scope=%s，Q=%d，M=%d）", scope, len(queries), len(memory))
    return overlaps


def assert_memory_view_sufficient(
    view_size: int,
    min_size: int,
    target_id: str,
    fold: int,
) -> bool:
    """§9.1 的记忆库规模闸门。

    Args:
        view_size: ``|M_view|``。
        min_size: 下限（冻结为 200）。
        target_id: 靶点。
        fold: fold 编号。

    Returns:
        ``True`` 表示视图充足；``False`` 表示该 (fold, 任务) 标记
        ``insufficient_memory``、``g̃ ≡ 0``，**且必须计入报告**。
        这里返回布尔而不是抛异常，是因为 §9.1 规定的是"标记并计入报告"，
        不是"终止构建"—— 与 §5.4 的四条断言语义不同。
    """
    if view_size >= min_size:
        return True
    _LOGGER.warning(
        "记忆库视图不足：target=%s fold=%d |M_view|=%d < %d ⇒ 标记 insufficient_memory，g̃≡0，计入报告",
        target_id, fold, view_size, min_size,
    )
    return False


def audit_query_pool(
    queries: Sequence[QueryRecord],
    blacklist: NaturalProductBlacklist,
    chembl_ids: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """复核事实 B 的五项计数（Stage 0 报告用）。

    Args:
        queries: 查询集。
        blacklist: NP 黑名单。
        chembl_ids: ``{inchikey: chembl_id}``，来自 NPASS generalinfo。

    Returns:
        与 method.md 事实 B 表格同构的计数字典。跑出来的数应与
        3,915 / 3,769 / 3,084 / 3,220 / 435 同量级；差得多说明
        查询集构建口径变了，需要先解释再往下走。
    """
    unique_keys = {q.inchikey for q in queries if q.inchikey}
    n_total = len(unique_keys)
    in_union = sum(1 for k in unique_keys if k in blacklist.full_keys)
    in_coconut = sum(1 for k in unique_keys if k in blacklist.coconut_keys)
    in_lotus = sum(1 for k in unique_keys if k in blacklist.lotus_keys)
    n_glycoside = sum(1 for k in unique_keys if blacklist.is_glycoside(k))
    n_with_chembl = (
        sum(1 for k in unique_keys if chembl_ids and chembl_ids.get(k)) if chembl_ids else -1
    )

    def _pct(n: int) -> float:
        return round(100 * n / n_total, 1) if n_total else 0.0

    return {
        "n_unique_np_in_query_pool": n_total,
        "in_coconut_or_lotus": {"n": in_union, "pct": _pct(in_union)},
        "in_coconut": {"n": in_coconut, "pct": _pct(in_coconut)},
        "in_lotus": {"n": in_lotus, "pct": _pct(in_lotus)},
        "with_chembl_id": {"n": n_with_chembl, "pct": _pct(n_with_chembl) if n_with_chembl >= 0 else None},
        "flagged_glycoside": {"n": n_glycoside, "pct": _pct(n_glycoside)},
        "reference_method_md_fact_b": {
            "n_unique": 4075, "in_union": 3915, "in_coconut": 3769,
            "in_lotus": 3084, "with_chembl_id": 3220, "glycoside": 435,
        },
    }
