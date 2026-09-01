"""NPASS 3.0 读取与靶点入选协议 R1–R8 (§3.1)。

NPASS 文件的三个坑（已核实，不要重新踩）：
1. 制表符分隔，空值记号是字面量 ``"n.a."``，**不是空串**；
2. 某些行超过 csv 默认字段上限，必须 ``csv.field_size_limit(10**7)``；
3. 同一酶的不同物种被登记为不同 ``target_id``（事实 D），
   AChE 人 vs 电鳐共享 68% 的较小化合物集 —— R5 的直系同源合并
   就是为这个存在的。
"""

from __future__ import annotations

import csv
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

from sparc.common.logging_utils import get_logger
from sparc.data.schema import CensorFlag, TargetRecord

_LOGGER = get_logger(__name__)

NULL_TOKEN = "n.a."

# R1：只要单一蛋白靶点
ALLOWED_TARGET_TYPES = {"Individual protein", "Single protein"}

# 事实 E：转运体面板实为一项研究，硬排除（同时由 R4/R6 兜底）
TRANSPORTER_PANEL_TARGETS = {"NPT1028", "NPT1249", "NPT713"}   # MRP4 / MRP3 / BSEP
TRANSPORTER_PANEL_KEYWORDS = ("multidrug resistance-associated protein", "bile salt export pump", "MRP")


def _clean(value: Optional[str]) -> str:
    """把 ``n.a.`` 与空白归一化为空串。"""
    v = (value or "").strip()
    return "" if v == NULL_TOKEN else v


@dataclass
class TargetSelectionReport:
    """R1–R8 的逐条淘汰计数 —— §3.1 明确要求"不预填猜测值"，
    这些数字由 Stage 0 实测产出。"""

    n_targets_input: int = 0
    eliminated: Counter = field(default_factory=Counter)
    ortholog_groups: Dict[str, List[str]] = field(default_factory=dict)
    tier1: List[str] = field(default_factory=list)
    tier2: List[str] = field(default_factory=list)
    tier_x: List[str] = field(default_factory=list)
    n_pairs_selected: int = 0
    n_unique_compounds: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转成可写报告的字典。"""
        return {
            "n_targets_input": self.n_targets_input,
            "eliminated_by_rule": dict(self.eliminated),
            "n_ortholog_groups": len(self.ortholog_groups),
            "ortholog_groups_with_merges": {
                k: v for k, v in self.ortholog_groups.items() if len(v) > 1
            },
            "n_tier1": len(self.tier1),
            "n_tier2": len(self.tier2),
            "n_tier_x": len(self.tier_x),
            "tier1": sorted(self.tier1),
            "tier2": sorted(self.tier2),
            "tier_x": sorted(self.tier_x),
            "n_pairs_selected": self.n_pairs_selected,
            "n_unique_compounds": self.n_unique_compounds,
        }


class NPASSLoader:
    """NPASS 3.0 四张表的读取器。"""

    def __init__(self, npass_dir: Path, field_size_limit: int = 10 ** 7) -> None:
        """
        Args:
            npass_dir: ``data/origin_dataset/NPSS``。
            field_size_limit: csv 字段上限；NPASS 有超长行。
        """
        self.dir = Path(npass_dir)
        csv.field_size_limit(field_size_limit)

    # ------------------------------------------------------------------
    def _iter_tsv(self, filename: str) -> Iterator[Dict[str, str]]:
        """逐行读一张 NPASS 表（流式，不整表进内存）。"""
        path = self.dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"缺少 NPASS 文件：{path}")
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                yield row

    def load_targets(self) -> Dict[str, Dict[str, str]]:
        """读 ``NPASS3.0_target.txt``。

        Returns:
            ``{target_id: {target_type, target_name, uniprot_id, tax_id, organism}}``。
        """
        targets: Dict[str, Dict[str, str]] = {}
        for row in self._iter_tsv("NPASS3.0_target.txt"):
            tid = _clean(row.get("target_id"))
            if not tid:
                continue
            targets[tid] = {
                "target_type": _clean(row.get("target_type")),
                "target_name": _clean(row.get("target_name")),
                "uniprot_id": _clean(row.get("uniprot_id")),
                "organism_tax_id": _clean(row.get("target_organism_tax_id")),
                "organism": _clean(row.get("target_organism")),
            }
        _LOGGER.info("NPASS 靶点表：%d 个 target_id", len(targets))
        return targets

    def load_structures(self) -> Dict[str, Dict[str, str]]:
        """读 ``NPASS3.0_naturalproducts_structure.txt``。

        Returns:
            ``{np_id: {inchikey, smiles}}``。
        """
        structures: Dict[str, Dict[str, str]] = {}
        for row in self._iter_tsv("NPASS3.0_naturalproducts_structure.txt"):
            np_id = _clean(row.get("np_id"))
            key = _clean(row.get("InChIKey"))
            smiles = _clean(row.get("SMILES"))
            if np_id and key and smiles:
                structures[np_id] = {"inchikey": key, "smiles": smiles}
        _LOGGER.info("NPASS 结构表：%d 个可用结构", len(structures))
        return structures

    def load_general_info(self) -> Dict[str, Dict[str, str]]:
        """读 ``NPASS3.0_naturalproducts_generalinfo.txt``。

        ``chembl_id`` 列是事实 B（79.0% 查询自带 ChEMBL ID）的来源，
        NP-purge 报告需要它。
        """
        info: Dict[str, Dict[str, str]] = {}
        for row in self._iter_tsv("NPASS3.0_naturalproducts_generalinfo.txt"):
            np_id = _clean(row.get("np_id"))
            if not np_id:
                continue
            info[np_id] = {
                "inchikey": _clean(row.get("inchikey")),
                "chembl_id": _clean(row.get("chembl_id")),
                "pref_name": _clean(row.get("pref_name")),
            }
        _LOGGER.info("NPASS 通用信息表：%d 条", len(info))
        return info

    def iter_activities(
        self,
        activity_type_grouped: str = "IC50",
        target_ids: Optional[Set[str]] = None,
    ) -> Iterator[Dict[str, str]]:
        """流式读活性表，可按 ``activity_type_grouped`` 与靶点过滤 (R2)。

        Args:
            activity_type_grouped: 冻结为 ``"IC50"``（R2）。
            target_ids: 只保留这些靶点；``None`` 表示不过滤。

        Yields:
            清洗过 ``n.a.`` 的活性行。
        """
        for row in self._iter_tsv("NPASS3.0_activities.txt"):
            if activity_type_grouped and _clean(row.get("activity_type_grouped")) != activity_type_grouped:
                continue
            tid = _clean(row.get("target_id"))
            if target_ids is not None and tid not in target_ids:
                continue
            yield {k: _clean(v) for k, v in row.items()}


# ======================================================================
# R5：直系同源分组
# ======================================================================
_SPECIES_SUFFIX = re.compile(
    r"\s*\((?:human|rat|mouse|bovine|sheep|ovine|electric\s+\w+|torpedo|mushroom|yeast|rabbit|"
    r"porcine|pig|chicken|dog|canine|zebrafish|drosophila|e\.?\s*coli)[^)]*\)\s*$",
    flags=re.IGNORECASE,
)
_TRAILING_SPECIES = re.compile(
    r"\s*(?:from|of)\s+[A-Z][a-z]+\s+[a-z]+\s*$", flags=re.IGNORECASE
)


def normalize_enzyme_name(name: str) -> str:
    """把靶点名归一化到"酶功能"层级，去掉物种修饰。

    R5 要求"同 (酶功能, EC 号) 组内只保留化合物数最多的一个 target_id"。
    NPASS 没有 EC 号列，因此用归一化后的酶名作为分组键；
    有 UniProt 时优先用同一 UniProt 作为更强的证据。

    Args:
        name: NPASS ``target_name``。

    Returns:
        归一化名（小写、去物种、去多余空白）。
    """
    cleaned = _SPECIES_SUFFIX.sub("", name or "")
    cleaned = _TRAILING_SPECIES.sub("", cleaned)
    cleaned = re.sub(r"[^a-zA-Z0-9\s\-]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    return cleaned


def build_ortholog_groups(
    target_meta: Dict[str, Dict[str, str]],
    compound_sets: Optional[Dict[str, Set[str]]] = None,
) -> Dict[str, str]:
    """按 (酶功能, 物种) 构建 ``ortholog_group_id`` (R5, §6.3)。

    **合并判据只有两条，都对应"同一个酶"**：
      1. 归一化酶名相同（去掉物种修饰后）；
      2. UniProt accession 相同。

    **化合物集重叠不作为合并判据。** 这一条是有意为之：NPASS 里
    CYP3A4 / CYP2D6 / CYP1A1 / CYP1B1 / CYP1A2 / CYP2C9 / CYP2C19
    被同一批天然产物筛过，两两重叠远超 50%，但它们是 **7 个不同的酶**，
    不是彼此的直系同源物。按重叠合并会把这 7 个靶点压成 1 个，
    连带 MET/VEGFR/SRC 等激酶也被压成 1 个 —— 实测会让 R1–R8 的
    存活靶点从 31 个掉到 13 个，直接抹掉事实 H 的"6.4 倍免费解"。

    规范引用的三个事实 D 案例都不需要重叠判据：AChE（人/电鳗/电鳐）、
    COX-1（绵羊/人）、酪氨酸酶（双孢蘑菇/人）在 NPASS 里 ``target_name``
    **完全相同**，仅 ``target_organism`` 不同，条款 1 即可全部捕获。

    化合物重叠仍然重要，但它的位置是**诊断报告**而不是合并判据，
    见 :func:`compound_overlap_report`。

    Args:
        target_meta: ``{target_id: {target_name, uniprot_id, ...}}``。
        compound_sets: 仅用于在日志中报告合并组的规模；不参与合并决策。

    Returns:
        ``{target_id: ortholog_group_id}``。
    """
    from sparc.chem.scaffold import _UnionFind  # noqa: PLC0415 - 复用并查集

    tids = sorted(target_meta)
    uf = _UnionFind(tids)

    # 条款 1：归一化酶名相同
    by_name: Dict[str, List[str]] = defaultdict(list)
    for tid in tids:
        name = normalize_enzyme_name(target_meta[tid].get("target_name", ""))
        if name:
            by_name[name].append(tid)
    for group in by_name.values():
        for other in group[1:]:
            uf.union(group[0], other)

    # 条款 2：同一 UniProt accession（同一蛋白被登记了多次）
    by_accession: Dict[str, List[str]] = defaultdict(list)
    for tid in tids:
        accession = target_meta[tid].get("uniprot_id", "")
        if accession:
            by_accession[accession].append(tid)
    for group in by_accession.values():
        for other in group[1:]:
            uf.union(group[0], other)

    groups: Dict[str, str] = {tid: f"OG_{uf.find(tid)}" for tid in tids}
    merged = {g: m for g, m in _invert(groups).items() if len(m) > 1}
    _LOGGER.info(
        "直系同源分组：%d 组，其中 %d 组发生了合并（合并判据：同酶名 / 同 UniProt）",
        len(set(groups.values())), len(merged),
    )
    if compound_sets:
        for group_id, members in sorted(merged.items(), key=lambda kv: -len(kv[1]))[:5]:
            detail = [(t, target_meta[t].get("organism", "?"), len(compound_sets.get(t, ())))
                      for t in members]
            _LOGGER.debug("同源组 %s：%s", group_id, detail)
    return groups


def compound_overlap_report(
    target_meta: Dict[str, Dict[str, str]],
    compound_sets: Dict[str, Set[str]],
    min_compounds: int = 10,
    min_overlap_frac: float = 0.15,
    same_enzyme_only: bool = False,
) -> List[Dict[str, Any]]:
    """靶点间化合物重叠诊断（复现事实 D / 事实 E 的表）。

    **只报告，不合并。** 合并由 :func:`build_ortholog_groups` 按酶名做，
    见那里的说明。

    Args:
        target_meta: 靶点表。
        compound_sets: ``{target_id: {inchikey}}``。
        min_compounds: 参与比较的最小化合物数。
        min_overlap_frac: 报告阈值（占较小集的比例）。
        same_enzyme_only: 只报告归一化酶名相同的配对（事实 D 的口径）。

    Returns:
        重叠配对列表，按重叠比例降序。事实 D 的四对应出现在表头附近；
        事实 E 的 MRP4 ∩ MRP3 = 100% 也会出现，但它们是**不同的酶**，
        由 R4（censored_frac）与显式排除处理，不走 R5。
    """
    candidates = [t for t in sorted(compound_sets) if len(compound_sets[t]) >= min_compounds]
    rows: List[Dict[str, Any]] = []
    for i, a in enumerate(candidates):
        set_a = compound_sets[a]
        name_a = normalize_enzyme_name(target_meta.get(a, {}).get("target_name", ""))
        for b in candidates[i + 1:]:
            set_b = compound_sets[b]
            name_b = normalize_enzyme_name(target_meta.get(b, {}).get("target_name", ""))
            if same_enzyme_only and name_a != name_b:
                continue
            smaller = min(len(set_a), len(set_b))
            if not smaller:
                continue
            overlap = len(set_a & set_b)
            frac = overlap / smaller
            if frac < min_overlap_frac:
                continue
            rows.append({
                "target_a": a, "name_a": target_meta.get(a, {}).get("target_name", ""),
                "organism_a": target_meta.get(a, {}).get("organism", ""), "n_a": len(set_a),
                "target_b": b, "name_b": target_meta.get(b, {}).get("target_name", ""),
                "organism_b": target_meta.get(b, {}).get("organism", ""), "n_b": len(set_b),
                "intersection": overlap, "frac_of_smaller": round(frac, 4),
                "same_enzyme_name": name_a == name_b,
            })
    rows.sort(key=lambda r: -r["frac_of_smaller"])
    _LOGGER.info("化合物重叠诊断：%d 个配对超过 %.0f%%（其中同酶名 %d 个）",
                 len(rows), 100 * min_overlap_frac, sum(1 for r in rows if r["same_enzyme_name"]))
    return rows


def _invert(mapping: Dict[str, str]) -> Dict[str, List[str]]:
    """``{k: v}`` → ``{v: [k]}``。"""
    out: Dict[str, List[str]] = defaultdict(list)
    for k, v in mapping.items():
        out[v].append(k)
    return dict(out)


# ======================================================================
# R1–R8 主流程
# ======================================================================
def select_targets(
    target_meta: Dict[str, Dict[str, str]],
    per_target_compounds: Dict[str, Set[str]],
    per_target_pactivities: Dict[str, List[float]],
    per_target_censor_flags: Dict[str, List[CensorFlag]],
    available_uniprot: Set[str],
    min_compounds: int = 50,
    tier1_min_compounds: int = 100,
    max_censored_frac: float = 0.40,
    min_pactivity_std: float = 0.50,
) -> Tuple[Dict[str, TargetRecord], TargetSelectionReport]:
    """执行 §3.1 的 R1–R8 并分层 (§3.2)。

    规则（依次施加，**不允许跳过任何一条**）::

        R1  target_type ∈ {Individual protein, Single protein}
        R2  activity_type_grouped == 'IC50'          （在调用方过滤活性时施加）
        R3  唯一化合物数 ≥ 50
        R4  censored_frac < 0.40                     # 排除转运体面板（事实 E）
        R5  直系同源冗余：同组只保留化合物数最多者，其余并入同一 ortholog_group_id
        R6  pIC50 标准差 ≥ 0.50                       # 标签方差退化（事实 E）
        R7  必须有可获取的 UniProt 序列
        R8  记录 organism_tax_id，检索时按同一 tax_id 硬过滤（事实 F）

    Args:
        target_meta: NPASS 靶点表。
        per_target_compounds: ``{target_id: {inchikey}}``。
        per_target_pactivities: ``{target_id: [pIC50]}``（已聚合到化合物级）。
        per_target_censor_flags: ``{target_id: [CensorFlag]}``。
        available_uniprot: 本地已有 FASTA 的 UniProt accession 集合（R7）。
        min_compounds: R3 阈值。
        tier1_min_compounds: Tier-1 阈值 (§3.2)。
        max_censored_frac: R4 阈值。
        min_pactivity_std: R6 阈值。

    Returns:
        ``({target_id: TargetRecord}, TargetSelectionReport)``。被淘汰的靶点
        也在返回值里，带 ``exclusion_reasons`` —— Tier-X 需要它们
        （MRP4 必须出现在论文里作为反例，§3.2）。
    """
    report = TargetSelectionReport(n_targets_input=len(target_meta))
    ortholog_of = build_ortholog_groups(target_meta, per_target_compounds)
    report.ortholog_groups = _invert(ortholog_of)

    # R5 预计算：每个同源组里化合物数最多的 target_id 为代表
    group_leader: Dict[str, str] = {}
    for group_id, members in report.ortholog_groups.items():
        group_leader[group_id] = max(members, key=lambda t: len(per_target_compounds.get(t, ())))

    records: Dict[str, TargetRecord] = {}
    for tid, meta in target_meta.items():
        reasons: List[str] = []
        compounds = per_target_compounds.get(tid, set())
        pacts = per_target_pactivities.get(tid, [])
        flags = per_target_censor_flags.get(tid, [])

        n_compounds = len(compounds)
        censored_frac = (sum(1 for f in flags if f.is_censored) / len(flags)) if flags else 0.0
        pact_std = statistics.pstdev(pacts) if len(pacts) > 1 else 0.0

        if meta.get("target_type") not in ALLOWED_TARGET_TYPES:
            reasons.append("R1_target_type")
        if n_compounds < min_compounds:
            reasons.append("R3_too_few_compounds")
        if censored_frac >= max_censored_frac:
            reasons.append("R4_censored_frac")
        group_id = ortholog_of.get(tid, f"OG_{tid}")
        if group_leader.get(group_id) != tid:
            reasons.append("R5_ortholog_redundant")
        if pact_std < min_pactivity_std:
            reasons.append("R6_label_variance_degenerate")
        uniprot = meta.get("uniprot_id", "")
        if not uniprot or uniprot not in available_uniprot:
            reasons.append("R7_no_sequence")
        if not meta.get("organism_tax_id"):
            reasons.append("R8_no_tax_id")
        if tid in TRANSPORTER_PANEL_TARGETS:
            reasons.append("factE_transporter_panel")

        for reason in reasons:
            report.eliminated[reason] += 1

        if reasons:
            tier = "tier_x"
        elif n_compounds >= tier1_min_compounds:
            tier = "tier1"
        else:
            tier = "tier2"

        records[tid] = TargetRecord(
            target_id=tid,
            target_name=meta.get("target_name", ""),
            target_type=meta.get("target_type", ""),
            uniprot_id=uniprot,
            organism_tax_id=meta.get("organism_tax_id", ""),
            organism=meta.get("organism", ""),
            ortholog_group_id=group_id,
            n_unique_compounds=n_compounds,
            censored_frac=censored_frac,
            pactivity_std=pact_std,
            tier=tier,
            has_sequence=uniprot in available_uniprot,
            exclusion_reasons=tuple(reasons),
        )

    report.tier1 = [t for t, r in records.items() if r.tier == "tier1"]
    report.tier2 = [t for t, r in records.items() if r.tier == "tier2"]
    # Tier-X 只保留"曾经有资格但被 R4/R6 淘汰"的靶点，它们是论文的失败边界展示
    report.tier_x = [
        t for t, r in records.items()
        if r.tier == "tier_x"
        and r.n_unique_compounds >= min_compounds
        and any(x in r.exclusion_reasons for x in ("R4_censored_frac", "R6_label_variance_degenerate",
                                                   "factE_transporter_panel"))
    ]
    selected = set(report.tier1) | set(report.tier2)
    report.n_pairs_selected = sum(len(per_target_compounds.get(t, ())) for t in selected)
    report.n_unique_compounds = len(set().union(*[per_target_compounds.get(t, set()) for t in selected])) if selected else 0

    _LOGGER.info(
        "R1–R8 完成：Tier-1 %d / Tier-2 %d / Tier-X %d；(靶点,化合物) 对 %d；唯一化合物 %d",
        len(report.tier1), len(report.tier2), len(report.tier_x),
        report.n_pairs_selected, report.n_unique_compounds,
    )
    return records, report


def load_available_uniprot(fasta_dir: Path) -> Dict[str, str]:
    """扫描本地 FASTA 目录，返回 ``{accession: 序列}`` (R7)。

    Args:
        fasta_dir: ``data/origin_dataset/UniProt/npass_targets``。

    Returns:
        ``{accession: 氨基酸序列}``。目录下有 68 个 accession；
        ``Q7ZJM1``/``Q9YQ12`` 已废弃（§2.2 的待办 T3）。
    """
    sequences: Dict[str, str] = {}
    for path in sorted(Path(fasta_dir).glob("*.fasta")):
        accession = path.stem.split("_")[0]
        lines = path.read_text(encoding="utf-8").splitlines()
        seq = "".join(line.strip() for line in lines if line and not line.startswith(">"))
        if seq:
            sequences[accession] = seq
    _LOGGER.info("本地 UniProt 序列：%d 条（来自 %s）", len(sequences), fasta_dir)
    return sequences
