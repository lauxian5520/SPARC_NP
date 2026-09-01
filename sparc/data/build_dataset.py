"""Stage 0 数据构建编排 (§5, §13.1)。

顺序（**不可调换**）::

    1. NPASS → R1–R3 初筛 → 活性聚合 → R4–R8 → targets.yaml
    2. 标准化查询分子（四把家族钥匙）
    3. ChEMBL + BindingDB → 候选记忆池 → 按 InChIKey 去重合并
    4. NP-purge（COCONUT ∪ LOTUS）           ← §5.3，先于任何其它步骤
    5. 骨架家族构建（四条规则）
    6. §5.4 四条零重叠断言                    ← fail hard
    7. 协议 S/T/A/X 划分 + 指纹落盘
    8. Table 0（化学域偏移量化）+ Stage 0 闸门

"NP-purge before anything else"：第 4 步必须在第 5–7 步之前。
否则骨架家族会把"查询分子自己"当成记忆库邻居一起并进家族，
让后面的断言看起来通过、实际泄漏。
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from sparc.common.config import ExperimentConfig
from sparc.common.logging_utils import get_logger
from sparc.data.blacklist import NaturalProductBlacklist
from sparc.data.chembl import RawActivity
from sparc.data.purge import NPPurger, PurgeReport, assert_zero_overlap, audit_query_pool
from sparc.data.schema import CensorFlag, Domain, MemoryRecord, QueryRecord, TargetRecord
from sparc.data.splits import SplitAssignment, SplitBuilder, summarize_split
from sparc.data.units import UnitConversionReport, aggregate_pactivity, to_pactivity

_LOGGER = get_logger(__name__)


@dataclass
class MoleculeAnnotation:
    """一个分子的标准化结果 + 四把家族钥匙 + 描述符。"""

    inchikey: str
    smiles: str
    skeleton14: str
    deglyco_core_hash: str
    tautomer_family_id: str
    murcko_smiles: str = ""
    is_glycoside: bool = False
    descriptors: Dict[str, float] = field(default_factory=dict)


class MoleculeAnnotator:
    """把 SMILES 批量转成 :class:`MoleculeAnnotation`。

    需要 RDKit。本机（树莓派）不可用 —— 这一步在 CUDA 服务器上跑。
    """

    def __init__(self, blacklist: Optional[NaturalProductBlacklist] = None, compute_descriptors: bool = False) -> None:
        """
        Args:
            blacklist: 提供 COCONUT 糖苷标记，加速 :class:`SugarStripper`。
            compute_descriptors: 是否计算 Table 0 描述符（较慢，Stage 0 才需要）。
        """
        from sparc.chem.fingerprints import FingerprintCalculator  # noqa: PLC0415
        from sparc.chem.standardize import MoleculeStandardizer  # noqa: PLC0415
        from sparc.chem.sugar import SugarStripper  # noqa: PLC0415

        self.standardizer = MoleculeStandardizer()
        self.stripper = SugarStripper(coconut_glycoside_flags=(blacklist.glycoside_flags if blacklist else None))
        self.fingerprints = FingerprintCalculator()
        self.blacklist = blacklist
        self.compute_descriptors = compute_descriptors
        self.n_failed = 0

    def annotate(self, smiles: str) -> Optional[MoleculeAnnotation]:
        """标准化并计算四把钥匙。

        Args:
            smiles: 原始 SMILES。

        Returns:
            :class:`MoleculeAnnotation`；标准化失败返回 ``None`` 并计数。
        """
        std = self.standardizer.standardize(smiles)
        if not std.ok or not std.inchikey or not std.canonical_smiles:
            self.n_failed += 1
            return None
        deglyco = self.stripper.deglyco_core_hash(std.canonical_smiles, std.inchikey)
        descriptors: Dict[str, float] = {}
        if self.compute_descriptors:
            desc = self.fingerprints.descriptors(std.canonical_smiles)
            if desc:
                descriptors = desc.to_dict()
            if self.blacklist and std.inchikey in self.blacklist.np_likeness:
                descriptors["np_likeness"] = self.blacklist.np_likeness[std.inchikey]
        return MoleculeAnnotation(
            inchikey=std.inchikey,
            smiles=std.canonical_smiles,
            skeleton14=std.skeleton14 or std.inchikey[:14],
            deglyco_core_hash=deglyco.deglyco_core_hash or std.skeleton14 or std.inchikey[:14],
            tautomer_family_id=std.tautomer_family_id or std.skeleton14 or std.inchikey[:14],
            murcko_smiles=std.murcko_smiles or "",
            is_glycoside=(deglyco.n_sugars_removed > 0 or deglyco.used_coconut_flag),
            descriptors=descriptors,
        )

    def annotate_many(self, smiles_list: Sequence[str], log_every: int = 5000) -> Dict[str, MoleculeAnnotation]:
        """批量标注，按输入 SMILES 建索引。"""
        out: Dict[str, MoleculeAnnotation] = {}
        for i, smi in enumerate(smiles_list, 1):
            if smi in out:
                continue
            ann = self.annotate(smi)
            if ann:
                out[smi] = ann
            if i % log_every == 0:
                _LOGGER.info("标准化进度 %d/%d（失败 %d）", i, len(smiles_list), self.n_failed)
        _LOGGER.info("标准化完成：%d 成功 / %d 失败", len(out), self.n_failed)
        return out


# ======================================================================
# 记忆库构建
# ======================================================================
def merge_memory_sources(
    raw_activities: Iterable[RawActivity],
    annotations: Dict[str, MoleculeAnnotation],
    target_by_uniprot: Dict[Tuple[str, str], TargetRecord],
    config: ExperimentConfig,
    molecular_weights: Optional[Dict[str, float]] = None,
) -> Tuple[List[MemoryRecord], UnitConversionReport, Dict[str, int]]:
    """把 ChEMBL/BindingDB 原始活性合并成候选记忆池 (§5.2 步骤 2–4)。

    Args:
        raw_activities: 两个来源的原始活性流。
        annotations: ``{原始 smiles: MoleculeAnnotation}``。
        target_by_uniprot: ``{(uniprot_id, tax_id): TargetRecord}``。
            **按 (accession, tax_id) 而不是只按 accession 索引** ——
            事实 F：NaFM 的"人源靶点"里酪氨酸酶是双孢蘑菇、COX-1 是绵羊，
            跨物种靶点必须按同一物种酶匹配 (R8)。
        config: 实验配置（提供聚合方式与极差上限）。
        molecular_weights: ``{inchikey: MW}``，质量浓度单位换算需要。

    Returns:
        ``(候选记忆池, 单位换算报告, QC 计数)``。
    """
    data_cfg = config.hparams.data
    unit_report = UnitConversionReport()
    qc: Dict[str, int] = defaultdict(int)

    # (inchikey, target_id) -> [(pactivity, censor_flag, assay_id, source_db, smiles)]
    grouped: Dict[Tuple[str, str], List[Tuple[float, CensorFlag, str, str, str]]] = defaultdict(list)

    for raw in raw_activities:
        ann = annotations.get(raw.smiles)
        if ann is None:
            qc["drop_standardize_failed"] += 1
            continue
        target = target_by_uniprot.get((raw.uniprot_id, raw.organism_tax_id))
        if target is None:
            # R8：tax_id 对不上的记录不能当作同靶点证据
            qc["drop_target_or_taxid_mismatch"] += 1
            continue
        mw = (molecular_weights or {}).get(ann.inchikey)
        pact = to_pactivity(raw.value, raw.units, molecular_weight=mw, report=unit_report)
        if pact is None:
            qc["drop_unit_conversion"] += 1
            continue
        flag = CensorFlag(data_cfg.censor_flag(raw.relation))
        grouped[(ann.inchikey, target.target_id)].append((pact, flag, raw.assay_id, raw.source_db, ann.smiles))

    memory: List[MemoryRecord] = []
    for (inchikey, target_id), entries in grouped.items():
        values = [e[0] for e in entries]
        aggregated, status = aggregate_pactivity(values, data_cfg.aggregate, data_cfg.max_pactivity_spread)
        if status == "high_spread":
            qc["qc_high_spread_drop"] += 1
            continue
        if aggregated is None:
            qc["drop_empty_group"] += 1
            continue
        # 审查标志的聚合：全部同向才保留审查语义，混合则退回 none 并计数
        flags = {e[1] for e in entries}
        if len(flags) == 1:
            flag = next(iter(flags))
        else:
            flag = CensorFlag.NONE
            qc["mixed_censor_downgraded"] += 1

        smiles = entries[0][4]
        target = next(t for t in target_by_uniprot.values() if t.target_id == target_id)
        ann = next((a for a in [None]), None)  # 占位，实际用下方 annotation 索引
        memory.append(MemoryRecord(
            memory_id=f"{target_id}::{inchikey}",
            inchikey=inchikey,
            smiles=smiles,
            target_id=target_id,
            uniprot_id=target.uniprot_id,
            organism_tax_id=target.organism_tax_id,
            pactivity=aggregated,
            censor_flag=flag,
            skeleton14="", deglyco_core_hash="", tautomer_family_id="",   # 下方回填
            domain=Domain.DRUG,
            assay_family=_assay_family_id(entries[0][2]),
            endpoint="IC50",
            source_db=entries[0][3],
            n_source_records=len(entries),
        ))

    # 回填四把钥匙（用 inchikey 反查 annotation）
    by_key = {a.inchikey: a for a in annotations.values()}
    filled: List[MemoryRecord] = []
    for rec in memory:
        ann = by_key.get(rec.inchikey)
        if ann is None:
            qc["drop_no_annotation"] += 1
            continue
        filled.append(MemoryRecord(
            **{**rec.__dict__,
               "skeleton14": ann.skeleton14,
               "deglyco_core_hash": ann.deglyco_core_hash,
               "tautomer_family_id": ann.tautomer_family_id,
               "murcko_smiles": ann.murcko_smiles}
        ))

    _LOGGER.info("候选记忆池：%d 条；单位换算 %s；QC %s",
                 len(filled), unit_report.to_dict(), dict(qc))
    return filled, unit_report, dict(qc)


def _assay_family_id(assay_id: str, n_families: int = 32) -> int:
    """把 assay_id 映射到 0..31 的 assay family（Evidence 的 embedding 索引）。

    §8.3.3 规定 assay family embedding 为 32 族 × 8 维。ChEMBL 的
    assay 数远超 32，因此用稳定哈希分桶 —— 目的不是语义分类，
    而是给 Evidence 一个低维、跨 fold 稳定的 assay 上下文。
    """
    return (hash(assay_id) if assay_id else 0) % n_families if assay_id else 0


# ======================================================================
# Table 0：化学域偏移量化
# ======================================================================
def build_table0(
    queries: Sequence[QueryRecord],
    memory: Sequence[MemoryRecord],
    query_descriptors: Dict[str, Dict[str, float]],
    memory_descriptors: Dict[str, Dict[str, float]],
    top1_tanimoto: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Table 0 —— 化学域偏移量化 (§12.4)。

    **必须在主表之前产出。** §13.1 的 Stage 0 闸门：若药物记忆库与
    天然产物查询集在化学空间上分不开（Top-1 Tanimoto 中位数 > 0.6），
    所谓"跨域"不成立 ⇒ 停止，重新定义域。

    Args:
        queries: 清洗后的查询集。
        memory: 清洗后的药物记忆库。
        query_descriptors: ``{inchikey: 描述符字典}``。
        memory_descriptors: 同上。
        top1_tanimoto: 每个查询对记忆库的 Top-1 Tanimoto；闸门判定用。

    Returns:
        Table 0 的内容字典（可直接渲染成 markdown）。
    """
    import statistics  # noqa: PLC0415

    def _column_stats(keys: Iterable[str], source: Dict[str, Dict[str, float]], column: str) -> Dict[str, float]:
        values = [source[k][column] for k in keys if k in source and column in source[k]]
        if not values:
            return {"n": 0}
        return {
            "n": len(values),
            "mean": round(statistics.fmean(values), 4),
            "median": round(statistics.median(values), 4),
            "std": round(statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
            "p10": round(sorted(values)[int(0.10 * (len(values) - 1))], 4),
            "p90": round(sorted(values)[int(0.90 * (len(values) - 1))], 4),
        }

    q_keys = sorted({q.inchikey for q in queries})
    m_keys = sorted({m.inchikey for m in memory})
    columns = ["np_likeness", "fraction_csp3", "n_stereocenters", "ring_complexity",
               "heavy_atom_count", "n_rings", "n_aromatic_rings", "mol_weight", "logp"]

    table: Dict[str, Any] = {
        "n_query_molecules": len(q_keys),
        "n_memory_molecules": len(m_keys),
        "descriptors": {
            col: {
                "natural_products": _column_stats(q_keys, query_descriptors, col),
                "drug_memory": _column_stats(m_keys, memory_descriptors, col),
            }
            for col in columns
        },
    }

    # 骨架重叠率 —— 清洗后应为 0（§5.4 断言 2 保证）
    q_skel = {q.skeleton14 for q in queries}
    m_skel = {m.skeleton14 for m in memory}
    table["scaffold_overlap"] = {
        "n_query_skeletons": len(q_skel),
        "n_memory_skeletons": len(m_skel),
        "n_intersection": len(q_skel & m_skel),
        "overlap_rate": round(len(q_skel & m_skel) / len(q_skel), 6) if q_skel else 0.0,
    }

    if top1_tanimoto:
        values = sorted(top1_tanimoto)
        table["top1_tanimoto"] = {
            "n": len(values),
            "median": round(statistics.median(values), 4),
            "mean": round(statistics.fmean(values), 4),
            "p10": round(values[int(0.10 * (len(values) - 1))], 4),
            "p90": round(values[int(0.90 * (len(values) - 1))], 4),
            "frac_below_0.35": round(sum(1 for v in values if v < 0.35) / len(values), 4),
        }
    return table


def check_stage0_gate(table0: Dict[str, Any], config: ExperimentConfig) -> Tuple[bool, List[str]]:
    """执行 §13.1 的 Stage 0 闸门。

    Args:
        table0: :func:`build_table0` 的产出。
        config: 实验配置（提供闸门阈值）。

    Returns:
        ``(是否通过, 违规原因列表)``。**不通过时调用方必须停止**，
        §13.1 原话："说明所谓'跨域'不成立 ⇒ 停止，重新定义域"。
    """
    gate = config.prereg.stage0_gate
    violations: List[str] = []

    top1 = table0.get("top1_tanimoto", {})
    if top1:
        median = top1.get("median", 0.0)
        limit = gate["max_median_top1_tanimoto"]
        if median > limit:
            violations.append(
                f"Top-1 Tanimoto 中位数 {median:.3f} > {limit} —— 药物记忆库与天然产物查询集"
                "在化学空间上分不开，'跨域'前提不成立（§13.1 Stage 0 闸门）"
            )
    else:
        violations.append("Table 0 缺少 top1_tanimoto，无法判定 Stage 0 闸门")

    overlap = table0.get("scaffold_overlap", {}).get("n_intersection", 0)
    if overlap:
        violations.append(f"清洗后仍有 {overlap} 个骨架块重叠 —— §5.4 断言 2 本应拦住它")

    return (not violations), violations


# ======================================================================
# 落盘
# ======================================================================
def write_stage0_outputs(
    output_dir: Path,
    queries: Sequence[QueryRecord],
    memory: Sequence[MemoryRecord],
    targets: Dict[str, TargetRecord],
    splits: Dict[str, SplitAssignment],
    reports: Dict[str, Any],
) -> Dict[str, Path]:
    """把 Stage 0 的全部产物落盘。

    Args:
        output_dir: ``data/s0_data/outputs``。
        queries: 查询集。
        memory: 记忆库。
        targets: 靶点记录。
        splits: ``{协议: SplitAssignment}``。
        reports: 各类报告（NP-purge / 单位换算 / Table 0 / R1–R8）。

    Returns:
        ``{产物名: 路径}``。
    """
    import csv  # noqa: PLC0415

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}

    def _write_tsv(name: str, rows: Sequence[Dict[str, Any]]) -> None:
        path = output_dir / name
        if not rows:
            path.write_text("", encoding="utf-8")
            written[name] = path
            return
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
        written[name] = path

    _write_tsv("queries.tsv", [q.to_dict() for q in queries])
    _write_tsv("memory.tsv", [m.to_dict() for m in memory])
    _write_tsv("targets.tsv", [t.to_dict() for t in targets.values()])

    for protocol, assignment in splits.items():
        path = output_dir / f"split_{protocol}.json"
        path.write_text(json.dumps(assignment.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        written[f"split_{protocol}"] = path

    report_path = output_dir / "stage0_report.json"
    report_path.write_text(json.dumps(reports, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    written["stage0_report"] = report_path

    _LOGGER.info("Stage 0 产物已落盘：%s", {k: str(v.name) for k, v in written.items()})
    return written
