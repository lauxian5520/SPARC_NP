"""Stage 0 的完整管线（步骤 3–8）—— 需要 RDKit 与已解包的 ChEMBL 37。

由 :mod:`run_s0_data` 在非 ``--dry-run`` 时调用。拆成独立模块是为了
让 ``--dry-run`` 路径完全不 import RDKit —— 树莓派上必须能跑通 dry-run。

**顺序纪律**：NP-purge（步骤 4）必须在骨架家族（步骤 5）与划分（步骤 7）之前。
否则骨架家族会把"查询分子自己"当成记忆库邻居一起并进家族，
让后面的断言看起来通过、实际泄漏。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from sparc.data.blacklist import NaturalProductBlacklist
from sparc.data.build_dataset import (
    MoleculeAnnotator,
    build_table0,
    check_stage0_gate,
    merge_memory_sources,
    write_stage0_outputs,
)
from sparc.data.purge import NPPurger, apply_memory_gate, assert_zero_overlap
from sparc.data.schema import CensorFlag, Domain, MemoryRecord, QueryRecord, TargetRecord
from sparc.data.splits import DegenerateSplitError, SplitBuilder


def run_full_pipeline(
    config: Any,
    logger: Any,
    loader: Any,
    targets: Dict[str, TargetRecord],
    selection_report: Any,
    structures: Dict[str, Dict[str, str]],
    blacklist: NaturalProductBlacklist,
    per_target_compounds: Dict[str, Set[str]],
    outputs: Path,
    run_name: str,
) -> Dict[str, Any]:
    """执行步骤 3–8。

    Args:
        config: 实验配置。
        logger: 日志器。
        loader: :class:`~sparc.data.npass.NPASSLoader`。
        targets: R1–R8 的靶点记录。
        selection_report: 靶点筛选报告。
        structures: NPASS 结构表。
        blacklist: COCONUT ∪ LOTUS 黑名单。
        per_target_compounds: ``{target_id: {inchikey}}``。
        outputs: 产物目录。
        run_name: run 名称。

    Returns:
        追加到 Stage 0 报告的字典。
    """
    paths = config.paths
    hparams = config.hparams
    report: Dict[str, Any] = {}
    selected = {tid: rec for tid, rec in targets.items() if rec.tier in ("tier1", "tier2")}
    logger.info("入选靶点 %d 个（Tier-1 %d / Tier-2 %d）",
                len(selected), len(selection_report.tier1), len(selection_report.tier2))

    # ---------- 步骤 3：标准化查询分子 ----------
    logger.info("== 步骤 3：标准化查询分子（四把家族钥匙） ==")
    annotator = MoleculeAnnotator(blacklist=blacklist, compute_descriptors=True)
    key_to_smiles = {rec["inchikey"]: rec["smiles"] for rec in structures.values()}
    query_keys = sorted(set().union(*[per_target_compounds[t] for t in selected])) if selected else []
    annotations = annotator.annotate_many([key_to_smiles[k] for k in query_keys if k in key_to_smiles])
    by_inchikey = {ann.inchikey: ann for ann in annotations.values()}
    report["standardization"] = {"n_input": len(query_keys), "n_ok": len(annotations),
                                 "n_failed": annotator.n_failed}

    queries = _build_queries(loader, selected, per_target_compounds, key_to_smiles,
                             by_inchikey, blacklist, config, logger)
    logger.info("查询集 Q：%d 条", len(queries))

    # ---------- 步骤 4：候选记忆池 + NP-purge ----------
    logger.info("== 步骤 4：ChEMBL/BindingDB 抽取 → NP-purge（先于一切） ==")
    memory_candidates, memory_annotations, year_by_inchikey, memory_report = _build_memory(
        config, logger, selected, blacklist, annotator, outputs,
    )
    report["memory_extraction"] = memory_report

    # 协议 A 的年份必须在划分（步骤 7）之前就位
    queries = _backfill_query_years(queries, year_by_inchikey, logger)

    purger = NPPurger(blacklist, min_memory_per_target=hparams.retrieval.min_memory_view)
    # 传入靶点全集，否则 |M_t| = 0 的靶点不会出现在 per_target_kept 里，
    # 于是恰恰逃过为它设的闸门（§5.3）
    memory, purge_report = purger.purge(memory_candidates, all_target_ids=selected.keys())
    report["np_purge"] = purge_report.to_dict()
    logger.info("NP-purge：%d → %d（清洗率 %.1f%%）",
                purge_report.n_input, purge_report.n_kept, 100 * purge_report.np_purge_rate)

    # ---------- 步骤 4b：§5.3 硬闸门（|M_t| < 200 ⇒ 降级或排除） ----------
    # 必须在骨架家族与划分之前执行：被排除的靶点连同其查询一起离场，
    # 否则它们会带着一个恒为 insufficient_memory（g̃ ≡ 0）的检索分支
    # 进入 H1(b)/H3(b) 的靶点级分母，纯粹稀释信号。
    selected, queries, memory, gate_summary = apply_memory_gate(
        selected, purge_report, queries, memory)
    # 闸门改的是 selected 的 tier，必须回写进 targets —— targets.yaml 是从它落盘的，
    # 不回写的话降级/排除只存在于内存里，后续阶段读到的仍是旧 tier。
    targets = {**targets, **selected}
    report["memory_gate"] = gate_summary
    logger.info("§5.3 闸门后：靶点 %d 个（排除 %d）、查询 %d 条、记忆 %d 条",
                sum(1 for r in selected.values() if r.tier in ("tier1", "tier2")),
                gate_summary["n_excluded"], len(queries), len(memory))

    # ---------- 步骤 5：骨架家族 ----------
    logger.info("== 步骤 5：骨架家族（四条规则） ==")
    queries, memory, family_summary = _assign_scaffold_families(queries, memory, by_inchikey,
                                                               memory_annotations, config, logger)
    report["scaffold_families"] = family_summary

    # ---------- 步骤 6：§5.4 四条断言 ----------
    logger.info("== 步骤 6：§5.4 四条零重叠断言（fail hard） ==")
    report["leakage_assertions"] = assert_zero_overlap(queries, memory, scope="stage0_global")

    # ---------- 步骤 7：协议划分 ----------
    logger.info("== 步骤 7：协议 S/T/A/X 划分 ==")
    splits = {}
    split_summary = {}
    builder = SplitBuilder(
        hparams.split.outer_test_frac, hparams.split.dev_train_frac,
        hparams.split.dev_inner_frac, hparams.split.dev_calib_frac, seed=config.effective_seed,
    )
    for protocol in hparams.split.protocols:
        try:
            assignment = builder.build(queries, protocol)
            splits[protocol] = assignment
            split_summary[protocol] = {"counts": assignment.counts(), "units": assignment.unit_counts(),
                                       "fingerprint": assignment.fingerprint}
        except (ValueError, DegenerateSplitError) as exc:
            # 协议 S 是主协议，全部主结果都建立在它上面 —— 它退化就没有实验可做。
            # T/A/X 只进 Table 3 的压力测试，失败记录下来继续。
            if protocol == "S":
                logger.error("主协议 S 划分失败，Stage 0 终止：%s", exc)
                raise
            logger.warning("协议 %s 划分失败（Table 3 该行将缺失）：%s", protocol, exc)
            split_summary[protocol] = {"error": str(exc)}
    report["splits"] = split_summary

    # ---------- 步骤 8：Table 0 + 闸门 ----------
    logger.info("== 步骤 8：Table 0 与 Stage 0 闸门 ==")
    top1 = _top1_tanimoto(queries, memory, config, logger)
    table0 = build_table0(
        queries, memory,
        {k: a.descriptors for k, a in by_inchikey.items() if a.descriptors},
        {k: a.descriptors for k, a in memory_annotations.items() if a.descriptors},
        top1_tanimoto=top1,
    )
    report["table0"] = table0

    passed, violations = check_stage0_gate(table0, config)
    report["stage0_gate"] = {"passed": passed, "violations": violations}
    if not passed:
        logger.error("Stage 0 闸门未通过：\n%s", "\n".join(violations))
        raise SystemExit(
            "Stage 0 闸门未通过（§13.1）。所谓「跨域」不成立 ⇒ 停止，重新定义域。\n"
            + "\n".join(violations)
        )
    logger.info("Stage 0 闸门通过。")

    # ---------- 落盘 ----------
    from sparc.eval.tables import render_table0, write_report  # noqa: PLC0415

    write_stage0_outputs(outputs, queries, memory, targets, splits, report)
    write_report(render_table0(table0), paths.get("reports") / "table0.md")
    return report


# ======================================================================
def _build_queries(loader, selected, per_target_compounds, key_to_smiles, by_inchikey,
                   blacklist, config, logger) -> List[QueryRecord]:
    """把 NPASS 活性组装成查询集 Q。"""
    from collections import defaultdict  # noqa: PLC0415

    from sparc.data.units import aggregate_pactivity, to_pactivity  # noqa: PLC0415

    data_cfg = config.hparams.data
    organisms = blacklist.load_lotus_organisms(config.paths.file("lotus_gz", "lotus_dir"), rank="genus")

    grouped = defaultdict(list)
    refs: Dict[Tuple[str, str], str] = {}
    flags = defaultdict(list)
    structures = loader.load_structures()
    np_to_key = {np_id: rec["inchikey"] for np_id, rec in structures.items()}

    for row in loader.iter_activities("IC50", target_ids=set(selected)):
        key = np_to_key.get(row.get("np_id", ""))
        target_id = row.get("target_id", "")
        if not key or target_id not in selected:
            continue
        ann = by_inchikey.get(key)
        if ann is None:
            continue
        try:
            value = float(row.get("activity_value") or "nan")
        except ValueError:
            continue
        pact = to_pactivity(value, row.get("activity_units", ""),
                            molecular_weight=None if not ann else None)
        if pact is None:
            continue
        grouped[(target_id, key)].append(pact)
        flags[(target_id, key)].append(CensorFlag(data_cfg.censor_flag(row.get("activity_relation", "="))))
        # NPASS 只给 ref_id / ref_id_type（多为 PMID），**没有年份列**。
        # 年份在 _build_memory 之后由 ChEMBL 的 docs.year 按 InChIKey 回填
        # （事实 B：79.0% 的查询自带 ChEMBL ID）。这里只记来源，不再塞 None ——
        # 旧代码 `years.setdefault(key, None)` 写的是字面 None，
        # 于是协议 A 把所有查询归进同一个 "year_unknown" 桶而不报错。
        ref = row.get("ref_id", "")
        if ref.isdigit():
            refs[(target_id, key)] = ref

    queries: List[QueryRecord] = []
    for (target_id, key), values in grouped.items():
        aggregated, status = aggregate_pactivity(values, data_cfg.aggregate, data_cfg.max_pactivity_spread)
        if aggregated is None:
            continue
        ann = by_inchikey[key]
        target = selected[target_id]
        flag_set = set(flags[(target_id, key)])
        flag = next(iter(flag_set)) if len(flag_set) == 1 else CensorFlag.NONE
        queries.append(QueryRecord(
            query_id=f"{target_id}::{key}", compound_id=key, inchikey=key, smiles=ann.smiles,
            target_id=target_id, uniprot_id=target.uniprot_id, organism_tax_id=target.organism_tax_id,
            pactivity=aggregated, censor_flag=flag,
            skeleton14=ann.skeleton14, deglyco_core_hash=ann.deglyco_core_hash,
            tautomer_family_id=ann.tautomer_family_id, murcko_smiles=ann.murcko_smiles,
            ortholog_group_id=target.ortholog_group_id,
            source_organism_family=organisms.get(key, "unknown"),
            reference_year=None,   # 由 _backfill_query_years() 在记忆库建好后回填
            n_source_records=len(values), is_glycoside=ann.is_glycoside,
        ))
    logger.info("查询集自带 PMID 引用 %d/%d 条（NPASS 无年份列，年份改由 ChEMBL docs.year 回填）",
                len(refs), len(queries))
    return queries


def _backfill_query_years(queries, year_by_inchikey, logger) -> List[QueryRecord]:
    """用 ChEMBL 的 ``docs.year`` 按 InChIKey 回填查询的 ``reference_year``。

    NPASS 只给 ``ref_id``/``ref_id_type``（多为 PMID），没有年份列；而事实 B 说
    **79.0% 的查询池自带 ChEMBL ID**，所以绝大多数查询分子在 ChEMBL 里有对应的
    活性记录，可以直接取其最早发表年份。

    这是协议 A（时序外推，§6.1）唯一的年份来源。修复前 ``reference_year`` 恒为
    ``None``，:meth:`SplitAssigner._unit_function` 于是把**所有**查询归进同一个
    ``"year_unknown"`` 桶 —— 划分退化成一个单位，而且不报错。

    Args:
        queries: 待回填的查询集。
        year_by_inchikey: ``{inchikey: 最早年份}``，来自 ChEMBL 原始活性流。
        logger: 日志器。

    Returns:
        回填后的查询集（新对象；``QueryRecord`` 是 frozen dataclass）。
    """
    filled = [QueryRecord(**{**q.__dict__,
                             "reference_year": year_by_inchikey.get(q.inchikey)})
              for q in queries]
    n_known = sum(1 for q in filled if q.reference_year is not None)
    coverage = n_known / len(filled) if filled else 0.0
    logger.info("查询年份回填：%d/%d（%.1f%%）有年份", n_known, len(filled), coverage * 100)
    if filled and coverage < 0.50:
        logger.warning(
            "查询年份覆盖 %.1f%% < 50%%：协议 A（时序外推）的划分单位会被 "
            "\"year_unknown\" 主导，Table 3 的该行不可信。SplitAssigner 会在超限时 fail hard。",
            coverage * 100)
    return filled


def _build_memory(config, logger, selected, blacklist, annotator, outputs):
    """从 ChEMBL + BindingDB 抽取候选记忆池并标准化。"""
    from sparc.data.bindingdb import BindingDBExtractor  # noqa: PLC0415
    from sparc.data.chembl import ChEMBLExtractor  # noqa: PLC0415

    paths = config.paths
    accessions = sorted({rec.uniprot_id for rec in selected.values() if rec.uniprot_id})
    tax_map = {rec.uniprot_id: rec.organism_tax_id for rec in selected.values()}
    target_by_uniprot = {(rec.uniprot_id, rec.organism_tax_id): rec for rec in selected.values()}

    raw: List[Any] = []
    db_path = paths.get("interim") / paths.raw["files"]["chembl_sqlite"]
    with ChEMBLExtractor(db_path) as extractor:
        logger.info("ChEMBL 表规模：%s", extractor.table_counts())
        raw.extend(extractor.extract(accessions, "IC50"))

    bindingdb_zip = paths.file("bindingdb_zip", "bindingdb_dir")
    if bindingdb_zip.is_file():
        raw.extend(BindingDBExtractor(bindingdb_zip).extract(accessions, tax_map))
    else:
        logger.warning("BindingDB 压缩包不存在，跳过该来源（§2.3-T2）")

    smiles_list = sorted({r.smiles for r in raw if r.smiles})
    logger.info("候选记忆池原始记录 %d 条，唯一 SMILES %d 个", len(raw), len(smiles_list))
    annotations = annotator.annotate_many(smiles_list)
    by_inchikey = {ann.inchikey: ann for ann in annotations.values()}

    # 供 _backfill_query_years() 用：同一分子在多条记录里出现时取最早年份。
    # 走 annotations 把原始 SMILES 映成标准 InChIKey，才能与查询集对上。
    year_by_inchikey: Dict[str, int] = {}
    for record in raw:
        if record.year is None:
            continue
        ann = annotations.get(record.smiles)
        if ann is None:
            continue
        current = year_by_inchikey.get(ann.inchikey)
        if current is None or record.year < current:
            year_by_inchikey[ann.inchikey] = record.year
    logger.info("ChEMBL 年份索引：%d 个 InChIKey 有发表年份", len(year_by_inchikey))

    molecular_weights: Dict[str, float] = {}
    memory, unit_report, qc = merge_memory_sources(raw, annotations, target_by_uniprot, config, molecular_weights)
    return memory, by_inchikey, year_by_inchikey, {
        "n_raw": len(raw), "unit_conversion": unit_report.to_dict(), "qc": qc,
        "n_inchikey_with_year": len(year_by_inchikey),
    }


def _assign_scaffold_families(queries, memory, query_annotations, memory_annotations, config, logger):
    """给查询与记忆库统一构建骨架家族（同一个并查集，否则划分无意义）。"""
    from sparc.chem.fingerprints import FingerprintCalculator  # noqa: PLC0415
    from sparc.chem.scaffold import ScaffoldFamilyBuilder, ScaffoldKeys  # noqa: PLC0415

    split_cfg = config.hparams.split
    calculator = FingerprintCalculator(config.hparams.retrieval.ecfp_radius,
                                       config.hparams.retrieval.ecfp_n_bits)

    keys: List[ScaffoldKeys] = []
    seen: Set[str] = set()
    for record in list(queries) + list(memory):
        if record.inchikey in seen:
            continue
        seen.add(record.inchikey)
        keys.append(ScaffoldKeys(
            molecule_id=record.inchikey, inchikey=record.inchikey,
            skeleton14=record.skeleton14, deglyco_core_hash=record.deglyco_core_hash,
            tautomer_family_id=record.tautomer_family_id, murcko_smiles=record.murcko_smiles or None,
        ))

    murcko_fps = {}
    for key in keys:
        if key.murcko_smiles and key.murcko_smiles not in murcko_fps:
            fingerprint = calculator.ecfp4(key.murcko_smiles)
            if fingerprint is not None:
                murcko_fps[key.murcko_smiles] = fingerprint

    result = ScaffoldFamilyBuilder(
        split_cfg.murcko_tanimoto_threshold, split_cfg.use_inchikey_skeleton,
        split_cfg.use_deglyco_core, split_cfg.use_tautomer_family,
        # 渗流护栏必须从冻结配置取值：之前漏传，靠类默认值恰好也是 0.20 才没出事，
        # 一旦有人在 frozen_hparams.yaml 里收紧这个阈值，冻结哈希会变而行为不会变。
        max_largest_family_frac=split_cfg.max_largest_family_frac,
    ).build(keys, murcko_fps)

    queries = [QueryRecord(**{**q.__dict__, "scaffold_family_id": result.family_of.get(q.inchikey, "")})
               for q in queries]
    memory = [MemoryRecord(**{**m.__dict__, "scaffold_family_id": result.family_of.get(m.inchikey, "")})
              for m in memory]
    return queries, memory, result.summary()


def _top1_tanimoto(queries, memory, config, logger) -> List[float]:
    """计算每个查询对同靶点记忆库的 Top-1 Tanimoto（Stage 0 闸门判据）。"""
    from collections import defaultdict  # noqa: PLC0415

    from sparc.chem.fingerprints import FingerprintCalculator, tanimoto_matrix  # noqa: PLC0415

    calculator = FingerprintCalculator(config.hparams.retrieval.ecfp_radius,
                                       config.hparams.retrieval.ecfp_n_bits)
    memory_by_target = defaultdict(list)
    for record in memory:
        memory_by_target[record.target_id].append(record)

    values: List[float] = []
    for target_id, group in memory_by_target.items():
        fingerprints, ok = calculator.ecfp4_batch([r.smiles for r in group])
        if not len(ok):
            continue
        target_queries = [q for q in queries if q.target_id == target_id]
        if not target_queries:
            continue
        query_fps, query_ok = calculator.ecfp4_batch([q.smiles for q in target_queries])
        if not len(query_ok):
            continue
        similarity = tanimoto_matrix(query_fps, fingerprints)
        values.extend(similarity.max(axis=1).tolist())
    logger.info("Top-1 Tanimoto 计算完成：%d 个查询", len(values))
    return values
