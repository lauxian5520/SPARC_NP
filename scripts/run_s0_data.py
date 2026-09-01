#!/usr/bin/env python3
"""Stage 0 —— 数据构建 (§5, §13.1)。

顺序（不可调换）::

    NPASS → R1–R8 → 标准化 → ChEMBL/BindingDB → NP-purge
    → 骨架家族 → §5.4 四条断言 → 协议 S/T/A/X 划分 → Table 0 → Stage 0 闸门

用法::

    python scripts/run_s0_data.py --run-name s0_v1
    python scripts/run_s0_data.py --dry-run          # 只跑靶点筛选与黑名单，不碰 ChEMBL

**需要 RDKit**（除 ``--dry-run``）。本机（树莓派）跑不了完整流程；
``--dry-run`` 可以在本机跑，用来核对 R1–R8 的淘汰计数与事实 B 的五项。
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

from _bootstrap import base_parser, setup

from sparc.data.blacklist import NaturalProductBlacklist
from sparc.data.npass import NPASSLoader, load_available_uniprot, select_targets
from sparc.data.schema import CensorFlag
from sparc.data.units import UnitConversionReport, aggregate_pactivity, to_pactivity

STAGE = "s0_data"


def main() -> int:
    """Stage 0 主流程。"""
    parser = base_parser(STAGE, __doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="只跑 NPASS 靶点筛选与黑名单统计（无需 RDKit / ChEMBL）")
    parser.add_argument("--min-compounds", type=int, default=50, help="R3 阈值")
    parser.add_argument("--tier1-min-compounds", type=int, default=100, help="Tier-1 阈值")
    args = parser.parse_args()

    config, logger, _, guard = setup(STAGE, args, need_device=False)
    paths = config.paths
    data_cfg = config.hparams.data
    outputs = paths.stage_outputs(STAGE)
    report: Dict[str, object] = {"stage": STAGE, "run_name": args.run_name}

    # ---------- 1. NPASS + R1–R8 ----------
    logger.info("== 步骤 1：NPASS 读取与 R1–R8 靶点筛选 ==")
    loader = NPASSLoader(paths.get("npass_dir"), data_cfg.csv_field_size_limit)
    target_meta = loader.load_targets()
    structures = loader.load_structures()
    general_info = loader.load_general_info()
    sequences = load_available_uniprot(paths.get("uniprot_dir") / "npass_targets")

    # 分子量表：ug.mL-1 换算必需。NPASS 不提供 MW，只能由 SMILES 算 ——
    # 因此需要 RDKit。缺 RDKit 时会丢掉全部质量浓度记录（约 19.6k 条），
    # §5.2 步骤 3 要求"丢弃并计数"，这里如实执行并在报告中标明口径差异。
    molecular_weights = _build_mw_table(structures, logger)

    # 聚合活性到 (靶点, 化合物) 级
    per_target_values: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    per_target_flags: Dict[str, List[CensorFlag]] = defaultdict(list)
    per_target_raw_compounds: Dict[str, Set[str]] = defaultdict(set)   # 规范 §1.2 事实 H 的口径
    unit_report = UnitConversionReport()
    n_rows = 0
    for row in loader.iter_activities("IC50"):
        n_rows += 1
        np_id = row.get("np_id", "")
        target_id = row.get("target_id", "")
        structure = structures.get(np_id)
        if not structure or not target_id:
            continue
        per_target_raw_compounds[target_id].add(structure["inchikey"])
        try:
            value = float(row.get("activity_value") or "nan")
        except ValueError:
            continue
        pact = to_pactivity(
            value, row.get("activity_units", ""),
            molecular_weight=molecular_weights.get(structure["inchikey"]), report=unit_report,
        )
        if pact is None:
            continue
        per_target_values[target_id][structure["inchikey"]].append(pact)
        per_target_flags[target_id].append(CensorFlag(data_cfg.censor_flag(row.get("activity_relation", "="))))

    per_target_compounds: Dict[str, Set[str]] = {}
    per_target_pactivities: Dict[str, List[float]] = {}
    n_dropped_spread = 0
    for target_id, by_compound in per_target_values.items():
        compounds: Set[str] = set()
        values: List[float] = []
        for inchikey, records in by_compound.items():
            aggregated, status = aggregate_pactivity(records, data_cfg.aggregate, data_cfg.max_pactivity_spread)
            if status == "high_spread":
                n_dropped_spread += 1
                continue
            if aggregated is not None:
                compounds.add(inchikey)
                values.append(aggregated)
        per_target_compounds[target_id] = compounds
        per_target_pactivities[target_id] = values

    targets, selection_report = select_targets(
        target_meta, per_target_compounds, per_target_pactivities, per_target_flags,
        available_uniprot=set(sequences),
        min_compounds=args.min_compounds, tier1_min_compounds=args.tier1_min_compounds,
        max_censored_frac=0.40, min_pactivity_std=0.50,
    )
    # 两个口径并列，让 §1.2 事实 H 的 70/25/7314/4075 与换算后的实际数可比
    spec_scale = _spec_scale_audit(target_meta, per_target_raw_compounds)
    converted_scale = _converted_scale_audit(target_meta, per_target_compounds)
    report["npass"] = {
        "n_ic50_rows_scanned": n_rows,
        "unit_conversion": unit_report.to_dict(),
        "qc_high_spread_drop": n_dropped_spread,
        "scale_spec_basis": spec_scale,          # 不做单位换算，对应 method.md 事实 H
        "scale_after_conversion": converted_scale,
        "mw_table_available": bool(molecular_weights),
    }
    logger.info("规模（事实 H 口径，不换算）：%s", spec_scale)
    logger.info("规模（单位换算后）        ：%s", converted_scale)
    if not molecular_weights:
        logger.warning(
            "无 RDKit ⇒ 无法计算分子量 ⇒ %d 条 ug.mL-1 记录被丢弃，"
            "靶点规模会低于事实 H 的 70/25。这是口径差异而非数据错误；"
            "在装有 RDKit 的服务器上重跑即可恢复。",
            unit_report.n_dropped_no_mw,
        )
    report["target_selection"] = selection_report.to_dict()

    # 事实 D / 事实 E 的重叠诊断（只报告，不参与 R5 合并 —— 见 build_ortholog_groups 的说明）
    from sparc.data.npass import compound_overlap_report  # noqa: PLC0415

    report["compound_overlap_diagnostics"] = {
        "same_enzyme_cross_species": compound_overlap_report(
            target_meta, per_target_compounds, min_compounds=20,
            min_overlap_frac=0.15, same_enzyme_only=True)[:20],
        "cross_enzyme_high_overlap": [
            r for r in compound_overlap_report(
                target_meta, per_target_compounds, min_compounds=50, min_overlap_frac=0.60)
            if not r["same_enzyme_name"]
        ][:20],
    }
    logger.info("R1–R8：Tier-1 %d / Tier-2 %d / Tier-X %d",
                len(selection_report.tier1), len(selection_report.tier2), len(selection_report.tier_x))

    # 写 targets.yaml（§15 的配置产物）
    targets_yaml = paths.config_dir / "targets.yaml"
    _write_targets_yaml(targets_yaml, targets, selection_report)
    logger.info("靶点清单已写入 %s", targets_yaml)

    # ---------- 2. NP 黑名单 ----------
    logger.info("== 步骤 2：构建 COCONUT ∪ LOTUS 黑名单 ==")
    blacklist = NaturalProductBlacklist.build(
        paths.file("coconut_lite", "coconut_dir"),
        paths.file("lotus_gz", "lotus_dir"),
    )
    report["blacklist"] = blacklist.summary()

    # 复核事实 B（查询池自检索面）
    selected = set(selection_report.tier1) | set(selection_report.tier2)
    query_keys = set().union(*[per_target_compounds.get(t, set()) for t in selected]) if selected else set()
    chembl_by_key = {
        info["inchikey"]: info["chembl_id"]
        for info in general_info.values() if info.get("inchikey") and info.get("chembl_id")
    }
    report["fact_b_audit"] = _audit_fact_b(query_keys, blacklist, chembl_by_key)
    logger.info("事实 B 复核：%s", json.dumps(report["fact_b_audit"], ensure_ascii=False))

    if args.dry_run:
        path = outputs / f"{args.run_name}_dryrun_report.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        logger.info("dry-run 完成，报告：%s", path)
        logger.info("完整流程需要 RDKit 与已解包的 ChEMBL 37（§2.3-T1/T2），请在 CUDA 服务器上运行")
        return 0

    # ---------- 3–8：完整流程（需要 RDKit 与 ChEMBL） ----------
    logger.info("== 步骤 3–8：标准化 / 记忆库 / NP-purge / 断言 / 划分 / Table 0 ==")
    from run_s0_full import run_full_pipeline  # noqa: PLC0415

    report.update(run_full_pipeline(
        config=config, logger=logger, loader=loader, targets=targets,
        selection_report=selection_report, structures=structures,
        blacklist=blacklist, per_target_compounds=per_target_compounds,
        outputs=outputs, run_name=args.run_name,
    ))

    guard.complete(STAGE, config.freeze_manifest(),
                   artifacts={"outputs": str(outputs)},
                   metrics={"n_tier1": len(selection_report.tier1), "n_tier2": len(selection_report.tier2)})
    logger.info("Stage 0 完成。")
    return 0


def _build_mw_table(structures: Dict[str, Dict[str, str]], logger) -> Dict[str, float]:
    """由 SMILES 计算分子量表 ``{inchikey: MW}``（``ug.mL-1`` 换算必需）。

    Args:
        structures: NPASS 结构表 ``{np_id: {inchikey, smiles}}``。
        logger: 日志器。

    Returns:
        ``{inchikey: 分子量}``；无 RDKit 时返回空字典（调用方会据此告警）。
    """
    from sparc.chem.rdkit_backend import RDKIT_AVAILABLE  # noqa: PLC0415

    if not RDKIT_AVAILABLE:
        logger.warning("RDKit 不可用，跳过分子量表构建（见 §5.2 步骤 3 的计数纪律）")
        return {}

    from sparc.chem.fingerprints import FingerprintCalculator  # noqa: PLC0415

    calculator = FingerprintCalculator()
    weights: Dict[str, float] = {}
    for record in structures.values():
        key = record["inchikey"]
        if key in weights:
            continue
        value = calculator.molecular_weight(record["smiles"])
        if value:
            weights[key] = value
    logger.info("分子量表构建完成：%d 个分子", len(weights))
    return weights


def _spec_scale_audit(target_meta: Dict[str, Dict[str, str]],
                      raw_compounds: Dict[str, Set[str]]) -> Dict[str, int]:
    """复核 method.md §1.2 事实 H 的四个数（不做单位换算的口径）。"""
    from sparc.data.npass import ALLOWED_TARGET_TYPES  # noqa: PLC0415

    ge50 = [t for t, keys in raw_compounds.items()
            if target_meta.get(t, {}).get("target_type") in ALLOWED_TARGET_TYPES and len(keys) >= 50]
    ge100 = [t for t in ge50 if len(raw_compounds[t]) >= 100]
    unique = set().union(*[raw_compounds[t] for t in ge50]) if ge50 else set()
    return {
        "n_targets_ge50": len(ge50), "n_targets_ge100": len(ge100),
        "n_target_compound_pairs": sum(len(raw_compounds[t]) for t in ge50),
        "n_unique_natural_products": len(unique),
        "method_md_fact_h": {"n_targets_ge50": 70, "n_targets_ge100": 25,
                             "n_target_compound_pairs": 7314, "n_unique_natural_products": 4075},
    }


def _converted_scale_audit(target_meta: Dict[str, Dict[str, str]],
                           compounds: Dict[str, Set[str]]) -> Dict[str, int]:
    """单位换算 + high-spread 过滤之后的实际规模。"""
    from sparc.data.npass import ALLOWED_TARGET_TYPES  # noqa: PLC0415

    ge50 = [t for t, keys in compounds.items()
            if target_meta.get(t, {}).get("target_type") in ALLOWED_TARGET_TYPES and len(keys) >= 50]
    ge100 = [t for t in ge50 if len(compounds[t]) >= 100]
    return {
        "n_targets_ge50": len(ge50), "n_targets_ge100": len(ge100),
        "n_target_compound_pairs": sum(len(compounds[t]) for t in ge50),
    }


def _audit_fact_b(query_keys: Set[str], blacklist: NaturalProductBlacklist,
                  chembl_by_key: Dict[str, str]) -> Dict[str, object]:
    """复核 method.md 事实 B 的五项计数。"""
    n = len(query_keys)

    def pct(count: int) -> float:
        """占比（百分数，保留一位小数）。"""
        return round(100.0 * count / n, 1) if n else 0.0

    in_coconut = sum(1 for k in query_keys if k in blacklist.coconut_keys)
    in_lotus = sum(1 for k in query_keys if k in blacklist.lotus_keys)
    in_union = sum(1 for k in query_keys if k in blacklist.full_keys)
    with_chembl = sum(1 for k in query_keys if chembl_by_key.get(k))
    glycoside = sum(1 for k in query_keys if blacklist.is_glycoside(k))
    return {
        "n_unique_np_in_query_pool": n,
        "in_coconut_or_lotus": {"n": in_union, "pct": pct(in_union)},
        "in_coconut": {"n": in_coconut, "pct": pct(in_coconut)},
        "in_lotus": {"n": in_lotus, "pct": pct(in_lotus)},
        "with_chembl_id": {"n": with_chembl, "pct": pct(with_chembl)},
        "flagged_glycoside": {"n": glycoside, "pct": pct(glycoside)},
        "method_md_reference": {"n_unique": 4075, "in_union_pct": 96.1, "in_coconut_pct": 92.5,
                                "in_lotus_pct": 75.7, "with_chembl_pct": 79.0, "glycoside_pct": 10.7},
    }


def _write_targets_yaml(path: Path, targets, selection_report) -> None:
    """把 R1–R8 的结果写成 ``configs/targets.yaml`` (§15)。"""
    import yaml  # noqa: PLC0415

    # 只写"曾经有资格"的靶点：通过 R1–R3 的，加上被 R4/R6 淘汰的 Tier-X 反例。
    # NPASS 共 8,764 个 target_id，其中绝大多数是细胞系或化合物数个位数的条目，
    # 全部写进配置只会让这份文件无法阅读。完整的淘汰计数在 summary 里。
    keep = {
        tid: record for tid, record in targets.items()
        if record.tier in ("tier1", "tier2")
        or (record.tier == "tier_x" and record.n_unique_compounds >= 50)
    }
    payload = {
        "spec_version": "1.0.0",
        "generated_by": "scripts/run_s0_data.py",
        "note": ("由 Stage 0 生成。R4–R8 的淘汰数为实测值，method.md §3.1 明确规定不预填猜测值。"
                 "targets 段只含通过 R1–R3 的靶点与 Tier-X 反例（≥50 化合物）；"
                 "全部 8,764 个 target_id 的淘汰计数见 summary.eliminated_by_rule。"),
        "summary": selection_report.to_dict(),
        "n_targets_in_npass": len(targets),
        "n_targets_written": len(keep),
        "targets": {tid: record.to_dict() for tid, record in sorted(keep.items())},
    }
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
