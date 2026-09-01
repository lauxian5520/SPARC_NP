"""活性单位统一 (§5.2 步骤 3)。

全部换算为 pIC50 = −log10(IC50 [M])。

**必须计数的两条**：
1. ``ug.mL-1`` 记录需要 MW 换算；MW 缺失者丢弃并计数 ——
   NPASS 中 ``ug.mL-1`` 是第二大单位（211,385 行，仅次于 nM 的 448,509 行），
   静默丢弃会悄悄改变数据集构成；
2. 不可换算的单位（``%``、``cells.uL-1``、``mm``、``s`` 等）同样计数。
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

# 浓度单位 → 摩尔浓度(M) 的乘数
_MOLAR_UNITS: Dict[str, float] = {
    "m": 1.0,
    "mm": 1e-3,          # 注意：NPASS 中 "mm" 同时被用作长度单位，见下方 ambiguity 处理
    "um": 1e-6,
    "µm": 1e-6,
    "nm": 1e-9,
    "pm": 1e-12,
    "fm": 1e-15,
    "mol.l-1": 1.0,
    "mmol.l-1": 1e-3,
    "umol.l-1": 1e-6,
    "nmol.l-1": 1e-9,
}

# 质量浓度单位 → g/L 的乘数（需 MW 才能转摩尔）
_MASS_CONC_UNITS: Dict[str, float] = {
    "g.l-1": 1.0,
    "mg.l-1": 1e-3,
    "mg.ml-1": 1.0,       # 1 mg/mL = 1 g/L
    "ug.ml-1": 1e-3,      # 1 µg/mL = 1 mg/L = 1e-3 g/L
    "ug ml-1": 1e-3,
    "µg.ml-1": 1e-3,
    "ng.ml-1": 1e-6,
    "ug.l-1": 1e-6,
}

# 明确不可换算 —— 出现即丢弃并计数，绝不猜测
_NON_CONVERTIBLE = {
    "%", "n.a.", "", "cells.ul-1", "s", "min", "h", "iu.l-1", "u.l-1",
    "meq.l-1", "g", "pg", "mg.kg-1", "ratio", "-", "fold",
}

# NPASS 中 "mm" 语义歧义（毫摩尔 vs 毫米）。§5 要求不静默猜测，
# 因此默认按不可换算处理并计数；需要时由调用方显式开启。
AMBIGUOUS_UNITS = {"mm"}


@dataclass
class UnitConversionReport:
    """单位换算的计数报告 —— 必须进 Stage 0 报告。"""

    n_input: int = 0
    n_converted: int = 0
    n_dropped_no_mw: int = 0
    n_dropped_bad_unit: int = 0
    n_dropped_nonpositive: int = 0
    n_dropped_ambiguous_unit: int = 0
    dropped_units: Counter = field(default_factory=Counter)

    def to_dict(self) -> Dict[str, object]:
        """转成可写报告的字典。"""
        return {
            "n_input": self.n_input,
            "n_converted": self.n_converted,
            "n_dropped_no_mw": self.n_dropped_no_mw,
            "n_dropped_bad_unit": self.n_dropped_bad_unit,
            "n_dropped_nonpositive": self.n_dropped_nonpositive,
            "n_dropped_ambiguous_unit": self.n_dropped_ambiguous_unit,
            "convert_rate": round(self.n_converted / self.n_input, 4) if self.n_input else 0.0,
            "top_dropped_units": self.dropped_units.most_common(12),
        }


def normalize_unit(units: str) -> str:
    """把单位字符串归一化到小写无空格形式。"""
    return (units or "").strip().lower().replace("μ", "u").replace(" ", "").replace("/", ".")


def to_pactivity(
    value: float,
    units: str,
    molecular_weight: Optional[float] = None,
    report: Optional[UnitConversionReport] = None,
    allow_ambiguous_mm_as_millimolar: bool = False,
) -> Optional[float]:
    """把一条活性记录换算成 pIC50。

    Args:
        value: 原始活性数值。
        units: 原始单位字符串（NPASS/ChEMBL 原样）。
        molecular_weight: 分子量 (g/mol)；质量浓度单位必需。
        report: 计数报告；传入则原地累加。
        allow_ambiguous_mm_as_millimolar: 是否把歧义单位 ``mm`` 当毫摩尔。
            默认 ``False``（丢弃并计数）—— §5 的纪律是不猜测。

    Returns:
        pIC50 值；无法换算时返回 ``None``。
    """
    if report is not None:
        report.n_input += 1

    unit = normalize_unit(units)
    if unit in AMBIGUOUS_UNITS and not allow_ambiguous_mm_as_millimolar:
        if report is not None:
            report.n_dropped_ambiguous_unit += 1
            report.dropped_units[unit] += 1
        return None

    if value is None or not math.isfinite(value) or value <= 0:
        if report is not None:
            report.n_dropped_nonpositive += 1
        return None

    molar: Optional[float] = None
    if unit in _MOLAR_UNITS:
        molar = value * _MOLAR_UNITS[unit]
    elif unit in _MASS_CONC_UNITS:
        if molecular_weight is None or molecular_weight <= 0:
            if report is not None:
                report.n_dropped_no_mw += 1
                report.dropped_units[unit] += 1
            return None
        molar = (value * _MASS_CONC_UNITS[unit]) / molecular_weight    # (g/L) / (g/mol) = mol/L
    else:
        if report is not None:
            report.n_dropped_bad_unit += 1
            report.dropped_units[unit] += 1
        return None

    if molar is None or molar <= 0:
        if report is not None:
            report.n_dropped_nonpositive += 1
        return None

    if report is not None:
        report.n_converted += 1
    return -math.log10(molar)


def aggregate_pactivity(
    values: list[float],
    method: str = "median",
    max_spread: float = 1.5,
) -> Tuple[Optional[float], str]:
    """聚合同 (化合物, 靶点) 的多条记录 (§5.2 步骤 4)。

    Args:
        values: 同一 (化合物, 靶点) 的多个 pIC50。
        method: ``"median"``（冻结默认）或 ``"mean"``。
        max_spread: 极差上限；超过则整条丢弃（``qc_high_spread_drop``）。

    Returns:
        ``(聚合值 或 None, 状态)``；状态取值 ``"ok"`` / ``"high_spread"`` / ``"empty"``。
    """
    if not values:
        return None, "empty"
    if len(values) > 1 and (max(values) - min(values)) > max_spread:
        return None, "high_spread"
    if method == "mean":
        return sum(values) / len(values), "ok"
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    median = ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])
    return median, "ok"
