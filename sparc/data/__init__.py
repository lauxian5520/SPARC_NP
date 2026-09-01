"""数据构建管线 (§5) 与防泄漏协议 (§6)。"""

from sparc.data.schema import (
    ActivityRecord,
    CensorFlag,
    MemoryRecord,
    QueryRecord,
    TargetRecord,
)
from sparc.data.npass import NPASSLoader, TargetSelectionReport, select_targets
from sparc.data.blacklist import NaturalProductBlacklist
from sparc.data.purge import LeakageAssertionError, NPPurger, PurgeReport, assert_zero_overlap
from sparc.data.splits import SplitAssignment, SplitBuilder
from sparc.data.units import UnitConversionReport, to_pactivity

__all__ = [
    "ActivityRecord",
    "CensorFlag",
    "MemoryRecord",
    "QueryRecord",
    "TargetRecord",
    "NPASSLoader",
    "TargetSelectionReport",
    "select_targets",
    "NaturalProductBlacklist",
    "LeakageAssertionError",
    "NPPurger",
    "PurgeReport",
    "assert_zero_overlap",
    "SplitAssignment",
    "SplitBuilder",
    "UnitConversionReport",
    "to_pactivity",
]
