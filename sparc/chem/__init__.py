"""化学工具层：标准化、指纹、脱糖、骨架家族。

**惰性导入纪律**：RDKit 只在真正需要时导入（见 :func:`require_rdkit`）。
本机（树莓派 aarch64）没有 RDKit，数据盘点与规范工作仍需能 import 本包。
"""

from sparc.chem.rdkit_backend import RDKIT_AVAILABLE, require_rdkit
from sparc.chem.standardize import MoleculeStandardizer, StandardizedMolecule
from sparc.chem.fingerprints import FingerprintCalculator, tanimoto, tanimoto_matrix
from sparc.chem.sugar import SugarStripper
from sparc.chem.scaffold import ScaffoldFamilyBuilder, ScaffoldKeys

__all__ = [
    "RDKIT_AVAILABLE",
    "require_rdkit",
    "MoleculeStandardizer",
    "StandardizedMolecule",
    "FingerprintCalculator",
    "tanimoto",
    "tanimoto_matrix",
    "SugarStripper",
    "ScaffoldFamilyBuilder",
    "ScaffoldKeys",
]
