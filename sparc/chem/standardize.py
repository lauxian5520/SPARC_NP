"""分子标准化 (§5.2 步骤 2)。

管线：去盐 → 中性化 → 规范互变体 → InChIKey。

三条与本项目泄漏设计直接相关的产出：
* ``inchikey``          —— 精确匹配键（NP-purge 与断言 1）；
* ``skeleton14``        —— InChIKey 前 14 位，骨架块匹配（断言 2）；
* ``tautomer_family_id``—— 规范互变体的 InChIKey 前 14 位（断言 4）。

之所以把互变体家族单独算一遍：``TautomerEnumerator().Canonicalize()``
会把不同互变体写法折叠到同一个规范式，而 InChIKey 的第一块本身
**不保证**折叠所有互变体（尤其是酮–烯醇）。二者是两个键，不是一个。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Iterable, List, Optional

from sparc.chem.rdkit_backend import require_rdkit
from sparc.common.logging_utils import get_logger

_LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class StandardizedMolecule:
    """一个标准化后的分子及其全部去重/泄漏键。"""

    input_smiles: str
    canonical_smiles: Optional[str]
    inchikey: Optional[str]
    skeleton14: Optional[str]
    tautomer_smiles: Optional[str]
    tautomer_family_id: Optional[str]
    murcko_smiles: Optional[str]
    n_heavy_atoms: int
    ok: bool
    error: str = ""

    def to_dict(self) -> Dict[str, object]:
        """转成可写 TSV 的字典。"""
        return {
            "input_smiles": self.input_smiles,
            "canonical_smiles": self.canonical_smiles,
            "inchikey": self.inchikey,
            "skeleton14": self.skeleton14,
            "tautomer_family_id": self.tautomer_family_id,
            "murcko_smiles": self.murcko_smiles,
            "n_heavy_atoms": self.n_heavy_atoms,
            "ok": self.ok,
            "error": self.error,
        }


class MoleculeStandardizer:
    """RDKit 标准化器。

    对 drug 与 natural product 使用**完全相同**的标准化路径 ——
    §7.1 的接口契约要求，也是跨域比较有意义的前提。
    """

    def __init__(self, max_tautomer_atoms: int = 120, enumerate_tautomers: bool = True) -> None:
        """
        Args:
            max_tautomer_atoms: 超过此重原子数则跳过互变体规范化（RDKit
                的枚举在大分子上会爆炸；天然产物里确实有这种分子）。
            enumerate_tautomers: 是否做互变体规范化。关闭时
                ``tautomer_family_id`` 退化为 ``skeleton14``，
                此时 §5.4 的第 4 条断言强度下降，必须在报告中说明。
        """
        self.max_tautomer_atoms = max_tautomer_atoms
        self.enumerate_tautomers = enumerate_tautomers
        self._chem = require_rdkit()
        from rdkit.Chem import SaltRemover  # noqa: PLC0415
        from rdkit.Chem.MolStandardize import rdMolStandardize  # noqa: PLC0415

        self._salt_remover = SaltRemover.SaltRemover()
        self._uncharger = rdMolStandardize.Uncharger()
        self._largest_fragment = rdMolStandardize.LargestFragmentChooser()
        self._tautomer_enumerator = rdMolStandardize.TautomerEnumerator()

    # ------------------------------------------------------------------
    def standardize(self, smiles: str) -> StandardizedMolecule:
        """标准化单个 SMILES。

        Args:
            smiles: 输入 SMILES。

        Returns:
            :class:`StandardizedMolecule`；解析失败时 ``ok=False`` 且带原因，
            调用方必须计数而不是静默丢弃（§5 步骤 3 的同一条纪律）。
        """
        chem = self._chem
        if not smiles or not smiles.strip():
            return self._failed(smiles, "empty_smiles")
        try:
            mol = chem.MolFromSmiles(smiles)
            if mol is None:
                return self._failed(smiles, "parse_failed")

            mol = self._largest_fragment.choose(self._salt_remover.StripMol(mol, dontRemoveEverything=True))
            if mol is None or mol.GetNumAtoms() == 0:
                return self._failed(smiles, "empty_after_desalt")
            mol = self._uncharger.uncharge(mol)
            chem.SanitizeMol(mol)

            canonical_smiles = chem.MolToSmiles(mol, canonical=True)
            inchikey = chem.MolToInchiKey(mol) or None
            skeleton14 = inchikey[:14] if inchikey else None

            tautomer_smiles, tautomer_family_id = self._canonical_tautomer(mol, skeleton14)
            murcko_smiles = self._murcko(mol)

            return StandardizedMolecule(
                input_smiles=smiles,
                canonical_smiles=canonical_smiles,
                inchikey=inchikey,
                skeleton14=skeleton14,
                tautomer_smiles=tautomer_smiles,
                tautomer_family_id=tautomer_family_id,
                murcko_smiles=murcko_smiles,
                n_heavy_atoms=mol.GetNumHeavyAtoms(),
                ok=inchikey is not None,
                error="" if inchikey else "inchikey_failed",
            )
        except Exception as exc:  # noqa: BLE001 - RDKit 会抛各种类型
            return self._failed(smiles, f"exception:{type(exc).__name__}")

    def standardize_many(self, smiles_list: Iterable[str]) -> List[StandardizedMolecule]:
        """批量标准化。"""
        return [self.standardize(s) for s in smiles_list]

    # ------------------------------------------------------------------
    def _canonical_tautomer(self, mol: object, fallback: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        """计算规范互变体及其家族 ID。"""
        chem = self._chem
        if not self.enumerate_tautomers or mol.GetNumHeavyAtoms() > self.max_tautomer_atoms:
            return None, fallback
        try:
            canon = self._tautomer_enumerator.Canonicalize(mol)
            key = chem.MolToInchiKey(canon)
            return chem.MolToSmiles(canon, canonical=True), (key[:14] if key else fallback)
        except Exception:  # noqa: BLE001
            return None, fallback

    def _murcko(self, mol: object) -> Optional[str]:
        """通用 Murcko 骨架（generic framework，元素与键级抹平）。"""
        from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: PLC0415

        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold is None or scaffold.GetNumAtoms() == 0:
                return None
            generic = MurckoScaffold.MakeScaffoldGeneric(scaffold)
            return self._chem.MolToSmiles(generic, canonical=True)
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _failed(smiles: str, reason: str) -> StandardizedMolecule:
        """构造一个失败记录。"""
        return StandardizedMolecule(
            input_smiles=smiles, canonical_smiles=None, inchikey=None, skeleton14=None,
            tautomer_smiles=None, tautomer_family_id=None, murcko_smiles=None,
            n_heavy_atoms=0, ok=False, error=reason,
        )


@lru_cache(maxsize=1)
def default_standardizer() -> MoleculeStandardizer:
    """进程内共享的默认标准化器（RDKit 对象构造较重）。"""
    return MoleculeStandardizer()
