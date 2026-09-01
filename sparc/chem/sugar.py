"""糖苷–苷元泄漏通道 (§6.2, 事实 C)。

**为什么这个文件必须存在**：COCONUT 2026-08 中 15.8% 的分子是糖苷。
Murcko 骨架保留糖环，于是芦丁与槲皮素落入*不同*骨架家族 ——
苷元可以合法地留在其糖苷查询的记忆库里，而且往往就是相似度第一名。
这不是边缘案例，是 1/6 的分子。

``deglyco_core_hash`` 的定义 (§6.2)::

    if COCONUT.is_glycoside[inchikey] or RDKit_sugar_detected(mol):
        core = removeCircularAndLinearSugars(mol)
        return canonical_inchikey(core)[:14]
    return inchikey[:14]

RDKit 没有 CDK 的 ``SugarRemovalUtility``，因此这里按
Schaub, Zielesny, Steinbeck & Sorokina (*J. Cheminform.* 12:67, 2020)
描述的判据自行实现 SMARTS 版本：环状糖（呋喃糖/吡喃糖）与
线性糖（开链多羟基碳链）分别检测并迭代剥离末端糖单元。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

from sparc.chem.rdkit_backend import require_rdkit
from sparc.common.logging_utils import get_logger

_LOGGER = get_logger(__name__)

# 环状糖：5/6 元环，环内恰好一个氧，其余为 sp3 碳
_CIRCULAR_SUGAR_SMARTS: Tuple[str, ...] = (
    # 吡喃糖骨架（6 元，1 个环氧）
    "[C;R1]1[C;R1][C;R1][C;R1][C;R1][O;R1]1",
    # 呋喃糖骨架（5 元，1 个环氧）
    "[C;R1]1[C;R1][C;R1][C;R1][O;R1]1",
)

# 线性糖：连续多羟基碳链（至少 4 个连续带 O 的 sp3 碳）
_LINEAR_SUGAR_SMARTS: Tuple[str, ...] = (
    "[CX4;!R]([OX2H1,OX2H0])[CX4;!R]([OX2H1,OX2H0])[CX4;!R]([OX2H1,OX2H0])[CX4;!R]([OX2H1,OX2H0])",
    "[CX4;!R]([OX2H1])[CX4;!R]([OX2H1])[CX4;!R]([OX2H1])[CX4;!R]=[OX1]",
)

# 糖苷键：环内碳 — 桥氧 — 环外原子
_GLYCOSIDIC_BOND_SMARTS = "[C;R][O;X2;!R][C,O,N,S]"


@dataclass(frozen=True)
class DeglycosylationResult:
    """一次脱糖的结果。"""

    input_smiles: str
    core_smiles: Optional[str]
    core_inchikey: Optional[str]
    deglyco_core_hash: Optional[str]     # 母核 InChIKey 前 14 位
    n_sugars_removed: int
    detected_circular: bool
    detected_linear: bool
    used_coconut_flag: bool
    ok: bool
    error: str = ""


class SugarStripper:
    """环状 + 线性糖剥离器。

    优先信任 COCONUT 自带的 ``is_glycoside`` / ``contains_sugar`` 标记
    （覆盖 738,827 个分子，§2.2），不在 COCONUT 中的分子回退到
    SMARTS 检测。
    """

    def __init__(
        self,
        coconut_glycoside_flags: Optional[Dict[str, bool]] = None,
        min_core_heavy_atoms: int = 5,
        max_iterations: int = 8,
        preserve_ring_count: int = 1,
    ) -> None:
        """
        Args:
            coconut_glycoside_flags: ``{inchikey: is_glycoside}``，来自
                ``coconut_npkey_lite.tsv``。可为 ``None``（纯 SMARTS 模式）。
            min_core_heavy_atoms: 剥离后母核的重原子数下限；低于此值说明
                整个分子就是糖，此时**不剥离**（否则 hash 会塌到空分子上，
                把所有单糖误并成一个骨架家族）。
            max_iterations: 迭代剥离的最大轮数（多糖苷如芦丁有 2 个糖单元）。
            preserve_ring_count: 母核必须保留的最少环数；剥离后环数低于
                此值则回退，避免把糖环本身当成母核骨架剥没。
        """
        self.coconut_flags = coconut_glycoside_flags or {}
        self.min_core_heavy_atoms = min_core_heavy_atoms
        self.max_iterations = max_iterations
        self.preserve_ring_count = preserve_ring_count
        self._chem = require_rdkit()
        self._circular_patterns = [self._chem.MolFromSmarts(s) for s in _CIRCULAR_SUGAR_SMARTS]
        self._linear_patterns = [self._chem.MolFromSmarts(s) for s in _LINEAR_SUGAR_SMARTS]
        self._glycosidic_pattern = self._chem.MolFromSmarts(_GLYCOSIDIC_BOND_SMARTS)

    # ------------------------------------------------------------------
    def deglyco_core_hash(self, smiles: str, inchikey: Optional[str] = None) -> DeglycosylationResult:
        """计算 §6.2 的 ``deglyco_core_hash``。

        Args:
            smiles: 标准化后的 SMILES。
            inchikey: 该分子的 InChIKey；用于查 COCONUT 糖苷标记，
                并在"不是糖苷"时直接返回其前 14 位。

        Returns:
            :class:`DeglycosylationResult`。非糖苷分子的
            ``deglyco_core_hash`` 就是 ``inchikey[:14]``。
        """
        chem = self._chem
        mol = chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            return DeglycosylationResult(
                smiles, None, None, (inchikey[:14] if inchikey else None),
                0, False, False, False, ok=False, error="parse_failed",
            )

        coconut_flag = bool(self.coconut_flags.get(inchikey, False)) if inchikey else False
        has_circular = self._has_circular_sugar(mol)
        has_linear = self._has_linear_sugar(mol)

        if not (coconut_flag or has_circular or has_linear):
            # 非糖苷：母核 hash 退化为骨架块，与 §6.2 的伪代码一致
            return DeglycosylationResult(
                smiles, chem.MolToSmiles(mol), inchikey, (inchikey[:14] if inchikey else None),
                0, False, False, coconut_flag, ok=True,
            )

        core, n_removed = self._strip(mol)
        core_smiles = chem.MolToSmiles(core, canonical=True)
        core_key = chem.MolToInchiKey(core) or None
        return DeglycosylationResult(
            input_smiles=smiles,
            core_smiles=core_smiles,
            core_inchikey=core_key,
            deglyco_core_hash=(core_key[:14] if core_key else (inchikey[:14] if inchikey else None)),
            n_sugars_removed=n_removed,
            detected_circular=has_circular,
            detected_linear=has_linear,
            used_coconut_flag=coconut_flag,
            ok=core_key is not None,
            error="" if core_key else "core_inchikey_failed",
        )

    def is_glycoside(self, smiles: str, inchikey: Optional[str] = None) -> bool:
        """判断是否为糖苷（COCONUT 标记优先，否则 SMARTS 检测）。"""
        if inchikey and inchikey in self.coconut_flags:
            return bool(self.coconut_flags[inchikey])
        mol = self._chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            return False
        return self._has_circular_sugar(mol) or self._has_linear_sugar(mol)

    # ------------------------------------------------------------------
    # 内部：检测
    # ------------------------------------------------------------------
    def _has_circular_sugar(self, mol: object) -> bool:
        """是否含环状糖（需同时满足环骨架匹配与羟基密度判据）。"""
        for pattern in self._circular_patterns:
            for match in mol.GetSubstructMatches(pattern):
                if self._ring_is_sugar_like(mol, match):
                    return True
        return False

    def _has_linear_sugar(self, mol: object) -> bool:
        """是否含线性糖。"""
        return any(mol.HasSubstructMatch(p) for p in self._linear_patterns)

    def _ring_is_sugar_like(self, mol: object, ring_atoms: Sequence[int]) -> bool:
        """糖环判据：环上碳至少一半带外接氧（Schaub et al. 的羟基密度判据）。"""
        carbons = [i for i in ring_atoms if mol.GetAtomWithIdx(i).GetSymbol() == "C"]
        if not carbons:
            return False
        with_oxygen = 0
        for idx in carbons:
            atom = mol.GetAtomWithIdx(idx)
            if any(nb.GetSymbol() == "O" and not nb.IsInRing() for nb in atom.GetNeighbors()):
                with_oxygen += 1
        return with_oxygen >= max(2, len(carbons) // 2)

    # ------------------------------------------------------------------
    # 内部：剥离
    # ------------------------------------------------------------------
    def _strip(self, mol: object) -> Tuple[object, int]:
        """迭代剥离末端糖单元，返回 ``(母核, 剥离数)``。

        每轮：找出所有糖环原子 → 断开糖苷键 → 取重原子数最大的碎片。
        当碎片过小或环数不足时回退到上一轮结果（宁可不剥，也不能把
        母核剥没 —— 那会制造一个假的骨架家族合并）。
        """
        chem = self._chem
        current = mol
        removed = 0
        for _ in range(self.max_iterations):
            sugar_atoms = self._collect_sugar_atoms(current)
            if not sugar_atoms:
                break
            candidate = self._remove_atoms(current, sugar_atoms)
            if candidate is None:
                break
            if candidate.GetNumHeavyAtoms() < self.min_core_heavy_atoms:
                break
            if candidate.GetRingInfo().NumRings() < self.preserve_ring_count and current.GetRingInfo().NumRings() >= self.preserve_ring_count:
                break
            current = candidate
            removed += 1
        try:
            chem.SanitizeMol(current)
        except Exception:  # noqa: BLE001
            _LOGGER.debug("脱糖后 sanitize 失败，回退到原分子")
            return mol, 0
        return current, removed

    def _collect_sugar_atoms(self, mol: object) -> Set[int]:
        """收集所有属于糖单元的原子下标（环状 + 其外接羟基氧）。"""
        sugar: Set[int] = set()
        ring_info = mol.GetRingInfo()
        for pattern in self._circular_patterns:
            for match in mol.GetSubstructMatches(pattern):
                if not self._ring_is_sugar_like(mol, match):
                    continue
                # 只剥"末端"糖：糖环与母核之间恰好一条糖苷键
                if self._is_terminal_sugar(mol, match, ring_info):
                    sugar.update(match)
                    for idx in match:
                        for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
                            if nb.GetSymbol() == "O" and not nb.IsInRing() and nb.GetDegree() == 1:
                                sugar.add(nb.GetIdx())
                            elif nb.GetSymbol() == "C" and not nb.IsInRing() and self._is_exocyclic_ch2oh(mol, nb):
                                sugar.add(nb.GetIdx())
                                for sub in nb.GetNeighbors():
                                    if sub.GetSymbol() == "O" and sub.GetDegree() == 1:
                                        sugar.add(sub.GetIdx())
        return sugar

    @staticmethod
    def _is_exocyclic_ch2oh(mol: object, atom: object) -> bool:
        """是否为糖环外接的 CH2OH 侧链（属于糖单元）。"""
        if atom.GetTotalNumHs() < 2:
            return False
        return any(nb.GetSymbol() == "O" and nb.GetDegree() == 1 for nb in atom.GetNeighbors())

    def _is_terminal_sugar(self, mol: object, ring_atoms: Sequence[int], ring_info: object) -> bool:
        """糖环是否为末端糖：它与分子其余部分只通过糖苷键相连。

        "外连"只算**真正通向母核**的键。糖环自身的装饰基团不算：

        * 度为 1 的重原子取代基 —— 羟基 ``-OH``、**脱氧糖的甲基 ``-CH3``**、
          以及修饰糖上的卤素。它们是死端，不可能连向母核；
        * 环外 ``-CH2OH`` 侧链（C6），由 :meth:`_is_exocyclic_ch2oh` 判定。

        .. note::
           ``-CH3`` 这一条是必须的，不是锦上添花。**6-脱氧糖**（鼠李糖、
           岩藻糖、鸡纳糖）的 C6 是甲基而不是 CH2OH，漏掉它就会把甲基
           误计成外连键，于是这类糖**永远剥不掉**。§6.2 的强制夹具
           芦丁 = 槲皮素 + 芸香糖（鼠李糖-(1→6)-葡萄糖）正好踩中：
           鼠李糖剥不掉 ⇒ 葡萄糖永远等不到变成末端 ⇒ ``n_removed=0`` ⇒
           芦丁与槲皮素落在不同骨架家族 ⇒ 苷元可以合法地待在它自己
           糖苷的记忆库里当最近邻。这正是 §6.2 要堵的那条泄漏通道。
        """
        ring_set = set(ring_atoms)
        external_links = 0
        for idx in ring_atoms:
            for nb in mol.GetAtomWithIdx(idx).GetNeighbors():
                if nb.GetIdx() in ring_set:
                    continue
                # 度为 1 的重原子取代基（-OH / -CH3 / 卤素）是死端，不是外连
                if nb.GetDegree() == 1:
                    continue
                # 环外 CH2OH 侧链属于糖单元
                if nb.GetSymbol() == "C" and self._is_exocyclic_ch2oh(mol, nb):
                    continue
                external_links += 1
        return external_links <= 1

    def _remove_atoms(self, mol: object, atom_ids: Set[int]) -> Optional[object]:
        """删除给定原子并返回最大碎片。"""
        chem = self._chem
        editable = chem.RWMol(mol)
        for idx in sorted(atom_ids, reverse=True):
            editable.RemoveAtom(idx)
        try:
            result = editable.GetMol()
            chem.SanitizeMol(result)
        except Exception:  # noqa: BLE001
            return None
        frags = chem.GetMolFrags(result, asMols=True, sanitizeFrags=True)
        if not frags:
            return None
        return max(frags, key=lambda m: m.GetNumHeavyAtoms())


def load_coconut_glycoside_flags(coconut_lite_tsv: str) -> Dict[str, bool]:
    """从 ``coconut_npkey_lite.tsv`` 读糖苷标记。

    Args:
        coconut_lite_tsv: 派生文件路径（列见 CLAUDE.md 的再生脚本）。

    Returns:
        ``{inchikey: is_glycoside or contains_sugar}``。两个标记取或 ——
        §6.2 的伪代码写的是 ``is_glycoside``，但 ``contains_sugar``
        覆盖了一部分未被 NPClassifier 判为糖苷的含糖分子，
        在泄漏检测上宁可宽一点。
    """
    import csv  # noqa: PLC0415

    flags: Dict[str, bool] = {}
    with open(coconut_lite_tsv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            key = (row.get("inchikey") or "").strip()
            if not key:
                continue
            is_gly = (row.get("is_glycoside") or "").strip().lower() == "true"
            has_sugar = (row.get("contains_sugar") or "").strip().lower() == "true"
            flags[key] = is_gly or has_sugar
    return flags
