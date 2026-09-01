"""指纹与相似度 (§9.1, §12.4)。

三类指纹对应 §9.1 的三个召回源：
* ECFP4(2048)  —— LSH 源，同时是 Tanimoto 门控特征的来源；
* Model A embedding —— HNSW 源（在 :mod:`sparc.retrieval.index` 中处理）；
* 药效团指纹    —— ANN 源。

另外提供 Table 0（化学域偏移量化）需要的分子描述符。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from sparc.chem.rdkit_backend import require_rdkit


def tanimoto(a: np.ndarray, b: np.ndarray) -> float:
    """两个二值指纹的 Tanimoto 相似度。

    Args:
        a: 形状 ``(n_bits,)`` 的 0/1 数组。
        b: 同上。

    Returns:
        ``|a∩b| / |a∪b|``；并集为 0 时返回 0.0。
    """
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    inter = float(np.count_nonzero(a_bool & b_bool))
    union = float(np.count_nonzero(a_bool | b_bool))
    return inter / union if union > 0 else 0.0


def tanimoto_matrix(query: np.ndarray, memory: np.ndarray) -> np.ndarray:
    """批量 Tanimoto：``(n_q, n_bits) x (n_m, n_bits) -> (n_q, n_m)``。

    用 uint8 矩阵乘法实现，避免逐对 Python 循环 —— 候选池 K₀=256、
    查询数千级时这是热点路径。

    Args:
        query: 查询指纹矩阵（0/1）。
        memory: 记忆库指纹矩阵（0/1）。

    Returns:
        相似度矩阵，形状 ``(n_q, n_m)``，dtype float32。
    """
    q = np.ascontiguousarray(query, dtype=np.float32)
    m = np.ascontiguousarray(memory, dtype=np.float32)
    inter = q @ m.T
    q_pop = q.sum(axis=1, keepdims=True)
    m_pop = m.sum(axis=1, keepdims=True).T
    union = q_pop + m_pop - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        sim = np.where(union > 0, inter / union, 0.0)
    return sim.astype(np.float32)


@dataclass(frozen=True)
class MolecularDescriptors:
    """Table 0（化学域偏移量化，§12.4）所需的描述符。"""

    heavy_atom_count: int
    fraction_csp3: float
    n_stereocenters: int
    n_rings: int
    n_aromatic_rings: int
    ring_complexity: float          # 环系复杂度 = 环原子数 / 重原子数
    mol_weight: float
    logp: float
    tpsa: float
    n_hbd: int
    n_hba: int
    n_rotatable: int

    def to_dict(self) -> Dict[str, float]:
        """转成字典（Table 0 直接按列聚合）。"""
        return {
            "heavy_atom_count": float(self.heavy_atom_count),
            "fraction_csp3": self.fraction_csp3,
            "n_stereocenters": float(self.n_stereocenters),
            "n_rings": float(self.n_rings),
            "n_aromatic_rings": float(self.n_aromatic_rings),
            "ring_complexity": self.ring_complexity,
            "mol_weight": self.mol_weight,
            "logp": self.logp,
            "tpsa": self.tpsa,
            "n_hbd": float(self.n_hbd),
            "n_hba": float(self.n_hba),
            "n_rotatable": float(self.n_rotatable),
        }


class FingerprintCalculator:
    """ECFP4 / 药效团指纹与描述符计算器。"""

    def __init__(self, radius: int = 2, n_bits: int = 2048) -> None:
        """
        Args:
            radius: Morgan 半径；``2`` 即 ECFP4（冻结超参）。
            n_bits: 折叠位数，默认 2048（冻结超参）。
        """
        self.radius = radius
        self.n_bits = n_bits
        self._chem = require_rdkit()
        from rdkit.Chem import rdFingerprintGenerator  # noqa: PLC0415

        self._morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)

    # ------------------------------------------------------------------
    def ecfp4(self, smiles: str) -> Optional[np.ndarray]:
        """计算 ECFP4 位向量。

        Args:
            smiles: 标准化后的 SMILES。

        Returns:
            ``(n_bits,)`` uint8 数组；解析失败返回 ``None``。
        """
        mol = self._chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = self._morgan_gen.GetFingerprintAsNumPy(mol)
        return fp.astype(np.uint8)

    def ecfp4_batch(self, smiles_list: Sequence[str]) -> tuple[np.ndarray, List[int]]:
        """批量计算 ECFP4。

        Returns:
            ``(fp_matrix, ok_indices)``：``fp_matrix`` 只含解析成功的行，
            ``ok_indices`` 给出它们在输入中的下标。
        """
        rows: List[np.ndarray] = []
        ok: List[int] = []
        for i, smi in enumerate(smiles_list):
            fp = self.ecfp4(smi)
            if fp is not None:
                rows.append(fp)
                ok.append(i)
        matrix = np.vstack(rows) if rows else np.zeros((0, self.n_bits), dtype=np.uint8)
        return matrix, ok

    # ------------------------------------------------------------------
    def pharmacophore(self, smiles: str, n_bits: int = 2048) -> Optional[np.ndarray]:
        """2D 药效团指纹（Gobbi 定义，无 3D 构象）。

        §1.1 声明全流程 no-3D，因此这里用 2D 拓扑距离的药效团对，
        而不是 3D pharmacophore。

        Args:
            smiles: 标准化后的 SMILES。
            n_bits: 折叠位数。

        Returns:
            ``(n_bits,)`` uint8 数组；失败返回 ``None``。
        """
        from rdkit.Chem.Pharm2D import Generate, Gobbi_Pharm2D  # noqa: PLC0415

        mol = self._chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            sparse_fp = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)
        except Exception:  # noqa: BLE001
            return None
        dense = np.zeros(n_bits, dtype=np.uint8)
        for bit in sparse_fp.GetOnBits():
            dense[bit % n_bits] = 1
        return dense

    # ------------------------------------------------------------------
    def descriptors(self, smiles: str) -> Optional[MolecularDescriptors]:
        """计算 Table 0 描述符。"""
        from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors  # noqa: PLC0415

        mol = self._chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        ring_info = mol.GetRingInfo()
        n_ring_atoms = sum(1 for atom in mol.GetAtoms() if atom.IsInRing())
        n_heavy = mol.GetNumHeavyAtoms()
        try:
            n_stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
        except Exception:  # noqa: BLE001 - 未赋值立体中心会抛错
            n_stereo = 0
        return MolecularDescriptors(
            heavy_atom_count=n_heavy,
            fraction_csp3=float(rdMolDescriptors.CalcFractionCSP3(mol)),
            n_stereocenters=int(n_stereo),
            n_rings=int(ring_info.NumRings()),
            n_aromatic_rings=int(rdMolDescriptors.CalcNumAromaticRings(mol)),
            ring_complexity=(n_ring_atoms / n_heavy) if n_heavy else 0.0,
            mol_weight=float(Descriptors.MolWt(mol)),
            logp=float(Crippen.MolLogP(mol)),
            tpsa=float(rdMolDescriptors.CalcTPSA(mol)),
            n_hbd=int(Lipinski.NumHDonors(mol)),
            n_hba=int(Lipinski.NumHAcceptors(mol)),
            n_rotatable=int(Lipinski.NumRotatableBonds(mol)),
        )

    def molecular_weight(self, smiles: str) -> Optional[float]:
        """分子量 —— §5.2 步骤 3 的 ``ug.mL-1`` 单位换算需要它。

        Returns:
            分子量；解析失败返回 ``None``（调用方必须计数后丢弃，
            NPASS 中 ``ug.mL-1`` 是第二大单位，不能沉默丢弃）。
        """
        from rdkit.Chem import Descriptors  # noqa: PLC0415

        mol = self._chem.MolFromSmiles(smiles)
        return float(Descriptors.MolWt(mol)) if mol is not None else None
