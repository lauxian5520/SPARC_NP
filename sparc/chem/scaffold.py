"""骨架家族构建 (§6.1，事实 C)。

家族定义（v1.0 修订，**四把钥匙缺一不可**）：
    节点 = 分子；以下任一成立即连边
      1. 通用 Murcko 骨架的 ECFP4 Tanimoto ≥ 0.50
      2. ``inchikey_skeleton`` 相同（InChIKey 前 14 位）
      3. ``deglyco_core_hash`` 相同（脱糖母核）      ← v1.0 新增
      4. ``tautomer_family_id`` 相同                 ← v1.0 新增
    连通分量 = 骨架家族；划分以家族为原子单位。

条款 2–4 是 v1.0 新增。仅用条款 1（旧方案）会漏掉 15.8% 的糖苷通道 ——
芦丁与槲皮素在只看 Murcko 时属于不同家族，于是苷元可以合法地留在
它自己糖苷的记忆库里，还是 Top-1。

实现上：条款 2–4 是等价类（哈希桶），条款 1 是阈值图。
先用哈希桶做并查集（O(n)），再只在**桶间代表元**上算 Tanimoto，
把 O(n²) 的相似度计算压到骨架去重后的规模。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from sparc.chem.fingerprints import tanimoto_matrix
from sparc.common.logging_utils import get_logger

_LOGGER = get_logger(__name__)


@dataclass
class ScaffoldKeys:
    """一个分子的四把家族钥匙。"""

    molecule_id: str
    inchikey: str
    skeleton14: Optional[str] = None
    deglyco_core_hash: Optional[str] = None
    tautomer_family_id: Optional[str] = None
    murcko_smiles: Optional[str] = None

    def hash_keys(self) -> List[Tuple[str, str]]:
        """返回参与哈希桶合并的 ``(键类型, 键值)`` 列表（条款 2–4）。"""
        keys: List[Tuple[str, str]] = []
        if self.skeleton14:
            keys.append(("skeleton14", self.skeleton14))
        if self.deglyco_core_hash:
            keys.append(("deglyco", self.deglyco_core_hash))
        if self.tautomer_family_id:
            keys.append(("tautomer", self.tautomer_family_id))
        return keys


class _UnionFind:
    """按秩合并 + 路径压缩的并查集。"""

    def __init__(self, items: Iterable[str]) -> None:
        """用给定元素初始化，每个元素自成一集。"""
        self.parent: Dict[str, str] = {item: item for item in items}
        self.rank: Dict[str, int] = {item: 0 for item in self.parent}

    def find(self, x: str) -> str:
        """找根（路径压缩）。"""
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        """合并两个集合。"""
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


@dataclass
class ScaffoldFamilyResult:
    """骨架家族构建结果。"""

    family_of: Dict[str, str]                    # molecule_id -> family_id
    members: Dict[str, List[str]]                # family_id -> [molecule_id]
    edge_counts: Dict[str, int] = field(default_factory=dict)   # 各条款贡献的边数

    @property
    def n_families(self) -> int:
        """家族数。"""
        return len(self.members)

    @property
    def n_molecules(self) -> int:
        """分子数。"""
        return len(self.family_of)

    def size_histogram(self) -> Dict[int, int]:
        """家族大小分布 ``{大小: 家族数}``。"""
        hist: Dict[int, int] = defaultdict(int)
        for ids in self.members.values():
            hist[len(ids)] += 1
        return dict(sorted(hist.items()))

    def summary(self) -> Dict[str, object]:
        """写进 Stage 0 报告的摘要。"""
        sizes = [len(v) for v in self.members.values()]
        return {
            "n_molecules": self.n_molecules,
            "n_families": self.n_families,
            "largest_family": max(sizes) if sizes else 0,
            "median_family_size": float(np.median(sizes)) if sizes else 0.0,
            "singleton_families": sum(1 for s in sizes if s == 1),
            "edges_by_rule": dict(self.edge_counts),
        }


class ScaffoldFamilyBuilder:
    """把 §6.1 的四条规则实现成一次并查集。"""

    def __init__(
        self,
        murcko_tanimoto_threshold: float = 0.50,
        use_inchikey_skeleton: bool = True,
        use_deglyco_core: bool = True,
        use_tautomer_family: bool = True,
        murcko_block_size: int = 2048,
    ) -> None:
        """
        Args:
            murcko_tanimoto_threshold: 条款 1 阈值（冻结为 0.50）。
            use_inchikey_skeleton: 启用条款 2。
            use_deglyco_core: 启用条款 3（糖苷通道，**不建议关闭**）。
            use_tautomer_family: 启用条款 4。
            murcko_block_size: Tanimoto 分块矩阵的块大小（控制内存）。
        """
        self.threshold = murcko_tanimoto_threshold
        self.use_skeleton = use_inchikey_skeleton
        self.use_deglyco = use_deglyco_core
        self.use_tautomer = use_tautomer_family
        self.block_size = murcko_block_size
        if not (use_deglyco_core and use_tautomer_family and use_inchikey_skeleton):
            _LOGGER.warning(
                "骨架家族的条款 2–4 被部分关闭 —— §6.1 明确规定这会漏掉糖苷/互变体泄漏通道，"
                "仅允许在敏感性分析中这样做，且必须在报告中声明。"
            )

    # ------------------------------------------------------------------
    def build(
        self,
        keys: Sequence[ScaffoldKeys],
        murcko_fingerprints: Optional[Dict[str, np.ndarray]] = None,
    ) -> ScaffoldFamilyResult:
        """构建骨架家族。

        Args:
            keys: 每个分子的四把钥匙。
            murcko_fingerprints: ``{murcko_smiles: ECFP4 位向量}``；
                为 ``None`` 时跳过条款 1（只用哈希桶），此时必须在
                报告中标注，因为家族会偏碎。

        Returns:
            :class:`ScaffoldFamilyResult`。
        """
        mol_ids = [k.molecule_id for k in keys]
        if len(set(mol_ids)) != len(mol_ids):
            raise ValueError("molecule_id 必须唯一 —— 重复 ID 会让并查集把无关分子并到一起")
        uf = _UnionFind(mol_ids)
        edge_counts: Dict[str, int] = defaultdict(int)

        # --- 条款 2–4：哈希桶等价类 ---
        buckets: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        enabled = {
            "skeleton14": self.use_skeleton,
            "deglyco": self.use_deglyco,
            "tautomer": self.use_tautomer,
        }
        for key in keys:
            for kind, value in key.hash_keys():
                if enabled.get(kind, False):
                    buckets[(kind, value)].append(key.molecule_id)
        for (kind, _), group in buckets.items():
            for other in group[1:]:
                if uf.find(group[0]) != uf.find(other):
                    edge_counts[kind] += 1
                uf.union(group[0], other)

        # --- 条款 1：Murcko 骨架 Tanimoto ≥ 阈值 ---
        if murcko_fingerprints:
            edge_counts["murcko_tanimoto"] += self._link_by_murcko(keys, murcko_fingerprints, uf)
        else:
            _LOGGER.warning("未提供 Murcko 指纹，跳过条款 1；家族会偏碎，须在 Stage 0 报告中声明")

        members: Dict[str, List[str]] = defaultdict(list)
        family_of: Dict[str, str] = {}
        for mol_id in mol_ids:
            root = uf.find(mol_id)
            family_of[mol_id] = f"FAM_{root}"
            members[f"FAM_{root}"].append(mol_id)

        result = ScaffoldFamilyResult(family_of=family_of, members=dict(members), edge_counts=dict(edge_counts))
        _LOGGER.info("骨架家族构建完成：%s", result.summary())
        return result

    # ------------------------------------------------------------------
    def _link_by_murcko(
        self,
        keys: Sequence[ScaffoldKeys],
        murcko_fingerprints: Dict[str, np.ndarray],
        uf: _UnionFind,
    ) -> int:
        """按 Murcko 骨架 Tanimoto 连边。

        先把分子按 ``murcko_smiles`` 去重成骨架代表元，只在代表元之间
        算相似度 —— 分子数 n 万级时，骨架数通常只有千级，这一步把
        O(n²) 压成 O(m²)。

        Returns:
            新增的连边数（同骨架合并 + 跨骨架相似合并）。
        """
        scaffold_to_mols: Dict[str, List[str]] = defaultdict(list)
        for key in keys:
            if key.murcko_smiles:
                scaffold_to_mols[key.murcko_smiles].append(key.molecule_id)
        scaffolds = [s for s in scaffold_to_mols if s in murcko_fingerprints]
        if not scaffolds:
            return 0

        n_edges = 0
        # 同一 Murcko 骨架直接合并（Tanimoto = 1.0 ≥ 阈值）
        for scaffold in scaffolds:
            group = scaffold_to_mols[scaffold]
            for other in group[1:]:
                if uf.find(group[0]) != uf.find(other):
                    n_edges += 1
                uf.union(group[0], other)

        matrix = np.vstack([murcko_fingerprints[s] for s in scaffolds]).astype(np.uint8)
        n = len(scaffolds)
        for start in range(0, n, self.block_size):
            stop = min(start + self.block_size, n)
            sims = tanimoto_matrix(matrix[start:stop], matrix)
            rows, cols = np.nonzero(sims >= self.threshold)
            for r, c in zip(rows.tolist(), cols.tolist()):
                i = start + r
                if i >= c:                      # 只处理上三角，避免重复
                    continue
                a = scaffold_to_mols[scaffolds[i]][0]
                b = scaffold_to_mols[scaffolds[c]][0]
                if uf.find(a) != uf.find(b):
                    n_edges += 1
                uf.union(a, b)
        return n_edges
