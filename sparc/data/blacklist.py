"""天然产物黑名单：COCONUT ∪ LOTUS (§5.3)。

事实 B：70 靶点查询池共 4,075 个唯一天然产物，其中 **79.0% 自带 ChEMBL ID**。
不做 NP-purge 的话，"药物→天然产物跨域检索"实际是"分子检索它自己"，
Top-1 邻居很可能就是查询分子本身，H1/H2/H3 全部失效。

黑名单两个粒度：
* ``BLACKLIST_FULL``     —— 完整 InChIKey，精确匹配；
* ``BLACKLIST_SKELETON`` —— 前 14 位骨架块。

COCONUT 738,827 + LOTUS 227,319 唯一 InChIKey，并集约 764,721。
骨架块集合装内存约 764k × 14 字符，Python set 下大致 60–80 MB，可接受。
"""

from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

from sparc.common.logging_utils import get_logger

_LOGGER = get_logger(__name__)


@dataclass
class NaturalProductBlacklist:
    """COCONUT ∪ LOTUS 的 InChIKey 黑名单。"""

    full_keys: Set[str] = field(default_factory=set)
    skeleton_keys: Set[str] = field(default_factory=set)
    coconut_keys: Set[str] = field(default_factory=set)
    lotus_keys: Set[str] = field(default_factory=set)
    glycoside_flags: Dict[str, bool] = field(default_factory=dict)
    np_likeness: Dict[str, float] = field(default_factory=dict)
    pathway: Dict[str, str] = field(default_factory=dict)
    superclass: Dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------
    @classmethod
    def build(
        cls,
        coconut_lite_tsv: Path,
        lotus_csv_gz: Path,
        load_annotations: bool = True,
        field_size_limit: int = 10 ** 7,
    ) -> "NaturalProductBlacklist":
        """从 COCONUT lite TSV 与 LOTUS gz CSV 构建黑名单。

        Args:
            coconut_lite_tsv: ``COCONUT/coconut_npkey_lite.tsv``（派生文件，
                再生脚本见 CLAUDE.md）。
            lotus_csv_gz: ``LOTUS/260413_frozen.csv.gz``。
            load_annotations: 是否同时载入 ``is_glycoside`` / ``np_likeness`` /
                NPClassifier 标签。§6.2 的糖苷通道与 §12.4 的 Table 0 需要它们。
            field_size_limit: csv 字段上限。

        Returns:
            :class:`NaturalProductBlacklist`。
        """
        csv.field_size_limit(field_size_limit)
        obj = cls()
        obj._load_coconut(Path(coconut_lite_tsv), load_annotations)
        obj._load_lotus(Path(lotus_csv_gz))
        obj.full_keys = obj.coconut_keys | obj.lotus_keys
        obj.skeleton_keys = {k[:14] for k in obj.full_keys if len(k) >= 14}
        _LOGGER.info(
            "NP 黑名单构建完成：COCONUT %d + LOTUS %d → 并集 %d 精确键 / %d 骨架块",
            len(obj.coconut_keys), len(obj.lotus_keys), len(obj.full_keys), len(obj.skeleton_keys),
        )
        return obj

    def _load_coconut(self, path: Path, load_annotations: bool) -> None:
        """读 COCONUT lite。"""
        if not path.is_file():
            raise FileNotFoundError(
                f"缺少 {path}。该文件是派生产物，可用 CLAUDE.md 中的脚本从 "
                "coconut_csv_lite-08-2026.zip 再生。"
            )
        with open(path, "r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                key = (row.get("inchikey") or "").strip()
                if len(key) < 14:
                    continue
                self.coconut_keys.add(key)
                if not load_annotations:
                    continue
                is_gly = (row.get("is_glycoside") or "").strip().lower() == "true"
                has_sugar = (row.get("contains_sugar") or "").strip().lower() == "true"
                self.glycoside_flags[key] = is_gly or has_sugar
                raw_npl = (row.get("np_likeness") or "").strip()
                if raw_npl:
                    try:
                        self.np_likeness[key] = float(raw_npl)
                    except ValueError:
                        pass
                if row.get("pathway"):
                    self.pathway[key] = row["pathway"].strip()
                if row.get("superclass"):
                    self.superclass[key] = row["superclass"].strip()

    def _load_lotus(self, path: Path) -> None:
        """读 LOTUS gz CSV（结构–生物来源对，取唯一 InChIKey）。"""
        if not path.is_file():
            raise FileNotFoundError(f"缺少 LOTUS 文件：{path}")
        with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row.get("structure_inchikey") or "").strip()
                if len(key) >= 14:
                    self.lotus_keys.add(key)

    # ------------------------------------------------------------------
    def contains(self, inchikey: str) -> Optional[str]:
        """判断一个 InChIKey 是否命中黑名单。

        Args:
            inchikey: 待检 InChIKey。

        Returns:
            ``"np_exact"`` / ``"np_skeleton"`` / ``None``（未命中）。
            §5.3 要求这两种命中分别计数并在论文中报告。
        """
        if not inchikey:
            return None
        if inchikey in self.full_keys:
            return "np_exact"
        if len(inchikey) >= 14 and inchikey[:14] in self.skeleton_keys:
            return "np_skeleton"
        return None

    def is_natural_product(self, inchikey: str) -> bool:
        """是否被 COCONUT/LOTUS 确认为天然产物（只看精确键）。"""
        return inchikey in self.full_keys

    def is_glycoside(self, inchikey: str) -> bool:
        """COCONUT 的糖苷标记 (§6.2)。未收录的分子返回 ``False``，
        调用方应回退到 :class:`~sparc.chem.sugar.SugarStripper` 的 SMARTS 检测。"""
        return bool(self.glycoside_flags.get(inchikey, False))

    def load_lotus_organisms(self, lotus_csv_gz: Path, rank: str = "genus") -> Dict[str, str]:
        """读 LOTUS 的生物来源，用于协议 X（生物来源划分）与 C-BIO-SOURCE。

        Args:
            lotus_csv_gz: LOTUS gz CSV。
            rank: ``"genus"``（取二名法的属名）或 ``"full"``（完整学名）。

        Returns:
            ``{inchikey: 来源名}``；一个分子有多个来源时取首个（按文件顺序）。
        """
        organisms: Dict[str, str] = {}
        with gzip.open(lotus_csv_gz, "rt", encoding="utf-8", errors="replace", newline="") as handle:
            for row in csv.DictReader(handle):
                key = (row.get("structure_inchikey") or "").strip()
                name = (row.get("organism_name") or "").strip()
                if not key or not name or key in organisms:
                    continue
                organisms[key] = name.split()[0] if rank == "genus" else name
        _LOGGER.info("LOTUS 生物来源：%d 个分子有 %s 级标注", len(organisms), rank)
        return organisms

    def summary(self) -> Dict[str, int]:
        """写进 Stage 0 报告的规模摘要。"""
        return {
            "n_coconut": len(self.coconut_keys),
            "n_lotus": len(self.lotus_keys),
            "n_union_exact": len(self.full_keys),
            "n_union_skeleton": len(self.skeleton_keys),
            "n_glycoside_flagged": sum(1 for v in self.glycoside_flags.values() if v),
        }
