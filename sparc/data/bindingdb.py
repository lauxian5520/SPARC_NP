"""BindingDB 抽取 —— 药物记忆库补充来源 (§5.2 步骤 1, §2.3-T2)。

BindingDB 全量 TSV 有 ~200 列且体积很大（解包后数 GB），因此这里
**流式读取 zip 内的 TSV**，只保留需要的列，从不整表进内存。

与 ChEMBL 按 InChIKey 去重合并（§2.3-T2）由
:func:`sparc.data.build_dataset.merge_memory_sources` 负责。
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set

from sparc.common.logging_utils import get_logger
from sparc.data.chembl import RawActivity

_LOGGER = get_logger(__name__)

# BindingDB 列名（202607 版）。IC50 列名带单位后缀，是历史遗留格式。
_COL_SMILES = "Ligand SMILES"
_COL_INCHIKEY = "Ligand InChI Key"
_COL_IC50 = "IC50 (nM)"
_COL_UNIPROT_PRIMARY = "UniProt (SwissProt) Primary ID of Target Chain"
_COL_UNIPROT_ALT = "UniProt (TrEMBL) Primary ID of Target Chain"
_COL_ORGANISM = "Target Source Organism According to Curator or DataSource"
_COL_YEAR = "Article DOI"      # BindingDB 无独立年份列；PMID/DOI 由调用方另行解析
_COL_PMID = "PMID"


class BindingDBExtractor:
    """BindingDB 全量 TSV（zip 内）的流式抽取器。"""

    def __init__(self, zip_path: Path, inner_name: Optional[str] = None) -> None:
        """
        Args:
            zip_path: ``BindingDB_All_202607_tsv.zip``。
            inner_name: zip 内 TSV 的文件名；``None`` 时自动取第一个 ``.tsv``。
        """
        self.zip_path = Path(zip_path)
        if not self.zip_path.is_file():
            raise FileNotFoundError(f"找不到 BindingDB 压缩包：{self.zip_path}（§2.3-T2 未完成）")
        self.inner_name = inner_name

    def _resolve_inner(self, archive: zipfile.ZipFile) -> str:
        """确定 zip 内的 TSV 文件名。"""
        if self.inner_name:
            return self.inner_name
        names = [n for n in archive.namelist() if n.lower().endswith((".tsv", ".txt"))]
        if not names:
            raise ValueError(f"{self.zip_path} 内没有 TSV 文件")
        return names[0]

    # ------------------------------------------------------------------
    def extract(
        self,
        uniprot_accessions: Sequence[str],
        organism_tax_ids: Optional[Dict[str, str]] = None,
        max_rows: Optional[int] = None,
    ) -> Iterator[RawActivity]:
        """流式抽取 IC50 记录。

        Args:
            uniprot_accessions: 只保留这些 accession 的记录。
            organism_tax_ids: ``{accession: tax_id}``，用于回填 tax_id ——
                BindingDB 只给物种名，而 R8 要求按 tax_id 硬过滤。
                缺失时 tax_id 为空串，调用方须按"物种不可确认"处理。
            max_rows: 调试用的行数上限。

        Yields:
            :class:`RawActivity`（``source_db="bindingdb"``）。
        """
        wanted: Set[str] = {a for a in uniprot_accessions if a}
        tax_map = organism_tax_ids or {}
        csv.field_size_limit(10 ** 7)

        n_seen = 0
        n_yield = 0
        with zipfile.ZipFile(self.zip_path) as archive:
            inner = self._resolve_inner(archive)
            _LOGGER.info("流式读取 BindingDB：%s → %s", self.zip_path.name, inner)
            with archive.open(inner) as raw:
                stream = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
                reader = csv.DictReader(stream, delimiter="\t")
                for row in reader:
                    n_seen += 1
                    if max_rows and n_seen > max_rows:
                        break
                    accession = (row.get(_COL_UNIPROT_PRIMARY) or row.get(_COL_UNIPROT_ALT) or "").strip()
                    if accession not in wanted:
                        continue
                    raw_value = (row.get(_COL_IC50) or "").strip()
                    if not raw_value:
                        continue
                    relation, value = _parse_bindingdb_value(raw_value)
                    if value is None:
                        continue
                    inchikey = (row.get(_COL_INCHIKEY) or "").strip()
                    smiles = (row.get(_COL_SMILES) or "").strip()
                    if not inchikey or not smiles:
                        continue
                    yield RawActivity(
                        inchikey=inchikey,
                        smiles=smiles,
                        uniprot_id=accession,
                        organism_tax_id=tax_map.get(accession, ""),
                        activity_type="IC50",
                        relation=relation,
                        value=value,
                        units="nM",
                        assay_id=(row.get(_COL_PMID) or "").strip() or "bindingdb",
                        year=None,
                        source_db="bindingdb",
                    )
                    n_yield += 1
        _LOGGER.info("BindingDB 抽取完成：扫描 %d 行 → 命中 %d 条", n_seen, n_yield)


def _parse_bindingdb_value(raw: str) -> tuple[str, Optional[float]]:
    """解析 BindingDB 的活性值，保留关系符。

    BindingDB 把关系符写在数值里，如 ``">10000"``。§5.2 步骤 5 规定
    ``>``/``<`` 不转点值，因此必须在这里就把它拆出来。

    Args:
        raw: 原始字段值。

    Returns:
        ``(关系符, 数值 或 None)``。
    """
    text = raw.strip()
    relation = "="
    for symbol in (">=", "<=", ">", "<"):
        if text.startswith(symbol):
            relation = symbol
            text = text[len(symbol):].strip()
            break
    try:
        return relation, float(text)
    except ValueError:
        return relation, None


def load_bindingdb_uniprot(mapping_file: Path) -> Dict[str, str]:
    """读 ``BindingDB_UniProt.txt``，返回 ``{accession: 名称}``。"""
    mapping: Dict[str, str] = {}
    with open(mapping_file, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0].strip():
                mapping[parts[0].strip()] = parts[1].strip()
    _LOGGER.info("BindingDB↔UniProt 映射：%d 个 accession", len(mapping))
    return mapping
