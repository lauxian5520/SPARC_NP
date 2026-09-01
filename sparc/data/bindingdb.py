"""BindingDB 抽取 —— 药物记忆库补充来源 (§5.2 步骤 1, §2.3-T2)。

BindingDB 全量 TSV 有 ~200 列且体积很大（解包后数 GB），因此这里
**流式读取 zip 内的 TSV**，只保留需要的列，从不整表进内存。

与 ChEMBL 按 InChIKey 去重合并（§2.3-T2）由
:func:`sparc.data.build_dataset.merge_memory_sources` 负责。
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

from sparc.common.logging_utils import get_logger
from sparc.data.chembl import RawActivity

_LOGGER = get_logger(__name__)

# BindingDB 列名（202607 版）。IC50 列名带单位后缀，是历史遗留格式。
_COL_SMILES = "Ligand SMILES"
_COL_INCHIKEY = "Ligand InChI Key"
_COL_IC50 = "IC50 (nM)"
# ⚠️ BindingDB 的靶点链列名**带链号后缀**：``... of Target Chain 1`` / ``Chain 2`` …
# 全量 TSV（202607，640 列）里 SwissProt / TrEMBL 各有 50 条链的列。
# 不带后缀的列名在表头里**出现 0 次** —— 按它去 ``row.get()`` 永远拿到 None，
# 于是每一行的 accession 都是空串、全部被跳过，最终"扫描完成但命中 0 条"。
# 这个失败是静默的：不报错、不警告，只是安静地返回空记忆库。
_UNIPROT_CHAIN_RE = re.compile(
    r"UniProt \((SwissProt|TrEMBL)\) Primary ID of Target Chain (\d+)"
)


def _uniprot_columns(fieldnames: Sequence[str]) -> Tuple[List[str], List[str]]:
    """从真实表头里解析出按链号排序的 SwissProt / TrEMBL 主 ID 列。

    Args:
        fieldnames: ``csv.DictReader.fieldnames``。

    Returns:
        ``(swissprot 列名列表, trembl 列名列表)``，均按链号升序。

    Raises:
        ValueError: 一条链都没找到 —— 说明 BindingDB 换了表头格式，
            必须先核对再继续，不允许静默产出空记忆库。
    """
    swiss: List[Tuple[int, str]] = []
    trembl: List[Tuple[int, str]] = []
    for name in fieldnames or ():
        match = _UNIPROT_CHAIN_RE.fullmatch(name.strip())
        if not match:
            continue
        (swiss if match.group(1) == "SwissProt" else trembl).append((int(match.group(2)), name))
    if not swiss and not trembl:
        raise ValueError(
            "BindingDB 表头里找不到任何 'UniProt (SwissProt|TrEMBL) Primary ID of Target Chain N' 列。"
            f"实际表头前 12 列：{list(fieldnames or ())[:12]}。"
            "BindingDB 可能改了列名格式 —— 先核对再继续，不要让它静默产出空记忆库。"
        )
    return [n for _, n in sorted(swiss)], [n for _, n in sorted(trembl)]
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
                swiss_cols, trembl_cols = _uniprot_columns(reader.fieldnames)
                _LOGGER.info("BindingDB 靶点链列：SwissProt %d 条 / TrEMBL %d 条",
                             len(swiss_cols), len(trembl_cols))
                for row in reader:
                    n_seen += 1
                    if max_rows and n_seen > max_rows:
                        break
                    accession = ""
                    for column in swiss_cols:          # 先 SwissProt，按链号顺序取第一个非空
                        value = (row.get(column) or "").strip()
                        if value:
                            accession = value
                            break
                    if not accession:
                        for column in trembl_cols:
                            value = (row.get(column) or "").strip()
                            if value:
                                accession = value
                                break
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
        if n_yield == 0 and n_seen > 0:
            _LOGGER.error(
                "BindingDB 扫描了 %d 行却命中 0 条。BindingDB 覆盖 ~9,500 个靶点，"
                "对常见人源靶点（AChE/COX-2/CYP3A4/BACE1 等）命中 0 条几乎不可能 —— "
                "优先怀疑列名或 accession 口径，不要当成'本轮确实没有数据'。", n_seen,
            )
        else:
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
