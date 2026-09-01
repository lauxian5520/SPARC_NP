"""ChEMBL 37 抽取 —— 药物记忆库主来源 (§5.2 步骤 1)。

**环境事实**：本机没有 ``sqlite3`` CLI，但 Python stdlib 的
``sqlite3`` 模块可用（3.46.1）。因此 ChEMBL 解包后可以直接查询。

解包（§2.3-T1，展开约 25 GB）::

    tar -xzf chembl_37_sqlite.tar.gz -C data/interim/
    # 解包后建索引：
    #   CREATE INDEX ix_act_assay ON activities(assay_id);
    #   CREATE INDEX ix_act_mol   ON activities(molregno);
    #   CREATE INDEX ix_cs_ik     ON compound_structures(standard_inchi_key);

见 :meth:`ChEMBLExtractor.create_indexes`。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set

from sparc.common.logging_utils import get_logger

_LOGGER = get_logger(__name__)

# 只取 IC50，并保留关系符（§5.2 步骤 5：审查值不转点值）
_ACTIVITY_SQL = """
SELECT
    cs.standard_inchi_key                AS inchikey,
    cs.canonical_smiles                  AS smiles,
    cseq.accession                       AS uniprot_id,
    td.organism                          AS target_organism,
    td.tax_id                            AS organism_tax_id,
    act.standard_type                    AS standard_type,
    act.standard_relation                AS standard_relation,
    act.standard_value                   AS standard_value,
    act.standard_units                   AS standard_units,
    act.assay_id                         AS assay_id,
    a.assay_type                         AS assay_type,
    a.confidence_score                   AS confidence_score,
    docs.year                            AS year
FROM activities            AS act
JOIN assays                AS a     ON a.assay_id       = act.assay_id
JOIN target_dictionary     AS td    ON td.tid           = a.tid
JOIN target_components     AS tc    ON tc.tid           = td.tid
JOIN component_sequences   AS cseq  ON cseq.component_id = tc.component_id
JOIN compound_structures   AS cs    ON cs.molregno      = act.molregno
LEFT JOIN docs                       ON docs.doc_id     = act.doc_id
WHERE cseq.accession IN ({placeholders})
  AND act.standard_type = ?
  AND act.standard_value IS NOT NULL
  AND act.standard_value > 0
  AND cs.standard_inchi_key IS NOT NULL
  AND (act.data_validity_comment IS NULL OR act.data_validity_comment = '')
  AND act.potential_duplicate = 0
  AND a.confidence_score >= ?
"""

_INDEX_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS ix_sparc_act_assay ON activities(assay_id)",
    "CREATE INDEX IF NOT EXISTS ix_sparc_act_mol   ON activities(molregno)",
    "CREATE INDEX IF NOT EXISTS ix_sparc_act_type  ON activities(standard_type)",
    "CREATE INDEX IF NOT EXISTS ix_sparc_cs_ik     ON compound_structures(standard_inchi_key)",
    "CREATE INDEX IF NOT EXISTS ix_sparc_cseq_acc  ON component_sequences(accession)",
    "CREATE INDEX IF NOT EXISTS ix_sparc_tc_tid    ON target_components(tid)",
)


@dataclass(frozen=True)
class RawActivity:
    """一条来自 ChEMBL/BindingDB 的原始活性（尚未标准化、未换算单位）。"""

    inchikey: str
    smiles: str
    uniprot_id: str
    organism_tax_id: str
    activity_type: str
    relation: str
    value: float
    units: str
    assay_id: str
    year: Optional[int]
    source_db: str
    confidence_score: Optional[int] = None


class ChEMBLExtractor:
    """ChEMBL 37 SQLite 抽取器（stdlib sqlite3，只读模式）。"""

    def __init__(self, db_path: Path, read_only: bool = True) -> None:
        """
        Args:
            db_path: 解包后的 ``chembl_37.db``。
            read_only: 以 URI 只读模式打开。建索引时需传 ``False``。

        Raises:
            FileNotFoundError: 数据库文件不存在（多半是还没解包，见 §2.3-T1）。
        """
        self.db_path = Path(db_path)
        if not self.db_path.is_file():
            raise FileNotFoundError(
                f"找不到 ChEMBL SQLite：{self.db_path}\n"
                "ChEMBL 37 尚未解包（§2.3-T1）。请先执行：\n"
                f"  tar -xzf .../chembl_37_sqlite.tar.gz -C {self.db_path.parent}\n"
                "解包约需 25 GB 磁盘。"
            )
        uri = f"file:{self.db_path}?mode=ro" if read_only else str(self.db_path)
        self.conn = sqlite3.connect(uri, uri=read_only)
        self.conn.row_factory = sqlite3.Row

    def close(self) -> None:
        """关闭连接。"""
        self.conn.close()

    def __enter__(self) -> "ChEMBLExtractor":
        """进入上下文。"""
        return self

    def __exit__(self, *exc: object) -> None:
        """退出上下文并关闭连接。"""
        self.close()

    # ------------------------------------------------------------------
    def create_indexes(self) -> None:
        """建 §2.3-T1 要求的索引（需以可写模式打开）。"""
        cursor = self.conn.cursor()
        for statement in _INDEX_STATEMENTS:
            _LOGGER.info("建索引：%s", statement)
            cursor.execute(statement)
        self.conn.commit()

    def table_counts(self) -> Dict[str, int]:
        """四张核心表的行数，用于确认解包完整性。"""
        counts: Dict[str, int] = {}
        for table in ("activities", "assays", "compound_structures", "target_dictionary"):
            try:
                counts[table] = self.conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            except sqlite3.Error as exc:
                counts[table] = -1
                _LOGGER.error("读取表 %s 失败：%s", table, exc)
        return counts

    # ------------------------------------------------------------------
    def extract(
        self,
        uniprot_accessions: Sequence[str],
        activity_type: str = "IC50",
        min_confidence: int = 8,
        chunk_size: int = 400,
    ) -> Iterator[RawActivity]:
        """抽取指定 UniProt accession 上的活性三元组。

        Args:
            uniprot_accessions: §3.1 入选靶点的 UniProt accession。
            activity_type: 冻结为 ``"IC50"``（R2）。
            min_confidence: ChEMBL ``confidence_score`` 下限。8 = "直接指定
                单一蛋白"，低于此值的 assay 靶点归属不可靠，会把噪声记录
                当成"同靶点证据"喂给检索。
            chunk_size: SQL ``IN`` 子句的分块大小（SQLite 变量上限 999）。

        Yields:
            :class:`RawActivity`。**注意 tax_id 一并带出** —— 事实 F 要求
            跨物种靶点按同一 tax_id 硬过滤 (R8)。
        """
        accessions = sorted(set(a for a in uniprot_accessions if a))
        if not accessions:
            _LOGGER.warning("UniProt accession 列表为空，ChEMBL 抽取跳过")
            return

        n_yielded = 0
        for start in range(0, len(accessions), chunk_size):
            chunk = accessions[start:start + chunk_size]
            sql = _ACTIVITY_SQL.format(placeholders=",".join("?" * len(chunk)))
            params: List[Any] = [*chunk, activity_type, min_confidence]
            for row in self.conn.execute(sql, params):
                yield RawActivity(
                    inchikey=row["inchikey"] or "",
                    smiles=row["smiles"] or "",
                    uniprot_id=row["uniprot_id"] or "",
                    organism_tax_id=str(row["organism_tax_id"] or ""),
                    activity_type=row["standard_type"] or activity_type,
                    relation=(row["standard_relation"] or "=").strip(),
                    value=float(row["standard_value"]),
                    units=(row["standard_units"] or "").strip(),
                    assay_id=str(row["assay_id"]),
                    year=int(row["year"]) if row["year"] else None,
                    source_db="chembl",
                    confidence_score=row["confidence_score"],
                )
                n_yielded += 1
        _LOGGER.info("ChEMBL 抽取完成：%d 条原始活性（%d 个 accession）", n_yielded, len(accessions))


def load_chembl_uniprot_mapping(mapping_file: Path) -> Dict[str, List[str]]:
    """读 ``chembl_uniprot_mapping.txt``。

    Args:
        mapping_file: ``ChEMBL37/chembl_uniprot_mapping.txt``。

    Returns:
        ``{uniprot_accession: [chembl_target_id]}``。
    """
    mapping: Dict[str, List[str]] = {}
    with open(mapping_file, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            mapping.setdefault(parts[0].strip(), []).append(parts[1].strip())
    _LOGGER.info("ChEMBL↔UniProt 映射：%d 个 accession", len(mapping))
    return mapping
