"""数据记录的强类型定义。

整个管线只在这四种记录之间流动，避免"字典里到底有哪些键"的猜谜。
所有记录都带 ``censor_flag`` —— §10.1 规定审查值进 Tobit 似然，
不转点值。**这是本项目相对 NaFM 的一个干净的技术改进**，
因此审查标志不允许在任何中间环节被丢掉。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class CensorFlag(str, Enum):
    """审查标志 (§5.2 步骤 5)。"""

    NONE = "none"       # 精确测定值
    LEFT = "left"       # 实测 < y，真值更小
    RIGHT = "right"     # 实测 > y，真值更大

    @property
    def is_censored(self) -> bool:
        """是否为审查记录。"""
        return self is not CensorFlag.NONE


class Domain(str, Enum):
    """记录来源域。Evidence 编码的第 5 维 ``1[domain=drug]`` 用它。"""

    DRUG = "drug"                       # ChEMBL / BindingDB
    NATURAL_PRODUCT = "natural_product" # NPASS


@dataclass(frozen=True)
class TargetRecord:
    """一个入选靶点 (§3.1)。"""

    target_id: str                  # NPASS target_id，如 NPT204
    target_name: str
    target_type: str
    uniprot_id: str
    organism_tax_id: str
    organism: str
    ortholog_group_id: str          # R5：直系同源分组，整组同侧划分 (§6.3)
    n_unique_compounds: int
    censored_frac: float
    pactivity_std: float
    tier: str                       # "tier1" / "tier2" / "tier_x"
    has_sequence: bool
    exclusion_reasons: tuple[str, ...] = ()

    @property
    def is_selected(self) -> bool:
        """是否通过 R1–R8。"""
        return not self.exclusion_reasons

    def to_dict(self) -> Dict[str, Any]:
        """转成可写 YAML/TSV 的字典。"""
        return {
            "target_id": self.target_id,
            "target_name": self.target_name,
            "target_type": self.target_type,
            "uniprot_id": self.uniprot_id,
            "organism_tax_id": self.organism_tax_id,
            "organism": self.organism,
            "ortholog_group_id": self.ortholog_group_id,
            "n_unique_compounds": self.n_unique_compounds,
            "censored_frac": round(self.censored_frac, 4),
            "pactivity_std": round(self.pactivity_std, 4),
            "tier": self.tier,
            "has_sequence": self.has_sequence,
            "exclusion_reasons": list(self.exclusion_reasons),
        }


@dataclass(frozen=True)
class ActivityRecord:
    """一条聚合前的原始活性记录。"""

    compound_id: str
    target_id: str
    activity_type: str              # 统一为 IC50
    value: float                    # 原始数值
    units: str
    relation: str                   # 原始关系符
    censor_flag: CensorFlag
    assay_id: str = ""
    assay_organism_tax_id: str = ""
    reference_id: str = ""
    reference_year: Optional[int] = None
    domain: Domain = Domain.NATURAL_PRODUCT


@dataclass(frozen=True)
class QueryRecord:
    """一条天然产物查询 (§5.1 的查询集 Q)。"""

    query_id: str                   # f"{target_id}::{inchikey}"
    compound_id: str
    inchikey: str
    smiles: str
    target_id: str
    uniprot_id: str
    organism_tax_id: str
    pactivity: float                # pIC50
    censor_flag: CensorFlag
    # 四把家族钥匙 —— §5.4 的四条断言直接用它们
    skeleton14: str
    deglyco_core_hash: str
    tautomer_family_id: str
    murcko_smiles: str = ""
    scaffold_family_id: str = ""
    ortholog_group_id: str = ""
    source_organism_family: str = ""   # 协议 X（生物来源）划分用
    reference_year: Optional[int] = None
    n_source_records: int = 1
    is_glycoside: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """转成可写 TSV 的字典。"""
        return {
            "query_id": self.query_id,
            "compound_id": self.compound_id,
            "inchikey": self.inchikey,
            "smiles": self.smiles,
            "target_id": self.target_id,
            "uniprot_id": self.uniprot_id,
            "organism_tax_id": self.organism_tax_id,
            "pactivity": self.pactivity,
            "censor_flag": self.censor_flag.value,
            "skeleton14": self.skeleton14,
            "deglyco_core_hash": self.deglyco_core_hash,
            "tautomer_family_id": self.tautomer_family_id,
            "murcko_smiles": self.murcko_smiles,
            "scaffold_family_id": self.scaffold_family_id,
            "ortholog_group_id": self.ortholog_group_id,
            "source_organism_family": self.source_organism_family,
            "reference_year": self.reference_year,
            "n_source_records": self.n_source_records,
            "is_glycoside": self.is_glycoside,
        }


@dataclass(frozen=True)
class MemoryRecord:
    """一条药物记忆库记录 (§5.2 的 Drug Memory M)。

    结构与 :class:`QueryRecord` 对齐是有意的：``C_FORCED_RETRIEVAL``
    诊断分支 (§13.4) 会让天然产物记录也进记忆库，届时两者必须
    走完全相同的编码与检索路径。
    """

    memory_id: str
    inchikey: str
    smiles: str
    target_id: str
    uniprot_id: str
    organism_tax_id: str
    pactivity: float
    censor_flag: CensorFlag
    skeleton14: str
    deglyco_core_hash: str
    tautomer_family_id: str
    murcko_smiles: str = ""
    scaffold_family_id: str = ""
    domain: Domain = Domain.DRUG
    assay_family: int = 0            # 0..31，Evidence 的 assay family embedding
    endpoint: str = "IC50"
    source_db: str = "chembl"
    n_source_records: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转成可写 TSV 的字典。"""
        return {
            "memory_id": self.memory_id,
            "inchikey": self.inchikey,
            "smiles": self.smiles,
            "target_id": self.target_id,
            "uniprot_id": self.uniprot_id,
            "organism_tax_id": self.organism_tax_id,
            "pactivity": self.pactivity,
            "censor_flag": self.censor_flag.value,
            "skeleton14": self.skeleton14,
            "deglyco_core_hash": self.deglyco_core_hash,
            "tautomer_family_id": self.tautomer_family_id,
            "murcko_smiles": self.murcko_smiles,
            "scaffold_family_id": self.scaffold_family_id,
            "domain": self.domain.value,
            "assay_family": self.assay_family,
            "endpoint": self.endpoint,
            "source_db": self.source_db,
            "n_source_records": self.n_source_records,
        }
