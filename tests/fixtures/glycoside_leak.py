"""糖苷–苷元泄漏夹具 (§6.2 强制回归夹具)。

**槲皮素（``REFJWTPEDVJJIY``）与芦丁（``IKGXIBQEEMLURG``）必须落在
同一骨架家族；此测试失败则 CI 红灯。**

为什么是这两个分子：芦丁 = 槲皮素 + 芸香糖（鼠李糖-葡萄糖二糖）。
Murcko 骨架保留糖环，于是二者的 Murcko 完全不同、Tanimoto 远低于 0.50，
在旧方案（只用 Murcko + 精确 InChIKey）下属于**不同**家族 ——
于是槲皮素可以合法地留在芦丁查询的记忆库里，而且往往是 Top-1。

COCONUT 2026-08 中 15.8% 的分子是糖苷。这不是边缘案例，是 1/6 的分子。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

# 标准 SMILES（来自 PubChem，已 canonical 化）
QUERCETIN_SMILES = "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12"
RUTIN_SMILES = (
    "C[C@@H]1O[C@@H](OC[C@H]2O[C@@H](Oc3c(-c4ccc(O)c(O)c4)oc4cc(O)cc(O)c4c3=O)"
    "[C@H](O)[C@@H](O)[C@@H]2O)[C@H](O)[C@H](O)[C@H]1O"
)

QUERCETIN_SKELETON14 = "REFJWTPEDVJJIY"
RUTIN_SKELETON14 = "IKGXIBQEEMLURG"

# 阴性对照：与二者都无关，必须落在**不同**家族
NEGATIVE_CONTROL_SMILES = "CC(=O)Oc1ccccc1C(=O)O"          # 阿司匹林
NEGATIVE_CONTROL_SKELETON14 = "BSYNRYMUTXBXSQ"


@dataclass(frozen=True)
class GlycosideCase:
    """一个糖苷–苷元配对用例。"""

    name: str
    aglycone_smiles: str
    glycoside_smiles: str
    aglycone_skeleton14: str
    glycoside_skeleton14: str
    note: str = ""


# 主夹具 + 两个补充用例（覆盖 O-糖苷与 C-糖苷两种连接方式）
GLYCOSIDE_CASES: List[GlycosideCase] = [
    GlycosideCase(
        name="quercetin_rutin",
        aglycone_smiles=QUERCETIN_SMILES,
        glycoside_smiles=RUTIN_SMILES,
        aglycone_skeleton14=QUERCETIN_SKELETON14,
        glycoside_skeleton14=RUTIN_SKELETON14,
        note="§6.2 指定的强制夹具：芦丁 = 槲皮素 + 芸香糖（O-二糖苷）",
    ),
    GlycosideCase(
        name="apigenin_vitexin",
        aglycone_smiles="O=c1cc(-c2ccc(O)cc2)oc2cc(O)cc(O)c12",
        glycoside_smiles="O=c1cc(-c2ccc(O)cc2)oc2cc(O)c([C@@H]3O[C@H](CO)[C@@H](O)[C@H](O)[C@H]3O)c(O)c12",
        aglycone_skeleton14="KZNIFHPLKGYRTM",
        glycoside_skeleton14="XJYKYQXMDQEUAO",
        note="C-糖苷（牡荆素）：糖环通过 C–C 键连接，比 O-糖苷更难剥离",
    ),
    GlycosideCase(
        name="genistein_genistin",
        aglycone_smiles="O=c1c(-c2ccc(O)cc2)coc2cc(O)cc(O)c12",
        glycoside_smiles="O=c1c(-c2ccc(O)cc2)coc2cc(O[C@@H]3O[C@H](CO)[C@@H](O)[C@H](O)[C@H]3O)cc(O)c12",
        aglycone_skeleton14="TZBJGXHYKVUXJN",
        glycoside_skeleton14="ZCOLJUOHXJRHDI",
        note="O-单糖苷（染料木苷）",
    ),
]


def scaffold_keys_from_annotations(annotations: Dict[str, object]) -> Dict[str, str]:
    """把标注结果转成 ``{名称: deglyco_core_hash}``，便于断言。"""
    return {name: getattr(ann, "deglyco_core_hash", "") for name, ann in annotations.items()}
