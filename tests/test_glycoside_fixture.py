"""§6.2 的强制回归夹具：槲皮素 / 芦丁必须落在同一骨架家族。

**此测试失败则 CI 红灯**（§6.2 原文）。

它保护的是事实 C：COCONUT 2026-08 中 15.8% 的分子是糖苷，Murcko 骨架
保留糖环，于是苷元可以合法地留在其糖苷查询的记忆库里、还是 Top-1。
只用 Murcko + 精确 InChIKey（旧方案）会漏掉这 1/6 的分子。

分两层：
* :class:`TestGlycosideFamilyMerge` —— 不需要 RDKit，用已知的 InChIKey
  验证家族合并逻辑本身，树莓派上也必须绿；
* :class:`TestSugarStripper` —— 需要 RDKit，验证脱糖实现能真的把
  芦丁还原成槲皮素母核。
"""

from __future__ import annotations

import pytest

from sparc.chem.scaffold import ScaffoldFamilyBuilder, ScaffoldKeys
from fixtures.glycoside_leak import (
    GLYCOSIDE_CASES,
    NEGATIVE_CONTROL_SKELETON14,
    NEGATIVE_CONTROL_SMILES,
    QUERCETIN_SKELETON14,
    QUERCETIN_SMILES,
    RUTIN_SKELETON14,
    RUTIN_SMILES,
)

from conftest import requires_rdkit


class TestGlycosideFamilyMerge:
    """家族合并逻辑（不依赖 RDKit）。"""

    def test_quercetin_rutin_same_family(self):
        """§6.2 指定的夹具：二者的 InChIKey 与骨架块都不同，只有脱糖母核相同。"""
        keys = [
            ScaffoldKeys("quercetin", f"{QUERCETIN_SKELETON14}-UHFFFAOYSA-N",
                         skeleton14=QUERCETIN_SKELETON14,
                         deglyco_core_hash=QUERCETIN_SKELETON14,
                         tautomer_family_id=QUERCETIN_SKELETON14),
            ScaffoldKeys("rutin", f"{RUTIN_SKELETON14}-NVPNHPEKSA-N",
                         skeleton14=RUTIN_SKELETON14,
                         deglyco_core_hash=QUERCETIN_SKELETON14,     # 脱糖后回到槲皮素母核
                         tautomer_family_id=RUTIN_SKELETON14),
            ScaffoldKeys("aspirin", f"{NEGATIVE_CONTROL_SKELETON14}-UHFFFAOYSA-N",
                         skeleton14=NEGATIVE_CONTROL_SKELETON14,
                         deglyco_core_hash=NEGATIVE_CONTROL_SKELETON14,
                         tautomer_family_id=NEGATIVE_CONTROL_SKELETON14),
        ]
        result = ScaffoldFamilyBuilder().build(keys)
        assert result.family_of["quercetin"] == result.family_of["rutin"], (
            "槲皮素与芦丁必须落在同一骨架家族（§6.2 强制夹具）。"
            "不满足意味着苷元可以合法地留在其糖苷查询的记忆库里。"
        )
        assert result.family_of["aspirin"] != result.family_of["rutin"], "阴性对照被错误合并"
        assert result.edge_counts.get("deglyco", 0) >= 1

    def test_disabling_deglyco_reproduces_the_leak(self):
        """关掉条款 3 就会复现旧方案的泄漏 —— 证明这把钥匙确实是必需的。"""
        keys = [
            ScaffoldKeys("quercetin", "A", skeleton14=QUERCETIN_SKELETON14,
                         deglyco_core_hash=QUERCETIN_SKELETON14, tautomer_family_id="TQ"),
            ScaffoldKeys("rutin", "B", skeleton14=RUTIN_SKELETON14,
                         deglyco_core_hash=QUERCETIN_SKELETON14, tautomer_family_id="TR"),
        ]
        leaky = ScaffoldFamilyBuilder(use_deglyco_core=False).build(keys)
        assert leaky.family_of["quercetin"] != leaky.family_of["rutin"], (
            "关掉脱糖键后本应复现泄漏；若仍合并，说明测试没有真正检验条款 3"
        )

    def test_tautomer_key_merges_independently(self):
        """条款 4：互变体家族相同也应合并。"""
        keys = [
            ScaffoldKeys("keto", "A", skeleton14="KKKKKKKKKKKKKK",
                         deglyco_core_hash="KKKKKKKKKKKKKK", tautomer_family_id="TAUT1"),
            ScaffoldKeys("enol", "B", skeleton14="EEEEEEEEEEEEEE",
                         deglyco_core_hash="EEEEEEEEEEEEEE", tautomer_family_id="TAUT1"),
        ]
        result = ScaffoldFamilyBuilder().build(keys)
        assert result.family_of["keto"] == result.family_of["enol"]

    def test_builder_rejects_duplicate_ids(self):
        """重复 molecule_id 会让并查集把无关分子并到一起，必须报错。"""
        keys = [ScaffoldKeys("dup", "A", skeleton14="S1"), ScaffoldKeys("dup", "B", skeleton14="S2")]
        with pytest.raises(ValueError, match="molecule_id 必须唯一"):
            ScaffoldFamilyBuilder().build(keys)


@requires_rdkit
class TestSugarStripper:
    """脱糖实现（需要 RDKit）。"""

    def test_rutin_strips_to_quercetin_core(self):
        """芦丁脱糖后的母核 hash 必须等于槲皮素的骨架块。"""
        from sparc.chem.standardize import MoleculeStandardizer
        from sparc.chem.sugar import SugarStripper

        standardizer = MoleculeStandardizer()
        stripper = SugarStripper()

        quercetin = standardizer.standardize(QUERCETIN_SMILES)
        rutin = standardizer.standardize(RUTIN_SMILES)
        assert quercetin.ok and rutin.ok

        rutin_core = stripper.deglyco_core_hash(rutin.canonical_smiles, rutin.inchikey)
        assert rutin_core.ok
        assert rutin_core.n_sugars_removed >= 1, "芦丁的两个糖单元一个都没剥掉"
        assert rutin_core.deglyco_core_hash == quercetin.skeleton14, (
            f"芦丁脱糖母核 {rutin_core.deglyco_core_hash} != 槲皮素骨架块 {quercetin.skeleton14}"
        )

    def test_quercetin_is_unchanged(self):
        """非糖苷分子的母核 hash 就是它自己的骨架块，不应被剥。"""
        from sparc.chem.standardize import MoleculeStandardizer
        from sparc.chem.sugar import SugarStripper

        standardizer = MoleculeStandardizer()
        quercetin = standardizer.standardize(QUERCETIN_SMILES)
        result = SugarStripper().deglyco_core_hash(quercetin.canonical_smiles, quercetin.inchikey)
        assert result.n_sugars_removed == 0
        assert result.deglyco_core_hash == quercetin.skeleton14

    def test_negative_control_not_stripped(self):
        """阿司匹林不含糖，不应被误剥。"""
        from sparc.chem.standardize import MoleculeStandardizer
        from sparc.chem.sugar import SugarStripper

        standardizer = MoleculeStandardizer()
        aspirin = standardizer.standardize(NEGATIVE_CONTROL_SMILES)
        result = SugarStripper().deglyco_core_hash(aspirin.canonical_smiles, aspirin.inchikey)
        assert result.n_sugars_removed == 0

    @pytest.mark.parametrize("case", GLYCOSIDE_CASES, ids=lambda c: c.name)
    def test_all_glycoside_cases(self, case):
        """三个配对用例（O-二糖苷 / C-糖苷 / O-单糖苷）都必须合并到同一母核。"""
        from sparc.chem.standardize import MoleculeStandardizer
        from sparc.chem.sugar import SugarStripper

        standardizer = MoleculeStandardizer()
        stripper = SugarStripper()

        aglycone = standardizer.standardize(case.aglycone_smiles)
        glycoside = standardizer.standardize(case.glycoside_smiles)
        assert aglycone.ok and glycoside.ok, f"{case.name}：SMILES 解析失败"

        glycoside_core = stripper.deglyco_core_hash(glycoside.canonical_smiles, glycoside.inchikey)
        assert glycoside_core.deglyco_core_hash == aglycone.skeleton14, (
            f"{case.name}（{case.note}）：脱糖母核 {glycoside_core.deglyco_core_hash} "
            f"!= 苷元骨架块 {aglycone.skeleton14}"
        )
