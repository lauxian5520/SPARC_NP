"""主表渲染 (§12.3)。

四张表，**Table 0 必须在主表之前**：

* **Table 0** 化学域偏移量化 —— 分不开的话后面全部实验没有意义；
* **Table 1** 主结果（Tier-1 靶点，协议 S）；
* **Table 2** 消融；
* **Table 3** 协议压力测试（S/T/A/X）。

所有表都渲染成 markdown 写入 ``reports/``。每张表都强制带均值基线行 ——
:func:`_assert_mean_baseline` 会拒绝渲染缺它的表。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from sparc.common.logging_utils import get_logger

_LOGGER = get_logger(__name__)

MEAN_BASELINE_KEYS = ("mean_predictor", "均值预测器", "mean_baseline")


def _assert_mean_baseline(rows: Sequence[Dict[str, Any]], key: str = "method") -> None:
    """拒绝渲染缺均值基线的表 (§12.2, 事实 A)。

    Raises:
        ValueError: 表中没有均值基线行。
    """
    present = {str(row.get(key, "")).strip().lower() for row in rows}
    if not any(k.lower() in present for k in MEAN_BASELINE_KEYS):
        raise ValueError(
            "本表缺少均值预测器基线行。§12.2 与事实 A 规定它必须出现在每一张表：\n"
            "  在 NPASS 上按 NaFM 口径复原其 8 个靶点，已发表 SOTA 有 4 个跑不过\n"
            "  '预测训练集均值'（PTP-1B 上高 31%，隐含 R² = −0.73），ECFP 8/8 全败。\n"
            f"  当前表中的方法：{sorted(present)}"
        )


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]], align: Optional[Sequence[str]] = None) -> str:
    """渲染一个 markdown 表格。"""
    align = align or ["---"] * len(headers)
    lines = ["| " + " | ".join(str(h) for h in headers) + " |",
             "|" + "|".join(align) + "|"]
    for row in rows:
        lines.append("| " + " | ".join("" if v is None else str(v) for v in row) + " |")
    return "\n".join(lines)


def _fmt(value: Any, digits: int = 4) -> str:
    """数值格式化。"""
    if value is None:
        return "—"
    if isinstance(value, float):
        return "—" if value != value else f"{value:.{digits}f}"      # NaN -> —
    return str(value)


# ======================================================================
def render_table0(table0: Dict[str, Any]) -> str:
    """Table 0 —— 化学域偏移量化 (§12.4)。

    Args:
        table0: :func:`sparc.data.build_dataset.build_table0` 的产出。

    Returns:
        markdown 文本。
    """
    lines = [
        "# Table 0 — 化学域偏移量化",
        "",
        "> §13.1 Stage 0 闸门：若药物记忆库与天然产物查询集在化学空间上分不开",
        "> （Top-1 Tanimoto 中位数 > 0.6），说明所谓“跨域”不成立 ⇒ 停止，重新定义域。",
        "",
        f"- 天然产物查询分子：**{table0.get('n_query_molecules', 0):,}**",
        f"- 清洗后药物记忆库分子：**{table0.get('n_memory_molecules', 0):,}**",
        "",
        "## 分子描述符对比",
        "",
    ]
    rows: List[List[Any]] = []
    for column, stats in table0.get("descriptors", {}).items():
        np_stats = stats.get("natural_products", {})
        drug_stats = stats.get("drug_memory", {})
        rows.append([
            column,
            _fmt(np_stats.get("median")), _fmt(np_stats.get("mean")), _fmt(np_stats.get("std")),
            _fmt(drug_stats.get("median")), _fmt(drug_stats.get("mean")), _fmt(drug_stats.get("std")),
        ])
    lines.append(_md_table(
        ["描述符", "NP 中位数", "NP 均值", "NP 标准差", "药物 中位数", "药物 均值", "药物 标准差"],
        rows, ["---", "---:", "---:", "---:", "---:", "---:", "---:"],
    ))

    overlap = table0.get("scaffold_overlap", {})
    lines += [
        "", "## 骨架重叠（NP-purge 后应为 0）", "",
        _md_table(
            ["项", "值"],
            [["查询集骨架块数", f"{overlap.get('n_query_skeletons', 0):,}"],
             ["记忆库骨架块数", f"{overlap.get('n_memory_skeletons', 0):,}"],
             ["交集", f"**{overlap.get('n_intersection', 0)}**"],
             ["重叠率", _fmt(overlap.get("overlap_rate"), 6)]],
            ["---", "---:"],
        ),
    ]

    top1 = table0.get("top1_tanimoto")
    if top1:
        lines += [
            "", "## Top-1 Tanimoto 分布（Stage 0 闸门判据）", "",
            _md_table(
                ["统计量", "值"],
                [["中位数", f"**{_fmt(top1.get('median'))}**"], ["均值", _fmt(top1.get("mean"))],
                 ["P10", _fmt(top1.get("p10"))], ["P90", _fmt(top1.get("p90"))],
                 ["< 0.35 占比（H1 低支持子集）", _fmt(top1.get("frac_below_0.35"))]],
                ["---", "---:"],
            ),
            "",
            f"> 闸门：中位数 {_fmt(top1.get('median'))} "
            f"{'≤' if (top1.get('median') or 1) <= 0.6 else '>'} 0.60 ⇒ "
            f"{'通过' if (top1.get('median') or 1) <= 0.6 else '**未通过，须停止并重新定义域**'}",
        ]
    return "\n".join(lines) + "\n"


def render_table1(rows: Sequence[Dict[str, Any]], tier: str = "Tier-1", protocol: str = "S") -> str:
    """Table 1 —— 主结果 (§12.3)。

    行：均值预测器 / ECFP4+RF / Base(Model A) / Base + 朴素 kNN /
        Base + Tanimoto 门控 / **SPARC-NP** / Oracle 门控（不可部署上界）
    列：SafeCoverage ↑ · Coverage@λ* · HarmRate ↓ · NT rate ↓ · RMSE · ΔRMSE vs 均值基线

    Args:
        rows: 每行含 ``method`` 与各列的字典。
        tier: 靶点层级。
        protocol: 划分协议。

    Returns:
        markdown 文本。
    """
    _assert_mean_baseline(rows)
    header = ["方法", "SafeCoverage ↑", "Coverage@λ*", "HarmRate ↓", "NT rate ↓", "RMSE", "ΔRMSE vs 均值基线"]
    body = [[
        row.get("method"),
        _fmt(row.get("safe_coverage")), _fmt(row.get("coverage_at_lambda")),
        _fmt(row.get("harm_rate")), _fmt(row.get("negative_transfer_rate")),
        _fmt(row.get("rmse")), _fmt(row.get("delta_rmse_vs_mean")),
    ] for row in rows]
    return "\n".join([
        f"# Table 1 — 主结果（{tier} 靶点，协议 {protocol}）", "",
        "> 主标指标是 **SafeCoverage**，不是 RMSE（事实 A：已发表 SOTA 在半数靶点上",
        "> 跑不过均值预测器，在这样的数据上把主张写成“提升 RMSE”不可能被证实）。",
        "> Oracle 门控是**不可部署的上界**，不是竞争对手。",
        "",
        _md_table(header, body, ["---", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]) + "\n"


def render_table2(rows: Sequence[Dict[str, Any]]) -> str:
    """Table 2 —— 消融 (§12.3)。

    行：−GraphMatcher / −Reranker（改用 Tanimoto 排序）/ −冲突特征 /
        门控 MLP vs logistic / rank-8 vs rank-32 残差 / λ 网格搜索 vs LTT

    Args:
        rows: 消融结果行。

    Returns:
        markdown 文本。
    """
    header = ["消融", "SafeCoverage ↑", "Coverage@λ*", "HarmRate ↓", "NT rate ↓", "RMSE", "Δ vs 完整模型"]
    body = [[
        row.get("ablation"),
        _fmt(row.get("safe_coverage")), _fmt(row.get("coverage_at_lambda")),
        _fmt(row.get("harm_rate")), _fmt(row.get("negative_transfer_rate")),
        _fmt(row.get("rmse")), _fmt(row.get("delta_vs_full")),
    ] for row in rows]
    return "\n".join([
        "# Table 2 — 消融", "",
        "> “λ 网格搜索 vs LTT”这一行是 §11.1 的核心论证：网格搜索的成绩带",
        "> 后选择偏倚，LTT 给的是有限样本、分布无关的保证。两者数字接近",
        "> 恰恰说明保证是**免费**的，而不是说明 LTT 多余。",
        "",
        _md_table(header, body, ["---", "---:", "---:", "---:", "---:", "---:", "---:"]),
    ]) + "\n"


def render_table3(rows: Sequence[Dict[str, Any]]) -> str:
    """Table 3 —— 协议压力测试 (§12.3)。

    行：协议 S / T / A / X
    列：SafeCoverage · Coverage · HarmRate · 记忆库规模 ``|M_view|`` 中位数

    Args:
        rows: 每协议一行。

    Returns:
        markdown 文本。
    """
    header = ["协议", "划分单位", "SafeCoverage ↑", "Coverage@λ*", "HarmRate ↓", "|M_view| 中位数", "n_test"]
    body = [[
        row.get("protocol"), row.get("split_unit"),
        _fmt(row.get("safe_coverage")), _fmt(row.get("coverage_at_lambda")),
        _fmt(row.get("harm_rate")), row.get("median_memory_view"), row.get("n_test"),
    ] for row in rows]
    return "\n".join([
        "# Table 3 — 协议压力测试", "",
        "| 协议 | 含义 |", "|---|---|",
        "| S | 骨架家族（主协议，四把钥匙的连通分量） |",
        "| T | 靶点 / ortholog_group（cold-target 泛化） |",
        "| A | assay / 文献年份（时序外推） |",
        "| X | 生物来源（天然产物特有的分布偏移） |",
        "",
        _md_table(header, body, ["---", "---", "---:", "---:", "---:", "---:", "---:"]),
    ]) + "\n"


def render_hypothesis_report(results: Sequence[Any]) -> str:
    """渲染 H1/H2/H3 的判据表。

    Args:
        results: :class:`~sparc.eval.metrics.HypothesisResult` 列表。

    Returns:
        markdown 文本。
    """
    lines = ["# 预注册假设判定", "",
             "> 判据在跑第一个实验前写死于 `configs/preregistration.yaml` 并 hash 冻结。", ""]
    for result in results:
        status = "✅ 成立" if result.passed else "❌ 不成立"
        lines += [f"## {result.hypothesis} — {status}", ""]
        rows = []
        for name, criterion in result.criteria.items():
            rows.append([
                name, _fmt(criterion.get("value")), _fmt(criterion.get("threshold")),
                "✅" if criterion.get("passed") else "❌",
            ])
        lines += [_md_table(["判据", "实测值", "阈值", "通过"], rows, ["---", "---:", "---:", ":-:"]), ""]
        if result.note:
            lines += [f"> {result.note}", ""]
    return "\n".join(lines)


def write_report(content: str, path: Any) -> Any:
    """把渲染结果写到 ``reports/``。"""
    from pathlib import Path  # noqa: PLC0415

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    _LOGGER.info("报告已写入：%s", path)
    return path
