"""主图 (§12.4)。

* **Figure 1** Risk–Coverage 曲线，含零伤害线与 ``c*`` 交点标注；
  四条曲线（朴素 / Tanimoto / SPARC-NP / Oracle）。
* **Figure 2** 门控系数图：28 个系数带 bootstrap CI，按绝对值排序，
  按"相似度类 / 冲突类 / 不确定性类"着色。
  **这张图只有在门控是 29 参数 logistic 时才画得出来 ——
  它是本项目可解释性主张的全部载体。**

matplotlib 惰性导入：本机（树莓派）没装也不影响其余模块。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from sparc.common.logging_utils import get_logger

_LOGGER = get_logger(__name__)

# Figure 2 的分组配色（色盲友好；不依赖红/绿区分）
GROUP_COLORS: Dict[str, str] = {
    "similarity": "#3B7EA1",     # 蓝 —— 相似度类
    "conflict": "#D97706",       # 橙 —— 冲突类
    "uncertainty": "#6B7280",    # 灰 —— 不确定性类
}


def _require_matplotlib() -> Any:
    """惰性导入 matplotlib。"""
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")        # 服务器无显示，强制非交互后端
        import matplotlib.pyplot as plt  # noqa: PLC0415

        return plt
    except ImportError as exc:
        raise ImportError("绘图需要 matplotlib：pip install matplotlib") from exc


def plot_risk_coverage(
    curves: Dict[str, Sequence[Dict[str, float]]],
    safe_coverage_values: Optional[Dict[str, float]] = None,
    output_path: Optional[Path] = None,
    title: str = "Figure 1 — Risk–Coverage",
    dpi: int = 200,
) -> Any:
    """Figure 1 —— Risk–Coverage 曲线。

    Args:
        curves: ``{方法名: risk_coverage_curve 的产出}``；建议包含
            ``naive`` / ``tanimoto`` / ``sparc_np`` / ``oracle`` 四条。
        safe_coverage_values: ``{方法名: c*}``，用于标注交点。
        output_path: 保存路径；``None`` 时不保存。
        title: 图标题。
        dpi: 分辨率。

    Returns:
        ``matplotlib`` figure。
    """
    plt = _require_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    styles = {
        "naive": ("#9CA3AF", "--", "朴素检索"),
        "tanimoto": ("#3B7EA1", "-.", "Tanimoto 门控"),
        "sparc_np": ("#B91C1C", "-", "SPARC-NP"),
        "oracle": ("#059669", ":", "Oracle 门控（不可部署上界）"),
    }

    for name, curve in curves.items():
        color, linestyle, label = styles.get(name, ("#374151", "-", name))
        coverage = np.array([row["coverage"] for row in curve])
        mean_delta = np.array([row["mean_delta_loss"] for row in curve])
        ucb = np.array([row["ucb"] for row in curve])
        ucb_plot = np.where(np.isfinite(ucb), ucb, np.nan)

        axes[0].plot(coverage, mean_delta, color=color, linestyle=linestyle, label=label, linewidth=1.8)
        axes[0].plot(coverage, ucb_plot, color=color, linestyle=linestyle, linewidth=0.9, alpha=0.45)
        axes[1].plot(coverage, [row["harm_rate"] for row in curve],
                     color=color, linestyle=linestyle, label=label, linewidth=1.8)

        c_star = (safe_coverage_values or {}).get(name)
        if c_star:
            axes[0].axvline(c_star, color=color, linestyle=":", linewidth=0.8, alpha=0.6)
            axes[0].annotate(f"c*={c_star:.2f}", xy=(c_star, 0), xytext=(c_star, 0.02),
                             color=color, fontsize=8, ha="center")

    # 零伤害线 —— c* 就是曲线（的 UCB）与它的交点
    axes[0].axhline(0.0, color="#111827", linewidth=1.0)
    axes[0].set_xlabel("Coverage")
    axes[0].set_ylabel(r"$E[\ell_{SPARC} - \ell_{base}]$（细线为 95% UCB）")
    axes[0].set_title("平均伤害 vs 覆盖率")
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].grid(alpha=0.2)

    axes[1].set_xlabel("Coverage")
    axes[1].set_ylabel(r"Harm rate $P(\ell_{SPARC} > \ell_{base} + \epsilon)$")
    axes[1].set_title("伤害率 vs 覆盖率")
    axes[1].grid(alpha=0.2)

    fig.suptitle(title)
    fig.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        _LOGGER.info("Figure 1 已保存：%s", output_path)
    return fig


def plot_gate_coefficients(
    report: Dict[str, Any],
    output_path: Optional[Path] = None,
    title: str = "Figure 2 — 门控系数（29 参数 logistic）",
    dpi: int = 200,
    show_sign_prior: bool = True,
) -> Any:
    """Figure 2 —— 门控系数图。

    Args:
        report: :meth:`~sparc.models.gate.GateCoefficientReport.to_dict` 的产出。
        output_path: 保存路径。
        title: 图标题。
        dpi: 分辨率。
        show_sign_prior: 是否标注与化学先验冲突的系数（H2 判据 (d)）。

    Returns:
        ``matplotlib`` figure。
    """
    plt = _require_matplotlib()
    features = sorted(report["features"], key=lambda f: abs(f["coefficient"]))

    names = [f["name"] for f in features]
    coefficients = [f["coefficient"] for f in features]
    colors = [GROUP_COLORS.get(f["group"], "#374151") for f in features]
    flips = set(report.get("sign_flips", []))

    fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(features) + 1.6))
    y = np.arange(len(features))
    ax.barh(y, coefficients, color=colors, height=0.7)

    if features[0].get("ci_lower") is not None:
        lower = np.array([f["ci_lower"] for f in features])
        upper = np.array([f["ci_upper"] for f in features])
        ax.errorbar(coefficients, y, xerr=[np.array(coefficients) - lower, upper - np.array(coefficients)],
                    fmt="none", ecolor="#111827", elinewidth=0.8, capsize=2)

    labels = [f"{n} *" if (show_sign_prior and n in flips) else n for n in names]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0.0, color="#111827", linewidth=1.0)
    ax.set_xlabel(r"$w_g$ 系数（$x_g$ 已 z-score 标准化）")
    ax.set_title(f"{title}\n截距 $b_g$ = {report.get('intercept', 0.0):.3f}"
                 + (f"；符号翻转 {len(flips)}/28（H2 判据 d 上限 2）" if show_sign_prior else ""))
    ax.grid(axis="x", alpha=0.2)

    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in GROUP_COLORS.values()]
    ax.legend(handles, ["相似度类", "冲突类", "不确定性类"], fontsize=8, frameon=False, loc="lower right")

    fig.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        _LOGGER.info("Figure 2 已保存：%s", output_path)
    return fig


def plot_domain_shift(
    table0: Dict[str, Any],
    output_path: Optional[Path] = None,
    dpi: int = 200,
) -> Any:
    """Table 0 的配图：描述符分布对比（可选，用于附录）。

    Args:
        table0: :func:`~sparc.data.build_dataset.build_table0` 的产出。
        output_path: 保存路径。
        dpi: 分辨率。

    Returns:
        ``matplotlib`` figure。
    """
    plt = _require_matplotlib()
    descriptors = table0.get("descriptors", {})
    columns = [c for c in descriptors if descriptors[c]["natural_products"].get("n", 0) > 0]
    if not columns:
        raise ValueError("Table 0 中没有可画的描述符")

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(columns) + 1.5))
    y = np.arange(len(columns))
    np_median = [descriptors[c]["natural_products"].get("median", np.nan) for c in columns]
    drug_median = [descriptors[c]["drug_memory"].get("median", np.nan) for c in columns]

    ax.barh(y - 0.2, np_median, height=0.35, color="#059669", label="天然产物查询集")
    ax.barh(y + 0.2, drug_median, height=0.35, color="#3B7EA1", label="药物记忆库（NP-purge 后）")
    ax.set_yticks(y)
    ax.set_yticklabels(columns, fontsize=8)
    ax.set_xlabel("中位数")
    ax.set_title("Table 0 — 化学域偏移")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="x", alpha=0.2)

    fig.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return fig
