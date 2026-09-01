#!/usr/bin/env python3
"""Stage 5 —— 测试集单次评估 (§10.4, §12)。

**只允许跑一次。** §10.4：S4 与 S5 之间不允许任何回溯修改；
任何在 S5 之后的调整都必须重新走 S0 并声明为新的实验轮次。
:class:`~sparc.train.stage_guard.StageGuard` 会拒绝在 S4 之后进入 S0–S3。

产出：
* **Table 0** 化学域偏移量化（Stage 0 已产出，这里复制进 reports）
* **Table 1** 主结果（Tier-1，协议 S）
* **Table 2** 消融
* **Table 3** 协议压力测试
* **Figure 1** Risk–Coverage 曲线（四条：朴素/Tanimoto/SPARC-NP/Oracle）
* **Figure 2** 门控系数图
* H1/H2/H3 判定表

每张表都强制带均值预测器基线（事实 A）；缺它 :func:`render_table1` 会拒绝渲染。

用法::

    python scripts/run_s5_eval.py --run-name s5_final --predictions data/s3_gate/outputs/test_predictions.npz \
        --lambda-star 0.42 --deterministic
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from _bootstrap import base_parser, setup

STAGE = "s5_eval"


def main() -> int:
    """Stage 5 主流程。"""
    parser = base_parser(STAGE, __doc__)
    parser.add_argument("--predictions", required=True,
                        help="npz：各方法在测试集上的 gate/loss/prediction，以及 y_true/target_ids")
    parser.add_argument("--lambda-star", type=float, required=True, help="S4 给出的 λ*")
    parser.add_argument("--tier", default="Tier-1")
    parser.add_argument("--protocol", default="S")
    args = parser.parse_args()

    config, logger, _, guard = setup(STAGE, args, need_device=False)
    from sparc.calibrate.ltt import validate_on_test  # noqa: PLC0415
    from sparc.eval.baselines import evaluate_baselines  # noqa: PLC0415
    from sparc.eval.figures import plot_risk_coverage  # noqa: PLC0415
    from sparc.eval.metrics import evaluate_h3, harm_rate, negative_transfer_rate  # noqa: PLC0415
    from sparc.eval.safe_coverage import oracle_gate_values, safe_coverage, tanimoto_gate_values  # noqa: PLC0415
    from sparc.eval.tables import render_hypothesis_report, render_table1, render_table3, write_report  # noqa: PLC0415

    data = np.load(Path(args.predictions), allow_pickle=True)
    y_true = data["y_true"]
    target_ids = [str(t) for t in data["target_ids"]]
    loss_base = data["loss_base"]
    loss_sparc = data["loss_sparc"]
    gate = data["gate"]
    prereg = config.prereg

    logger.warning("=== S5：测试集单次评估。此后任何修改都必须开启新的实验轮次（§10.4）===")

    # ---- 各方法的门控值 ----
    gate_variants: Dict[str, np.ndarray] = {
        "sparc_np": gate,
        "naive": np.ones_like(gate),                                  # 朴素检索：永远接受
        "oracle": oracle_gate_values(loss_sparc, loss_base),           # 不可部署上界
    }
    if "top1_tanimoto" in data.files:
        gate_variants["tanimoto"] = tanimoto_gate_values(data["top1_tanimoto"])

    loss_variants = {
        "sparc_np": loss_sparc,
        "naive": data["loss_naive"] if "loss_naive" in data.files else loss_sparc,
        "oracle": loss_sparc,
        "tanimoto": data["loss_tanimoto"] if "loss_tanimoto" in data.files else loss_sparc,
    }

    curves: Dict[str, Any] = {}
    safe_values: Dict[str, float] = {}
    table1_rows: List[Dict[str, Any]] = []

    # ---- 强制基线行（事实 A）----
    predictions = {k: data[k] for k in data.files if k.startswith("pred_")}
    predictions = {k.replace("pred_", ""): v for k, v in predictions.items()}
    baseline_rows = evaluate_baselines(predictions, y_true, target_ids) if predictions else []
    macro_rmse = {r.name: r.rmse for r in baseline_rows if r.target_id == "__macro__"}
    macro_delta = {r.name: r.delta_rmse_vs_mean for r in baseline_rows if r.target_id == "__macro__"}

    for name in ("mean_predictor", "ecfp4_rf", "base"):
        if name in macro_rmse:
            table1_rows.append({"method": name, "rmse": macro_rmse[name],
                                "delta_rmse_vs_mean": macro_delta.get(name)})

    for name, gate_values in gate_variants.items():
        losses = loss_variants.get(name, loss_sparc)
        result = safe_coverage(
            gate_values, losses, loss_base,
            coverage_grid_step=prereg.safe_coverage["coverage_grid_step"],
            ci_level=prereg.safe_coverage["ci_level"],
            bootstrap_n=prereg.safe_coverage["bootstrap_n"], seed=config.effective_seed,
        )
        curves[name] = result.curve
        safe_values[name] = result.c_star
        table1_rows.append({
            "method": name,
            "safe_coverage": result.c_star,
            "coverage_at_lambda": float((gate_values >= args.lambda_star).mean()),
            "harm_rate": harm_rate(losses, loss_base, prereg.ltt.epsilon),
            "negative_transfer_rate": negative_transfer_rate(losses, loss_base),
            "rmse": macro_rmse.get(name),
            "delta_rmse_vs_mean": macro_delta.get(name),
        })
        logger.info("%-12s SafeCoverage %.3f | Coverage@λ* %.3f | HarmRate %.3f",
                    name, result.c_star, table1_rows[-1]["coverage_at_lambda"], table1_rows[-1]["harm_rate"])

    # ---- H3 判定 ----
    per_target = _per_target_safe_coverage(gate, loss_sparc, loss_base, target_ids, config)
    test_risk = _test_risk(gate, loss_sparc, loss_base, args.lambda_star, prereg.ltt.epsilon)
    h3 = evaluate_h3(
        safe_values["sparc_np"], per_target, test_risk, prereg.ltt.alpha,
        prereg.h3["criteria"]["a_min_safe_coverage"],
        prereg.h3["criteria"]["b_min_target_positive_frac"],
    )
    logger.info("H3 判定：%s —— %s", h3.passed, h3.note)

    # ---- 落盘 ----
    reports_dir = config.paths.get("reports")
    write_report(render_table1(table1_rows, args.tier, args.protocol), reports_dir / "table1.md")
    if "protocol_rows" in data.files:
        write_report(render_table3(json.loads(str(data["protocol_rows"]))), reports_dir / "table3.md")
    write_report(render_hypothesis_report([h3]), reports_dir / "hypotheses.md")

    try:
        plot_risk_coverage(curves, safe_values, config.paths.get("figures") / "figure1_risk_coverage.png")
    except ImportError:
        logger.warning("matplotlib 缺失，跳过 Figure 1")

    output = {
        "stage": STAGE, "lambda_star": args.lambda_star,
        "safe_coverage": safe_values, "table1": table1_rows,
        "h3": h3.to_dict(), "test_risk": test_risk,
        "per_target_safe_coverage": per_target,
    }
    path = config.paths.stage_outputs(STAGE) / f"{args.run_name}_final.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info("Stage 5 完成。报告：%s；表与图在 %s", path, reports_dir)

    guard.complete(STAGE, config.freeze_manifest(), artifacts={"report": str(path)},
                   metrics={"safe_coverage": safe_values["sparc_np"], "h3_passed": h3.passed})
    return 0


def _per_target_safe_coverage(gate, loss_sparc, loss_base, target_ids, config) -> Dict[str, float]:
    """分靶点 SafeCoverage（H3 判据 (b) 的分母）。"""
    from sparc.eval.safe_coverage import safe_coverage  # noqa: PLC0415

    ids = np.asarray(target_ids)
    out: Dict[str, float] = {}
    for target_id in sorted(set(target_ids)):
        mask = ids == target_id
        if mask.sum() < 20:            # 少于 min_samples 时 c* 必为 0，不必浪费 bootstrap
            out[target_id] = 0.0
            continue
        out[target_id] = safe_coverage(
            gate[mask], loss_sparc[mask], loss_base[mask], bootstrap_n=2000,
            seed=config.effective_seed,
        ).c_star
    return out


def _test_risk(gate, loss_sparc, loss_base, lam: float, epsilon: float) -> float:
    """λ* 在测试集上的实际风险（H3 判据 (c)）。"""
    accepted = gate >= lam
    if not accepted.any():
        return 0.0
    return float((loss_sparc[accepted] > loss_base[accepted] + epsilon).mean())


if __name__ == "__main__":
    raise SystemExit(main())
