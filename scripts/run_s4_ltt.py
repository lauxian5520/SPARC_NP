#!/usr/bin/env python3
"""Stage 4 —— Learn-then-Test 标定 λ (§11)。

**这不是超参搜索。** ``talk.md`` 的做法是"枚举候选 τ_g，挑验证集上
伤害率最低的" —— 那是调参，报告出来的成绩带后选择偏倚。

```
R(λ) = P( ℓ_SPARC(q) > ℓ_base(q) + ε | g(q) ≥ λ )
λ* = 通过 FWER ≤ δ 检验的最小 λ
保证：P( R(λ*) ≤ α ) ≥ 1 − δ        有限样本、分布无关
```

α = 0.10、δ = 0.05、ε = 0.01（冻结在 preregistration.yaml）。

**标定折的独立性是保证成立的前提**：该折不参与 Θ_B/Θ_R 训练，
且其骨架家族已从自身的记忆库视图中剔除（§11.2）。本脚本要求
显式传 ``--confirm-holdout``。

CPU 即可，分钟级。跑完后 :class:`StageGuard` 会锁死 S0–S3 —— §10.4：
S4 与 S5 之间不允许任何回溯修改。

用法::

    python scripts/run_s4_ltt.py --run-name s4_v1 --losses data/s3_gate/outputs/calib_losses.npz --confirm-holdout
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from _bootstrap import base_parser, setup

STAGE = "s4_ltt"


def main() -> int:
    """Stage 4 主流程。"""
    parser = base_parser(STAGE, __doc__)
    parser.add_argument("--losses", required=True,
                        help="npz，含 calib_gate/calib_loss_sparc/calib_loss_base"
                             "（可选 inner_* 用于固定序列定序）")
    parser.add_argument("--confirm-holdout", action="store_true",
                        help="确认标定折未参与 Θ_B/Θ_R 训练，且其骨架家族已从自身记忆库视图剔除（§11.2）")
    parser.add_argument("--alpha", type=float, default=None, help="覆盖 α（仅用于 R4 的敏感性曲线）")
    args = parser.parse_args()

    config, logger, _, guard = setup(STAGE, args, need_device=False)
    from sparc.calibrate.ltt import (  # noqa: PLC0415
        diagnose_fixed_sequence_power,
        fixed_sequence_order_from_inner_fold,
        learn_then_test,
    )
    from sparc.eval.safe_coverage import safe_coverage  # noqa: PLC0415

    if not args.confirm_holdout:
        logger.error(
            "必须传 --confirm-holdout。§11.2：标定折不得参与 Θ_B/Θ_R 训练，"
            "且该折的骨架家族必须从其自身的记忆库视图中剔除。"
            "若标定折参与过 Base 训练，H3 的三条判据同时失效 —— 而失效不会报错。"
        )
        return 1

    data = np.load(Path(args.losses), allow_pickle=False)
    gate = data["calib_gate"]
    loss_sparc = data["calib_loss_sparc"]
    loss_base = data["calib_loss_base"]
    ltt_cfg = config.prereg.ltt
    alpha = args.alpha if args.alpha is not None else ltt_cfg.alpha
    logger.info("标定折样本 %d 条；α=%.2f δ=%.2f ε=%.3f；多重校正=%s",
                len(gate), alpha, ltt_cfg.delta, ltt_cfg.epsilon, ltt_cfg.multiple_testing)

    # 固定序列的功效诊断（本项目标定折规模下这一步经常决定成败）
    diagnosis = diagnose_fixed_sequence_power(gate, alpha, ltt_cfg.delta, ltt_cfg.lambda_grid())
    logger.info("固定序列功效诊断：%s",
                json.dumps({k: v for k, v in diagnosis.items() if k != "per_lambda"}, ensure_ascii=False))

    lambda_order = None
    if ltt_cfg.multiple_testing == "fixed_sequence" and ltt_cfg.fixed_sequence_order_source == "inner_fold":
        if all(k in data.files for k in ("inner_gate", "inner_loss_sparc", "inner_loss_base")):
            lambda_order = fixed_sequence_order_from_inner_fold(
                data["inner_gate"], data["inner_loss_sparc"], data["inner_loss_base"],
                alpha, ltt_cfg.epsilon, ltt_cfg.lambda_grid(),
            )
        else:
            logger.warning("npz 中没有 inner_* 数组，固定序列退化为 λ 从大到小（功效可能不足）")

    result = learn_then_test(
        gate, loss_sparc, loss_base, alpha=alpha, delta=ltt_cfg.delta, epsilon=ltt_cfg.epsilon,
        lambda_grid=ltt_cfg.lambda_grid(), multiple_testing=ltt_cfg.multiple_testing,
        lambda_order=lambda_order, calibration_is_held_out=True,
    )

    coverage = safe_coverage(
        gate, loss_sparc, loss_base,
        coverage_grid_step=config.prereg.safe_coverage["coverage_grid_step"],
        ci_level=config.prereg.safe_coverage["ci_level"],
        bootstrap_n=config.prereg.safe_coverage["bootstrap_n"],
        seed=config.effective_seed,
    )

    output: Dict[str, Any] = {
        "stage": STAGE, "ltt": result.to_dict(), "safe_coverage_on_calib": coverage.to_dict(),
        "fixed_sequence_diagnosis": {k: v for k, v in diagnosis.items() if k != "per_lambda"},
    }

    if not result.found:
        logger.warning(
            "在 α=%.2f 下不存在满足风险约束的 λ。§16 R4：这本身是有效结论 —— "
            "报告它，并同时给出 α=0.20 的 Risk–Coverage 曲线。", alpha,
        )
        relaxed = learn_then_test(
            gate, loss_sparc, loss_base, alpha=0.20, delta=ltt_cfg.delta, epsilon=ltt_cfg.epsilon,
            lambda_grid=ltt_cfg.lambda_grid(), multiple_testing=ltt_cfg.multiple_testing,
            lambda_order=lambda_order, calibration_is_held_out=True,
        )
        output["ltt_alpha_020"] = relaxed.to_dict()

    path = config.paths.stage_outputs(STAGE) / f"{args.run_name}_ltt.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info("Stage 4 报告：%s", path)
    logger.info(
        "λ* = %s。**S4 已完成：从此刻起不允许任何回溯修改（§10.4）**，"
        "S5 只允许对测试集评估一次。", result.lambda_star,
    )

    guard.complete(STAGE, config.freeze_manifest(), artifacts={"report": str(path)},
                   metrics={"lambda_star": result.lambda_star,
                            "coverage_at_lambda_star": result.coverage_at_star(),
                            "safe_coverage_calib": coverage.c_star})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
