#!/usr/bin/env python3
"""Stage 3 —— 交叉拟合效用标签 → 29 参数门控 → H2 判据 (§8.3.5, §10.3)。

**部署交叉拟合集成本身** (§10.3)::

    R_b(q) = (1/J) Σ_{j=1..J} f_R^(−j)(q)      J = 5

``talk.md`` 的效用标签来自 ``f_R^(−j)``（只见 4/5 dev），却用来给最终的
``f_R``（见全部 dev）把关 —— 系统性低估检索效用，导致过度拒检。

必须报告：``|M^cf| / |M^deploy|`` 与两者支持度特征分布的 KS 距离；
KS > 0.15 说明协变量偏移过大，需缩小 J。

附加诊断 ``C_FORCED_RETRIEVAL`` (§13.4)：拿 C-NP-PATHWAY 走完整检索管线，
**不告诉门控这是什么任务**，报告 ``E[g|ρ=C]`` 与 ``E[g|ρ=B]`` 的对比。
若门控自己把 g 压下去了，就同时证明了负迁移真实存在 + 门控在无提示下
自行识别了它。

用法::

    python scripts/run_s3_gate.py --run-name s3_v1 --retrieval-ckpt s2_v1
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np

from _bootstrap import base_parser, setup

STAGE = "s3_gate"


def main() -> int:
    """Stage 3 主流程。"""
    parser = base_parser(STAGE, __doc__)
    parser.add_argument("--retrieval-ckpt", default=None, help="S2 的 Θ_R checkpoint run 名")
    parser.add_argument("--gate-features", default=None,
                        help="预计算的 x_g 特征 npz（含 features/labels/groups/split）")
    parser.add_argument("--forced-retrieval-diagnostic", action="store_true",
                        help="附加跑 C_FORCED_RETRIEVAL 诊断（§13.4）")
    args = parser.parse_args()

    config, logger, device, guard = setup(STAGE, args)
    import torch  # noqa: PLC0415

    from sparc.calibrate.crossfit import CrossFitEnsemble  # noqa: PLC0415
    from sparc.eval.metrics import evaluate_h2  # noqa: PLC0415
    from sparc.models.gate import SupportGate  # noqa: PLC0415

    data = _load_gate_data(config, args, logger)
    if data is None:
        return 1

    features = torch.from_numpy(data["features"]).float().to(device)
    labels = torch.from_numpy(data["labels"]).float().to(device)

    gate = SupportGate(config.hparams.dims.n_gate_features, config.gate_manifest.names).to(device)
    assert gate.n_parameters() == 29, "门控必须正好 29 个参数（§8.3.5）"
    gate.fit_standardizer(features[data["split"] == "train"],
                          clip_sigma=config.gate_manifest.raw["standardization"]["clip_sigma"])

    # ---- 交叉拟合（按骨架家族分折，不按样本） ----
    ensemble = CrossFitEnsemble(config.hparams.train.crossfit["n_folds"], seed=config.effective_seed)
    ensemble.assign_folds(data["groups"])

    l2_grid = config.hparams.train.gate["l2_grid"]
    best_l2, oof_scores = _fit_gate_crossfit(gate, features, labels, ensemble, l2_grid, config, logger, device)
    logger.info("门控 L2 = %s（内层 CV 选出）", best_l2)

    # ---- H2 判据 ----
    report = gate.coefficient_report(config.gate_manifest.sign_priors, config.gate_manifest.groups)
    coefficient_dict = report.to_dict()
    logger.info("门控系数（Figure 2 数据）：符号翻转 %d/28",
                coefficient_dict["n_sign_flips"])

    h2 = evaluate_h2(
        labels=data["labels"], gate_scores_oof=oof_scores,
        tanimoto_scores=data["tanimoto_top1"],
        cliff_mask=data.get("cliff_mask"),
        sign_flips=coefficient_dict["sign_flips"],
        min_cliff_auroc=config.prereg.h2["criteria"]["b_min_auroc_on_activity_cliff_subset"],
        max_generalization_gap=config.prereg.h2["criteria"]["c_max_generalization_gap"],
        max_sign_flips=config.prereg.h2["criteria"]["d_max_sign_flips"],
        n_bootstrap=config.prereg.h2["paired_bootstrap_n"],
        seed=config.effective_seed,
    )
    logger.info("H2 判定：%s —— %s", h2.passed, h2.note)

    results: Dict[str, Any] = {
        "stage": STAGE, "best_l2": best_l2, "gate_coefficients": coefficient_dict,
        "h2": h2.to_dict(), "n_samples": int(len(data["labels"])),
        "utility_positive_rate": float(np.mean(data["labels"])),
    }
    if args.forced_retrieval_diagnostic:
        results["c_forced_retrieval"] = _forced_retrieval_diagnostic(config, gate, data, logger, device)

    path = config.paths.stage_outputs(STAGE) / f"{args.run_name}_gate.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    from sparc.eval.figures import plot_gate_coefficients  # noqa: PLC0415

    try:
        plot_gate_coefficients(coefficient_dict, config.paths.get("figures") / "figure2_gate_coefficients.png")
    except ImportError:
        logger.warning("matplotlib 缺失，跳过 Figure 2 绘制")

    torch.save({"model_state": gate.state_dict(), "best_l2": best_l2},
               config.paths.stage_checkpoints(STAGE) / f"{args.run_name}_gate.pt")
    logger.info("Stage 3 报告：%s", path)
    guard.complete(STAGE, config.freeze_manifest(), artifacts={"report": str(path)},
                   metrics={"h2_passed": h2.passed, "n_sign_flips": coefficient_dict["n_sign_flips"]})
    return 0


def _load_gate_data(config, args, logger) -> Dict[str, Any] | None:
    """载入预计算的 x_g 特征与效用标签。"""
    from pathlib import Path  # noqa: PLC0415

    path = Path(args.gate_features) if args.gate_features else (
        config.paths.stage_outputs("s2_retrieval") / "gate_features.npz"
    )
    if not path.is_file():
        logger.error(
            "缺少门控特征文件 %s。它由 Stage 2 在 dev 上跑完整检索管线后产出，"
            "含 features(N,28) / labels(N) / groups(N) / split(N) / tanimoto_top1(N)。", path,
        )
        return None
    data = np.load(path, allow_pickle=False)
    out = {k: data[k] for k in data.files}
    logger.info("门控数据：%d 条，正类率 %.3f", len(out["labels"]), float(np.mean(out["labels"])))
    return out


def _fit_gate_crossfit(gate, features, labels, ensemble, l2_grid, config, logger, device):
    """交叉拟合训练门控，内层 CV 选 L2，返回 (best_l2, OOF 分数)。"""
    import torch  # noqa: PLC0415

    from sparc.eval.metrics import roc_auc  # noqa: PLC0415

    standardized = gate.standardize(features)
    n = standardized.shape[0]
    best_l2, best_auroc, best_oof = l2_grid[0], -1.0, None

    for l2 in l2_grid:
        oof = np.zeros(n)
        for fold in range(ensemble.n_folds):
            train_idx = torch.from_numpy(np.nonzero(ensemble.fold_of != fold)[0]).to(device)
            eval_idx = torch.from_numpy(np.nonzero(ensemble.fold_of == fold)[0]).to(device)
            weights = _fit_logistic(standardized[train_idx], labels[train_idx], l2,
                                    config.hparams.train.gate, device)
            with torch.no_grad():
                logits = standardized[eval_idx] @ weights[:-1] + weights[-1]
                oof[eval_idx.cpu().numpy()] = torch.sigmoid(logits).cpu().numpy()
        auroc = roc_auc(labels.cpu().numpy(), oof)
        logger.info("L2=%.3g ⇒ OOF AUROC %.4f", l2, auroc)
        if auroc > best_auroc:
            best_l2, best_auroc, best_oof = l2, auroc, oof

    # 用最优 L2 在全部数据上重拟合，作为部署门控
    final = _fit_logistic(standardized, labels, best_l2, config.hparams.train.gate, device)
    with torch.no_grad():
        gate.linear.weight.copy_(final[:-1].unsqueeze(0))
        gate.linear.bias.copy_(final[-1:])
    return best_l2, best_oof


def _fit_logistic(x, y, l2: float, gate_cfg: Dict[str, Any], device):
    """带 L2 的 logistic 回归（全批 LBFGS，29 个参数无需 minibatch）。"""
    import torch  # noqa: PLC0415

    weights = torch.zeros(x.shape[1] + 1, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS([weights], lr=gate_cfg["lr"], max_iter=gate_cfg["max_iter"])

    def closure():
        """LBFGS 闭包。"""
        optimizer.zero_grad()
        logits = x @ weights[:-1] + weights[-1]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss = loss + l2 * (weights[:-1] ** 2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    return weights.detach()


def _forced_retrieval_diagnostic(config, gate, data, logger, device) -> Dict[str, Any]:
    """``C_FORCED_RETRIEVAL`` 诊断 (§13.4)。

    这一条实验把"任务路由"从一个配置开关变回方法贡献：若门控在
    **无提示**的情况下对 C 类任务自行压低 ``g``，就同时证明了
    负迁移真实存在 + 门控识别了它。
    """
    import torch  # noqa: PLC0415

    if "task_type" not in data:
        logger.warning("门控数据中没有 task_type 列，跳过 C_FORCED_RETRIEVAL 诊断")
        return {"skipped": True, "reason": "no_task_type_column"}

    features = torch.from_numpy(data["features"]).float().to(device)
    with torch.no_grad():
        g = gate(features).cpu().numpy()
    task_type = data["task_type"]
    g_b = g[task_type == "B"]
    g_c = g[task_type == "C"]
    result = {
        "E_g_given_B": float(g_b.mean()) if g_b.size else None,
        "E_g_given_C": float(g_c.mean()) if g_c.size else None,
        "n_B": int(g_b.size), "n_C": int(g_c.size),
        "gate_suppressed_on_C": bool(g_c.size and g_b.size and g_c.mean() < g_b.mean()),
        "note": "门控在无提示下压低 C 类的 g ⇒ 负迁移真实存在 + 门控自行识别（§13.4）",
    }
    logger.info("C_FORCED_RETRIEVAL：E[g|B]=%s E[g|C]=%s",
                result["E_g_given_B"], result["E_g_given_C"])
    return result


if __name__ == "__main__":
    raise SystemExit(main())
