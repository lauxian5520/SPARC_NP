#!/usr/bin/env python3
"""Stage 0-B —— Model A 主干对比实验 (§7.4)。

**项目的第一个真实实验，约一天算力，无需任何 SCAR 组件。**

设置：冻结编码器 + 线性头，Tier-1 靶点，协议 S，3 个种子。

必须出现的对比对象::

    均值预测器            平凡基线（必须有，事实 A）
    ECFP4(2048)+RF        非神经基线（必须有，NaFM Table 2）
    ECFP4+SVR             非神经基线
    A1/A2/A3              至少三个 Model A 候选
    D-MPNN（从零训练）     上界参照
    NaFM（冻结+线性头）    天然产物侧上界参照（**绝不作 Model A**）

选型判据（预注册）：通过判据 10（优于均值基线；不劣于 ECFP4+RF 超过 5%）
且在 Tier-1 上 macro-averaged RMSE 最低者为主 Model A；
第二名进入 §13.3 的 backbone 敏感性分析。

> 若三个候选全部通不过判据 10，这本身是可发表的结论
> （"药物域基础模型在天然产物低数据区不可用"），项目转向 Plan-C，**不是失败**。

用法::

    python scripts/run_s0b_backbone.py --candidates molformer_xl chemberta2_mlm --run-name s0b_v1
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np

from _bootstrap import base_parser, setup

STAGE = "s0b_backbone"


def main() -> int:
    """Stage 0-B 主流程。"""
    parser = base_parser(STAGE, __doc__)
    parser.add_argument("--candidates", nargs="+", default=["molformer_xl"],
                        help="model_registry.yaml 中的 Model A 候选名")
    parser.add_argument("--tier", default="tier1", choices=["tier1", "tier2", "all"])
    parser.add_argument("--skip-stability-check", action="store_true",
                        help="跳过判据 11（输出稳定性）—— 仅调试用")
    parser.add_argument("--allow-unverified", nargs="*", default=["c10"],
                        help="显式放行的 §7.2 P0 判据。默认 ['c10'] —— 低数据稳定性"
                             "本来就要由本阶段产出，跑之前它必然是 unverified。"
                             "放行清单会写进 run manifest 留痕；放行 c9（预训练域门禁）"
                             "必须在实验记录里给出 r_NP 的独立测算依据。")
    args = parser.parse_args()

    config, logger, device, guard = setup(STAGE, args)
    from sparc.common.config import resolve_model_a_entry  # noqa: PLC0415
    from sparc.eval.baselines import MeanPredictor, evaluate_baselines  # noqa: PLC0415
    from sparc.models.model_a import build_model_a, check_output_stability  # noqa: PLC0415

    queries = _load_queries(config, args.tier, logger)
    if not queries:
        logger.error("没有查询数据。请先跑 scripts/run_s0_data.py。")
        return 1

    thresholds = config.model_registry["eligibility_thresholds"]
    results: Dict[str, Any] = {
        "stage": STAGE, "n_queries": len(queries), "candidates": {},
        # §7.2 留痕：本次放行了哪些判据，进 run manifest
        "eligibility_waivers": sorted(args.allow_unverified or []),
    }

    # ---- 强制基线（事实 A）----
    logger.info("== 强制基线：均值预测器 + ECFP4+RF ==")
    predictions, y_true, target_ids = _run_mandatory_baselines(config, queries, logger)

    # ---- Model A 候选 ----
    for name in args.candidates:
        # blocked 名单 + §7.2 P0 判据闸门（c8/c9/c10）在此硬拦
        entry = resolve_model_a_entry(config.model_registry, name,
                                      allow_unverified=args.allow_unverified)
        logger.info("== 候选 Model A：%s ==", name)
        adapter = build_model_a(entry, device=str(device), cache_dir=config.paths.get("model_a_cache"))

        stability: Dict[str, Any] = {"skipped": True}
        if not args.skip_stability_check:
            stability = check_output_stability(
                adapter, [q["smiles"] for q in queries[:64]],
                min_cosine=thresholds["c11_min_cosine_canonical_vs_random_smiles"],
            )
            logger.info("判据 11（输出稳定性）：%s", json.dumps(stability, ensure_ascii=False))

        per_seed: List[np.ndarray] = []
        for seed in config.hparams.train.seeds_multi:
            per_seed.append(_frozen_linear_probe(config, adapter, queries, seed, device, logger))
        predictions[name] = np.mean(per_seed, axis=0)
        results["candidates"][name] = {
            "fingerprint": adapter.fingerprint().to_dict(),
            "c11_output_stability": stability,
            "n_seeds": len(per_seed),
        }

    # ---- 评估 + 判据 10 ----
    rows = evaluate_baselines(predictions, y_true, target_ids)
    results["rows"] = [r.to_dict() for r in rows]
    results["criterion_10"] = _evaluate_criterion_10(rows, args.candidates, thresholds, logger)
    results["selection"] = _select_model_a(results["criterion_10"], logger)

    path = config.paths.stage_outputs(STAGE) / f"{args.run_name}_backbone_report.json"
    path.write_text(json.dumps(results, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info("Stage 0-B 报告：%s", path)

    guard.complete(STAGE, config.freeze_manifest(), artifacts={"report": str(path)},
                   metrics={"selected": results["selection"].get("primary")})
    return 0


def _load_queries(config, tier: str, logger) -> List[Dict[str, Any]]:
    """从 Stage 0 产物读回查询集。"""
    import csv  # noqa: PLC0415

    import yaml  # noqa: PLC0415

    path = config.paths.stage_outputs("s0_data") / "queries.tsv"
    if not path.is_file():
        logger.error("找不到 %s —— 请先跑 run_s0_data.py（完整模式）", path)
        return []
    targets_yaml = config.paths.config_dir / "targets.yaml"
    tiers = {}
    if targets_yaml.is_file():
        tiers = {k: v["tier"] for k, v in yaml.safe_load(targets_yaml.read_text(encoding="utf-8"))["targets"].items()}

    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if tier != "all" and tiers:
        rows = [r for r in rows if tiers.get(r["target_id"]) == tier]
    logger.info("载入查询 %d 条（tier=%s）", len(rows), tier)
    return rows


def _run_mandatory_baselines(config, queries, logger) -> tuple:
    """跑均值预测器与 ECFP4+RF（§12.2 强制）。"""
    from sparc.chem.fingerprints import FingerprintCalculator  # noqa: PLC0415
    from sparc.eval.baselines import EcfpRandomForest, EcfpSVR, MeanPredictor  # noqa: PLC0415

    y = np.array([float(q["pactivity"]) for q in queries])
    target_ids = [q["target_id"] for q in queries]
    split = np.array([q.get("split", "train") for q in queries])
    train_mask = split != "test"

    predictions: Dict[str, np.ndarray] = {}
    mean_model = MeanPredictor().fit(y[train_mask], [t for t, m in zip(target_ids, train_mask) if m])
    predictions["mean_predictor"] = mean_model.predict(len(y), target_ids)

    calculator = FingerprintCalculator(config.hparams.retrieval.ecfp_radius,
                                       config.hparams.retrieval.ecfp_n_bits)
    fingerprints, ok = calculator.ecfp4_batch([q["smiles"] for q in queries])
    if len(ok) == len(queries):
        rf = EcfpRandomForest(seed=config.effective_seed).fit(fingerprints[train_mask], y[train_mask])
        predictions["ecfp4_rf"] = rf.predict(fingerprints)
        try:
            predictions["ecfp4_svr"] = EcfpSVR().fit(fingerprints[train_mask], y[train_mask]).predict(fingerprints)
        except ImportError:
            logger.warning("scikit-learn 缺失，跳过 SVR（可选基线）")
    else:
        logger.error("ECFP 计算失败 %d 条，跳过 RF/SVR 基线", len(queries) - len(ok))
    return predictions, y, target_ids


def _frozen_linear_probe(config, adapter, queries, seed, device, logger) -> np.ndarray:
    """冻结编码器 + 线性头（岭回归闭式解，无需 torch 训练循环）。

    §7.4 的设置就是"冻结编码器 + 线性头"。用岭回归闭式解而不是
    SGD，是因为它无超参、确定性、且在 n≈10² 时是线性探针的正解 ——
    把"编码器好不好"与"探针训得好不好"分开。
    """
    from sparc.models.whitening import FrozenPCAWhitening  # noqa: PLC0415

    rng = np.random.default_rng(seed)
    y = np.array([float(q["pactivity"]) for q in queries])
    split = np.array([q.get("split", "train") for q in queries])
    train_mask = split != "test"

    embeddings = adapter.encode_cached([q["smiles"] for q in queries])
    n_components = min(config.hparams.dims.d_pca_ligand, int(train_mask.sum()) - 1, embeddings.shape[1])
    whitening = FrozenPCAWhitening(n_components).fit(embeddings[train_mask], f"s0b_seed{seed}")
    features = whitening.transform(embeddings)

    design = np.hstack([features, np.ones((features.shape[0], 1), dtype=np.float32)])
    ridge = 1.0
    gram = design[train_mask].T @ design[train_mask] + ridge * np.eye(design.shape[1], dtype=np.float64)
    weights = np.linalg.solve(gram, design[train_mask].T @ y[train_mask])
    logger.debug("线性探针 seed=%d 完成（PCA %d 维）", seed, n_components)
    return design @ weights


def _evaluate_criterion_10(rows, candidates, thresholds, logger) -> Dict[str, Any]:
    """§7.2 判据 10：低数据稳定性。"""
    macro = {r.name: r.rmse for r in rows if r.target_id == "__macro__"}
    mean_rmse = macro.get("mean_predictor", float("nan"))
    ecfp_rmse = macro.get("ecfp4_rf", float("nan"))
    limit = thresholds["c10_max_worse_than_ecfp_rf"]

    result: Dict[str, Any] = {"macro_rmse": macro, "mean_baseline": mean_rmse, "ecfp4_rf": ecfp_rmse}
    for name in candidates:
        rmse = macro.get(name, float("nan"))
        beats_mean = rmse < mean_rmse
        vs_ecfp = (rmse - ecfp_rmse) / ecfp_rmse if ecfp_rmse else float("inf")
        passed = bool(beats_mean and vs_ecfp <= limit)
        result[name] = {"rmse": rmse, "beats_mean_baseline": bool(beats_mean),
                        "relative_to_ecfp_rf": vs_ecfp, "passed": passed}
        logger.info("判据 10 / %s：RMSE %.4f（均值基线 %.4f，ECFP+RF %.4f）⇒ %s",
                    name, rmse, mean_rmse, ecfp_rmse, "通过" if passed else "未通过")
    return result


def _select_model_a(criterion_10: Dict[str, Any], logger) -> Dict[str, Any]:
    """按预注册判据选主 Model A 与备选。"""
    passing = {k: v for k, v in criterion_10.items()
               if isinstance(v, dict) and v.get("passed") and "rmse" in v}
    if not passing:
        logger.warning(
            "全部候选未通过判据 10 ⇒ 触发 Plan-C（§13.5）："
            "论文定位为「药物域基础模型在天然产物低数据区的失效边界刻画」+ 基准贡献。"
            "这不是失败。"
        )
        return {"primary": None, "secondary": None, "plan": "Plan-C"}
    ranked = sorted(passing.items(), key=lambda kv: kv[1]["rmse"])
    selection = {"primary": ranked[0][0],
                 "secondary": ranked[1][0] if len(ranked) > 1 else None,
                 "plan": "GO"}
    logger.info("主 Model A = %s；备选 = %s（进入 §13.3 backbone 敏感性分析）",
                selection["primary"], selection["secondary"])
    return selection


if __name__ == "__main__":
    raise SystemExit(main())
