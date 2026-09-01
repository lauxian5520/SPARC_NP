#!/usr/bin/env python3
"""Stage 2 —— SCAR 检索分支训练 (§8.3, §10.2)。

可训练：Θ_R \\ gate（56,026 − 29 = 55,997 参数）。**Θ_B 冻结。**

```
L_R = L_task + λ_rank·L_rank + λ_utility·L_utility-gate + λ_cal·L_cal + 1e-5·||Θ_R||²
L_utility-gate = L_utility + 0.5·L_harm
```

**训练期使用连续 ``g``**（§8.3.5）。硬阈值 ``λ`` 只在 Stage 4 之后生效 ——
按字面实现 ``g̃ = g·1[g≥λ]`` 会让门控从 ``L_task`` 得到的梯度恒为零。

每个 epoch 结束跑一次 §8.3.4 的退化断言（fp32），不等训练完才发现它坏了。

用法::

    python scripts/run_s2_retrieval.py --run-name s2_v1 --base-ckpt s1_v1_frozen
"""

from __future__ import annotations

import json
from typing import Any, Dict

from _bootstrap import base_parser, setup

STAGE = "s2_retrieval"


def main() -> int:
    """Stage 2 主流程。"""
    parser = base_parser(STAGE, __doc__)
    parser.add_argument("--base-ckpt", default=None, help="S1-F 冻结的 Θ_B checkpoint run 名")
    parser.add_argument("--model-a", default="molformer_xl")
    parser.add_argument("--protocol", default="S", choices=["S", "T", "A", "X"])
    parser.add_argument("--lambda-rank", type=float, default=None, help="覆盖冻结默认值（须在网格内）")
    parser.add_argument("--lambda-utility", type=float, default=None)
    parser.add_argument("--lambda-cal", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--skeleton-only", action="store_true",
                        help="只验证模型构造与参数预算，不训练。会写账本但标记 trained=False。")
    args = parser.parse_args()

    config, logger, device, guard = setup(STAGE, args)
    weights = _resolve_loss_weights(config, args, logger)

    import torch  # noqa: PLC0415

    from sparc.common.checkpoint import CheckpointManager  # noqa: PLC0415
    from sparc.models.base import BasePredictor  # noqa: PLC0415
    from sparc.models.evidence import EvidenceLite  # noqa: PLC0415
    from sparc.models.gate import SupportGate  # noqa: PLC0415
    from sparc.models.graphmatcher import GraphMatcherLite  # noqa: PLC0415
    from sparc.models.reranker import RerankerLite  # noqa: PLC0415
    from sparc.models.residual import ResidualHead, UncertaintyHead  # noqa: PLC0415
    from sparc.models.sparc_model import SparcNP  # noqa: PLC0415
    from sparc.train import Trainer, TrainerConfig  # noqa: PLC0415

    # ---- 组装模型并载入冻结的 Θ_B ----
    base = BasePredictor(dropout=config.hparams.train.base["dropout"])
    if args.base_ckpt:
        manager = CheckpointManager(config.paths.stage_checkpoints("s1_base"), args.base_ckpt)
        extra = manager.load_weights_only(base, map_location=device)
        logger.info("已载入 S1-F 冻结的 Θ_B（label_mean=%.3f, label_std=%.3f）",
                    extra.get("label_mean", float("nan")), extra.get("label_std", float("nan")))
    else:
        logger.warning("未指定 --base-ckpt，Θ_B 为随机初始化 —— 仅可用于形状调试，不得进入论文")
    base.freeze()

    sinkhorn = config.hparams.sinkhorn
    model = SparcNP(
        base=base,
        graph_matcher=GraphMatcherLite(tau_m=sinkhorn.tau_m, sinkhorn_iters=sinkhorn.iters,
                                       with_dustbin=sinkhorn.with_dustbin, eps=sinkhorn.eps),
        reranker=RerankerLite(), evidence=EvidenceLite(tau_rank=config.hparams.loss.tau_rank),
        residual=ResidualHead(rank=config.hparams.dims.residual_rank),
        uncertainty=UncertaintyHead(),
        gate=SupportGate(config.hparams.dims.n_gate_features, config.gate_manifest.names),
        detach_base_hidden=config.hparams.train.retrieval["detach_base_hidden"],
    ).to(device)

    budget = model.parameter_budget()
    logger.info("参数预算：%s", json.dumps(budget, ensure_ascii=False))
    expected = config.hparams.param_budget
    for key in ("theta_b", "theta_r", "total_trainable"):
        assert budget[key] == expected[key], f"{key} 参数量 {budget[key]} != 规范 {expected[key]}"

    logger.info(
        "损失权重：λ_rank=%.2f λ_utility=%.2f λ_cal=%.2f（harm_weight=%.1f）",
        weights["lambda_rank"], weights["lambda_utility"], weights["lambda_cal"],
        config.hparams.loss.harm_weight,
    )
    logger.info("训练期使用**连续 g**；硬阈值 λ 只在 Stage 4 之后生效（§8.3.5）")

    trainer_config = TrainerConfig.from_hparams(
        STAGE, config.hparams.train.retrieval, config.hparams.train.amp, args.run_name
    )
    if args.epochs:
        trainer_config.epochs = args.epochs

    Trainer(
        model, trainer_config, device,
        config.paths.stage_logs(STAGE), config.paths.stage_checkpoints(STAGE),
        trainable_parameters=model.trainable_parameters("s2"),
        config_sha256=config.freeze_manifest(),
    )
    logger.info(
        "Stage 2 骨架已就绪：模型已构造、参数预算已核对、Trainer 已配置。\n"
        "训练数据装配需要 Stage 0 的记忆库与 Model A 缓存 —— 接线时必须用：\n"
        "    pipeline = RetrievalPipeline.for_model(model, k_src=..., k0=..., k_top=...)\n"
        "**不要自己 new 一个 GraphMatcherLite**：管线与模型用不同实例时，"
        "node_encoder / edge_encoder / 3 层 GINE 共 18,883 个参数（全部可训练参数的 16.7%）"
        "永远拿不到梯度，而损失曲线与参数量核对都一切正常 —— 这个失败是静默的。"
        "接好之后调用 sparc.retrieval.pipeline.assert_graph_matcher_shared(model, pipeline) 自检。"
    )
    if not args.skeleton_only:
        logger.error(
            "Stage 2 尚未接入训练数据，Θ_R 没有被训练过。**不会**把本阶段标记为完成 —— "
            "否则账本会声称 S2 已完成，而 S3/S4/S5 会在一个随机初始化的 Θ_R 上继续跑，"
            "且 S4 之后就再也回不来了（§10.4）。\n"
            "接入数据后再跑；只想验证骨架请显式加 --skeleton-only。"
        )
        return 1

    logger.warning(
        "--skeleton-only：仅验证模型构造与参数预算，Θ_R 未训练。"
        "本次会写入账本但标记 trained=False，S3 会拒绝使用它。"
    )
    guard.complete(STAGE, config.freeze_manifest(),
                   metrics={"loss_weights": weights, "budget": budget, "trained": False})
    return 0


def _resolve_loss_weights(config, args, logger) -> Dict[str, float]:
    """解析损失权重，并强制它们落在冻结网格内 (§10.2)。

    Raises:
        ValueError: 取值不在冻结网格内。未冻结的搜索网格是数据泄漏
            最常见的入口，因此这里不接受"就试一下"。
    """
    loss = config.hparams.loss
    resolved = {
        "lambda_rank": args.lambda_rank if args.lambda_rank is not None else loss.lambda_rank,
        "lambda_utility": args.lambda_utility if args.lambda_utility is not None else loss.lambda_utility,
        "lambda_cal": args.lambda_cal if args.lambda_cal is not None else loss.lambda_cal,
    }
    grids = {"lambda_rank": loss.grid_rank, "lambda_utility": loss.grid_utility, "lambda_cal": loss.grid_cal}
    for name, value in resolved.items():
        if value not in grids[name]:
            raise ValueError(
                f"{name}={value} 不在冻结网格 {grids[name]} 内。§10.2：网格必须在跑第一个实验前"
                "冻结并写入 configs/frozen_hparams.yaml —— 未冻结的搜索网格是数据泄漏最常见的入口。"
            )
    return resolved


if __name__ == "__main__":
    raise SystemExit(main())
