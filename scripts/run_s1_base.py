#!/usr/bin/env python3
"""Stage 1 —— Base 预测器训练 + H1 检验 (§8.2, §10.1, §13.3)。

两件事：
1. 训练 Θ_B（57,218 参数，Tobit 似然，含审查记录）；
2. **H1 检验** —— 只需 Base + 朴素 kNN。不需要门控、不需要重排器、
   不需要图匹配、不需要交叉拟合。**这是决定整个项目有没有立足点的实验，
   一天能跑完。**

止损条件（硬性，§13.3）：若 H1 在 ≥ 2/3 的 Tier-1 靶点上，低支持子集
Δℓ 的 95% CI 下界 ≤ 0 ⇒ 无负迁移可控 ⇒ 立即执行 §13.5 Plan-B。

训练完成后自动执行 **S1-F**：冻结 Θ_B 并保存独立 checkpoint。

用法::

    python scripts/run_s1_base.py --run-name s1_v1 --model-a molformer_xl
    python scripts/run_s1_base.py --h1-only --run-name s1_h1     # 只跑 H1 检验
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np

from _bootstrap import base_parser, setup

STAGE = "s1_base"


def main() -> int:
    """Stage 1 主流程。"""
    parser = base_parser(STAGE, __doc__)
    parser.add_argument("--model-a", default="molformer_xl", help="主 Model A（由 S0-B 选出）")
    parser.add_argument("--protocol", default="S", choices=["S", "T", "A", "X"])
    parser.add_argument("--h1-only", action="store_true", help="跳过训练，只跑 H1 检验")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖冻结的 epoch 数")
    args = parser.parse_args()

    config, logger, device, guard = setup(STAGE, args)
    import torch  # noqa: PLC0415

    from sparc.common.checkpoint import CheckpointManager  # noqa: PLC0415
    from sparc.losses.tobit import CENSOR_LEFT, CENSOR_NONE, CENSOR_RIGHT, per_sample_loss  # noqa: PLC0415
    from sparc.models.base import BasePredictor  # noqa: PLC0415
    from sparc.train import Trainer, TrainerConfig  # noqa: PLC0415

    bundle = _load_features(config, args, logger, device)
    if bundle is None:
        return 1

    model = BasePredictor(dropout=config.hparams.train.base["dropout"])
    logger.info("Θ_B 参数量 = %d（规范 §8.3.6 要求 57,218）", model.n_parameters())
    assert model.n_parameters() == config.hparams.param_budget["theta_b"], "Θ_B 参数量与规范不符"

    if not args.h1_only:
        trainer_config = TrainerConfig.from_hparams(
            STAGE, config.hparams.train.base, config.hparams.train.amp, args.run_name
        )
        if args.epochs:
            trainer_config.epochs = args.epochs
        trainer = Trainer(
            model, trainer_config, device,
            config.paths.stage_logs(STAGE), config.paths.stage_checkpoints(STAGE),
            trainable_parameters=model.parameters(), config_sha256=config.freeze_manifest(),
        )
        trainer.fit(
            train_step=lambda m, batch: _train_step(m, batch, per_sample_loss),
            train_loader=bundle["train_loader"],
            validate=lambda m: _validate(m, bundle["inner"], per_sample_loss),
            val_metric_key="val_loss",
            extra_state={"pca_ligand": bundle["pca_ligand"].state_dict(),
                         "pca_protein": bundle["pca_protein"].state_dict(),
                         "label_mean": bundle["label_mean"], "label_std": bundle["label_std"]},
        )
        trainer.load_best()

        # ---- S1-F：冻结 Θ_B 并另存独立 checkpoint ----
        model.freeze()
        frozen = CheckpointManager(config.paths.stage_checkpoints("s1_base"), f"{args.run_name}_frozen",
                                   config_sha256=config.freeze_manifest())
        frozen.save(model, epoch=-1, metric=None,
                    extra={"stage": "S1-F", "frozen": True,
                           "pca_ligand": bundle["pca_ligand"].state_dict(),
                           "pca_protein": bundle["pca_protein"].state_dict(),
                           "label_mean": bundle["label_mean"], "label_std": bundle["label_std"]})
        logger.info("S1-F 完成：Θ_B 已冻结并另存 %s", frozen.last_path)
        guard.complete("s1f_freeze", config.freeze_manifest(), artifacts={"frozen": str(frozen.last_path)})

    # ---- H1 检验 ----
    logger.info("== H1 检验：朴素跨域检索在低支持区造成负迁移 ==")
    h1 = _evaluate_h1(config, model, bundle, logger, device)
    path = config.paths.stage_outputs(STAGE) / f"{args.run_name}_h1.json"
    path.write_text(json.dumps(h1, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info("H1 报告：%s", path)

    if not h1.get("passed"):
        logger.error(
            "H1 不成立 ⇒ 无负迁移可控 ⇒ 按 §13.3 止损条件立即执行 §13.5 Plan-B：\n"
            "  转为「基准完整性 + 分析论文」：事实 A（SOTA vs 均值基线）、\n"
            "  转运体面板伪装成 4 个靶点、直系同源泄漏、糖苷通道、79%% 查询自检索、\n"
            "  censored 当点值；交付带四种协议的重构基准。\n"
            "  投稿目标：NeurIPS D&B / J. Cheminformatics / JCIM。**这不是失败。**"
        )

    guard.complete(STAGE, config.freeze_manifest(), artifacts={"h1": str(path)},
                   metrics={"h1_passed": h1.get("passed")})
    return 0


def _load_features(config, args, logger, device) -> Dict[str, Any] | None:
    """载入查询、编码、拟合冻结 PCA、构造 loader。"""
    import csv  # noqa: PLC0415

    import torch  # noqa: PLC0415

    from sparc.common.config import resolve_model_a_entry  # noqa: PLC0415
    from sparc.data.npass import load_available_uniprot  # noqa: PLC0415
    from sparc.models.model_a import build_model_a  # noqa: PLC0415
    from sparc.models.protein import ProteinEncoder  # noqa: PLC0415
    from sparc.models.whitening import FrozenPCAWhitening  # noqa: PLC0415

    query_path = config.paths.stage_outputs("s0_data") / "queries.tsv"
    split_path = config.paths.stage_outputs("s0_data") / f"split_{args.protocol}.json"
    if not query_path.is_file() or not split_path.is_file():
        logger.error("缺少 Stage 0 产物（%s / %s），请先跑 run_s0_data.py", query_path, split_path)
        return None

    with open(query_path, "r", encoding="utf-8", newline="") as handle:
        queries = list(csv.DictReader(handle, delimiter="\t"))
    split_of = json.loads(split_path.read_text(encoding="utf-8"))["split_of"]
    for query in queries:
        query["split"] = split_of.get(query["query_id"], "train")

    # 配体编码
    # §7.2：S1 不给任何默认放行 —— 到这一步 c9/c10 都该已经是 pass 了。
    # c10 由 Stage 0-B 产出，跑完 S0-B 必须把结果回填进 model_registry.yaml。
    entry = resolve_model_a_entry(config.model_registry, args.model_a)
    adapter = build_model_a(entry, device=str(device), cache_dir=config.paths.get("model_a_cache"))
    ligand_raw = adapter.encode_cached([q["smiles"] for q in queries])

    # 蛋白编码（ESM-2 mean-pool，按 accession 缓存）
    sequences = load_available_uniprot(config.paths.get("uniprot_dir") / "npass_targets")
    protein_encoder = ProteinEncoder(config.paths.get("esm2_dir"), device=str(device),
                                     cache_dir=config.paths.get("esm2_cache"))
    protein_vectors = protein_encoder.encode_targets(
        {a: s for a, s in sequences.items() if a in {q["uniprot_id"] for q in queries}}
    )
    protein_raw = np.stack([protein_vectors[q["uniprot_id"]] for q in queries])

    split = np.array([q["split"] for q in queries])
    train_mask = split == "train"

    # **只在 fold-train 上拟合** PCA（§8.2；用全量拟合是一条隐蔽的泄漏通道）
    pca_ligand = FrozenPCAWhitening(config.hparams.dims.d_pca_ligand).fit(ligand_raw[train_mask], "fold_train")
    pca_protein = FrozenPCAWhitening(config.hparams.dims.d_pca_protein).fit(protein_raw[train_mask], "fold_train")
    ligand = torch.from_numpy(pca_ligand.transform(ligand_raw)).float()
    protein = torch.from_numpy(pca_protein.transform(protein_raw)).float()

    y = torch.tensor([float(q["pactivity"]) for q in queries], dtype=torch.float32)
    censor = torch.tensor([{"none": 0, "left": 1, "right": 2}[q["censor_flag"]] for q in queries], dtype=torch.long)

    def subset(mask: np.ndarray) -> Dict[str, Any]:
        """按掩码切一份数据。"""
        idx = torch.from_numpy(np.nonzero(mask)[0])
        return {"ligand": ligand[idx].to(device), "protein": protein[idx].to(device),
                "y": y[idx].to(device), "censor": censor[idx].to(device),
                "queries": [queries[i] for i in idx.tolist()]}

    batch_size = config.hparams.train.base["batch_size"]
    train_data = subset(train_mask)

    class _Loader:
        """按 epoch 重新打乱的极简 loader（数据量小，无需 DataLoader）。"""

        def __iter__(self):
            """每个 epoch 重新打乱后按 batch 产出。"""
            order = torch.randperm(train_data["y"].shape[0], device=device)
            for start in range(0, order.numel(), batch_size):
                idx = order[start:start + batch_size]
                yield {k: v[idx] for k, v in train_data.items() if k != "queries"}

    return {
        "queries": queries, "train_loader": _Loader(),
        "train": train_data, "inner": subset(split == "inner"),
        "calib": subset(split == "calib"), "test": subset(split == "test"),
        "pca_ligand": pca_ligand, "pca_protein": pca_protein,
        "label_mean": float(y[train_mask].mean()), "label_std": float(y[train_mask].std()),
    }


def _train_step(model, batch, loss_fn):
    """一个训练步：Tobit 负对数似然。"""
    out = model(batch["ligand"], batch["protein"])
    loss = loss_fn(out.mu, out.log_var, batch["y"], batch["censor"]).mean()
    return {"total": loss}


def _validate(model, data, loss_fn) -> Dict[str, float]:
    """验证：Tobit 损失 + RMSE。"""
    import torch  # noqa: PLC0415

    model.eval()
    with torch.no_grad():
        out = model(data["ligand"], data["protein"])
        loss = loss_fn(out.mu, out.log_var, data["y"], data["censor"]).mean()
        rmse = torch.sqrt(((out.mu - data["y"]) ** 2).mean())
    model.train()
    return {"val_loss": float(loss), "val_rmse": float(rmse)}


def _evaluate_h1(config, model, bundle, logger, device) -> Dict[str, Any]:
    """H1：低支持子集上 Base + 朴素 kNN 是否严格劣于 Base。"""
    import torch  # noqa: PLC0415

    from sparc.eval.metrics import evaluate_h1  # noqa: PLC0415
    from sparc.losses.tobit import per_sample_loss  # noqa: PLC0415
    from sparc.retrieval.pipeline import naive_knn_residual  # noqa: PLC0415

    prereg = config.prereg.h1
    threshold = prereg["low_support_tanimoto_threshold"]
    data = bundle["inner"]

    model.eval()
    with torch.no_grad():
        out = model(data["ligand"], data["protein"])
        loss_base = per_sample_loss(out.mu, out.log_var, data["y"], data["censor"]).cpu().numpy()

    neighbours = _naive_neighbours(config, data["queries"], logger)
    mu = out.mu.cpu().numpy()
    residuals = np.array([
        naive_knn_residual(n["tanimoto"], n["labels"], mu[i], k=config.hparams.retrieval.k_top)
        for i, n in enumerate(neighbours)
    ])
    top1 = np.array([float(n["tanimoto"].max()) if n["tanimoto"].size else 0.0 for n in neighbours])

    with torch.no_grad():
        mu_naive = out.mu + torch.from_numpy(residuals).float().to(device)
        loss_naive = per_sample_loss(mu_naive, out.log_var, data["y"], data["censor"]).cpu().numpy()

    delta = loss_naive - loss_base
    low_support = top1 < threshold
    logger.info("低支持子集（Top-1 Tanimoto < %.2f）：%d / %d 条", threshold, int(low_support.sum()), len(top1))

    per_target: Dict[str, float] = {}
    target_ids = np.array([q["target_id"] for q in data["queries"]])
    for target_id in sorted(set(target_ids.tolist())):
        mask = (target_ids == target_id) & low_support
        if mask.sum() >= 5:
            per_target[target_id] = float(delta[mask].mean())

    result = evaluate_h1(
        delta[low_support], per_target,
        ci_level=config.prereg.raw["h1"]["ci_level"],
        n_bootstrap=config.prereg.raw["h1"]["bootstrap_n"],
        min_target_agreement=prereg["criteria"]["b_min_target_agreement_frac"],
        min_cohens_d=prereg["criteria"]["c_min_cohens_d"],
        seed=config.effective_seed,
    )
    logger.info("H1 判定：%s —— %s", result.passed, result.note)
    return result.to_dict()


def _naive_neighbours(config, queries, logger) -> List[Dict[str, np.ndarray]]:
    """为 H1 检索朴素 kNN 邻居（同靶点药物记忆库，无重排、无门控）。"""
    import csv  # noqa: PLC0415
    from collections import defaultdict  # noqa: PLC0415

    from sparc.chem.fingerprints import FingerprintCalculator, tanimoto_matrix  # noqa: PLC0415

    memory_path = config.paths.stage_outputs("s0_data") / "memory.tsv"
    with open(memory_path, "r", encoding="utf-8", newline="") as handle:
        memory = list(csv.DictReader(handle, delimiter="\t"))
    by_target = defaultdict(list)
    for record in memory:
        by_target[record["target_id"]].append(record)

    calculator = FingerprintCalculator(config.hparams.retrieval.ecfp_radius,
                                       config.hparams.retrieval.ecfp_n_bits)
    cache: Dict[str, Any] = {}
    out: List[Dict[str, np.ndarray]] = []
    for query in queries:
        group = by_target.get(query["target_id"], [])
        if not group:
            out.append({"tanimoto": np.zeros(0), "labels": np.zeros(0)})
            continue
        if query["target_id"] not in cache:
            fingerprints, ok = calculator.ecfp4_batch([r["smiles"] for r in group])
            labels = np.array([float(group[i]["pactivity"]) for i in ok])
            cache[query["target_id"]] = (fingerprints, labels)
        fingerprints, labels = cache[query["target_id"]]
        query_fp = calculator.ecfp4(query["smiles"])
        if query_fp is None or fingerprints.shape[0] == 0:
            out.append({"tanimoto": np.zeros(0), "labels": np.zeros(0)})
            continue
        similarity = tanimoto_matrix(query_fp[None, :], fingerprints)[0]
        out.append({"tanimoto": similarity, "labels": labels})
    logger.info("朴素 kNN 邻居检索完成：%d 个查询", len(out))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
