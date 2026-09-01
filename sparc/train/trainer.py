"""通用训练循环，含自动断点续训与混合精度 (§10.4)。

规范要求：
* 训练逻辑必须实现自动保存与无缝加载 checkpoints（模型权重、优化器状态、epoch 数）；
* 自动适配硬件（CUDA / MPS / CPU）；显式配置随机种子；
* 日志与 Monitor 文件统一落盘至 ``data/<模块名>/logs/``。

本 Trainer 在此之上还做两件与本项目纪律直接相关的事：

1. **每个 epoch 结束后跑一次退化断言**（可配置频率）。``g̃=0`` 必须精确
   退化回 Base —— 这是"拒检"概念的定义本身，不能等到训练完才发现它坏了。
2. **混合精度只用于训练**；退化断言强制在 fp32 下执行 (§8.3.4)。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch import nn

from sparc.common.checkpoint import CheckpointManager
from sparc.common.logging_utils import MetricMonitor, get_logger

_LOGGER = get_logger(__name__)


@dataclass
class TrainerConfig:
    """训练配置（全部来自 ``frozen_hparams.yaml``，不在代码里写死）。"""

    stage: str
    run_name: str = "default"
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 5.0
    early_stop_patience: int = 30
    amp_enabled: bool = True
    greater_is_better: bool = False
    log_every: int = 10
    checkpoint_every: int = 0
    fallback_check_every: int = 1      # 每 N 个 epoch 跑一次 g̃=0 退化断言
    scheduler: str = "cosine"          # "cosine" / "plateau" / "none"
    min_lr: float = 1e-6

    @classmethod
    def from_hparams(cls, stage: str, section: Dict[str, Any], amp: Dict[str, Any], run_name: str = "default",
                     greater_is_better: bool = False) -> "TrainerConfig":
        """从 ``frozen_hparams.yaml`` 的某个 ``train.*`` 段构造。

        Args:
            stage: 阶段名。
            section: ``hparams.train.base`` / ``.retrieval`` / ``.gate``。
            amp: ``hparams.train.amp``。
            run_name: run 名称。
            greater_is_better: 主指标方向。

        Returns:
            :class:`TrainerConfig`。
        """
        return cls(
            stage=stage, run_name=run_name,
            epochs=int(section.get("epochs", 100)),
            lr=float(section.get("lr", 1e-3)),
            weight_decay=float(section.get("weight_decay", 1e-4)),
            grad_clip=float(section.get("grad_clip", 5.0)),
            early_stop_patience=int(section.get("early_stop_patience", 30)),
            amp_enabled=bool(amp.get("enabled", True)),
            greater_is_better=greater_is_better,
        )


@dataclass
class EpochResult:
    """一个 epoch 的结果。"""

    epoch: int
    train_loss: float
    val_metric: float
    lr: float
    seconds: float
    extra: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转成可写 Monitor 的字典。"""
        return {"epoch": self.epoch, "train_loss": self.train_loss, "val_metric": self.val_metric,
                "lr": self.lr, "seconds": round(self.seconds, 2), **self.extra}


class Trainer:
    """带断点续训的训练循环。"""

    def __init__(
        self,
        model: nn.Module,
        config: TrainerConfig,
        device: torch.device,
        log_dir: Path,
        checkpoint_dir: Path,
        trainable_parameters: Optional[Iterable[nn.Parameter]] = None,
        config_sha256: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Args:
            model: 待训练模型。
            config: 训练配置。
            device: 设备（由 :func:`~sparc.common.device.resolve_device` 给出）。
            log_dir: ``data/<模块名>/logs/``。
            checkpoint_dir: ``data/<模块名>/checkpoints/``。
            trainable_parameters: 本阶段应训练的参数；``None`` 时取全部
                ``requires_grad`` 的参数。分阶段冻结见
                :meth:`~sparc.models.sparc_model.SparcNP.trainable_parameters`。
            config_sha256: 冻结配置指纹，写入 checkpoint 并在续训时校验。
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.monitor = MetricMonitor(log_dir, config.run_name)

        params = list(trainable_parameters) if trainable_parameters is not None else [
            p for p in model.parameters() if p.requires_grad
        ]
        if not params:
            raise ValueError(f"阶段 '{config.stage}' 没有任何可训练参数 —— 检查冻结顺序（§4.2）")
        self.optimizer = torch.optim.AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
        self.scheduler = self._build_scheduler()
        self.scaler = (
            torch.amp.GradScaler(device.type)
            if (config.amp_enabled and device.type == "cuda") else None
        )
        self.checkpoints = CheckpointManager(
            checkpoint_dir, config.run_name,
            keep_every=config.checkpoint_every,
            greater_is_better=config.greater_is_better,
            config_sha256=config_sha256,
        )
        self.n_trainable = sum(p.numel() for p in params)
        _LOGGER.info(
            "Trainer 就绪：阶段 %s / run %s / 设备 %s / 可训练参数 %d / AMP %s",
            config.stage, config.run_name, device, self.n_trainable, "on" if self.scaler else "off",
        )

    # ------------------------------------------------------------------
    def _build_scheduler(self) -> Optional[Any]:
        """构建学习率调度器。"""
        if self.config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.epochs, eta_min=self.config.min_lr
            )
        if self.config.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max" if self.config.greater_is_better else "min",
                patience=max(self.config.early_stop_patience // 3, 3), min_lr=self.config.min_lr,
            )
        return None

    # ------------------------------------------------------------------
    def fit(
        self,
        train_step: Callable[[nn.Module, Any], Dict[str, torch.Tensor]],
        train_loader: Iterable[Any],
        validate: Callable[[nn.Module], Dict[str, float]],
        val_metric_key: str = "val_loss",
        fallback_check: Optional[Callable[[nn.Module], None]] = None,
        extra_state: Optional[Dict[str, Any]] = None,
    ) -> List[EpochResult]:
        """训练主循环。

        Args:
            train_step: ``fn(model, batch) -> {"total": 标量损失, ...}``。
            train_loader: 可多次迭代的数据加载器。
            validate: ``fn(model) -> {指标名: 值}``。
            val_metric_key: 用于 early stopping 与 best checkpoint 的指标键。
            fallback_check: ``fn(model)``，退化断言 (§8.3.4)。**S2 起必须提供**；
                不提供时会记一条警告，因为"忘了跑断言"与"断言通过"在日志上
                不该长得一样。
            extra_state: 额外写进 checkpoint 的字段（PCA 白化参数、标准化统计量等）。

        Returns:
            每个 epoch 的结果列表。
        """
        resume = self.checkpoints.load_for_resume(
            self.model, self.optimizer, self.scheduler, self.scaler, map_location=self.device
        )
        start_epoch = resume.start_epoch if resume else 0
        global_step = resume.global_step if resume else 0
        best_metric = resume.best_metric if resume else (
            -float("inf") if self.config.greater_is_better else float("inf")
        )
        patience = 0
        history: List[EpochResult] = []

        if fallback_check is None and self.config.stage.startswith("s2"):
            _LOGGER.warning(
                "阶段 %s 未提供 fallback_check —— §8.3.4 的 g̃=0 退化断言不会被执行。"
                "这不是可选项：'拒检'的定义依赖于它精确成立。", self.config.stage,
            )

        for epoch in range(start_epoch, self.config.epochs):
            tic = time.time()
            self.model.train()
            losses: List[float] = []
            components: Dict[str, List[float]] = {}

            for batch in train_loader:
                self.optimizer.zero_grad(set_to_none=True)
                if self.scaler is not None:
                    with torch.amp.autocast(self.device.type):
                        out = train_step(self.model, batch)
                    self.scaler.scale(out["total"]).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for group in self.optimizer.param_groups for p in group["params"]],
                        self.config.grad_clip,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    out = train_step(self.model, batch)
                    out["total"].backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for group in self.optimizer.param_groups for p in group["params"]],
                        self.config.grad_clip,
                    )
                    self.optimizer.step()

                losses.append(float(out["total"].detach().item()))
                for key, value in out.items():
                    if key != "total" and torch.is_tensor(value) and value.numel() == 1:
                        components.setdefault(key, []).append(float(value.item()))
                global_step += 1

            train_loss = float(np.mean(losses)) if losses else float("nan")
            metrics = validate(self.model)
            val_metric = float(metrics.get(val_metric_key, float("nan")))

            if self.scheduler is not None:
                if self.config.scheduler == "plateau":
                    self.scheduler.step(val_metric)
                else:
                    self.scheduler.step()

            # §8.3.4 退化断言：训练中就要跑，不能等训练完
            if fallback_check is not None and self.config.fallback_check_every > 0 \
                    and epoch % self.config.fallback_check_every == 0:
                fallback_check(self.model)

            result = EpochResult(
                epoch=epoch, train_loss=train_loss, val_metric=val_metric,
                lr=float(self.optimizer.param_groups[0]["lr"]),
                seconds=time.time() - tic,
                extra={**{k: float(np.mean(v)) for k, v in components.items()},
                       **{k: float(v) for k, v in metrics.items() if k != val_metric_key}},
            )
            history.append(result)
            self.monitor.log(step=epoch, **result.to_dict())

            self.checkpoints.save(
                self.model, self.optimizer, epoch=epoch, global_step=global_step,
                metric=val_metric, scheduler=self.scheduler, scaler=self.scaler,
                extra=extra_state,
            )

            improved = (val_metric > best_metric) if self.config.greater_is_better else (val_metric < best_metric)
            if improved:
                best_metric = val_metric
                patience = 0
            else:
                patience += 1

            if epoch % self.config.log_every == 0 or improved:
                _LOGGER.info(
                    "epoch %3d | train %.5f | %s %.5f | lr %.2e | %.1fs%s",
                    epoch, train_loss, val_metric_key, val_metric,
                    result.lr, result.seconds, "  ← best" if improved else "",
                )

            if patience >= self.config.early_stop_patience:
                _LOGGER.info("early stopping：%d 个 epoch 未改善（best %s = %.5f）",
                             patience, val_metric_key, best_metric)
                break

        _LOGGER.info("阶段 %s 训练结束：best %s = %.5f，checkpoint 在 %s",
                     self.config.stage, val_metric_key, best_metric, self.checkpoints.dir)
        return history

    # ------------------------------------------------------------------
    def load_best(self) -> Dict[str, Any]:
        """载入 best checkpoint（S1-F 冻结、S2 起始都用它）。"""
        return self.checkpoints.load_weights_only(self.model, map_location=self.device)
