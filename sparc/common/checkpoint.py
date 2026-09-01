"""断点续训（Checkpointing）。

规范要求：训练逻辑必须实现自动保存与无缝加载 checkpoint
（模型权重、优化器状态及 epoch 数）。本模块在此基础上额外保存：

* ``scaler_state``  —— 混合精度 GradScaler，续训后 loss scale 不回退；
* ``scheduler_state``
* ``rng_state``     —— python/numpy/torch/cuda 全部 RNG，保证续训后
  数据顺序与 dropout 掩码可复现；
* ``config_sha256`` —— 冻结配置指纹。**载入时指纹不一致会报错**，
  防止"改了 frozen_hparams 又接着旧 checkpoint 训"这种静默污染。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from sparc.common.logging_utils import get_logger

_LOGGER = get_logger(__name__)


@dataclass
class ResumeState:
    """续训时回填给训练循环的状态。"""

    start_epoch: int
    global_step: int
    best_metric: float
    extra: Dict[str, Any]


class CheckpointManager:
    """管理某一阶段某一 run 的 checkpoint 目录。

    目录布局::

        data/<模块名>/checkpoints/<run_name>/
            last.pt          # 每个 epoch 覆盖写，用于续训
            best.pt          # 主指标最优，用于下一阶段冻结载入
            epoch_0042.pt    # 可选的周期性快照
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        run_name: str,
        keep_every: int = 0,
        greater_is_better: bool = False,
        config_sha256: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Args:
            checkpoint_dir: ``data/<模块名>/checkpoints/``。
            run_name: run 名称，作为子目录名。
            keep_every: 每 N 个 epoch 额外存一份带编号快照；0 表示不存。
            greater_is_better: 主指标是否越大越好（RMSE 类为 False，AUROC 类为 True）。
            config_sha256: 冻结配置指纹，写入并在续训时校验。
        """
        self.dir = Path(checkpoint_dir) / run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.keep_every = keep_every
        self.greater_is_better = greater_is_better
        self.config_sha256 = dict(config_sha256 or {})
        self.best_metric = -float("inf") if greater_is_better else float("inf")

    # ------------------------------------------------------------------
    @property
    def last_path(self) -> Path:
        """``last.pt`` 路径。"""
        return self.dir / "last.pt"

    @property
    def best_path(self) -> Path:
        """``best.pt`` 路径。"""
        return self.dir / "best.pt"

    def is_improvement(self, metric: float) -> bool:
        """判断指标是否优于当前最优。"""
        return metric > self.best_metric if self.greater_is_better else metric < self.best_metric

    # ------------------------------------------------------------------
    def save(
        self,
        model: Any,
        optimizer: Any = None,
        epoch: int = 0,
        global_step: int = 0,
        metric: Optional[float] = None,
        scheduler: Any = None,
        scaler: Any = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """保存 checkpoint 到 ``last.pt``，必要时同步 ``best.pt``。

        Args:
            model: ``nn.Module``。
            optimizer: 优化器；``None`` 表示不保存优化器状态。
            epoch: 已完成的 epoch 数（续训从 ``epoch + 1`` 开始）。
            global_step: 全局 step。
            metric: 主监控指标；提供时用于判定 best。
            scheduler: 学习率调度器。
            scaler: ``torch.amp.GradScaler``。
            extra: 额外落盘的自由字段（如 PCA 白化参数、标准化统计量）。

        Returns:
            ``last.pt`` 的路径。
        """
        import torch  # noqa: PLC0415

        payload: Dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "metric": metric,
            "best_metric": self.best_metric,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
            "rng_state": self._collect_rng_state(),
            "config_sha256": self.config_sha256,
            "extra": extra or {},
        }
        tmp = self.last_path.with_suffix(".pt.tmp")
        torch.save(payload, tmp)
        tmp.replace(self.last_path)          # 原子替换，避免写一半被中断

        if metric is not None and self.is_improvement(metric):
            self.best_metric = metric
            payload["best_metric"] = metric
            torch.save(payload, self.best_path)
            _LOGGER.info("epoch %d：主指标 %.6f 刷新最优，已写 best.pt", epoch, metric)

        if self.keep_every and epoch % self.keep_every == 0:
            torch.save(payload, self.dir / f"epoch_{epoch:04d}.pt")
        return self.last_path

    # ------------------------------------------------------------------
    def load_for_resume(
        self,
        model: Any,
        optimizer: Any = None,
        scheduler: Any = None,
        scaler: Any = None,
        map_location: Any = "cpu",
        strict_config: bool = True,
        restore_rng: bool = True,
    ) -> Optional[ResumeState]:
        """若存在 ``last.pt`` 则无缝续训，否则返回 ``None``（从头训）。

        Args:
            model: 待载入权重的模型。
            optimizer: 待载入状态的优化器。
            scheduler: 待载入状态的调度器。
            scaler: 待载入状态的 GradScaler。
            map_location: ``torch.load`` 的设备映射。
            strict_config: 冻结配置指纹不一致时是否报错。**默认 True**，
                因为"改了冻结超参又接着旧 checkpoint 训"是无法被审计的污染。
            restore_rng: 是否恢复 RNG 状态。

        Returns:
            :class:`ResumeState` 或 ``None``。
        """
        if not self.last_path.is_file():
            _LOGGER.info("未发现 checkpoint（%s），从头开始训练", self.last_path)
            return None

        import torch  # noqa: PLC0415

        payload = torch.load(self.last_path, map_location=map_location, weights_only=False)
        self._check_config(payload.get("config_sha256", {}), strict_config)

        model.load_state_dict(payload["model_state"])
        if optimizer is not None and payload.get("optimizer_state") is not None:
            optimizer.load_state_dict(payload["optimizer_state"])
        if scheduler is not None and payload.get("scheduler_state") is not None:
            scheduler.load_state_dict(payload["scheduler_state"])
        if scaler is not None and payload.get("scaler_state") is not None:
            scaler.load_state_dict(payload["scaler_state"])
        if restore_rng and payload.get("rng_state"):
            self._restore_rng_state(payload["rng_state"])

        self.best_metric = payload.get("best_metric", self.best_metric)
        state = ResumeState(
            start_epoch=int(payload["epoch"]) + 1,
            global_step=int(payload.get("global_step", 0)),
            best_metric=float(self.best_metric),
            extra=payload.get("extra", {}),
        )
        _LOGGER.info(
            "已从 %s 续训：epoch %d 起，global_step=%d，best=%.6f",
            self.last_path, state.start_epoch, state.global_step, state.best_metric,
        )
        return state

    def load_weights_only(self, model: Any, path: Optional[Path] = None, map_location: Any = "cpu") -> Dict[str, Any]:
        """只载入权重（用于 S1-F 之后把 Θ_B 冻结着交给 S2）。

        Args:
            model: 目标模型。
            path: checkpoint 路径；``None`` 时用 ``best.pt``（不存在则 ``last.pt``）。
            map_location: 设备映射。

        Returns:
            checkpoint 的 ``extra`` 字段（含 PCA 白化等冻结参数）。
        """
        import torch  # noqa: PLC0415

        target = Path(path) if path else (self.best_path if self.best_path.is_file() else self.last_path)
        if not target.is_file():
            raise FileNotFoundError(f"找不到 checkpoint：{target}")
        payload = torch.load(target, map_location=map_location, weights_only=False)
        model.load_state_dict(payload["model_state"])
        _LOGGER.info("已从 %s 载入权重（epoch=%s, metric=%s）", target, payload.get("epoch"), payload.get("metric"))
        return payload.get("extra", {})

    # ------------------------------------------------------------------
    def _check_config(self, saved: Dict[str, str], strict: bool) -> None:
        """校验冻结配置指纹。"""
        if not saved or not self.config_sha256:
            return
        mismatched = {k: (saved.get(k), v) for k, v in self.config_sha256.items() if saved.get(k) != v}
        if not mismatched:
            return
        msg = (
            "checkpoint 的冻结配置指纹与当前配置不一致，续训会产生无法审计的混合结果：\n"
            + "\n".join(f"  {k}: checkpoint={old} 当前={new}" for k, (old, new) in mismatched.items())
        )
        if strict:
            raise RuntimeError(msg + "\n如确认要接着训，请显式传 strict_config=False 并在 run manifest 中说明。")
        _LOGGER.warning(msg)

    @staticmethod
    def _collect_rng_state() -> Dict[str, Any]:
        """收集 python/numpy/torch/cuda 的 RNG 状态。"""
        import torch  # noqa: PLC0415

        state: Dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state: Dict[str, Any]) -> None:
        """恢复 RNG 状态。CUDA 设备数变化时跳过 cuda 部分并告警。"""
        import torch  # noqa: PLC0415

        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"].cpu() if hasattr(state["torch"], "cpu") else state["torch"])
        if "cuda" in state and torch.cuda.is_available():
            if len(state["cuda"]) == torch.cuda.device_count():
                torch.cuda.set_rng_state_all(state["cuda"])
            else:
                _LOGGER.warning("CUDA 设备数与 checkpoint 不一致，跳过 CUDA RNG 恢复")
