"""日志与 Monitor 文件落盘。

规范要求：日志与 Monitor 文件统一落盘至 ``data/<模块名>/logs/``。
本模块提供两个东西：

1. :func:`setup_stage_logging` —— 同时输出到控制台与
   ``data/<stage>/logs/<run_name>_<时间戳>.log``，并落盘 run manifest；
2. :class:`MetricMonitor` —— 逐 epoch 追加的 JSONL 指标监视文件，
   便于训练中断后直接看曲线，也便于把多次 run 拼起来比较。
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str) -> logging.Logger:
    """取得一个模块级 logger（不重复添加 handler）。"""
    return logging.getLogger(name)


def setup_stage_logging(
    log_dir: Path,
    stage: str,
    run_name: str = "default",
    level: int = logging.INFO,
    manifest: Optional[Dict[str, Any]] = None,
) -> tuple[logging.Logger, Path]:
    """配置某一阶段的日志输出。

    Args:
        log_dir: ``data/<模块名>/logs/``。
        stage: 阶段名，用作 logger 名前缀。
        run_name: run 名称。
        level: 日志级别。
        manifest: 若提供，则同时落盘 ``<run>_<ts>.manifest.json``。
            **训练脚本应始终提供** —— 没有 manifest 的 run 无法复现。

    Returns:
        ``(logger, log_file_path)``。
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{run_name}_{stamp}.log"

    root = logging.getLogger("sparc")
    root.setLevel(level)
    root.handlers.clear()
    root.propagate = False

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    root.addHandler(stream)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    logger = logging.getLogger(f"sparc.{stage}")
    logger.info("阶段 %s / run %s 日志已建立：%s", stage, run_name, log_file)

    if manifest is not None:
        manifest_path = log_dir / f"{run_name}_{stamp}.manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
        )
        logger.info("run manifest 已落盘：%s", manifest_path)
        logger.info("冻结配置 sha256：%s", json.dumps(manifest.get("config_sha256", {}), ensure_ascii=False))
    return logger, log_file


class MetricMonitor:
    """逐步追加的 JSONL 指标监视器。

    每行一条 JSON 记录，训练中断也不会丢历史；重新开始训练时
    以 ``append`` 模式续写，配合 :class:`~sparc.common.checkpoint.CheckpointManager`
    实现"无缝续训 + 指标不断档"。
    """

    def __init__(self, log_dir: Path, run_name: str, filename: Optional[str] = None) -> None:
        """
        Args:
            log_dir: ``data/<模块名>/logs/``。
            run_name: run 名称。
            filename: 覆盖默认文件名 ``<run_name>_monitor.jsonl``。
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / (filename or f"{run_name}_monitor.jsonl")

    def log(self, step: int, **metrics: Any) -> None:
        """追加一条指标记录。

        Args:
            step: 全局 step 或 epoch。
            **metrics: 任意标量指标。
        """
        record = {"ts": datetime.now().isoformat(timespec="seconds"), "step": step}
        record.update({k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()})
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def read_all(self) -> list[Dict[str, Any]]:
        """读回全部记录（用于 notebook 画曲线）。"""
        if not self.path.is_file():
            return []
        with open(self.path, "r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
