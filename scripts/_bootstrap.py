"""脚本公共引导：把 ``Project/code`` 加入 sys.path 并提供通用 CLI 参数。

所有 ``run_s*.py`` 都从这里起手，保证：
* 无需 ``pip install -e .`` 也能直接 ``python scripts/run_s0_data.py`` 跑；
* 种子、设备、run 名称、阶段账本的处理在所有阶段完全一致。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from sparc.common import load_experiment_config, seed_everything, setup_stage_logging  # noqa: E402
from sparc.common.config import ExperimentConfig  # noqa: E402
from sparc.train import StageGuard  # noqa: E402


def base_parser(stage: str, description: str) -> argparse.ArgumentParser:
    """构造带通用参数的 ArgumentParser。

    Args:
        stage: 阶段名。
        description: 脚本说明。

    Returns:
        已加入通用参数的 parser。
    """
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config-dir", type=str, default=None, help="配置目录（默认 code/configs）")
    parser.add_argument("--run-name", type=str, default=f"{stage}_default", help="run 名称")
    parser.add_argument("--seed", type=int, default=None, help="覆盖冻结种子")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--device-index", type=int, default=0, help="CUDA 设备序号")
    parser.add_argument("--round-id", type=int, default=1, help="实验轮次（§10.4：S5 后要改只能开新轮次）")
    parser.add_argument("--deterministic", action="store_true", help="开启 cuDNN 确定性（S5 建议开启）")
    parser.add_argument("--skip-stage-guard", action="store_true",
                        help="跳过阶段纪律检查 —— 只应在调试时使用，且必须在 run manifest 中说明")
    return parser


def setup(stage: str, args: argparse.Namespace, need_device: bool = True) -> Tuple[ExperimentConfig, Any, Any, StageGuard]:
    """统一的阶段初始化。

    Args:
        stage: 阶段名。
        args: 解析后的命令行参数。
        need_device: 是否需要解析 torch 设备（S0/S4 是纯 CPU 步骤）。

    Returns:
        ``(config, logger, device_or_None, stage_guard)``。
    """
    config = load_experiment_config(
        config_dir=args.config_dir, stage=stage, run_name=args.run_name, seed=args.seed,
        extra={"cli": vars(args)},
    )
    seed = seed_everything(config.effective_seed, deterministic=args.deterministic)

    device = None
    device_info = None
    if need_device:
        from sparc.common.device import resolve_device  # noqa: PLC0415

        device, info = resolve_device(args.device, args.device_index)
        device_info = info.to_dict()

    logger, _ = setup_stage_logging(
        config.paths.stage_logs(stage), stage, args.run_name,
        manifest=config.run_manifest(device_info=device_info),
    )
    logger.info("随机种子 = %d（deterministic=%s）", seed, args.deterministic)
    if device is not None:
        logger.info("设备 = %s", device)

    guard = StageGuard(config.paths.get("stage_root") / "stage_ledger.json", round_id=args.round_id)
    if not args.skip_stage_guard:
        guard.check_can_enter(stage)
    else:
        logger.warning("已跳过阶段纪律检查（--skip-stage-guard）—— 本次 run 的结果不得进入论文")
    return config, logger, device, guard
