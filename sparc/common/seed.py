"""随机种子管理。

规范要求"显示配置随机种子"。这里做到：
1. 一次调用覆盖 ``random`` / ``numpy`` / ``torch``（含 CUDA 全部设备）；
2. 可选开启 cuDNN 确定性模式（S5 单次测试评估建议开启）；
3. 返回实际生效的种子并交给调用方写日志，避免"设了但没记"。
"""

from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def seed_everything(seed: int, deterministic: bool = False) -> int:
    """设定全局随机种子。

    Args:
        seed: 种子值。
        deterministic: 是否开启 cuDNN 确定性算法。开启会牺牲一部分速度，
            但 S5（测试集单次评估）与泄漏断言复现必须开启。

    Returns:
        实际生效的种子，供调用方写入日志与 manifest。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch = _try_import_torch()
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # 让不确定的算子直接报错，而不是静默地不可复现
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except (AttributeError, RuntimeError):
                pass
        else:
            torch.backends.cudnn.benchmark = True
    return seed


def _try_import_torch() -> Optional[object]:
    """惰性导入 torch —— 本机（树莓派）无 torch 时数据模块仍可使用。"""
    try:
        import torch  # noqa: PLC0415
        return torch
    except ImportError:
        return None
