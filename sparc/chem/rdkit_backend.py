"""RDKit 惰性导入网关。

本项目在两台机器上跑：
* 树莓派 aarch64 —— 只做规范、数据盘点与精确算术分析，**没有 RDKit**；
* CUDA 服务器 —— 跑完整管线。

因此所有 RDKit 调用都必须经过本模块，且给出可执行的错误信息，
而不是在 import 时炸掉整个包。
"""

from __future__ import annotations

from typing import Any

_RDKIT_HINT = (
    "本步骤需要 RDKit，但当前环境没有安装。\n"
    "  · 树莓派(aarch64)：不要在此运行化学管线；本机只做规范与数据盘点。\n"
    "  · CUDA 服务器：conda install -c conda-forge rdkit>=2024.03 "
    "（或 pip install rdkit）后重试。"
)

try:  # pragma: no cover - 取决于运行环境
    from rdkit import Chem  # noqa: F401
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")   # 关掉 RDKit 的解析告警刷屏
    RDKIT_AVAILABLE = True
except ImportError:  # pragma: no cover
    RDKIT_AVAILABLE = False


def require_rdkit() -> Any:
    """返回 ``rdkit.Chem`` 模块，不可用时抛出带指引的错误。

    Returns:
        ``rdkit.Chem`` 模块对象。

    Raises:
        ImportError: 环境中没有 RDKit。
    """
    if not RDKIT_AVAILABLE:
        raise ImportError(_RDKIT_HINT)
    from rdkit import Chem  # noqa: PLC0415

    return Chem
