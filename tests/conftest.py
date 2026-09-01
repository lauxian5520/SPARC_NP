"""pytest 公共夹具。

**测试的分层**：本项目在两台机器上跑，因此测试也分两层：
* 无标记的测试：纯 Python/numpy，树莓派上就能跑，覆盖泄漏断言、
  划分协议、参数预算、LTT、SafeCoverage 等**规范正确性**；
* 带 ``requires_torch`` / ``requires_rdkit`` 标记的测试：只在服务器上跑，
  覆盖张量形状、退化等价性、糖苷夹具等**实现正确性**。

CI 的红灯标准：无标记测试**必须**全绿；带标记的测试在服务器上必须全绿。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from sparc.common import load_experiment_config  # noqa: E402


def _has(module: str) -> bool:
    """检查某个可选依赖是否可用。"""
    try:
        __import__(module)
        return True
    except ImportError:
        return False


HAS_TORCH = _has("torch")
HAS_RDKIT = _has("rdkit")

requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="需要 PyTorch（在 CUDA 服务器上运行）")
requires_rdkit = pytest.mark.skipif(not HAS_RDKIT, reason="需要 RDKit（在 CUDA 服务器上运行）")


@pytest.fixture(scope="session")
def config():
    """载入冻结配置（三份 YAML + 路径 + 模型注册表）。"""
    return load_experiment_config(stage="test", run_name="pytest")


@pytest.fixture(scope="session")
def dims(config):
    """维度契约。"""
    return config.hparams.dims


@pytest.fixture(scope="session")
def code_root() -> Path:
    """``Project/code`` 绝对路径。"""
    return CODE_ROOT
