"""硬件设备自动适配（CUDA / MPS / CPU）。

本项目在树莓派上做规范与数据工作，在装有 CUDA 的服务器上训练。
所有训练脚本都通过 :func:`resolve_device` 拿设备，不在业务代码里
写 ``"cuda"`` 字面量。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class DeviceInfo:
    """设备描述，会被完整写进 run manifest 以便复现。"""

    kind: str                    # "cuda" / "mps" / "cpu"
    name: str
    index: Optional[int]
    n_devices: int
    supports_amp: bool
    total_memory_gb: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        """转成可 JSON 序列化的字典。"""
        return {
            "kind": self.kind,
            "name": self.name,
            "index": self.index,
            "n_devices": self.n_devices,
            "supports_amp": self.supports_amp,
            "total_memory_gb": self.total_memory_gb,
        }

    def __str__(self) -> str:
        """人类可读的设备描述（写进日志）。"""
        mem = f", {self.total_memory_gb:.1f} GB" if self.total_memory_gb else ""
        return f"{self.kind}:{self.index if self.index is not None else '-'} [{self.name}{mem}]"


def resolve_device(prefer: str = "auto", index: int = 0) -> tuple[Any, DeviceInfo]:
    """解析并返回 ``(torch.device, DeviceInfo)``。

    Args:
        prefer: ``"auto"``（默认，按 CUDA > MPS > CPU 顺序）、``"cuda"``、
            ``"mps"`` 或 ``"cpu"``。显式指定但不可用时报错而非静默降级 ——
            "以为在 GPU 上跑了两天其实在 CPU 上"是本项目负担不起的错误。
        index: CUDA 设备序号。

    Returns:
        ``(device, info)`` 二元组。

    Raises:
        ImportError: 未安装 torch。
        RuntimeError: 显式请求的设备不可用。
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "训练需要 PyTorch。本机（树莓派）只做规范与数据工作；"
            "请在 CUDA 服务器上 `pip install -r code/requirements.txt`。"
        ) from exc

    has_cuda = torch.cuda.is_available()
    has_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

    if prefer == "auto":
        kind = "cuda" if has_cuda else ("mps" if has_mps else "cpu")
    else:
        kind = prefer
        if kind == "cuda" and not has_cuda:
            raise RuntimeError("显式请求 CUDA 但 torch.cuda.is_available() 为 False —— 拒绝静默降级到 CPU。")
        if kind == "mps" and not has_mps:
            raise RuntimeError("显式请求 MPS 但不可用 —— 拒绝静默降级到 CPU。")

    if kind == "cuda":
        device = torch.device(f"cuda:{index}")
        props = torch.cuda.get_device_properties(index)
        info = DeviceInfo(
            kind="cuda",
            name=props.name,
            index=index,
            n_devices=torch.cuda.device_count(),
            supports_amp=True,
            total_memory_gb=props.total_memory / (1024 ** 3),
        )
    elif kind == "mps":
        device = torch.device("mps")
        info = DeviceInfo("mps", "Apple Silicon GPU", None, 1, False, None)
    else:
        device = torch.device("cpu")
        info = DeviceInfo("cpu", "CPU", None, 1, False, None)
    return device, info
