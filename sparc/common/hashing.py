"""哈希冻结工具。

§10.2 与 §11.1 的核心纪律：预注册文件、冻结超参网格、门控特征清单
都必须带 sha256 写入每一次 run 的 manifest。没有 hash 的"冻结"不是冻结。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


def file_sha256(path: Path | str, chunk_size: int = 1 << 20) -> str:
    """按块计算文件 sha256（支持 GB 级文件）。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_sha256(obj: Any) -> str:
    """对任意可 JSON 序列化对象做与键顺序无关的 sha256。"""
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_freeze_manifest(config_paths: Dict[str, Path]) -> Dict[str, str]:
    """为一组冻结配置文件生成 ``{名称: sha256}`` 清单。

    Args:
        config_paths: 例如 ``{"preregistration": .../preregistration.yaml}``。

    Returns:
        名称到 sha256 的映射；文件缺失时值为 ``"MISSING"``（由调用方判定是否致命）。
    """
    return {
        name: (file_sha256(path) if Path(path).is_file() else "MISSING")
        for name, path in config_paths.items()
    }
