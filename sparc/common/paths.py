"""路径解析：全项目唯一的相对路径入口。

设计约束（来自项目规范）：
1. 严禁硬编码绝对路径。``configs/paths.yaml`` 中的所有相对路径都以
   **该 YAML 文件所在目录** 为基准点解析，因此把整个 ``Project/``
   目录原样拷到 CUDA 服务器上即可运行，无需改动任何一行配置。
2. 代码与数据绝对隔离：代码只允许写 ``Project/data/`` 与
   ``Project/reports/`` 下的路径，``origin_dataset`` 视为只读。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml

# 本文件位于 Project/code/sparc/common/paths.py
_THIS_FILE = Path(__file__).resolve()
CODE_ROOT: Path = _THIS_FILE.parents[2]          # Project/code
DEFAULT_CONFIG_DIR: Path = CODE_ROOT / "configs"
DEFAULT_PATHS_YAML: Path = DEFAULT_CONFIG_DIR / "paths.yaml"

# 只读源数据目录名 —— 任何写入尝试都会被 PathConfig.assert_writable 拒绝
READ_ONLY_MARKERS = ("origin_dataset", "assets_weight")


@dataclass
class PathConfig:
    """把 ``paths.yaml`` 解析成绝对路径的容器。

    Attributes:
        config_dir: ``paths.yaml`` 所在目录，即所有相对路径的基准点。
        raw: 原始 YAML 字典，保留以便写入 run manifest。
    """

    config_dir: Path
    raw: Dict[str, Any] = field(default_factory=dict)
    _resolved: Dict[str, Path] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, paths_yaml: Path | str | None = None) -> "PathConfig":
        """从 YAML 载入路径配置。

        Args:
            paths_yaml: ``paths.yaml`` 路径；``None`` 时使用包内默认位置。

        Returns:
            解析完成的 :class:`PathConfig`。
        """
        yaml_path = Path(paths_yaml).resolve() if paths_yaml else DEFAULT_PATHS_YAML
        if not yaml_path.is_file():
            raise FileNotFoundError(f"找不到路径配置文件：{yaml_path}")
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        cfg = cls(config_dir=yaml_path.parent, raw=raw)
        cfg._resolve_all()
        return cfg

    def _resolve_all(self) -> None:
        """把 YAML 中所有字符串路径解析为绝对路径并缓存。"""
        for key, value in self.raw.items():
            if isinstance(value, str):
                self._resolved[key] = (self.config_dir / value).resolve()
        for key, value in self.raw.get("stage_dirs", {}).items():
            self._resolved[f"stage:{key}"] = (self.config_dir / value).resolve()

    # ------------------------------------------------------------------
    # 访问器
    # ------------------------------------------------------------------
    def get(self, key: str) -> Path:
        """按 ``paths.yaml`` 的顶层键取绝对路径。"""
        if key not in self._resolved:
            raise KeyError(f"paths.yaml 中没有键 '{key}'（可用键：{sorted(self._resolved)}）")
        return self._resolved[key]

    def file(self, key: str, parent_key: str) -> Path:
        """取 ``files:`` 段中登记的具体文件。

        Args:
            key: ``files`` 段下的键，例如 ``npass_activities``。
            parent_key: 该文件所在目录的顶层键，例如 ``npass_dir``。
        """
        files = self.raw.get("files", {})
        if key not in files:
            raise KeyError(f"paths.yaml 的 files 段中没有键 '{key}'")
        return self.get(parent_key) / files[key]

    def stage_dir(self, stage: str) -> Path:
        """取某个训练阶段的模块数据目录（如 ``s1_base``）。"""
        return self.get(f"stage:{stage}") if f"stage:{stage}" in self._resolved else self.get("stage_root") / stage

    def stage_logs(self, stage: str) -> Path:
        """阶段日志目录 ``data/<模块名>/logs/``，不存在则创建。"""
        return self.ensure_dir(self.stage_dir(stage) / "logs")

    def stage_checkpoints(self, stage: str) -> Path:
        """阶段 checkpoint 目录 ``data/<模块名>/checkpoints/``。"""
        return self.ensure_dir(self.stage_dir(stage) / "checkpoints")

    def stage_outputs(self, stage: str) -> Path:
        """阶段产物目录 ``data/<模块名>/outputs/``。"""
        return self.ensure_dir(self.stage_dir(stage) / "outputs")

    # ------------------------------------------------------------------
    # 写入保护
    # ------------------------------------------------------------------
    @staticmethod
    def assert_writable(path: Path) -> Path:
        """拒绝对只读源数据目录的写入。

        Raises:
            PermissionError: 目标路径落在 ``origin_dataset``/``assets_weight`` 内。
        """
        parts = set(Path(path).resolve().parts)
        hit = parts.intersection(READ_ONLY_MARKERS)
        if hit:
            raise PermissionError(
                f"拒绝写入只读源数据目录（命中 {sorted(hit)}）：{path}\n"
                "§2 规定 origin_dataset 只读；派生产物请写到 data/interim 或 data/processed。"
            )
        return path

    @classmethod
    def ensure_dir(cls, path: Path) -> Path:
        """创建目录（含父目录），并做只读保护检查。"""
        cls.assert_writable(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
