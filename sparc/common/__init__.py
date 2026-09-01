"""通用基础设施：配置、路径、设备、随机种子、日志、断点续训、哈希冻结。"""

from sparc.common.config import (
    DataConfig,
    ExperimentConfig,
    GateFeatureManifest,
    PathConfig,
    ModelAEligibilityError,
    PreregistrationConfig,
    load_experiment_config,
    resolve_model_a_entry,
)
from sparc.common.device import DeviceInfo, resolve_device
from sparc.common.checkpoint import CheckpointManager
from sparc.common.logging_utils import get_logger, setup_stage_logging
from sparc.common.seed import seed_everything
from sparc.common.hashing import file_sha256, object_sha256

__all__ = [
    "DataConfig",
    "ExperimentConfig",
    "GateFeatureManifest",
    "PathConfig",
    "ModelAEligibilityError",
    "PreregistrationConfig",
    "load_experiment_config",
    "resolve_model_a_entry",
    "DeviceInfo",
    "resolve_device",
    "CheckpointManager",
    "get_logger",
    "setup_stage_logging",
    "seed_everything",
    "file_sha256",
    "object_sha256",
]
