"""模型层。

冻结顺序 (§4.2)::

    Model A（外部，永不训练）
       ↓ 冻结 PCA 白化（fold-train 拟合）
    Θ_B  Base 预测器          Stage 1 训练 → 冻结
       ↓
    Θ_R  SCAR 检索分支         Stage 2 训练（Θ_B 冻结）
       ↓
    Gate 29 参数 logistic      Stage 3 交叉拟合训练
       ↓
    λ    拒检阈值              Stage 4 Learn-then-Test

**惰性导入**：依赖 torch 的符号通过模块级 ``__getattr__`` 按需加载。
本机（树莓派）没有 torch，但仍需 import ``sparc.models.param_budget``
来验证 §8.3.6 的参数预算 —— 参数预算是规范的硬约束，
必须能在写规范的那台机器上验证。
"""

from typing import Any

from sparc.models.whitening import FrozenPCAWhitening
from sparc.models.param_budget import full_budget, supervision_ratio

# 符号名 -> 所属子模块（torch 依赖，按需导入）
_LAZY_EXPORTS = {
    "ModelAAdapter": "model_a", "ModelAFingerprint": "model_a", "MODEL_A_REGISTRY": "model_a",
    "build_model_a": "model_a", "check_output_stability": "model_a",
    "ProteinEncoder": "protein",
    "BasePredictor": "base", "BaseOutput": "base",
    "GraphMatcherLite": "graphmatcher", "MolecularGraph": "graphmatcher",
    "MolecularGraphFeaturizer": "graphmatcher", "log_domain_sinkhorn": "graphmatcher",
    "RerankerLite": "reranker", "RerankOutput": "reranker",
    "EvidenceLite": "evidence", "EvidenceOutput": "evidence",
    "ResidualHead": "residual", "UncertaintyHead": "residual",
    "SupportGate": "gate", "GateCoefficientReport": "gate",
    "SparcNP": "sparc_model", "SparcOutput": "sparc_model",
}

__all__ = ["FrozenPCAWhitening", "full_budget", "supervision_ratio", *sorted(_LAZY_EXPORTS)]


def __getattr__(name: str) -> Any:
    """按需导入依赖 torch 的符号（PEP 562）。"""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'sparc.models' has no attribute '{name}'")
    import importlib  # noqa: PLC0415

    module = importlib.import_module(f"sparc.models.{module_name}")
    return getattr(module, name)
