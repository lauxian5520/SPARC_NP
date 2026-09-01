"""冻结 PCA 白化 (§7.1, §8.2)。

**这是 Model A 可插拔的全部机制。** ``d_A`` 之后立刻接一个在
fold-train 上拟合的冻结 PCA 白化到 128 维，因此 Θ_B / Θ_R 的
参数量与 Model A 无关 —— 切换 Model A 不改动任何其它模块。

三条纪律：
1. **只在 fold-train 上拟合**。用 dev 全量或全数据拟合 PCA 是一条
   隐蔽的泄漏通道：标定折的协变量分布会通过主成分方向漏进训练。
2. **拟合后冻结**，作为 buffer 存进 checkpoint，不产生梯度。
3. 白化（除以 sqrt(特征值)）而不只是投影 —— 后续 LayerNorm 假定
   输入各维尺度可比。

实现用 numpy 的 SVD，不依赖 sklearn（服务器环境也未必装）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from sparc.common.logging_utils import get_logger

_LOGGER = get_logger(__name__)


@dataclass
class FrozenPCAWhitening:
    """冻结的 PCA 白化变换 ``z = (u − mean) @ components.T / sqrt(eigenvalues)``。"""

    n_components: int
    mean: Optional[np.ndarray] = None            # (d_in,)
    components: Optional[np.ndarray] = None      # (n_components, d_in)
    scale: Optional[np.ndarray] = None           # (n_components,)
    explained_variance_ratio: Optional[np.ndarray] = None
    fitted_on: str = ""                          # 记录拟合数据的描述，写进 manifest
    eps: float = 1e-6

    @property
    def is_fitted(self) -> bool:
        """是否已拟合。"""
        return self.components is not None

    # ------------------------------------------------------------------
    def fit(self, features: np.ndarray, fitted_on: str = "fold_train") -> "FrozenPCAWhitening":
        """在 fold-train 特征上拟合。

        Args:
            features: ``(n_samples, d_in)`` 的 Model A / ESM-2 输出。
            fitted_on: 拟合数据的描述（如 ``"fold3_train"``），写入 manifest。

        Returns:
            ``self``（便于链式调用）。

        Raises:
            ValueError: 样本数少于目标维数 —— 此时白化会把噪声方向放大到
                与信号同量级，必须减小 ``n_components`` 或增加样本。
        """
        x = np.asarray(features, dtype=np.float64)
        n_samples, d_in = x.shape
        if n_samples < self.n_components:
            raise ValueError(
                f"PCA 白化需要至少 {self.n_components} 个样本，当前只有 {n_samples}。"
                "样本数少于目标维数时白化会放大噪声方向 —— 请减小 n_components。"
            )
        self.mean = x.mean(axis=0)
        centered = x - self.mean
        # 经济型 SVD：比协方差特征分解在 d_in 大时更稳
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        k = min(self.n_components, vt.shape[0])
        eigenvalues = (singular[:k] ** 2) / max(n_samples - 1, 1)
        self.components = vt[:k].astype(np.float32)
        self.scale = np.sqrt(eigenvalues + self.eps).astype(np.float32)
        total_variance = float((singular ** 2).sum() / max(n_samples - 1, 1))
        self.explained_variance_ratio = (eigenvalues / total_variance).astype(np.float32) if total_variance > 0 else None
        self.fitted_on = fitted_on

        cumulative = float(self.explained_variance_ratio.sum()) if self.explained_variance_ratio is not None else float("nan")
        _LOGGER.info(
            "PCA 白化拟合完成：%d → %d 维，累计解释方差 %.3f（拟合于 %s，n=%d）",
            d_in, k, cumulative, fitted_on, n_samples,
        )
        if k < self.n_components:
            _LOGGER.warning("实际主成分数 %d < 目标 %d（输入秩不足），下游维度契约将不满足", k, self.n_components)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """应用白化变换。

        Args:
            features: ``(n_samples, d_in)``。

        Returns:
            ``(n_samples, n_components)`` float32。

        Raises:
            RuntimeError: 尚未拟合。
        """
        if not self.is_fitted:
            raise RuntimeError("PCA 白化尚未拟合 —— 必须先在 fold-train 上 fit() 再 transform()")
        x = np.asarray(features, dtype=np.float32)
        return ((x - self.mean.astype(np.float32)) @ self.components.T) / self.scale

    def fit_transform(self, features: np.ndarray, fitted_on: str = "fold_train") -> np.ndarray:
        """拟合并变换（只允许对 fold-train 使用）。"""
        return self.fit(features, fitted_on).transform(features)

    # ------------------------------------------------------------------
    def state_dict(self) -> Dict[str, Any]:
        """导出为可写进 checkpoint 的字典。"""
        if not self.is_fitted:
            raise RuntimeError("未拟合的 PCA 白化不应被保存")
        return {
            "n_components": self.n_components,
            "mean": self.mean,
            "components": self.components,
            "scale": self.scale,
            "explained_variance_ratio": self.explained_variance_ratio,
            "fitted_on": self.fitted_on,
            "eps": self.eps,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> "FrozenPCAWhitening":
        """从 checkpoint 恢复。"""
        obj = cls(n_components=int(state["n_components"]), eps=float(state.get("eps", 1e-6)))
        obj.mean = np.asarray(state["mean"])
        obj.components = np.asarray(state["components"])
        obj.scale = np.asarray(state["scale"])
        evr = state.get("explained_variance_ratio")
        obj.explained_variance_ratio = np.asarray(evr) if evr is not None else None
        obj.fitted_on = state.get("fitted_on", "")
        return obj

    def as_torch_module(self) -> Any:
        """包成一个只含 buffer、无可训练参数的 ``nn.Module``。

        Returns:
            ``nn.Module``，``forward(x) -> whitened``。所有张量注册为
            buffer 而非 Parameter，因此 ``parameters()`` 为空 ——
            这保证它不会污染 §8.3.6 的参数预算核对。
        """
        import torch  # noqa: PLC0415
        from torch import nn  # noqa: PLC0415

        if not self.is_fitted:
            raise RuntimeError("未拟合的 PCA 白化无法转成模块")

        class _Whitening(nn.Module):
            """冻结白化层（无可训练参数）。"""

            def __init__(self, mean: np.ndarray, components: np.ndarray, scale: np.ndarray) -> None:
                """把 PCA 白化参数注册为 buffer（无可训练参数）。"""
                super().__init__()
                self.register_buffer("mean", torch.from_numpy(np.asarray(mean, dtype=np.float32)))
                self.register_buffer("components", torch.from_numpy(np.asarray(components, dtype=np.float32)))
                self.register_buffer("scale", torch.from_numpy(np.asarray(scale, dtype=np.float32)))

            def forward(self, x: "torch.Tensor") -> "torch.Tensor":
                """``(B, d_in) -> (B, n_components)``。"""
                return ((x - self.mean) @ self.components.t()) / self.scale

        return _Whitening(self.mean, self.components, self.scale)
