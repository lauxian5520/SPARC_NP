"""支持度门控 —— 29 参数 logistic (§8.3.5)。

```
g = σ(w_gᵀ x_g + b_g),   w_g ∈ R^28
g̃ = g · 1[g ≥ λ]
```

**为什么用 logistic 而不是 MLP**：门控只占原参数量的 0.2%，换掉它省不了
多少参数 —— 换的目的是**可解释性**。29 个系数可以直接印进论文，
"什么让药物证据对天然产物可信"变成一张可读的表。这比 MLP 高 0.01 的
AUROC 有价值得多，也是 H2 判据 (d)（系数符号与化学先验一致）
能够存在的前提。

**训练期梯度（这条在 talk.md 中丢失，v1.0 恢复）**：训练阶段使用
**连续 ``g``**，硬阈值 ``λ`` 只在 Stage 4 之后生效。按字面实现
``g̃ = g·1[g≥λ]`` 会让门控从 ``L_task`` 得到的梯度**恒为零**。
:meth:`SupportGate.forward` 的 ``apply_threshold`` 默认为 ``False``
就是这条纪律的代码形式。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from sparc.common.logging_utils import get_logger

_LOGGER = get_logger(__name__)


@dataclass
class GateCoefficientReport:
    """Figure 2（门控系数图）的数据 —— 本项目可解释性主张的全部载体。"""

    names: List[str]
    coefficients: List[float]
    sign_priors: List[int]
    groups: List[str]
    intercept: float
    ci_lower: Optional[List[float]] = None
    ci_upper: Optional[List[float]] = None

    def sign_flips(self) -> List[str]:
        """符号与化学先验冲突的特征名 —— H2 判据 (d)：翻转数 ≤ 2/28。

        Note:
            系数为 0（或 CI 跨 0）不算翻转 —— 那是"没学到"，不是"学反了"。
        """
        flips: List[str] = []
        for name, coef, prior in zip(self.names, self.coefficients, self.sign_priors):
            if coef != 0.0 and np.sign(coef) != np.sign(prior):
                flips.append(name)
        return flips

    def to_dict(self) -> Dict[str, Any]:
        """转成可写报告/画图的字典。"""
        return {
            "intercept": self.intercept,
            "features": [
                {
                    "name": n, "coefficient": c, "sign_prior": s, "group": g,
                    "ci_lower": self.ci_lower[i] if self.ci_lower else None,
                    "ci_upper": self.ci_upper[i] if self.ci_upper else None,
                }
                for i, (n, c, s, g) in enumerate(zip(self.names, self.coefficients, self.sign_priors, self.groups))
            ],
            "sign_flips": self.sign_flips(),
            "n_sign_flips": len(self.sign_flips()),
        }


class SupportGate(nn.Module):
    """29 参数 logistic 门控。"""

    def __init__(self, n_features: int = 28, feature_names: Optional[Sequence[str]] = None) -> None:
        """
        Args:
            n_features: 特征维数（冻结为 28）。
            feature_names: 特征名（来自 ``gate_feature_manifest.yaml``），
                只用于报告与断言，不影响计算。
        """
        super().__init__()
        self.linear = nn.Linear(n_features, 1)      # 28 + 1 = 29
        nn.init.zeros_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 2.0)    # σ(2.0) ≈ 0.88：起点倾向接受检索，
                                                    # 让 S2 早期能积累到有效的 d_i 信号
        self.n_features = n_features
        self.feature_names = list(feature_names) if feature_names else [f"x_g_{i+1}" for i in range(n_features)]
        if len(self.feature_names) != n_features:
            raise ValueError(f"特征名数量 {len(self.feature_names)} != n_features {n_features}")

        # x_g 的标准化统计量（fold-train 拟合后冻结，随 checkpoint 落盘）
        self.register_buffer("feature_mean", torch.zeros(n_features))
        self.register_buffer("feature_std", torch.ones(n_features))
        self.register_buffer("clip_sigma", torch.tensor(5.0))

    # ------------------------------------------------------------------
    def fit_standardizer(self, features: Tensor, clip_sigma: float = 5.0) -> None:
        """在 fold-train 上拟合 x_g 的 z-score 统计量。

        Args:
            features: ``(N, 28)`` fold-train 的门控特征。
            clip_sigma: 标准化后的截断（防止极端支持度值主导 logistic）。
        """
        self.feature_mean.copy_(features.mean(dim=0))
        std = features.std(dim=0)
        # 常数特征（如硬过滤后恒为 1 的 assay_compatible_ratio）std=0，
        # 置 1 使其标准化后恒为 0，系数不可辨识但不会产生 NaN
        self.feature_std.copy_(torch.where(std > 1e-6, std, torch.ones_like(std)))
        self.clip_sigma.fill_(clip_sigma)
        n_dead = int((std <= 1e-6).sum())
        if n_dead:
            dead = [self.feature_names[i] for i in torch.nonzero(std <= 1e-6).flatten().tolist()]
            _LOGGER.warning(
                "门控特征中有 %d 维在 fold-train 上方差为 0（%s）——"
                "manifest 应把它们标为 dead，其系数不可解释", n_dead, dead,
            )

    def standardize(self, features: Tensor) -> Tensor:
        """应用 z-score 标准化并截断。"""
        z = (features - self.feature_mean) / self.feature_std
        return z.clamp(min=-self.clip_sigma, max=self.clip_sigma)

    # ------------------------------------------------------------------
    def forward(
        self,
        gate_features: Tensor,
        apply_threshold: bool = False,
        lam: Optional[float] = None,
        already_standardized: bool = False,
    ) -> Tensor:
        """计算门控值。

        Args:
            gate_features: ``(B, 28)`` 支持度特征。
            apply_threshold: 是否施加硬阈值。**训练期必须为 ``False``**
                （§8.3.5：训练用连续 ``g``，硬阈值只在 Stage 4 之后生效）。
            lam: 阈值 ``λ``，由 Learn-then-Test 给出，**不是超参搜索的结果**。
            already_standardized: ``gate_features`` 是否已标准化。

        Returns:
            ``(B,)``：``apply_threshold=False`` 时为连续 ``g``；
            否则为 ``g̃ = g · 1[g ≥ λ]``。

        Raises:
            ValueError: 要求施加阈值但未提供 ``λ``。
        """
        x = gate_features if already_standardized else self.standardize(gate_features)
        g = torch.sigmoid(self.linear(x).squeeze(-1))
        if not apply_threshold:
            return g
        if lam is None:
            raise ValueError(
                "apply_threshold=True 但未提供 λ。λ 必须来自 Learn-then-Test 标定 (§11)，"
                "不允许在此处即兴取值 —— 那正是 §11.1 要替换掉的做法。"
            )
        return g * (g >= lam).to(g.dtype)

    # ------------------------------------------------------------------
    def coefficient_report(
        self,
        sign_priors: Sequence[int],
        groups: Sequence[str],
        bootstrap_coefficients: Optional[np.ndarray] = None,
        ci_level: float = 0.95,
    ) -> GateCoefficientReport:
        """产出 Figure 2 的数据。

        Args:
            sign_priors: 来自 ``gate_feature_manifest.yaml`` 的化学先验符号。
            groups: 特征分组（着色依据）。
            bootstrap_coefficients: ``(n_bootstrap, 28)``，提供则计算 CI。
            ci_level: 置信水平。

        Returns:
            :class:`GateCoefficientReport`。
        """
        weights = self.linear.weight.detach().cpu().numpy().flatten()
        ci_lower = ci_upper = None
        if bootstrap_coefficients is not None and len(bootstrap_coefficients):
            alpha = (1.0 - ci_level) / 2.0
            ci_lower = np.quantile(bootstrap_coefficients, alpha, axis=0).tolist()
            ci_upper = np.quantile(bootstrap_coefficients, 1.0 - alpha, axis=0).tolist()
        return GateCoefficientReport(
            names=list(self.feature_names),
            coefficients=weights.tolist(),
            sign_priors=list(sign_priors),
            groups=list(groups),
            intercept=float(self.linear.bias.detach().cpu().item()),
            ci_lower=ci_lower, ci_upper=ci_upper,
        )

    def n_parameters(self) -> int:
        """可训练参数量（预算核对：29）。"""
        return sum(p.numel() for p in self.parameters())
