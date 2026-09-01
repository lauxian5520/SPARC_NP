"""交叉拟合与部署一致性 (§10.3)。

**问题**：``talk.md`` 的交叉拟合效用标签来自 ``f_R^(−j)``（只见 4/5 dev），
却用来给最终的 ``f_R``（见全部 dev）把关 —— 系统性低估检索效用，
导致**过度拒检**。

**v1.0 方案：部署交叉拟合集成本身。**

    R_b(q) = (1/J) Σ_{j=1..J} f_R^(−j)(q)      J = 5

三个好处：
① 效用标签与部署模型分布一致；
② ``Var_j f_R^(−j)(q)`` 成为免费的第 29 维门控特征（若启用，门控参数变 30）；
③ 集成本身降低小样本方差。

**必须报告**：``|M^cf| / |M^deploy|``（交叉拟合视图与部署视图的记忆库
规模比）与两者支持度特征分布的 KS 距离。若 KS > 0.15，说明协变量
偏移过大，需缩小 J。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from sparc.common.logging_utils import get_logger

_LOGGER = get_logger(__name__)


def ks_distance(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """两样本 Kolmogorov–Smirnov 距离（纯 numpy，不依赖 scipy）。

    Args:
        sample_a: 一维样本。
        sample_b: 一维样本。

    Returns:
        ``sup_x |F_A(x) − F_B(x)|``。
    """
    a = np.sort(np.asarray(sample_a, dtype=np.float64))
    b = np.sort(np.asarray(sample_b, dtype=np.float64))
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    return float(np.max(np.abs(cdf_a - cdf_b)))


@dataclass
class CrossFitReport:
    """§10.3 要求必须报告的两项。"""

    n_folds: int
    memory_size_crossfit: float
    memory_size_deploy: float
    ks_by_feature: Dict[str, float] = field(default_factory=dict)
    max_ks_threshold: float = 0.15

    @property
    def memory_ratio(self) -> float:
        """``|M^cf| / |M^deploy|``。"""
        return self.memory_size_crossfit / self.memory_size_deploy if self.memory_size_deploy else float("nan")

    @property
    def max_ks(self) -> float:
        """支持度特征上的最大 KS 距离。"""
        values = [v for v in self.ks_by_feature.values() if np.isfinite(v)]
        return max(values) if values else float("nan")

    @property
    def shift_too_large(self) -> bool:
        """KS > 阈值 ⇒ 协变量偏移过大，需缩小 J。"""
        return bool(np.isfinite(self.max_ks) and self.max_ks > self.max_ks_threshold)

    def to_dict(self) -> Dict[str, Any]:
        """转成可写报告的字典。"""
        return {
            "n_folds": self.n_folds,
            "memory_size_crossfit": self.memory_size_crossfit,
            "memory_size_deploy": self.memory_size_deploy,
            "memory_ratio": round(self.memory_ratio, 4) if np.isfinite(self.memory_ratio) else None,
            "max_ks": round(self.max_ks, 4) if np.isfinite(self.max_ks) else None,
            "max_ks_threshold": self.max_ks_threshold,
            "shift_too_large": self.shift_too_large,
            "ks_by_feature": {k: round(v, 4) for k, v in sorted(self.ks_by_feature.items())},
            "action_if_shift": "缩小 J（§10.3）",
        }


class CrossFitEnsemble:
    """J 折交叉拟合集成 —— **部署的就是这个集成本身**。

    与"训 J 个模型做平均"的常规集成不同，这里的关键是：**用来生成
    效用标签的模型与部署的模型是同一个对象**。``f_R^(−j)`` 只见 4/5 dev，
    如果部署 ``f_R``（见全部 dev），效用标签就会系统性低估检索效用。
    """

    def __init__(self, n_folds: int = 5, seed: int = 42) -> None:
        """
        Args:
            n_folds: J（冻结为 5）。
            seed: 折划分种子。
        """
        self.n_folds = n_folds
        self.seed = seed
        self.models: List[Any] = []
        self.fold_of: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    def assign_folds(self, group_ids: Sequence[str]) -> np.ndarray:
        """按**组**（骨架家族）分折，不按样本分折。

        Args:
            group_ids: ``(N,)`` 每个样本的骨架家族 ID。

        Returns:
            ``(N,)`` 折号。按样本随机分折会让同一骨架家族的分子
            跨折出现 —— 那正是协议 S 要防的泄漏。
        """
        rng = np.random.default_rng(self.seed)
        unique = sorted(set(group_ids))
        shuffled = rng.permutation(len(unique))
        group_fold = {group: int(shuffled[i] % self.n_folds) for i, group in enumerate(unique)}
        self.fold_of = np.array([group_fold[g] for g in group_ids], dtype=np.int64)
        counts = np.bincount(self.fold_of, minlength=self.n_folds)
        _LOGGER.info("交叉拟合分折（按骨架家族）：各折样本数 %s", counts.tolist())
        return self.fold_of

    def fit(
        self,
        train_fn: Callable[[np.ndarray, int], Any],
        n_samples: int,
    ) -> List[Any]:
        """训练 J 个 ``f_R^(−j)``。

        Args:
            train_fn: ``fn(train_mask, fold_index) -> model``。
            n_samples: 样本总数。

        Returns:
            J 个模型。
        """
        if self.fold_of is None:
            raise RuntimeError("请先调用 assign_folds()")
        self.models = []
        for j in range(self.n_folds):
            train_mask = self.fold_of != j
            _LOGGER.info("交叉拟合第 %d/%d 折：训练样本 %d", j + 1, self.n_folds, int(train_mask.sum()))
            self.models.append(train_fn(train_mask, j))
        return self.models

    def predict_ensemble(self, predict_fn: Callable[[Any], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """部署预测：``R_b(q) = (1/J) Σ_j f_R^(−j)(q)``。

        Args:
            predict_fn: ``fn(model) -> (N,) 预测``。

        Returns:
            ``(集成均值, 集成方差)``。方差即 ``Var_j f_R^(−j)(q)``，
            是可选的第 29 维门控特征（启用后门控参数变 30，
            须同步改 ``gate_feature_manifest.yaml`` 的 ``n_features``/``n_params``）。
        """
        if not self.models:
            raise RuntimeError("尚未训练交叉拟合模型")
        predictions = np.stack([predict_fn(model) for model in self.models])   # (J, N)
        return predictions.mean(axis=0), predictions.var(axis=0)

    def oof_predict(self, predict_fn: Callable[[Any, np.ndarray], np.ndarray], n_samples: int) -> np.ndarray:
        """out-of-fold 预测 —— 效用标签由它生成。

        Args:
            predict_fn: ``fn(model, sample_mask) -> 该子集的预测``。
            n_samples: 样本总数。

        Returns:
            ``(N,)`` OOF 预测。
        """
        if self.fold_of is None or not self.models:
            raise RuntimeError("请先 assign_folds() 并 fit()")
        out = np.zeros(n_samples, dtype=np.float64)
        for j, model in enumerate(self.models):
            mask = self.fold_of == j
            out[mask] = predict_fn(model, mask)
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def build_report(
        crossfit_features: np.ndarray,
        deploy_features: np.ndarray,
        feature_names: Sequence[str],
        memory_size_crossfit: float,
        memory_size_deploy: float,
        n_folds: int = 5,
        max_ks_threshold: float = 0.15,
    ) -> CrossFitReport:
        """生成 §10.3 要求的报告。

        Args:
            crossfit_features: ``(N, 28)`` 交叉拟合视图下的支持度特征。
            deploy_features: ``(N, 28)`` 部署视图下的支持度特征。
            feature_names: 28 个特征名。
            memory_size_crossfit: ``|M^cf|`` 均值。
            memory_size_deploy: ``|M^deploy|`` 均值。
            n_folds: J。
            max_ks_threshold: KS 阈值（0.15）。

        Returns:
            :class:`CrossFitReport`。
        """
        ks = {
            name: ks_distance(crossfit_features[:, i], deploy_features[:, i])
            for i, name in enumerate(feature_names)
        }
        report = CrossFitReport(
            n_folds=n_folds,
            memory_size_crossfit=memory_size_crossfit,
            memory_size_deploy=memory_size_deploy,
            ks_by_feature=ks,
            max_ks_threshold=max_ks_threshold,
        )
        if report.shift_too_large:
            _LOGGER.warning(
                "交叉拟合与部署视图的支持度特征分布偏移过大：max KS = %.3f > %.2f ⇒ 需缩小 J（§10.3）",
                report.max_ks, max_ks_threshold,
            )
        return report
