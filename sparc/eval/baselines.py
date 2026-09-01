"""基线预测器 (§7.4, §12.2)。

**均值预测器必须进入每一张主表**（事实 A）。这不是形式要求：

| 靶点 | 均值基线 RMSE | NaFM RMSE | 隐含 R² |
|---|---:|---:|---:|
| PTP-1B | **0.627** | 0.8243 | **−0.73** |
| AChE (人) | **1.099** | 1.1227 | −0.04 |
| COX-2 (人) | **0.854** | 0.9239 | −0.17 |
| HIV-1 RT | **1.037** | 1.0802 | −0.09 |

已发表 SOTA 在 8 个靶点中有 4 个跑不过"预测训练集均值"；
ECFP 在 8 个靶点上**全部**劣于均值基线。缺了这一行的表，
读者无法判断任何数字是否有意义。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from sparc.common.logging_utils import get_logger
from sparc.eval.metrics import r2_score, rmse, spearman

_LOGGER = get_logger(__name__)


class MeanPredictor:
    """训练集均值预测器 —— 平凡基线，也是本项目最重要的参照系。"""

    def __init__(self, statistic: str = "mean") -> None:
        """
        Args:
            statistic: ``"mean"`` 或 ``"median"``。
        """
        self.statistic = statistic
        self.value_: Optional[float] = None
        self.per_target_: Dict[str, float] = {}

    def fit(self, y: np.ndarray, target_ids: Optional[Sequence[str]] = None) -> "MeanPredictor":
        """在训练集上拟合。

        Args:
            y: ``(N,)`` 训练标签。
            target_ids: ``(N,)`` 靶点 ID；提供时**按靶点分别拟合**。
                这是正确的口径：事实 A 表里的均值基线是每个靶点各自
                的训练均值，不是全局均值。

        Returns:
            ``self``。
        """
        y = np.asarray(y, dtype=np.float64)
        func = np.mean if self.statistic == "mean" else np.median
        self.value_ = float(func(y)) if y.size else 0.0
        if target_ids is not None:
            for tid in sorted(set(target_ids)):
                mask = np.array([t == tid for t in target_ids])
                if mask.any():
                    self.per_target_[tid] = float(func(y[mask]))
        return self

    def predict(self, n: int, target_ids: Optional[Sequence[str]] = None) -> np.ndarray:
        """预测。

        Args:
            n: 样本数。
            target_ids: ``(n,)`` 靶点 ID；提供时用对应靶点的训练均值，
                未见过的靶点回退到全局均值。

        Returns:
            ``(n,)``。
        """
        if self.value_ is None:
            raise RuntimeError("MeanPredictor 尚未 fit()")
        if target_ids is None:
            return np.full(n, self.value_, dtype=np.float64)
        return np.array([self.per_target_.get(t, self.value_) for t in target_ids], dtype=np.float64)


class EcfpRandomForest:
    """ECFP4 + RandomForest —— §7.4 的必备非神经基线。

    需要 scikit-learn。**不提供 numpy 手写替代** —— 基线跑不起来时
    应该报错，而不是悄悄换成一个没人能复现的东西。
    """

    def __init__(self, n_estimators: int = 500, max_depth: Optional[int] = None, seed: int = 42, n_jobs: int = -1) -> None:
        """
        Args:
            n_estimators: 树数。
            max_depth: 最大深度。
            seed: 随机种子。
            n_jobs: 并行度。
        """
        self.params = {"n_estimators": n_estimators, "max_depth": max_depth,
                       "random_state": seed, "n_jobs": n_jobs}
        self.model = None

    def fit(self, fingerprints: np.ndarray, y: np.ndarray) -> "EcfpRandomForest":
        """拟合。"""
        try:
            from sklearn.ensemble import RandomForestRegressor  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "ECFP4+RF 基线需要 scikit-learn（§7.4 列为必备基线）。"
                "请 pip install scikit-learn；不要用别的东西顶替这一行。"
            ) from exc
        self.model = RandomForestRegressor(**self.params)
        self.model.fit(np.asarray(fingerprints, dtype=np.float32), np.asarray(y, dtype=np.float64))
        return self

    def predict(self, fingerprints: np.ndarray) -> np.ndarray:
        """预测。"""
        if self.model is None:
            raise RuntimeError("EcfpRandomForest 尚未 fit()")
        return self.model.predict(np.asarray(fingerprints, dtype=np.float32))


class EcfpSVR:
    """ECFP4 + SVR —— §7.4 的可选非神经基线。"""

    def __init__(self, c: float = 10.0, epsilon: float = 0.1, gamma: str = "scale") -> None:
        """
        Args:
            c: 正则化强度。
            epsilon: ε-不敏感带宽。
            gamma: RBF 核宽度。
        """
        self.params = {"C": c, "epsilon": epsilon, "gamma": gamma}
        self.model = None

    def fit(self, fingerprints: np.ndarray, y: np.ndarray) -> "EcfpSVR":
        """拟合。"""
        from sklearn.svm import SVR  # noqa: PLC0415

        self.model = SVR(**self.params)
        self.model.fit(np.asarray(fingerprints, dtype=np.float32), np.asarray(y, dtype=np.float64))
        return self

    def predict(self, fingerprints: np.ndarray) -> np.ndarray:
        """预测。"""
        if self.model is None:
            raise RuntimeError("EcfpSVR 尚未 fit()")
        return self.model.predict(np.asarray(fingerprints, dtype=np.float32))


@dataclass
class BaselineRow:
    """一个基线在一个靶点上的结果。"""

    name: str
    target_id: str
    rmse: float
    r2: float
    spearman: float
    n_test: int
    delta_rmse_vs_mean: float = float("nan")

    def to_dict(self) -> Dict[str, Any]:
        """转成表格行。"""
        return {
            "baseline": self.name, "target_id": self.target_id,
            "rmse": round(self.rmse, 4), "r2": round(self.r2, 4),
            "spearman": round(self.spearman, 4), "n_test": self.n_test,
            "delta_rmse_vs_mean": round(self.delta_rmse_vs_mean, 4),
        }


def evaluate_baselines(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    target_ids: Sequence[str],
    mean_baseline_name: str = "mean_predictor",
) -> List[BaselineRow]:
    """按靶点评估一组基线，并计算相对均值基线的 ΔRMSE。

    Args:
        predictions: ``{基线名: (N,) 预测}``。
        y_true: ``(N,)``。
        target_ids: ``(N,)``。
        mean_baseline_name: 均值基线的键名。

    Returns:
        逐 (基线, 靶点) 的结果行 + 每个基线的 ``macro`` 汇总行。

    Raises:
        ValueError: 缺少均值基线 —— §12.2 规定它必须出现在每张表。
    """
    if mean_baseline_name not in predictions:
        raise ValueError(
            f"缺少均值基线 '{mean_baseline_name}'。§12.2 与事实 A 规定："
            "均值预测器必须出现在每一张表 —— 已发表 SOTA 在 8 个 NPASS 靶点中"
            "有 4 个跑不过它，缺了这一行读者无法判断任何数字是否有意义。"
        )

    y_true = np.asarray(y_true, dtype=np.float64)
    targets = sorted(set(target_ids))
    tid_array = np.asarray(target_ids)

    mean_rmse_by_target = {
        tid: rmse(y_true[tid_array == tid], predictions[mean_baseline_name][tid_array == tid])
        for tid in targets
    }

    rows: List[BaselineRow] = []
    for name, pred in predictions.items():
        pred = np.asarray(pred, dtype=np.float64)
        per_target_rmse: List[float] = []
        for tid in targets:
            mask = tid_array == tid
            if not mask.any():
                continue
            value = rmse(y_true[mask], pred[mask])
            per_target_rmse.append(value)
            rows.append(BaselineRow(
                name=name, target_id=tid, rmse=value,
                r2=r2_score(y_true[mask], pred[mask]),
                spearman=spearman(y_true[mask], pred[mask]),
                n_test=int(mask.sum()),
                delta_rmse_vs_mean=value - mean_rmse_by_target[tid],
            ))
        # macro 平均（靶点等权，不是样本等权 —— 靶点规模差 10 倍以上）
        macro = float(np.mean(per_target_rmse)) if per_target_rmse else float("nan")
        macro_mean_baseline = float(np.mean(list(mean_rmse_by_target.values())))
        rows.append(BaselineRow(
            name=name, target_id="__macro__", rmse=macro,
            r2=r2_score(y_true, pred), spearman=spearman(y_true, pred),
            n_test=int(y_true.size), delta_rmse_vs_mean=macro - macro_mean_baseline,
        ))

    n_lose = sum(1 for r in rows if r.target_id != "__macro__" and r.name != mean_baseline_name
                 and r.delta_rmse_vs_mean > 0)
    n_compare = sum(1 for r in rows if r.target_id != "__macro__" and r.name != mean_baseline_name)
    if n_compare:
        _LOGGER.info(
            "基线评估完成：%d/%d 个 (方法, 靶点) 组合跑不过均值基线（事实 A 的复核口径）",
            n_lose, n_compare,
        )
    return rows
