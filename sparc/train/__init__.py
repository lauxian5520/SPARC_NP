"""训练阶段 (§10.4)。

```
S0    数据构建、NP-purge、划分、泄漏断言        —— 无训练
S0-B  Model A 主干对比                        —— 只训线性头
S1    Base 预测器训练                          可训 Θ_B
S1-F  冻结 Θ_B，保存独立 checkpoint
S2    检索分支训练（L_rank + L_task，连续 g）    可训 Θ_R \\ gate
S3    交叉拟合效用标签 → 门控 logistic 拟合      可训 gate (29)
S4    Learn-then-Test 标定 λ                   全部冻结
S5    测试集单次评估                            全部冻结
```

**S4 与 S5 之间不允许任何回溯修改。** 任何在 S5 之后的调整都必须
重新走 S0 并声明为新的实验轮次。:class:`~sparc.train.stage_guard.StageGuard`
把这条纪律实现成一个会落盘的状态机。
"""

from typing import Any

from sparc.train.stage_guard import StageGuard, StageViolationError

# trainer 依赖 torch，按需导入（本机树莓派无 torch，但阶段守卫仍需可用）
_LAZY_EXPORTS = {"EpochResult": "trainer", "Trainer": "trainer", "TrainerConfig": "trainer"}

__all__ = ["StageGuard", "StageViolationError", *sorted(_LAZY_EXPORTS)]


def __getattr__(name: str) -> Any:
    """按需导入依赖 torch 的符号。"""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'sparc.train' has no attribute '{name}'")
    import importlib  # noqa: PLC0415

    return getattr(importlib.import_module(f"sparc.train.{module_name}"), name)
