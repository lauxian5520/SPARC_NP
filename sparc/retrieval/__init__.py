"""检索管线 (§9)。

**v1.0 强制顺序**（修正 talk.md §3.7.1 与 §3.7.2 的矛盾）::

    1. 视图构建：先按 [同 UniProt + 同 tax_id + 同 endpoint + fold 可见性]
       构建过滤后的记忆视图 M_view
    2. 在 M_view 上分别构建三个索引：ECFP4 LSH | Model A HNSW | 药效团 ANN
    3. 每源召回 K_src = min(512, |M_view|)
    4. 并集 → 去重 → 按 Stage-1 分数取 K₀ = min(256, |union|)
    5. assert compatible_ratio == 1.0        # 视图已过滤，此处必须恒真
    6. Sinkhorn 图匹配 → Reranker → Top-K = 16

按旧顺序（三源各召回 512 → 并集 → 再过滤）实现，兼容记录占比 ~10⁻³ 时
512 个里平均只剩 0.5 个可用 ⇒ 候选集静默为空 ⇒ 永久走 fallback 而不报错。
"""

from typing import Any

from sparc.retrieval.memory import MemoryView, MemoryViewBuilder
from sparc.retrieval.features import GateFeatureExtractor, GateFeatureBundle

_LAZY_EXPORTS = {
    "RetrievalIndex": "index", "CandidatePool": "index",
    "RetrievalPipeline": "pipeline", "RetrievalBatch": "pipeline",
}

__all__ = ["MemoryView", "MemoryViewBuilder", "GateFeatureExtractor", "GateFeatureBundle", *sorted(_LAZY_EXPORTS)]


def __getattr__(name: str) -> Any:
    """按需导入依赖 torch 的符号。"""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'sparc.retrieval' has no attribute '{name}'")
    import importlib  # noqa: PLC0415

    return getattr(importlib.import_module(f"sparc.retrieval.{module_name}"), name)
