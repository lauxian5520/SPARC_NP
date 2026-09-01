"""蛋白编码：ESM-2 mean-pool + 冻结 PCA (§8.2)。

**为什么是 mean-pool 而不是 attention pooling**：原设计的
``W_p, W_p,out ∈ R^{256×640}`` 共 328,448 参数 —— **68 个蛋白序列
养不起 32.8 万参数**。在单靶点设定下，那等于用 32.8 万参数计算一个常数，
是一条 256 维的过拟合通道。v1.0 改为冻结 mean-pool + 冻结 PCA + 64×64
共 4,288 参数。

ESM-2 推理结果按 UniProt accession 缓存到 ``data/cache/esm2/``，
训练时不再跑 ESM-2（§14：主要显存占用可完全离线化）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from sparc.common.logging_utils import get_logger

_LOGGER = get_logger(__name__)


class ProteinEncoder:
    """ESM-2 150M 编码器（冻结，只做推理）。"""

    def __init__(
        self,
        model_dir: Path,
        device: str = "cpu",
        max_length: int = 1022,
        batch_size: int = 4,
        cache_dir: Optional[Path] = None,
    ) -> None:
        """
        Args:
            model_dir: ``data/assets_weight/ESM-2-150M``（hidden 640, 30 层）。
            device: 推理设备。
            max_length: 序列截断长度（ESM-2 位置上限 1026，留出特殊 token）。
            batch_size: 批大小。
            cache_dir: ``data/cache/esm2``。
        """
        self.model_dir = Path(model_dir)
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._model = None
        self._tokenizer = None
        self.d_protein = 640          # 与规范的 H_p^raw ∈ R^{L×640} 一致

    def _ensure_loaded(self) -> None:
        """惰性载入 ESM-2 并冻结。"""
        if self._model is not None:
            return
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

        self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self._model = AutoModel.from_pretrained(str(self.model_dir))
        self._model.eval().to(self.device)
        for param in self._model.parameters():
            param.requires_grad_(False)
        self.d_protein = int(self._model.config.hidden_size)
        _LOGGER.info("ESM-2 已载入：%s，hidden=%d，layers=%d",
                     self.model_dir, self.d_protein, self._model.config.num_hidden_layers)

    # ------------------------------------------------------------------
    def encode(self, sequences: Sequence[str]) -> np.ndarray:
        """对序列做掩码均值池化。

        Args:
            sequences: 氨基酸序列列表。

        Returns:
            ``(N, 640)`` float32。
        """
        import torch  # noqa: PLC0415

        self._ensure_loaded()
        outputs: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(sequences), self.batch_size):
                batch = [s[:self.max_length] for s in sequences[start:start + self.batch_size]]
                encoded = self._tokenizer(batch, padding=True, truncation=True,
                                          max_length=self.max_length, return_tensors="pt").to(self.device)
                hidden = self._model(**encoded).last_hidden_state          # (B, L, 640)
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                # 去掉 CLS/EOS：只对真实氨基酸位置做平均
                mask[:, 0] = 0.0
                lengths = encoded["attention_mask"].sum(dim=1) - 1
                for i, length in enumerate(lengths.tolist()):
                    mask[i, int(length)] = 0.0
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                outputs.append(pooled.float().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32)

    def encode_targets(self, sequences: Dict[str, str], cache_name: str = "esm2_targets.npz") -> Dict[str, np.ndarray]:
        """编码全部靶点序列并缓存。

        Args:
            sequences: ``{uniprot_accession: 序列}``。
            cache_name: 缓存文件名。

        Returns:
            ``{accession: (640,) 向量}``。缓存命中时不再跑 ESM-2。
        """
        cache_path = (self.cache_dir / cache_name) if self.cache_dir else None
        if cache_path and cache_path.is_file():
            data = np.load(cache_path, allow_pickle=False)
            cached = {k: v.astype(np.float32) for k, v in zip(data["accessions"].tolist(), data["embeddings"])}
            missing = set(sequences) - set(cached)
            if not missing:
                _LOGGER.info("ESM-2 缓存命中：%s（%d 个靶点）", cache_path, len(cached))
                return {k: cached[k] for k in sequences}
            _LOGGER.info("ESM-2 缓存缺 %d 个靶点，重新编码", len(missing))

        accessions = sorted(sequences)
        vectors = self.encode([sequences[a] for a in accessions])
        result = {a: v for a, v in zip(accessions, vectors)}
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, accessions=np.array(accessions), embeddings=vectors)
            _LOGGER.info("ESM-2 编码已缓存：%s", cache_path)
        return result
