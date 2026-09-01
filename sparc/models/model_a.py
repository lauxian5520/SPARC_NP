"""Model A：冻结的药物域编码器，可插拔 (§7)。

**Model A 永不训练。** 它是外部输入，通过 :class:`ModelAAdapter` 协议接入。
下游只依赖 ``d_A``；``d_A`` 之后立刻接冻结 PCA 白化到 128 维
(:mod:`sparc.models.whitening`)，因此切换 Model A 不改动任何其它模块。

§7.2 的 11 条资格判据里，有三条是代码能直接检查的：
* 判据 8/9（几何依赖、预训练域门禁）—— 由注册表的 ``blocked`` 段硬阻断，
  见 :func:`sparc.common.config.resolve_model_a_entry`；
* 判据 11（输出稳定性）—— :func:`check_output_stability`；
* 判据 3（可钉版本）—— :meth:`ModelAAdapter.fingerprint`。

其余判据（低数据稳定性等）由 Stage 0-B 的实测决定，见
``scripts/run_s0b_backbone.py``。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from sparc.common.logging_utils import get_logger

_LOGGER = get_logger(__name__)


@dataclass(frozen=True)
class ModelAFingerprint:
    """Model A 的可复现指纹 (§7.1)。"""

    weights_sha256: str
    tokenizer_sha256: str
    normalization_spec: str
    revision: str
    d_a: int

    def to_dict(self) -> Dict[str, Any]:
        """转成可写 manifest 的字典。"""
        return {
            "weights_sha256": self.weights_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "normalization_spec": self.normalization_spec,
            "revision": self.revision,
            "d_a": self.d_a,
        }


@runtime_checkable
class ModelAAdapter(Protocol):
    """§7.1 的接口契约。

    实现方必须保证：
    * ``encode`` 确定性、eval mode、无 dropout、无随机构象；
    * 对 drug 与 natural product 使用**完全相同**的标准化与推理路径。
      跨域比较的全部意义系于这一条 —— 两侧走不同的预处理，
      "域偏移"测出来的就是预处理差异。
    """

    name: str
    d_a: int
    revision: str

    def encode(self, smiles: Sequence[str]) -> np.ndarray:
        """把 SMILES 列表编码成 ``(N, d_A)`` float32 矩阵。"""
        ...

    def fingerprint(self) -> ModelAFingerprint:
        """返回权重/分词器 sha256 与标准化规格。"""
        ...


class _BaseAdapter:
    """适配器公共部分：缓存、批处理、canonicalization。"""

    name: str = "base"
    d_a: int = 0
    revision: str = "unpinned"

    def __init__(
        self,
        batch_size: int = 64,
        device: str = "cpu",
        cache_dir: Optional[Path] = None,
        canonicalize: bool = True,
    ) -> None:
        """
        Args:
            batch_size: 推理批大小。
            device: ``"cuda"`` / ``"cpu"``。
            cache_dir: embedding 缓存目录；§14 建议把 Model A 推理
                完全离线化成缓存，训练时不再跑编码器。
            canonicalize: 是否在 encode 内固化 canonicalization。
                判据 11 不满足时**必须**开启，并记入 ``normalization_spec``。
        """
        self.batch_size = batch_size
        self.device = device
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.canonicalize = canonicalize
        self._cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    def _canonical(self, smiles: Sequence[str]) -> List[str]:
        """统一 canonicalization —— drug 与 NP 走同一条路径。"""
        if not self.canonicalize:
            return list(smiles)
        from sparc.chem.rdkit_backend import require_rdkit  # noqa: PLC0415

        chem = require_rdkit()
        out: List[str] = []
        for smi in smiles:
            mol = chem.MolFromSmiles(smi)
            out.append(chem.MolToSmiles(mol, canonical=True) if mol is not None else smi)
        return out

    @property
    def normalization_spec(self) -> str:
        """标准化规格描述，写进指纹。"""
        return f"canonicalize={self.canonicalize};desalt=upstream;tautomer=upstream"

    def encode_cached(self, smiles: Sequence[str]) -> np.ndarray:
        """带内存缓存的编码（同一分子在多个靶点上重复出现时省一次推理）。"""
        missing = [s for s in smiles if s not in self._cache]
        if missing:
            vectors = self.encode(missing)
            for smi, vec in zip(missing, vectors):
                self._cache[smi] = vec
        return np.stack([self._cache[s] for s in smiles]).astype(np.float32)

    def encode(self, smiles: Sequence[str]) -> np.ndarray:  # pragma: no cover - 抽象
        """子类实现。"""
        raise NotImplementedError

    def fingerprint(self) -> ModelAFingerprint:  # pragma: no cover - 抽象
        """子类实现。"""
        raise NotImplementedError

    # ------------------------------------------------------------------
    def save_cache(self, path: Path, keys: Sequence[str]) -> Path:
        """把 embedding 缓存落盘为 npz（§14 的离线化）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        matrix = np.stack([self._cache[k] for k in keys if k in self._cache])
        np.savez_compressed(path, keys=np.array([k for k in keys if k in self._cache]), embeddings=matrix)
        _LOGGER.info("Model A embedding 缓存已落盘：%s（%d 条）", path, matrix.shape[0])
        return path

    def load_cache(self, path: Path) -> int:
        """从 npz 载入 embedding 缓存。"""
        data = np.load(path, allow_pickle=False)
        keys = data["keys"].tolist()
        embeddings = data["embeddings"]
        for key, vec in zip(keys, embeddings):
            self._cache[key] = vec.astype(np.float32)
        _LOGGER.info("Model A embedding 缓存已载入：%s（%d 条）", path, len(keys))
        return len(keys)


class HuggingFaceSmilesEncoder(_BaseAdapter):
    """HuggingFace SMILES 语言模型适配器（MoLFormer-XL / ChemBERTa-2）。"""

    def __init__(
        self,
        hf_id: str,
        revision: str,
        name: str = "hf_smiles",
        pooling: str = "mean",
        trust_remote_code: bool = False,
        max_length: int = 512,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            hf_id: HuggingFace 模型 ID。
            revision: **必须钉死的 commit/revision**（判据 3）。
            name: 注册表键。
            pooling: ``"mean"``（掩码均值池化）或 ``"cls"``。
            trust_remote_code: MoLFormer-XL 需要 ``True``。
            max_length: 分词最大长度。
        """
        super().__init__(**kwargs)
        if not revision or revision.startswith("REQUIRED"):
            raise ValueError(
                f"Model A '{name}' 的 revision 未钉死（当前 '{revision}'）。"
                "§7.2 判据 3 要求 commit/HF revision 必须钉死才能注册。"
            )
        self.hf_id = hf_id
        self.revision = revision
        self.name = name
        self.pooling = pooling
        self.trust_remote_code = trust_remote_code
        self.max_length = max_length
        self._model = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        """惰性加载模型与分词器，并设为 eval + 禁梯度。"""
        if self._model is not None:
            return
        import torch  # noqa: PLC0415
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.hf_id, revision=self.revision, trust_remote_code=self.trust_remote_code
        )
        self._model = AutoModel.from_pretrained(
            self.hf_id, revision=self.revision, trust_remote_code=self.trust_remote_code
        )
        self._model.eval().to(self.device)
        for param in self._model.parameters():
            param.requires_grad_(False)          # Model A 永不训练
        self.d_a = int(self._model.config.hidden_size)
        _LOGGER.info("Model A '%s' 已载入：%s@%s，d_A=%d", self.name, self.hf_id, self.revision, self.d_a)
        del torch

    def encode(self, smiles: Sequence[str]) -> np.ndarray:
        """确定性编码。"""
        import torch  # noqa: PLC0415

        self._ensure_loaded()
        texts = self._canonical(smiles)
        outputs: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start:start + self.batch_size]
                encoded = self._tokenizer(
                    batch, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
                ).to(self.device)
                hidden = self._model(**encoded).last_hidden_state          # (B, L, d_A)
                if self.pooling == "cls":
                    pooled = hidden[:, 0]
                else:
                    mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                outputs.append(pooled.float().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32)

    def fingerprint(self) -> ModelAFingerprint:
        """权重与分词器指纹。"""
        self._ensure_loaded()
        return ModelAFingerprint(
            weights_sha256=_hash_module_weights(self._model),
            tokenizer_sha256=hashlib.sha256(
                repr(sorted(self._tokenizer.get_vocab().items())).encode("utf-8")
            ).hexdigest(),
            normalization_spec=self.normalization_spec,
            revision=self.revision,
            d_a=self.d_a,
        )


class MolCLRGraphEncoder(_BaseAdapter):
    """MolCLR (GIN) 适配器 —— 需要外部仓库代码与 checkpoint。

    NaFM Table 2 中 MolCLR 在 8 个靶点上全胜 ECFP（事实 G），
    是本项目最强的正面证据，因此列为必选候选。
    """

    name = "molclr_gin"
    d_a = 512

    def __init__(self, ckpt_path: Path, revision: str, repo_path: Optional[Path] = None, **kwargs: Any) -> None:
        """
        Args:
            ckpt_path: ``pretrained_gin/model.pth``。
            revision: 仓库 commit（判据 3）。
            repo_path: MolCLR 仓库根目录（其 ``models/ginet_molclr.py`` 需要在 sys.path 上）。
        """
        super().__init__(**kwargs)
        self.ckpt_path = Path(ckpt_path)
        self.repo_path = Path(repo_path) if repo_path else None
        self.revision = revision
        self._model = None

    def _ensure_loaded(self) -> None:
        """载入 MolCLR GIN 主干。"""
        if self._model is not None:
            return
        import sys  # noqa: PLC0415

        import torch  # noqa: PLC0415

        if self.repo_path and str(self.repo_path) not in sys.path:
            sys.path.insert(0, str(self.repo_path))
        try:
            from models.ginet_molclr import GINet  # type: ignore  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "找不到 MolCLR 的 GINet 定义。请 clone github.com/yuyangw/MolCLR "
                "并把仓库根目录作为 repo_path 传入。"
            ) from exc
        model = GINet()
        state = torch.load(self.ckpt_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
        model.eval().to(self.device)
        for param in model.parameters():
            param.requires_grad_(False)
        self._model = model
        _LOGGER.info("Model A 'molclr_gin' 已载入：%s", self.ckpt_path)

    def encode(self, smiles: Sequence[str]) -> np.ndarray:
        """图编码。需要 torch_geometric 的批处理。"""
        import torch  # noqa: PLC0415

        self._ensure_loaded()
        from sparc.models.graphmatcher import MolecularGraphFeaturizer  # noqa: PLC0415

        featurizer = MolecularGraphFeaturizer()
        texts = self._canonical(smiles)
        outputs: List[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch = [featurizer.to_molclr_data(s) for s in texts[start:start + self.batch_size]]
                batch = [b for b in batch if b is not None]
                if not batch:
                    continue
                from torch_geometric.data import Batch  # noqa: PLC0415

                data = Batch.from_data_list(batch).to(self.device)
                embedding, _ = self._model(data)
                outputs.append(embedding.float().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32) if outputs else np.zeros((0, self.d_a), np.float32)

    def fingerprint(self) -> ModelAFingerprint:
        """权重指纹（图模型无分词器）。"""
        self._ensure_loaded()
        return ModelAFingerprint(
            weights_sha256=_hash_module_weights(self._model),
            tokenizer_sha256="n/a:graph_model",
            normalization_spec=self.normalization_spec,
            revision=self.revision,
            d_a=self.d_a,
        )


class EcfpBaselineEncoder(_BaseAdapter):
    """ECFP4(2048) "编码器" —— 不是 Model A，是 §7.4 的非神经基线。

    放在这里是为了让 S0-B 的对比实验能走完全相同的下游路径
    （同一个 PCA 白化、同一个 Θ_B），把"编码器差异"与
    "下游实现差异"分开。
    """

    name = "ecfp4"
    revision = "rdkit_morgan_r2_2048"

    def __init__(self, radius: int = 2, n_bits: int = 2048, **kwargs: Any) -> None:
        """
        Args:
            radius: Morgan 半径（2 = ECFP4）。
            n_bits: 位数。
        """
        super().__init__(**kwargs)
        self.d_a = n_bits
        self.radius = radius
        self.n_bits = n_bits
        self._calculator = None

    def encode(self, smiles: Sequence[str]) -> np.ndarray:
        """返回 0/1 指纹矩阵（float32）。"""
        from sparc.chem.fingerprints import FingerprintCalculator  # noqa: PLC0415

        if self._calculator is None:
            self._calculator = FingerprintCalculator(self.radius, self.n_bits)
        rows = []
        for smi in self._canonical(smiles):
            fp = self._calculator.ecfp4(smi)
            rows.append(fp if fp is not None else np.zeros(self.n_bits, dtype=np.uint8))
        return np.vstack(rows).astype(np.float32)

    def fingerprint(self) -> ModelAFingerprint:
        """确定性指纹（无权重）。"""
        return ModelAFingerprint(
            weights_sha256="n/a:deterministic_fingerprint",
            tokenizer_sha256="n/a",
            normalization_spec=self.normalization_spec,
            revision=self.revision,
            d_a=self.d_a,
        )


def _hash_module_weights(module: Any) -> str:
    """对 ``nn.Module`` 的全部参数做确定性 sha256。"""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


# ======================================================================
# 注册表与判据 11
# ======================================================================
MODEL_A_REGISTRY: Dict[str, Callable[..., _BaseAdapter]] = {
    "HuggingFaceSmilesEncoder": HuggingFaceSmilesEncoder,
    "MolCLRGraphEncoder": MolCLRGraphEncoder,
    "EcfpBaselineEncoder": EcfpBaselineEncoder,
}


def build_model_a(entry: Dict[str, Any], device: str = "cpu", cache_dir: Optional[Path] = None, **overrides: Any) -> _BaseAdapter:
    """按 ``model_registry.yaml`` 的条目构造适配器。

    Args:
        entry: :func:`sparc.common.config.resolve_model_a_entry` 的返回值。
            （该函数已执行 blocked 硬阻断，此处不重复检查。）
        device: 推理设备。
        cache_dir: embedding 缓存目录。
        **overrides: 覆盖条目字段（如临时换 revision）。

    Returns:
        构造好的适配器。

    Raises:
        ValueError: adapter 字段格式不对，或类名不在注册表中。
    """
    adapter_spec = overrides.pop("adapter", entry.get("adapter", ""))
    class_name = adapter_spec.rsplit(":", 1)[-1]
    if class_name not in MODEL_A_REGISTRY:
        raise ValueError(f"未知的 Model A adapter '{class_name}'；已注册：{sorted(MODEL_A_REGISTRY)}")

    kwargs: Dict[str, Any] = {"device": device, "cache_dir": cache_dir}
    if class_name == "HuggingFaceSmilesEncoder":
        kwargs.update(
            hf_id=entry["hf_id"], revision=entry["revision"], name=entry["name"],
            pooling=entry.get("pooling", "mean"), trust_remote_code=entry.get("trust_remote_code", False),
        )
    elif class_name == "MolCLRGraphEncoder":
        kwargs.update(ckpt_path=entry["ckpt_rel"], revision=entry["revision"], repo_path=entry.get("repo_path"))
    kwargs.update(overrides)
    return MODEL_A_REGISTRY[class_name](**kwargs)


def check_output_stability(
    adapter: _BaseAdapter,
    smiles_list: Sequence[str],
    n_random_variants: int = 3,
    min_cosine: float = 0.99,
    seed: int = 42,
) -> Dict[str, Any]:
    """§7.2 判据 11：输出稳定性检查。

    同一 SMILES 的两种等价写法（canonical / random SMILES），
    embedding 余弦相似度必须 ≥ 0.99。不满足者需在 ``encode()`` 内
    固化 canonicalization，并记入 ``normalization_spec``。

    Args:
        adapter: 待检适配器。
        smiles_list: 测试分子。
        n_random_variants: 每个分子生成的随机 SMILES 数。
        min_cosine: 阈值（冻结为 0.99）。
        seed: 随机 SMILES 的种子。

    Returns:
        ``{"passed": bool, "min_cosine": float, "mean_cosine": float, ...}``。
    """
    from sparc.chem.rdkit_backend import require_rdkit  # noqa: PLC0415

    chem = require_rdkit()
    rng = np.random.default_rng(seed)

    canonical: List[str] = []
    variants: List[str] = []
    owner: List[int] = []
    for i, smi in enumerate(smiles_list):
        mol = chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canonical.append(chem.MolToSmiles(mol, canonical=True))
        for _ in range(n_random_variants):
            variants.append(chem.MolToSmiles(mol, canonical=False, doRandom=True,
                                             randomSeed=int(rng.integers(1, 10 ** 6))))
            owner.append(len(canonical) - 1)

    if not canonical:
        return {"passed": False, "reason": "no_parsable_smiles"}

    base = adapter.encode(canonical)
    variant_vectors = adapter.encode(variants)
    base_norm = base / (np.linalg.norm(base, axis=1, keepdims=True) + 1e-12)
    var_norm = variant_vectors / (np.linalg.norm(variant_vectors, axis=1, keepdims=True) + 1e-12)
    cosines = np.array([float(base_norm[o] @ var_norm[i]) for i, o in enumerate(owner)])

    result = {
        "criterion": "c11_output_stability",
        "n_molecules": len(canonical),
        "n_comparisons": int(cosines.size),
        "min_cosine": float(cosines.min()),
        "mean_cosine": float(cosines.mean()),
        "frac_below_threshold": float((cosines < min_cosine).mean()),
        "threshold": min_cosine,
        "passed": bool(cosines.min() >= min_cosine),
    }
    if not result["passed"]:
        _LOGGER.warning(
            "判据 11 未通过：最小余弦 %.4f < %.2f。必须在 encode() 内固化 canonicalization "
            "并记入 normalization_spec。", result["min_cosine"], min_cosine,
        )
    return result
