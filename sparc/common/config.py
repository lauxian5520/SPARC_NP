"""配置对象：把 5 个冻结 YAML 汇成一个可传递、可哈希的实验配置。

规范要求"超参数与路径统一提取至配置类/配置文件中，严禁硬编码"。
本模块是全项目唯一读取 YAML 的地方；其余模块只接受 :class:`ExperimentConfig`
或其字段作为入参。

冻结纪律（§10.2 / §11.1）：``preregistration.yaml``、``frozen_hparams.yaml``、
``gate_feature_manifest.yaml`` 的 sha256 会在 :meth:`ExperimentConfig.freeze_manifest`
中计算，并由训练脚本写进每一次 run 的 manifest。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import yaml

from sparc.common.hashing import build_freeze_manifest, object_sha256
from sparc.common.paths import DEFAULT_CONFIG_DIR, PathConfig


def _load_yaml(path: Path) -> Dict[str, Any]:
    """读取 YAML 并保证返回字典。"""
    if not path.is_file():
        raise FileNotFoundError(f"缺少冻结配置文件：{path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"配置文件内容不是映射：{path}")
    return data


# ======================================================================
# 预注册
# ======================================================================
@dataclass(frozen=True)
class LTTConfig:
    """Learn-then-Test 参数 (§11.2)。"""

    alpha: float
    delta: float
    epsilon: float
    lambda_grid_start: float
    lambda_grid_stop: float
    lambda_grid_step: float
    multiple_testing: str
    pvalue_method: str
    # 仅当 multiple_testing == "fixed_sequence" 时生效：检验顺序的来源。
    # "inner_fold" 表示由 dev 内层折（与标定折互斥）预先指定，
    # 以在获得功效的同时保持有限样本有效性（见 sparc.calibrate.ltt）。
    fixed_sequence_order_source: str = "inner_fold"

    def lambda_grid(self) -> List[float]:
        """候选阈值网格 Λ = {0.00, 0.01, ..., 0.99}。"""
        n = int(round((self.lambda_grid_stop - self.lambda_grid_start) / self.lambda_grid_step)) + 1
        return [round(self.lambda_grid_start + i * self.lambda_grid_step, 10) for i in range(n)]


@dataclass(frozen=True)
class PreregistrationConfig:
    """§1.3 的三个可证伪假设判据 + §11.2 的 α/δ/ε。"""

    raw: Dict[str, Any]
    ltt: LTTConfig
    source_path: Path

    @classmethod
    def load(cls, path: Path) -> "PreregistrationConfig":
        """从 ``preregistration.yaml`` 载入。"""
        raw = _load_yaml(path)
        return cls(raw=raw, ltt=LTTConfig(**raw["ltt"]), source_path=path)

    @property
    def h1(self) -> Dict[str, Any]:
        """H1（负迁移存在性）判据。"""
        return self.raw["h1"]

    @property
    def h2(self) -> Dict[str, Any]:
        """H2（支持度可识别有害检索）判据。"""
        return self.raw["h2"]

    @property
    def h3(self) -> Dict[str, Any]:
        """H3（门控残差获得非平凡安全覆盖率）判据。"""
        return self.raw["h3"]

    @property
    def stage0_gate(self) -> Dict[str, Any]:
        """Stage 0 闸门（跨域可分性、|M_t| 下限）。"""
        return self.raw["stage0_gate"]

    @property
    def safe_coverage(self) -> Dict[str, Any]:
        """SafeCoverage 的估计参数。"""
        return self.raw["safe_coverage"]

    @property
    def mandatory_baselines(self) -> List[str]:
        """必须出现在每张表的基线（事实 A）。"""
        return list(self.raw["mandatory_baselines"])


# ======================================================================
# 冻结超参
# ======================================================================
@dataclass(frozen=True)
class DimConfig:
    """维度契约 (§8.2 / §8.3)，由 ``tests/test_dim_manifest.py`` 校验。"""

    d_pca_ligand: int
    d_pca_protein: int
    d_ligand_hidden: int
    d_protein_ctx: int
    bilinear_rank: int
    d_bilinear_block: int
    d_base_fusion_in: int
    d_base_hidden: int
    d_base_head: int
    d_query_repr: int
    d_gm_node_feat: int
    d_gm_edge_feat: int
    d_gm_hidden: int
    d_gm_proj: int
    d_align_summary: int
    d_rank_feature: int
    d_rank_hidden: int
    d_evidence_raw: int
    d_evidence_ctx: int
    d_evidence_y: int
    d_assay_family_emb: int
    n_assay_family: int
    d_candidate_meta: int
    residual_rank: int
    n_gate_features: int

    def validate(self) -> None:
        """自洽性检查：维度之间的加法关系必须成立。

        Raises:
            ValueError: 任一恒等式不成立。这些等式直接决定参数量能否对上
                §8.3.6 的 113,244，因此不允许"差不多"。
            """
        checks = [
            (self.d_bilinear_block == self.bilinear_rank ** 2,
             "s_bil 外积块必须是 bilinear_rank²（4×4=16）"),
            (self.d_base_fusion_in == self.d_ligand_hidden + self.d_protein_ctx + self.d_bilinear_block,
             "x_q = [z_q'(128); c_t(64); s_bil⊗(16)] = 208"),
            (self.d_query_repr == self.d_ligand_hidden + self.d_base_hidden,
             "v_q = [z_q'(128); h_q(128)] = 256"),
            (self.d_rank_feature == 3 * self.d_gm_proj + 7,
             "x^rank = [u_q⊙u_i(32); |u_q−u_i|(32); a_i(32); 7 标量] = 103"),
            (self.d_evidence_raw == self.d_gm_proj + self.d_align_summary + self.d_evidence_y
             + self.d_assay_family_emb + self.d_candidate_meta,
             "e_i^raw = [u_i(32); a_i(32); e_y(16); e_a(8); t_i(8)] = 96"),
        ]
        for ok, msg in checks:
            if not ok:
                raise ValueError(f"维度契约不自洽：{msg}")


@dataclass(frozen=True)
class RetrievalConfig:
    """检索管线参数 (§9.1)。"""

    k_src: int
    k0: int
    k_top: int
    min_memory_view: int
    sources: List[str]
    ecfp_radius: int
    ecfp_n_bits: int
    view_filter_keys: List[str]


@dataclass(frozen=True)
class SinkhornConfig:
    """Sinkhorn 图匹配参数 (§8.3.1)。``tau_m`` 是 v1.0 补齐的冻结超参。"""

    tau_m: float
    iters: int
    with_dustbin: bool
    log_domain: bool
    max_atoms: int
    eps: float


@dataclass(frozen=True)
class LossWeightConfig:
    """损失权重与内层 CV 网格 (§10.2)。"""

    lambda_rank: float
    lambda_utility: float
    lambda_cal: float
    harm_weight: float
    l2_theta_r: float
    tau_rank: float
    grid_rank: List[float]
    grid_utility: List[float]
    grid_cal: List[float]

    def inner_cv_grid(self) -> List[Dict[str, float]]:
        """展开内层 CV 的完整网格（笛卡尔积）。

        Returns:
            每个元素形如 ``{"lambda_rank": .., "lambda_utility": .., "lambda_cal": ..}``。
        """
        return [
            {"lambda_rank": r, "lambda_utility": u, "lambda_cal": c}
            for r in self.grid_rank
            for u in self.grid_utility
            for c in self.grid_cal
        ]


@dataclass(frozen=True)
class SplitConfig:
    """外层 60:20:20 三分与骨架家族定义 (§6.1 / §11.2)。"""

    outer_test_frac: float
    dev_train_frac: float
    dev_inner_frac: float
    dev_calib_frac: float
    protocol_default: str
    protocols: List[str]
    murcko_tanimoto_threshold: float
    use_inchikey_skeleton: bool
    use_deglyco_core: bool
    use_tautomer_family: bool

    def validate(self) -> None:
        """dev 三分必须和为 1。"""
        total = self.dev_train_frac + self.dev_inner_frac + self.dev_calib_frac
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"dev 三分之和必须为 1.0，当前 {total}")


@dataclass(frozen=True)
class DataConfig:
    """数据构建参数 (§5)。"""

    lambda_d: float
    aggregate: str
    max_pactivity_spread: float
    censor_relations: Dict[str, List[str]]
    null_token: str
    csv_field_size_limit: int

    def censor_flag(self, relation: str) -> str:
        """把 NPASS/ChEMBL 的 ``activity_relation`` 映射到审查标志。

        Args:
            relation: 原始关系符，如 ``">"``、``"="``、``"n.a."``。

        Returns:
            ``"left"`` / ``"none"`` / ``"right"``；未知关系按 ``"none"`` 处理
            但调用方应计数（§5 步骤 5 要求审查值不得静默转点值）。
        """
        rel = (relation or "").strip()
        for flag, symbols in self.censor_relations.items():
            if rel in symbols:
                return flag
        return "none"


@dataclass(frozen=True)
class TrainConfig:
    """训练超参（分阶段）。"""

    seed: int
    seeds_multi: List[int]
    base: Dict[str, Any]
    retrieval: Dict[str, Any]
    gate: Dict[str, Any]
    crossfit: Dict[str, Any]
    amp: Dict[str, Any]


@dataclass(frozen=True)
class FrozenHparams:
    """``frozen_hparams.yaml`` 的强类型视图。"""

    raw: Dict[str, Any]
    dims: DimConfig
    param_budget: Dict[str, int]
    retrieval: RetrievalConfig
    sinkhorn: SinkhornConfig
    loss: LossWeightConfig
    split: SplitConfig
    data: DataConfig
    train: TrainConfig
    source_path: Path

    @classmethod
    def load(cls, path: Path) -> "FrozenHparams":
        """从 ``frozen_hparams.yaml`` 载入并做自洽性校验。"""
        raw = _load_yaml(path)
        dims = DimConfig(**raw["dims"])
        dims.validate()

        r = raw["retrieval"]
        retrieval = RetrievalConfig(
            k_src=r["k_src"], k0=r["k0"], k_top=r["k_top"],
            min_memory_view=r["min_memory_view"], sources=list(r["sources"]),
            ecfp_radius=r["ecfp"]["radius"], ecfp_n_bits=r["ecfp"]["n_bits"],
            view_filter_keys=list(r["view_filter_keys"]),
        )
        lw = raw["loss_weights"]
        loss = LossWeightConfig(
            lambda_rank=lw["lambda_rank"]["default"],
            lambda_utility=lw["lambda_utility"]["default"],
            lambda_cal=lw["lambda_cal"]["default"],
            harm_weight=lw["harm_weight"],
            l2_theta_r=lw["l2_theta_r"],
            tau_rank=lw["tau_rank"],
            grid_rank=list(lw["lambda_rank"]["grid"]),
            grid_utility=list(lw["lambda_utility"]["grid"]),
            grid_cal=list(lw["lambda_cal"]["grid"]),
        )
        sp = raw["split"]
        split = SplitConfig(
            outer_test_frac=sp["outer_test_frac"], dev_train_frac=sp["dev_train_frac"],
            dev_inner_frac=sp["dev_inner_frac"], dev_calib_frac=sp["dev_calib_frac"],
            protocol_default=sp["protocol_default"], protocols=list(sp["protocols"]),
            murcko_tanimoto_threshold=sp["scaffold_family"]["murcko_tanimoto_threshold"],
            use_inchikey_skeleton=sp["scaffold_family"]["use_inchikey_skeleton"],
            use_deglyco_core=sp["scaffold_family"]["use_deglyco_core"],
            use_tautomer_family=sp["scaffold_family"]["use_tautomer_family"],
        )
        split.validate()
        d = raw["data"]
        data = DataConfig(
            lambda_d=d["lambda_d"], aggregate=d["aggregate"],
            max_pactivity_spread=d["max_pactivity_spread"],
            censor_relations={k: list(v) for k, v in d["censor_relations"].items()},
            null_token=d["null_token"], csv_field_size_limit=d["csv_field_size_limit"],
        )
        t = raw["train"]
        train = TrainConfig(
            seed=t["seed"], seeds_multi=list(t["seeds_multi"]), base=dict(t["base"]),
            retrieval=dict(t["retrieval"]), gate=dict(t["gate"]),
            crossfit=dict(t["crossfit"]), amp=dict(t["amp"]),
        )
        return cls(
            raw=raw, dims=dims, param_budget=dict(raw["param_budget"]),
            retrieval=retrieval, sinkhorn=SinkhornConfig(**raw["sinkhorn"]),
            loss=loss, split=split, data=data, train=train, source_path=path,
        )


# ======================================================================
# 门控特征清单
# ======================================================================
@dataclass(frozen=True)
class GateFeatureManifest:
    """28 维门控特征清单 (§8.3.5)。

    这是 Figure 2（门控系数图）的全部载体，也是 H2 判据 (d)
    "系数符号与化学先验一致"能够存在的前提。
    """

    raw: Dict[str, Any]
    source_path: Path

    @classmethod
    def load(cls, path: Path) -> "GateFeatureManifest":
        """从 ``gate_feature_manifest.yaml`` 载入并校验条目数。"""
        raw = _load_yaml(path)
        obj = cls(raw=raw, source_path=path)
        n_declared = raw["n_features"]
        if len(raw["features"]) != n_declared:
            raise ValueError(f"门控特征清单条目数 {len(raw['features'])} != 声明的 {n_declared}")
        if raw["n_params"] != n_declared + 1:
            raise ValueError("n_params 必须等于 n_features + 1（截距）")
        idxs = [f["idx"] for f in raw["features"]]
        if idxs != list(range(1, n_declared + 1)):
            raise ValueError("门控特征 idx 必须是从 1 开始的连续整数，且顺序即 w_g 分量顺序")
        return obj

    @property
    def n_features(self) -> int:
        """特征维数（默认 28）。"""
        return int(self.raw["n_features"])

    @property
    def names(self) -> List[str]:
        """按 idx 顺序的特征名 —— 即 ``w_g`` 的分量顺序。"""
        return [f["name"] for f in self.raw["features"]]

    @property
    def sign_priors(self) -> List[int]:
        """化学先验符号，用于 H2 判据 (d) 的符号翻转计数。"""
        return [int(f["sign_prior"]) for f in self.raw["features"]]

    @property
    def groups(self) -> List[str]:
        """特征分组（similarity / conflict / uncertainty），Figure 2 的着色依据。"""
        return [f["group"] for f in self.raw["features"]]

    @property
    def pools(self) -> List[str]:
        """每维特征的计算池（top_k / k0 / both / model）。"""
        return [f["pool"] for f in self.raw["features"]]

    def index_of(self, name: str) -> int:
        """按名字取 0-based 下标。"""
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise KeyError(f"门控特征清单中没有 '{name}'") from exc


# ======================================================================
# 顶层实验配置
# ======================================================================
@dataclass
class ExperimentConfig:
    """一次 run 的完整配置：路径 + 三份冻结 YAML + 模型注册表。"""

    paths: PathConfig
    prereg: PreregistrationConfig
    hparams: FrozenHparams
    gate_manifest: GateFeatureManifest
    model_registry: Dict[str, Any]
    stage: str = "unset"
    run_name: str = "default"
    seed: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def effective_seed(self) -> int:
        """本次 run 实际使用的种子（未显式指定则取冻结默认值）。"""
        return self.seed if self.seed is not None else self.hparams.train.seed

    def freeze_manifest(self) -> Dict[str, str]:
        """三份冻结配置 + 模型注册表的 sha256。

        Returns:
            ``{"preregistration": sha, "frozen_hparams": sha, ...}``。
            训练脚本必须把它写进 run manifest，否则"冻结"无法被审计。
        """
        return build_freeze_manifest({
            "preregistration": self.prereg.source_path,
            "frozen_hparams": self.hparams.source_path,
            "gate_feature_manifest": self.gate_manifest.source_path,
            "paths": self.paths.config_dir / "paths.yaml",
            "model_registry": self.paths.config_dir / "model_registry.yaml",
        })

    def run_manifest(self, device_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """生成完整的 run manifest（落盘到 ``data/<stage>/logs/``）。"""
        manifest: Dict[str, Any] = {
            "spec_version": self.hparams.raw.get("spec_version"),
            "stage": self.stage,
            "run_name": self.run_name,
            "seed": self.effective_seed,
            "config_sha256": self.freeze_manifest(),
            "device": device_info,
            "extra": self.extra,
        }
        manifest["manifest_sha256"] = object_sha256(manifest)
        return manifest


def load_experiment_config(
    config_dir: Path | str | None = None,
    stage: str = "unset",
    run_name: str = "default",
    seed: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> ExperimentConfig:
    """载入一次实验所需的全部配置。

    Args:
        config_dir: 配置目录；``None`` 时用 ``Project/code/configs``。
        stage: 阶段名，同时决定 ``data/<stage>/{logs,checkpoints,outputs}``。
        run_name: run 名称，用于区分同一阶段的多次运行。
        seed: 覆盖冻结种子；``None`` 表示用 ``frozen_hparams.yaml`` 的默认值。
        extra: 额外写入 manifest 的自由字段。

    Returns:
        :class:`ExperimentConfig`。
    """
    cfg_dir = Path(config_dir).resolve() if config_dir else DEFAULT_CONFIG_DIR
    paths = PathConfig.load(cfg_dir / "paths.yaml")
    registry = _load_yaml(cfg_dir / "model_registry.yaml")
    return ExperimentConfig(
        paths=paths,
        prereg=PreregistrationConfig.load(cfg_dir / "preregistration.yaml"),
        hparams=FrozenHparams.load(cfg_dir / "frozen_hparams.yaml"),
        gate_manifest=GateFeatureManifest.load(cfg_dir / "gate_feature_manifest.yaml"),
        model_registry=registry,
        stage=stage,
        run_name=run_name,
        seed=seed,
        extra=dict(extra or {}),
    )


# §7.2 的 P0 判据：三条里任意一条不是 pass，候选即不可注册。
# c8  几何依赖排除     —— 3D 构象/几何预训练一律 blocked
# c9  预训练域门禁     —— r_NP ≤ 0.05；**unknown ⇒ blocked**，不得按"大概是药物库"放行
# c10 低数据稳定性     —— 由 Stage 0-B 实测回填（优于均值基线，且不劣于 ECFP4+RF 超 5%）
P0_ELIGIBILITY_CRITERIA: Tuple[Tuple[str, str], ...] = (
    ("c8", "几何依赖排除（3D 构象/几何重建/构象对齐预训练）"),
    ("c9", "预训练域门禁 r_NP ≤ 0.05（unknown ⇒ blocked）"),
    ("c10", "低数据稳定性（Stage 0-B 实测回填）"),
)


class ModelAEligibilityError(ValueError):
    """Model A 未通过 §7.2 的资格判据。

    独立于普通 ``ValueError`` 的类型，目的与 :class:`LeakageAssertionError` 一致：
    让"放行一个没验过的主干"在代码里显得刻意，而不是顺手。
    """


def resolve_model_a_entry(
    registry: Dict[str, Any],
    name: str,
    allow_unverified: Optional[Sequence[str]] = None,
    enforce_eligibility: bool = True,
) -> Dict[str, Any]:
    """从注册表取一个 Model A 条目，并执行 §7.2 的硬阻断与资格闸门。

    两道闸门：

    1. **名单阻断** —— 命中 ``blocked:`` 即报错。挡住的是"已经想到的那几个"
       （GraphMVP / Uni-Mol / GEM / NaFM）。
    2. **判据闸门** —— P0 判据（c8/c9/c10）为 ``fail`` 或 ``unverified`` 即报错。
       挡住的是**还没想到的**。``model_registry.yaml`` 头部写着"unverified 不等于
       pass；判据 9 明确规定 unknown ⇒ blocked"，这道闸门就是那句话的执行者 ——
       在它存在之前，四个候选的 ``c9`` 全是 ``unverified`` 而 resolve 一路畅通。

    Args:
        registry: ``model_registry.yaml`` 内容。
        name: 候选名，如 ``"molformer_xl"``。
        allow_unverified: 显式放行的判据编号，如 ``["c10"]``。**只用于 Stage 0-B
            本身** —— c10 要靠 S0-B 跑出来才能回填，跑之前它必然是 ``unverified``。
            放行清单会被调用方写进 run manifest，因此"我临时放过了哪一条"永远
            留痕。放行 ``c9`` 需要在 PR/实验记录里给出 r_NP 的独立测算依据。
        enforce_eligibility: 关掉判据闸门（仅供单元测试构造夹具用）。
            **业务代码不得传 False**；名单阻断在任何情况下都不可关闭。

    Returns:
        候选条目字典。

    Raises:
        ValueError: 命中 blocked 清单（判据 8 几何依赖 / 判据 9 预训练域门禁），
            或候选不存在。**blocked 不可用配置绕过** —— NaFM 作 Model A
            会直接抹掉本项目的跨域前提。
        ModelAEligibilityError: P0 判据未通过，且未被 ``allow_unverified`` 显式放行。
    """
    for blocked in registry.get("blocked", []):
        if blocked["name"] == name:
            raise ValueError(
                f"Model A '{name}' 被硬阻断，不可注册：{blocked['reason']}（method.md §7.2）"
            )

    entry: Optional[Dict[str, Any]] = None
    for candidate in registry.get("candidates", []):
        if candidate["name"] == name:
            entry = candidate
            break
    if entry is None:
        known: Sequence[str] = [c["name"] for c in registry.get("candidates", [])]
        raise ValueError(f"注册表中没有 Model A '{name}'；已登记：{list(known)}")

    if enforce_eligibility:
        _assert_p0_eligibility(entry, name, set(allow_unverified or ()))
    return entry


def _assert_p0_eligibility(entry: Dict[str, Any], name: str, waived: Set[str]) -> None:
    """检查 §7.2 的三条 P0 判据。

    Args:
        entry: 候选条目。
        name: 候选名（错误信息用）。
        waived: 被显式放行的判据编号。

    Raises:
        ModelAEligibilityError: 存在未放行的 ``fail`` / ``unverified`` / 缺失判据。
    """
    eligibility = entry.get("eligibility") or {}
    failures: List[str] = []
    for code, description in P0_ELIGIBILITY_CRITERIA:
        if code in waived:
            continue
        status = str(eligibility.get(code, "missing")).strip().lower()
        if status != "pass":
            failures.append(f"  · 判据 {code}（{description}）：{status}")

    if not failures:
        return

    raise ModelAEligibilityError(
        f"Model A '{name}' 未通过 §7.2 的 P0 资格判据 —— 拒绝注册。\n"
        + "\n".join(failures)
        + "\n\nmodel_registry.yaml 头部写明：**unverified 不等于 pass**；"
        "判据 9 明确规定 unknown ⇒ blocked，不得按'大概是药物库'放行。\n"
        "合法的推进方式：\n"
        "  1. 实测后把 model_registry.yaml 里对应判据改成 pass（c9 需给出 r_NP 的测算过程）；\n"
        f"  2. 若正在跑 Stage 0-B（c10 必须由它产出），显式传 "
        f"allow_unverified=['c10'] —— 放行清单会写进 run manifest 留痕。\n"
        "  绝不要把 enforce_eligibility 关掉：那会让整个 §7.2 形同虚设，\n"
        "  而判据 9 是'跨域'这个前提唯一的守卫。"
    )
