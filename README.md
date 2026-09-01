# SPARC-NP v1.0 —— 代码

> **S**elective **P**rediction with **A**nalogical **R**etrieval under **C**hemical shift — **N**atural **P**roducts
>
> 唯一实现依据：`../../method.md` v1.0.0。任何与之冲突的实现一律视为缺陷。

---

## 迁移到 CUDA 服务器

整个 `Project/` 目录原样拷过去即可 —— **所有路径都是相对路径**，以
`code/configs/paths.yaml` 所在目录为基准点解析，无需改动任何一行配置。

```bash
# 1. 环境（rdkit 用 conda 更省事）
conda create -n sparc python=3.11 && conda activate sparc
conda install -c conda-forge "rdkit=2024.03.*"
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install -e .

# 2. 解包源数据（§2.3-T1/T2；ChEMBL 展开约 25 GB）
tar -xzf ../data/origin_dataset/ChEMBL37/chembl_37_sqlite.tar.gz -C ../data/interim/
python -c "
from sparc.data.chembl import ChEMBLExtractor
with ChEMBLExtractor('../data/interim/chembl_37.db', read_only=False) as e:
    e.create_indexes(); print(e.table_counts())
"

# 3. 自检（无 GPU 也能跑）
pytest -q                      # 全部测试
pytest -q -m "not slow"

# 4. 按阶段跑
python scripts/run_s0_data.py       --run-name s0_v1
python scripts/run_s0b_backbone.py  --run-name s0b_v1 --candidates molformer_xl kpgt chemberta2_mlm
python scripts/run_s1_base.py       --run-name s1_v1 --model-a <S0-B 选出的主 Model A>
python scripts/run_s2_retrieval.py  --run-name s2_v1 --base-ckpt s1_v1_frozen
python scripts/run_s3_gate.py       --run-name s3_v1 --retrieval-ckpt s2_v1 --forced-retrieval-diagnostic
python scripts/run_s4_ltt.py        --run-name s4_v1 --losses <calib_losses.npz> --confirm-holdout
python scripts/run_s5_eval.py       --run-name s5_final --predictions <test.npz> --lambda-star <λ*> --deterministic
```

在树莓派上可以跑的只有：`pytest`（带标记的会 skip）、
`python scripts/run_s0_data.py --dry-run`、以及全部 notebook 的前几节。

---

## 目录

```
code/
├── configs/                     ← 冻结配置，全部带 sha256 写进每次 run 的 manifest
│   ├── preregistration.yaml     H1/H2/H3 判据、α=0.10 δ=0.05 ε=0.01
│   ├── frozen_hparams.yaml      维度契约、参数预算、τ_m=0.10、损失网格、划分比例
│   ├── gate_feature_manifest.yaml   28 维门控特征（顺序即 w_g 分量顺序）
│   ├── model_registry.yaml      Model A 候选 + 11 条资格判据 + 硬阻断清单
│   ├── paths.yaml               全部相对路径
│   └── targets.yaml             由 Stage 0 生成（R1–R8 结果）
├── sparc/
│   ├── common/    配置 / 路径 / 设备 / 种子 / 日志 / 断点续训 / 哈希冻结
│   ├── chem/      标准化 / 指纹 / 脱糖 / 骨架家族
│   ├── data/      NPASS / ChEMBL / BindingDB / 黑名单 / NP-purge / 划分 / 单位
│   ├── models/    Model A 适配器 / 冻结 PCA / Θ_B / 图匹配 / 重排 / 证据 / 残差 / 门控
│   ├── retrieval/ 记忆库视图 / 三源索引 / 28 维特征 / 管线
│   ├── losses/    Tobit / 排序 / 效用·伤害·校准
│   ├── train/     阶段守卫 / 通用训练循环
│   ├── calibrate/ Learn-then-Test / 交叉拟合
│   └── eval/      SafeCoverage / 指标 / 基线 / 表 / 图
├── scripts/       每个阶段一个 CLI 入口
├── experiments/   notebook（只做交互式调用与可视化）
└── tests/         泄漏断言 / 维度与参数预算 / 退化等价性 / 标定 / 门控特征 / 糖苷夹具
```

数据一一对应写到 `../data/<模块名>/{logs,checkpoints,outputs}/`。

---

## 六条不可静默放宽的约束

代码里每一条都有对应的守卫，绕过它们都需要一个**显式动作**。

| # | 约束 | 守卫 |
|---|---|---|
| 1 | §5.4 四条零重叠断言 **fail hard** | `LeakageAssertionError`，不是 warning |
| 2 | 骨架家族需要**四把钥匙**，不是一把 | `ScaffoldFamilyBuilder`；关掉任一把会打日志告警 |
| 3 | `λ` 来自 Learn-then-Test，不是网格搜索 | `learn_then_test(calibration_is_held_out=...)` 必须显式确认 |
| 4 | 训练期用**连续 `g`** | `SupportGate.forward(apply_threshold=False)` 是默认值 |
| 5 | 回退等价性在 **fp32** 下断言 | `assert_fallback_identity` 拒绝非 fp32 输入 |
| 6 | Model A 冻结，NaFM / 3D 预训练模型**硬阻断** | `resolve_model_a_entry` 抛异常，配置绕不过 |

另外两条工程纪律：

* **参数预算零容差**：`tests/test_dim_manifest.py` 三方比对
  解析计算 == `frozen_hparams.yaml` == 实际 `nn.Module`，
  合计必须是 **113,244**（Θ_B 57,218 + Θ_R 56,026）。
* **S4 之后不允许回溯**：`StageGuard` 把 §10.4 实现成落盘状态机。
  S4 完成后再进入 S0–S3 会抛 `StageViolationError`，
  唯一出路是 `--round-id` 递增并从 S0 重来。

---

## 两处实现层面的判断（与规范一致，但值得知道）

**1. LTT 的多重校正默认改用 Bonferroni。**
§11.2 允许 `fixed_sequence` 与 `bonferroni` 二选一。固定序列按「λ 从大到小」
从 λ=0.99 起测，而标定折在最大 λ 上通常只有个位数被接受样本 ——
经验风险为 0 时 Hoeffding p 值仍需 `n ≥ ln(1/δ)/(2α²) = 150` 才能拒绝，
序列在第一步就中断、λ\* 恒为 `None`。这会把 H3 变成不可检验，
而它与「真的不存在安全工作点」是两回事。
诊断见 `diagnose_fixed_sequence_power`；若要用固定序列，
用 `fixed_sequence_order_from_inner_fold` 在 **dev 内层折**（与标定折互斥）
上预先指定检验顺序，这样既有功效又保持有限样本有效性。

**2. R5 的直系同源合并只按酶名，不按化合物重叠。**
规范 R5 说的是「同 (酶功能, EC 号) 组」，即同一个酶的不同物种。
NPASS 里 CYP3A4/2D6/1A1/1B1/1A2/2C9/2C19 被同一批天然产物筛过、两两重叠远超 50%，
但它们是 **7 个不同的酶**。按重叠合并实测会让 R1–R8 的存活靶点从 31 掉到 13，
直接抹掉事实 H 的「6.4 倍免费解」。规范引用的三个事实 D 案例
（AChE 人/电鳗/电鳐、COX-1 绵羊/人、酪氨酸酶蘑菇/人）在 NPASS 里
`target_name` **完全相同**，酶名归一化即可全部捕获。
化合物重叠仍然重要，但它的位置是 `compound_overlap_report` 这个**诊断**，
不是合并判据。

---

## 已在本机核实的数字

| 项 | 实测 | `method.md` |
|---|---:|---:|
| `>=50` 化合物靶点 | 70 | 70 |
| `>=100` 化合物靶点 | 25 | 25 |
| (靶点, 化合物) 对 | 7,314 | 7,314 |
| 唯一天然产物 | 4,075 | 4,075 |
| COCONUT ∪ LOTUS 唯一 InChIKey | 764,721 | 764,721 |
| AChE 人 vs 电鳐 交集/较小集 | 17/25 = 68.0% | 68.0% |
| 参数合计 | 113,244 | 113,244 |
| Θ_R : 独立监督（70 靶点池） | 10.17 : 1 | 10.2 : 1 |

> 单位换算后的靶点数会低于 70（本机 59），因为 19,581 条 `ug.mL-1` 记录
> 缺分子量无法换算 —— §5.2 步骤 3 要求丢弃并计数。服务器上有 RDKit
> 即可由 SMILES 算出 MW 补回。两个口径都写在 Stage 0 报告里。

---

## 语言约定

文档、注释、日志用中文；代码标识符与配置键用英文。
