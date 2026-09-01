#!/usr/bin/env python3
"""把一次 run 的产物压成可粘贴的进度摘要。

**用途**：服务器与分析端不在同一台机器时，把"我该关心的那几十个数字"
从几百 MB 的日志和 JSON 里抽出来，压成一屏能贴进对话的文本。

用法::

    python scripts/report_digest.py                    # 全部阶段
    python scripts/report_digest.py --stage s0_data    # 只看一个阶段
    python scripts/report_digest.py --errors-only      # 只要报错与告警
    python scripts/report_digest.py -o digest.txt      # 写文件（便于 scp / 粘贴）

设计原则：**只读不写，不碰任何实验产物**。带 ``⚠`` 的行是与
``method.md`` 期望值不符、需要人看一眼的地方。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_bootstrap = Path(__file__).resolve().parent
import sys                                          # noqa: E402
sys.path.insert(0, str(_bootstrap.parent))

STAGES = ("s0_data", "s0b_backbone", "s1_base", "s2_retrieval", "s3_gate", "s4_ltt", "s5_eval")

# method.md 的期望值。None 表示"没有先验，只报告实测"。
EXPECTED: Dict[str, Tuple[Any, str]] = {
    "npass.scale_spec_basis.n_targets_ge50": (70, "事实 H"),
    "npass.scale_spec_basis.n_targets_ge100": (25, "事实 H"),
    "npass.scale_spec_basis.n_target_compound_pairs": (7314, "事实 H"),
    "npass.scale_spec_basis.n_unique_natural_products": (4075, "事实 H"),
    "blacklist.n_union_exact": (764721, "COCONUT∪LOTUS"),
    "target_selection.n_tier1": (13, "v1.0.2 实测（含 RDKit MW）"),
    "target_selection.n_tier2": (28, "v1.0.2 实测"),
    "target_selection.n_pairs_selected": (4250, "v1.0.2 实测"),
}

# 日志里值得单独拎出来的关键行
LOG_PATTERNS: List[Tuple[str, str]] = [
    (r"§5\.4.*断言.*(通过|失败)", "泄漏断言"),
    (r"NP-purge：.*", "NP-purge"),
    (r"R1–R8.*", "靶点筛选"),
    (r"BindingDB.*命中\s*(\d+)\s*条", "BindingDB"),
    (r"ChEMBL.*(抽取|命中|标准化).*", "ChEMBL"),
    (r"记忆库视图不足.*", "记忆库闸门"),
    (r"降级或排除.*", "靶点降级"),
    (r"Stage 0 闸门.*", "Stage 0 闸门"),
    (r"参数预算.*total_trainable.*", "参数预算"),
    (r"LTT (完成|未找到).*", "LTT"),
    (r"SafeCoverage = .*", "SafeCoverage"),
    (r"H[123].*(成立|不成立|passed).*", "假设判据"),
]


def _flatten(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """把嵌套 dict 压平成 ``a.b.c -> value``，跳过大列表。"""
    out: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else key
            if isinstance(value, list) and len(value) > 8:
                out[path] = f"<list len={len(value)}>"
            elif isinstance(value, dict) and len(value) > 40:
                out[path] = f"<dict len={len(value)}>"
            else:
                out.update(_flatten(value, path))
    elif isinstance(obj, list) and obj and not isinstance(obj[0], (dict, list)):
        out[prefix] = obj if len(obj) <= 8 else f"<list len={len(obj)}>"
    else:
        out[prefix] = obj
    return out


def _fmt(value: Any) -> str:
    """紧凑地格式化一个值。"""
    if isinstance(value, float):
        return f"{value:.4g}"
    text = str(value)
    return text if len(text) <= 78 else text[:75] + "..."


def digest_outputs(stage_dir: Path, lines: List[str]) -> None:
    """摘录 ``outputs/*.json``。"""
    files = sorted((stage_dir / "outputs").glob("*.json")) if (stage_dir / "outputs").is_dir() else []
    for path in files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:                     # noqa: BLE001
            lines.append(f"  ! 读不动 {path.name}：{type(exc).__name__}")
            continue
        flat = _flatten(data)
        lines.append(f"  ── {path.name}（{len(flat)} 个字段）")
        for key, value in flat.items():
            if isinstance(value, str) and value.startswith("<"):
                continue
            mark = ""
            if key in EXPECTED:
                want, source = EXPECTED[key]
                mark = f"   [期望 {want}（{source}）]" if value != want else "  ✓"
                if value != want:
                    mark = "  ⚠" + mark
            if mark or _is_interesting(key):
                lines.append(f"     {key:56s} = {_fmt(value)}{mark}")


def _is_interesting(key: str) -> bool:
    """字段名是否值得进摘要。"""
    keys = ("n_", "rate", "frac", "pct", "count", "total", "median", "min_", "max_",
            "passed", "lambda", "coverage", "risk", "auroc", "rmse", "safe", "tier",
            "purge", "overlap", "budget", "theta", "r_np", "eliminated")
    return any(k in key.lower() for k in keys)


def digest_logs(stage_dir: Path, lines: List[str], errors_only: bool, tail: int) -> None:
    """摘录 ``logs/*.log`` 里的关键行 + 全部 WARNING/ERROR。"""
    log_dir = stage_dir / "logs"
    if not log_dir.is_dir():
        return
    logs = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return
    latest = logs[-1]
    text = latest.read_text(encoding="utf-8", errors="replace").splitlines()
    lines.append(f"  ── {latest.name}（{len(text)} 行，{latest.stat().st_size / 1024:.0f} KB）")

    problems = [ln for ln in text if "| ERROR" in ln or "| WARNING" in ln]
    if problems:
        lines.append(f"     ⚠ WARNING/ERROR {len(problems)} 条：")
        for ln in problems[:20]:
            lines.append(f"       {_strip_ts(ln)[:110]}")
        if len(problems) > 20:
            lines.append(f"       …还有 {len(problems) - 20} 条")
    else:
        lines.append("     ✓ 无 WARNING/ERROR")

    if errors_only:
        return

    hits: List[str] = []
    for pattern, label in LOG_PATTERNS:
        matched = [ln for ln in text if re.search(pattern, ln)]
        for ln in matched[-2:]:
            hits.append(f"       [{label}] {_strip_ts(ln)[:104]}")
    if hits:
        lines.append("     关键行：")
        lines.extend(hits)

    if tail:
        lines.append(f"     末 {tail} 行：")
        for ln in text[-tail:]:
            lines.append(f"       {_strip_ts(ln)[:110]}")


def _strip_ts(line: str) -> str:
    """去掉时间戳与 logger 名，只留正文。"""
    parts = line.split("|")
    return parts[-1].strip() if len(parts) >= 4 else line.strip()


def digest_manifest(stage_dir: Path, lines: List[str]) -> None:
    """摘录最新 run manifest 的配置哈希（用于确认配置没被改过）。"""
    log_dir = stage_dir / "logs"
    files = sorted(log_dir.glob("*.manifest.json"), key=lambda p: p.stat().st_mtime) if log_dir.is_dir() else []
    if not files:
        return
    data = json.loads(files[-1].read_text(encoding="utf-8"))
    sha = data.get("config_sha256", {})
    short = {k: (v[:8] if isinstance(v, str) else v) for k, v in sha.items()}
    lines.append(f"  ── manifest：seed={data.get('seed')} device={data.get('device')} 配置哈希={short}")


def main() -> int:
    """入口。"""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", nargs="*", default=None, help=f"限定阶段，可选：{' '.join(STAGES)}")
    parser.add_argument("--errors-only", action="store_true", help="只要 WARNING/ERROR")
    parser.add_argument("--tail", type=int, default=6, help="每个日志附带的末尾行数（0=不要）")
    parser.add_argument("-o", "--out", default=None, help="写入文件而不是打印")
    args = parser.parse_args()

    from sparc.common import load_experiment_config                   # noqa: PLC0415

    config = load_experiment_config()
    lines: List[str] = []
    lines.append("=" * 78)
    lines.append("SPARC-NP 进度摘要")
    lines.append("=" * 78)

    ledger = Path(config.paths.get("stage_root")) / "stage_ledger.json"
    if ledger.is_file():
        data = json.loads(ledger.read_text(encoding="utf-8"))
        done = [f"{r['stage']}(r{r['round_id']})" for r in data.get("records", [])]
        lines.append(f"账本：round_id={data.get('round_id')}  已完成：{done or '（空）'}")
    else:
        lines.append("账本：尚未创建（还没有阶段跑完）")

    targets = Path(config.paths.config_dir) / "targets.yaml"
    if targets.is_file():
        import yaml                                                   # noqa: PLC0415
        summary = (yaml.safe_load(targets.read_text(encoding="utf-8")) or {}).get("summary", {})
        lines.append(f"靶点：Tier-1 {len(summary.get('tier1') or [])} / "
                     f"Tier-2 {len(summary.get('tier2') or [])} / "
                     f"Tier-X {len(summary.get('tier_x') or [])}")

    for stage in (args.stage or STAGES):
        stage_dir = Path(config.paths.get("stage_root")) / stage
        if not stage_dir.is_dir():
            continue
        has = any((stage_dir / sub).is_dir() and any((stage_dir / sub).iterdir())
                  for sub in ("logs", "outputs", "checkpoints"))
        if not has:
            continue
        lines.append("")
        lines.append(f"【{stage}】")
        digest_manifest(stage_dir, lines)
        digest_logs(stage_dir, lines, args.errors_only, args.tail)
        if not args.errors_only:
            digest_outputs(stage_dir, lines)

    reports = Path(config.paths.get("reports"))
    tables = sorted(reports.glob("table*.md")) if reports.is_dir() else []
    if tables:
        lines.append("")
        lines.append("【已产出的表】" + "  ".join(f"{t.name}({t.stat().st_size}B)" for t in tables))

    lines.append("")
    lines.append("=" * 78)
    lines.append("⚠ 标记的行 = 与 method.md 期望值不符，需要人看一眼")
    lines.append("=" * 78)

    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"已写入 {args.out}（{len(text)} 字符，{len(lines)} 行）")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
