"""阶段纪律守卫 (§10.4)。

**S4 与 S5 之间不允许任何回溯修改。** 这是一句纪律，但纪律靠人记
在多周的项目里是记不住的 —— 尤其是"跑完 S5 发现结果不好，回去
调一下 S2 再跑一遍"这个动作，在当下每一步看起来都合理。

本模块把它实现成一个**落盘的状态机**：阶段完成时写一个带 hash 的
戳记；若在 S4 完成后再试图进入 S1/S2/S3，直接抛异常，并给出
唯一合法的出路 —— 声明新的实验轮次（``round_id`` 递增），
从 S0 重新开始。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from sparc.common.hashing import object_sha256
from sparc.common.logging_utils import get_logger

_LOGGER = get_logger(__name__)

STAGE_ORDER = ("s0_data", "s0b_backbone", "s1_base", "s1f_freeze",
               "s2_retrieval", "s3_gate", "s4_ltt", "s5_eval")

# S4 之后不允许再进入的阶段
LOCKED_AFTER_S4 = ("s0_data", "s0b_backbone", "s1_base", "s1f_freeze", "s2_retrieval", "s3_gate")


class StageViolationError(RuntimeError):
    """违反阶段纪律 (§10.4)。"""


@dataclass
class StageRecord:
    """一个阶段的完成戳记。"""

    stage: str
    round_id: int
    completed_at: str
    config_sha256: Dict[str, str] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    record_sha256: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转成可落盘的字典。"""
        return {
            "stage": self.stage, "round_id": self.round_id, "completed_at": self.completed_at,
            "config_sha256": self.config_sha256, "artifacts": self.artifacts,
            "metrics": self.metrics, "record_sha256": self.record_sha256,
        }


class StageGuard:
    """阶段状态机，状态落盘在 ``data/stage_ledger.json``。"""

    def __init__(self, ledger_path: Path, round_id: int = 1) -> None:
        """
        Args:
            ledger_path: 状态文件路径（建议 ``data/stage_ledger.json``）。
            round_id: 实验轮次。S5 之后要改任何东西，唯一合法出路是
                递增 ``round_id`` 并从 S0 重来。
        """
        self.path = Path(ledger_path)
        self.round_id = round_id
        self.records: List[StageRecord] = []
        if self.path.is_file():
            self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        """读回已完成的阶段。"""
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self.records = [StageRecord(**row) for row in data.get("records", [])]
        _LOGGER.info("阶段账本已载入：%s（%d 条记录）", self.path, len(self.records))

    def _save(self) -> None:
        """落盘。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"round_id": self.round_id, "records": [r.to_dict() for r in self.records]}
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------
    def completed_stages(self, round_id: Optional[int] = None) -> List[str]:
        """本轮已完成的阶段。"""
        rid = round_id if round_id is not None else self.round_id
        return [r.stage for r in self.records if r.round_id == rid]

    def is_completed(self, stage: str, round_id: Optional[int] = None) -> bool:
        """某阶段是否已完成。"""
        return stage in self.completed_stages(round_id)

    # ------------------------------------------------------------------
    def check_can_enter(self, stage: str) -> None:
        """进入某阶段前的检查。

        Args:
            stage: 目标阶段。

        Raises:
            StageViolationError: 违反前置顺序，或试图在 S4 完成后回溯。
            ValueError: 未知阶段名。
        """
        if stage not in STAGE_ORDER:
            raise ValueError(f"未知阶段 '{stage}'，合法阶段：{STAGE_ORDER}")

        completed = set(self.completed_stages())

        # 核心纪律：S4 之后不允许回溯
        if "s4_ltt" in completed and stage in LOCKED_AFTER_S4:
            raise StageViolationError(
                f"S4（Learn-then-Test 标定）已在轮次 {self.round_id} 完成，不允许再进入 '{stage}'。\n"
                "§10.4：S4 与 S5 之间不允许任何回溯修改；任何在 S5 之后的调整都必须\n"
                "重新走 S0 并声明为新的实验轮次。\n"
                f"唯一合法出路：StageGuard(ledger_path, round_id={self.round_id + 1}) 并从 s0_data 重来。\n"
                "这条限制存在的理由：λ 由 LTT 给出的有限样本保证，前提是标定折在\n"
                "看到它之前模型已经完全固定。回溯修改会让保证失效，而失效不会报错。"
            )

        # 前置顺序
        index = STAGE_ORDER.index(stage)
        missing = [s for s in STAGE_ORDER[:index] if s not in completed and s != "s0b_backbone"]
        if missing:
            _LOGGER.warning("进入 '%s' 时以下前置阶段尚未完成：%s（若为有意跳过，请在 run manifest 中说明）",
                            stage, missing)

    def complete(
        self,
        stage: str,
        config_sha256: Optional[Dict[str, str]] = None,
        artifacts: Optional[Dict[str, str]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> StageRecord:
        """标记某阶段完成并落盘。

        Args:
            stage: 阶段名。
            config_sha256: 冻结配置指纹。
            artifacts: 产物路径。
            metrics: 关键指标。

        Returns:
            :class:`StageRecord`。
        """
        record = StageRecord(
            stage=stage, round_id=self.round_id,
            completed_at=datetime.now().isoformat(timespec="seconds"),
            config_sha256=dict(config_sha256 or {}),
            artifacts={k: str(v) for k, v in (artifacts or {}).items()},
            metrics=dict(metrics or {}),
        )
        record.record_sha256 = object_sha256(record.to_dict())
        self.records = [r for r in self.records if not (r.stage == stage and r.round_id == self.round_id)]
        self.records.append(record)
        self._save()
        _LOGGER.info("阶段 '%s'（轮次 %d）已标记完成，账本：%s", stage, self.round_id, self.path)
        return record

    def start_new_round(self) -> int:
        """开启新的实验轮次（S5 之后要改东西的唯一合法出路）。

        Returns:
            新的 ``round_id``。
        """
        self.round_id = max((r.round_id for r in self.records), default=0) + 1
        _LOGGER.info("已开启新的实验轮次：round_id=%d，必须从 s0_data 重新开始", self.round_id)
        self._save()
        return self.round_id

    def summary(self) -> Dict[str, Any]:
        """账本摘要。"""
        return {
            "ledger": str(self.path),
            "current_round": self.round_id,
            "completed_this_round": self.completed_stages(),
            "all_rounds": sorted({r.round_id for r in self.records}),
            "s4_locked": "s4_ltt" in set(self.completed_stages()),
        }
