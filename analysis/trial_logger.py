from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

from common.types import ExecutionResult, Observation, PolicyAction, RefinedGrasp, RobotState, SafeAction


class TrialLogger:
    """Append one JSON record per trial for later analysis."""

    def __init__(self, log_dir: str) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "trial_log.jsonl"

    def log_trial(
        self,
        *,
        instruction: str,
        observation: Observation,
        policy_action: PolicyAction,
        safe_action: SafeAction,
        refined_grasp: RefinedGrasp,
        result: ExecutionResult,
        final_robot_state: RobotState,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "instruction": instruction,
            "observation": asdict(observation),
            "policy_action": asdict(policy_action),
            "safe_action": asdict(safe_action),
            "refined_grasp": asdict(refined_grasp),
            "result": asdict(result),
            "final_robot_state": asdict(final_robot_state),
            "metadata": dict(metadata or {}),
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
