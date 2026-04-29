"""
agent/trajectory_logger.py — Episode trajectory logging.

Saves every step to JSONL for future SFT / rejection-sampling / RL.
Each episode gets its own directory under runs/prompt_agent/
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from .schemas import ToolAction, ToolResult, TrajectoryStep


class TrajectoryLogger:
    """
    Logs one episode to:

        runs/prompt_agent/{timestamp}_{scene}_{task_slug}/
            config.json
            trajectory.jsonl          — one JSON line per step
            raw_gemini/               — raw Gemini text responses
            images/                   — observation frames
            crops/                    — crop frames

            final_result.json

    File paths logged here become the training corpus for future fine-tuning.
    """

    def __init__(
        self,
        log_root: str,
        episode_id: str,
        task: str,
        config: dict[str, Any],
    ):
        self.episode_id = episode_id
        self.task       = task
        self.step_count = 0

        self.ep_dir = Path(log_root)
        for sub in ("raw_gemini", "prompts", "images", "crops"):
            (self.ep_dir / sub).mkdir(parents=True, exist_ok=True)

        # Write config
        (self.ep_dir / "config.json").write_text(
            json.dumps(config, indent=2, default=str), encoding="utf-8")

        # Open trajectory JSONL
        self._traj_path = self.ep_dir / "trajectory.jsonl"
        self._traj_file = self._traj_path.open("a", encoding="utf-8")

        print(f"[TrajectoryLogger] episode_dir={self.ep_dir}")

    # ── Public API ────────────────────────────────────────────────────────────

    def log_step(
        self,
        action: ToolAction,
        result: ToolResult,
        current_obs: dict[str, Any],
        prompt_text: Optional[str] = None,
        raw_gemini_text: Optional[str] = None,
        success: Optional[bool] = None,
    ) -> TrajectoryStep:
        """Write one trajectory step to JSONL. Returns the TrajectoryStep."""
        ts = time.strftime("%H:%M:%S")

        raw_path: Optional[str] = None
        if raw_gemini_text:
            raw_path = self._save_text(
                raw_gemini_text, "raw_gemini", f"step{self.step_count:04d}_{ts.replace(':','')}.txt")

        prompt_path: Optional[str] = None
        if prompt_text:
            prompt_path = self._save_text(
                prompt_text, "prompts", f"step{self.step_count:04d}_{ts.replace(':','')}.txt")

        step = TrajectoryStep(
            episode_id=self.episode_id,
            step_idx=self.step_count,
            task=self.task,
            timestamp=ts,
            current_obs=current_obs,
            action=action.to_dict(),
            result=result.to_dict(),
            raw_gemini_path=raw_path,
            prompt_path=prompt_path,
            success=success,
        )

        self._traj_file.write(step.to_json() + "\n")
        self._traj_file.flush()
        self.step_count += 1

        return step

    def save_final_result(
        self,
        success: bool,
        answer: Optional[str],
        total_steps: int,
        extra: Optional[dict] = None,
    ) -> str:
        """Write final_result.json and close the JSONL file."""
        result = {
            "episode_id": self.episode_id,
            "task": self.task,
            "success": success,
            "answer": answer,
            "total_steps": total_steps,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **(extra or {}),
        }
        path = self.ep_dir / "final_result.json"
        path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"[TrajectoryLogger] final_result → {path}")
        return str(path)

    def close(self) -> None:
        try:
            self._traj_file.close()
        except Exception:
            pass

    @property
    def episode_dir(self) -> str:
        return str(self.ep_dir)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _save_text(self, text: str, subdir: str, filename: str) -> str:
        path = self.ep_dir / subdir / filename
        try:
            path.write_text(text, encoding="utf-8")
        except Exception:
            pass
        return str(path)


