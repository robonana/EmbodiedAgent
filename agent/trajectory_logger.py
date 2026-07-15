"""
agent/trajectory_logger.py — Episode trajectory logging.

Saves every step to JSONL for future SFT / rejection-sampling / RL.
Each episode gets its own directory under runs/prompt_agent/

Design notes:
  * JSONL (one self-contained JSON object per line), not a single JSON array, so an
    episode that crashes mid-run still leaves a readable, parseable prefix — and so a
    later training job can stream the corpus without loading whole episodes.
  * Every line is flushed immediately. Episodes take minutes and can die inside the
    simulator; buffering would routinely lose the last (most interesting) steps.
  * Large blobs (the full prompt, the raw VLM completion, images) go to sidecar files
    and are referenced by path from the JSONL row. Keeps lines small, and lets the
    viewer lazily load only what it renders.
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
        self.step_count = 0   # monotonically increasing; becomes TrajectoryStep.step_idx

        # Create the episode directory and every subdirectory up front, so downstream
        # writers (the toolbox saving frames/crops) never have to check for existence.
        self.ep_dir = Path(log_root)
        for sub in ("raw_gemini", "prompts", "images", "crops"):
            (self.ep_dir / sub).mkdir(parents=True, exist_ok=True)

        # Write config
        # Snapshot of the run's settings (model, scene, thresholds, ...). `default=str`
        # because config routinely holds Paths and other non-JSON objects that we'd
        # rather stringify than crash on.
        (self.ep_dir / "config.json").write_text(
            json.dumps(config, indent=2, default=str), encoding="utf-8")

        # Open trajectory JSONL
        # Append mode: a resumed/retried run adds to the same episode rather than
        # silently truncating the steps already recorded.
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

        # Spill the two big text blobs to sidecar files first, so the JSONL row can just
        # carry their paths. Filenames are step-ordered (zero-padded) *and* timestamped,
        # which keeps them sorting correctly while staying unique across a resumed run.
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

        # Flush per step — see module docstring: episodes die in the simulator, and an
        # unflushed buffer would lose exactly the steps that explain why.
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
        """Write final_result.json and close the JSONL file.

        This file is the unit of aggregation for evaluation: compare_results.py globs for
        it across run directories. `extra` lets a specific runner add benchmark-specific
        metrics (e.g. OVMM's per-phase success flags) without changing this schema.
        """
        result = {
            "episode_id": self.episode_id,
            "task": self.task,
            "success": success,
            "answer": answer,          # populated for QA-style tasks; None otherwise
            "total_steps": total_steps,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **(extra or {}),
        }
        path = self.ep_dir / "final_result.json"
        path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"[TrajectoryLogger] final_result → {path}")
        return str(path)

    def close(self) -> None:
        """Best-effort close. Swallows errors: this runs in episode teardown, often
        already on an exception path, and a double-close must not mask the real error."""
        try:
            self._traj_file.close()
        except Exception:
            pass

    @property
    def episode_dir(self) -> str:
        return str(self.ep_dir)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _save_text(self, text: str, subdir: str, filename: str) -> str:
        """Write a sidecar text blob and return its path.

        Deliberately non-fatal: losing a debug artefact (a prompt dump) must never abort
        an episode that is otherwise running fine. The path is returned regardless, so a
        reader that later finds it missing can just skip it.
        """
        path = self.ep_dir / subdir / filename
        try:
            path.write_text(text, encoding="utf-8")
        except Exception:
            pass
        return str(path)
