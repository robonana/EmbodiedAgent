"""
agent/prompt_agent.py — PromptEmbodiedAgent main loop.

Gemini-2.5-Pro decides one tool call per step.  The loop runs until:
  - finish() is called
  - max_agent_steps is reached
  - 3 identical consecutive actions detected (repeat warning, then halt if 10)
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Optional

from .gemini_client import GeminiClient
from .toolbox import ToolboxProtocol
from .prompts import SYSTEM_PROMPT, build_policy_prompt
from .schemas import ToolAction, ToolResult, VALID_TOOLS
from .tasks import get_task_config
from .trajectory_logger import TrajectoryLogger


class PromptEmbodiedAgent:
    """
    Closed-loop embodied agent:

        observe → Gemini picks tool → execute → update history → repeat

    One Gemini call per step.  Current observation image always attached.
    Relevant memory/crop images attached only when the last result includes them.
    """

    def __init__(
        self,
        toolbox: ToolboxProtocol,
        gemini_client: GeminiClient,
        log_dir: str,
        max_agent_steps: int = 40,
        history_window: int = 8,
        max_monitor_cycles: int = 5,
    ):
        self.toolbox      = toolbox
        self.gemini       = gemini_client
        self.log_dir      = log_dir
        self.max_steps    = max_agent_steps
        self.hist_window  = history_window
        self.max_monitors = max_monitor_cycles

    # ── Main entry point ──────────────────────────────────────────────────────

    def run(
        self,
        task: str,
        episode_config: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Run a complete episode.

        Returns a summary dict with keys:
            success, answer, total_steps, episode_id, episode_dir
        """
        episode_id = uuid.uuid4().hex[:12]
        task_cfg   = get_task_config(task)

        cfg_for_log = {
            "task": task,
            "episode_id": episode_id,
            "model": self.gemini.model_name,
            "max_agent_steps": self.max_steps,
            "history_window": self.hist_window,
            **(episode_config or {}),
        }
        logger = TrajectoryLogger(
            log_root=self.log_dir,
            episode_id=episode_id,
            task=task,
            config=cfg_for_log,
        )

        # Redirect toolbox log_dir images to this episode's directory
        # (toolbox already writes there; just forward the reference)

        print(f"\n{'='*60}")
        print(f"[PromptAgent] task={task!r}  max_steps={self.max_steps}")
        print(f"[PromptAgent] episode_id={episode_id}")
        print(f"{'='*60}")

        # ── Episode state ─────────────────────────────────────────────────────
        trajectory: list[dict]      = []   # full log (for JSONL only)
        transient_memory: list[str] = []   # accumulated progress_analysis (for prompt)
        current_obs: dict           = {}   # last observe() data dict
        last_result: Optional[ToolResult] = None
        last_action: Optional[ToolAction] = None
        last_action_key        = None  # for repeat detection
        repeat_count           = 0
        final_answer: Optional[str] = None
        success                = False
        step_idx               = 0

        # ── Step 0: initial observation ───────────────────────────────────────
        print(f"\n[PromptAgent] Step 0: initial observe()")
        obs_result = self.toolbox.observe()
        if obs_result.ok:
            current_obs = obs_result.data.copy()
        last_result = obs_result

        # ── Main loop ─────────────────────────────────────────────────────────
        for step_idx in range(self.max_steps):
            print(f"\n[PromptAgent] Step {step_idx + 1}/{self.max_steps}  "
                  f"task={task[:50]!r}")

            # Fresh observation before every policy call
            obs_result = self.toolbox.observe()
            if obs_result.ok:
                current_obs = obs_result.data.copy()

            # ── Build prompt ─────────────────────────────────────────────────
            repeat_warning = None
            if repeat_count >= 3:
                repeat_warning = (
                    f"You repeated the same action {repeat_count} times without "
                    f"progress."
                )

            user_prompt = build_policy_prompt(
                task=task,
                step_idx=step_idx + 1,
                timestamp=time.strftime("%H:%M:%S"),
                current_observation=current_obs,
                transient_memory=transient_memory,
                last_action=last_action.to_dict() if last_action else None,
                last_result=last_result.to_dict() if last_result else None,
                repeat_warning=repeat_warning,
            )

            # ── Collect labeled images for policy call ────────────────────────
            labeled_images: list[tuple[str, str]] = []
            obs_path = self.toolbox._last_image_path
            if obs_path:
                labeled_images.append((f"Current observation image ({obs_path}):",
                                       obs_path))
            if last_result and last_result.image_paths:
                for i, p in enumerate(last_result.image_paths):
                    if p != obs_path and os.path.exists(p):
                        labeled_images.append((
                            f"Last tool result image {i + 1} ({os.path.basename(p)}):",
                            p,
                        ))

            # ── Call Gemini policy ────────────────────────────────────────────
            print(f"  [Gemini] calling policy (images={len(labeled_images)})")
            raw_response_dict = self.gemini.call_policy(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                labeled_images=labeled_images,
            )

            # ── Parse action ─────────────────────────────────────────────────
            if not raw_response_dict or "tool" not in raw_response_dict:
                print(f"  [PromptAgent] Gemini returned no valid action at step "
                      f"{step_idx + 1}")
                action = ToolAction(
                    rationale="Gemini returned no valid JSON; waiting one step.",
                    tool="wait",
                    arguments={"seconds": 1},
                )
            else:
                action = ToolAction.from_dict(raw_response_dict)

            # Sanitise: unknown tool → wait
            if action.tool not in VALID_TOOLS:
                print(f"  [PromptAgent] unknown tool '{action.tool}', using wait")
                action = ToolAction(
                    rationale=f"Tool '{action.tool}' is invalid; waiting one step.",
                    tool="wait",
                    arguments={"seconds": 1},
                )

            if action.progress_analysis:
                print(f"  progress_analysis: {action.progress_analysis}")
            print(f"  rationale={action.rationale!r}\n"
                  f"  action: {action.tool}  args={action.arguments}")

            # ── Repeat detection ─────────────────────────────────────────────
            action_key = f"{action.tool}:{_canonical_args(action.arguments)}"
            if action_key == last_action_key:
                repeat_count += 1
            else:
                repeat_count = 0
            last_action_key = action_key

            if repeat_count >= 10:
                print(f"  [PromptAgent] repeated same action 10 times — halting.")
                last_result = ToolResult(
                    ok=False, tool="finish",
                    summary="Halted: repeated same action 10 times without progress.",
                )
                logger.log_step(action, last_result, current_obs)
                break

            # ── Execute action ────────────────────────────────────────────────
            result = self.toolbox.execute(action)
            last_result = result

            print(f"  result: ok={result.ok}  {result.summary}")

            # ── Log step ─────────────────────────────────────────────────────
            logger.log_step(
                action=action,
                result=result,
                current_obs=current_obs,
                prompt_text=user_prompt,
                raw_gemini_text=self.gemini._last_raw_response,
                success=None,
            )

            # ── Update last action + transient memory ────────────────────────
            last_action = action
            if action.progress_analysis:
                transient_memory.append(action.progress_analysis)

            # ── Update observation state ──────────────────────────────────────
            if result.ok and result.data.get("image_path"):
                current_obs["summary"] = result.summary
                if result.data.get("robot_pose"):
                    current_obs["robot_pose"] = result.data["robot_pose"]

            # ── Append to trajectory log (JSONL only — not used in prompt) ────
            trajectory.append({
                "step_idx":         step_idx + 1,
                "progress_analysis": action.progress_analysis or "",
                "rationale":        action.rationale or "",
                "expected_progress": action.expected_progress or "",
                "tool":             action.tool,
                "arguments":        action.arguments,
                "ok":               result.ok,
                "summary":          result.summary,
            })

            # ── Check terminal conditions ────────────────────────────────────
            if action.tool == "finish":
                final_answer = action.arguments.get("answer") or result.data.get("answer")
                success = result.ok
                print(f"  [PromptAgent] finish called. success={success}")
                break

            if result.data.get("task_done"):
                success = True
                break

        # ── Episode complete ──────────────────────────────────────────────────
        print(f"\n[PromptAgent] Episode done — "
              f"steps={step_idx + 1}  success={success}  answer={final_answer!r}")

        final_path = logger.save_final_result(
            success=success,
            answer=final_answer,
            total_steps=step_idx + 1,
        )
        logger.close()

        return {
            "success": success,
            "answer": final_answer,
            "total_steps": step_idx + 1,
            "episode_id": episode_id,
            "episode_dir": logger.episode_dir,
            "final_result_path": final_path,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _canonical_args(args: dict) -> str:
    """Stable string representation of arguments for repeat detection."""
    import json as _json
    try:
        return _json.dumps(args, sort_keys=True)
    except Exception:
        return str(args)
