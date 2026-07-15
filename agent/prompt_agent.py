"""
agent/prompt_agent.py — PromptEmbodiedAgent main loop.

Gemini-2.5-Pro decides one tool call per step.  The loop runs until:
  - finish() is called
  - max_agent_steps is reached
  - 3 identical consecutive actions detected (repeat warning, then halt if 10)

This is the whole "agent": there is no planner, no state machine, no learned policy.
Each step re-observes the world, rebuilds a prompt from scratch, asks the VLM for one
tool call, executes it, and folds the result into the context for the next step.

Two things carry across steps and are worth naming:
  * `transient_memory` — the running list of the VLM's own progress_analysis strings.
    This is the agent's working memory (what it thinks it has accomplished), distinct
    from EpisodicMemory (what it has *seen*, retrievable by similarity).
  * `last_action` / `last_result` — replayed into the next prompt so the model can
    verify whether its previous action did what it predicted.

Everything is backend-agnostic: `toolbox` is any ToolboxProtocol, `gemini_client` is
GeminiClient or its OpenAI-compatible subclass.
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
        self.max_steps    = max_agent_steps    # hard episode budget; also bounds cost
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
        task_cfg   = get_task_config(task)   # eval metadata only — never enters a prompt

        # Everything needed to reproduce this run, snapshotted into config.json.
        cfg_for_log = {
            "task": task,
            "episode_id": episode_id,
            "model": self.gemini.model_name,
            "max_agent_steps": self.max_steps,
            "history_window": self.hist_window,
            **(episode_config or {}),   # runner-specific extras (scene id, seed, ...)
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
        # Seeds current_obs / last_result before the first policy call, so step 1's
        # prompt already describes the world instead of an empty scene.
        print(f"\n[PromptAgent] Step 0: initial observe()")
        obs_result = self.toolbox.observe()
        if obs_result.ok:
            current_obs = obs_result.data.copy()
        last_result = obs_result

        # ── Main loop ─────────────────────────────────────────────────────────
        for step_idx in range(self.max_steps):
            print(f"\n[PromptAgent] Step {step_idx + 1}/{self.max_steps}  "
                  f"task={task[:50]!r}")

            # Fresh observation before every policy call.
            # Re-observing (rather than reusing the previous step's result) matters
            # because the world may have moved on its own — physics settling, a dropped
            # object rolling — since the last action returned.
            obs_result = self.toolbox.observe()
            if obs_result.ok:
                current_obs = obs_result.data.copy()

            # ── Build prompt ─────────────────────────────────────────────────
            # Nudge the model out of a loop *before* halting it. Three identical actions
            # is usually the model retrying something that silently fails (e.g. grasping
            # from too far away); telling it so is often enough to break the cycle.
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
            # Each image gets a caption that says *what it is*; the client interleaves
            # caption-then-image so the model can tell them apart. `attached` dedupes:
            # the current frame is often also present in the last result's image list.
            labeled_images: list[tuple[str, str]] = []
            attached: set[str] = set()
            obs_path = self.toolbox._last_image_path
            if obs_path:
                labeled_images.append((
                    f"Current head-camera observation (forward-facing) ({obs_path}):",
                    obs_path))
                attached.add(obs_path)
            if last_result and last_result.image_paths:
                # Map each candidate image path → its memory_id so the policy sees
                # that e.g. the file "000034.png" IS memory mem_000034 (the navigate
                # target) — it never has to guess the correspondence or re-retrieve
                # just to recover a memory_id for a frame it already identified.
                path_to_mem: dict[str, str] = {}
                for c in (last_result.data or {}).get("candidates", []):
                    cp, mid = c.get("rgb_path"), c.get("memory_id")
                    if cp and mid:
                        path_to_mem[cp] = mid
                for i, p in enumerate(last_result.image_paths):
                    if p not in attached and os.path.exists(p):
                        mem_id = path_to_mem.get(p)
                        if mem_id:
                            # Spell out the exact `navigate` call for this frame. The
                            # model reliably picks the right *image* but frequently
                            # garbles the id syntax; handing it the literal argument
                            # removes that failure mode.
                            label = (
                                f"Last tool result image {i + 1} "
                                f"(file {os.path.basename(p)} = memory_id {mem_id}; "
                                f'navigate here with target={{"memory_id":"{mem_id}"}}):'
                            )
                        else:
                            # A crop or a detection overlay — no memory id to cite.
                            label = (
                                f"Last tool result image {i + 1} "
                                f"({os.path.basename(p)}):"
                            )
                        labeled_images.append((label, p))
                        attached.add(p)

            # ── Call VLM policy ───────────────────────────────────────────────
            _vlm = type(self.gemini).__name__
            print(f"  [VLM] {_vlm} model={self.gemini.model_name} "
                  f"calling policy (images={len(labeled_images)})")
            raw_response_dict = self.gemini.call_policy(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                labeled_images=labeled_images,
            )
            # Signal renderer to pause before executing this decision
            # (the interactive viewer holds the frame here so a human can read the
            # rationale before the robot moves; headless toolboxes lack this attribute).
            if hasattr(self.toolbox, "vlm_just_decided"):
                self.toolbox.vlm_just_decided = True

            # ── Parse action ─────────────────────────────────────────────────
            # Two degradations, both to `wait` rather than an exception: a step where the
            # model produced nothing usable costs one step, not the episode. `wait` is the
            # safe no-op — it lets the sim settle and gives the model a fresh observation
            # to try again from.
            if not raw_response_dict or "tool" not in raw_response_dict:
                # The client already exhausted its own JSON-repair retries.
                print(f"  [PromptAgent] {self.gemini.model_name} returned no valid "
                      f"action at step {step_idx + 1}")
                action = ToolAction(
                    rationale="VLM returned no valid JSON; waiting one step.",
                    tool="wait",
                    arguments={"seconds": 1},
                    previous_action_verification=(
                        "Previous action verification: unable to verify because "
                        "the VLM response was not valid JSON."
                    ),
                )
            else:
                action = ToolAction.from_dict(raw_response_dict)

            # Sanitise: unknown tool → wait.
            # Preserve the model's previous_action_verification through the swap — that
            # reasoning was about the *last* step and is still valid context.
            if action.tool not in VALID_TOOLS:
                print(f"  [PromptAgent] unknown tool '{action.tool}', using wait")
                action = ToolAction(
                    rationale=f"Tool '{action.tool}' is invalid; waiting one step.",
                    tool="wait",
                    arguments={"seconds": 1},
                    previous_action_verification=action.previous_action_verification,
                )

            if action.previous_action_verification:
                print(f"  previous_action_verification: "
                      f"{action.previous_action_verification}")
            if action.progress_analysis:
                print(f"  progress_analysis: {action.progress_analysis}")
            print(f"  rationale={action.rationale!r}\n"
                  f"  action: {action.tool}  args={action.arguments}")

            # ── Repeat detection ─────────────────────────────────────────────
            # Identity is (tool, canonicalised args) — so re-issuing the same navigate
            # counts as a repeat, but navigating somewhere else does not. Any different
            # action resets the counter to 0, i.e. we only catch *consecutive* loops.
            action_key = f"{action.tool}:{_canonical_args(action.arguments)}"
            if action_key == last_action_key:
                repeat_count += 1
            else:
                repeat_count = 0
            last_action_key = action_key

            # Hard stop. By 10 the warning injected at 3 has demonstrably not helped, and
            # the agent is burning API calls on a wedged state.
            if repeat_count >= 10:
                print(f"  [PromptAgent] repeated same action 10 times — halting.")
                last_result = ToolResult(
                    ok=False, tool="finish",
                    summary="Halted: repeated same action 10 times without progress.",
                )
                logger.log_step(action, last_result, current_obs)
                break

            # ── Execute action ────────────────────────────────────────────────
            # The toolbox validates and either performs the action or returns ok=False
            # with an explanation, which becomes the model's feedback next step.
            result = self.toolbox.execute(action)
            last_result = result

            print(f"  result: ok={result.ok}  {result.summary}")

            # ── Log step ─────────────────────────────────────────────────────
            # success=None: per-step success is unknown; only the episode has a verdict.
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
                # Grows unboundedly across the episode by design: the model's own
                # narrative of what it has achieved is the cheapest long-horizon memory
                # we have, and at ≤ max_steps entries it stays well inside the context.
                transient_memory.append(action.progress_analysis)

            # ── Update observation state ──────────────────────────────────────
            # A tool that produced a new image (e.g. navigate ending somewhere new) has
            # effectively re-observed the world; fold that into current_obs so the state
            # the next prompt describes is not stale. Note the next iteration calls
            # observe() anyway — this mainly keeps `summary` in sync for the logged obs.
            if result.ok and result.data.get("image_path"):
                current_obs["summary"] = result.summary
                if result.data.get("robot_pose"):
                    current_obs["robot_pose"] = result.data["robot_pose"]

            # ── Append to trajectory log (JSONL only — not used in prompt) ────
            # A flattened, human-readable mirror of the step, kept separate from the
            # logger's richer record. Handy when eyeballing an episode.
            trajectory.append({
                "step_idx":         step_idx + 1,
                "previous_action_verification": (
                    action.previous_action_verification or ""
                ),
                "progress_analysis": action.progress_analysis or "",
                "rationale":        action.rationale or "",
                "expected_progress": action.expected_progress or "",
                "tool":             action.tool,
                "arguments":        action.arguments,
                "ok":               result.ok,
                "summary":          result.summary,
            })

            # ── Check terminal conditions ────────────────────────────────────
            # (1) The agent declares itself done. Note `success = result.ok`, not True:
            #     the toolbox gets the final word on whether `finish` was legitimate
            #     (e.g. OVMM checks the object really was placed).
            if action.tool == "finish":
                final_answer = action.arguments.get("answer") or result.data.get("answer")
                success = result.ok
                print(f"  [PromptAgent] finish called. success={success}")
                break

            # (2) The environment declares the task complete — a benchmark's own success
            #     detector fired, so we stop even though the agent hasn't noticed yet.
            if result.data.get("task_done"):
                success = True
                break

        # Falling out of the for-loop without a break == hit max_steps == failure
        # (success stays False).

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
    """Stable string representation of arguments for repeat detection.

    sort_keys makes the comparison insensitive to the key order the VLM happened to emit,
    so {"skill":"grasp","target":"mug"} and {"target":"mug","skill":"grasp"} are correctly
    recognised as the same action. Falls back to str() for anything unserialisable —
    imperfect but still stable enough to catch a literal repeat.
    """
    import json as _json
    try:
        return _json.dumps(args, sort_keys=True)
    except Exception:
        return str(args)
