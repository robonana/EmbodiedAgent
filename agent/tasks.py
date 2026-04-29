"""
agent/tasks.py — Task configuration for the prompt-agent baseline.

No task-specific tools are defined here.  This module only provides
lightweight metadata (success criteria hints, suggested start tools)
used for logging and evaluation — not exposed to the Gemini policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TaskConfig:
    """
    Optional metadata for a demo task.  All fields are for evaluation /
    logging only — the Gemini policy receives only the task string.
    """
    task_str: str
    category: str = "object_retrieval"   # object_retrieval | search | qa | cleanup | maintenance
    success_condition_hint: Optional[str] = None   # for eval only
    # Suggested first tool — agent may choose differently
    suggested_first_tool: str = "observe"
    max_steps_hint: int = 40


# ── Lightweight task registry ─────────────────────────────────────────────────
# These are example task strings that demonstrate the baseline's capabilities.
# The agent treats them as free-form task strings.

DEMO_TASKS: dict[str, TaskConfig] = {
    "bring_water_bottle": TaskConfig(
        task_str="Bring me the water bottle.",
        category="object_retrieval",
        success_condition_hint="the robot is holding the water bottle",
    ),
    "find_mug": TaskConfig(
        task_str="Find the mug I saw earlier.",
        category="search",
        success_condition_hint="the mug is visible in the current view",
    ),
    "describe_desk": TaskConfig(
        task_str="What is on the computer desk?",
        category="qa",
        success_condition_hint="finish with a descriptive answer",
        suggested_first_tool="inspect",
    ),
    "clean_table": TaskConfig(
        task_str="Clean the table.",
        category="cleanup",
        success_condition_hint="the table surface is clear",
        max_steps_hint=60,
    ),
    "keep_board_clean": TaskConfig(
        task_str="Keep the board clean.",
        category="maintenance",
        success_condition_hint="the board has no visible writing",
        max_steps_hint=40,
    ),
}


def get_task_config(task_str: str) -> TaskConfig:
    """Return a TaskConfig for the given task string, or a generic one."""
    task_lower = task_str.lower()
    for cfg in DEMO_TASKS.values():
        if cfg.task_str.lower() == task_lower:
            return cfg
    # Generic fallback
    return TaskConfig(task_str=task_str)
