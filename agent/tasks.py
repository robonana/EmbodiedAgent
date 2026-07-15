"""
agent/tasks.py — Task configuration for the prompt-agent baseline.

No task-specific tools are defined here.  This module only provides
lightweight metadata (success criteria hints, suggested start tools)
used for logging and evaluation — not exposed to the Gemini policy.

The separation matters for the claim the baseline makes: the policy sees *only* the
free-form task sentence, exactly as a user would type it. Everything in this file
(category, success hints, step budgets) is consumed by the harness — for grouping runs
in eval and for sizing the step limit — and never enters a prompt. If any of it leaked
into the prompt, the agent would be getting task-specific priors it is supposed to
derive on its own.
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
    # Coarse bucket used to aggregate success rates across runs of the same *kind* of task.
    category: str = "object_retrieval"   # object_retrieval | search | qa | cleanup | maintenance
    # Natural-language statement of what "done" means, read by the offline verifier /
    # by a human grading an episode. Never shown to the policy.
    success_condition_hint: Optional[str] = None   # for eval only
    # Suggested first tool — agent may choose differently
    suggested_first_tool: str = "observe"
    # Rough step budget for this task type; the runner uses it to size --max_steps when
    # the user doesn't pass one. A hint, not a hard cap enforced here.
    max_steps_hint: int = 40


# ── Lightweight task registry ─────────────────────────────────────────────────
# These are example task strings that demonstrate the baseline's capabilities.
# The agent treats them as free-form task strings.
#
# The five entries span the behaviour classes we care about: fetch an object,
# re-find something seen earlier (exercises episodic memory), answer a question about
# a scene (terminates via `finish` with an answer rather than an action), and two
# open-ended multi-object tasks that need a bigger step budget.

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
        # Pure QA: no navigation needed if the desk is already in view, so start by looking.
        suggested_first_tool="inspect",
    ),
    "clean_table": TaskConfig(
        task_str="Clean the table.",
        category="cleanup",
        success_condition_hint="the table surface is clear",
        # Multi-object: one pick-and-place cycle per item, so budget more steps.
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
    """Return a TaskConfig for the given task string, or a generic one.

    Matching is by exact (case-insensitive) sentence, not by key: callers pass the task
    the way the user phrased it, and only an exact match can safely inherit that task's
    hand-written success hint. Anything unrecognised gets a default TaskConfig, so an
    arbitrary user-supplied task still runs — just without eval metadata.
    """
    task_lower = task_str.lower()
    for cfg in DEMO_TASKS.values():
        if cfg.task_str.lower() == task_lower:
            return cfg
    # Generic fallback
    return TaskConfig(task_str=task_str)
