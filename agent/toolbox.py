"""
agent/toolbox.py — Protocol defining the interface PromptEmbodiedAgent expects.

Any toolbox implementation (ManiSkill, real robot, other sim) must satisfy
this interface to work with PromptEmbodiedAgent.

The agent loop is deliberately backend-agnostic: it only ever calls observe() and
execute(), so swapping Habitat for a real robot means writing a new class that
satisfies this Protocol — no change to the agent, the prompts, or the schemas.

Structural (not nominal) typing: implementations do NOT subclass this. `Protocol`
means "any object with these members type-checks", and `@runtime_checkable` lets
`isinstance(obj, ToolboxProtocol)` verify member *presence* at runtime (it does not
check signatures).
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from .schemas import ToolAction, ToolResult


@runtime_checkable
class ToolboxProtocol(Protocol):
    # Path to the most recently captured RGB frame. The agent reads this directly to
    # decide which image to attach to the next VLM call, so it is part of the contract
    # rather than an implementation detail. None before the first observe().
    _last_image_path: Optional[str]

    def observe(self) -> ToolResult:
        """Capture the current sensor state (RGB frame, pose, held object, ...).

        Called once per agent step *before* the VLM is prompted. Must have no side
        effects on the world.
        """
        ...

    def execute(self, action: ToolAction) -> ToolResult:
        """Validate and run one VLM-proposed action, returning a uniform ToolResult.

        Implementations own validation: an action naming an unknown tool, or carrying
        bad arguments, must come back as ok=False with an explanatory `summary` (which
        is fed to the VLM as feedback) rather than raising.
        """
        ...
