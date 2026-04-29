"""
agent/toolbox.py — Protocol defining the interface PromptEmbodiedAgent expects.

Any toolbox implementation (ManiSkill, real robot, other sim) must satisfy
this interface to work with PromptEmbodiedAgent.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from .schemas import ToolAction, ToolResult


@runtime_checkable
class ToolboxProtocol(Protocol):
    _last_image_path: Optional[str]

    def observe(self) -> ToolResult: ...
    def execute(self, action: ToolAction) -> ToolResult: ...
