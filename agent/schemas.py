"""
agent/schemas.py — Data schemas for PromptEmbodiedAgent and EpisodicMemory.

Uses stdlib dataclasses (no Pydantic dependency).
All fields that reach Gemini prompts are strings or JSON-serialisable primitives.

Memory policy: MemoryEntry is built ONLY from robot sensor data and VLM-generated
observations.  Simulator ground-truth (object names, poses, segmentation) MUST
NOT appear in any MemoryEntry field.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


# ── Tool name registry ────────────────────────────────────────────────────────

VALID_TOOLS: frozenset[str] = frozenset({
    "detect", "inspect", "retrieve_memory", "retrieve_trajectory",
    "navigate", "approach", "manipulate", "verify", "wait", "finish",
})

SUPPORTED_SKILLS: frozenset[str] = frozenset({"grasp", "place", "drop"})




# ── Core agent schemas ────────────────────────────────────────────────────────

@dataclass
class ToolAction:
    """One structured tool call output by Gemini."""
    tool: str                            # must be in VALID_TOOLS
    arguments: dict[str, Any]
    rationale: str = ""
    expected_progress: Optional[str] = None
    progress_analysis: Optional[str] = None  # agent's self-assessment of task progress

    @classmethod
    def from_dict(cls, d: dict) -> "ToolAction":
        return cls(
            tool=str(d.get("tool", "")),
            arguments=d.get("arguments", {}),
            rationale=str(d.get("rationale", "")),
            expected_progress=d.get("expected_progress"),
            progress_analysis=d.get("progress_analysis"),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ToolResult:
    """Structured result returned by every AgentToolbox method."""
    ok: bool
    tool: str
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    image_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)



@dataclass
class SensorData:
    """Raw sensor readings at the time of capture."""
    image_path: str
    robot_pose: list[float]       # [x, y, yaw_rad]
    timestamp: str                # HH:MM:SS or ISO8601


@dataclass
class EmbeddingRefs:
    """Paths to pre-computed embeddings (populated by EmbeddingWorker)."""
    image_embedding_path: Optional[str] = None
    embedding_model: str = "unknown"


@dataclass
class MemorySource:
    """Provenance of a memory entry."""
    source_type: str = "agent_observe"   # "scan_wasd" | "agent_observe"
    episode_id: Optional[str] = None


@dataclass
class MemoryEntry:
    """
    One episodic memory record.

    Built exclusively from robot sensor data (pose + image path).
    NEVER stores simulator ground-truth (no oracle names, poses, or segmentation).
    """
    memory_id: str
    sensor: SensorData
    embeddings: EmbeddingRefs = field(default_factory=EmbeddingRefs)
    source: MemorySource = field(default_factory=MemorySource)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        s = d["sensor"]
        sensor = SensorData(
            image_path=s["image_path"],
            robot_pose=s["robot_pose"],
            timestamp=s["timestamp"],
        )
        emb = d.get("embeddings", {})
        embeddings = EmbeddingRefs(
            image_embedding_path=emb.get("image_embedding_path"),
            embedding_model=emb.get("embedding_model", "unknown"),
        )
        src = d.get("source", {})
        source = MemorySource(
            source_type=src.get("source_type", "agent_observe"),
            episode_id=src.get("episode_id"),
        )
        return cls(
            memory_id=d["memory_id"],
            sensor=sensor,
            embeddings=embeddings,
            source=source,
        )


# ── Retrieval candidate ───────────────────────────────────────────────────────

@dataclass
class MemoryCandidate:
    """One candidate returned by retrieve_memory, enriched from EpisodicMemory."""
    memory_id: str                  # e.g. "mem_000042"
    image_path: str                 # also exposed as rgb_path in agent-facing dicts
    robot_pose: list[float]         # [x, y, yaw_rad]
    retrieval_score: float
    timestamp: Optional[str] = None
    frame_idx: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def pose_str(self) -> str:
        if len(self.robot_pose) >= 3:
            import math
            x, y, yaw = self.robot_pose[:3]
            return f"({x:.2f}, {y:.2f}, yaw={math.degrees(yaw):.0f}°)"
        return str(self.robot_pose)


@dataclass
class TrajectoryStep:
    """One step in the episode trajectory log (written to JSONL)."""
    episode_id: str
    step_idx: int
    task: str
    timestamp: str
    current_obs: dict[str, Any]      # full observation dict before the action
    action: dict[str, Any]           # serialised ToolAction
    result: dict[str, Any]           # serialised ToolResult (includes image_paths)
    raw_gemini_path: Optional[str] = None   # path to raw Gemini text file
    prompt_path: Optional[str] = None       # path to full prompt text file
    success: Optional[bool] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())
