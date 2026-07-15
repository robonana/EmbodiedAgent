"""
agent/schemas.py — Data schemas for PromptEmbodiedAgent and EpisodicMemory.

Uses stdlib dataclasses (no Pydantic dependency).
All fields that reach Gemini prompts are strings or JSON-serialisable primitives.

Memory policy: MemoryEntry is built ONLY from robot sensor data and VLM-generated
observations.  Simulator ground-truth (object names, poses, segmentation) MUST
NOT appear in any MemoryEntry field.

Layout of this module:
  1. Tool-name registry      — the closed vocabulary the VLM is allowed to emit.
  2. Normalisation helpers   — coerce sloppy VLM output into that vocabulary.
  3. Core agent schemas      — ToolAction (VLM -> toolbox) / ToolResult (toolbox -> VLM).
  4. Memory schemas          — what a single episodic memory record looks like on disk.
  5. Retrieval / logging     — MemoryCandidate (search hit) and TrajectoryStep (JSONL row).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


# ── Tool name registry ────────────────────────────────────────────────────────
# This is the *closed vocabulary* of the agent. The VLM is prompted to emit one of
# these names in `ToolAction.tool`; anything else is rejected before it can reach
# the simulator. Keeping the registry here (rather than inside the toolbox) lets the
# validator, the prompt builder and the toolbox all agree on one source of truth.

VALID_TOOLS: frozenset[str] = frozenset({
    # Perception (no side effects on the sim):
    "detect",              # open-vocabulary detection on the current RGB frame
    "inspect",             # ask the VLM a free-form question about the current view
    "retrieve_memory",     # nearest-neighbour lookup over past frames
    "retrieve_trajectory", # replay/lookup of previously executed steps
    # Actuation (mutates simulator state):
    "navigate",            # point-goal / object-goal navigation to a target
    "base_move",           # single discrete base motion primitive (see BASE_MOVE_MOTIONS)
    "manipulate",          # arm skill: grasp / place / drop (see SUPPORTED_SKILLS)
    # Control flow:
    "wait",                # no-op step (lets the sim settle, e.g. after a drop)
    "finish",              # agent declares the task complete -> ends the episode
})

# The only arm skills the manipulation backend implements. `manipulate` actions with
# any other `skill` argument are rejected rather than silently no-oped.
SUPPORTED_SKILLS: frozenset[str] = frozenset({"grasp", "place", "drop"})

# Canonical spellings of the discrete base motions. These exact strings are what the
# toolbox switches on, and what the prompt advertises to the VLM.
BASE_MOVE_MOTIONS: frozenset[str] = frozenset({
    "forward",
    "backward",
    "left",
    "right",
    "rotate 30 degrees",
    "rotate -30 degrees",
})

# VLMs paraphrase. Rather than fail an otherwise-correct action because the model
# wrote "turn left" or "rotate_30", we map every plausible surface form onto the one
# canonical motion string above. Keys are pre-normalised (lowercase, underscores ->
# spaces, whitespace collapsed) — see normalize_base_move_motion().
_BASE_MOVE_ALIASES: dict[str, str] = {
    "forward": "forward",
    "backward": "backward",
    "back": "backward",
    "left": "left",
    "right": "right",
    # Positive yaw == counter-clockwise == "turn left" in Habitat's convention.
    "rotate 30 degrees": "rotate 30 degrees",
    "rotate +30 degrees": "rotate 30 degrees",
    "rotate 30": "rotate 30 degrees",
    "rotate +30": "rotate 30 degrees",
    "rotate_30": "rotate 30 degrees",
    "rotate_30_degrees": "rotate 30 degrees",
    "turn left": "rotate 30 degrees",
    # Negative yaw == clockwise == "turn right".
    "rotate -30 degrees": "rotate -30 degrees",
    "rotate -30": "rotate -30 degrees",
    "rotate_-30": "rotate -30 degrees",
    "rotate_-30_degrees": "rotate -30 degrees",
    "turn right": "rotate -30 degrees",
}


def normalize_base_move_motion(motion: object) -> Optional[str]:
    """Map a VLM-produced motion string onto its canonical form.

    Accepts anything (the VLM may hand us a non-string, or nothing at all), lowercases
    it, turns underscores into spaces and collapses runs of whitespace, then looks the
    result up in _BASE_MOVE_ALIASES.

    Returns the canonical motion (a member of BASE_MOVE_MOTIONS), or None if the input
    is not a motion we support — callers treat None as "reject this action".
    """
    key = str(motion or "").strip().lower().replace("_", " ")
    key = " ".join(key.split())  # collapse internal runs of whitespace
    return _BASE_MOVE_ALIASES.get(key)


def normalize_memory_id(memory_id: object) -> Optional[str]:
    """Normalize memory ids such as mem42 / mem_000042 to mem_000042.

    Memory ids are canonically ``mem_`` + a zero-padded 6-digit counter, but the VLM
    frequently drops the underscore or the padding when it copies an id back out of a
    retrieval result. We accept any ``mem<sep><digits>`` form and re-pad it.

    Returns the canonical id, or None if the input isn't recognisably a memory id (in
    which case the caller reports a lookup failure to the VLM instead of guessing).
    """
    if not isinstance(memory_id, str):
        return None
    raw = memory_id.strip()
    if not raw:
        return None
    # "mem42", "mem-42", "mem_000042" -> re-pad the numeric part to 6 digits.
    numeric = re.fullmatch(r"mem[_-]?(\d+)", raw, flags=re.IGNORECASE)
    if numeric:
        return f"mem_{int(numeric.group(1)):06d}"
    # Already-prefixed but non-numeric suffix (e.g. a scan-phase id): pass through.
    if raw.lower().startswith("mem_"):
        return raw
    return None


# ── Core agent schemas ────────────────────────────────────────────────────────

@dataclass
class ToolAction:
    """One structured tool call output by Gemini.

    This is the *only* channel by which the VLM influences the world: the JSON block it
    emits each step is parsed into one of these. The three optional reasoning fields
    (`previous_action_verification`, `progress_analysis`, `expected_progress`) carry no
    control flow — they are chain-of-thought slots that measurably improve action
    quality and are replayed back into the next prompt as context.
    """
    tool: str                            # must be in VALID_TOOLS
    arguments: dict[str, Any]            # tool-specific; validated by the toolbox
    rationale: str = ""                  # why this action, in the VLM's own words
    expected_progress: Optional[str] = None          # what the VLM predicts will happen
    previous_action_verification: Optional[str] = None  # did the *last* action do what it expected?
    progress_analysis: Optional[str] = None  # agent's self-assessment of task progress

    @classmethod
    def from_dict(cls, d: dict) -> "ToolAction":
        """Build from freshly-parsed VLM JSON.

        Deliberately lenient: missing keys become empty/None rather than raising, so a
        partially-malformed response still yields a ToolAction that the validator can
        reject with a *specific* error message fed back to the VLM.
        """
        return cls(
            tool=str(d.get("tool", "")),
            arguments=d.get("arguments", {}),
            rationale=str(d.get("rationale", "")),
            expected_progress=d.get("expected_progress"),
            previous_action_verification=d.get("previous_action_verification"),
            progress_analysis=d.get("progress_analysis"),
        )

    def to_dict(self) -> dict:
        """Serialise for the trajectory log and for replay into later prompts.

        Key order is intentional: it mirrors the order the VLM is asked to *think* in
        (verify last action -> analyse progress -> justify -> act -> predict), so the
        replayed history reads as a coherent monologue.
        """
        return {
            "previous_action_verification": self.previous_action_verification,
            "progress_analysis": self.progress_analysis,
            "rationale": self.rationale,
            "tool": self.tool,
            "arguments": self.arguments,
            "expected_progress": self.expected_progress,
        }


@dataclass
class ToolResult:
    """Structured result returned by every AgentToolbox method.

    Uniform across all tools so the agent loop never has to special-case a tool's
    return type:
      - `ok`          : did the tool run without error? (a *failed grasp* is still ok=True;
                        ok=False means the tool itself could not execute)
      - `summary`     : one-line natural-language result, injected verbatim into the next prompt
      - `data`        : machine-readable payload (detections, poses, memory hits, ...)
      - `image_paths` : any frames produced; the agent loop attaches these to the next VLM call
    """
    ok: bool
    tool: str
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    image_paths: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SensorData:
    """Raw sensor readings at the time of capture.

    Sensor-only by design — see the memory policy in the module docstring. The image is
    stored as a path, not bytes, so memory records stay small and JSON-serialisable.
    """
    image_path: str
    robot_pose: list[float]       # [x, y, yaw_rad]
    timestamp: str                # HH:MM:SS or ISO8601


@dataclass
class EmbeddingRefs:
    """Paths to pre-computed embeddings (populated by EmbeddingWorker).

    Embeddings live in .npy files beside the images rather than inline, so the memory
    index stays human-readable and the (large) vectors can be memory-mapped or rebuilt
    with a different model without rewriting the records.
    """
    image_embedding_path: Optional[str] = None
    embedding_model: str = "unknown"   # which encoder produced the vector, for invalidation


@dataclass
class MemorySource:
    """Provenance of a memory entry.

    Distinguishes frames captured during the offline scene scan ("scan_wasd", a
    teleoperated sweep of the scene done before the episode) from frames the agent
    captured itself while acting ("agent_observe"). Retrieval can weight or filter on this.
    """
    source_type: str = "agent_observe"   # "scan_wasd" | "agent_observe"
    episode_id: Optional[str] = None     # None for pre-episode scan frames


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
        """Rehydrate from a JSON row of the on-disk memory index.

        `sensor` and `memory_id` are required (a record without them is corrupt and we
        want the KeyError); `embeddings` and `source` are optional so that indexes
        written by older versions — before embeddings/provenance existed — still load.
        """
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
    """One candidate returned by retrieve_memory, enriched from EpisodicMemory.

    A flattened view of a MemoryEntry plus its similarity score. Flattened because this
    is what gets rendered into the prompt — the VLM sees a list of (id, pose, score)
    triples alongside the retrieved images, and may cite an id in a later `navigate`.
    """
    memory_id: str                  # e.g. "mem_000042"
    image_path: str                 # also exposed as rgb_path in agent-facing dicts
    robot_pose: list[float]         # [x, y, yaw_rad]
    retrieval_score: float          # higher == more similar to the query
    timestamp: Optional[str] = None
    frame_idx: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def pose_str(self) -> str:
        """Human/VLM-readable pose, with yaw in degrees.

        Radians are what the sim uses but degrees are what the VLM reasons about
        reliably, so the conversion happens here at the presentation boundary.
        """
        if len(self.robot_pose) >= 3:
            import math
            x, y, yaw = self.robot_pose[:3]
            return f"({x:.2f}, {y:.2f}, yaw={math.degrees(yaw):.0f}°)"
        # Degenerate pose (shouldn't happen) — fall back to the raw list rather than crash.
        return str(self.robot_pose)


@dataclass
class TrajectoryStep:
    """One step in the episode trajectory log (written to JSONL).

    Captures everything needed to (a) score the episode offline, (b) replay it in the
    viewer, and (c) reconstruct the exact prompt that produced the action — the raw VLM
    text and the full prompt are written to separate files and referenced by path here,
    keeping each JSONL line small enough to stream.
    """
    episode_id: str
    step_idx: int
    task: str
    timestamp: str
    current_obs: dict[str, Any]      # full observation dict before the action
    action: dict[str, Any]           # serialised ToolAction
    result: dict[str, Any]           # serialised ToolResult (includes image_paths)
    raw_gemini_path: Optional[str] = None   # path to raw Gemini text file
    prompt_path: Optional[str] = None       # path to full prompt text file
    success: Optional[bool] = None   # set on the final step only; None while in progress

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        """One JSONL line — no trailing newline; the writer adds it."""
        return json.dumps(self.to_dict())
