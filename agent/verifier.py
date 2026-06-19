"""
agent/verifier.py — Argument validation for agent tool calls.

NOT a learned precondition scorer.  Rule-based checks only.
"""

from __future__ import annotations

from typing import Any, Optional

from .schemas import (
    BASE_MOVE_MOTIONS,
    SUPPORTED_SKILLS,
    VALID_TOOLS,
    normalize_base_move_motion,
    normalize_memory_id,
)


# ── JSON / action validation ──────────────────────────────────────────────────

def check_json_validity(raw: str) -> tuple[bool, Optional[str]]:
    """Return (is_valid, error_message).  Checks parseable + is a dict."""
    import json
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return False, f"Expected JSON object, got {type(obj).__name__}"
        return True, None
    except json.JSONDecodeError as e:
        return False, f"JSON decode error: {e}"


def check_tool_argument_validity(
    tool: str,
    arguments: dict[str, Any],
) -> tuple[bool, Optional[str]]:
    """
    Lightweight argument validation — only checks structure, not semantics.
    Never calls the simulator or Gemini.
    Returns (is_valid, failure_reason).
    """
    if tool not in VALID_TOOLS:
        return False, "invalid_tool"

    if tool == "inspect":
        image_paths = arguments.get("image_paths")
        if not arguments.get("image_path") and not (
            isinstance(image_paths, (list, tuple)) and len(image_paths) > 0
        ):
            return False, "invalid_arguments"
        if not arguments.get("question"):
            return False, "invalid_arguments"
        bbox = arguments.get("bbox")
        if bbox is not None:
            if not isinstance(bbox, (list, tuple)) or len(bbox) == 0:
                return False, "invalid_arguments"
            # Accept single box [x1,y1,x2,y2] or list of boxes [[...], ...]
            if isinstance(bbox[0], (int, float)):
                boxes = [bbox]
            else:
                boxes = bbox
            for box in boxes:
                if not (isinstance(box, (list, tuple)) and len(box) == 4):
                    return False, "invalid_arguments"
                if any(not isinstance(v, (int, float)) for v in box):
                    return False, "invalid_arguments"

    elif tool == "retrieve_memory":
        if not arguments.get("query"):
            return False, "invalid_arguments"
        top_k = arguments.get("top_k")
        if top_k is not None and int(top_k) != -1 and int(top_k) < 1:
            return False, "invalid_arguments"
        for tf in ("time_from", "time_to"):
            val = arguments.get(tf)
            if val is not None and not isinstance(val, str):
                return False, "invalid_arguments"

    elif tool == "retrieve_trajectory":
        if not arguments.get("time_from") or not arguments.get("time_to"):
            return False, "invalid_arguments"

    elif tool == "navigate":
        # navigate only accepts a memory_id (a retrieve_memory candidate);
        # raw coordinate/pose goals are not allowed.
        tgt = arguments.get("target")
        if isinstance(tgt, str):
            if normalize_memory_id(tgt) is None:
                return False, "invalid_arguments"
        elif isinstance(tgt, dict):
            has_memory_id = normalize_memory_id(
                tgt.get("memory_id") or tgt.get("mem_id") or tgt.get("memory")
            ) is not None
            if not has_memory_id:
                return False, "invalid_arguments"
        else:
            return False, "invalid_arguments"

    elif tool == "base_move":
        if normalize_base_move_motion(arguments.get("motion")) not in BASE_MOVE_MOTIONS:
            return False, "invalid_arguments"

    elif tool == "manipulate":
        if not arguments.get("skill"):
            return False, "invalid_arguments"
        if arguments["skill"] not in SUPPORTED_SKILLS:
            return False, "unsupported_skill"
        if not arguments.get("target"):
            return False, "invalid_arguments"

    return True, None


# ── Progress inference ────────────────────────────────────────────────────────

def infer_progress_from_tool_result(
    tool: str,
    result_ok: bool,
    result_data: dict[str, Any],
) -> Optional[bool]:
    """
    Infer whether progress was made based on the tool and result.
    Returns True / False / None (unknown).
    """
    if not result_ok:
        return False

    if tool in ("inspect", "retrieve_memory", "retrieve_trajectory"):
        return bool(result_data)

    if tool == "navigate":
        dist = result_data.get("distance_to_goal")
        if dist is not None:
            return float(dist) < 2.0
        return result_ok

    if tool == "base_move":
        return bool(result_data.get("completed", False))

    if tool == "manipulate":
        skill = result_data.get("skill", "")
        if skill == "grasp":
            return None  # next policy response must verify grasp success from observation
        return result_ok

    return None
