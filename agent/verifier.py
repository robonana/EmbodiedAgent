"""
agent/verifier.py — Argument validation for agent tool calls.

NOT a learned precondition scorer.  Rule-based checks only.
"""

from __future__ import annotations

from typing import Any, Optional

from .schemas import VALID_TOOLS, SUPPORTED_SKILLS


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
        if not arguments.get("image_path"):
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
        tgt = arguments.get("target")
        if not isinstance(tgt, dict):
            return False, "invalid_arguments"
        if not tgt.get("pose"):
            return False, "invalid_arguments"

    elif tool == "approach":
        if not arguments.get("target"):
            return False, "invalid_arguments"

    elif tool == "manipulate":
        if not arguments.get("skill"):
            return False, "invalid_arguments"
        if arguments["skill"] not in SUPPORTED_SKILLS:
            return False, "unsupported_skill"
        if not arguments.get("target"):
            return False, "invalid_arguments"

    elif tool == "verify":
        if not arguments.get("condition"):
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

    if tool == "approach":
        return result_data.get("reached_desired_distance", False)

    if tool == "manipulate":
        skill = result_data.get("skill", "")
        if skill == "grasp":
            return None  # agent must call verify() to confirm grasp success
        return result_ok

    if tool == "verify":
        satisfied = result_data.get("satisfied")
        return satisfied if satisfied is not None else None

    return None


