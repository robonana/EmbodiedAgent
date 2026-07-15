"""
agent/verifier.py — Argument validation for agent tool calls.

NOT a learned precondition scorer.  Rule-based checks only.

This is the gate between the VLM's free-form JSON and the simulator. It runs *before*
any action touches the world, is pure (never calls the sim or the VLM), and is cheap
enough to run on every step. Its failure strings ("invalid_arguments", ...) are surfaced
back to the VLM as feedback so the model can correct itself on the next step, which is
why they're stable machine-readable tokens rather than prose.

Philosophy: check *structure*, not *semantics*. "Is `bbox` four numbers?" is our job.
"Is that bbox actually around the mug?" is the world's job to answer.
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
    """Return (is_valid, error_message).  Checks parseable + is a dict.

    The top-level type check matters: a model that emits a bare list or string produces
    valid JSON that would then blow up on `.get()` downstream. Failing here yields a
    message we can hand back to the VLM ("Expected JSON object, got list").
    """
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

    Tools not named below (`detect`, `wait`, `finish`) have no required arguments and
    fall through to the trailing `return True, None`.
    """
    if tool not in VALID_TOOLS:
        return False, "invalid_tool"

    if tool == "inspect":
        # Needs *an* image (either a single path or a non-empty list) and *a* question;
        # without both there is nothing to ask the VLM about.
        image_paths = arguments.get("image_paths")
        if not arguments.get("image_path") and not (
            isinstance(image_paths, (list, tuple)) and len(image_paths) > 0
        ):
            return False, "invalid_arguments"
        if not arguments.get("question"):
            return False, "invalid_arguments"
        # bbox is optional (it crops/highlights the region to ask about), but if given it
        # must be well-formed or the crop will silently produce garbage.
        bbox = arguments.get("bbox")
        if bbox is not None:
            if not isinstance(bbox, (list, tuple)) or len(bbox) == 0:
                return False, "invalid_arguments"
            # Accept single box [x1,y1,x2,y2] or list of boxes [[...], ...]
            # Disambiguate by peeking at the first element: a number means it's one flat box.
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
        # top_k == -1 is the sentinel for "return everything"; anything else must be >= 1.
        top_k = arguments.get("top_k")
        if top_k is not None and int(top_k) != -1 and int(top_k) < 1:
            return False, "invalid_arguments"
        # Time filters are compared as strings against the stored timestamps, so a
        # non-string here (e.g. the VLM emitting a number) would silently match nothing.
        for tf in ("time_from", "time_to"):
            val = arguments.get(tf)
            if val is not None and not isinstance(val, str):
                return False, "invalid_arguments"

    elif tool == "retrieve_trajectory":
        # Unlike retrieve_memory, the time window is mandatory here — it is the query.
        if not arguments.get("time_from") or not arguments.get("time_to"):
            return False, "invalid_arguments"

    elif tool == "navigate":
        # navigate only accepts a memory_id (a retrieve_memory candidate);
        # raw coordinate/pose goals are not allowed.
        #
        # This is a deliberate capability restriction, not an oversight: letting the VLM
        # emit raw (x, y) goals would let it navigate to coordinates it hallucinated or
        # read out of ground-truth, bypassing perception. Forcing every goal to be a
        # previously-observed memory frame keeps the agent honest — it can only go
        # somewhere it has actually seen.
        tgt = arguments.get("target")
        if isinstance(tgt, str):
            # Bare-string form: target is the memory id itself.
            if normalize_memory_id(tgt) is None:
                return False, "invalid_arguments"
        elif isinstance(tgt, dict):
            # Dict form: accept the several key names the VLM tends to invent.
            has_memory_id = normalize_memory_id(
                tgt.get("memory_id") or tgt.get("mem_id") or tgt.get("memory")
            ) is not None
            if not has_memory_id:
                return False, "invalid_arguments"
        else:
            # Anything else (a list of coords, a number, None) is a coordinate goal in
            # disguise, or nonsense. Reject.
            return False, "invalid_arguments"

    elif tool == "base_move":
        # normalize_* folds aliases ("turn left") into canonical motions; if the result
        # still isn't in the registry, the motion is one we cannot execute.
        if normalize_base_move_motion(arguments.get("motion")) not in BASE_MOVE_MOTIONS:
            return False, "invalid_arguments"

    elif tool == "manipulate":
        if not arguments.get("skill"):
            return False, "invalid_arguments"
        # Distinct reason code: the model asked for a *coherent* skill we simply don't
        # implement (e.g. "push"), which is worth telling it apart from malformed args.
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

    This is the harness's *own* opinion about whether a step helped, kept separate from
    the VLM's self-assessment (ToolAction.previous_action_verification) so the two can be
    compared during evaluation. The tri-state return is important: None means "we
    genuinely cannot tell from the backend", and is not the same as False.
    """
    # A tool that failed to execute cannot have made progress, whatever it was.
    if not result_ok:
        return False

    # Pure perception/retrieval: progress == we learned something (non-empty payload).
    if tool in ("inspect", "retrieve_memory", "retrieve_trajectory"):
        return bool(result_data)

    if tool == "navigate":
        # Navigation "succeeds" whenever the planner terminates, so use the residual
        # distance instead. 2.0 m is roughly interaction range — close enough that the
        # target should now be visible and graspable.
        dist = result_data.get("distance_to_goal")
        if dist is not None:
            return float(dist) < 2.0
        return result_ok  # backend didn't report a distance; fall back to the ok flag

    if tool == "base_move":
        # `completed` is False when the motion was blocked by a collision — the robot
        # tried to move and didn't, which is the definition of no progress.
        return bool(result_data.get("completed", False))

    if tool == "manipulate":
        skill = result_data.get("skill", "")
        if skill == "grasp":
            grasped = result_data.get("grasped")
            if grasped is not None:
                return bool(grasped)  # ground-truth grasp state from the backend
            return None  # backend can't introspect; policy verifies from observation
        # place/drop: no cheap ground-truth signal, so trust that the skill ran.
        return result_ok

    # wait / finish / detect: no meaningful notion of progress.
    return None
