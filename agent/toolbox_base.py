"""
agent/toolbox_base.py — BaseToolbox: backend-independent agent toolbox.

Contains all tool logic and state management.  Backend-specific behaviour is
delegated to abstract primitive methods that each backend must implement:

    _step()                        → dict          advance one control cycle; return obs dict
    _capture_rgb(obs)              → ndarray|None  extract RGB from obs
    _get_robot_pose()              → [x, y, yaw]   current robot pose
    _navigate_step(bearing)        → None          send one navigation command toward bearing
    _base_move_step(motion)        → None          send one discrete base motion tick
    _plan_path(start, goal)        → [ndarray]     list of XY waypoints
    _grasp(target)                 → (localized, name, dist)  # localized=attempted, not succeeded
    _release(...)                  → (ok, summary)
    _forward_step()                → None          send one forward movement command
    _get_depth_and_intrinsics(obs) → (depth, K, E) | None
"""

from __future__ import annotations

import math
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .schemas import (
    BASE_MOVE_MOTIONS,
    ToolAction,
    ToolResult,
    SUPPORTED_SKILLS,
    VALID_TOOLS,
    normalize_base_move_motion,
    normalize_memory_id,
)
from .verifier import check_tool_argument_validity


# ── Time-window helpers ───────────────────────────────────────────────────────

def _parse_hhmmss(s: Optional[str]) -> Optional[int]:
    """'HH:MM:SS' or 'HH:MM' → total seconds since midnight."""
    if not s:
        return None
    try:
        parts = s.strip().split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        pass
    return None


def _in_time_window(
    timestamp: Optional[str],
    t_from: Optional[int],
    t_to: Optional[int],
) -> bool:
    ts = _parse_hhmmss(timestamp)
    if ts is None:
        return True
    if t_from is not None and ts < t_from:
        return False
    if t_to is not None and ts > t_to:
        return False
    return True


# ── BaseToolbox ───────────────────────────────────────────────────────────────

class BaseToolbox(ABC):
    """
    Backend-independent implementation of all agent tools.

    Concrete subclasses implement the primitive methods below for either
    a simulator or a real robot.
    """

    _NAV_MAX_STEPS  = 2000
    _BASE_MOVE_MAX_STEPS = 120
    _BASE_MOVE_DISTANCE_M = 0.30
    _BASE_MOVE_ROTATION_RAD = math.radians(30)
    _EVENT_EVERY    = 10

    def __init__(
        self,
        gemini_client,
        log_dir: str,
        capture_out_dir: str,
        embedding_worker=None,
        episodic_memory=None,
        retrieval_model: str = "siglip_base",
        retrieval_data_root: Optional[str] = None,
        scene_id: Optional[str] = None,
        event_callback: Optional[Callable] = None,
        grounding_dino=None,
    ):
        self.gemini          = gemini_client
        self.log_dir         = Path(log_dir)
        self.capture_out_dir = Path(capture_out_dir)
        self.embedding_worker  = embedding_worker
        self.episodic_memory   = episodic_memory
        self.retrieval_model   = retrieval_model
        self.retrieval_data_root = retrieval_data_root or str(self.capture_out_dir.parent)
        self.scene_id        = scene_id or self.capture_out_dir.name
        self.event_callback  = event_callback
        self._grounding_dino = grounding_dino

        self._last_obs:        Optional[dict]       = None
        self._last_rgb:        Optional[np.ndarray] = None
        self._last_image_path: Optional[str]        = None
        self._step_counter:    int                  = 0

        for sub in ("images", "crops"):
            (self.log_dir / sub).mkdir(parents=True, exist_ok=True)

    # ── Abstract primitives ───────────────────────────────────────────────────

    @abstractmethod
    def _step(self) -> dict:
        """Advance one control cycle and return the current obs dict.

        Sim backend: step physics + render.
        Real-robot backend: sleep one control period, read sensors.
        """

    @abstractmethod
    def _capture_rgb(self, obs: dict) -> Optional[np.ndarray]:
        """Extract RGB ndarray (H,W,3 uint8) from obs, or None."""

    def _grasp_state(self) -> Optional[dict]:
        """Ground-truth grasp state, or None if the backend cannot introspect it.

        Returns ``{"grasped": bool, "object": Optional[str]}`` where ``grasped``
        is whether the gripper is holding anything and ``object`` is the handle
        of the held object (if known). Sim backends override this from the
        physics grasp manager; real-robot/MCP backends without a reliable held
        signal return None so the policy falls back to visual judgment.
        """
        return None

    @abstractmethod
    def _get_robot_pose(self) -> list[float]:
        """Return current robot pose as [x, y, yaw]."""

    @abstractmethod
    def _navigate_step(self, bearing: float) -> None:
        """Advance robot one kinematic step toward bearing (radians)."""

    @abstractmethod
    def _base_move_step(self, motion: str) -> None:
        """Advance the robot one small base-motion tick."""

    @abstractmethod
    def _plan_path(
        self, start_xy: np.ndarray, goal_xy: np.ndarray
    ) -> list[np.ndarray]:
        """Return list of XY waypoints from start to goal."""

    @abstractmethod
    def _grasp(self, target: str) -> tuple[bool, str, float]:
        """Attempt to grasp the object the target pixel lands on.
        Returns (localized, object_name, distance_m), where `localized` means a
        graspable object was found and a grasp was ATTEMPTED — not that the
        grasp succeeded. Whether the object is actually held is left to the
        verify step; the tool must not decide task success here."""

    @abstractmethod
    def _release(
        self,
        target: str = "",
        destination: Optional[str] = None,
        target_region: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Release held object. Returns (success, short_summary)."""

    @abstractmethod
    def _forward_step(self) -> None:
        """Move robot forward one small step."""

    @abstractmethod
    def _get_depth_and_intrinsics(
        self, obs: dict
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Return (depth_HxW float32, K_3x3, E_4x4) from obs, or None."""

    # ── Tool dispatcher ───────────────────────────────────────────────────────

    def execute(self, action: ToolAction) -> ToolResult:
        tool = action.tool
        args = action.arguments or {}

        if tool not in VALID_TOOLS:
            return ToolResult(ok=False, tool=tool,
                              summary=f"Unknown tool '{tool}'.")

        valid, reason = check_tool_argument_validity(tool, args)
        if not valid:
            return ToolResult(ok=False, tool=tool,
                              summary=f"Invalid arguments for '{tool}': {reason}.")

        try:
            dispatch = {
                "detect":              self.detect,
                "inspect":             self.inspect,
                "retrieve_memory":     self.retrieve_memory,
                "retrieve_trajectory": self.retrieve_trajectory,
                "navigate":            self.navigate,
                "base_move":           self.base_move,
                "manipulate":          self.manipulate,
                "wait":                self.wait,
                "finish":              self.finish,
            }
            return dispatch[tool](**args)
        except Exception as e:
            print(f"[Toolbox] ERROR in {tool}: {e}")
            return ToolResult(ok=False, tool=tool,
                              summary=f"{tool} raised an unexpected error: {e}")

    # ── observe ───────────────────────────────────────────────────────────────

    def observe(self) -> ToolResult:
        try:
            obs = self._step()
            self._pump_events()

            rgb = self._capture_rgb(obs)
            if rgb is None:
                return ToolResult(ok=False, tool="observe",
                                  summary="Camera returned no image.")

            img_path = self._save_frame(rgb, "obs")
            self._last_obs        = obs
            self._last_rgb        = rgb
            self._last_image_path = img_path

            pose    = self._get_robot_pose()
            summary = f"Camera captured. pose={self._pose_str(pose)}"
            return ToolResult(
                ok=True, tool="observe", summary=summary,
                data={"robot_pose": pose, "image_path": img_path},
                image_paths=[img_path],
            )
        except Exception as e:
            return ToolResult(ok=False, tool="observe",
                              summary=f"observe failed: {e}")

    # ── inspect ───────────────────────────────────────────────────────────────

    def _save_detection_overlay(
        self,
        image_path: str,
        image: np.ndarray,
        query: str,
        detections: list[dict],
    ) -> Optional[str]:
        """Save an annotated copy of a detect() image with bbox overlays."""
        if not detections:
            return None
        try:
            from PIL import Image as PILImage, ImageDraw

            img = PILImage.fromarray(image.astype(np.uint8)).convert("RGB")
            draw = ImageDraw.Draw(img)
            W, H = img.size
            for i, det in enumerate(detections, 1):
                bbox = det.get("bbox", [])
                if len(bbox) < 4:
                    continue
                x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
                x1 = max(0, min(W - 1, x1))
                y1 = max(0, min(H - 1, y1))
                x2 = max(x1 + 1, min(W, x2))
                y2 = max(y1 + 1, min(H, y2))
                color = (255, 50, 50)
                draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
                label = str(det.get("label") or query)
                score = det.get("score")
                suffix = f" {score:.2f}" if isinstance(score, (int, float)) else ""
                text = f"{i}: {label}{suffix}"
                text_bbox = draw.textbbox((x1, y1), text)
                text_h = text_bbox[3] - text_bbox[1]
                ty = max(0, y1 - text_h - 6)
                draw.rectangle(
                    (x1, ty, min(W, x1 + text_bbox[2] - text_bbox[0] + 8),
                     min(H, ty + text_h + 6)),
                    fill=(0, 0, 0),
                )
                draw.text((x1 + 4, ty + 3), text, fill=(255, 255, 255))

            src = Path(image_path)
            safe_query = re.sub(r"[^A-Za-z0-9_.-]+", "_", query.strip()).strip("_")
            safe_query = safe_query[:48] or "query"
            out_path = src.with_name(f"{src.stem}_detect_{safe_query}{src.suffix}")
            img.save(out_path)
            return str(out_path)
        except Exception as exc:
            print(f"[detect] bbox overlay save failed: {exc}", flush=True)
            return None

    def detect(self, image_path: str, query: str, **_) -> ToolResult:
        """GroundingDINO open-set bounding box proposals."""
        if self._grounding_dino is None:
            return ToolResult(ok=False, tool="detect",
                              summary="GroundingDINO not available.")
        try:
            from PIL import Image as PILImage
            img = np.array(PILImage.open(image_path).convert("RGB"))
        except Exception as e:
            return ToolResult(ok=False, tool="detect",
                              summary=f"Could not load image: {e}")
        try:
            detections = self._grounding_dino.detect(img, query)
        except Exception as e:
            return ToolResult(ok=False, tool="detect",
                              summary=f"GroundingDINO error: {e}")
        if not detections:
            return ToolResult(ok=False, tool="detect",
                              summary=f"No detections for '{query}'.",
                              data={"query": query, "detections": []})
        overlay_path = self._save_detection_overlay(
            image_path, img, query, detections
        )
        image_paths = [image_path] + ([overlay_path] if overlay_path else [])
        return ToolResult(
            ok=True, tool="detect",
            summary=(f"Found {len(detections)} detection(s) for '{query}'."
                     + (f" Saved bbox overlay: {overlay_path}"
                        if overlay_path else "")),
            data={"query": query, "detections": detections,
                  "bboxes": [d["bbox"] for d in detections],
                  "annotated_image_path": overlay_path},
            image_paths=image_paths,
        )

    def inspect(
        self,
        image_path: Optional[str] = None,
        question: str = "What do you see?",
        bbox: Optional[list] = None,
        image_paths: Optional[list] = None,
        **_,
    ) -> ToolResult:
        # Accept a single image (image_path) or several (image_paths). They are
        # all sent to the VLM together so it can compare/reason across views.
        raw_paths: list[str] = []
        if image_paths:
            raw_paths = [p for p in image_paths if p]
        elif image_path:
            raw_paths = [image_path]
        paths = [p for p in raw_paths if os.path.exists(p)]
        if not paths:
            return ToolResult(ok=False, tool="inspect",
                              summary="Image not available.")

        # bbox cropping is only meaningful for a single image; with multiple
        # images each is inspected in full and bbox is ignored.
        bboxes_raw: list[list] = []
        if bbox is not None and len(paths) == 1:
            if isinstance(bbox[0], (int, float)):
                bboxes_raw = [bbox]
            else:
                bboxes_raw = list(bbox)

        query_paths:  list[str]  = []
        query_labels: list[str]  = []
        crop_infos:   list[list] = []

        if not bboxes_raw:
            for i, p in enumerate(paths):
                query_paths.append(p)
                query_labels.append(
                    f"Full image ({p}):" if len(paths) == 1
                    else f"Image {i + 1} of {len(paths)} ({p}):")
        else:
            from PIL import Image as _PIL
            crops_dir = str(self.log_dir / "crops")
            single    = paths[0]
            img       = _PIL.open(single).convert("RGB")
            W, H      = img.size
            for i, raw_box in enumerate(bboxes_raw):
                x1, y1, x2, y2 = [int(v) for v in raw_box[:4]]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if x2 <= x1 or y2 <= y1:
                    return ToolResult(ok=False, tool="inspect",
                                      summary=f"Invalid bbox[{i}]: [{x1},{y1},{x2},{y2}]")
                try:
                    crop = img.crop((x1, y1, x2, y2)).resize(
                        (512, 512), _PIL.LANCZOS)
                    crop_path = os.path.join(
                        crops_dir,
                        f"{self._step_counter * 100 + i:06d}.png"
                    )
                    os.makedirs(crops_dir, exist_ok=True)
                    crop.save(crop_path)
                    query_paths.append(crop_path)
                    query_labels.append(
                        f"Crop {i+1} of {len(bboxes_raw)} "
                        f"(bbox=[{x1},{y1},{x2},{y2}] from {single}):")
                    crop_infos.append([x1, y1, x2, y2])
                except Exception as e:
                    return ToolResult(ok=False, tool="inspect",
                                      summary=f"Crop[{i}] failed: {e}")

        result = self.gemini.inspect_image(query_paths, question,
                                           labels=query_labels)
        if not result:
            return ToolResult(ok=True, tool="inspect",
                              summary="Gemini returned no answer.",
                              image_paths=query_paths,
                              data={"bboxes": crop_infos or None})

        answer      = result.get("answer", "no answer")
        evidence    = result.get("evidence", "")
        confidence  = float(result.get("confidence", 0.5))
        cand_bboxes = result.get("candidate_bboxes", [])
        label       = f"{crop_infos}" if crop_infos else "full"

        srcs = ", ".join(paths)
        return ToolResult(
            ok=True, tool="inspect",
            summary=f"inspect {srcs} [{label}] | {question} | {answer}",
            data={"question": question, "answer": answer, "evidence": evidence,
                  "confidence": confidence, "candidate_bboxes": cand_bboxes,
                  "bboxes": crop_infos or None},
            image_paths=query_paths,
        )

    # ── retrieve_memory ───────────────────────────────────────────────────────

    def retrieve_memory(
        self,
        query: str,
        top_k: int = 5,
        time_from: Optional[str] = None,
        time_to: Optional[str] = None,
        **_,
    ) -> ToolResult:
        from memory.retrieval import retrieve_memory_candidates

        top_k    = -1 if int(top_k) == -1 else max(1, int(top_k))
        index_dir = os.path.join(
            self.retrieval_data_root, self.scene_id,
            f"retrieval_index_{self.retrieval_model}",
        )

        if not os.path.exists(os.path.join(index_dir, "index.bin")):
            return ToolResult(ok=False, tool="retrieve_memory",
                              summary="No memory index found. Scan the scene first.")

        is_image_query   = os.path.isfile(query)
        query_image_paths = [query] if is_image_query else None
        fetch_k = -1 if (time_from or time_to) else top_k

        try:
            candidates = retrieve_memory_candidates(
                query=query,
                index_dir=index_dir,
                capture_out_dir=str(self.capture_out_dir),
                top_k=fetch_k,
                model=self.retrieval_model,
                retrieval_data_root=self.retrieval_data_root,
                scene_id=self.scene_id,
                episodic_memory=self.episodic_memory,
                embedding_worker=self.embedding_worker,
                query_image_paths=query_image_paths,
            )
        except Exception as e:
            return ToolResult(ok=False, tool="retrieve_memory",
                              summary=f"Retrieval failed: {e}")

        if not candidates:
            return ToolResult(ok=False, tool="retrieve_memory",
                              summary="No matching frames found in memory.")

        if time_from or time_to:
            t_from = _parse_hhmmss(time_from) if time_from else None
            t_to   = _parse_hhmmss(time_to)   if time_to   else None
            before = len(candidates)
            candidates = [c for c in candidates
                          if _in_time_window(c.timestamp, t_from, t_to)]
            print(f"[Toolbox] time filter [{time_from}–{time_to}]: "
                  f"{before} → {len(candidates)}")
            if not candidates:
                return ToolResult(ok=False, tool="retrieve_memory",
                                  summary=f"No candidates in window {time_from}–{time_to}.")

        # Gemini rerank
        analysis_by_id: dict = {}
        try:
            img_paths = [c.image_path for c in candidates
                         if os.path.exists(c.image_path)]
            rerank_result = self.gemini.rerank_memory_candidates(
                query=query,
                candidates=[c.to_dict() for c in candidates],
                image_paths=img_paths,
                query_image_path=query if is_image_query else None,
            )
            ranked_ids = rerank_result.get("ranked_ids", [])
            for entry in rerank_result.get("candidates_analysis", []):
                mid = entry.get("memory_id")
                if mid:
                    analysis_by_id[mid] = entry
            if ranked_ids and len(ranked_ids) == len(candidates):
                id_map   = {c.memory_id: c for c in candidates}
                reordered = [id_map[mid] for mid in ranked_ids if mid in id_map]
                reordered += [c for c in candidates
                              if c.memory_id not in set(ranked_ids)]
                candidates = reordered
        except Exception as e:
            print(f"[Toolbox] rerank error (non-fatal): {e}")

        if top_k != -1:
            candidates = candidates[:top_k]

        cand_dicts = []
        for c in candidates:
            analysis = analysis_by_id.get(c.memory_id, {})
            entry: dict = {
                "memory_id":       c.memory_id,
                "navigate_target": {"memory_id": c.memory_id},
                "rgb_path":        c.image_path,
                "robot_pose":      c.robot_pose,
                "retrieval_score": round(c.retrieval_score, 4),
                "frame_idx":       c.frame_idx,
                "timestamp":       c.timestamp,
            }
            if analysis:
                entry["rerank_object_location"] = analysis.get("object_location", "")
                entry["rerank_confidence"]       = analysis.get("confidence", 0.0)
                entry["rerank_reasoning"]        = analysis.get("reasoning", "")
            cand_dicts.append(entry)

        return ToolResult(
            ok=True, tool="retrieve_memory",
            summary=(f"retrieve_memory['{query[:40]}']: "
                     f"{len(candidates)} candidates."),
            data={"query": query, "top_k": top_k, "candidates": cand_dicts},
            image_paths=[c.image_path for c in candidates
                         if os.path.exists(c.image_path)],
        )

    # ── retrieve_trajectory ───────────────────────────────────────────────────

    def retrieve_trajectory(
        self,
        time_from: str,
        time_to: str,
        **_,
    ) -> ToolResult:
        import json as _json

        t_from = _parse_hhmmss(time_from)
        t_to   = _parse_hhmmss(time_to)
        if t_from is None or t_to is None:
            return ToolResult(ok=False, tool="retrieve_trajectory",
                              summary=f"Cannot parse time bounds: {time_from!r}–{time_to!r}")

        traj_path = self.log_dir / "trajectory.jsonl"
        if not traj_path.exists():
            return ToolResult(ok=False, tool="retrieve_trajectory",
                              summary="No trajectory.jsonl found.")

        steps = []
        try:
            with open(traj_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = _json.loads(line)
                    except _json.JSONDecodeError:
                        continue
                    if not _in_time_window(entry.get("timestamp"), t_from, t_to):
                        continue
                    action = entry.get("action", {})
                    result = entry.get("result", {})
                    obs    = entry.get("current_obs", {})
                    steps.append({
                        "step_idx":          entry.get("step_idx"),
                        "timestamp":         entry.get("timestamp"),
                        "tool":              action.get("tool"),
                        "arguments":         action.get("arguments"),
                        "previous_action_verification": action.get(
                            "previous_action_verification", ""),
                        "progress_analysis": action.get("progress_analysis", ""),
                        "ok":                result.get("ok"),
                        "summary":           result.get("summary", ""),
                        "robot_pose":        obs.get("robot_pose"),
                    })
        except Exception as e:
            return ToolResult(ok=False, tool="retrieve_trajectory",
                              summary=f"Error reading trajectory: {e}")

        if not steps:
            return ToolResult(ok=False, tool="retrieve_trajectory",
                              summary=f"No steps in window {time_from}–{time_to}.")

        return ToolResult(
            ok=True, tool="retrieve_trajectory",
            summary=f"retrieve_trajectory [{time_from}–{time_to}]: {len(steps)} steps",
            data={"time_from": time_from, "time_to": time_to, "steps": steps},
        )

    # ── navigate ─────────────────────────────────────────────────────────────

    def navigate(self, target: dict, **_) -> ToolResult:
        target_xy, target_yaw, nav_label = self._resolve_navigate_target(target)
        if target_xy is None:
            return ToolResult(ok=False, tool="navigate",
                              summary=f"Could not resolve navigate target: {target}")

        start_xy   = np.array(self._get_robot_pose()[:2])
        waypoints  = self._plan_path(start_xy, target_xy)
        total_dist = float(np.linalg.norm(target_xy - start_xy))
        print(f"[Toolbox] navigate: {len(waypoints)} waypoints  "
              f"target=({target_xy[0]:.2f},{target_xy[1]:.2f})  "
              f"dist={total_dist:.2f}m")

        NAV_STOP_DIST = 0.5
        WP_ADV = 0.3
        wp_idx = 0
        outcome = "incomplete"
        dist = total_dist

        for step in range(self._NAV_MAX_STEPS):
            if step % self._EVENT_EVERY == 0:
                self._pump_events()

            pose    = self._get_robot_pose()
            curr_xy = np.array(pose[:2])
            curr_yaw = pose[2]
            dist    = float(np.linalg.norm(target_xy - curr_xy))

            if dist < NAV_STOP_DIST:
                outcome = f"reached  dist={dist:.2f}m"
                break

            if (wp_idx == len(waypoints) - 1 and
                    np.linalg.norm(waypoints[-1] - curr_xy) < WP_ADV):
                outcome = f"best_effort  dist={dist:.2f}m"
                break

            while wp_idx < len(waypoints) - 1:
                if np.linalg.norm(waypoints[wp_idx] - curr_xy) < WP_ADV:
                    wp_idx += 1
                else:
                    break

            delta   = waypoints[wp_idx] - curr_xy
            bearing = math.atan2(delta[1], delta[0]) - curr_yaw
            bearing = (bearing + math.pi) % (2 * math.pi) - math.pi

            self._navigate_step(bearing)
            self._last_obs = self._step()

        else:
            outcome = f"max_steps  dist={dist:.2f}m"

        if target_yaw is not None:
            self._align_yaw(target_yaw)

        obs_result = self.observe()
        final_xy   = np.array(self._get_robot_pose()[:2])
        final_dist = float(np.linalg.norm(target_xy - final_xy))
        reached    = "reached" in outcome or final_dist < NAV_STOP_DIST + 0.5

        return ToolResult(
            ok=reached or "best_effort" in outcome,
            tool="navigate",
            summary=f"navigate[{nav_label}]: {outcome}  final_dist={final_dist:.2f}m",
            data={"target": target, "nav_label": nav_label, "outcome": outcome,
                  "final_pose": self._get_robot_pose(),
                  "distance_to_goal": final_dist},
            image_paths=obs_result.image_paths,
        )

    # ── base_move ────────────────────────────────────────────────────────────

    def base_move(
        self,
        motion: str,
        **_,
    ) -> ToolResult:
        motion = normalize_base_move_motion(motion) or ""
        if motion not in BASE_MOVE_MOTIONS:
            return ToolResult(
                ok=False, tool="base_move",
                summary=(f"base_move: invalid motion {motion!r}. "
                         f"Allowed: {sorted(BASE_MOVE_MOTIONS)}"),
            )

        start_pose = self._get_robot_pose()
        start_xy = np.array(start_pose[:2], dtype=np.float64)
        start_yaw = float(start_pose[2]) if len(start_pose) >= 3 else 0.0
        rotate = motion.startswith("rotate")
        target = (self._BASE_MOVE_ROTATION_RAD if rotate
                  else self._BASE_MOVE_DISTANCE_M)
        steps = 0

        for step in range(self._BASE_MOVE_MAX_STEPS):
            if step % self._EVENT_EVERY == 0:
                self._pump_events()

            self._base_move_step(motion)
            self._last_obs = self._step()
            steps += 1

            pose = self._get_robot_pose()
            curr_xy = np.array(pose[:2], dtype=np.float64)
            curr_yaw = float(pose[2]) if len(pose) >= 3 else start_yaw
            if rotate:
                yaw_delta = self._angle_diff(curr_yaw, start_yaw)
                if motion == "rotate 30 degrees":
                    progress = yaw_delta
                else:
                    progress = -yaw_delta
            else:
                progress = float(np.linalg.norm(curr_xy - start_xy))

            if progress >= target * 0.95:
                break

        final_pose = self._get_robot_pose()
        final_xy = np.array(final_pose[:2], dtype=np.float64)
        final_yaw = float(final_pose[2]) if len(final_pose) >= 3 else start_yaw
        yaw_delta = self._angle_diff(final_yaw, start_yaw)
        distance = float(np.linalg.norm(final_xy - start_xy))
        if rotate:
            progress = yaw_delta if motion == "rotate 30 degrees" else -yaw_delta
            completed = progress >= target * 0.75
        else:
            progress = distance
            completed = progress >= target * 0.5

        obs_result = self.observe()
        return ToolResult(
            ok=completed, tool="base_move",
            summary=(f"base_move[{motion}]: steps={steps} "
                     f"distance={distance:.2f}m yaw_delta={math.degrees(yaw_delta):.1f}deg "
                     f"completed={completed}"),
            data={
                "motion": motion,
                "steps": steps,
                "start_pose": start_pose,
                "final_pose": final_pose,
                "distance_m": distance,
                "yaw_delta_deg": math.degrees(yaw_delta),
                "completed": completed,
            },
            image_paths=obs_result.image_paths,
        )

    # ── manipulate ───────────────────────────────────────────────────────────

    def _post_manipulate(self) -> None:
        """Hook run after every grasp/place attempt, whatever the outcome.
        Backends with an arm override this to retract it; default is a no-op."""

    def manipulate(
        self,
        skill: str,
        target: str = "",
        destination: Optional[str] = None,
        target_region: Optional[str] = None,
        **_,
    ) -> ToolResult:
        skill = skill.lower().strip()

        if skill not in SUPPORTED_SKILLS:
            return ToolResult(
                ok=False, tool="manipulate",
                summary=(f"Skill '{skill}' is not supported. "
                         f"Supported: {sorted(SUPPORTED_SKILLS)}."),
            )

        if skill == "grasp":
            localized, obj_name, dist = self._grasp(target)
            if not localized:
                # No graspable object under the aimed pixel — a real inability
                # to act, so report failure (this is not a task-success verdict).
                return ToolResult(ok=False, tool="manipulate",
                                  summary=f"Could not localize '{target}' to grasp.",
                                  data={"skill": skill})

            # A grasp was attempted on a real target. Query the backend's
            # ground-truth grasp state and report it directly so the policy does
            # not have to infer success from the image.
            self._post_manipulate()   # retract the arm regardless of outcome
            obs_result = self.observe()
            gstate    = self._grasp_state()
            grasped   = gstate.get("grasped") if gstate else None
            held_obj  = gstate.get("object")  if gstate else None
            if grasped is True:
                held_str = f" (holding '{held_obj}')" if held_obj else ""
                summary  = f"Grasp on {obj_name} SUCCEEDED: object is held{held_str}."
            elif grasped is False:
                summary  = f"Grasp on {obj_name} FAILED: gripper is empty (no object held)."
            else:
                summary  = f"Grasp attempted on {obj_name} (grasp state unknown)."
            return ToolResult(
                # When ground truth is available, ok reflects the actual grasp
                # outcome; otherwise stay neutral (attempt made) and defer.
                ok=(grasped is not False),
                tool="manipulate",
                summary=summary,
                data={"skill": skill, "object": obj_name, "distance": dist,
                      "grasped": grasped, "held_object": held_obj},
                image_paths=obs_result.image_paths,
            )

        if skill in ("place", "drop"):
            released, release_summary = self._release(
                target=target,
                destination=destination,
                target_region=target_region,
            )
            self._post_manipulate()   # retract the arm regardless of outcome
            obs_result = self.observe()
            return ToolResult(
                ok=released, tool="manipulate",
                summary=f"{skill}: {release_summary}",
                data={"skill": skill, "released": released,
                      "destination": destination,
                      "target_region": target_region},
                image_paths=obs_result.image_paths,
            )

        return ToolResult(ok=False, tool="manipulate",
                          summary=f"Unhandled skill: {skill}")

    # ── wait ─────────────────────────────────────────────────────────────────

    def wait(self, seconds: float = 1.0, **_) -> ToolResult:
        seconds = float(min(max(seconds, 0.1), 100.0))
        n_steps = max(1, int(seconds * 30))
        try:
            for i in range(n_steps):
                if i % self._EVENT_EVERY == 0:
                    self._pump_events()
                self._last_obs = self._step()
    
        except Exception as e:
            return ToolResult(ok=False, tool="wait", summary=f"wait failed: {e}")

        return ToolResult(
            ok=True, tool="wait",
            summary=f"wait: stepped {n_steps} steps (~{seconds:.1f}s)",
            data={"seconds": seconds, "n_steps": n_steps},
        )

    # ── finish ────────────────────────────────────────────────────────────────

    def finish(self, answer: Optional[str] = None, **_) -> ToolResult:
        return ToolResult(
            ok=True, tool="finish",
            summary=f"finish: {answer}" if answer else "finish",
            data={"answer": answer, "task_done": True},
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_navigate_target(
        self, target: Any
    ) -> tuple[Optional[np.ndarray], Optional[float], str]:
        if isinstance(target, str):
            memory_id = normalize_memory_id(target)
            if memory_id:
                xy, yaw = self._pose_from_memory_id(memory_id)
                if xy is not None:
                    return xy, yaw, memory_id
                return None, None, f"memory_id {memory_id} not found"
            return None, None, "navigate target string must be a memory_id"

        if not isinstance(target, dict):
            return None, None, "navigate target requires a dict or memory_id string"

        memory_id = normalize_memory_id(
            target.get("memory_id") or target.get("mem_id") or target.get("memory")
        )
        if memory_id:
            xy, yaw = self._pose_from_memory_id(memory_id)
            if xy is not None:
                return xy, yaw, memory_id
            return None, None, f"memory_id {memory_id} not found"

        # Raw coordinate goals are NOT accepted: navigate only goes to a
        # remembered observation (a retrieve_memory candidate's memory_id).
        return None, None, (
            "navigate target must be a memory_id from retrieve_memory "
            "(raw coordinates/pose are not accepted)"
        )

    def _pose_from_memory_id(
        self, memory_id: str
    ) -> tuple[Optional[np.ndarray], Optional[float]]:
        memory_id = normalize_memory_id(memory_id) or str(memory_id)
        try:
            idx_str   = memory_id.replace("mem_", "").lstrip("0") or "0"
            frame_idx = int(idx_str)
            pose_path = self.capture_out_dir / "robot_xy" / f"{frame_idx:06d}.txt"
            if pose_path.exists():
                data = np.loadtxt(str(pose_path)).flatten()
                if len(data) >= 3:
                    return np.array([data[0], data[1]]), float(data[2])
                if len(data) >= 2:
                    return np.array([data[0], data[1]]), None
        except Exception as e:
            print(f"[Toolbox] _pose_from_memory_id file lookup error: {e}")

        if self.episodic_memory is not None:
            result = self.episodic_memory.get_pose(memory_id)
            if result is not None:
                return result
        return None, None

    def _estimate_object_xy_from_depth(
        self, target: str
    ) -> Optional[np.ndarray]:
        """Back-project depth + Gemini bbox → world XY."""
        obs = self._last_obs
        if obs is None:
            return None

        bbox: Optional[list] = None
        if self._grounding_dino is not None and self._last_image_path:
            try:
                from PIL import Image as _PIL
                img_pil = _PIL.open(self._last_image_path).convert("RGB")
                detections = self._grounding_dino.detect(
                    np.array(img_pil), target
                )
                if detections:
                    bbox = detections[0]["bbox"]
                    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
                    W, H = img_pil.size
                    x1, y1 = max(0, min(W - 1, x1)), max(0, min(H - 1, y1))
                    x2, y2 = max(x1 + 1, min(W, x2)), max(y1 + 1, min(H, y2))
                    bbox = [x1, y1, x2, y2]
                    try:
                        crops_dir = str(self.log_dir / "crops")
                        os.makedirs(crops_dir, exist_ok=True)
                        crop = img_pil.crop((x1, y1, x2, y2)).resize(
                            (512, 512), _PIL.LANCZOS)
                        crop_path = os.path.join(
                            crops_dir, f"{self._step_counter * 100:06d}_gdino.png")
                        crop.save(crop_path)
                    except Exception as _ce:
                        print(f"[depth] crop save failed: {_ce}")
            except Exception as _e:
                print(f"[depth] GroundingDINO failed: {_e}")

        if bbox is None:
            inspect_result = self.inspect(
                image_path=self._last_image_path or "",
                question=f"Locate the {target}. Return its pixel bounding box.",
            )
            if inspect_result.ok:
                bboxes = inspect_result.data.get("candidate_bboxes", [])
                if bboxes:
                    bbox = bboxes[0]

        cam = self._get_depth_and_intrinsics(obs)
        if cam is None:
            print("[Toolbox] _estimate_object_xy: no depth/camera params")
            return None
        depth_np, K, E = cam
        H, W = depth_np.shape[:2]

        if bbox and len(bbox) >= 4:
            x1, y1, x2, y2 = (int(bbox[0]), int(bbox[1]),
                               int(bbox[2]), int(bbox[3]))
            x1, y1 = max(0, min(W - 1, x1)), max(0, min(H - 1, y1))
            x2, y2 = max(x1 + 1, min(W, x2)), max(y1 + 1, min(H, y2))
            u_c = max(0, min(W - 1, (x1 + x2) // 2))
            v_c = max(0, min(H - 1, (y1 + y2) // 2))
            patch = depth_np[y1:y2, x1:x2]
            valid = patch[(patch > 0.05) & (patch < 10.0)]
            if valid.size > 0:
                d = float(np.median(valid))
            else:
                d = float(depth_np[v_c, u_c])  # fall back to bbox centre pixel
        else:
            u_c, v_c = W // 2, H // 2
            d = float(depth_np[v_c, u_c])

        if d < 0.05 or d > 10.0:
            print(f"[Toolbox] _estimate_object_xy: invalid depth {d:.3f}m")
            return None

        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        x_c    = (u_c - cx) * d / fx
        y_c    = (v_c - cy) * d / fy
        p_cam  = np.array([x_c, y_c, d, 1.0])
        p_world = np.linalg.inv(E) @ p_cam

        obj_xy = p_world[:2].copy()
        print(f"[Toolbox] _estimate_object_xy: '{target}'  "
              f"bbox={bbox}  depth={d:.2f}m  "
              f"world_xy=({obj_xy[0]:.2f},{obj_xy[1]:.2f})")
        return obj_xy

    def _align_yaw(self, target_yaw: float) -> None:
        # kinematic_nav_step moves FORWARD when |bearing| < ROT_THRESH (~8°).
        # Clamp bearing to always exceed that threshold so we only rotate here.
        _MIN_ROT = math.radians(9)
        for _ in range(400):
            pose    = self._get_robot_pose()
            bearing = ((target_yaw - pose[2] + math.pi) % (2 * math.pi) - math.pi)
            if abs(bearing) < math.radians(5):
                break
            if abs(bearing) < _MIN_ROT:
                bearing = math.copysign(_MIN_ROT, bearing)
            self._navigate_step(bearing)
            self._step()

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        return (a - b + math.pi) % (2 * math.pi) - math.pi

    def _snap_to_navmesh(self, xy: np.ndarray) -> np.ndarray:
        """Snap xy to nearest navmesh vertex if available. Override to add navmesh."""
        return xy

    def _save_frame(self, rgb: np.ndarray, tag: str) -> str:
        from PIL import Image as _PIL
        self._step_counter += 1
        path = str(self.log_dir / "images" / f"{self._step_counter:04d}_{tag}.png")
        _PIL.fromarray(rgb).save(path)
        return path

    def _save_companion_frame(self, rgb: np.ndarray, tag: str) -> str:
        """Save an extra frame for the CURRENT step without advancing the step
        counter, so it pairs with the head frame just saved."""
        from PIL import Image as _PIL
        path = str(self.log_dir / "images" / f"{self._step_counter:04d}_{tag}.png")
        _PIL.fromarray(rgb).save(path)
        return path


    def _pose_str(self, pose: list) -> str:
        if len(pose) >= 3:
            return f"({pose[0]:.2f},{pose[1]:.2f} yaw={math.degrees(pose[2]):.0f}°)"
        return str(pose)

    def _pump_events(self) -> None:
        if self.event_callback:
            try:
                self.event_callback()
            except Exception:
                pass
