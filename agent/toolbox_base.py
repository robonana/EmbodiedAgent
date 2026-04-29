"""
agent/toolbox_base.py — BaseToolbox: backend-independent agent toolbox.

Contains all tool logic and state management.  Backend-specific behaviour is
delegated to ten abstract primitive methods that each backend must implement:

    _step()                        → dict          advance one control cycle; return obs dict
    _capture_rgb(obs)              → ndarray|None  extract RGB from obs
    _get_robot_pose()              → [x, y, yaw]   current robot pose
    _navigate_step(bearing)        → None          send one navigation command toward bearing
    _plan_path(start, goal)        → [ndarray]     list of XY waypoints
    _grasp(target)                 → (ok, name, dist)
    _release()                     → None
    _forward_step()                → None          send one forward movement command
    _get_depth_and_intrinsics(obs) → (depth, K, E) | None
"""

from __future__ import annotations

import math
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .schemas import (
    ToolAction,
    ToolResult,
    SUPPORTED_SKILLS,
    VALID_TOOLS,
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

    Concrete subclasses implement the ten primitive methods below for either
    a simulator or a real robot.
    """

    _NAV_MAX_STEPS  = 2000
    _APPROACH_STEPS = 600
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

    @abstractmethod
    def _get_robot_pose(self) -> list[float]:
        """Return current robot pose as [x, y, yaw]."""

    @abstractmethod
    def _navigate_step(self, bearing: float) -> None:
        """Advance robot one kinematic step toward bearing (radians)."""

    @abstractmethod
    def _plan_path(
        self, start_xy: np.ndarray, goal_xy: np.ndarray
    ) -> list[np.ndarray]:
        """Return list of XY waypoints from start to goal."""

    @abstractmethod
    def _grasp(self, target: str) -> tuple[bool, str, float]:
        """Attempt to grasp nearest object matching target.
        Returns (success, object_name, distance_m)."""

    @abstractmethod
    def _release(self) -> None:
        """Release held object."""

    @abstractmethod
    def _forward_step(self) -> None:
        """Move robot forward one small step (blind approach fallback)."""

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
                "approach":            self.approach,
                "manipulate":          self.manipulate,
                "verify":              self.verify,
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
            )
        except Exception as e:
            return ToolResult(ok=False, tool="observe",
                              summary=f"observe failed: {e}")

    # ── inspect ───────────────────────────────────────────────────────────────

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
        return ToolResult(
            ok=True, tool="detect",
            summary=f"Found {len(detections)} detection(s) for '{query}'.",
            data={"query": query, "detections": detections,
                  "bboxes": [d["bbox"] for d in detections]},
            image_paths=[image_path],
        )

    def inspect(
        self,
        image_path: str,
        question: str = "What do you see?",
        bbox: Optional[list] = None,
        **_,
    ) -> ToolResult:
        if not image_path or not os.path.exists(image_path):
            return ToolResult(ok=False, tool="inspect",
                              summary="Image not available.")

        bboxes_raw: list[list] = []
        if bbox is not None:
            if isinstance(bbox[0], (int, float)):
                bboxes_raw = [bbox]
            else:
                bboxes_raw = list(bbox)

        query_paths:  list[str]  = []
        query_labels: list[str]  = []
        crop_infos:   list[list] = []

        if not bboxes_raw:
            query_paths  = [image_path]
            query_labels = [f"Full image ({image_path}):"]
        else:
            from PIL import Image as _PIL
            crops_dir = str(self.log_dir / "crops")
            img       = _PIL.open(image_path).convert("RGB")
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
                        f"(bbox=[{x1},{y1},{x2},{y2}] from {image_path}):")
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

        return ToolResult(
            ok=True, tool="inspect",
            summary=f"inspect {image_path} [{label}] | {question} | {answer}",
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
        from ..memory.retrieval import retrieve_memory_candidates

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
                        "progress_analysis": action.get("progress_analysis", "")[:200],
                        "ok":                result.get("ok"),
                        "summary":           result.get("summary", "")[:150],
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

    # ── approach ─────────────────────────────────────────────────────────────

    def approach(
        self,
        target: str = "view_center",
        desired_distance: float = 0.55,
        **_,
    ) -> ToolResult:
        desired_distance = float(desired_distance)

        obj_xy: Optional[np.ndarray] = None
        if target != "view_center":
            obj_xy = self._estimate_object_xy_from_depth(target)
            if obj_xy is None:
                print(f"[Toolbox] approach: depth estimation failed for "
                      f"'{target}', falling back to forward steps")
                return self._forward_approach(desired_distance, target)

        start_xy = np.array(self._get_robot_pose()[:2])

        dir_vec = start_xy - obj_xy
        d = float(np.linalg.norm(dir_vec))
        dir_vec = dir_vec / d if d > 1e-6 else np.array([1.0, 0.0])
        approach_xy = obj_xy + desired_distance * dir_vec
        approach_xy = self._snap_to_navmesh(approach_xy)

        waypoints = self._plan_path(start_xy, approach_xy)
        WP_ADV = 0.15
        wp_idx = 0
        reached = False

        for step in range(self._APPROACH_STEPS):
            if step % self._EVENT_EVERY == 0:
                self._pump_events()

            pose     = self._get_robot_pose()
            curr_xy  = np.array(pose[:2])
            dist_obj = float(np.linalg.norm(obj_xy - curr_xy))

            if dist_obj <= desired_distance:
                reached = True
                break

            if (wp_idx == len(waypoints) - 1 and
                    np.linalg.norm(waypoints[-1] - curr_xy) < WP_ADV):
                break

            while wp_idx < len(waypoints) - 1:
                if np.linalg.norm(waypoints[wp_idx] - curr_xy) < WP_ADV:
                    wp_idx += 1
                else:
                    break

            curr_yaw = pose[2]
            delta    = waypoints[wp_idx] - curr_xy
            bearing  = math.atan2(delta[1], delta[0]) - curr_yaw
            bearing  = (bearing + math.pi) % (2 * math.pi) - math.pi

            self._navigate_step(bearing)
            self._last_obs = self._step()


        # Nav stopped at navmesh boundary; close remaining gap with forward steps
        final_xy   = np.array(self._get_robot_pose()[:2])
        final_dist = float(np.linalg.norm(obj_xy - final_xy))
        if not reached and final_dist > desired_distance:
            gap        = final_dist - desired_distance
            extra_steps = max(1, int(gap / 0.015))
            target_yaw  = math.atan2(obj_xy[1] - final_xy[1], obj_xy[0] - final_xy[0])
            self._align_yaw(target_yaw)
            for _ in range(extra_steps):
                curr_xy = np.array(self._get_robot_pose()[:2])
                if float(np.linalg.norm(obj_xy - curr_xy)) <= desired_distance:
                    break
                self._forward_step()
                self._last_obs = self._step()

        final_xy   = np.array(self._get_robot_pose()[:2])
        final_dist = float(np.linalg.norm(obj_xy - final_xy))
        reached    = reached or final_dist <= desired_distance + 0.1
        obs_result = self.observe()

        return ToolResult(
            ok=reached, tool="approach",
            summary=(f"approach['{target[:40]}']: "
                     f"final_dist={final_dist:.2f}m  reached={reached}"),
            data={"target": target, "desired_distance": desired_distance,
                  "final_distance": final_dist, "reached_desired_distance": reached},
            image_paths=obs_result.image_paths,
        )

    # ── manipulate ───────────────────────────────────────────────────────────

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
            success, obj_name, dist = self._grasp(target)
            if not success:
                return ToolResult(ok=False, tool="manipulate",
                                  summary=f"No object found to grasp.")

            obs_result = self.observe()
            return ToolResult(
                ok=True, tool="manipulate",
                summary=(f"grasp attempted: {obj_name} at dist={dist:.2f}m — "),
                data={"skill": skill, "object": obj_name, "distance": dist},
                image_paths=obs_result.image_paths,
            )

        if skill in ("place", "drop"):
            self._release()
            obs_result = self.observe()
            return ToolResult(
                ok=True, tool="manipulate",
                summary=f"{skill}: gripper opened.",
                data={"skill": skill},
                image_paths=obs_result.image_paths,
            )

        return ToolResult(ok=False, tool="manipulate",
                          summary=f"Unhandled skill: {skill}")

    # ── verify ────────────────────────────────────────────────────────────────

    def verify(
        self,
        condition: str,
        target: Optional[str] = None,
        **_,
    ) -> ToolResult:
        img_path = self._last_image_path
        if img_path is None or not os.path.exists(img_path):
            obs = self.observe()
            if not obs.ok:
                return ToolResult(ok=False, tool="verify",
                                  summary="Cannot verify: no image available.")
            img_path = self._last_image_path

        vlm    = self.gemini.verify_condition(img_path, condition, target)
        sat    = bool(vlm.get("satisfied", False))
        conf   = float(vlm.get("confidence", 0.0))
        evid   = vlm.get("evidence", "")

        return ToolResult(
            ok=True, tool="verify",
            summary=(f"verify['{condition[:60]}']: "
                     f"satisfied={sat}  conf={conf:.2f}  {evid[:60]}"),
            data={"condition": condition, "target": target,
                  "satisfied": sat, "confidence": conf, "evidence": evid},
            image_paths=[img_path],
        )

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
        self, target: dict
    ) -> tuple[Optional[np.ndarray], Optional[float], str]:
        pose = target.get("pose", [])
        if len(pose) >= 2:
            xy  = np.array([float(pose[0]), float(pose[1])])
            yaw = float(pose[2]) if len(pose) >= 3 else None
            return xy, yaw, f"pose({xy[0]:.2f},{xy[1]:.2f})"
        return None, None, "navigate target requires pose: [x, y] or [x, y, theta]"

    def _pose_from_memory_id(
        self, memory_id: str
    ) -> tuple[Optional[np.ndarray], Optional[float]]:
        if self.episodic_memory is not None:
            result = self.episodic_memory.get_pose(memory_id)
            if result is not None:
                return result
        try:
            idx_str   = memory_id.replace("mem_", "").lstrip("0") or "0"
            frame_idx = int(idx_str)
            pose_path = self.capture_out_dir / "robot_xy" / f"{frame_idx:06d}.txt"
            if not pose_path.exists():
                return None, None
            data = np.loadtxt(str(pose_path)).flatten()
            if len(data) >= 3:
                return np.array([data[0], data[1]]), float(data[2])
            elif len(data) >= 2:
                return np.array([data[0], data[1]]), None
        except Exception as e:
            print(f"[Toolbox] _pose_from_memory_id error: {e}")
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

    def _forward_approach(
        self, desired_distance: float, target_label: str
    ) -> ToolResult:
        # Walk at most desired_distance + 0.5m (≈ reach margin) forward.
        # _forward_step moves 0.015 m per call (NAV_FWD_M_PER_STEP).
        n_steps = max(1, int((desired_distance + 0.5) / 0.015))
        for i in range(n_steps):
            if i % self._EVENT_EVERY == 0:
                self._pump_events()
            self._forward_step()
            self._last_obs = self._step()


        obs_result = self.observe()
        return ToolResult(
            ok=True, tool="approach",
            summary=(f"approach(blind forward {n_steps} steps / "
                     f"{n_steps*0.015:.2f}m): '{target_label}'"),
            data={"target": target_label, "approximation": "blind_forward"},
            image_paths=obs_result.image_paths,
        )

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

    def _snap_to_navmesh(self, xy: np.ndarray) -> np.ndarray:
        """Snap xy to nearest navmesh vertex if available. Override to add navmesh."""
        return xy

    def _save_frame(self, rgb: np.ndarray, tag: str) -> str:
        from PIL import Image as _PIL
        self._step_counter += 1
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
