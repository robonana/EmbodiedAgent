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

This is the largest single piece of the system, and the split is deliberate: the *tools*
(what the agent can ask for, how arguments are validated, how results are summarised for
the VLM) are identical whether we are driving Habitat, an MCP-controlled robot, or the RL
env server. Only the *primitives* — take a physics step, read a pose, send a velocity —
differ. Subclasses implement ~9 small methods; everything else is inherited.

Note the recurring shape of every tool: it never raises, it always comes back as a
ToolResult whose `summary` is written to be read by the VLM on the next step, and its
`ok` flag means "the tool executed", not "the task progressed".
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
    """'HH:MM:SS' or 'HH:MM' → total seconds since midnight.

    Converting to a single integer makes window comparison a pair of integer compares,
    and sidesteps string-ordering bugs. Returns None (rather than raising) on anything
    unparseable — the VLM writes these strings, so malformed input is expected traffic,
    and callers interpret None as "no bound".
    """
    if not s:
        return None
    try:
        parts = s.strip().split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        if len(parts) == 2:
            # "MM:SS" — episodes are minutes long, so a 2-part time is minutes:seconds,
            # not hours:minutes.
            return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        pass
    return None


def _in_time_window(
    timestamp: Optional[str],
    t_from: Optional[int],
    t_to: Optional[int],
) -> bool:
    """Inclusive window test used by both retrieve_memory and retrieve_trajectory.

    Fails *open*: an entry with a missing or unparseable timestamp is kept rather than
    dropped. A filter that silently hides real memories is worse than one that lets a few
    extra through — the VLM can discard an irrelevant frame, but it cannot recover one it
    was never shown. Either bound may be None, meaning unbounded on that side.
    """
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

    # Safety bounds and motion quanta. The *_MAX_STEPS values are watchdogs, not targets:
    # they stop a control loop spinning forever when the robot is wedged against geometry
    # and making no progress toward its goal.
    _NAV_MAX_STEPS  = 2000          # ~ generous; a cross-room navigate uses a few hundred
    _BASE_MOVE_MAX_STEPS = 120      # one base_move is a *small* nudge, so this is plenty
    _BASE_MOVE_DISTANCE_M = 0.30    # how far "forward"/"left"/... travels
    _BASE_MOVE_ROTATION_RAD = math.radians(30)   # matches the "rotate 30 degrees" motions
    _EVENT_EVERY    = 10            # pump GUI events every N control steps (not every
                                    # step — the callback is comparatively expensive)

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
        # The VLM. Used for the *nested* calls a tool makes while executing (inspect's
        # visual QA, retrieve_memory's rerank) — not for policy decisions, which the
        # agent loop owns.
        self.gemini          = gemini_client
        self.log_dir         = Path(log_dir)            # this episode's output dir
        self.capture_out_dir = Path(capture_out_dir)    # the scene's pre-scan frames

        # Retrieval stack. All three may be None in a run without memory; the tools that
        # need them check and degrade to an ok=False ToolResult.
        self.embedding_worker  = embedding_worker
        self.episodic_memory   = episodic_memory
        self.retrieval_model   = retrieval_model
        # The index lives at {root}/{scene_id}/retrieval_index_{model}/. Both parts
        # default off capture_out_dir so a caller that only knows the scan directory
        # still resolves the right index.
        self.retrieval_data_root = retrieval_data_root or str(self.capture_out_dir.parent)
        self.scene_id        = scene_id or self.capture_out_dir.name
        self.event_callback  = event_callback     # GUI pump; None when headless
        self._grounding_dino = grounding_dino     # None ⇒ `detect` is unavailable

        # Rolling "what the robot last saw". Written by observe(), read by the agent loop
        # (_last_image_path) and by the depth back-projection (_last_obs).
        self._last_obs:        Optional[dict]       = None
        self._last_rgb:        Optional[np.ndarray] = None
        self._last_image_path: Optional[str]        = None
        # Increments once per saved observation frame; also names the image files, so
        # frame numbering and file numbering can never drift apart.
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
        """The single entry point from the agent loop: validate, dispatch, never raise.

        Three layers of defence, because the input is model-generated:
          1. tool name must be in the registry,
          2. arguments must pass the structural verifier,
          3. anything the tool itself throws is caught and returned as ok=False.

        Every failure path produces a ToolResult whose summary explains *what was wrong*,
        because that text is what the VLM reads next step — an exception would kill the
        episode, whereas a good error message often fixes the next action.
        """
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
            # Every handler takes **_ so that extra keys the VLM hallucinates are absorbed
            # rather than raising TypeError — the verifier already checked the keys we care
            # about, and a spurious argument shouldn't cost a step.
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
        """Advance one control cycle, capture the head camera, and cache the result.

        Not a VLM-callable tool — the agent loop calls this itself before every policy
        step, and several tools call it after acting so their ToolResult carries a fresh
        post-action frame. The side effect (updating _last_obs / _last_rgb /
        _last_image_path) is the point: everything downstream reads that cache.
        """
        try:
            obs = self._step()
            self._pump_events()

            rgb = self._capture_rgb(obs)
            if rgb is None:
                # Sensor failure. ok=False rather than an exception so the loop survives
                # a transient render glitch and simply tries again next step.
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
        """Save an annotated copy of a detect() image with bbox overlays.

        The overlay is attached to the ToolResult and therefore *shown to the VLM* next
        step. That is the real purpose: the model reads boxes far more reliably when it can
        see them drawn on the image than when it has to map numeric coordinates onto pixels
        in its head. Numbering the boxes ("1: bowl 0.87") also lets it refer to one by index.

        Returns None if there is nothing to draw or anything goes wrong — a missing overlay
        degrades detect() cosmetically, so it must never fail the tool.
        """
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
                # Clamp into the image. The detector can emit slightly out-of-frame boxes,
                # and the x2>x1 / y2>y1 floors keep the rectangle non-degenerate.
                x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
                x1 = max(0, min(W - 1, x1))
                y1 = max(0, min(H - 1, y1))
                x2 = max(x1 + 1, min(W, x2))
                y2 = max(y1 + 1, min(H, y2))
                color = (255, 50, 50)
                draw.rectangle((x1, y1, x2, y2), outline=color, width=3)

                # Caption: "<index>: <label> <score>", drawn on a filled black plate so it
                # stays legible over any scene content.
                label = str(det.get("label") or query)
                score = det.get("score")
                suffix = f" {score:.2f}" if isinstance(score, (int, float)) else ""
                text = f"{i}: {label}{suffix}"
                text_bbox = draw.textbbox((x1, y1), text)
                text_h = text_bbox[3] - text_bbox[1]
                # Prefer to sit the caption above the box; if the box is at the top edge,
                # max(0, ...) drops it inside the frame instead of off-screen.
                ty = max(0, y1 - text_h - 6)
                draw.rectangle(
                    (x1, ty, min(W, x1 + text_bbox[2] - text_bbox[0] + 8),
                     min(H, ty + text_h + 6)),
                    fill=(0, 0, 0),
                )
                draw.text((x1 + 4, ty + 3), text, fill=(255, 255, 255))

            # Name the overlay after the source frame + the query, so a directory of
            # overlays is self-describing. The query is model-authored free text, so
            # sanitise it hard before it becomes a filename.
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
        """GroundingDINO open-set bounding box proposals.

        Exists because VLM-guessed bounding boxes are unreliable, and the boxes feed two
        things that need real coordinates: `inspect` crops and the depth back-projection
        that turns a pixel into a world XY for grasping. The system prompt tells the model
        to get boxes from here rather than inventing them.

        Note it reads the image from `image_path` rather than using the cached frame — the
        model may legitimately want to detect in a *memory* frame, not just the live view.
        """
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
            # ok=False so the model treats "not found here" as a real signal — it should
            # move or look elsewhere rather than re-detecting the same empty frame.
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
        """Visual QA over the current view, a memory frame, or cropped regions of one.

        The crop path is the interesting one. A 512×512 head-camera frame renders a mug
        across the room as a handful of pixels — far too small for the VLM to identify.
        Cropping the detector's bbox and upscaling it to a full 512×512 gives the model a
        close-up it can actually read. This "detect → crop → inspect" chain is how the
        agent identifies small or distant objects, and it is why `bbox` exists at all.
        """
        # Accept a single image (image_path) or several (image_paths). They are
        # all sent to the VLM together so it can compare/reason across views.
        raw_paths: list[str] = []
        if image_paths:
            raw_paths = [p for p in image_paths if p]
        elif image_path:
            raw_paths = [image_path]
        # Drop paths that don't exist (the model sometimes invents plausible filenames)
        # rather than failing outright — any surviving image still answers the question.
        paths = [p for p in raw_paths if os.path.exists(p)]
        if not paths:
            return ToolResult(ok=False, tool="inspect",
                              summary="Image not available.")

        # bbox cropping is only meaningful for a single image; with multiple
        # images each is inspected in full and bbox is ignored.
        bboxes_raw: list[list] = []
        if bbox is not None and len(paths) == 1:
            # Same one-box-or-many disambiguation as the verifier: a numeric first element
            # means this is a single flat [x1,y1,x2,y2].
            if isinstance(bbox[0], (int, float)):
                bboxes_raw = [bbox]
            else:
                bboxes_raw = list(bbox)

        # Whatever we end up sending to the VLM: either the full images, or the crops.
        query_paths:  list[str]  = []   # what to attach
        query_labels: list[str]  = []   # caption for each (interleaved by the client)
        crop_infos:   list[list] = []   # the clamped boxes, echoed back in the result

        if not bboxes_raw:
            # No bbox: inspect each image whole. Label differs for 1 vs many so the model
            # knows whether it's answering about one view or comparing several.
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
                # Clamp to the frame. Boxes come from the detector or the model, and
                # either can overhang the edge.
                x1, y1, x2, y2 = [int(v) for v in raw_box[:4]]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if x2 <= x1 or y2 <= y1:
                    # Degenerate after clamping (inverted or entirely off-frame). Tell the
                    # model precisely which box was bad so it can re-issue a fixed one.
                    return ToolResult(ok=False, tool="inspect",
                                      summary=f"Invalid bbox[{i}]: [{x1},{y1},{x2},{y2}]")
                try:
                    # Upscale every crop to 512×512 with LANCZOS. This is the whole point
                    # of cropping — a tiny region becomes a full-resolution image for the
                    # VLM. Aspect ratio is intentionally not preserved; the model tolerates
                    # the stretch far better than it tolerates a 30×20 thumbnail.
                    crop = img.crop((x1, y1, x2, y2)).resize(
                        (512, 512), _PIL.LANCZOS)
                    # step*100 + i keeps crop filenames unique and ordered even when one
                    # step produces several crops (up to 100 per step).
                    crop_path = os.path.join(
                        crops_dir,
                        f"{self._step_counter * 100 + i:06d}.png"
                    )
                    os.makedirs(crops_dir, exist_ok=True)
                    crop.save(crop_path)
                    query_paths.append(crop_path)
                    # Caption carries the source frame and the original box, so the model
                    # can map what it sees in the close-up back onto the full view.
                    query_labels.append(
                        f"Crop {i+1} of {len(bboxes_raw)} "
                        f"(bbox=[{x1},{y1},{x2},{y2}] from {single}):")
                    crop_infos.append([x1, y1, x2, y2])
                except Exception as e:
                    return ToolResult(ok=False, tool="inspect",
                                      summary=f"Crop[{i}] failed: {e}")

        # The nested VLM call. Note ok=True even when the model returns nothing usable:
        # inspect *ran*, it just didn't learn anything, and the images are still attached
        # for the policy to look at itself.
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
        """Two-stage visual memory search: embedding recall, then VLM rerank.

        Pipeline:
          1. FAISS nearest-neighbours over the scene's frame embeddings (cheap, high
             recall, noisy — finds frames that *look like* the query).
          2. Show every survivor to the VLM and have it rank them by whether the target is
             genuinely visible (expensive, high precision).
          3. Truncate to top_k and return, with the frames attached as images.

        The result's `navigate_target` field spells out the exact argument for a follow-up
        navigate() call, so the model never has to construct one by hand.
        """
        from memory.retrieval import retrieve_memory_candidates

        # -1 is the "everything" sentinel; anything else is floored at 1.
        top_k    = -1 if int(top_k) == -1 else max(1, int(top_k))
        index_dir = os.path.join(
            self.retrieval_data_root, self.scene_id,
            f"retrieval_index_{self.retrieval_model}",
        )

        if not os.path.exists(os.path.join(index_dir, "index.bin")):
            return ToolResult(ok=False, tool="retrieve_memory",
                              summary="No memory index found. Scan the scene first.")

        # The query is polymorphic: an existing file path means "find frames like this
        # image", anything else means "find frames matching this text".
        is_image_query   = os.path.isfile(query)
        query_image_paths = [query] if is_image_query else None
        # When a time filter is active, fetch *everything* from the index and filter
        # afterwards. Fetching only top_k first would let the k nearest frames all fall
        # outside the window, yielding nothing even though matching frames exist.
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

        # Post-filter by time window (see fetch_k above — we deliberately over-fetched).
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

        # ── Stage 2: Gemini rerank ────────────────────────────────────────────
        # Entirely best-effort. If the rerank call fails, or the model returns a malformed
        # ranking, we keep the embedding order — degraded but still useful. Retrieval must
        # not fail just because the reranker had a bad day.
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
            # Per-candidate reasoning, keyed by id so it can be merged back below.
            for entry in rerank_result.get("candidates_analysis", []):
                mid = entry.get("memory_id")
                if mid:
                    analysis_by_id[mid] = entry
            # Only trust the ranking if it's a full permutation. A short list means the
            # model dropped candidates, which would silently discard real memories — in
            # that case keep the embedding order rather than a partial one.
            if ranked_ids and len(ranked_ids) == len(candidates):
                id_map   = {c.memory_id: c for c in candidates}
                reordered = [id_map[mid] for mid in ranked_ids if mid in id_map]
                # Anything the model omitted or hallucinated an id for still gets appended,
                # so no candidate is ever lost — just demoted to the end.
                reordered += [c for c in candidates
                              if c.memory_id not in set(ranked_ids)]
                candidates = reordered
        except Exception as e:
            print(f"[Toolbox] rerank error (non-fatal): {e}")

        # Truncate *after* reranking: the point of over-fetching is to let the VLM promote
        # a frame that embedding similarity ranked 20th into the top few.
        if top_k != -1:
            candidates = candidates[:top_k]

        # Flatten into the dicts the prompt renders, merging in the rerank reasoning.
        cand_dicts = []
        for c in candidates:
            analysis = analysis_by_id.get(c.memory_id, {})
            entry: dict = {
                "memory_id":       c.memory_id,
                # Pre-built argument for navigate(). Handing the model the literal call
                # removes the most common source of malformed navigate actions.
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
        """Textual memory: what the agent *did* in a time window, and how it turned out.

        The counterpart to retrieve_memory (which is visual memory of what it *saw*).
        Reads back the very trajectory.jsonl the TrajectoryLogger is writing as the episode
        runs — the agent is literally querying its own log. This is what lets it answer
        "have I already tried grasping this?" without keeping every step in context.
        """
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
                        # Tolerate a torn final line: the logger flushes per step, but we
                        # may be reading while a write is in flight. Skip, don't fail.
                        continue
                    if not _in_time_window(entry.get("timestamp"), t_from, t_to):
                        continue
                    # Project each fat JSONL row down to the fields worth spending prompt
                    # tokens on: what was tried, what the agent expected, what happened,
                    # and where the robot was. Images and raw prompts are omitted.
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
        """Drive the base to a remembered observation's pose, then face the way it faced.

        Structure: plan a path once, then run a pure-pursuit-style loop — repeatedly steer
        toward the current waypoint, advancing to the next as each is reached. The heavy
        lifting (obstacle-aware planning) is in the backend's _plan_path; this loop only
        follows the polyline and decides when to stop.

        Only a memory_id is accepted as a target (see _resolve_navigate_target for why).
        After arriving, we replay the remembered *yaw* too, so the robot ends up looking at
        what the frame was looking at — arriving at the right spot facing a wall is useless.
        """
        target_xy, target_yaw, nav_label = self._resolve_navigate_target(target)
        if target_xy is None:
            # nav_label carries the specific reason (bad id / unknown id / not a memory_id).
            return ToolResult(ok=False, tool="navigate",
                              summary=f"Could not resolve navigate target: {target}")

        start_xy   = np.array(self._get_robot_pose()[:2])
        waypoints  = self._plan_path(start_xy, target_xy)
        total_dist = float(np.linalg.norm(target_xy - start_xy))
        print(f"[Toolbox] navigate: {len(waypoints)} waypoints  "
              f"target=({target_xy[0]:.2f},{target_xy[1]:.2f})  "
              f"dist={total_dist:.2f}m")

        NAV_STOP_DIST = 0.5   # "arrived": within half a metre of the goal
        WP_ADV = 0.3          # waypoint capture radius — advance once this close
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

            # (a) Success: close enough to the goal.
            if dist < NAV_STOP_DIST:
                outcome = f"reached  dist={dist:.2f}m"
                break

            # (b) Ran out of path. We consumed the final waypoint but are still further
            # than NAV_STOP_DIST from the goal — typically because the goal sits just off
            # the navmesh (e.g. a pose recorded against a wall). Stop where the path ends
            # rather than grinding into geometry; "best_effort" still counts as ok, since
            # the robot did get as close as the map allows.
            if (wp_idx == len(waypoints) - 1 and
                    np.linalg.norm(waypoints[-1] - curr_xy) < WP_ADV):
                outcome = f"best_effort  dist={dist:.2f}m"
                break

            # Advance the waypoint cursor past every waypoint we're already within
            # WP_ADV of. A `while` (not an `if`) because a tight corner can put several
            # closely-spaced waypoints inside the capture radius at once, and stepping
            # one per control tick would stall the robot.
            while wp_idx < len(waypoints) - 1:
                if np.linalg.norm(waypoints[wp_idx] - curr_xy) < WP_ADV:
                    wp_idx += 1
                else:
                    break

            # Bearing = angle to the active waypoint, expressed in the robot's own frame
            # (hence subtracting curr_yaw), wrapped to (-pi, pi] so the robot turns the
            # short way round rather than taking the long way.
            delta   = waypoints[wp_idx] - curr_xy
            bearing = math.atan2(delta[1], delta[0]) - curr_yaw
            bearing = (bearing + math.pi) % (2 * math.pi) - math.pi

            self._navigate_step(bearing)
            self._last_obs = self._step()

        else:
            # for/else: the loop ran to _NAV_MAX_STEPS without breaking — the robot is
            # stuck (wedged on geometry, or oscillating). Report it and give up.
            outcome = f"max_steps  dist={dist:.2f}m"

        # Reproduce the remembered heading, so the target that was visible in the memory
        # frame is visible again now.
        if target_yaw is not None:
            self._align_yaw(target_yaw)

        obs_result = self.observe()
        final_xy   = np.array(self._get_robot_pose()[:2])
        final_dist = float(np.linalg.norm(target_xy - final_xy))
        # Re-measure rather than trusting `outcome`: _align_yaw may have nudged the base.
        # The +0.5 slack forgives that drift.
        reached    = "reached" in outcome or final_dist < NAV_STOP_DIST + 0.5

        return ToolResult(
            # best_effort is a success: the robot went as far as the map permits, and the
            # model should now look around rather than retry the same navigate.
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
        """One small, bounded base motion — the agent's last-mile adjustment primitive.

        Closed-loop, not open-loop: rather than commanding a fixed number of ticks, we keep
        stepping until the *measured* pose change reaches the quantum (0.30 m or 30°), or
        we hit the step cap. That distinction is what makes `completed` meaningful — if the
        robot is pressed against a table, the loop runs out of steps having moved nothing,
        and the model is told the move was blocked instead of believing it succeeded.
        """
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
        # What "done" means for this motion: an angle for rotations, a distance otherwise.
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
                # Signed, wrapped yaw change. Negate for the clockwise motion so `progress`
                # is always a positive quantity counting *toward* the target.
                yaw_delta = self._angle_diff(curr_yaw, start_yaw)
                if motion == "rotate 30 degrees":
                    progress = yaw_delta
                else:
                    progress = -yaw_delta
            else:
                # Straight-line displacement from the start. Unsigned by design: the caller
                # picked the direction, and we only care that the base actually travelled.
                progress = float(np.linalg.norm(curr_xy - start_xy))

            # 95% — stop just shy of the target rather than overshooting past it while
            # waiting for an exact hit that floating-point control will never produce.
            if progress >= target * 0.95:
                break

        # Re-measure from the final pose. The in-loop `progress` is stale by one tick, and
        # for the max-steps case it never reached the break at all.
        final_pose = self._get_robot_pose()
        final_xy = np.array(final_pose[:2], dtype=np.float64)
        final_yaw = float(final_pose[2]) if len(final_pose) >= 3 else start_yaw
        yaw_delta = self._angle_diff(final_yaw, start_yaw)
        distance = float(np.linalg.norm(final_xy - start_xy))
        # Asymmetric completion thresholds. Rotation is nearly unobstructable, so we hold it
        # to 75% of the target. Translation routinely gets cut short by furniture and the
        # navmesh, and a partial 50% nudge is still a genuinely useful reposition — failing
        # it would make the model retry a move that did in fact help.
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
        """Arm skills: grasp / place / drop.

        The key subtlety is in grasp, and it is worth stating plainly. `_grasp` returns
        `localized`, which means "we found something graspable under the aimed pixel and
        attempted a grasp" — NOT "we are now holding it". Whether an object is actually held
        is a separate question, answered by `_grasp_state()` from the physics engine. Two
        distinct failure modes therefore get two distinct messages:

            not localized  → "Could not localize 'X'"      (couldn't even try — aim elsewhere)
            grasped=False  → "Grasp on X FAILED: gripper is empty"  (tried, missed — retry/reposition)

        Conflating them would leave the model unable to tell "look somewhere else" from
        "get closer and try again".
        """
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
        """Let the world run without commanding anything.

        Used to let physics settle (a dropped object still rolling), and as the safe
        fallback the agent loop substitutes when the VLM emits an unusable action.

        Note this is *simulated* time, not wall-clock: we step the backend 30 times per
        requested second (the control rate) rather than sleeping. The clamp to [0.1, 100]
        matches the bound advertised in the system prompt.
        """
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
        """The agent declares the episode over.

        Unconditionally ok=True here: the base toolbox has no notion of what the task was,
        so it cannot judge whether finishing was justified. Benchmark backends (e.g. OVMM)
        override this to check the environment's real success condition and can return
        ok=False, which the agent loop propagates straight into episode success.
        """
        return ToolResult(
            ok=True, tool="finish",
            summary=f"finish: {answer}" if answer else "finish",
            # task_done is the flag the agent loop watches to break out of its step loop.
            data={"answer": answer, "task_done": True},
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_navigate_target(
        self, target: Any
    ) -> tuple[Optional[np.ndarray], Optional[float], str]:
        """Turn a VLM-supplied navigate target into (xy, yaw, label).

        Only memory_ids are accepted — never raw coordinates. This is a deliberate
        capability restriction: a coordinate goal could come from a hallucination or from
        ground-truth the agent isn't supposed to have, letting it teleport to objects it
        never perceived. Requiring a memory_id means every navigation goal is somewhere the
        robot has provably already stood and photographed.

        The third element is a human-readable label that doubles as the failure reason when
        xy is None, so the caller can hand the model a specific explanation.
        """
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
        """memory_id → (xy, yaw), trying the scan directory first and EpisodicMemory second.

        Two sources because memories have two origins. Frames from the offline scene scan
        live on disk as robot_xy/{idx}.txt sidecars and can be resolved without any of the
        memory machinery being initialised. Frames the agent captured during the episode
        only exist in EpisodicMemory. Checking the file first is also the cheaper path.

        `.lstrip("0") or "0"` guards the edge case of mem_000000, whose numeric part strips
        to the empty string.
        """
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
                    return np.array([data[0], data[1]]), None   # position only, no heading
        except Exception as e:
            print(f"[Toolbox] _pose_from_memory_id file lookup error: {e}")

        # Fall back to the in-episode memory store.
        if self.episodic_memory is not None:
            result = self.episodic_memory.get_pose(memory_id)
            if result is not None:
                return result
        return None, None   # unknown id — navigate() reports it to the model

    def _estimate_object_xy_from_depth(
        self, target: str
    ) -> Optional[np.ndarray]:
        """Back-project depth + Gemini bbox → world XY.

        The perception→geometry bridge: turn the *name* of an object into a metric position
        the robot can act on. Three stages:

          1. Localise it in the image. Prefer GroundingDINO (accurate boxes); fall back to
             asking the VLM for a box; fall back again to the image centre.
          2. Read a depth value at that box. Median over the box interior, not the centre
             pixel — the centre may land on a hole, a specular highlight, or a background
             surface seen through a handle, whereas the median is robust to all three.
          3. Un-project (u, v, d) through the intrinsics K into camera coordinates, then
             through the inverse extrinsics E into world coordinates.

        Returns None whenever any stage fails; callers must handle that (no position means
        no grasp).
        """
        obs = self._last_obs
        if obs is None:
            return None

        # ── Stage 1a: GroundingDINO (preferred) ───────────────────────────────
        bbox: Optional[list] = None
        if self._grounding_dino is not None and self._last_image_path:
            try:
                from PIL import Image as _PIL
                img_pil = _PIL.open(self._last_image_path).convert("RGB")
                detections = self._grounding_dino.detect(
                    np.array(img_pil), target
                )
                if detections:
                    # Highest-scoring detection wins (the list is score-sorted).
                    bbox = detections[0]["bbox"]
                    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
                    W, H = img_pil.size
                    x1, y1 = max(0, min(W - 1, x1)), max(0, min(H - 1, y1))
                    x2, y2 = max(x1 + 1, min(W, x2)), max(y1 + 1, min(H, y2))
                    bbox = [x1, y1, x2, y2]
                    # Debug artefact only — lets a human confirm from the logs that the
                    # detector locked onto the right thing before the robot grabbed at it.
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

        # ── Stage 1b: fall back to asking the VLM for a box ───────────────────
        # Less accurate than the detector, but it can name things the detector misses.
        if bbox is None:
            inspect_result = self.inspect(
                image_path=self._last_image_path or "",
                question=f"Locate the {target}. Return its pixel bounding box.",
            )
            if inspect_result.ok:
                bboxes = inspect_result.data.get("candidate_bboxes", [])
                if bboxes:
                    bbox = bboxes[0]

        # ── Stage 2: sample depth ─────────────────────────────────────────────
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
            # The pixel we un-project is the box centre; the depth we use is the median
            # over the whole box. Mixing the two is intentional — the centre is the best
            # single guess at the object's image position, while the median depth is a far
            # more robust distance estimate than that one pixel's reading.
            u_c = max(0, min(W - 1, (x1 + x2) // 2))
            v_c = max(0, min(H - 1, (y1 + y2) // 2))
            patch = depth_np[y1:y2, x1:x2]
            # Discard sensor invalids: 0/near-0 means "no return", >10 m is beyond anything
            # in an indoor scene and is usually a sky/void pixel.
            valid = patch[(patch > 0.05) & (patch < 10.0)]
            if valid.size > 0:
                d = float(np.median(valid))
            else:
                d = float(depth_np[v_c, u_c])  # fall back to bbox centre pixel
        else:
            # ── Stage 1c: no box at all — assume the object is what we're looking at.
            u_c, v_c = W // 2, H // 2
            d = float(depth_np[v_c, u_c])

        # Reject an implausible depth outright rather than back-projecting a point to
        # infinity and sending the arm somewhere absurd.
        if d < 0.05 or d > 10.0:
            print(f"[Toolbox] _estimate_object_xy: invalid depth {d:.3f}m")
            return None

        # ── Stage 3: un-project pixel + depth → world point ───────────────────
        # Standard pinhole inverse: (u - cx) * d / fx gives the camera-frame X offset, and
        # likewise for Y. Homogeneous, so the 4×4 extrinsics can be applied directly.
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        x_c    = (u_c - cx) * d / fx
        y_c    = (v_c - cy) * d / fy
        p_cam  = np.array([x_c, y_c, d, 1.0])
        # E is world→camera, so its inverse takes the camera-frame point back to the world.
        p_world = np.linalg.inv(E) @ p_cam

        # Only XY: the robot navigates on the floor plane, so height is discarded here.
        obj_xy = p_world[:2].copy()
        print(f"[Toolbox] _estimate_object_xy: '{target}'  "
              f"bbox={bbox}  depth={d:.2f}m  "
              f"world_xy=({obj_xy[0]:.2f},{obj_xy[1]:.2f})")
        return obj_xy

    def _align_yaw(self, target_yaw: float) -> None:
        """Rotate in place until the robot faces target_yaw (±5°).

        The clamp is the whole trick, and it is easy to break by "simplifying".
        """
        # kinematic_nav_step moves FORWARD when |bearing| < ROT_THRESH (~8°).
        # Clamp bearing to always exceed that threshold so we only rotate here.
        #
        # Without the clamp, a small residual bearing (say 6°) would be read by
        # _navigate_step as "close enough to aligned — drive forward", so the robot would
        # creep away from the pose it just navigated to instead of turning the last few
        # degrees. Forcing |bearing| ≥ 9° keeps every command a pure rotation, while the
        # 5° exit tolerance (below the 8° threshold) guarantees the loop still terminates.
        _MIN_ROT = math.radians(9)
        for _ in range(400):   # watchdog: bail out rather than spin forever
            pose    = self._get_robot_pose()
            # Wrapped to (-pi, pi] so we always turn the short way round.
            bearing = ((target_yaw - pose[2] + math.pi) % (2 * math.pi) - math.pi)
            if abs(bearing) < math.radians(5):
                break
            if abs(bearing) < _MIN_ROT:
                bearing = math.copysign(_MIN_ROT, bearing)   # preserve turn direction
            self._navigate_step(bearing)
            self._step()

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Signed a-b, wrapped to (-pi, pi]. Handles the ±pi discontinuity correctly."""
        return (a - b + math.pi) % (2 * math.pi) - math.pi

    def _snap_to_navmesh(self, xy: np.ndarray) -> np.ndarray:
        """Snap xy to nearest navmesh vertex if available. Override to add navmesh.

        Identity in the base class — a backend with no map (e.g. a real robot) has nothing
        to snap to. Sim backends override it so a goal that lands inside furniture is pulled
        back onto walkable floor.
        """
        return xy

    def _save_frame(self, rgb: np.ndarray, tag: str) -> str:
        """Persist an observation frame and advance the step counter.

        The counter both names the file and identifies the step, so image numbering and
        step numbering are the same sequence by construction.
        """
        from PIL import Image as _PIL
        self._step_counter += 1
        path = str(self.log_dir / "images" / f"{self._step_counter:04d}_{tag}.png")
        _PIL.fromarray(rgb).save(path)
        return path

    def _save_companion_frame(self, rgb: np.ndarray, tag: str) -> str:
        """Save an extra frame for the CURRENT step without advancing the step
        counter, so it pairs with the head frame just saved.

        For secondary cameras (e.g. the wrist cam): the file shares the head frame's number
        and differs only in `tag`, so the two sort together and are obviously one step's
        worth of observation.
        """
        from PIL import Image as _PIL
        path = str(self.log_dir / "images" / f"{self._step_counter:04d}_{tag}.png")
        _PIL.fromarray(rgb).save(path)
        return path

    def _pose_str(self, pose: list) -> str:
        """Compact pose for logs and tool summaries — yaw in degrees, since that is what
        both humans and the VLM read reliably."""
        if len(pose) >= 3:
            return f"({pose[0]:.2f},{pose[1]:.2f} yaw={math.degrees(pose[2]):.0f}°)"
        return str(pose)

    def _pump_events(self) -> None:
        """Give the GUI a slice of time mid-control-loop, if a viewer is attached.

        Called from inside the long-running loops (navigate, base_move, wait) — without it
        the window would freeze for the whole motion. Swallows exceptions: a rendering
        glitch must never take down a control loop that is driving a robot.
        """
        if self.event_callback:
            try:
                self.event_callback()
            except Exception:
                pass
