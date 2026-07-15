"""
sim/habitat_toolbox.py — HabitatToolbox: Habitat/ReplicaCAD backend for PromptEmbodiedAgent.

Uses the same FetchRobot + ReplicaCAD + oracle navigation setup as OWMM-Agent,
enabling a fair side-by-side comparison.

Coordinate convention
---------------------
Habitat world: X right, Y up, Z back (right-hand).
Navigation plane: XZ (Y is vertical).
BaseToolbox "2D pose":  [x_nav, y_nav, yaw]  maps to  [world.X, world.Z, heading].
oracle_nav_coord_action: [world.X, world.Y_floor, world.Z, orientation_rad].

Camera convention (Habitat depth sensor)
-----------------------------------------
Camera space: X right, Y down, Z forward-into-scene but depth is positive along +Z_cam.
Habitat uses -Z_cam convention (point in front → z_cam = -depth).  The depth_rot /
depth_trans sensors expose the camera-to-world 3 × 3 rotation and 3-D translation.

Reading guide — this file has four fairly independent parts:

  1. Module constants (below): the thresholds that decide when a grasp/place is allowed to
     fire. These are the numbers you actually tune when manipulation misbehaves.
  2. The pygame display plumbing: a subprocess, for a non-obvious reason documented there.
  3. HabitatToolbox itself: the BaseToolbox primitives (_step, _get_robot_pose, _grasp, …)
     re-implemented against habitat.Env.
  4. Overrides of the *high-level* tools — navigate() in particular, which does NOT use
     BaseToolbox's pure-pursuit loop but delegates to Habitat's oracle_nav_coord_action, so
     that this backend's navigation is identical to OWMM-Agent's and the comparison is fair.

The single biggest source of confusion here is the coordinate mismatch spelled out above:
Habitat is Y-up, so the floor plane is XZ, while BaseToolbox's generic code assumes the
floor is XY. Every pose crossing that boundary is remapped ([x, z, yaw] ⇄ [X, Y_floor, Z]),
and the remapping is easy to get subtly wrong.
"""

from __future__ import annotations

import math
import os
import threading
import time
from typing import Callable, Optional

import numpy as np

from agent.toolbox_base import BaseToolbox
from agent.schemas import ToolResult

# ── Manipulation thresholds ──────────────────────────────────────────────────
# These govern when a grasp or place is permitted to succeed. They are the knobs to reach
# for first when the arm is "almost" working: too tight and the agent can never grasp
# anything, too loose and it grasps through walls.

# OWMM-style IK-reach pick: max arm steps and how close the end-effector must
# get to the *target* object before the magic-grasp snap is allowed.
_ARM_MAX_STEPS      = 90     # watchdog: give up if IK can't converge in this many steps
_GRASP_REACH_THRESH = 0.12   # metres (EE → target object centre)
# When a grasp ray hits the receptacle under a small object, snap to a pickable
# scene object within this radius of the hit point (see _raycast_pixel_to_object).
# Horizontal radius is tight (honesty); vertical is looser since objects rest ABOVE
# the surface the ray lands on. Override via GRASP_SNAP_RADIUS to tune.
_GRASP_SNAP_RADIUS   = float(os.environ.get("GRASP_SNAP_RADIUS", "0.30"))   # metres, horizontal
_GRASP_SNAP_RADIUS_Y = float(os.environ.get("GRASP_SNAP_RADIUS_Y", "0.45")) # metres, vertical
# Suction: start closing the gripper once the EE is this close to the target so
# the fingers actually engage (an open gripper straddles small objects without
# ever touching them → no contact → no grasp).
_GRASP_CLOSE_DIST   = 0.18   # metres
_PLACE_MAX_STEPS    = 90
# Looser than the grasp threshold (0.20 vs 0.12 m): a receptacle is a large surface and the
# target point is only an estimate of somewhere on it, whereas a grasp has to hit a specific
# small object.
_PLACE_REACH_THRESH = 0.20   # metres (EE → visible receptacle/top point)
# After releasing, keep stepping so the object actually falls and comes to rest before the
# next observation — otherwise the agent photographs it mid-air and mis-verifies the place.
_PLACE_SETTLE_STEPS = 25
# Arm retract after a manipulation: blend this fraction of the way from the
# current pose toward the rest pose (0 = stay put, 1 = full tuck). Partial so
# the arm just clears the workspace without folding all the way back.
_ARM_RETRACT_FRAC = 0.4
# Wall-clock seconds between live memory writes while the robot performs a task.
# Rate-limits episodic-memory ingestion: the control loop runs far faster than this, and
# indexing every frame would fill memory with near-duplicates.
_LIVE_INGEST_INTERVAL = 3.0
# Base velocity magnitudes for kinematic steps
_FWD_VEL   = 0.25       # m/s forward speed for base_velocity
_TURN_VEL  = 0.5        # rad/s turn speed
# Turn-then-drive threshold: if the goal is more than 20° off the nose, rotate in place
# rather than arcing toward it. (Wider than the ManiSkill backend's 8° — Habitat's velocity
# controller arcs more gracefully, so it tolerates a larger error before a pure turn helps.)
_ROT_THRESH = math.radians(20)   # steer-only threshold

# ── pygame rendering (same env-var interface as OWMM-Agent) ──────────────────
# Habitat-sim (EGL/headless) conflicts with pygame (GLX) in the same process on
# Ubuntu. Fix: run the pygame display in a forked subprocess launched BEFORE
# habitat loads, communicating frames via multiprocessing.Queue.

_RENDER = os.environ.get("HABITAT_RENDER", "0") == "1"   # show a live pygame window
_STEP   = os.environ.get("HABITAT_STEP",   "0") == "1"   # pause for SPACE before each action
_RECORD_THIRD_PERSON = os.environ.get("HABITAT_RECORD_THIRD_PERSON", "0") == "1"
# Grasp model: "suction" = grab only once the gripper is in physics CONTACT
# with the target (realistic trigger), seating it in the hand (force=True);
# "magic" = snap once the EE is within _GRASP_REACH_THRESH of the target, no
# contact required. Override with GRASP_MODE=magic.
_GRASP_MODE = os.environ.get("GRASP_MODE", "suction").lower()
# Render every Nth sim step inside long action loops (oracle-nav can run
# hundreds of steps); throttled so we don't flood the display mp.Queue.
_RENDER_EVERY = 4

import multiprocessing as _mp

_display_queue: Optional["_mp.Queue"] = None  # frame queue to display process
_ack_queue:     Optional["_mp.Queue"] = None  # acknowledgement for step-mode
_display_proc:  Optional["_mp.Process"] = None


def _display_worker(frame_q, ack_q):
    """Runs in a separate process (no Habitat EGL — can safely use X11/pygame).

    Loop: pull a (frame, pause) tuple off the queue, blit it, and — if paused — block until
    the user presses SPACE, then acknowledge on ack_q. The ack is what makes step-mode work:
    the *main* process blocks reading ack_q, so the simulator genuinely waits for the human.
    """
    import pygame
    screen = None
    while True:
        item = frame_q.get()
        if item is None:          # sentinel → quit
            if screen:
                pygame.quit()
            return
        frame, pause = item
        # Lazily create (or recreate) the window: the frame size isn't known until the first
        # frame arrives, and it changes if the third-person recorder is toggled on.
        if screen is None or screen.get_size() != (frame.shape[1], frame.shape[0]):
            pygame.init()
            screen = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
            print("[pygame] window opened — SPACE to step, Q to quit", flush=True)

        # Drain the event queue even when not pausing — an unpumped SDL window is marked
        # unresponsive by the window manager and stops repainting.
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                return

        # numpy is (row, col) = (y, x); pygame surfaces are (x, y). Hence the transpose.
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        cap = "EmbodiedAgent  [SPACE to execute]" if pause else "EmbodiedAgent"
        pygame.display.set_caption(cap)
        pygame.display.flip()

        if pause:
            # Block here until the user advances. The 20 ms sleep keeps this from spinning
            # a core while waiting.
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        ack_q.put("quit")   # tell the main process we're gone
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key in (pygame.K_SPACE, pygame.K_RETURN, pygame.K_RIGHT):
                            waiting = False
                        elif event.key == pygame.K_q:
                            ack_q.put("quit")
                            return
                threading.Event().wait(0.02)
            ack_q.put("ok")     # unblocks _pg_show in the main process


def start_display_process():
    """Launch the pygame display worker before habitat-sim is imported.

    The ordering in the name is a hard requirement, not a preference — see the comment above
    on the EGL/GLX conflict. Once habitat-sim has initialised EGL in this process, forking a
    child that tries to open an X11 window will fail or crash.
    """
    global _display_queue, _ack_queue, _display_proc
    if _display_proc is not None:
        return   # idempotent
    ctx = _mp.get_context("spawn")   # spawn avoids inheriting any EGL state
                                     # (fork would copy the parent's GL context and deadlock)
    _display_queue = ctx.Queue()
    _ack_queue     = ctx.Queue()
    _display_proc  = ctx.Process(
        target=_display_worker,
        args=(_display_queue, _ack_queue),
        daemon=True,   # never outlive the main process
    )
    _display_proc.start()
    print("[pygame] display process started", flush=True)


def stop_display_process():
    """Ask the display worker to exit (the None sentinel breaks its loop)."""
    if _display_queue is not None:
        _display_queue.put(None)


def _pg_show(frame: np.ndarray, pause: bool = False):
    """Send a frame to the display process.

    No-op when rendering is off, so call sites don't need to guard. The .copy() is required:
    the frame is a view into a buffer the simulator will overwrite, and the queue serialises
    it asynchronously.
    """
    if not _RENDER or _display_queue is None:
        return
    _display_queue.put((frame.copy(), pause))
    if pause:
        _ack_queue.get()   # block until user presses SPACE


class HabitatToolbox(BaseToolbox):
    """
    Habitat/ReplicaCAD backend for PromptEmbodiedAgent.

    Wraps ``habitat.Env`` and implements the abstract primitives from
    ``BaseToolbox``.  High-level ``navigate()`` is overridden to use
    Habitat's oracle_nav_coord_action (identical to OWMM-Agent's navigation
    primitive) for a fair comparison.
    """

    # ── Navigation stop-distance — matches oracle_nav's dist_thresh ────────────
    _NAV_STOP_DIST = 0.4    # metres
    _NAV_MAX_STEPS = 2000   # hard cap on oracle-nav iterations

    def __init__(
        self,
        hab_env,                           # habitat.Env instance (already reset)
        gemini_client,
        log_dir: str,
        capture_out_dir: str,
        scene_id: Optional[str] = None,
        embedding_worker=None,
        episodic_memory=None,
        retrieval_model: str = "siglip_base",
        retrieval_data_root: Optional[str] = None,
        event_callback: Optional[Callable] = None,
        grounding_dino=None,
        initial_obs: Optional[dict] = None,
        display: bool = False,
        primary_camera: str = "head",
    ):
        super().__init__(
            gemini_client=gemini_client,
            log_dir=log_dir,
            capture_out_dir=capture_out_dir,
            embedding_worker=embedding_worker,
            episodic_memory=episodic_memory,
            retrieval_model=retrieval_model,
            retrieval_data_root=retrieval_data_root,
            scene_id=scene_id,
            event_callback=event_callback,
            grounding_dino=grounding_dino,
        )
        self._env = hab_env
        self._last_obs: Optional[dict] = initial_obs
        self._display = display
        # Which camera feeds the agent's observation: "head" (forward-facing,
        # correct for navigation/OVMM) or "arm_workspace" (head view + arm
        # reachability overlay — the OWMM baseline's eval camera, for parity).
        self._primary_camera = primary_camera
        # Two fix-ups applied to the freshly-reset env before anything runs. Both are
        # working around Habitat defaults that are wrong for this agent — see each method.
        self._configure_gripper_camera()
        self._force_dynamic_scene_objects()
        self._third_person_frame_idx = 0
        self._third_person_frame_dir = self.capture_out_dir / "third_person"
        if _RECORD_THIRD_PERSON:
            self._third_person_frame_dir.mkdir(parents=True, exist_ok=True)

        # ── Live memory ingestion ─────────────────────────────────────────────
        # Fold the head-camera frames the robot sees WHILE performing the task
        # into the same FAISS + episodic memory built by scan_scene, so it
        # remembers what it observed on its way. Throttled by wall-clock so we
        # add roughly one frame every _LIVE_INGEST_INTERVAL seconds.
        self._live_navcap = None
        self._last_ingest_t = 0.0

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def _sim(self):
        """The raw habitat_sim Simulator, beneath the RL-env wrapper."""
        return self._env._sim

    def _robot(self):
        """The Fetch articulated agent. Single-agent setup, hence the hard-coded [0]."""
        return self._sim.agents_mgr[0].articulated_agent

    def _configure_gripper_camera(self) -> None:
        """Aim Fetch's arm camera at the gripper instead of the floor/body.

        Habitat's stock arm camera points at nothing useful, which makes it worthless for
        verifying a grasp. The block below re-mounts it. The geometry is fiddly and was
        derived empirically (with tools/tune_wrist_cam.py and tools/wrist_cam_diag.py), so
        the reasoning is recorded inline rather than left to be re-discovered.

        Non-fatal on failure: a bad camera costs a useful view, not the run.
        """
        try:
            import magnum as mn

            robot = self._robot()
            cam_info = robot.params.cameras.get("articulated_agent_arm")
            if cam_info is None:
                return

            # Camera-link frame (link 22, rigid to the gripper): +X → fingertips,
            # +Y → up, ±Z → lateral. The end-effector measures at ~(0.08, 0, 0)
            # in this frame for ANY arm pose (see tools/wrist_cam_diag.py).
            # The two fingers measure at l=(0.08,-0.04,0) and r=(0.08,0.056,0) in
            # this frame — they separate along the gripper's Y (open/close) axis,
            # finger length along +X. So a pure top-down view looks straight DOWN
            # that separation axis and the near finger hides the far one (only one
            # visible). To show BOTH fingers + the gap, look ACROSS them along Z.
            # "s_up_r90" mount (tuned via tools/tune_wrist_cam.py with the
            # production-faithful configure-at-tucked→view-at-reach flow): camera
            # to the +Z side and elevated (+Y) for a top-ish angle, looking at the
            # finger center (0.08, 0.008, 0); roll 90 lays the two fingers
            # side-by-side. Shows both fingers empty AND a held object between them.
            cam_info.cam_offset_pos  = mn.Vector3(0.07, 0.09, 0.10)
            cam_info.cam_look_at_pos = mn.Vector3(0.08, 0.008, 0.00)
            cam_info.relative_transform = mn.Matrix4.rotation_z(mn.Deg(90.0))
            robot.update()
            print("[camera] articulated_agent_arm_rgb: elevated cross-finger view "
                  "(offset 0.07,0.09,0.10 look_at 0.08,0.008,0 roll 90) — both "
                  "fingers visible", flush=True)
        except Exception as exc:
            print(f"[camera] gripper camera pose override skipped: {exc}", flush=True)

    def _force_dynamic_scene_objects(self) -> None:
        """Ensure loaded episode rigid objects participate in physics.

        ``OVMM_FORCE_ALL_RIGID_DYNAMIC=1`` also converts every rigid handle in
        the scene, but that can destabilize HSSD scenes because many large
        environment assets start interpenetrating other geometry.

        Why this is needed at all: OVMM episodes load their target objects as STATIC by
        default, so a "grasped" object would not move with the gripper and a placed one
        would hang in mid-air. Converting them to DYNAMIC is what makes manipulation
        physical rather than cosmetic.

        Why it is scoped by default: `scene_obj_ids` is just the episode's own objects.
        The env-var escape hatch converts *every* rigid body in the scene, which sounds more
        thorough but in practice explodes — HSSD scenes ship with large assets that start
        interpenetrating, and waking them all up launches the furniture across the room.
        """
        try:
            import habitat_sim
            import magnum as mn

            rom = self._sim.get_rigid_object_manager()
            force_all = os.environ.get("OVMM_FORCE_ALL_RIGID_DYNAMIC", "0") == "1"
            if force_all:
                objects = [
                    rom.get_object_by_handle(handle)
                    for handle in rom.get_object_handles()
                ]
            else:
                objects = [
                    rom.get_object_by_id(obj_id)
                    for obj_id in getattr(self._sim, "scene_obj_ids", [])
                ]
            converted = 0
            for obj in objects:
                if obj is None:
                    continue
                obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
                obj.collidable = True
                # Zero the velocities: switching motion type can leave stale values that
                # would fling the object the instant physics resumes.
                obj.linear_velocity = mn.Vector3.zero_init()
                obj.angular_velocity = mn.Vector3.zero_init()
                # awake=True — a sleeping body ignores contacts until something wakes it, so
                # the gripper would pass straight through the object it is trying to grasp.
                obj.awake = True
                converted += 1
            scope = "all rigid handles" if force_all else "episode scene objects"
            print(f"[physics] forced {converted} {scope} dynamic/collidable",
                  flush=True)
        except Exception as exc:
            print(f"[physics] dynamic scene-object setup skipped: {exc}", flush=True)

    def _null_step_action(self) -> dict:
        """Action dict that advances physics without moving the robot.

        Habitat has no "just step" call — every step must carry an action — so a zero base
        velocity is how we let time pass (for physics to settle, or simply to grab a frame).
        """
        return {
            "action": "base_velocity",
            "action_args": {"base_vel": np.zeros(2, dtype=np.float32)},
        }

    # ── Abstract primitive implementations ───────────────────────────────────

    # Set True by PromptEmbodiedAgent after each VLM policy call.
    # Read (and cleared) by _step below to implement step-mode: the human sees the frame the
    # VLM decided on and presses SPACE before the robot acts on it.
    vlm_just_decided: bool = False

    def _step(self) -> dict:
        """Advance one simulation step (null action)."""
        # A finished episode cannot be stepped — Habitat raises. Hand back the last
        # observation so callers still in a control loop unwind cleanly.
        if self._env._episode_over:
            return self._last_obs or {}
        obs = self._env.step(self._null_step_action())
        self._last_obs = obs
        # Pause exactly once per VLM decision, not on every physics step of the action that
        # follows it — hence consuming the flag here.
        should_pause = _STEP and self.vlm_just_decided
        if should_pause:
            self.vlm_just_decided = False
        if _RENDER:
            self._show_frame(obs, pause=should_pause)
        elif self._display:
            self._show_frame(obs)
        # Memory accrues as a side effect of simply existing in the world — no explicit
        # "remember this" tool. Throttled internally.
        self._memorize_live(obs)
        return obs

    def _memorize_live(self, obs: dict, force: bool = False) -> None:
        """Fold the current head-camera frame into FAISS + episodic memory.

        Throttled to one frame every ``_LIVE_INGEST_INTERVAL`` seconds (wall
        clock) so that, as the robot drives and acts, it accumulates a memory of
        what it has seen — in the SAME on-disk format scan_scene produces
        (``color/<idx>.png`` + ``robot_xy/<idx>.txt``), so retrieval is unchanged.
        Call with ``force=True`` to bypass the throttle (e.g. at task start).
        """
        if self.embedding_worker is None or obs is None:
            return
        now = time.time()
        if not force and (now - self._last_ingest_t) < _LIVE_INGEST_INTERVAL:
            return

        # Always the HEAD camera, even when primary_camera is "arm_workspace": memory must
        # be visually consistent with the scan frames it is being indexed alongside, and a
        # gripper close-up is useless as a place to navigate back to.
        rgb = self._scan_rgb(obs)   # head camera regardless of primary_camera
        if rgb is None:
            return
        self._last_ingest_t = now   # only advance the throttle on an actual capture

        # Created on first use — a run with no memory never spins up the worker.
        # Writing into capture_out_dir means live frames land in the SAME dataset the
        # offline scan produced, and the frame counter continues from where the scan left
        # off (see NavCaptureWorker.__init__), so ids can't collide.
        if self._live_navcap is None:
            from sim.capture import NavCaptureWorker
            self._live_navcap = NavCaptureWorker(out_dir=str(self.capture_out_dir))

        pose = self._get_robot_pose()                 # [x_nav, z_nav, heading]
        xy   = np.array([pose[0], pose[1]], dtype=np.float32)
        idx  = self._live_navcap.enqueue(rgb, xy, pose[2])
        # Must match what NavCaptureWorker will write — this path is the join key between
        # the FAISS index and the images on disk.
        frame_path = os.path.join(str(self.capture_out_dir), "color", f"{idx:06d}.png")

        self.embedding_worker.enqueue(rgb=rgb, frame_path=frame_path,
                                      robot_xy=xy, robot_yaw=pose[2])
        if self.episodic_memory is not None:
            try:
                import datetime
                from agent.schemas import (MemoryEntry, MemorySource,
                                           SensorData, EmbeddingRefs)
                self.episodic_memory.add_entry(MemoryEntry(
                    memory_id = f"mem_{idx:06d}",
                    sensor    = SensorData(
                        image_path = frame_path,
                        robot_pose = [pose[0], pose[1], pose[2]],
                        timestamp  = datetime.datetime.now().isoformat(),
                    ),
                    embeddings = EmbeddingRefs(),
                    source     = MemorySource(source_type="live_task",
                                              episode_id=str(self.scene_id or "")),
                ))
            except Exception as exc:
                print(f"[memorize] episodic add failed: {exc}", flush=True)

    def close(self) -> None:
        """Flush the live-memory capture worker so its queued frames reach disk.

        The FAISS index already holds the vectors, but retrieval reads the PNGs
        back from disk, so they must be written before the process exits.
        """
        if self._live_navcap is not None:
            try:
                self._live_navcap.flush()
                self._live_navcap.stop()
            except Exception as exc:
                print(f"[memorize] live capture flush failed: {exc}", flush=True)
            self._live_navcap = None

    def _show_frame(self, obs: dict, pause: bool = False) -> None:
        """Display camera images via pygame: third-person, head.

        Builds a side-by-side canvas of whichever panels are available. Each slot is a tuple
        of candidate observation keys because the key names differ between single-agent and
        multi-agent Habitat configs ("head_rgb" vs "agent_0_head_rgb") — first hit wins.

        Panels are scaled to a common height so np.concatenate can join them horizontally
        (which requires matching row counts). Missing panels are simply skipped, so this
        renders whatever exists rather than requiring a particular sensor set.
        """
        if _RECORD_THIRD_PERSON:
            self._record_third_person_frame(obs)
        if not _RENDER:
            return
        try:
            panel_slots = (
                ("third_person_sensor",),                # what the robot looks like
                ("head_rgb", "agent_0_head_rgb"),        # what the robot sees
            )
            panels = []
            TARGET_H = 512
            from PIL import Image as _PILImage

            for slot in panel_slots:
                rgb = None
                for key in slot:
                    arr = obs.get(key)
                    if arr is None:
                        continue
                    arr = np.asarray(arr)
                    if arr.ndim == 3 and arr.shape[2] >= 3:
                        rgb = arr[..., :3].astype(np.uint8)   # drop alpha if present
                        break
                if rgb is None:
                    continue

                # Uniform height, aspect-preserving width.
                h, w = rgb.shape[:2]
                scale = TARGET_H / h
                new_w = max(1, int(w * scale))
                pil = _PILImage.fromarray(rgb).resize((new_w, TARGET_H))
                panels.append(np.array(pil))

            if not panels:
                return
            canvas = np.concatenate(panels, axis=1) if len(panels) > 1 else panels[0]
            _pg_show(canvas, pause=pause)
        except Exception as e:
            # Display is never worth failing an episode over.
            print(f"[display] {e}")

    def _record_third_person_frame(self, obs: dict) -> None:
        """Dump the third-person camera to a numbered PNG (enabled by env var).

        Independent of the live display: this is how an episode video gets made, so it runs
        even in a headless run. Frames are sequential, so `ffmpeg` can stitch them directly.
        """
        try:
            arr = obs.get("third_person_sensor")
            if arr is None:
                return
            rgb = np.asarray(arr)
            if rgb.ndim != 3 or rgb.shape[2] < 3:
                return
            rgb = rgb[..., :3].astype(np.uint8)
            from PIL import Image as _PILImage

            path = self._third_person_frame_dir / f"{self._third_person_frame_idx:06d}.png"
            _PILImage.fromarray(rgb).save(path)
            self._third_person_frame_idx += 1
        except Exception as e:
            print(f"[third-person-record] {e}", flush=True)

    def _capture_rgb(self, obs: dict) -> Optional[np.ndarray]:
        """
        Return HxWx3 uint8 RGB for the agent's observation. Priority:
          1. head_rgb / agent_0_head_rgb   (forward-facing head camera — the
             navigation/scene view; matches the explored memory frames)
          2. arm_workspace_rgb / articulated_agent_arm_rgb  (arm camera, fallback)
        The arm camera looks down at the gripper workspace, not forward, so it
        must NOT be the primary observation for navigation/scene reasoning.
        For OWMM-baseline parity set primary_camera="arm_workspace".
        """
        # The mode only reorders the preference list — every camera remains a fallback, so a
        # config missing the preferred sensor still yields *an* image rather than None.
        if self._primary_camera == "arm_workspace":
            order = ("arm_workspace_rgb", "articulated_agent_arm_rgb",
                     "head_rgb", "agent_0_head_rgb")
        else:
            order = ("head_rgb", "agent_0_head_rgb",
                     "arm_workspace_rgb", "articulated_agent_arm_rgb")
        for key in order:
            rgb = obs.get(key)
            if rgb is None:
                continue
            arr = np.asarray(rgb)
            if arr.ndim == 3 and arr.shape[2] >= 3:
                return arr[..., :3].astype(np.uint8)

        return None

    def _grasp_state(self) -> Optional[dict]:
        """Ground-truth grasp state from Habitat's magic-grasp manager.

        ``grasp_mgr.snap_idx`` is the rigid-object id currently constrained to
        the gripper (None when the hand is empty). This is the physics truth of
        whether an object is held, so the policy never has to infer it visually.
        """
        try:
            gm = self._sim.grasp_mgr
        except Exception:
            # No grasp manager at all → we genuinely cannot tell. None (not False) so the
            # policy falls back to visual judgment instead of being told "empty hand".
            return None
        idx = getattr(gm, "snap_idx", None)
        if idx is None:
            return {"grasped": False, "object": None}
        # Holding something. Resolving its handle is a nicety — if that lookup fails we still
        # report grasped=True, because *that* is the fact the policy needs.
        name = None
        try:
            obj = self._sim.get_rigid_object_manager().get_object_by_id(idx)
            name = obj.handle if obj is not None else None
        except Exception:
            pass
        return {"grasped": True, "object": name}

    def _get_robot_pose(self) -> list[float]:
        """
        Return [x_nav, z_nav, heading_rad] from localization_sensor.

        localization_sensor outputs [world.X, world.Y, world.Z, heading].
        Navigation plane is XZ, so nav coords are (world.X, world.Z).

        Note the index skip — [0], [2], [3], NOT [0], [1], [2]. Element 1 is world Y (the
        robot's height), which is not part of a planar pose. This is the coordinate remap
        described in the module docstring, and dropping the wrong axis here is a bug that
        manifests much later as navigation goals at nonsensical positions.
        """
        obs = self._last_obs or {}
        loc = obs.get("localization_sensor")
        if loc is None:
            # No localization sensor configured — derive the pose from the robot directly.
            try:
                robot = self._robot()
                pos = np.array(robot.base_pos)
                # Heading: transform the robot's local +X (forward) into world space, then
                # take its angle in the XZ plane. atan2(z, x) — again, Y is skipped.
                fwd = np.array([1.0, 0.0, 0.0])
                T = robot.base_transformation
                heading = math.atan2(
                    float(T.transform_vector(fwd)[2]),
                    float(T.transform_vector(fwd)[0]),
                )
                return [float(pos[0]), float(pos[2]), float(heading)]
            except Exception:
                return [0.0, 0.0, 0.0]   # last resort; better than raising mid-episode
        loc = np.asarray(loc).flatten()
        return [float(loc[0]), float(loc[2]), float(loc[3])]

    def _navigate_step(self, bearing: float) -> None:
        """
        Send one base_velocity command.  Used by BaseToolbox._align_yaw()
        for final orientation alignment.

        Turn-then-drive, exactly as in the ManiSkill backend: a bearing beyond _ROT_THRESH
        produces a pure rotation, anything smaller drives straight. _align_yaw relies on this
        by clamping its bearing above the threshold so it only ever rotates.
        """
        if self._env._episode_over:
            return
        # base_vel is [forward, angular]. copysign preserves the turn direction while
        # applying the fixed turn rate.
        if abs(bearing) > _ROT_THRESH:
            vel = np.array([0.0, math.copysign(_TURN_VEL, bearing)], dtype=np.float32)
        else:
            vel = np.array([_FWD_VEL, 0.0], dtype=np.float32)
        obs = self._env.step({
            "action": "base_velocity",
            "action_args": {"base_vel": vel},
        })
        self._last_obs = obs
        if self._display:
            self._show_frame(obs)

    def _base_move_step(self, motion: str) -> None:
        """One tick of a discrete base motion, routed through whichever mechanism can do it.

        Habitat's base_velocity action is configurable and often CANNOT do what the agent's
        motion vocabulary asks for:
          * a 2-D action space ([forward, angular]) has no lateral channel, so "left"/"right"
            (strafing) are unexpressible;
          * many configs set _allow_back=False, so "backward" is silently clamped to zero.

        Rather than degrade the agent's action set to the lowest common denominator, we fall
        back to _direct_planar_step (which moves the robot kinematically) for exactly those
        motions the controller can't express. The agent therefore always has the same six
        motions available, regardless of the Habitat config it is running against.
        """
        if self._env._episode_over:
            return

        has_lateral = self._base_velocity_has_lateral()
        if motion in ("left", "right") and not has_lateral:
            self._direct_planar_step(motion)
            return
        if motion == "backward" and not self._base_velocity_allows_backward():
            self._direct_planar_step(motion)
            return

        # 3-D action space: [forward, lateral, angular].
        if has_lateral:
            if motion == "forward":
                vel = np.array([_FWD_VEL, 0.0, 0.0], dtype=np.float32)
            elif motion == "backward":
                vel = np.array([-_FWD_VEL, 0.0, 0.0], dtype=np.float32)
            elif motion == "left":
                vel = np.array([0.0, _FWD_VEL, 0.0], dtype=np.float32)
            elif motion == "right":
                vel = np.array([0.0, -_FWD_VEL, 0.0], dtype=np.float32)
            elif motion == "rotate 30 degrees":
                vel = np.array([0.0, 0.0, _TURN_VEL], dtype=np.float32)
            elif motion == "rotate -30 degrees":
                vel = np.array([0.0, 0.0, -_TURN_VEL], dtype=np.float32)
            else:
                return
        else:
            # 2-D action space: [forward, angular]. left/right never reach here (they were
            # routed to _direct_planar_step above).
            if motion == "forward":
                vel = np.array([_FWD_VEL, 0.0], dtype=np.float32)
            elif motion == "backward":
                vel = np.array([-_FWD_VEL, 0.0], dtype=np.float32)
            elif motion == "rotate 30 degrees":
                vel = np.array([0.0, _TURN_VEL], dtype=np.float32)
            elif motion == "rotate -30 degrees":
                vel = np.array([0.0, -_TURN_VEL], dtype=np.float32)
            else:
                return

        obs = self._env.step({
            "action": "base_velocity",
            "action_args": {"base_vel": vel},
        })
        self._last_obs = obs
        if self._display:
            self._show_frame(obs)

    def _base_velocity_has_lateral(self) -> bool:
        """Does base_velocity accept a strafe channel? (action-space width >= 3)

        Introspects the gym space rather than the config, so it reflects what the action will
        actually accept. Conservative on failure: assume 2-D, which routes strafes through
        the kinematic fallback — always correct, just less physical.
        """
        try:
            action = self._env._task.actions.get("base_velocity")
            space = getattr(action, "action_space", None)
            spaces = getattr(space, "spaces", {})
            for subspace in spaces.values():
                shape = getattr(subspace, "shape", ())
                if shape and int(shape[0]) >= 3:
                    return True
        except Exception:
            pass
        return False

    def _base_velocity_allows_backward(self) -> bool:
        """Does the controller permit negative forward velocity?

        Many OVMM configs set _allow_back=False (the policy they were built for was
        forward-only), which silently CLAMPS a reverse command to zero rather than erroring —
        so without this check "backward" would appear to run and do nothing.

        Defaults to True when the attribute is absent, matching Habitat's own default.
        """
        try:
            action = self._env._task.actions.get("base_velocity")
            return bool(getattr(action, "_allow_back", True))
        except Exception:
            return True

    def _direct_planar_step(self, motion: str) -> None:
        try:
            import magnum as mn

            robot = self._robot()
            if motion == "forward":
                local = mn.Vector3(0.015, 0.0, 0.0)
            elif motion == "backward":
                local = mn.Vector3(-0.015, 0.0, 0.0)
            elif motion == "left":
                # Fetch's base_transformation has local X/Y in the ground
                # plane; local Z is vertical after the robot-frame rotation.
                local = mn.Vector3(0.0, 0.015, 0.0)
            elif motion == "right":
                local = mn.Vector3(0.0, -0.015, 0.0)
            else:
                return
            delta = robot.base_transformation.transform_vector(
                local)
            start = robot.base_pos
            end = start + delta
            try:
                end = self._sim.pathfinder.try_step(start, end)
            except Exception:
                pass
            robot.base_pos = end
            obs = self._env.step(self._null_step_action())
            self._last_obs = obs
            if self._display:
                self._show_frame(obs)
        except Exception as exc:
            print(f"[HabitatToolbox] direct base_move failed: {exc}", flush=True)

    def _plan_path(
        self, start_xy: np.ndarray, goal_xy: np.ndarray
    ) -> list[np.ndarray]:
        """
        Stub — oracle_nav_coord_action handles path planning internally.
        Returns a direct two-point path so BaseToolbox._align_yaw works.

        Unlike the ManiSkill backend (which plans with NavGrid), this backend overrides
        navigate() entirely and hands the goal to Habitat's oracle navigator, so no path is
        ever needed. The two-point stub exists only to satisfy the base class's signature.
        """
        return [start_xy.copy(), goal_xy.copy()]

    def _locate_target_object(self, target: str):
        """Aiming: localize `target` in the head image and return the matching
        scene-object id + its world position + pixel (u,v). Returns
        (obj_id, obj_world_xyz, (u,v)) or (None, None, None).

        This is the PERCEPTION-ONLY path to a grasp target, and it is the heart of what makes
        grasping here honest rather than an oracle:

            name → detector → pixel → ray → the object that pixel actually sees

        The final ray-cast step is what matters. Rather than trusting the depth estimate and
        then guessing which nearby object it must have meant, we ask the simulator "what
        surface does this pixel see?" and take that. If the answer is not one of the task's
        pickable objects — it's the floor, a wall, the robot's own arm — aiming FAILS. There
        is deliberately no "nearest object" fallback that would quietly rescue a bad aim, so
        the agent cannot grasp something it did not actually look at.
        """
        obs = self._last_obs
        if obs is None:
            return None, None, None
        # Sensor key differs across single/multi-agent configs.
        depth = None
        for dk in ("head_depth", "agent_0_head_depth"):
            if dk in obs:
                depth = np.asarray(obs[dk]).squeeze().astype(np.float32); break
        T = self._head_depth_cam_T()
        if depth is None or depth.ndim != 2 or T is None:
            return None, None, None
        H, W = depth.shape

        from PIL import Image as _PIL
        rgb_full = None
        if self._last_image_path and os.path.exists(self._last_image_path):
            try:
                rgb_full = np.array(_PIL.open(self._last_image_path).convert("RGB"))
            except Exception:
                rgb_full = None

        # ── Localise the target: gather ALL candidate boxes, aim at the first
        # whose ray lands on a pickable scene object. ─────────────────────────
        # An open-vocab detector routinely returns several boxes for one query, and
        # the HIGHEST-confidence one is often a false positive (e.g. a "toy vehicle"
        # query scoring a rug pattern above the actual toy). Trusting dets[0] blindly
        # then rays into empty floor and the grasp fails even though the real object
        # WAS detected at a lower rank. So we try every candidate and keep the first
        # that raycasts onto one of the episode's pickable objects — still no fuzzy
        # "nearest object" rescue, just "don't discard a correct lower-ranked box".
        candidates: list[tuple[list, str]] = []
        if self._grounding_dino is not None and rgb_full is not None:
            try:
                dets = self._grounding_dino.detect(rgb_full, target)
                candidates += [(det["bbox"], "gdino") for det in (dets or [])]
            except Exception as _e:
                print(f"[HabitatToolbox] GDino failed: {_e}")
        # Only pay for the VLM inspect call when the detector produced nothing.
        if not candidates:
            ir = self.inspect(image_path=self._last_image_path or "",
                              question=f"Locate the {target}. Return its pixel bounding box.")
            if ir.ok:
                candidates += [(b, "inspect") for b in ir.data.get("candidate_bboxes", [])]
        if not candidates:
            return None, None, None

        rom = self._sim.get_rigid_object_manager()
        fallback = None  # (u, v, hit_world) of the first tried point, for diagnostics
        for bbox, src in candidates:
            if not bbox or len(bbox) < 4:
                continue
            x1, y1, x2, y2 = [int(vv) for vv in bbox[:4]]
            x1 = max(0, min(W - 1, x1)); y1 = max(0, min(H - 1, y1))
            x2 = max(x1 + 1, min(W, x2)); y2 = max(y1 + 1, min(H, y2))
            cu, cv = (x1 + x2) // 2, (y1 + y2) // 2

            # Depth from the INNER 60% of the box (inset a fifth each side): border
            # pixels straddle the silhouette and read the wall behind. `or x2` guards
            # a tiny box that insets to an empty slice.
            mx, my = (x2 - x1) // 5, (y2 - y1) // 5
            patch = depth[y1 + my:y2 - my or y2, x1 + mx:x2 - mx or x2]
            valid = patch[(patch > 0.05) & (patch < 10.0)]
            d = float(np.median(valid)) if valid.size else float(depth[cv, cu])

            # Ray-cast a GRID of points inside the box, not just the centre. A small
            # object often does not fill its box, so the centre pixel can land on the
            # stool/floor behind it; sweeping a few interior points finds the one that
            # actually sees the object. Points are ordered centre-out so we prefer the
            # middle when several hit. Still strict: accept only a pickable scene object.
            us = sorted({int(round(x1 + (x2 - x1) * fx)) for fx in (0.5, 0.35, 0.65, 0.5, 0.5)},
                        key=lambda x: abs(x - cu))
            vs = sorted({int(round(y1 + (y2 - y1) * fy)) for fy in (0.5, 0.35, 0.65, 0.5, 0.5)},
                        key=lambda y: abs(y - cv))
            hit_id = None; hit_uv = (cu, cv)
            tried = 0
            for pv in vs:
                for pu in us:
                    tried += 1
                    oid, hw = self._raycast_pixel_to_object(pu, pv, (W, H), T)
                    if fallback is None:
                        fallback = (pu, pv, hw)
                    if oid is not None:
                        hit_id, hit_uv = oid, (pu, pv)
                        break
                if hit_id is not None:
                    break
            if os.environ.get("GRASP_DEBUG"):
                print(f"[GRASP_DEBUG] target={target!r} src={src} bbox={bbox[:4]} "
                      f"depth={d:.2f}m tried={tried}pts ray_obj_id={hit_id} "
                      f"hit_uv={hit_uv} hit={hit_id is not None} "
                      f"n_scene_obj_ids={len(getattr(self._sim, 'scene_obj_ids', []))}",
                      flush=True)
            if hit_id is not None:
                u, v = hit_uv
                self._save_localization_crop(rgb_full, (x1, y1, x2, y2), target, src, d)
                return hit_id, np.array(rom.get_object_by_id(hit_id).translation), (u, v)

        # No candidate box landed on a pickable object.
        u, v, hit_world = fallback if fallback else (None, None, None)
        return None, hit_world, (u, v)

    def _raycast_pixel_to_object(
        self, u: int, v: int, wh: tuple[int, int], T: np.ndarray
    ):
        """Cast a ray from the head camera through pixel ``(u, v)`` and return
        ``(obj_id, world_point)`` for the EXACT pickable scene object the pixel
        lands on. Returns ``(None, world_point_or_None)`` when the closest
        surface along the ray is not one of ``scene_obj_ids`` (e.g. floor,
        furniture, robot) — i.e. the pixel is not on a graspable object.

        ``T`` is the camera-to-world 4×4 from ``_head_depth_cam_T``; ``wh`` is
        the (width, height) of the head image the pixel indexes into.
        """
        try:
            import habitat_sim
            import magnum as mn
        except Exception:
            return None, None
        if T is None:
            return None, None
        W, H = wh
        hfov  = math.pi / 2.0
        f_inv = math.tan(hfov / 2.0)
        # Pixel → normalised camera-space direction (Habitat: forward = -Z_cam).
        # ys is flipped because image rows increase downward while camera Y increases up.
        xs = 2.0 * u / (W - 1) - 1.0
        ys = 1.0 - 2.0 * v / (H - 1)
        # Trailing 0.0 (not 1.0): this is a DIRECTION, not a point, so the transform must
        # rotate it without applying the camera's translation.
        dir_world = T @ np.array([xs * f_inv, ys * f_inv, -1.0, 0.0])
        dirw = np.array(dir_world[:3], dtype=np.float64)
        n = float(np.linalg.norm(dirw))
        if n < 1e-9:
            return None, None
        dirw /= n
        # The ray starts at the camera, which is T's translation column.
        origin = np.array([T[0, 3], T[1, 3], T[2, 3]], dtype=np.float64)

        ray = habitat_sim.geo.Ray(mn.Vector3(*origin), mn.Vector3(*dirw))
        res = self._sim.cast_ray(ray)
        if not res.has_hits:
            return None, None
        hit = res.hits[0]                       # closest surface along the ray
        world_pt = np.array([hit.point[0], hit.point[1], hit.point[2]])
        # The membership test that makes this strict: only the episode's own pickable
        # objects count. A hit on the floor, a wall, a table, or the robot returns None for
        # the id — the caller reads that as "the pixel is not on something graspable" and
        # refuses the grasp. The world point is still returned, since it is useful for
        # diagnostics even when the aim failed.
        scene_ids = {int(i) for i in getattr(self._sim, "scene_obj_ids", [])}
        if int(hit.object_id) in scene_ids:
            return int(hit.object_id), world_pt

        # The ray usually lands on the RECEPTACLE (stool/table/floor) a small object
        # sits on, not the object's own tiny collision mesh — habitat's cast_ray
        # returns the frontmost surface, and a toy a few cm tall barely occludes the
        # surface behind it. So if a PICKABLE scene object sits right at the hit
        # point, that is what the pixel is really aiming at. Tight radius keeps this
        # honest: it rescues "ray grazed the shelf under the mug", not "nearest object
        # anywhere". Objects resting ON the surface sit just above it, hence the
        # slightly larger vertical tolerance.
        rom = self._sim.get_rigid_object_manager()
        best_id, best_d = None, _GRASP_SNAP_RADIUS
        for sid in scene_ids:
            try:
                op = np.array(rom.get_object_by_id(sid).translation)
            except Exception:
                continue
            dxz = math.hypot(op[0] - world_pt[0], op[2] - world_pt[2])
            dy = abs(op[1] - world_pt[1])
            if dxz <= _GRASP_SNAP_RADIUS and dy <= _GRASP_SNAP_RADIUS_Y and dxz < best_d:
                best_id, best_d = int(sid), dxz
        if os.environ.get("GRASP_DEBUG"):
            print(f"[RAY_DEBUG] px=({u},{v}) raw_hit_id={int(hit.object_id)} "
                  f"hit_pt=({world_pt[0]:.2f},{world_pt[1]:.2f},{world_pt[2]:.2f}) "
                  f"snapped_scene_obj={best_id} (d={best_d:.2f}m)", flush=True)
        if best_id is not None:
            return best_id, world_pt
        return None, world_pt

    def _locate_place_pixel(self, destination: str):
        """Return a visible placement pixel and approximate world point.

        The receptacle bbox center often lies on the front face of a cabinet, so
        this aims slightly into the upper portion of the box, which is usually
        closer to the visible support surface.
        """
        obs = self._last_obs
        if obs is None:
            return None, None, "no_obs"

        depth = None
        for dk in ("head_depth", "agent_0_head_depth"):
            if dk in obs:
                depth = np.asarray(obs[dk]).squeeze().astype(np.float32)
                break
        T = self._head_depth_cam_T()
        if depth is None or depth.ndim != 2 or T is None:
            return None, None, "no_depth"
        H, W = depth.shape

        from PIL import Image as _PIL
        rgb_full = None
        if self._last_image_path and os.path.exists(self._last_image_path):
            try:
                rgb_full = np.array(_PIL.open(self._last_image_path).convert("RGB"))
            except Exception:
                rgb_full = None

        bbox, src = None, "none"
        if self._grounding_dino is not None and rgb_full is not None:
            try:
                dets = self._grounding_dino.detect(rgb_full, destination)
                if dets:
                    bbox = dets[0]["bbox"]
                    src = f"gdino(score={dets[0].get('score')})"
            except Exception as _e:
                print(f"[HabitatToolbox] GDino failed: {_e}")

        if bbox is None:
            ir = self.inspect(
                image_path=self._last_image_path or "",
                question=(f"Locate a clear placement point on top of the "
                          f"{destination}. Return its pixel bounding box."),
            )
            if ir.ok:
                candidates = ir.data.get("candidate_bboxes", [])
                if candidates:
                    bbox = candidates[0]
                    src = "inspect_top"

        if not bbox or len(bbox) < 4:
            return None, None, "no_bbox"

        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        x1 = max(0, min(W - 1, x1)); y1 = max(0, min(H - 1, y1))
        x2 = max(x1 + 1, min(W, x2)); y2 = max(y1 + 1, min(H, y2))
        u = (x1 + x2) // 2
        # NOT the box centre vertically — 22% down from the top edge. See the docstring: a
        # detector box around a "counter" or "cabinet" is dominated by its front FACE, whose
        # centre is a vertical surface you cannot put anything on. Aiming high in the box
        # biases toward the top edge, which is where the horizontal support surface is.
        v = y1 + max(1, int(0.22 * (y2 - y1)))

        # Small local window around the aim point (rather than the whole box, as in
        # _locate_target_object) — we want the depth AT the chosen placement point, not an
        # average over the whole receptacle, which spans a wide depth range.
        r = 5
        patch = depth[max(0, v - r):min(H, v + r + 1),
                      max(0, u - r):min(W, u + r + 1)]
        valid = patch[(patch > 0.05) & (patch < 10.0)]
        d = float(np.median(valid)) if valid.size else float(depth[v, u])
        self._save_localization_crop(rgb_full, (x1, y1, x2, y2),
                                     destination, src, d)
        if d < 0.05 or d > 10.0:
            # Implausible depth. Return the pixel anyway (the arm action can still aim at
            # it) but no world point, and say why in the source tag.
            return (float(u), float(v)), None, f"{src}:invalid_depth"

        # Un-project (u, v, d) to a world point. Trailing 1.0 → this IS a point, so the
        # camera translation applies (contrast the direction vector in _raycast_pixel_to_object).
        f_inv = math.tan((math.pi / 2) / 2)
        xs = 2.0 * u / (W - 1) - 1.0
        ys = 1.0 - 2.0 * v / (H - 1)
        world = T @ np.array([xs * d * f_inv, ys * d * f_inv, -d, 1.0])
        return (float(u), float(v)), np.array(world[:3]), src

    def _gripper_contact_obj(self, obj_id: int) -> bool:
        """True iff a gripper link is in physics contact with object ``obj_id``.

        Mirrors the contact test in the benchmark's SuctionGraspAction: scan the
        current physics contact points for one between the robot's gripper links
        and the target object.
        """
        try:
            from habitat.tasks.rearrange.utils import (
                coll_name_matches, coll_link_name_matches)
            robot = self._robot()
            robot_id = robot.sim_obj.object_id
            hand_links = list(robot.params.gripper_joints) + list(
                getattr(robot.params, "ee_links", []) or [])

            # Refresh contacts at the CURRENT pose (the benchmark does not
            # auto-step physics, so the cached contacts can be stale/empty).
            try:
                self._sim.perform_discrete_collision_detection()
            except Exception:
                pass
            contacts = self._sim.get_physics_contact_points()

            # Prefer a hand/gripper link touching the target object.
            for c in contacts:
                if (coll_name_matches(c, robot_id)
                        and coll_name_matches(c, obj_id)
                        and any(coll_link_name_matches(c, l) for l in hand_links)):
                    return True
            # Fallback: ANY robot link touching the target. gripper_joints are
            # joint indices and may not equal the contact link ids, so don't let
            # a numbering mismatch hide a real touch — during a grasp reach the
            # part contacting the target is the hand anyway.
            for c in contacts:
                if coll_name_matches(c, robot_id) and coll_name_matches(c, obj_id):
                    return True
            return False
        except Exception as e:
            print(f"[suction] contact check failed: {e}", flush=True)
            return False

    def _suction_snap(self, obj_id: int) -> bool:
        """Contact-gated grasp: grab ``obj_id`` ONLY once the gripper physically
        touches it (the realistic trigger), but SEAT it with force=True so it is
        placed at a canonical in-hand pose and held firmly.

        We don't preserve the contact pose (force=False): position-only IK can't
        aim the jaws, so the gripper often touches with its outer/back surface —
        constraining the object there gives a bad, unstable grasp. force=True
        repositions the object into the grasp frame (0.1 m in front of the EE
        link, like the benchmark MagicGrasp) so which surface touched no longer
        matters, and the object can't drift out and trip
        constraint_violation_drops_object during the retract. Returns True once
        the object is held.
        """
        if not self._gripper_contact_obj(obj_id):
            return False
        try:
            import magnum as mn
            keep_T = mn.Matrix4.translation(mn.Vector3(0.1, 0.0, 0.0))
            self._sim.grasp_mgr.snap_to_obj(
                int(obj_id),
                force=True,                        # seat the object in the hand
                rel_pos=mn.Vector3(0.1, 0.0, 0.0),
                keep_T=keep_T,
                should_open_gripper=False,
            )
            # Kill any momentum the approach imparted so the seated object stays
            # put rather than fighting the constraint on the next physics step.
            ro = self._sim.get_rigid_object_manager().get_object_by_id(obj_id)
            if ro is not None:
                ro.linear_velocity  = mn.Vector3.zero_init()
                ro.angular_velocity = mn.Vector3.zero_init()
                ro.awake = True
            return self._sim.grasp_mgr.snap_idx is not None
        except Exception as e:
            print(f"[suction] snap failed: {e}", flush=True)
            return False

    def _grasp(self, target: str) -> tuple[bool, str, float]:
        """
        OWMM-style pick: (1) AIM at the requested `target` object (the EXACT
        object the detected pixel lands on), (2) drive the arm toward it with IK
        (real arm motion via arm_pick_action / PixelArmAction), (3) grasp.

        Grasp model (``GRASP_MODE``):
          * "suction" (default) — grab only once the gripper is in physics
            CONTACT with the target, seating it in the hand (force=True). The
            arm keeps pressing/closing toward the object until contact or
            _ARM_MAX_STEPS.
          * "magic" — forced snap once the EE is within _GRASP_REACH_THRESH.

        Return contract — the first element means "a target was localized and a
        grasp was ATTEMPTED", NOT "the grasp succeeded". The tool never decides
        task success; whether the object is actually held is left for the verify
        step to judge from the resulting observation. So:
          * (False, target, inf)  — no graspable object under the aimed pixel.
          * (True,  name,  dist)  — localized; arm drove in and grasped if able.
                                    No "move closer" guidance is emitted.
        """
        sim, robot = self._sim, self._robot()
        rom = sim.get_rigid_object_manager()

        # ── AIM. Perception only — see _locate_target_object. ─────────────────
        obj_id, _obj_pos, pixel = self._locate_target_object(target)
        if obj_id is None:
            # Pixel is not on any pickable object → genuine inability to act.
            return False, target or "", float("inf")

        name = rom.get_object_by_id(obj_id).handle
        px, py = float(pixel[0]), float(pixel[1])
        suction = _GRASP_MODE == "suction"

        # Spread the fingers so the gripper can clear the approach; they are
        # closed onto the object once near (see _GRASP_CLOSE_DIST) — an open
        # gripper straddles small objects and never registers contact.
        if suction:
            try:
                robot.open_gripper()
            except Exception:
                pass

        def _ee_to_obj() -> float:
            """Live end-effector → object distance. Re-read every step: BOTH may move (the
            arm drives in, and the object can be nudged by contact)."""
            ee = np.array(robot.ee_transform().translation)
            return float(np.linalg.norm(np.array(rom.get_object_by_id(obj_id).translation) - ee))

        d = _ee_to_obj()
        grasped = False
        closing = False   # latches once the fingers have been commanded shut
        # ── ARM MOTION: drive the arm with IK toward the target's pixel ───────
        # via the registered arm_pick_action (PixelArmAction). Env.step works
        # for arm actions now that EmbodiedTask.step renders sensors for non-dict
        # action returns (the fork's ArmAction.step returns its ee_target).
        for step in range(_ARM_MAX_STEPS):
            if self._env._episode_over:
                break
            # magic: stop once within reach. suction: keep pressing to contact.
            # (In suction mode the loop only exits via a successful snap or the step cap —
            # proximity alone is not enough, physical contact is required.)
            if not suction and d < _GRASP_REACH_THRESH:
                break
            # The arm action is PIXEL-targeted, not coordinate-targeted: we hand it (px, py)
            # and it runs IK toward whatever that pixel sees. So the arm chases the same
            # perceptual target the detector found, rather than a ground-truth position.
            obs = self._env.step({
                "action": "arm_pick_action",
                "action_args": {
                    "arm_pick_action":  np.array([px, py, 1.0], dtype=np.float32),
                    "grip_pick_action": np.array([0.0], dtype=np.float32),  # snap manually
                                        # 0 = don't let the action auto-grasp; we control
                                        # exactly when the snap fires (below).
                },
            })
            self._last_obs = obs
            # Throttled: this loop can run 90 steps and the display queue would choke.
            if (_RENDER or self._display) and step % _RENDER_EVERY == 0:
                self._show_frame(obs)
            d = _ee_to_obj()
            if suction:
                # Once the hand is on the object, close the fingers so they
                # physically engage it (and keep pressing toward the pixel).
                # `closing` latches — issue the close once, then keep driving in; re-issuing
                # every step would fight the motor controller.
                if not closing and d < _GRASP_CLOSE_DIST:
                    try:
                        robot.close_gripper()
                    except Exception:
                        pass
                    closing = True
                # Poll for contact every step. Returns False until the fingers actually
                # touch, at which point it snaps and we're done.
                if self._suction_snap(obj_id):
                    grasped = True
                    break

        # ── SNAP. suction: handled in-loop on contact (force=False). magic:
        # forced snap once the arm physically reached. If neither grasps, we
        # leave the scene as-is and report the attempt neutrally — no advice, no
        # success/failure verdict; the verify step decides from the observation.
        if grasped:
            self._last_obs = self._env.step(self._null_step_action())  # settle
        elif not suction and d < _GRASP_REACH_THRESH:
            # Magic mode: proximity is sufficient, so snap now that the arm has arrived.
            try:
                sim.grasp_mgr.snap_to_obj(obj_id, force=True)
                self._last_obs = self._env.step(self._null_step_action())  # settle
            except Exception as e:
                print(f"[HabitatToolbox._grasp] snap_to_obj failed: {e}", flush=True)
        # Note: falling through here (neither branch taken) is NOT an error — it means the
        # arm reached for a real object and failed to get hold of it. We still return True,
        # because True means "attempted", and BaseToolbox.manipulate will consult
        # _grasp_state() for the physics truth of whether anything is actually held.
        return True, name, d

    def _retract_arm(self, settle_steps: int = 12, move_steps: int = 60) -> None:
        """After a manipulation has come to rest, pull the arm PARTWAY back
        toward its rest pose — enough to clear the workspace for navigation, but
        not a full tuck (see ``_ARM_RETRACT_FRAC``).

        A grasped object is carried along by the snap constraint, so the arm is
        moved GRADUALLY: the motor target is ramped from the start pose to the
        retract pose over ``move_steps`` so the held object tracks the hand. A
        single fast jump would let the object lag, violate the grasp constraint,
        and (with constraint_violation_drops_object) be dropped on the floor.

        The arm first holds its current pose for ``settle_steps`` so the
        manipulation (grasp constraint settling / dropped object coming to rest)
        FINISHES before any retract motion begins.
        """
        try:
            robot = self._robot()
        except Exception as e:
            print(f"[retract] no robot: {e}", flush=True)
            return

        # 1. Let the manipulation finish settling before moving anything.
        for _ in range(int(settle_steps)):
            if self._env._episode_over:
                break
            self._settle_release_physics(1)

        # 2. Target = a fraction of the way from the current pose to rest.
        try:
            start = np.asarray(robot.arm_joint_pos,         dtype=np.float32)
            rest  = np.asarray(robot.params.arm_init_params, dtype=np.float32)
            final = start + _ARM_RETRACT_FRAC * (rest - start)
        except Exception as e:
            print(f"[retract] could not compute retract pose: {e}", flush=True)
            return

        # 3. Ramp the motor target there gradually so a grasped object tracks
        #    the hand instead of being flung out of the constraint.
        #    alpha goes 1/n … 1, i.e. a linear interpolation from `start` to `final`.
        n = max(1, int(move_steps))
        for step in range(n):
            if self._env._episode_over:
                break
            alpha = (step + 1) / n
            try:
                robot.arm_motor_pos = start + alpha * (final - start)
            except Exception:
                break
            # Tick physics so the position motors actually drive to the target
            # (the benchmark config does not auto-step physics).
            self._settle_release_physics(1)
            if (_RENDER or self._display) and step % _RENDER_EVERY == 0:
                self._last_obs = self._env.step(self._null_step_action())
                self._show_frame(self._last_obs)
        if not self._env._episode_over:
            self._last_obs = self._env.step(self._null_step_action())  # refresh obs

    def _post_manipulate(self) -> None:
        """Always retract the arm after a manipulate attempt (grasp or place)."""
        self._retract_arm()

    def _release(
        self,
        target: str = "",
        destination: Optional[str] = None,
        target_region: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Place/drop held object, then verify Habitat cleared the grasp.

        Three phases:
          1. If a destination was named (and the env supports pixel-targeted placing), drive
             the arm over the receptacle — same perception path as grasping, via
             _locate_place_pixel.
          2. Break the grasp constraint (desnap), first making sure the object will actually
             fall (see _enable_released_object_physics).
          3. Step physics so it lands and settles before the agent looks again.

        Phase 1 is skipped entirely for a bare "drop", which just opens the hand where it is.
        """
        grasp_mgr = self._sim.grasp_mgr
        held_id = grasp_mgr.snap_idx
        if held_id is None:
            return True, "nothing was grasped."

        dest = target_region or destination
        place_pixel = None
        place_world = None
        place_src = "none"
        release_force = True

        # ── Phase 1: move the arm over the destination ────────────────────────
        if dest and "arm_place_action" in self._env._task.actions:
            place_pixel, place_world, place_src = self._locate_place_pixel(dest)
            if place_pixel is not None:
                px, py = place_pixel
                # Drive toward the placement pixel until the EE is close to the estimated
                # world point (or we run out of steps). Without a world point we cannot
                # measure arrival, so the loop just runs its full budget.
                for step in range(_PLACE_MAX_STEPS):
                    if self._env._episode_over:
                        break
                    obs = self._env.step({
                        "action": "arm_place_action",
                        "action_args": {
                            "arm_place_action": np.array([px, py, 1.0], dtype=np.float32),
                            "grip_place_action": np.array([0.0], dtype=np.float32),
                        },
                    })
                    self._last_obs = obs
                    if (_RENDER or self._display) and step % _RENDER_EVERY == 0:
                        self._show_frame(obs)
                    if place_world is not None:
                        ee = np.array(self._robot().ee_transform().translation)
                        if float(np.linalg.norm(ee - place_world)) < _PLACE_REACH_THRESH:
                            break

                # Signal release through the action itself (grip = -1) for a few steps, so
                # the env's own place logic gets a chance to run. The explicit desnap below
                # is the backstop for when it doesn't.
                for _ in range(3):
                    if self._env._episode_over:
                        break
                    obs = self._env.step({
                        "action": "arm_place_action",
                        "action_args": {
                            "arm_place_action": np.array([px, py, 0.0], dtype=np.float32),
                            "grip_place_action": np.array([-1.0], dtype=np.float32),
                        },
                    })
                    self._last_obs = obs

        # ── Phase 2: force the constraint open if it is still holding ─────────
        if grasp_mgr.snap_idx is not None:
            # Order matters: make the object dynamic BEFORE desnapping, or it is briefly
            # released while still kinematic and freezes in mid-air.
            # release_force = "let Habitat handle the drop" — only needed when we could not
            # set up the physics ourselves.
            release_force = not self._enable_released_object_physics(held_id)
            try:
                # Clear the magic-grasp constraint. We then tick physics
                # explicitly because fetch_vlm disables automatic physics
                # stepping for benchmark speed.
                grasp_mgr.desnap(release_force)
            except Exception as e:
                print(f"[HabitatToolbox._release] desnap failed: {e}")
            # Retry once. desnap occasionally no-ops when the constraint is in an odd state,
            # and an object welded to the hand would silently break every later step.
            if grasp_mgr.snap_idx is not None:
                try:
                    grasp_mgr.desnap(release_force)
                except Exception as e:
                    print(f"[HabitatToolbox._release] forced desnap failed: {e}")

        # ── Phase 3: let it fall ──────────────────────────────────────────────
        released = grasp_mgr.snap_idx is None
        if released:
            self._settle_release_physics(10)

        # Arm retraction is handled uniformly by _post_manipulate() after this
        # returns, so the object is left to settle in place here.
        for _ in range(_PLACE_SETTLE_STEPS):
            if self._env._episode_over:
                break
            self._settle_release_physics(2)
            self._last_obs = self._env.step(self._null_step_action())

        if not released:
            return False, f"release failed; still grasping object id {grasp_mgr.snap_idx}."
        if dest and place_pixel is None:
            return True, f"released object, but could not localize placement pixel for {dest}."
        if dest:
            return True, f"released object at {dest} using {place_src} placement pixel; physics settled."
        return True, "released object."

    def _enable_released_object_physics(self, obj_id: int) -> bool:
        """Make a formerly magic-grasped OVMM object respond to gravity.

        Some benchmark configs load scene objects as kinematic/non-collidable
        for speed. A plain desnap clears the grasp state but leaves such objects
        frozen in mid-air. Convert the released object to dynamic, and keep it
        temporarily in the held-object collision group so it can drop without
        colliding with overlapping gripper links.
        """
        try:
            import habitat_sim
            import magnum as mn
            from habitat_sim.physics import CollisionGroups

            obj = self._sim.get_rigid_object_manager().get_object_by_id(obj_id)
            if obj is None:
                return False
            obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
            obj.collidable = True
            obj.linear_velocity = mn.Vector3.zero_init()
            obj.angular_velocity = mn.Vector3.zero_init()
            obj.awake = True   # a sleeping body ignores gravity until something wakes it
            try:
                # Park it in the held-object collision group for the moment of release. At
                # the instant of desnap the object is still interpenetrating the gripper
                # links; in the normal group the solver would resolve that overlap by
                # violently ejecting it. UserGroup7 doesn't collide with the robot, so it
                # falls cleanly out of the hand instead of being shot across the room.
                obj.override_collision_group(CollisionGroups.UserGroup7)
            except Exception:
                pass
            return True
        except Exception as exc:
            print(f"[HabitatToolbox._release] dynamic release setup failed: {exc}",
                  flush=True)
            return False

    def _settle_release_physics(self, steps: int = 1) -> None:
        """Tick Habitat physics explicitly after desnap/retraction.

        The benchmark config disables automatic physics stepping (it is a large cost the
        RL policy doesn't need), so env.step() alone will NOT make a dropped object fall.
        Anything that needs the world to actually evolve — a release, an arm ramp — has to
        drive the simulator by hand, here.

        maybe_update_articulated_agent() re-syncs the robot's links after the physics tick;
        without it the arm's rendered pose lags the simulated one.
        """
        for _ in range(max(0, int(steps))):
            try:
                self._sim.step_physics(1.0 / 60.0)   # one 60 Hz tick
            except Exception as exc:
                print(f"[HabitatToolbox._release] physics settle failed: {exc}",
                      flush=True)
                return
            try:
                self._sim.maybe_update_articulated_agent()
            except Exception:
                pass

    def _forward_step(self) -> None:
        """Move base forward one step."""
        self._base_move_step("forward")

    def _head_depth_cam_T(self) -> Optional[np.ndarray]:
        """Camera-to-world 4x4 for the head-depth sensor, read straight from the
        simulator's sensor node. The obs `depth_rot`/`depth_trans` fields belong
        to a *different* camera (they place the head image ~8 m off the ground),
        so depth→world mapping must use this instead."""
        try:
            sensors = getattr(self._sim, "_sensors", {})
            key = ("head_depth" if "head_depth" in sensors
                   else next((k for k in sensors if "head_depth" in k), None))
            if key is None:
                return None
            node = sensors[key]._sensor_object.node
            return np.array(node.absolute_transformation(), dtype=np.float64)
        except Exception:
            return None

    def _get_depth_and_intrinsics(
        self, obs: dict
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Return (depth_HxW float32, K_3x3, E_4x4) for the head depth sensor.

        K uses the standard pinhole convention (pixel coordinates).
        E is the world-to-camera transform (BaseToolbox uses inv(E) to get
        camera-to-world when back-projecting depth).

        NOTE: For Habitat's coordinate system the back-projection result is
        [world.X, world.Y, world.Z].  Use _estimate_object_xy_from_depth_hab()
        instead of the base implementation when the nav-plane matters.
        """
        depth = None
        for dk in ("head_depth", "agent_0_head_depth"):
            if dk in obs:
                depth = np.asarray(obs[dk]).squeeze().astype(np.float32)
                break
        if depth is None or depth.ndim != 2:
            return None

        # camera-to-world from the sim sensor (NOT obs depth_rot/depth_trans,
        # which describe a different camera ~8 m off the ground)
        T_cw = self._head_depth_cam_T()
        if T_cw is None:
            return None

        H, W = depth.shape
        # Habitat's default head sensor uses 90° HFOV.
        # Synthesise the intrinsics from it: f = (W/2) / tan(hfov/2), principal point at the
        # image centre. Habitat exposes no K, so this reconstruction is the only source.
        hfov = math.pi / 2.0
        fx = fy = W / (2.0 * math.tan(hfov / 2.0))
        cx, cy = W / 2.0, H / 2.0
        K = np.array([[fx, 0.0, cx],
                      [0.0, fy, cy],
                      [0.0, 0.0, 1.0]], dtype=np.float32)

        # BaseToolbox expects world-to-camera so it can invert for backprojection.
        # T_cw is camera-to-world, hence the inverse here — the two cancel out in the base
        # class, but the base class's contract is what it is.
        E = np.linalg.inv(T_cw).astype(np.float32)

        return depth, K, E

    # ── Habitat-specific depth backprojection ─────────────────────────────────

    def _save_localization_crop(self, rgb_full, box, target, src, depth_m) -> None:
        """Save the bbox crop + an annotated full frame used for object
        localization, into <images_dir>/crops/ for inspection/debugging.

        Pure diagnostics, but the most valuable diagnostic in this file: when a grasp
        misses, these two images answer "did the detector find the right thing?" and "was
        the depth sane?" at a glance. The annotation carries the target name, WHICH localiser
        produced the box (gdino vs inspect), and the depth actually used.
        """
        if rgb_full is None or not self._last_image_path:
            return
        try:
            from PIL import Image as _PIL, ImageDraw as _Draw
            x1, y1, x2, y2 = box
            base = os.path.splitext(os.path.basename(self._last_image_path))[0]
            tgt = "".join(c if c.isalnum() else "_" for c in target)[:30]
            crop_dir = os.path.join(os.path.dirname(self._last_image_path), "crops")
            os.makedirs(crop_dir, exist_ok=True)
            crop_path = os.path.join(crop_dir, f"{base}_{tgt}_crop.png")
            _PIL.fromarray(rgb_full[y1:y2, x1:x2]).save(crop_path)
            ann = _PIL.fromarray(rgb_full.copy())
            dr = _Draw.Draw(ann)
            dr.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
            dr.text((x1 + 2, max(0, y1 - 12)), f"{target} {src} d={depth_m:.2f}m",
                    fill=(255, 0, 0))
            ann.save(os.path.join(crop_dir, f"{base}_{tgt}_box.png"))
            print(f"[HabitatToolbox] localization crop ({src}) → {crop_path}", flush=True)
        except Exception as _e:
            print(f"[HabitatToolbox] crop save failed: {_e}", flush=True)

    def _estimate_object_xy_from_depth_hab(
        self, target: str
    ) -> Optional[np.ndarray]:
        """
        Back-project depth + Gemini/GDino bbox → 2D nav position [world.X, world.Z].

        Uses Habitat's camera convention: depth sensor looks in -Z_cam direction,
        so a point at depth d has camera-space coords (x_c, y_c, -d).
        Navigation 2D position: [world.X, world.Z].
        """
        obs = self._last_obs
        if obs is None:
            return None

        depth = None
        for dk in ("head_depth", "agent_0_head_depth"):
            if dk in obs:
                depth = np.asarray(obs[dk]).squeeze().astype(np.float32)
                break
        if depth is None or depth.ndim != 2:
            return None

        # Camera-to-world straight from the sim sensor. The obs depth_rot/
        # depth_trans belong to a DIFFERENT camera (place the head image ~8 m
        # off the ground), so they must NOT be used here.
        T_cw = self._head_depth_cam_T()
        if T_cw is None:
            return None

        H, W = depth.shape
        hfov  = math.pi / 2.0
        f_inv = math.tan(hfov / 2.0)   # 1/f in normalised coords

        # ── Pixel target ──────────────────────────────────────────────────────
        from PIL import Image as _PIL
        rgb_full = None
        if self._last_image_path and os.path.exists(self._last_image_path):
            try:
                rgb_full = np.array(_PIL.open(self._last_image_path).convert("RGB"))
            except Exception:
                rgb_full = None

        bbox: Optional[list] = None
        bbox_src = "none"
        if self._grounding_dino is not None and rgb_full is not None:
            try:
                dets = self._grounding_dino.detect(rgb_full, target)
                if dets:
                    bbox = dets[0]["bbox"]
                    bbox_src = f"gdino(score={dets[0].get('score')})"
            except Exception as _e:
                print(f"[HabitatToolbox] GDino failed: {_e}")

        if bbox is None:
            inspect_res = self.inspect(
                image_path=self._last_image_path or "",
                question=f"Locate the {target}. Return its pixel bounding box.",
            )
            if inspect_res.ok:
                cand = inspect_res.data.get("candidate_bboxes", [])
                if cand:
                    bbox = cand[0]; bbox_src = "inspect"

        if bbox is not None and len(bbox) >= 4:
            x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
            x1 = max(0, min(W - 1, x1)); y1 = max(0, min(H - 1, y1))
            x2 = max(x1 + 1, min(W, x2)); y2 = max(y1 + 1, min(H, y2))
            u_c = (x1 + x2) // 2
            v_c = (y1 + y2) // 2
            # depth over the central 60% of the box (avoids background bleed at edges)
            mx = (x2 - x1) // 5; my = (y2 - y1) // 5
            patch = depth[y1 + my:y2 - my or y2, x1 + mx:x2 - mx or x2]
            valid = patch[(patch > 0.05) & (patch < 10.0)]
            d = float(np.median(valid)) if valid.size > 0 else float(depth[v_c, u_c])
            self._save_localization_crop(rgb_full, (x1, y1, x2, y2), target, bbox_src, d)
        else:
            u_c, v_c = W // 2, H // 2
            d = float(depth[v_c, u_c])

        if d < 0.05 or d > 10.0:
            print(f"[HabitatToolbox] invalid depth {d:.3f}m for '{target}'")
            return None

        # ── Normalised pixel → camera space (Habitat convention) ─────────────
        xs_n = (2.0 * u_c / (W - 1)) - 1.0   # ∈ [-1, +1]
        ys_n = 1.0 - (2.0 * v_c / (H - 1))   # ∈ [+1, -1]  (flipped: rows grow downward)

        x_c =  xs_n * d * f_inv   # camera X
        y_c =  ys_n * d * f_inv   # camera Y (down in image → positive)
        z_c = -d                  # Habitat: forward = -Z_cam

        p_cam_h = np.array([x_c, y_c, z_c, 1.0], dtype=np.float64)

        # ── Camera → world (T_cw from the sim sensor, set above) ──────────────
        p_world = T_cw @ p_cam_h

        # Nav 2D: (world.X, world.Z)  — Y is vertical in Habitat.
        # THIS is why the method exists rather than reusing BaseToolbox's version: the base
        # class takes p_world[:2] (X, Y), which in Habitat is "sideways and UP" — a
        # meaningless navigation goal. Here we take X and Z, the actual floor plane.
        obj_xy = np.array([float(p_world[0]), float(p_world[2])])
        print(f"[HabitatToolbox] depth: '{target}'  d={d:.2f}m  "
              f"world=({p_world[0]:.2f},{p_world[1]:.2f},{p_world[2]:.2f})")
        return obj_xy

    # ── navigate() override — oracle_nav_coord_action ─────────────────────────

    def navigate(self, target: dict, **_) -> ToolResult:  # type: ignore[override]
        """
        Navigate to a pose using Habitat's oracle_nav_coord_action.

        This is the same navigation primitive used by OWMM-Agent, ensuring a
        fair comparison.  The action plans and executes a path internally;
        we loop until the robot arrives or max steps are reached.

        target: {"memory_id": "mem_000042"} — a retrieve_memory candidate.
                Raw coordinate/pose goals are not accepted.
        """
        target_xy, target_yaw, nav_label = self._resolve_navigate_target(target)
        if target_xy is None:
            return ToolResult(ok=False, tool="navigate",
                              summary=f"navigate: bad target {target}")

        x_nav, z_nav = float(target_xy[0]), float(target_xy[1])

        # Floor height from the robot's current Y position
        try:
            floor_y = float(self._robot().base_pos[1])
        except Exception:
            floor_y = 0.0

        # OracleNavCoordAction contract (the registered impl is in
        # social_nav/oracle_social_nav_actions.py, subclass OracleNavDiffBaseAction):
        #   oracle_nav_coord_action = [world.X, floor_Y, world.Z, orientation_rad]
        #   - element [:3] is the drive-to position, element [3] the desired final yaw.
        #   - if orientation ∈ [-2π, 2π]  → if_orien=True,  dist_thresh tightens to 0.03 m
        #     and the robot also aligns to that yaw.
        #   - if orientation ∉ [-2π, 2π] (e.g. 10.0) → if_orien=False, dist_thresh=0.4 m,
        #     position-only (no final-yaw constraint).
        # It does NOT read oracle_nav_lookat_action / mode. The vector MUST be a 4-vec:
        # the action does nav_to_target_coord[3], so a 3-vec raises IndexError.
        # We navigate position-only (10.0) so the robot reliably arrives within 0.4 m
        # rather than chasing a 3 cm + exact-yaw goal it may never converge to.
        orientation = 10.0
        nav_target  = np.array([x_nav, floor_y, z_nav, orientation],
                               dtype=np.float32)

        dist    = float("inf")
        outcome = "incomplete"

        for step in range(self._NAV_MAX_STEPS):
            if step % self._EVENT_EVERY == 0:
                self._pump_events()
            if self._env._episode_over:
                outcome = "episode_over"
                break

            obs = self._env.step({
                "action": "oracle_nav_coord_action",
                "action_args": {"oracle_nav_coord_action": nav_target},
            })
            self._last_obs = obs

            # Memorize what the robot sees on its way (throttled to ~3 s).
            self._memorize_live(obs)

            # Live rendering during the drive (throttled). Without this the
            # display would freeze for the whole navigation and only refresh
            # at the closing observe().
            if (_RENDER or self._display) and step % _RENDER_EVERY == 0:
                self._show_frame(obs)

            # Distance check in nav plane (XZ). Note loc[0] and loc[2] — element 1 is the
            # robot's height, which must not enter a planar distance.
            loc     = np.asarray(obs.get("localization_sensor",
                                         [0.0, 0.0, 0.0, 0.0])).flatten()
            curr_2d = np.array([float(loc[0]), float(loc[2])])
            dist    = float(np.linalg.norm(curr_2d - np.array([x_nav, z_nav])))

            if dist < self._NAV_STOP_DIST:
                outcome = f"reached  dist={dist:.2f}m"
                break

            # Check oracle_nav skill_done flag.
            # The oracle planner sets this when it has finished its path — which can happen
            # while still short of the goal (the goal is off-navmesh, or behind a closed
            # door). Treat that as best-effort rather than spinning here until _NAV_MAX_STEPS.
            nav_act = self._env._task.actions.get("oracle_nav_coord_action")
            if nav_act is not None and getattr(nav_act, "skill_done", False):
                outcome = f"best_effort  dist={dist:.2f}m"
                break
        else:
            # for/else: never broke ⇒ the robot is stuck and the oracle never declared done.
            outcome = f"max_steps  dist={dist:.2f}m"

        # Honor a requested final orientation. oracle-nav drives position-only
        # (reliable arrival within 0.4 m) and leaves the robot facing an
        # arbitrary direction, so align the final yaw in place here — otherwise
        # the agent reaches the target but faces away and can't see it.
        if target_yaw is not None and not self._env._episode_over:
            try:
                # Set base_rot directly rather than driving _align_yaw's rotation loop: the
                # robot has arrived and there is nothing to collide with in a pure yaw, so
                # the loop would only burn steps.
                self._robot().base_rot = float(target_yaw)
                obs = self._env.step(self._null_step_action())
                self._last_obs = obs
                if _RENDER or self._display:
                    self._show_frame(obs)
                outcome += f"  yaw_aligned={float(target_yaw):.2f}"
            except Exception as e:
                print(f"[navigate] final yaw align failed: {e}", flush=True)

        obs_result = self.observe()
        # +0.5 m of slack over the stop distance — being just outside the threshold still
        # puts the target in view, and reporting failure would make the model retry a
        # navigate that essentially worked.
        reached    = dist < self._NAV_STOP_DIST + 0.5

        return ToolResult(
            ok=reached or "best_effort" in outcome,
            tool="navigate",
            summary=f"navigate[{nav_label}]: {outcome}  final_dist={dist:.2f}m",
            data={"target": target, "nav_label": nav_label, "outcome": outcome,
                  "final_pose": self._get_robot_pose(),
                  "distance_to_goal": dist},
            image_paths=obs_result.image_paths,
        )

    # ── Scene scan (memory pre-population) ────────────────────────────────────

    def _scan_rgb(self, obs: dict) -> Optional[np.ndarray]:
        """RGB for scene-scan memory. Prefer the wide head camera over the arm.

        A FIXED preference order, unlike _capture_rgb — it deliberately ignores
        primary_camera. Memory frames must all look alike for retrieval to work, and they
        must be frames the robot could plausibly navigate back to, which a gripper close-up
        is not.
        """
        for key in ("head_rgb", "agent_0_head_rgb", "third_person_sensor",
                    "arm_workspace_rgb", "articulated_agent_arm_rgb"):
            arr = obs.get(key)
            if arr is None:
                continue
            a = np.asarray(arr)
            if a.ndim == 3 and a.shape[2] >= 3:
                return a[..., :3].astype(np.uint8)
        return None

    def scan_scene(
        self,
        n_points: int,
        capture_dir: str,
        embedding_worker,
        episodic_memory,
        episode_id: str = "",
        yaws=(0.0, math.pi / 2, math.pi, 3 * math.pi / 2),
    ) -> int:
        """
        Teleport the robot to random navigable points and capture head-camera
        frames covering the scene, embedding each into FAISS memory.

        Replaces OWMM-Agent's 8 pre-coded scene-graph images with frames the
        robot actually observes.  The episode is reset afterwards so task
        metrics (num_steps, PDDL) start clean and the robot returns to its
        episode-defined start pose.

        Returns the number of frames captured.

        This is the pre-episode "look around the apartment" pass that gives the agent
        something to retrieve from on step 1. The robot is TELEPORTED between viewpoints
        rather than driven, which is legitimate precisely because of the reset at the end:
        the teleporting happens outside the episode, costs no steps, and leaves no trace in
        the task metrics. What it leaves behind is memory — frames + poses on disk, indexed
        in FAISS — exactly as if the robot had walked the route.

        n_points × len(yaws) frames: at each sampled floor position the robot spins through
        four cardinal headings, so a single position yields 360° of coverage.
        """
        import magnum as mn
        import datetime
        from sim.capture import NavCaptureWorker
        from agent.schemas import MemoryEntry, MemorySource, SensorData, EmbeddingRefs

        sim = self._sim
        pf  = sim.pathfinder
        # The pathfinder is what supplies navigable sample points. Without it we could only
        # teleport to arbitrary coordinates, most of which are inside walls.
        if not getattr(pf, "is_loaded", False):
            print("[scan] pathfinder not loaded — skipping scene scan", flush=True)
            return 0

        robot = self._robot()
        navcap = NavCaptureWorker(out_dir=str(capture_dir))
        saved  = 0

        print(f"[scan] scanning scene: {n_points} points × {len(yaws)} yaws", flush=True)
        for _ in range(int(n_points)):
            if self._env._episode_over:
                break
            # Random navigable point — guaranteed on the floor and reachable, so the frame
            # is one the robot could actually have taken.
            pt = np.asarray(pf.get_random_navigable_point(), dtype=np.float32)
            if not np.isfinite(pt).all():
                continue   # pathfinder occasionally returns NaN; just resample
            for yaw in yaws:
                if self._env._episode_over:
                    break
                try:
                    robot.base_pos = mn.Vector3(float(pt[0]), float(pt[1]), float(pt[2]))
                    robot.base_rot = float(yaw)
                except Exception as e:
                    print(f"[scan] teleport failed: {e}", flush=True)
                    continue

                # Null step renders the sensors at the teleported pose
                obs = self._env.step(self._null_step_action())
                self._last_obs = obs

                rgb = self._scan_rgb(obs)
                if rgb is None:
                    continue

                pose = self._get_robot_pose()           # [x_nav, z_nav, heading]
                xy   = np.array([pose[0], pose[1]], dtype=np.float32)
                idx  = navcap.enqueue(rgb, xy, pose[2])
                frame_path = os.path.join(str(capture_dir), "color", f"{idx:06d}.png")

                if embedding_worker is not None:
                    embedding_worker.enqueue(rgb=rgb, frame_path=frame_path,
                                             robot_xy=xy, robot_yaw=pose[2])
                if episodic_memory is not None:
                    episodic_memory.add_entry(MemoryEntry(
                        memory_id = f"mem_{idx:06d}",
                        sensor    = SensorData(
                            image_path = frame_path,
                            robot_pose = [pose[0], pose[1], pose[2]],
                            timestamp  = datetime.datetime.now().isoformat(),
                        ),
                        embeddings = EmbeddingRefs(),
                        source     = MemorySource(source_type="scene_scan",
                                                  episode_id=str(episode_id)),
                    ))
                saved += 1
                if _RENDER or self._display:
                    self._show_frame(obs)

        # Both flushes are required before the reset: retrieval reads PNGs back from disk,
        # so every queued frame must be written, and every vector must be in the index,
        # before the episode proper starts querying them.
        navcap.flush()
        navcap.stop()
        if embedding_worker is not None:
            embedding_worker.flush()

        # Restore the robot to its episode start pose + clean metrics.
        # This is what makes the scan "free": all the teleporting and stepping above is
        # erased from the task's step count and success metrics, leaving only the memory.
        self.reset_episode()
        print(f"[scan] captured {saved} frames → memory index "
              f"({getattr(embedding_worker, 'embedded', '?')} embedded)", flush=True)
        return saved

    def explore_frontier(
        self,
        capture_dir: str,
        embedding_worker,
        episodic_memory,
        episode_id: str = "",
        max_iters: int = 40,
        res: float = 0.10,
        max_range: float = 1.5,
        lam: float = 0.5,
        gain_radius: float = 3.0,
        min_gain: int = 1,
        max_candidates: int = 60,
        drive: bool = True,
        video_path: Optional[str] = None,
        yaws=(0.0, math.pi / 2, math.pi, 3 * math.pi / 2),
    ) -> int:
        """
        Frontier-based active exploration to build scene memory (drop-in
        alternative to scan_scene).

        Loop:
          1. observe at the current viewpoint (4 yaws): fuse head-depth into a
             top-down occupancy map and embed each RGB frame into FAISS memory;
          2. detect frontiers (FREE cells touching UNKNOWN);
          3. sample navigable viewpoints near frontiers, score each by
             information_gain (UNKNOWN cells within gain_radius) − lam·travel_cost
             (geodesic distance from the robot);
          4. if the best score < min_gain → stop; else move to the best viewpoint
             and repeat.

        Like scan_scene this is a pre-task memory bootstrap: it teleports between
        viewpoints (drive=False, fast) and reset_episode()s afterwards so task
        metrics start clean. Pass drive=True to actually navigate (oracle_nav)
        between viewpoints instead. Returns the number of frames captured.
        """
        import magnum as mn
        import datetime
        import habitat_sim
        from sim.capture import NavCaptureWorker
        from sim.frontier import OccupancyMap
        from agent.schemas import MemoryEntry, MemorySource, SensorData, EmbeddingRefs

        sim = self._sim
        pf = sim.pathfinder
        if not getattr(pf, "is_loaded", False):
            print("[explore] pathfinder not loaded — skipping", flush=True)
            return 0

        robot = self._robot()
        # A single floor height for the whole map. The occupancy fusion classifies points as
        # obstacle-or-floor by their height ABOVE this, and the geodesic queries need a Y to
        # sit their endpoints at. Single-storey scenes only.
        floor_y = float(robot.base_pos[1])
        # Size the map to the pathfinder's own bounds, taking X and Z (the floor plane).
        lo, hi = pf.get_bounds()
        omap = OccupancyMap(lo[0], lo[2], hi[0], hi[2], res=res)
        navcap = NavCaptureWorker(out_dir=str(capture_dir))
        saved = 0
        visited = []                          # viewpoints already observed from
        blocked = []                          # frontier targets that stalled nav
        frames = []                           # side-by-side video frames

        # target object(s) to mark (green) on the map panel.
        # PURELY for the debug video — ground-truth object positions, so this must never
        # feed exploration or memory. It lets a human see whether the explorer is heading
        # toward the object it will eventually need.
        _tmarks = []
        try:
            _rom0 = sim.get_rigid_object_manager()
            for _tk in (self._env.current_episode.targets or {}):
                _hh = [x for x in _rom0.get_object_handles() if _tk.split("_:")[0] in x]
                if _hh:
                    _tt = _rom0.get_object_by_handle(_hh[0]).translation
                    _tmarks.append((float(_tt.x), float(_tt.z), (0, 200, 0)))
        except Exception:
            pass

        def _push_frame(obs):
            """Compose [front RGB | depth | occupancy map] for the video."""
            if video_path is None:
                return
            try:
                from PIL import Image, ImageDraw
                rgb = obs.get("head_rgb")
                if rgb is None:
                    rgb = self._scan_rgb(obs)
                rgb = np.asarray(rgb)[..., :3].astype(np.uint8)
                dep = None
                for dk in ("agent_0_head_depth", "head_depth", "depth_obs"):
                    if dk in obs:
                        dep = np.asarray(obs[dk]).squeeze().astype(np.float32); break
                # Depth → greyscale, INVERTED (near = bright) and normalised by max_range,
                # so the visualisation shows exactly the band the mapper actually fuses.
                if dep is not None and dep.ndim == 2:
                    g = (255 * (1 - np.clip(dep / max_range, 0, 1))).astype(np.uint8)
                    dep_img = np.stack([g, g, g], axis=-1)
                else:
                    dep_img = np.zeros_like(rgb)
                pose = self._get_robot_pose()
                mp = omap.to_rgb(visited=visited, marks=_tmarks,
                                 robot=(pose[0], pose[1]))
                H = 384

                def fit(a, label):
                    im = Image.fromarray(a).resize(
                        (max(1, int(a.shape[1] * H / a.shape[0])), H))
                    ImageDraw.Draw(im).text((6, 4), label, fill=(255, 60, 60))
                    return np.asarray(im)[..., :3]
                frames.append(np.concatenate(
                    [fit(rgb, "front RGB"), fit(dep_img, "depth"),
                     fit(mp, "occupancy map (white=free black=obstacle grey=unknown)")],
                    axis=1))
            except Exception as e:
                print(f"[explore] video frame failed: {e}", flush=True)

        def _geodesic(x0, z0, x1, z1) -> float:
            """Walking distance between two floor points, respecting walls.

            This is the travel-cost term of the exploration objective, and it must be
            geodesic rather than straight-line: a frontier 2 m away through a wall is really
            a 15 m walk around, and Euclidean distance would send the robot chasing it.

            Falls back to Euclidean when no path exists (an unreachable frontier), which
            makes such candidates look cheap — but they get filtered out later by the
            navigability check on the sampled viewpoint.
            """
            sp = habitat_sim.ShortestPath()
            sp.requested_start = np.array([x0, floor_y, z0], dtype=np.float32)
            sp.requested_end = np.array([x1, floor_y, z1], dtype=np.float32)
            if pf.find_path(sp) and np.isfinite(sp.geodesic_distance):
                return float(sp.geodesic_distance)
            return float(np.hypot(x1 - x0, z1 - z0))

        def _fuse_and_capture(obs) -> None:
            """Fuse one head-depth frame into the map, embed its RGB, record video.

            The workhorse: every frame the explorer takes goes through here, doing double
            duty — the DEPTH updates the occupancy map that drives exploration, and the RGB
            goes into episodic memory, which is the actual product of the whole exercise.
            """
            nonlocal saved
            self._last_obs = obs
            # Use the ACTUAL head-depth camera-to-world from the sim (obs
            # depth_rot/depth_trans are a different camera → phantom walls).
            depth = None
            for dk in ("head_depth", "agent_0_head_depth"):
                if dk in obs:
                    depth = np.asarray(obs[dk]).squeeze()
                    break
            T = self._head_depth_cam_T()
            if depth is not None and depth.ndim == 2 and T is not None:
                omap.integrate(depth.astype(np.float32), T[:3, :3], T[:3, 3],
                               floor_y, max_range=max_range)
            rgb = self._scan_rgb(obs)
            if rgb is not None:
                pose = self._get_robot_pose()
                xy = np.array([pose[0], pose[1]], dtype=np.float32)
                idx = navcap.enqueue(rgb, xy, pose[2])
                frame_path = os.path.join(str(capture_dir), "color", f"{idx:06d}.png")
                if embedding_worker is not None:
                    embedding_worker.enqueue(rgb=rgb, frame_path=frame_path,
                                             robot_xy=xy, robot_yaw=pose[2])
                if episodic_memory is not None:
                    episodic_memory.add_entry(MemoryEntry(
                        memory_id=f"mem_{idx:06d}",
                        sensor=SensorData(
                            image_path=frame_path,
                            robot_pose=[pose[0], pose[1], pose[2]],
                            timestamp=datetime.datetime.now().isoformat(),
                        ),
                        embeddings=EmbeddingRefs(),
                        source=MemorySource(source_type="frontier_explore",
                                            episode_id=str(episode_id)),
                    ))
                saved += 1
            _push_frame(obs)
            if _RENDER or self._display:
                self._show_frame(obs)

        def _observe_and_map() -> None:
            """Rotate in place through the yaws, mapping/capturing at each.

            A single 90°-FOV frame maps only a narrow wedge, so spinning through four
            cardinal yaws at every viewpoint gives full 360° coverage from one position —
            far cheaper than visiting four positions.
            """
            for yaw in yaws:
                if self._env._episode_over:
                    break
                try:
                    robot.base_rot = float(yaw)
                except Exception:
                    continue
                _fuse_and_capture(self._env.step(self._null_step_action()))

        def _drive_and_map(tx, tz) -> tuple[bool, str, float, float, float]:
            """Continuously drive to (tx,tz) via oracle_nav, mapping + capturing
            frames en route (no teleporting)."""
            nav_target = np.array([tx, floor_y, tz, 10.0], dtype=np.float32)
            start_pose = self._get_robot_pose()
            last_progress_xy = np.array(start_pose[:2], dtype=np.float64)
            last_progress_dist = float(np.hypot(start_pose[0] - tx,
                                                start_pose[1] - tz))
            stagnant_steps = 0
            window_step = 0
            window_xy = last_progress_xy.copy()
            window_dist = last_progress_dist
            final_dist = last_progress_dist
            for step in range(self._NAV_MAX_STEPS):
                if self._env._episode_over:
                    break
                obs = self._env.step({
                    "action": "oracle_nav_coord_action",
                    "action_args": {"oracle_nav_coord_action": nav_target}})
                if step % 3 == 0:                 # throttle mapping/capture en route
                    _fuse_and_capture(obs)
                else:
                    self._last_obs = obs
                    if (_RENDER or self._display) and step % _RENDER_EVERY == 0:
                        self._show_frame(obs)
                loc = np.asarray(obs.get("localization_sensor",
                                         [0.0, 0.0, 0.0, 0.0])).flatten()
                curr_xy = np.array([float(loc[0]), float(loc[2])],
                                   dtype=np.float64)
                final_dist = float(np.hypot(curr_xy[0] - tx, curr_xy[1] - tz))
                if final_dist < self._NAV_STOP_DIST:
                    return True, "reached", curr_xy[0], curr_xy[1], final_dist
                nav_act = self._env._task.actions.get("oracle_nav_coord_action")
                if nav_act is not None and getattr(nav_act, "skill_done", False):
                    return True, "skill_done", curr_xy[0], curr_xy[1], final_dist

                # Oracle nav can get trapped in a wall/corner and keep rotating
                # forever. Detect yaw-only or sub-centimetre crawl so exploration
                # can blacklist this target and try another frontier.
                #
                # TWO stall detectors, because there are two distinct pathologies:
                #  (a) hard stall — the robot is wedged and moves < 5 cm per step for 120
                #      consecutive steps. Caught by `stagnant_steps` below.
                #  (b) slow crawl — it inches forward just enough to keep resetting the
                #      stagnant counter, but covers < 15 cm in 120 steps. Caught by the
                #      sliding `window_*` check further down.
                # Progress counts as EITHER moving in the world OR closing the distance to
                # the goal, so a legitimate detour around furniture isn't punished.
                moved = float(np.linalg.norm(curr_xy - last_progress_xy))
                dist_improved = last_progress_dist - final_dist
                if moved > 0.05 or dist_improved > 0.05:
                    last_progress_xy = curr_xy
                    last_progress_dist = final_dist
                    stagnant_steps = 0
                else:
                    stagnant_steps += 1
                # step >= 120 grace period: the oracle legitimately spends the first stretch
                # turning on the spot to face its path, which looks exactly like a stall.
                if step >= 120 and stagnant_steps >= 120:
                    print(
                        f"[explore] drive stalled near "
                        f"({curr_xy[0]:.2f},{curr_xy[1]:.2f}) while targeting "
                        f"({tx:.2f},{tz:.2f}); dist={final_dist:.2f}m — "
                        "blacklisting frontier",
                        flush=True,
                    )
                    return False, "stalled", curr_xy[0], curr_xy[1], final_dist
                if step - window_step >= 120:
                    window_moved = float(np.linalg.norm(curr_xy - window_xy))
                    window_improved = window_dist - final_dist
                    if window_moved < 0.15 and window_improved < 0.15:
                        print(
                            f"[explore] drive too slow near "
                            f"({curr_xy[0]:.2f},{curr_xy[1]:.2f}) while targeting "
                            f"({tx:.2f},{tz:.2f}); moved={window_moved:.2f}m "
                            f"improved={window_improved:.2f}m over 120 steps — "
                            "blacklisting frontier",
                            flush=True,
                        )
                        return False, "too_slow", curr_xy[0], curr_xy[1], final_dist
                    window_step = step
                    window_xy = curr_xy.copy()
                    window_dist = final_dist
            pose = self._get_robot_pose()
            return False, "max_steps", float(pose[0]), float(pose[1]), final_dist

        print(f"[explore] frontier exploration: ≤{max_iters} viewpoints "
              f"(res={res}m, λ={lam}, {'continuous-drive' if drive else 'teleport'})",
              flush=True)
        _observe_and_map()
        start_pose = self._get_robot_pose()
        visited.append((start_pose[0], start_pose[1]))

        for it in range(int(max_iters)):
            if self._env._episode_over:
                break
            clusters = omap.frontier_clusters(min_size=2)   # (wx, wz, size), largest first
            if not clusters:
                print("[explore] no frontiers left — fully explored", flush=True)
                break

            # ── Score every candidate frontier: gain − λ·cost ──────────────────
            # The classic Yamauchi objective. `size` (the frontier's boundary length) is the
            # information-gain proxy — a long frontier borders a lot of unknown space — and
            # geodesic distance is the price of going there. λ sets the exchange rate
            # between "how much I'd learn" and "how far I'd walk".
            rpose = self._get_robot_pose()
            rx, rz = rpose[0], rpose[1]
            best, best_score = None, -float("inf")
            for (wx, wz, size) in clusters[:max_candidates]:
                # Snap the frontier cell onto the navmesh: the cell itself is a map
                # abstraction and may not be a place the robot can actually stand.
                snap = np.asarray(pf.snap_point(
                    np.array([wx, floor_y, wz], dtype=np.float32)), dtype=np.float32)
                if not np.isfinite(snap).all():
                    continue   # unreachable — snap_point returns NaN off-navmesh
                sx, sz = float(snap[0]), float(snap[2])
                # only skip a candidate that is essentially a viewpoint we already
                # observed from (avoid exact re-picks / oscillation), not merely
                # *near* one — frontiers naturally sit close to the explored blob.
                # Hence the tight 0.5 m radius: a larger one would reject genuinely new
                # viewpoints just for being adjacent to explored space, which every frontier
                # is, by definition.
                if any(math.hypot(sx - vx, sz - vz) < 0.5 for vx, vz in visited):
                    continue
                # Blacklist radius is larger (1.0 m): if the robot couldn't reach one point,
                # it almost certainly can't reach its immediate neighbours either.
                if any(math.hypot(sx - bx, sz - bz) < 1.0 for bx, bz in blocked):
                    continue
                cost = _geodesic(rx, rz, sx, sz)
                # gain = frontier-cluster size (boundary length ∝ unknown area behind)
                score = size - lam * cost
                if score > best_score:
                    best_score, best = score, (sx, sz, size)

            # Termination: even the best frontier isn't worth the walk. Note this stops on
            # *diminishing returns*, not on total coverage — the remaining unknown space is
            # typically slivers behind furniture that cost more to reach than they teach.
            if best is None or best_score < min_gain:
                print(f"[explore] {len(clusters)} clusters but best score "
                      f"{best_score:.1f} < {min_gain} — stopping at iter {it}", flush=True)
                break

            sx, sz, size = best
            st = omap.stats()
            print(f"[explore] iter {it}: {len(clusters)} clusters → "
                  f"({sx:.2f},{sz:.2f}) cluster={size} score={best_score:.1f}  "
                  f"free={st['free']} occ={st['occ']} unknown={st['unknown']}", flush=True)

            if drive:
                ok, reason, fx, fz, dist = _drive_and_map(sx, sz)
                if ok:
                    _observe_and_map()        # look around once arrived
                    visited.append((sx, sz))
                else:
                    blocked.append((sx, sz))
                    if not any(math.hypot(fx - vx, fz - vz) < 0.5
                               for vx, vz in visited):
                        visited.append((fx, fz))
                        _observe_and_map()
                    print(
                        f"[explore] skipped frontier ({sx:.2f},{sz:.2f}) "
                        f"after {reason}; final=({fx:.2f},{fz:.2f}) "
                        f"dist={dist:.2f}m",
                        flush=True,
                    )
            else:
                try:
                    robot.base_pos = mn.Vector3(sx, floor_y, sz)
                except Exception as e:
                    print(f"[explore] teleport failed: {e}", flush=True)
                    break
                _observe_and_map()
                visited.append((sx, sz))

        # diagnostic: dump the occupancy map with the visited path + target marked
        try:
            marks = []
            rom2 = sim.get_rigid_object_manager()
            for tk in (self._env.current_episode.targets or {}):
                h = [x for x in rom2.get_object_handles() if tk.split("_:")[0] in x]
                if h:
                    t = rom2.get_object_by_handle(h[0]).translation
                    marks.append((float(t.x), float(t.z), (0, 200, 0)))
            omap.save_png(os.path.join(str(capture_dir), "explore_map.png"),
                          visited=visited, marks=marks)
            np.savez(os.path.join(str(capture_dir), "explore_grid.npz"),
                     grid=omap.grid, xmin=omap.xmin, zmin=omap.zmin,
                     res=omap.res, floor_y=floor_y)
            print(f"[explore] map image → {capture_dir}/explore_map.png", flush=True)
        except Exception as e:
            print(f"[explore] map dump failed: {e}", flush=True)

        navcap.flush(); navcap.stop()
        if embedding_worker is not None:
            embedding_worker.flush()

        if video_path is not None and frames:
            try:
                import imageio.v2 as imageio
                h = min(f.shape[0] for f in frames)
                w = min(f.shape[1] for f in frames)
                h -= h % 2; w -= w % 2          # libx264/yuv420p needs even dims
                with imageio.get_writer(video_path, fps=4, macro_block_size=None,
                                        codec="libx264") as wv:
                    for f in frames:
                        wv.append_data(np.ascontiguousarray(f[:h, :w]))
                print(f"[explore] video ({len(frames)} frames) → {video_path}", flush=True)
            except Exception as e:
                print(f"[explore] video write failed: {e}", flush=True)

        self.reset_episode()
        st = omap.stats()
        print(f"[explore] captured {saved} frames; map free={st['free']} "
              f"occ={st['occ']} unknown={st['unknown']}", flush=True)
        return saved

    # ── Episode management ────────────────────────────────────────────────────

    def reset_episode(self) -> dict:
        """Reset the Habitat env and return initial observations."""
        obs = self._env.reset()
        self._last_obs = obs
        self._configure_gripper_camera()
        return obs

    def get_metrics(self) -> dict:
        """Return Habitat task metrics (e.g. pddl_success, num_steps)."""
        try:
            return dict(self._env.get_metrics())
        except Exception:
            return {}

    def is_episode_over(self) -> bool:
        return bool(self._env._episode_over)
