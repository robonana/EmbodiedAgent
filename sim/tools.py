"""
sim/tools.py — AgentToolbox: ManiSkill/ReplicaCAD implementation of the agent toolbox.

Implements the abstract primitives of BaseToolbox for ManiSkill/SAPIEN.
All shared tool logic (navigate, base_move, manipulate, inspect, …) lives in BaseToolbox.

So the whole file is the ~10 small methods that say "here is how you take a physics step /
read a pose / close a gripper in *this* simulator", plus the arm IK those methods need. If
you are looking for what a tool *does*, it is in agent/toolbox_base.py; if you are looking
for how it touches the simulator, it is here.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Optional

import numpy as np

from agent.toolbox_base import BaseToolbox

# ── Arm geometry constants (from xlerobot.urdf) ───────────────────────────────
# Hard-coded rather than read from the URDF at runtime: the 2-link IK below needs link
# lengths and a shoulder origin in the base frame, and extracting those from SAPIEN's
# articulation tree is far more code than pinning the four numbers the arm actually has.
# They must be re-derived if the URDF changes.
#
# arm_base_joint xyz=(-0.135, -0.133, 0.760), rpy=(0,0,π/2) → Rotation joint
# adds (0.0452, 0, 0.0165) in base_link after the 90° transform
_ARM1_SHOULDER = np.array([-0.0898, -0.133, 0.7765])  # in robot base_link frame
_ARM_L1        = 0.1159   # upper-arm link length (m)
_ARM_L2        = 0.1350   # forearm link length (m)
_ARM_TIP       = 0.108    # wrist-to-EE-tip length (m)
_GRIPPER_OPEN  = 2.5      # qpos value for open gripper
_GRIPPER_CLOSE = 0.1      # qpos value for closed gripper


def _ik_2link(x: float, y: float) -> tuple[float, float]:
    """2-link planar IK for XLeRobot arm.  x=reach (m), y=height above shoulder (m).

    Returns (Pitch, Elbow) qpos values with offsets matching the URDF.
    Clamps to joint limits and workspace boundary silently.

    Closed-form elbow-up solution to the classic 2-link problem:
      * law of cosines gives the elbow angle from the target distance r,
      * the shoulder angle is the direction to the target (beta) plus the internal
        triangle angle (gamma).

    "Clamps silently" is a deliberate policy. An unreachable target is projected onto the
    workspace boundary and the arm stretches as far as it can, rather than raising. The
    grasp then simply misses, the backend reports grasped=False, and the VLM sees an honest
    failure it can recover from by driving closer — which is far more useful than an
    exception that kills the episode.
    """
    # The URDF's zero pose is not the IK's zero pose: the links have small perpendicular
    # offsets, so the joint frames sit at a slight angle to the link line. These constants
    # convert IK angles into joint angles.
    t1_off = math.atan2(0.028, 0.11257)
    t2_off = math.atan2(0.0052, 0.1349) + t1_off

    r = math.sqrt(x * x + y * y)
    r_max = _ARM_L1 + _ARM_L2          # fully extended
    r_min = abs(_ARM_L1 - _ARM_L2)     # fully folded — the hole in the middle of the annulus
    # Project onto the reachable annulus, preserving direction.
    if r > r_max:
        s = r_max / r; x *= s; y *= s; r = r_max
    elif 0 < r < r_min:
        s = r_min / r; x *= s; y *= s; r = r_min

    # Law of cosines. The clamp to [-1, 1] guards acos against floating-point overshoot
    # exactly at the boundary (where the ratio can come out as 1.0000000002).
    cos_t2 = max(-1.0, min(1.0, -(r * r - _ARM_L1 * _ARM_L1 - _ARM_L2 * _ARM_L2)
                           / (2 * _ARM_L1 * _ARM_L2)))
    t2 = math.pi - math.acos(cos_t2)                 # elbow
    beta  = math.atan2(y, x)                          # direction to target
    gamma = math.atan2(_ARM_L2 * math.sin(t2), _ARM_L1 + _ARM_L2 * math.cos(t2))
    t1    = beta + gamma                              # shoulder (elbow-up branch)

    # Apply the URDF offsets and clamp to the real joint limits.
    pitch = max(-0.1, min(3.45, t1 + t1_off))
    elbow = max(-0.2, min(math.pi, t2 + t2_off))
    return pitch, elbow


class AgentToolbox(BaseToolbox):
    """
    ManiSkill/ReplicaCAD backend for BaseToolbox.

    Adds five sim-only constructor params and implements the ten abstract
    primitive methods; everything else is inherited from BaseToolbox.
    """

    def __init__(
        self,
        env,                             # gym.Env
        agent,                           # uenv.agent (SAPIEN articulation wrapper)
        root_pos: np.ndarray,            # world XY offset of robot base
        action_shape: int,               # scalar action dim
        scene_builder,                   # uenv.scene_builder
        nav_verts: Optional[np.ndarray], # (N,2) navmesh XY positions or None
        capture_out_dir: str,
        gemini_client,
        log_dir: str,
        embedding_worker=None,
        episodic_memory=None,
        capture_worker=None,
        capture_interval: float = 3.0,
        sensory_tick=None,               # unused, kept for call-site compatibility
        retrieval_model: str = "siglip_base",
        retrieval_data_root: Optional[str] = None,
        scene_id: Optional[str] = None,
        event_callback: Optional[Callable] = None,
        grounding_dino=None,
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
        self.env           = env
        self.agent         = agent
        self.root_pos      = root_pos
        # Pre-allocated zero action. The base is driven kinematically (set_qpos), so every
        # env.step() we take is a *no-op action* whose only purpose is to advance physics
        # and re-render. Allocating it once avoids a fresh array on every control tick.
        self._action_zeros = np.zeros(action_shape, dtype=np.float32)
        self.scene_builder = scene_builder
        self.nav_verts     = nav_verts

        self._capture_worker   = capture_worker
        self._capture_interval = capture_interval
        # Far in the past, so the very first _step() triggers a capture immediately.
        self._last_bg_capture_t: float = -1e9

        # Lazy NavGrid — built on first navigate call
        # (rasterising + eroding the navmesh takes a moment, and a run that never navigates
        # should not pay for it).
        self._nav_grid = None

    # ── Abstract primitive implementations ───────────────────────────────────

    def _step(self) -> dict:
        """One control cycle: advance physics, render, opportunistically capture a frame.

        Every loop in BaseToolbox (navigate, base_move, wait) bottoms out here, which is
        why the background capture lives inside it — memory then accumulates automatically
        as the robot moves, with no explicit "capture" tool.
        """
        obs, _, _, _, _ = self.env.step(self._action_zeros)
        self.env.render()
        self._bg_capture(obs)
        return obs

    def _bg_capture(self, obs: dict) -> None:
        """Capture a frame into the scan index at the configured interval.

        This is how episodic memory gets built during an episode. Rate-limited by wall-clock
        time (default one frame per 3 s): _step() runs at hundreds of Hz during a long
        navigate, and indexing every frame would flood the embedding queue with near-identical
        images while starving retrieval of variety.

        One frame fans out to three consumers, and all three must agree on the frame index:
          capture_worker  → writes color/{idx}.png + robot_xy/{idx}.txt
          embedding_worker→ embeds it into the FAISS index under that same path
          episodic_memory → records the memory_id ⇄ pose ⇄ image mapping
        The index is allocated by capture_worker.enqueue() precisely so the other two can be
        told about the frame before it has actually hit the disk.
        """
        if self._capture_worker is None:
            return
        now = time.time()
        if now - self._last_bg_capture_t < self._capture_interval:
            return
        from .capture import _capture_nav_frame
        from .env import get_robot_xy, get_robot_yaw
        from agent.episodic_memory import frame_to_memory_id
        rgb = _capture_nav_frame(obs)
        if rgb is None:
            return   # bad frame — try again next tick, don't burn the interval
        curr_xy  = get_robot_xy(self.agent, self.root_pos)
        curr_yaw = get_robot_yaw(self.agent)
        fidx  = self._capture_worker.enqueue(rgb, curr_xy, curr_yaw)
        # Predict the path the capture worker will write to. It must match exactly, since
        # this is the key the embedding index and the memory store are joined on.
        fpath = str(self.capture_out_dir / "color" / f"{fidx:06d}.png")
        if self.embedding_worker is not None:
            self.embedding_worker.enqueue(rgb, fpath, curr_xy, curr_yaw)
        if self.episodic_memory is not None:
            self.episodic_memory.add_entry(self.episodic_memory.create_entry(
                memory_id=frame_to_memory_id(fidx),
                image_path=fpath,
                robot_pose=[float(curr_xy[0]), float(curr_xy[1]), float(curr_yaw)],
                embedding_model=self.retrieval_model,
                source_type="agent_scan",
            ))
        self._last_bg_capture_t = now

    def _capture_rgb(self, obs: dict) -> Optional[np.ndarray]:
        from .capture import _capture_nav_frame
        return _capture_nav_frame(obs)

    def _get_robot_pose(self) -> list[float]:
        # root_pos + qpos[:2]: see get_robot_xy — the base joints are a displacement from
        # the spawn origin, not a world position.
        from .env import get_robot_xy, get_robot_yaw
        xy  = get_robot_xy(self.agent, self.root_pos)
        yaw = get_robot_yaw(self.agent)
        return [float(xy[0]), float(xy[1]), float(yaw)]

    def _navigate_step(self, bearing: float) -> None:
        from .nav_grid import kinematic_nav_step
        kinematic_nav_step(self.agent, bearing)

    def _base_move_step(self, motion: str) -> None:
        """One tick of a discrete base motion, applied kinematically.

        The four translations are the same operation at four different headings — forward
        along yaw, backward at yaw+π, and strafing at yaw±π/2 — so the robot side-steps
        without turning, which is what makes "left"/"right" useful for peeking around an
        occlusion.

        Note this only advances by one *tick*; BaseToolbox.base_move calls it repeatedly and
        decides when the full 0.30 m / 30° quantum has been achieved.
        """
        from .env import NAV_FWD_M_PER_STEP, NAV_ROT_RAD_PER_STEP

        qp = self.agent.robot.get_qpos().cpu().numpy().flatten().copy()
        yaw = float(qp[2])
        if motion == "forward":
            step_yaw = yaw
            qp[0] += NAV_FWD_M_PER_STEP * math.cos(step_yaw)
            qp[1] += NAV_FWD_M_PER_STEP * math.sin(step_yaw)
        elif motion == "backward":
            step_yaw = yaw + math.pi
            qp[0] += NAV_FWD_M_PER_STEP * math.cos(step_yaw)
            qp[1] += NAV_FWD_M_PER_STEP * math.sin(step_yaw)
        elif motion == "left":
            step_yaw = yaw + math.pi / 2     # strafe: translate, don't rotate
            qp[0] += NAV_FWD_M_PER_STEP * math.cos(step_yaw)
            qp[1] += NAV_FWD_M_PER_STEP * math.sin(step_yaw)
        elif motion == "right":
            step_yaw = yaw - math.pi / 2
            qp[0] += NAV_FWD_M_PER_STEP * math.cos(step_yaw)
            qp[1] += NAV_FWD_M_PER_STEP * math.sin(step_yaw)
        elif motion == "rotate 30 degrees":
            qp[2] += NAV_ROT_RAD_PER_STEP    # CCW / left
        elif motion == "rotate -30 degrees":
            qp[2] -= NAV_ROT_RAD_PER_STEP    # CW / right
        self.agent.robot.set_qpos(qp)

    def _plan_path(
        self, start_xy: np.ndarray, goal_xy: np.ndarray
    ) -> list[np.ndarray]:
        # No navmesh (some staging scenes ship without one) ⇒ degrade to a straight line and
        # let navigate()'s progress check handle getting stuck.
        nav_grid = self._get_nav_grid()
        if nav_grid is not None:
            return nav_grid.plan(start_xy, goal_xy)
        return [goal_xy.copy()]

    def _grasp(self, target: str) -> tuple[bool, str, float]:
        """Aim the arm at the named object and close the gripper.

        Returns (localized, object_name, distance) — localized means "we found a target and
        ATTEMPTED a grasp", not that anything is now held. See BaseToolbox.manipulate.

        IMPORTANT caveat: this is an ORACLE grasp. It resolves the target by name against
        the simulator's ground-truth object list and reads the object's true pose, rather
        than using the agent's perception. That makes it a manipulation *stand-in* — it lets
        the agent be evaluated on navigation, search and memory without a real grasp policy
        confounding the results — but it means grasp success here is not evidence of
        perception working. (Contrast _estimate_object_xy_from_depth in BaseToolbox, which is
        the perception-only path.)

        Six stages: pick the object → get its world pose → find the shoulder in world →
        transform the object into the shoulder's frame → solve IK → open, move, close.
        """
        import torch as _torch
        import sapien
        from .env import get_robot_xy, get_actor_xy, find_objects

        # ── 1. Find closest matching object ──────────────────────────────────
        curr_xy    = get_robot_xy(self.agent, self.root_pos)
        candidates: list[tuple[float, str, object]] = []
        if target:
            for name, actor in find_objects(self.scene_builder, target):
                try:
                    d = float(np.linalg.norm(get_actor_xy(actor) - curr_xy))
                    candidates.append((d, name, actor))
                except Exception:
                    pass
        if not candidates:
            # Name didn't match anything (the VLM said "cup", the scene calls it "tumbler").
            # Fall back to *every* object and take the nearest — the robot has presumably
            # driven up to the thing it wants, so proximity is a decent disambiguator.
            for name, actor in self.scene_builder.scene_objects.items():
                try:
                    d = float(np.linalg.norm(get_actor_xy(actor) - curr_xy))
                    candidates.append((d, name, actor))
                except Exception:
                    pass
        candidates.sort(key=lambda x: x[0])
        if not candidates:
            return False, "", 999.0   # empty scene — nothing to grasp
        dist, obj_name, actor = candidates[0]

        # ── 2. Object world position ──────────────────────────────────────────
        obj_pose = actor.pose if hasattr(actor, "pose") else actor.get_pose()
        obj_p    = obj_pose.p
        if _torch.is_tensor(obj_p):
            obj_p = obj_p.cpu().numpy()
        obj_world = np.array(obj_p).flatten()[:3]

        # ── 3. Robot world position / yaw ─────────────────────────────────────
        qp        = self.agent.robot.get_qpos().cpu().numpy().flatten().copy()
        robot_xy  = np.array([self.root_pos[0] + qp[0], self.root_pos[1] + qp[1]])
        yaw       = qp[2]
        c, s      = math.cos(yaw), math.sin(yaw)

        # ── 4. Arm1 shoulder in world frame ───────────────────────────────────
        # Rotate the shoulder's base-frame offset by the robot's yaw, then translate.
        # (Standard 2-D rotation; the arm is mounted off-centre, so this offset matters.)
        sh_loc    = _ARM1_SHOULDER[:2]   # x,y offset in robot frame
        sh_world  = np.array([
            robot_xy[0] + c * sh_loc[0] - s * sh_loc[1],
            robot_xy[1] + s * sh_loc[0] + c * sh_loc[1],
        ])

        # ── 5. Object relative to shoulder in robot frame ─────────────────────
        # Inverse rotation (note the transposed signs) takes the world-frame delta into the
        # robot's frame, where the IK can reason about "forward" and "left".
        dx_w = obj_world[0] - sh_world[0]
        dy_w = obj_world[1] - sh_world[1]
        dx_r =  dx_w * c + dy_w * s    # forward in robot frame
        dy_r = -dx_w * s + dy_w * c    # left in robot frame
        dz   = obj_world[2] - _ARM1_SHOULDER[2]   # height above the shoulder

        # ── 6. Arm Rotation joint: θ such that [cos θ, -sin θ] ∝ (dx_r, dy_r) ─
        # arm points in [cos θ, -sin θ, 0] in base_link (forward at θ=0, right at θ=π/2)
        # Hence the negated dy_r: the joint's positive direction is to the RIGHT.
        arm_rot = math.atan2(-dy_r, dx_r)
        arm_rot = max(-2.1, min(2.1, arm_rot))    # joint limit

        # ── 7. 2-link IK (reach = horiz dist, height = dz) ───────────────────
        # The IK solves for the WRIST, so subtract the wrist→fingertip length — otherwise
        # the arm reaches until the wrist is at the object and the gripper closes past it.
        reach        = math.sqrt(dx_r * dx_r + dy_r * dy_r)
        reach_wrist  = max(0.0, reach - _ARM_TIP)  # IK targets wrist, not tip
        pitch, elbow = _ik_2link(reach_wrist, dz)
        # pitch - elbow cancels the two joints' rotations, leaving the end effector at
        # roughly world-horizontal regardless of the arm's configuration — a top-down-ish
        # approach that works for objects sitting on surfaces.
        wrist_pitch  = pitch - elbow      # keeps EE roughly horizontal
        print(f"[grasp IK] obj_world={obj_world.round(3)}  "
              f"dx_r={dx_r:.3f} dy_r={dy_r:.3f} dz={dz:.3f}  "
              f"reach={reach:.3f} reach_wrist={reach_wrist:.3f}  "
              f"arm_rot={math.degrees(arm_rot):.1f}°  "
              f"pitch={math.degrees(pitch):.1f}°  elbow={math.degrees(elbow):.1f}°")

        # ── 8. Open gripper, position arm kinematically ───────────────────────
        # These qpos indices are the arm's joints in URDF order. Set directly (no controller),
        # then step physics so the scene — and the object, if the fingers brush it — settles.
        qp[3]  = arm_rot
        qp[6]  = pitch
        qp[9]  = elbow
        qp[11] = wrist_pitch
        qp[13] = 1.57    # Wrist_Roll neutral
        qp[15] = _GRIPPER_OPEN
        self.agent.robot.set_qpos(qp)
        for _ in range(80):     # let the arm's motion resolve before closing
            self._step()

        # ── 9. Close gripper ──────────────────────────────────────────────────
        # Re-read qpos: physics may have moved the arm since step 8.
        qp     = self.agent.robot.get_qpos().cpu().numpy().flatten().copy()
        qp[15] = _GRIPPER_CLOSE
        self.agent.robot.set_qpos(qp)
        for _ in range(40):     # let contact/friction establish the grip
            self._step()

        # True == "we attempted a grasp on a real object", NOT "we are holding it".
        return True, obj_name, dist

    def _release(
        self,
        target: str = "",
        destination: Optional[str] = None,
        target_region: Optional[str] = None,
    ) -> tuple[bool, str]:
        """Open the gripper and let physics take the object.

        No placement logic: destination/target_region are accepted (the signature is shared
        across backends) but ignored here. Whatever is held simply drops from wherever the
        arm currently is, so the agent must position itself over the destination first. The
        20 steps let the object fall and settle before the next observation is taken.
        """
        qp     = self.agent.robot.get_qpos().cpu().numpy().flatten().copy()
        qp[15] = _GRIPPER_OPEN
        self.agent.robot.set_qpos(qp)
        for _ in range(20):
            self._step()
        return True, "gripper opened."

    def _forward_step(self) -> None:
        self._base_move_step("forward")

    def _get_depth_and_intrinsics(
        self, obs: dict
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Extract (depth, K, E) for the depth back-projection in BaseToolbox.

        Two ManiSkill quirks handled here:

        1. There is no depth buffer. The "Position" render target holds each pixel's
           CAMERA-SPACE position, and its Z channel is the depth — but negated, because
           OpenGL cameras look down -Z, so a point 2 m in front has z = -2. Hence the
           unary minus.

        2. extrinsic_cv is sometimes 3×4 (rotation + translation, no homogeneous row).
           The caller does `np.linalg.inv(E)`, which needs a square matrix, so pad it.

        Returns None (with a diagnostic listing the keys we *did* find) rather than raising:
        the caller treats a missing camera as "cannot localise", which is recoverable.
        """
        import torch as _torch
        cam_data = obs.get("sensor_data", {}).get("fetch_head", {})

        pos_raw = cam_data.get("Position")
        if pos_raw is None:
            print(f"[depth] no Position in sensor_data.fetch_head: {list(cam_data.keys())}")
            return None
        pos_np = (
            pos_raw.cpu().numpy() if _torch.is_tensor(pos_raw)
            else np.array(pos_raw)
        ).squeeze()
        depth_np = -pos_np[..., 2]  # negate: OpenGL Z convention (Z < 0 for objects in front)

        cam_param = obs.get("sensor_param", {}).get("fetch_head", {})
        # The _cv variants are in OpenCV convention (x right, y down, z forward), which is
        # what the pinhole un-projection in BaseToolbox assumes.
        K_raw = cam_param.get("intrinsic_cv")
        E_raw = cam_param.get("extrinsic_cv")
        if K_raw is None or E_raw is None:
            print(f"[depth] keys in sensor_param.fetch_head: {list(cam_param.keys())}")
            return None
        K = np.array(K_raw.cpu().numpy() if _torch.is_tensor(K_raw) else K_raw).squeeze()
        E = np.array(E_raw.cpu().numpy() if _torch.is_tensor(E_raw) else E_raw).squeeze()
        if E.shape == (3, 4):
            E = np.vstack([E, [0.0, 0.0, 0.0, 1.0]])   # → 4×4 so it can be inverted
        return depth_np, K, E

    # ── Sim-specific overrides ────────────────────────────────────────────────

    def _snap_to_navmesh(self, xy: np.ndarray) -> np.ndarray:
        """Pull a goal onto the nearest walkable navmesh vertex.

        Overrides the base class's identity implementation. Goals derived from memory poses
        or depth back-projection can land inside furniture; snapping keeps the planner from
        being handed an unreachable target.
        """
        if self.nav_verts is not None and len(self.nav_verts) >= 4:
            dists = np.linalg.norm(self.nav_verts - xy, axis=1)
            return self.nav_verts[np.argmin(dists)].copy()
        return xy

    # ── Sim-specific helpers ──────────────────────────────────────────────────

    def _get_nav_grid(self):
        """Build the NavGrid on first use, then memoise it.

        robot_radius=0.30 is slightly tighter than the 0.35 m default: XLeRobot's footprint
        is small, and over-inflating erodes ReplicaCAD's narrow doorways shut.

        On failure, caches None — so a scene whose navmesh cannot be rasterised does not
        retry (and re-log the failure) on every single navigate call.
        """
        if self._nav_grid is not None:
            return self._nav_grid
        if self.nav_verts is None or len(self.nav_verts) < 4:
            return None
        try:
            from .nav_grid import NavGrid
            self._nav_grid = NavGrid(self.nav_verts, robot_radius=0.30)
        except Exception as e:
            print(f"[AgentToolbox] NavGrid build failed: {e}")
            self._nav_grid = None
        return self._nav_grid
