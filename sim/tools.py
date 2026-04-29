"""
sim/tools.py — AgentToolbox: ManiSkill/ReplicaCAD implementation of the agent toolbox.

Implements the ten abstract primitives of BaseToolbox for ManiSkill/SAPIEN.
All shared tool logic (navigate, approach, manipulate, inspect, …) lives in BaseToolbox.
Ground-truth simulator lookups are used ONLY in the primitives below, never
surfaced in ToolResult summaries that reach the Gemini policy prompt.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Optional

import numpy as np

from ..agent.toolbox_base import BaseToolbox

# ── Arm geometry constants (from xlerobot.urdf) ───────────────────────────────
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
    """
    t1_off = math.atan2(0.028, 0.11257)
    t2_off = math.atan2(0.0052, 0.1349) + t1_off
    r = math.sqrt(x * x + y * y)
    r_max = _ARM_L1 + _ARM_L2
    r_min = abs(_ARM_L1 - _ARM_L2)
    if r > r_max:
        s = r_max / r; x *= s; y *= s; r = r_max
    elif 0 < r < r_min:
        s = r_min / r; x *= s; y *= s; r = r_min
    cos_t2 = max(-1.0, min(1.0, -(r * r - _ARM_L1 * _ARM_L1 - _ARM_L2 * _ARM_L2)
                           / (2 * _ARM_L1 * _ARM_L2)))
    t2 = math.pi - math.acos(cos_t2)
    beta  = math.atan2(y, x)
    gamma = math.atan2(_ARM_L2 * math.sin(t2), _ARM_L1 + _ARM_L2 * math.cos(t2))
    t1    = beta + gamma
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
        self._action_zeros = np.zeros(action_shape, dtype=np.float32)
        self.scene_builder = scene_builder
        self.nav_verts     = nav_verts

        self._capture_worker   = capture_worker
        self._capture_interval = capture_interval
        self._last_bg_capture_t: float = -1e9

        # Lazy NavGrid — built on first navigate/approach call
        self._nav_grid = None

    # ── Abstract primitive implementations ───────────────────────────────────

    def _step(self) -> dict:
        obs, _, _, _, _ = self.env.step(self._action_zeros)
        self.env.render()
        self._bg_capture(obs)
        return obs

    def _bg_capture(self, obs: dict) -> None:
        """Capture a frame into the scan index at the configured interval."""
        if self._capture_worker is None:
            return
        now = time.time()
        if now - self._last_bg_capture_t < self._capture_interval:
            return
        from .capture import _capture_nav_frame
        from .env import get_robot_xy, get_robot_yaw
        from ..agent.episodic_memory import frame_to_memory_id
        rgb = _capture_nav_frame(obs)
        if rgb is None:
            return
        curr_xy  = get_robot_xy(self.agent, self.root_pos)
        curr_yaw = get_robot_yaw(self.agent)
        fidx  = self._capture_worker.enqueue(rgb, curr_xy, curr_yaw)
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
        from .env import get_robot_xy, get_robot_yaw
        xy  = get_robot_xy(self.agent, self.root_pos)
        yaw = get_robot_yaw(self.agent)
        return [float(xy[0]), float(xy[1]), float(yaw)]

    def _navigate_step(self, bearing: float) -> None:
        from .nav_grid import kinematic_nav_step
        kinematic_nav_step(self.agent, bearing)

    def _plan_path(
        self, start_xy: np.ndarray, goal_xy: np.ndarray
    ) -> list[np.ndarray]:
        nav_grid = self._get_nav_grid()
        if nav_grid is not None:
            return nav_grid.plan(start_xy, goal_xy)
        return [goal_xy.copy()]

    def _grasp(self, target: str) -> tuple[bool, str, float]:
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
            for name, actor in self.scene_builder.scene_objects.items():
                try:
                    d = float(np.linalg.norm(get_actor_xy(actor) - curr_xy))
                    candidates.append((d, name, actor))
                except Exception:
                    pass
        candidates.sort(key=lambda x: x[0])
        if not candidates:
            return False, "", 999.0
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
        sh_loc    = _ARM1_SHOULDER[:2]   # x,y offset in robot frame
        sh_world  = np.array([
            robot_xy[0] + c * sh_loc[0] - s * sh_loc[1],
            robot_xy[1] + s * sh_loc[0] + c * sh_loc[1],
        ])

        # ── 5. Object relative to shoulder in robot frame ─────────────────────
        dx_w = obj_world[0] - sh_world[0]
        dy_w = obj_world[1] - sh_world[1]
        dx_r =  dx_w * c + dy_w * s    # forward in robot frame
        dy_r = -dx_w * s + dy_w * c    # left in robot frame
        dz   = obj_world[2] - _ARM1_SHOULDER[2]

        # ── 6. Arm Rotation joint: θ such that [cos θ, -sin θ] ∝ (dx_r, dy_r) ─
        # arm points in [cos θ, -sin θ, 0] in base_link (forward at θ=0, right at θ=π/2)
        arm_rot = math.atan2(-dy_r, dx_r)
        arm_rot = max(-2.1, min(2.1, arm_rot))

        # ── 7. 2-link IK (reach = horiz dist, height = dz) ───────────────────
        reach        = math.sqrt(dx_r * dx_r + dy_r * dy_r)
        reach_wrist  = max(0.0, reach - _ARM_TIP)  # IK targets wrist, not tip
        pitch, elbow = _ik_2link(reach_wrist, dz)
        wrist_pitch  = pitch - elbow      # keeps EE roughly horizontal
        print(f"[grasp IK] obj_world={obj_world.round(3)}  "
              f"dx_r={dx_r:.3f} dy_r={dy_r:.3f} dz={dz:.3f}  "
              f"reach={reach:.3f} reach_wrist={reach_wrist:.3f}  "
              f"arm_rot={math.degrees(arm_rot):.1f}°  "
              f"pitch={math.degrees(pitch):.1f}°  elbow={math.degrees(elbow):.1f}°")

        # ── 8. Open gripper, position arm kinematically ───────────────────────
        qp[3]  = arm_rot
        qp[6]  = pitch
        qp[9]  = elbow
        qp[11] = wrist_pitch
        qp[13] = 1.57    # Wrist_Roll neutral
        qp[15] = _GRIPPER_OPEN
        self.agent.robot.set_qpos(qp)
        for _ in range(80):
            self._step()

        # ── 9. Close gripper ──────────────────────────────────────────────────
        qp     = self.agent.robot.get_qpos().cpu().numpy().flatten().copy()
        qp[15] = _GRIPPER_CLOSE
        self.agent.robot.set_qpos(qp)
        for _ in range(40):
            self._step()

        return True, obj_name, dist

    def _release(self) -> None:
        qp     = self.agent.robot.get_qpos().cpu().numpy().flatten().copy()
        qp[15] = _GRIPPER_OPEN
        self.agent.robot.set_qpos(qp)
        for _ in range(20):
            self._step()

    def _forward_step(self) -> None:
        qp  = self.agent.robot.get_qpos().cpu().numpy().flatten().copy()
        yaw = qp[2]
        qp[0] += 0.015 * math.cos(yaw)
        qp[1] += 0.015 * math.sin(yaw)
        self.agent.robot.set_qpos(qp)

    def _get_depth_and_intrinsics(
        self, obs: dict
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
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
        K_raw = cam_param.get("intrinsic_cv")
        E_raw = cam_param.get("extrinsic_cv")
        if K_raw is None or E_raw is None:
            print(f"[depth] keys in sensor_param.fetch_head: {list(cam_param.keys())}")
            return None
        K = np.array(K_raw.cpu().numpy() if _torch.is_tensor(K_raw) else K_raw).squeeze()
        E = np.array(E_raw.cpu().numpy() if _torch.is_tensor(E_raw) else E_raw).squeeze()
        if E.shape == (3, 4):
            E = np.vstack([E, [0.0, 0.0, 0.0, 1.0]])
        return depth_np, K, E

    # ── Sim-specific overrides ────────────────────────────────────────────────

    def _snap_to_navmesh(self, xy: np.ndarray) -> np.ndarray:
        if self.nav_verts is not None and len(self.nav_verts) >= 4:
            dists = np.linalg.norm(self.nav_verts - xy, axis=1)
            return self.nav_verts[np.argmin(dists)].copy()
        return xy


    # ── Sim-specific helpers ──────────────────────────────────────────────────

    def _get_nav_grid(self):
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

