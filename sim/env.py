"""
env.py — ReplicaCAD environment setup.

Contains:
- XLeRobot / ManiSkill path injection
- ReplicaCADSceneBuilder monkey-patches (staging scenes + XLeRobot support)
- Navigation & manipulation tuning constants
- Spawn helpers (_load_replicacad_nav_positions, find_clear_spawn)
- Robot-state helpers (get_robot_xy, get_robot_yaw, get_actor_xy, find_objects)
- Manipulation helpers (find_objects, get_actor_xy)
"""

import math
import os
import sys

import numpy as np
import torch
import trimesh

# ── ManiSkill / SAPIEN / XLeRobot ────────────────────────────────────────────
XLER_MANISKILL_DIR = os.path.expanduser("~/Projects/XLeRobot/simulation/Maniskill")
if XLER_MANISKILL_DIR not in sys.path:
    sys.path.insert(0, XLER_MANISKILL_DIR)

import sapien
import gymnasium as gym
import agents.xlerobot.xlerobot  # noqa: registers xlerobot

from mani_skill.utils.scene_builder.replicacad.scene_builder import ReplicaCADSceneBuilder
from mani_skill.utils.structs import Articulation as MSArticulation
from envs.scenes.base_env import SceneManipulationEnv  # noqa: registers env

# ── patch: expose all 90 ReplicaCAD staging scenes ───────────────────────────
_orig_replicacad_init = ReplicaCADSceneBuilder.__init__


def _replicacad_patched_init(self, env, **kwargs):
    kwargs.setdefault("include_staging_scenes", True)
    _orig_replicacad_init(self, env, **kwargs)


ReplicaCADSceneBuilder.__init__ = _replicacad_patched_init


# ── patch: XLeRobot support in initialize() ───────────────────────────────────
def _replicacad_patched_initialize(self, env_idx):
    self.env.agent.robot.set_pose(sapien.Pose([-10, 0, -100]))
    for obj, pose in self._default_object_poses:
        obj.set_pose(pose)
        if isinstance(obj, MSArticulation):
            obj.set_qpos(obj.qpos[0] * 0)
            obj.set_qvel(obj.qvel[0] * 0)
    self.env.agent.reset(self.env.agent.keyframes["rest"].qpos)
    # Keep robot outside; main() relocates it to nav-mesh centroid
    self.env.agent.robot.set_pose(sapien.Pose([-10, 0, -100]))


ReplicaCADSceneBuilder.initialize = _replicacad_patched_initialize


# ── Navigation tuning ─────────────────────────────────────────────────────────
NAV_STOP_DIST              = 0.50   # metres — coarse-nav stop for object_type mode
RETRIEVAL_COARSE_STOP_DIST = 1.50   # metres — coarse-nav stop for retrieval mode
NAV_MAX_STEPS              = 6000
NAV_FWD_M_PER_STEP         = 0.015  # metres moved forward per env.step (kinematic)
NAV_ROT_RAD_PER_STEP       = 0.04   # radians rotated per env.step (kinematic)
ROT_THRESH                 = math.radians(8)   # align first if bearing > this
WARMUP_STEPS               = 60     # physics settle steps after spawn

# ── Manipulation tuning ───────────────────────────────────────────────────────
ARM_REACH = 0.60   # metres — last-mile stop distance for approach


# ── Spawn helpers ─────────────────────────────────────────────────────────────

def _load_replicacad_nav_positions(uenv) -> np.ndarray | None:
    """Load navigable XY positions from the scene's OBJ navmesh file."""
    try:
        import os.path as osp
        from mani_skill import ASSET_DIR
        sb = getattr(uenv, "scene_builder", None)
        if sb is None or sb.build_config_idxs is None:
            return None
        bci = sb.build_config_idxs[0]
        scenes_dir = osp.join(ASSET_DIR,
                              "scene_datasets/replica_cad_dataset/configs/scenes")
        nav_filename = sb.build_configs[bci].replace(
            ".scene_instance.json",
            ".scene_instance.fetch.navigable_positions.obj")
        nav_path = osp.join(scenes_dir, nav_filename)
        if not osp.exists(nav_path):
            return None
        verts = np.asarray(trimesh.load(nav_path).vertices, dtype=np.float32)
        return verts[:, :2]
    except Exception:
        return None


def find_clear_spawn(uenv, nav_verts: np.ndarray | None = None) -> np.ndarray:
    """Return a spawn XY near the navmesh centroid (or obstacle centroid fallback)."""
    if nav_verts is None:
        nav_verts = _load_replicacad_nav_positions(uenv)
    if nav_verts is not None and len(nav_verts) >= 2:
        centroid = nav_verts.mean(axis=0)
        return nav_verts[np.argmin(np.linalg.norm(nav_verts - centroid, axis=1))]
    xys = []
    sb = getattr(uenv, "scene_builder", None)
    if sb:
        for actor in sb.scene_objects.values():
            try:
                p = actor.pose.p
                if torch.is_tensor(p):
                    p = p.cpu().numpy()
                xys.append(np.array(p).flatten()[:2])
            except Exception:
                pass
    if xys:
        pts = np.stack(xys)
        return (pts.min(0) + pts.max(0)) / 2.0
    return np.zeros(2)


# ── Robot-state helpers ───────────────────────────────────────────────────────

def get_robot_xy(agent, root_pos: np.ndarray) -> np.ndarray:
    qp = agent.robot.get_qpos().cpu().numpy().flatten()
    return root_pos + qp[:2]


def get_robot_yaw(agent) -> float:
    return float(agent.robot.get_qpos().cpu().numpy().flatten()[2])


def get_actor_xy(actor) -> np.ndarray:
    pose = actor.pose if hasattr(actor, "pose") else actor.get_pose()
    p = pose.p
    if torch.is_tensor(p):
        p = p.cpu().numpy()
    return np.array(p).flatten()[:2]


def find_objects(scene_builder, query: str) -> list[tuple[str, object]]:
    """Return all scene objects whose name contains `query` (case-insensitive)."""
    q = query.lower()
    return [(k, v) for k, v in scene_builder.scene_objects.items()
            if q in k.lower()]


