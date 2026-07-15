"""
env.py — ReplicaCAD environment setup.

Contains:
- XLeRobot / ManiSkill path injection
- ReplicaCADSceneBuilder monkey-patches (staging scenes + XLeRobot support)
- Navigation & manipulation tuning constants
- Spawn helpers (_load_replicacad_nav_positions, find_clear_spawn)
- Robot-state helpers (get_robot_xy, get_robot_yaw, get_actor_xy, find_objects)
- Manipulation helpers (find_objects, get_actor_xy)

This is the ManiSkill/SAPIEN + ReplicaCAD backend (as distinct from sim/habitat_toolbox.py,
which is the Habitat one). Importing this module has SIDE EFFECTS by design: it puts
XLeRobot on sys.path, registers the robot and env with gym, and monkey-patches
ReplicaCADSceneBuilder. It must therefore be imported before any ReplicaCAD scene is built.
"""

import math
import os
import sys

import numpy as np
import torch
import trimesh

# ── ManiSkill / SAPIEN / XLeRobot ────────────────────────────────────────────
# XLeRobot is a sibling checkout, not a pip-installed package, so its ManiSkill directory
# has to go on sys.path before the imports below can resolve. Prepended (not appended) so
# its `agents`/`envs` packages win over any same-named installed module.
XLER_MANISKILL_DIR = os.path.expanduser("~/Projects/XLeRobot/simulation/Maniskill")
if XLER_MANISKILL_DIR not in sys.path:
    sys.path.insert(0, XLER_MANISKILL_DIR)

import sapien
import gymnasium as gym
# Import-for-side-effect: these modules register the robot and the environment with
# ManiSkill's global registries. Nothing here is referenced by name, hence the noqa.
import agents.xlerobot.xlerobot  # noqa: registers xlerobot

from mani_skill.utils.scene_builder.replicacad.scene_builder import ReplicaCADSceneBuilder
from mani_skill.utils.structs import Articulation as MSArticulation
from envs.scenes.base_env import SceneManipulationEnv  # noqa: registers env

# ── patch: expose all 90 ReplicaCAD staging scenes ───────────────────────────
# ManiSkill defaults to the ~5 "apartment" scenes and hides the 90 staging variants.
# We want the full set, so wrap __init__ to flip the flag. Done as a wrapper (rather than
# passing the kwarg at every call site) because the scene builder is constructed deep inside
# ManiSkill's env setup, where we have no hook.
_orig_replicacad_init = ReplicaCADSceneBuilder.__init__


def _replicacad_patched_init(self, env, **kwargs):
    # setdefault, not assignment — an explicit caller preference still wins.
    kwargs.setdefault("include_staging_scenes", True)
    _orig_replicacad_init(self, env, **kwargs)


ReplicaCADSceneBuilder.__init__ = _replicacad_patched_init


# ── patch: XLeRobot support in initialize() ───────────────────────────────────
def _replicacad_patched_initialize(self, env_idx):
    """Replaces ReplicaCADSceneBuilder.initialize wholesale (not a wrapper).

    The stock implementation hard-codes Fetch/Panda-specific spawn logic that does not
    apply to XLeRobot. This version does the parts that are robot-agnostic — reset every
    object to its authored pose, zero out articulation joint state, put the arm in its rest
    keyframe — and deliberately omits robot placement.

    The robot is parked far outside the scene (-10, 0, -100) both before and after object
    placement. Before, so it cannot collide with or displace objects as they are being
    positioned; after, so it stays out of the way until the caller relocates it to a real
    spawn point (see find_clear_spawn). The scene builder simply is not the right place to
    decide where this robot starts.
    """
    self.env.agent.robot.set_pose(sapien.Pose([-10, 0, -100]))
    for obj, pose in self._default_object_poses:
        obj.set_pose(pose)
        if isinstance(obj, MSArticulation):
            # Zero joint positions and velocities, so drawers/doors start closed and still.
            # `qpos[0] * 0` produces a correctly-shaped zero tensor without hard-coding DOF.
            obj.set_qpos(obj.qpos[0] * 0)
            obj.set_qvel(obj.qvel[0] * 0)
    self.env.agent.reset(self.env.agent.keyframes["rest"].qpos)
    # Keep robot outside; main() relocates it to nav-mesh centroid
    self.env.agent.robot.set_pose(sapien.Pose([-10, 0, -100]))


ReplicaCADSceneBuilder.initialize = _replicacad_patched_initialize


# ── Navigation tuning ─────────────────────────────────────────────────────────
# The base is driven KINEMATICALLY, not by torque: each env.step() teleports it a fixed
# increment. The two _PER_STEP constants below are therefore the robot's actual speed, and
# the *_MAX_STEPS budgets have to be read against them (6000 steps × 0.015 m ≈ 90 m of
# travel, generously more than any single ReplicaCAD apartment traverse).
NAV_STOP_DIST              = 0.50   # metres — coarse-nav stop for object_type mode
# Retrieval mode stops further out (1.5 m): the goal there is a viewpoint from which the
# target is *visible*, not a pose to manipulate from — and stopping closer tends to put the
# robot's nose in the furniture.
RETRIEVAL_COARSE_STOP_DIST = 1.50   # metres — coarse-nav stop for retrieval mode
NAV_MAX_STEPS              = 6000
NAV_FWD_M_PER_STEP         = 0.015  # metres moved forward per env.step (kinematic)
NAV_ROT_RAD_PER_STEP       = 0.04   # radians rotated per env.step (kinematic)
# Turn-then-drive: if the goal is more than 8° off the nose, rotate in place first rather
# than arcing toward it. Keeps paths tight and predictable in cluttered rooms.
ROT_THRESH                 = math.radians(8)   # align first if bearing > this
# The robot is dropped in rather than placed exactly on the floor; these steps let physics
# resolve that (and any object interpenetration) before the episode starts.
WARMUP_STEPS               = 60     # physics settle steps after spawn

# ── Manipulation tuning ───────────────────────────────────────────────────────
ARM_REACH = 0.60   # metres — last-mile stop distance for approach


# ── Spawn helpers ─────────────────────────────────────────────────────────────

def _load_replicacad_nav_positions(uenv) -> np.ndarray | None:
    """Load navigable XY positions from the scene's OBJ navmesh file.

    ReplicaCAD ships a precomputed navmesh per scene as an OBJ whose vertices are walkable
    points, named after the scene config. (The "fetch" in the filename refers to the robot
    the mesh was generated for — its footprint is close enough to XLeRobot's to reuse.)

    Returns None on any failure — a missing navmesh is normal for some staging scenes, and
    callers fall back to a geometric heuristic.
    """
    try:
        import os.path as osp
        from mani_skill import ASSET_DIR
        sb = getattr(uenv, "scene_builder", None)
        if sb is None or sb.build_config_idxs is None:
            return None
        bci = sb.build_config_idxs[0]   # single-env: only one scene is built
        scenes_dir = osp.join(ASSET_DIR,
                              "scene_datasets/replica_cad_dataset/configs/scenes")
        # The navmesh filename is derived from the scene config filename by substitution.
        nav_filename = sb.build_configs[bci].replace(
            ".scene_instance.json",
            ".scene_instance.fetch.navigable_positions.obj")
        nav_path = osp.join(scenes_dir, nav_filename)
        if not osp.exists(nav_path):
            return None
        verts = np.asarray(trimesh.load(nav_path).vertices, dtype=np.float32)
        return verts[:, :2]   # drop Z: navigation is planar
    except Exception:
        return None


def find_clear_spawn(uenv, nav_verts: np.ndarray | None = None) -> np.ndarray:
    """Return a spawn XY near the navmesh centroid (or obstacle centroid fallback).

    Preferred path: take the mean of all walkable points and snap to the nearest *actual*
    navmesh vertex. The snap matters — a room shaped like an L has a centroid that lands
    inside a wall, and spawning there would wedge the robot in geometry.

    Fallback (no navmesh): the centre of the bounding box of all scene objects. Crude, and
    it can land inside furniture, but it is at least inside the apartment.
    """
    if nav_verts is None:
        nav_verts = _load_replicacad_nav_positions(uenv)
    if nav_verts is not None and len(nav_verts) >= 2:
        centroid = nav_verts.mean(axis=0)
        # Snap to the nearest walkable vertex — see docstring.
        return nav_verts[np.argmin(np.linalg.norm(nav_verts - centroid, axis=1))]

    # ── Fallback: bounding-box centre of every scene object ───────────────────
    xys = []
    sb = getattr(uenv, "scene_builder", None)
    if sb:
        for actor in sb.scene_objects.values():
            try:
                p = actor.pose.p
                if torch.is_tensor(p):
                    p = p.cpu().numpy()   # ManiSkill poses are batched torch tensors
                xys.append(np.array(p).flatten()[:2])
            except Exception:
                pass   # skip anything without a readable pose
    if xys:
        pts = np.stack(xys)
        # Midpoint of the AABB, not the mean — the mean is dragged toward whichever room
        # happens to contain the most clutter.
        return (pts.min(0) + pts.max(0)) / 2.0
    return np.zeros(2)   # empty scene; world origin is as good a guess as any


# ── Robot-state helpers ───────────────────────────────────────────────────────

def get_robot_xy(agent, root_pos: np.ndarray) -> np.ndarray:
    """World XY of the base.

    XLeRobot's base is modelled as two prismatic joints plus a revolute one, so qpos[:2] is
    a base *displacement* in the robot's spawn frame, not a world position. Adding root_pos
    (where the robot was spawned) recovers world coordinates.
    """
    qp = agent.robot.get_qpos().cpu().numpy().flatten()
    return root_pos + qp[:2]


def get_robot_yaw(agent) -> float:
    """Base heading in radians — qpos[2] is the base's revolute joint (see get_robot_xy)."""
    return float(agent.robot.get_qpos().cpu().numpy().flatten()[2])


def get_actor_xy(actor) -> np.ndarray:
    """World XY of any scene actor, tolerating both the property and getter pose APIs."""
    pose = actor.pose if hasattr(actor, "pose") else actor.get_pose()
    p = pose.p
    if torch.is_tensor(p):
        p = p.cpu().numpy()
    # flatten() collapses ManiSkill's leading batch dimension (1, 3) → (3,).
    return np.array(p).flatten()[:2]


def find_objects(scene_builder, query: str) -> list[tuple[str, object]]:
    """Return all scene objects whose name contains `query` (case-insensitive).

    Substring rather than exact match, because ReplicaCAD names are of the form
    "frl_apartment_bowl_01_:0000" — a search for "bowl" has to hit that.

    NOTE this reads simulator ground truth (authored object names). It is used by the
    ManiSkill oracle/debug paths, and must not feed anything the agent's memory records —
    see the memory policy in agent/schemas.py.
    """
    q = query.lower()
    return [(k, v) for k, v in scene_builder.scene_objects.items()
            if q in k.lower()]
