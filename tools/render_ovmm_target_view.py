#!/usr/bin/env python3
"""Render a hero shot of an OVMM episode's TARGET OBJECT, in its scene, from a viewpoint
that actually shows it.

Answers the question "what am I even asking the robot to find?" — an OVMM episode names a
target object by config handle, but the only way to know what it looks like *in situ* is to
put a camera on it. Useful for sanity-checking an episode before running it, and for
debugging failures where the target turns out to be inside a cupboard.

Standalone: it drives habitat_sim DIRECTLY (no habitat-lab env, no robot, no task). It builds
a bare scene, places the episode's rigid objects itself, and flies a camera around.

The trick that makes it reliable is the semantic-ID search in render(): rather than guessing a
good camera pose, it tries several, tags the target with a unique semantic id, and picks the
pose whose semantic image contains the most target pixels — i.e. the view where the object is
genuinely most visible, occlusion included.

Usage:
    python tools/render_ovmm_target_view.py --split minival --episode-id 2
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
# The vendored habitat-lab fork is not pip-installed, so put its packages on sys.path.
HAB_LAB_ROOT = ROOT / "OWMM-Agent" / "sim" / "habitat-lab"
for rel in ("habitat-lab", "habitat-baselines", "habitat-mas"):
    p = HAB_LAB_ROOT / rel
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load_episode(split: str, episode_id: int) -> dict:
    """Find one episode by id in the CONVERTED split (see tools/convert_ovmm_episodes.py).

    Reads our own output, not the raw OVMM download — so rigid_objs already carries inline
    4×4 matrices rather than indices into a side-car .npy.
    """
    with gzip.open(ROOT / "data" / "datasets" / "ovmm" / f"{split}.json.gz") as f:
        data = json.load(f)
    for ep in data["episodes"]:
        if int(ep["episode_id"]) == episode_id:
            return ep
    raise ValueError(f"episode {episode_id} not found in split {split!r}")


def _matrix_translation(mat: list[list[float]]) -> np.ndarray:
    """Pull the position out of a 4×4 pose — the last column of the top 3 rows."""
    arr = np.array(mat, dtype=np.float32)
    return arr[:3, 3]


def _quat_from_rotation_matrix(rot: np.ndarray):
    """Rotation matrix → quaternion (Shepperd's method).

    Four branches, one per component that can be largest. The naive single-formula conversion
    divides by sqrt(1 + trace), which goes to zero for a 180° rotation and loses all
    precision near it; picking the branch whose denominator is largest keeps the result
    numerically stable for every possible rotation.

    Habitat wants an agent's rotation as a quaternion, so the look-at basis built below has to
    come through here.
    """
    import magnum as mn

    trace = float(rot[0, 0] + rot[1, 1] + rot[2, 2])
    # Branch 1: trace is positive — the safe, common case.
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    # Branches 2-4: trace <= 0, so use whichever diagonal element is largest.
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
    return mn.Quaternion(mn.Vector3(x, y, z), w).normalized()


def _look_rotation(source: np.ndarray, target: np.ndarray):
    """Camera rotation (as a quaternion) that points `source` at `target`.

    Builds an orthonormal basis by Gram-Schmidt: forward is fixed by the two points, right is
    forward × world-up, and true up is right × forward.
    """
    forward = target - source
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)   # Habitat is Y-up
    right = np.cross(forward, world_up)
    # Degenerate case: looking straight up or down makes forward parallel to world_up, so
    # their cross product vanishes and "right" is undefined. Pick a different up hint.
    if np.linalg.norm(right) < 1e-5:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # Habitat camera forward is local -Z.
    # Hence the NEGATED forward in the third column: the basis must map the camera's local
    # -Z onto the direction we want to look. Getting this sign wrong points the camera
    # exactly backwards.
    rot = np.column_stack([right, up, -forward])
    quat = _quat_from_rotation_matrix(rot)
    # Habitat wants [x, y, z, w] — scalar LAST, which is the opposite of magnum's own order.
    return [quat.vector.x, quat.vector.y, quat.vector.z, quat.scalar]


def _find_object_template(template_mgr, handle: str) -> str:
    """Resolve an object config name to a loaded template handle (substring match)."""
    matches = template_mgr.get_template_handles(handle)
    if not matches:
        raise RuntimeError(f"object template not loaded: {handle}")
    return matches[0]


def _add_episode_objects(
    sim,
    ep: dict,
    target_prefix: str,
    target_semantic_id: int,
) -> int | None:
    """Place every rigid object the episode specifies, and tag the target one.

    Because this bypasses habitat-lab entirely, the bare simulator loads only the static
    scene — none of the episode's *objects* exist until we add them here. Returns the target's
    object id (or None if it could not be placed).
    """
    import magnum as mn
    import habitat_sim

    template_mgr = sim.get_object_template_manager()
    rigid_mgr = sim.get_rigid_object_manager()
    # Object templates must be loaded before they can be instantiated by handle.
    for obj_dir in ep["additional_obj_config_paths"]:
        abs_dir = ROOT / obj_dir
        if abs_dir.is_dir():
            template_mgr.load_configs(str(abs_dir))

    target_id = None
    for cfg_name, transform in ep["rigid_objs"]:
        try:
            template_handle = _find_object_template(template_mgr, cfg_name)
            obj = rigid_mgr.add_object_by_template_handle(template_handle)
        except Exception as exc:
            # One unplaceable prop shouldn't sink the render — the target is what matters.
            print(f"[warn] skipping {cfg_name}: {exc}", flush=True)
            continue

        # The 4×4 was inlined by convert_ovmm_episodes.py, so it can be applied directly.
        mat = np.array(transform, dtype=np.float32)
        obj.transformation = mn.Matrix4(mat)
        if cfg_name.startswith(target_prefix):
            target_id = obj.object_id
            # THE KEY LINE. Painting the target with a unique, otherwise-unused semantic id
            # (default 1337) means the semantic camera renders it — and only it — with that
            # value. Counting those pixels is then an exact, occlusion-aware measure of how
            # visible the target is from a given viewpoint, which is what the camera search
            # in render() maximises.
            obj.semantic_id = target_semantic_id
            # KINEMATIC: freeze it at its authored pose. Dynamic, it would fall or settle
            # before we photographed it, and we want the episode's actual start state.
            obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    return target_id


def render(args: argparse.Namespace) -> Path:
    """Build the scene, sweep candidate cameras, save the most-visible view."""
    # Silence habitat's very chatty asset loader (only if the caller hasn't set a preference).
    if "MAGNUM_LOG" not in os.environ:
        os.environ["MAGNUM_LOG"] = "quiet"
    if "HABITAT_SIM_LOG" not in os.environ:
        os.environ["HABITAT_SIM_LOG"] = "quiet"

    import habitat_sim
    from habitat_sim.utils.common import d3_40_colors_rgb

    ep = _load_episode(args.split, args.episode_id)
    target_key = next(iter(ep["targets"]))     # first (usually only) target
    target_prefix = target_key.split("_:")[0]  # strip the instance suffix → config name
    target_goal = _matrix_translation(ep["targets"][target_key])
    # We want where the object STARTS (its rigid_objs pose), not where it must END UP (the
    # goal). Fall back to the goal only if the start pose can't be found.
    rigid_pose = next((x[1] for x in ep["rigid_objs"] if x[0].startswith(target_prefix)), None)
    target_pos = _matrix_translation(rigid_pose) if rigid_pose is not None else target_goal
    # Aim slightly above the object's origin — that origin is often at its base, so looking
    # straight at it frames the surface it sits on rather than the object.
    look_at = target_pos + np.array([0.0, args.look_height, 0.0], dtype=np.float32)

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(ROOT / ep["scene_id"])
    sim_cfg.scene_dataset_config_file = str(ROOT / ep["scene_dataset_config"])
    sim_cfg.enable_physics = True
    sim_cfg.gpu_device_id = args.gpu_id

    # Two co-located cameras: one RGB (the picture we save) and one SEMANTIC (the pixel count
    # that scores each viewpoint). Same resolution, same hfov, same position — so a pixel in
    # one corresponds exactly to a pixel in the other, which is what makes the score valid.
    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "color"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    rgb_spec.resolution = [args.height, args.width]
    rgb_spec.position = [0.0, 0.0, 0.0]
    rgb_spec.hfov = args.hfov

    sem_spec = habitat_sim.CameraSensorSpec()
    sem_spec.uuid = "semantic"
    sem_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    sem_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sem_spec.resolution = [args.height, args.width]
    sem_spec.position = [0.0, 0.0, 0.0]
    sem_spec.hfov = args.hfov

    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec, sem_spec]

    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    try:
        target_obj_id = _add_episode_objects(
            sim,
            ep,
            target_prefix,
            args.target_semantic_id,
        )

        # Candidate viewpoints, all relative to the target and all looking at it. The object
        # will be against a wall or inside a cupboard in some direction, so we try several:
        # four from the cardinal directions at eye-ish height (±X, ±Z, 1.2 m out, 0.55 m up)
        # and two raised diagonals that look down over any low obstruction.
        candidates = [
            target_pos + np.array([1.2, 0.55, 0.0], dtype=np.float32),
            target_pos + np.array([-1.2, 0.55, 0.0], dtype=np.float32),
            target_pos + np.array([0.0, 0.55, 1.2], dtype=np.float32),
            target_pos + np.array([0.0, 0.55, -1.2], dtype=np.float32),
            target_pos + np.array([1.0, 0.85, 1.0], dtype=np.float32),
            target_pos + np.array([-1.0, 0.85, -1.0], dtype=np.float32),
        ]
        # An explicit --camera-position is prepended, not substituted: it competes with the
        # automatic candidates and only wins if it genuinely shows the target better.
        if args.camera_position:
            candidates.insert(0, np.array(args.camera_position, dtype=np.float32))

        # ── The search: pick the viewpoint that SEES the most of the target ───
        # Counting semantic pixels is an honest visibility measure — it accounts for
        # occlusion, distance and framing all at once, which no geometric heuristic would.
        best_obs = None
        best_score = -1
        best_camera = None
        agent = sim.initialize_agent(0)
        for cam_pos in candidates:
            state = habitat_sim.AgentState()
            state.position = cam_pos
            state.rotation = _look_rotation(cam_pos, look_at)
            agent.set_state(state)
            obs = sim.get_sensor_observations()
            semantic = np.asarray(obs.get("semantic"))
            score = 0
            if target_obj_id is not None and semantic.size:
                score = int(np.count_nonzero(semantic == args.target_semantic_id))
            # Strict >, so ties keep the earlier (higher-priority) candidate. Note a score of
            # 0 still beats the -1 initial value, so we always return *some* image even when
            # the target is invisible from every candidate — and the saved metadata's
            # target_semantic_pixels=0 makes that plain.
            if score > best_score:
                best_score = score
                best_obs = obs
                best_camera = cam_pos

        if best_obs is None:
            raise RuntimeError("no camera observations returned")

        # Normalise the render output: strip alpha, and scale float [0,1] to uint8 [0,255].
        rgb = np.asarray(best_obs["color"])
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        if rgb.dtype != np.uint8:
            rgb = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)

        out = Path(args.output)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(out)

        # Sidecar JSON beside the PNG. target_semantic_pixels is the one to read: 0 means the
        # target was not visible from ANY candidate (likely inside a closed container), and
        # the saved image, whatever it shows, is not showing you the object.
        meta = {
            "split": args.split,
            "episode_id": args.episode_id,
            "target_key": target_key,
            "target_object_config": f"{target_prefix}.object_config.json",
            "target_object_position": target_pos.tolist(),
            "target_goal_position": target_goal.tolist(),
            "camera_position": best_camera.tolist() if best_camera is not None else None,
            "target_semantic_pixels": best_score,
            "target_semantic_id": args.target_semantic_id,
        }
        out.with_suffix(".json").write_text(json.dumps(meta, indent=2))

        # Optional: dump the semantic image too, false-coloured so it is human-readable.
        # Handy for confirming the target really is the blob you think it is.
        if args.semantic_output:
            sem = np.asarray(best_obs["semantic"], dtype=np.int32)
            # mod 40 wraps arbitrary semantic ids (incl. 1337) into the 40-colour palette.
            sem_rgb = d3_40_colors_rgb[np.mod(sem, 40)]
            sem_out = Path(args.semantic_output)
            if not sem_out.is_absolute():
                sem_out = ROOT / sem_out
            sem_out.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(sem_rgb.astype(np.uint8)).save(sem_out)

        print(f"saved {out}")
        print(json.dumps(meta, indent=2))
        return out
    finally:
        sim.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="minival")
    parser.add_argument("--episode-id", type=int, default=2)
    parser.add_argument("--output", default="runs/ovmm_target_views/minival_ep0002_battery_charger.png")
    parser.add_argument("--semantic-output", default="")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--hfov", type=float, default=70.0)
    parser.add_argument("--look-height", type=float, default=0.1,
                        help="Aim this far above the object's origin (its origin is usually "
                             "at its base).")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--camera-position", nargs=3, type=float,
                        help="Extra candidate viewpoint; competes with the automatic ones "
                             "rather than replacing them.")
    parser.add_argument("--target-semantic-id", type=int, default=1337,
                        help="Arbitrary sentinel painted onto the target so it can be "
                             "counted in the semantic image. Any value no scene object "
                             "already uses will do.")
    render(parser.parse_args())


if __name__ == "__main__":
    main()
