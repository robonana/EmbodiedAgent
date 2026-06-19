#!/usr/bin/env python3
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
HAB_LAB_ROOT = ROOT / "OWMM-Agent" / "sim" / "habitat-lab"
for rel in ("habitat-lab", "habitat-baselines", "habitat-mas"):
    p = HAB_LAB_ROOT / rel
    if p.is_dir() and str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load_episode(split: str, episode_id: int) -> dict:
    with gzip.open(ROOT / "data" / "datasets" / "ovmm" / f"{split}.json.gz") as f:
        data = json.load(f)
    for ep in data["episodes"]:
        if int(ep["episode_id"]) == episode_id:
            return ep
    raise ValueError(f"episode {episode_id} not found in split {split!r}")


def _matrix_translation(mat: list[list[float]]) -> np.ndarray:
    arr = np.array(mat, dtype=np.float32)
    return arr[:3, 3]


def _quat_from_rotation_matrix(rot: np.ndarray):
    import magnum as mn

    trace = float(rot[0, 0] + rot[1, 1] + rot[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
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
    forward = target - source
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-5:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)

    # Habitat camera forward is local -Z.
    rot = np.column_stack([right, up, -forward])
    quat = _quat_from_rotation_matrix(rot)
    return [quat.vector.x, quat.vector.y, quat.vector.z, quat.scalar]


def _find_object_template(template_mgr, handle: str) -> str:
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
    import magnum as mn
    import habitat_sim

    template_mgr = sim.get_object_template_manager()
    rigid_mgr = sim.get_rigid_object_manager()
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
            print(f"[warn] skipping {cfg_name}: {exc}", flush=True)
            continue

        mat = np.array(transform, dtype=np.float32)
        obj.transformation = mn.Matrix4(mat)
        if cfg_name.startswith(target_prefix):
            target_id = obj.object_id
            obj.semantic_id = target_semantic_id
            obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
    return target_id


def render(args: argparse.Namespace) -> Path:
    if "MAGNUM_LOG" not in os.environ:
        os.environ["MAGNUM_LOG"] = "quiet"
    if "HABITAT_SIM_LOG" not in os.environ:
        os.environ["HABITAT_SIM_LOG"] = "quiet"

    import habitat_sim
    from habitat_sim.utils.common import d3_40_colors_rgb

    ep = _load_episode(args.split, args.episode_id)
    target_key = next(iter(ep["targets"]))
    target_prefix = target_key.split("_:")[0]
    target_goal = _matrix_translation(ep["targets"][target_key])
    rigid_pose = next((x[1] for x in ep["rigid_objs"] if x[0].startswith(target_prefix)), None)
    target_pos = _matrix_translation(rigid_pose) if rigid_pose is not None else target_goal
    look_at = target_pos + np.array([0.0, args.look_height, 0.0], dtype=np.float32)

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(ROOT / ep["scene_id"])
    sim_cfg.scene_dataset_config_file = str(ROOT / ep["scene_dataset_config"])
    sim_cfg.enable_physics = True
    sim_cfg.gpu_device_id = args.gpu_id

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

        candidates = [
            target_pos + np.array([1.2, 0.55, 0.0], dtype=np.float32),
            target_pos + np.array([-1.2, 0.55, 0.0], dtype=np.float32),
            target_pos + np.array([0.0, 0.55, 1.2], dtype=np.float32),
            target_pos + np.array([0.0, 0.55, -1.2], dtype=np.float32),
            target_pos + np.array([1.0, 0.85, 1.0], dtype=np.float32),
            target_pos + np.array([-1.0, 0.85, -1.0], dtype=np.float32),
        ]
        if args.camera_position:
            candidates.insert(0, np.array(args.camera_position, dtype=np.float32))

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
            if score > best_score:
                best_score = score
                best_obs = obs
                best_camera = cam_pos

        if best_obs is None:
            raise RuntimeError("no camera observations returned")

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

        if args.semantic_output:
            sem = np.asarray(best_obs["semantic"], dtype=np.int32)
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
    parser.add_argument("--look-height", type=float, default=0.1)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--camera-position", nargs=3, type=float)
    parser.add_argument("--target-semantic-id", type=int, default=1337)
    render(parser.parse_args())


if __name__ == "__main__":
    main()
