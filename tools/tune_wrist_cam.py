#!/usr/bin/env python3
"""Render Fetch's arm (wrist/gripper) camera under candidate poses to tune framing.

Builds an OVMM episode, poses the arm in its ready pose, snaps a scene object
into the gripper (so there is something to frame), then for each candidate
(cam_offset_pos, cam_look_at_pos, roll_deg) re-aims the `articulated_agent_arm`
camera, steps once to re-render, and saves the wrist RGB to a PNG.

Usage:
  python tools/tune_wrist_cam.py --split train --episode_id 0 --out runs/wrist_cam_tune
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# (label, cam_offset_pos, cam_look_at_pos, roll_deg)
# Frame: +X→fingertips, ±Z→lateral side, Y→vertical. Current live setting is
# offset(-0.06,0,-0.11) look_at(0.32,0,0.03) roll 90 — look_at aims far past the
# gripper, so it only sees it partially. Candidates pull the look_at onto the
# gripper and the camera in closer.
# Measured: end-effector sits at (0.08, 0, 0) in the camera-link frame. Aim
# cam_look_at_pos at/just past the EE and place the camera a short distance away.
# EE is at (0.08,0,0) in cam-link frame for ALL arm poses (link rigid to gripper).
# Move the camera FORWARD (+X, past the white arm housing → external mount) but
# keep look_at ON the gripper (~0.09), not past it onto the floor.
# Straight-DOWN top views: camera directly above the gripper (same X & Z as the
# look_at, higher Y) so it looks straight down (-Y) and BOTH fingers are in the
# image plane. roll rotates which way the finger-separation axis lies, so we
# sweep roll to get the two fingers side-by-side rather than one behind the other.
# Fingers separate along +Y (l at y=-0.04, r at y=+0.056), finger length along +X,
# centered at ~(0.08, 0.008, 0). To see BOTH fingers + the gap, look along Z
# (across the fingers). Level views are pure-side; elevated (+Y) ones angle down
# for a "top-ish" feel while still showing both. roll swept to lay fingers
# side-by-side. Look at the finger CENTER (0.08, 0.008, 0).
CANDIDATES = [
    ("s_lvl_r0",  (0.08, 0.008, 0.11), (0.08, 0.008, 0.00),   0.0),
    ("s_lvl_r90", (0.08, 0.008, 0.11), (0.08, 0.008, 0.00),  90.0),
    ("s_up_r0",   (0.07, 0.09,  0.10), (0.08, 0.008, 0.00),   0.0),
    ("s_up_r90",  (0.07, 0.09,  0.10), (0.08, 0.008, 0.00),  90.0),
    ("s_up_rm90", (0.07, 0.09,  0.10), (0.08, 0.008, 0.00), -90.0),
    ("s_up_close",(0.06, 0.07,  0.08), (0.08, 0.008, 0.00),  90.0),
]


def _build_env(split, episode_id, gpu_id):
    import run_ovmm_embodied as r
    r._load_taskmap(split)
    get_config, HabEnv = r.base._import_habitat()
    args = SimpleNamespace(split=split, no_drop_missing=False, gpu_id=gpu_id, display=False)
    prev = os.getcwd()
    os.chdir(str(r.base._OWMM_ROOT))
    try:
        cfg = r._ovmm_build_episode_config(episode_id, args, get_config)
        env = HabEnv(config=cfg)
        obs = env.reset()
    finally:
        os.chdir(prev)
    return env, obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--episode_id", type=int, default=0)
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--out", default="runs/wrist_cam_tune")
    ap.add_argument("--no_snap", action="store_true",
                    help="Do not snap an object into the gripper (view the empty "
                         "gripper to judge how much of it is in frame)")
    args = ap.parse_args()

    import magnum as mn
    from sim.habitat_toolbox import HabitatToolbox

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env, obs = _build_env(args.split, args.episode_id, args.gpu_id)
    tb = HabitatToolbox(env, gemini_client=None,
                        log_dir=str(out_dir / "agent_log"),
                        capture_out_dir=str(out_dir / "captures"),
                        initial_obs=obs, display=False)
    sim = env._sim
    robot = sim.agents_mgr[0].articulated_agent

    cam = robot.params.cameras.get("articulated_agent_arm")
    if cam is None:
        print("[tune] no articulated_agent_arm camera on this robot", flush=True)
        return

    # Production-faithful flow: habitat bakes the camera orientation at the pose
    # present when the params are applied (in production that's the TUCKED init
    # pose, inside _configure_gripper_camera). So for each candidate we (1) set
    # the arm to the tucked init pose, (2) apply the cam params + update (bake),
    # then (3) move the arm to the P1 reaching pose to actually view the gripper.
    tucked = list(map(float, robot.params.arm_init_params))
    P1     = [0.0, -0.30, 0.0, 0.30, 0.0, 0.80, 0.0]   # reaching pose

    for label, off, look, roll in CANDIDATES:
        # (1) tucked, then (2) configure/bake at tucked
        robot.arm_joint_pos = tucked
        robot.gripper_joint_pos = robot.params.gripper_open_state
        robot.update()
        cam.cam_offset_pos  = mn.Vector3(*off)
        cam.cam_look_at_pos = mn.Vector3(*look)
        cam.relative_transform = mn.Matrix4.rotation_z(mn.Deg(roll))
        robot.update()
        # (3) move to the reaching pose and (optionally) snap an object
        robot.arm_joint_pos = P1
        robot.update()
        obs = env.step(tb._null_step_action())
        if sim.scene_obj_ids and not args.no_snap:
            try:
                sim.grasp_mgr.snap_to_obj(int(sim.scene_obj_ids[0]), force=True)
                obs = env.step(tb._null_step_action())
            except Exception as e:
                print(f"[tune] {label}: snap failed ({e})", flush=True)
        rgb = tb._capture_wrist_rgb(obs)
        if rgb is None:
            print(f"[tune] {label}: wrist rgb is None", flush=True)
            continue
        path = out_dir / f"{label}.png"
        Image.fromarray(rgb).save(path)
        print(f"[tune] {label}: off={off} look={look} roll={roll} -> {path}", flush=True)

    env.close()
    print(f"[tune] done -> {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
