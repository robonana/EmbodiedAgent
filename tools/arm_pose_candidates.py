#!/usr/bin/env python3
"""Render several extended Fetch arm poses (head + wrist + third-person if
available) so a human can pick the best demonstration/operating pose.

Outputs: runs/arm_pose_candidates/<label>_{head,wrist,tp}.png

A one-off visual sweep, not part of any pipeline. The problem it solves: choosing a "good"
arm pose by reading joint angles is hopeless — you have to see it. So this drives the arm to
each candidate, photographs it from every camera, and lets you compare the PNGs side by side.

Nothing here is imported by the agent; it exists to be run by hand.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image

# Run from anywhere: put the repo root on sys.path so `run_ovmm_embodied` etc. resolve.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Fetch 7-DOF: [shoulder_pan, shoulder_lift, upperarm_roll, elbow_flex,
#               forearm_roll, wrist_flex, wrist_roll]
# All candidates keep pan/roll at 0 (arm straight ahead, no twist) and vary only the three
# joints in the vertical plane — lift, elbow, wrist_flex. That is the sagittal slice that
# actually determines how high and how far forward the gripper ends up, which is the whole
# question here. Negative shoulder_lift raises the arm; positive lowers it.
POSES = {
    "P1_reach_fwd":  [0.0, -0.30, 0.0, 0.30, 0.0, 0.80, 0.0],
    "P2_level":      [0.0,  0.00, 0.0, 0.40, 0.0, 0.60, 0.0],
    "P3_down_table": [0.0,  0.25, 0.0, 0.55, 0.0, 0.95, 0.0],   # aimed at a table surface
    "P4_straight":   [0.0, -0.20, 0.0, 0.10, 0.0, 1.10, 0.0],   # elbow nearly extended
    "P5_high":       [0.0, -0.60, 0.0, 0.30, 0.0, 0.70, 0.0],   # shelf height
}


def _save(path, rgb):
    """Write a frame if the camera produced one; silently skip if not."""
    if rgb is not None:
        Image.fromarray(np.asarray(rgb)).save(path)


def main():
    import run_ovmm_embodied as r
    from sim.habitat_toolbox import HabitatToolbox
    # Reuse the OVMM runner's own env construction, so the robot/camera setup here is
    # identical to what the agent actually runs against — a pose that looks good in a
    # differently-configured env would be worthless.
    r._load_taskmap("train")
    gc, HabEnv = r.base._import_habitat()
    args = SimpleNamespace(split="train", no_drop_missing=False, gpu_id=0, display=False)
    # Hydra resolves its config search paths relative to the OWMM root, so chdir over the
    # build and restore afterwards.
    prev = os.getcwd(); os.chdir(str(r.base._OWMM_ROOT))
    try:
        # Episode 0 — any episode does; we only need a scene to stand the robot in.
        cfg = r._ovmm_build_episode_config(0, args, gc); env = HabEnv(config=cfg); obs = env.reset()
    finally:
        os.chdir(prev)

    out = Path("runs/arm_pose_candidates"); out.mkdir(parents=True, exist_ok=True)
    # The toolbox is built only for its camera helpers (_capture_rgb / _capture_wrist_rgb /
    # _null_step_action). No VLM, no memory — hence gemini_client=None.
    tb = HabitatToolbox(env, gemini_client=None, log_dir=str(out / "log"),
                        capture_out_dir=str(out / "cap"), initial_obs=obs, display=False)
    sim = env._sim
    robot = sim.agents_mgr[0].articulated_agent

    # The third-person sensor is not always configured; discover its key rather than
    # assuming one, and print the available sensors to make a missing one obvious.
    print("obs keys:", sorted(obs.keys()), flush=True)
    tp_key = next((k for k in obs if "third" in k.lower()), None)
    print("third-person key:", tp_key, flush=True)

    for name, pose in POSES.items():
        # Set joints directly (no controller, no IK) — we want exactly this pose.
        robot.arm_joint_pos = pose
        robot.gripper_joint_pos = robot.params.gripper_open_state
        robot.update()          # push the kinematic change into the sim's transforms
        # A null step is what re-renders the sensors at the new pose; without it every
        # frame would show the arm where it was before.
        obs = env.step(tb._null_step_action())
        # The end-effector position is the number that actually matters when judging a pose
        # ("can it reach a 0.8 m table?"), so print it alongside the images.
        ee = np.array(robot.ee_transform().translation)
        _save(out / f"{name}_head.png", tb._capture_rgb(obs))
        _save(out / f"{name}_wrist.png", tb._capture_wrist_rgb(obs))
        if tp_key and tp_key in obs:
            _save(out / f"{name}_tp.png", obs[tp_key])
        print(f"[{name}] EE={ee.round(3)} saved", flush=True)

    env.close()


if __name__ == "__main__":
    main()
