#!/usr/bin/env python3
"""Render several extended Fetch arm poses (head + wrist + third-person if
available) so a human can pick the best demonstration/operating pose.

Outputs: runs/arm_pose_candidates/<label>_{head,wrist,tp}.png
"""
from __future__ import annotations
import os, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Fetch 7-DOF: [shoulder_pan, shoulder_lift, upperarm_roll, elbow_flex,
#               forearm_roll, wrist_flex, wrist_roll]
POSES = {
    "P1_reach_fwd":  [0.0, -0.30, 0.0, 0.30, 0.0, 0.80, 0.0],
    "P2_level":      [0.0,  0.00, 0.0, 0.40, 0.0, 0.60, 0.0],
    "P3_down_table": [0.0,  0.25, 0.0, 0.55, 0.0, 0.95, 0.0],
    "P4_straight":   [0.0, -0.20, 0.0, 0.10, 0.0, 1.10, 0.0],
    "P5_high":       [0.0, -0.60, 0.0, 0.30, 0.0, 0.70, 0.0],
}


def _save(path, rgb):
    if rgb is not None:
        Image.fromarray(np.asarray(rgb)).save(path)


def main():
    import run_ovmm_embodied as r
    from sim.habitat_toolbox import HabitatToolbox
    r._load_taskmap("train")
    gc, HabEnv = r.base._import_habitat()
    args = SimpleNamespace(split="train", no_drop_missing=False, gpu_id=0, display=False)
    prev = os.getcwd(); os.chdir(str(r.base._OWMM_ROOT))
    try:
        cfg = r._ovmm_build_episode_config(0, args, gc); env = HabEnv(config=cfg); obs = env.reset()
    finally:
        os.chdir(prev)

    out = Path("runs/arm_pose_candidates"); out.mkdir(parents=True, exist_ok=True)
    tb = HabitatToolbox(env, gemini_client=None, log_dir=str(out / "log"),
                        capture_out_dir=str(out / "cap"), initial_obs=obs, display=False)
    sim = env._sim
    robot = sim.agents_mgr[0].articulated_agent

    print("obs keys:", sorted(obs.keys()), flush=True)
    tp_key = next((k for k in obs if "third" in k.lower()), None)
    print("third-person key:", tp_key, flush=True)

    for name, pose in POSES.items():
        robot.arm_joint_pos = pose
        robot.gripper_joint_pos = robot.params.gripper_open_state
        robot.update()
        obs = env.step(tb._null_step_action())
        ee = np.array(robot.ee_transform().translation)
        _save(out / f"{name}_head.png", tb._capture_rgb(obs))
        _save(out / f"{name}_wrist.png", tb._capture_wrist_rgb(obs))
        if tp_key and tp_key in obs:
            _save(out / f"{name}_tp.png", obs[tp_key])
        print(f"[{name}] EE={ee.round(3)} saved", flush=True)

    env.close()


if __name__ == "__main__":
    main()
