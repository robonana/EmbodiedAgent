#!/usr/bin/env python3
"""Stretch the Fetch arm to a reaching pose, then render third-person + head +
wrist so we can judge the wrist-camera framing with the gripper out in front.

Prints arm_init_params and EE position for each candidate arm pose.
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

# Fetch 7-DOF arm: [shoulder_pan, shoulder_lift, upperarm_roll, elbow_flex,
#                   forearm_roll, wrist_flex, wrist_roll]
ARM_POSES = {
    "tucked_init": None,  # use arm_init_params
    "reach_fwd":   [0.0, -0.30, 0.0, 0.30, 0.0, 0.80, 0.0],
    "reach_level": [0.0,  0.00, 0.0, 0.30, 0.0, 0.60, 0.0],
    "reach_far":   [0.0, -0.50, 0.0, 0.10, 0.0, 0.70, 0.0],
}


def _save(path, rgb):
    if rgb is not None:
        Image.fromarray(rgb).save(path)


def main():
    import magnum as mn
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

    out = Path("runs/arm_stretch"); out.mkdir(parents=True, exist_ok=True)
    tb = HabitatToolbox(env, gemini_client=None, log_dir=str(out / "log"),
                        capture_out_dir=str(out / "cap"), initial_obs=obs, display=False)
    sim = env._sim
    robot = sim.agents_mgr[0].articulated_agent
    print("arm_init_params:", list(np.asarray(robot.params.arm_init_params)))

    for name, pose in ARM_POSES.items():
        if pose is None:
            robot.arm_joint_pos = robot.params.arm_init_params
        else:
            robot.arm_joint_pos = pose
        robot.gripper_joint_pos = robot.params.gripper_open_state
        robot.update()
        obs = env.step(tb._null_step_action())
        ee = np.array(robot.ee_transform().translation)
        print(f"[{name}] EE world={ee.round(3)}")
        # third-person to see the arm; wrist to judge framing
        tp = None
        for k in ("third_person_sensor", "agent_0_third_person_sensor"):
            if k in obs:
                tp = obs[k]; break
        _save(out / f"{name}_thirdperson.png", tp)
        _save(out / f"{name}_wrist.png", tb._capture_wrist_rgb(obs))
        _save(out / f"{name}_head.png", tb._capture_rgb(obs))
        print(f"[{name}] saved third-person/head/wrist")

    env.close()


if __name__ == "__main__":
    main()
