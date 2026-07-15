#!/usr/bin/env python3
"""Stretch the Fetch arm to a reaching pose, then render third-person + head +
wrist so we can judge the wrist-camera framing with the gripper out in front.

Prints arm_init_params and EE position for each candidate arm pose.

Sibling of tools/arm_pose_candidates.py, aimed at a narrower question: the wrist camera is
mounted on the gripper (see HabitatToolbox._configure_gripper_camera), so its framing changes
completely as the arm moves. Judging that mount from the TUCKED pose is misleading — what
matters is what it sees when the arm is extended and about to grasp. So this renders the
tucked pose *and* several reaching poses, third-person (to see where the arm actually is)
alongside wrist (to see what the camera gets).

Manual diagnostic; nothing imports it.
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
# The tucked pose is the control: it is where the arm sits when the camera is CONFIGURED,
# and the point of this script is that it is not where the camera is USED.
ARM_POSES = {
    "tucked_init": None,  # use arm_init_params
    "reach_fwd":   [0.0, -0.30, 0.0, 0.30, 0.0, 0.80, 0.0],
    "reach_level": [0.0,  0.00, 0.0, 0.30, 0.0, 0.60, 0.0],
    "reach_far":   [0.0, -0.50, 0.0, 0.10, 0.0, 0.70, 0.0],   # elbow near-straight
}


def _save(path, rgb):
    """Write a frame if that camera exists in this config; otherwise skip."""
    if rgb is not None:
        Image.fromarray(rgb).save(path)


def main():
    import magnum as mn
    import run_ovmm_embodied as r
    from sim.habitat_toolbox import HabitatToolbox

    # Same env-construction path the real runner uses, so the camera mount under test is
    # exactly the one the agent will get. chdir because hydra's search paths are relative.
    r._load_taskmap("train")
    gc, HabEnv = r.base._import_habitat()
    args = SimpleNamespace(split="train", no_drop_missing=False, gpu_id=0, display=False)
    prev = os.getcwd(); os.chdir(str(r.base._OWMM_ROOT))
    try:
        cfg = r._ovmm_build_episode_config(0, args, gc); env = HabEnv(config=cfg); obs = env.reset()
    finally:
        os.chdir(prev)

    out = Path("runs/arm_stretch"); out.mkdir(parents=True, exist_ok=True)
    # Constructing the toolbox also runs _configure_gripper_camera — i.e. it applies the
    # wrist-camera mount we are here to evaluate. Built only for its camera helpers.
    tb = HabitatToolbox(env, gemini_client=None, log_dir=str(out / "log"),
                        capture_out_dir=str(out / "cap"), initial_obs=obs, display=False)
    sim = env._sim
    robot = sim.agents_mgr[0].articulated_agent
    # Printed so the tucked pose's actual joint values are on the record — they are the
    # baseline the reaching poses are being compared against.
    print("arm_init_params:", list(np.asarray(robot.params.arm_init_params)))

    for name, pose in ARM_POSES.items():
        if pose is None:
            robot.arm_joint_pos = robot.params.arm_init_params   # the tucked control
        else:
            robot.arm_joint_pos = pose
        # Open the gripper: a closed one hides the fingers, and finger visibility is
        # precisely what the wrist-camera mount is being judged on.
        robot.gripper_joint_pos = robot.params.gripper_open_state
        robot.update()                                # push kinematics into the sim
        obs = env.step(tb._null_step_action())        # re-render at the new pose
        ee = np.array(robot.ee_transform().translation)
        print(f"[{name}] EE world={ee.round(3)}")
        # third-person to see the arm; wrist to judge framing.
        # Key name varies with single/multi-agent config, so try both.
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
