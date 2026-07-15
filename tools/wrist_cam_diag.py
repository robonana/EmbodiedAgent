#!/usr/bin/env python3
"""Diagnostic: where is the end-effector relative to the arm-camera link?

Prints the arm camera params and computes the EE position expressed in the
camera-link frame, so we can aim cam_look_at_pos / place cam_offset_pos directly
at the gripper instead of guessing.

This is the measurement that produced the magic numbers in
HabitatToolbox._configure_gripper_camera. The camera mount is specified in the CAMERA-LINK
frame, but everything one naturally knows about the gripper (where the fingers are, where the
end-effector is) is in world coordinates. This script does the one transform that bridges
them — world → camera-link — and prints the answer, so the mount can be set from a
measurement rather than by trial and error.

Its key finding, recorded in that method's comments: the two fingers separate along the
camera-link Y axis, which is why a top-down camera sees only one finger and the mount instead
looks ACROSS them along Z.
"""
from __future__ import annotations
import os, sys
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main():
    import numpy as np
    import magnum as mn
    import run_ovmm_embodied as r
    r._load_taskmap("train")
    get_config, HabEnv = r.base._import_habitat()
    args = SimpleNamespace(split="train", no_drop_missing=False, gpu_id=0, display=False)
    prev = os.getcwd(); os.chdir(str(r.base._OWMM_ROOT))
    try:
        cfg = r._ovmm_build_episode_config(0, args, get_config)
        env = HabEnv(config=cfg); env.reset()
    finally:
        os.chdir(prev)

    sim = env._sim
    robot = sim.agents_mgr[0].articulated_agent
    # Use the stretched reaching pose (operational pose), not the tucked init.
    # This matters: the camera is rigid to the gripper link, so its view depends on where the
    # arm IS. Measuring in the tucked pose would tune the camera for a configuration it is
    # never actually used in. (The finger/EE offsets below come out the same either way —
    # they are rigid to the same link — but this keeps the whole measurement honest.)
    robot.arm_joint_pos = [0.0, -0.30, 0.0, 0.30, 0.0, 0.80, 0.0]
    robot.gripper_joint_pos = robot.params.gripper_open_state   # fingers apart, so their
                                                                # separation axis is visible
    robot.update()

    cam = robot.params.cameras.get("articulated_agent_arm")
    print("attached_link_id:", cam.attached_link_id)
    print("cam_offset_pos:", list(cam.cam_offset_pos))
    print("cam_look_at_pos:", list(cam.cam_look_at_pos))

    # Find the gripper finger links and express them in the camera-link frame so
    # we know which axis the two fingers separate along.
    #
    # This is THE question. Both mount vectors are specified in this frame, so knowing that
    # the fingers sit at roughly (0.08, ∓0.05, 0) tells us immediately that they separate
    # along Y — and therefore that the camera must be offset along Z to see between them.
    link_T = robot.sim_obj.get_link_scene_node(cam.attached_link_id).transformation
    inv = link_T.inverted()      # world → camera-link
    ao = robot.sim_obj
    for lid in ao.get_link_ids():
        try:
            nm = ao.get_link_name(lid)
        except Exception:
            nm = ""   # unnamed link; the name filter below just won't match it
        if "finger" in str(nm).lower() or "gripper" in str(nm).lower():
            wpos = ao.get_link_scene_node(lid).transformation.translation
            local = inv.transform_point(wpos)
            print(f"  link {lid} '{nm}': cam-link frame = "
                  f"{[round(float(x),4) for x in local]}", flush=True)

    # World transforms
    ee_T = robot.ee_transform()           # magnum Matrix4 (world)
    ee_world = ee_T.translation
    print("EE world:", list(ee_world))

    # Camera link world transform
    link_id = cam.attached_link_id
    try:
        # -1 is the convention for "attached to the base", which has no link node — its
        # transform is the articulated object's own.
        if link_id == -1:
            link_T = robot.sim_obj.transformation
        else:
            link_T = robot.sim_obj.get_link_scene_node(link_id).transformation
        print("cam link world translation:", list(link_T.translation))
        # EE expressed in camera-link frame:
        # The headline result — this vector IS the value to paste into cam_look_at_pos, and
        # it comes out at roughly (0.08, 0, 0) for any arm pose (the EE is rigid to this link).
        ee_in_link = link_T.inverted().transform_point(ee_world)
        print(">>> EE in camera-link frame (aim cam_look_at_pos here):",
              [round(float(x), 4) for x in ee_in_link])
    except Exception as e:
        print("link transform failed:", e)

    env.close()


if __name__ == "__main__":
    main()
