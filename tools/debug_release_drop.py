#!/usr/bin/env python3
"""Probe whether Habitat OVMM release lets a grasped object fall.

The benchmark config uses kinematic scene objects for speed. This script snaps
an object into the gripper, calls HabitatToolbox._release(), and measures
whether the object falls. It also has a comparison scenario that manually marks
the object dynamic before release.

A CONTROLLED EXPERIMENT, and it is the experiment that justifies
HabitatToolbox._enable_released_object_physics existing at all. Two scenarios, identical
except for one line:

    kinematic_release   release the object as the benchmark loads it (kinematic)
    dynamic_release     mark it DYNAMIC first, then release

The measurement is `drop_m` — how far the object's Y fell. If the kinematic scenario reports
~0 and the dynamic one reports a real drop, the object was freezing in mid-air on release,
and the toolbox is right to force it dynamic first. Both scenarios build a FRESH env, so
neither can contaminate the other.

Run it after touching anything in the release path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _save_rgb(path: Path, rgb) -> str | None:
    if rgb is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)
    return str(path)


def _vec3(v) -> list[float]:
    """magnum Vector3 → plain list, so it can go into the JSON summary."""
    return [float(v[0]), float(v[1]), float(v[2])]


def _motion_name(obj) -> str:
    """"MotionType.KINEMATIC" → "KINEMATIC". The whole point of the experiment is watching
    this value, so record it in a readable form at every stage."""
    try:
        return str(obj.motion_type).split(".")[-1]
    except Exception:
        return str(getattr(obj, "motion_type", "unknown"))


def _make_dynamic(obj) -> None:
    """The variable under test. Deliberately a trimmed copy of
    HabitatToolbox._enable_released_object_physics — it omits the UserGroup7 collision-group
    override, so this measures ONLY the effect of the motion-type change."""
    import habitat_sim
    import magnum as mn

    obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
    obj.collidable = True
    obj.linear_velocity = mn.Vector3.zero_init()
    obj.angular_velocity = mn.Vector3.zero_init()
    obj.awake = True   # a sleeping body ignores gravity entirely


def _build_env(split: str, episode_id: int, gpu_id: int):
    import run_ovmm_embodied as r

    r._load_taskmap(split)
    get_config, HabEnv = r.base._import_habitat()
    args = SimpleNamespace(
        split=split,
        no_drop_missing=False,
        gpu_id=gpu_id,
        display=False,
    )
    prev = os.getcwd()
    os.chdir(str(r.base._OWMM_ROOT))
    try:
        cfg = r._ovmm_build_episode_config(episode_id, args, get_config)
        env = HabEnv(config=cfg)
        obs = env.reset()
    finally:
        os.chdir(prev)
    return env, obs


def _run_scenario(
    *,
    name: str,
    split: str,
    episode_id: int,
    gpu_id: int,
    out_dir: Path,
    dynamic_before_release: bool,
) -> dict:
    """Run one arm of the experiment and return everything measured.

    Sequence: pose the arm → snap an object into the hand → record `before` → (optionally
    make it dynamic) → release → step physics and sample the object's position over 3
    seconds → record `final`. The headline number is drop_m in the returned dict.

    A fresh env per scenario (built here, closed in the finally) is what keeps the two arms
    independent.
    """
    from sim.habitat_toolbox import HabitatToolbox

    env, obs = _build_env(split, episode_id, gpu_id)
    try:
        scenario_dir = out_dir / name
        tb = HabitatToolbox(
            env,
            gemini_client=None,
            log_dir=str(scenario_dir / "agent_log"),
            capture_out_dir=str(scenario_dir / "captures"),
            initial_obs=obs,
            display=False,
        )
        sim = env._sim
        robot = sim.agents_mgr[0].articulated_agent
        rom = sim.get_rigid_object_manager()
        if not sim.scene_obj_ids:
            raise RuntimeError("episode has no scene_obj_ids to grasp")

        # Keep the arm in the configured extended/ready pose, then snap a scene
        # object into the gripper so we can isolate release behavior.
        robot.arm_joint_pos = robot.params.arm_init_params
        robot.gripper_joint_pos = robot.params.gripper_open_state
        robot.update()
        obs = env.step(tb._null_step_action())
        tb._last_obs = obs

        # force=True: snap the object into the hand regardless of contact. We are staging the
        # release, not testing the grasp, so the approach is skipped entirely.
        obj_id = int(sim.scene_obj_ids[0])
        obj = rom.get_object_by_id(obj_id)
        handle = obj.handle
        sim.grasp_mgr.snap_to_obj(obj_id, force=True)
        obs = env.step(tb._null_step_action())
        tb._last_obs = obs
        obj = rom.get_object_by_id(obj_id)   # re-fetch: the handle can be invalidated

        # Baseline. `translation` here is what final translation is subtracted from to get
        # drop_m, and `motion_type` records how the benchmark actually loaded the object.
        before = {
            "translation": _vec3(obj.translation),
            "motion_type": _motion_name(obj),
            "collidable": bool(obj.collidable),
            "snap_idx": sim.grasp_mgr.snap_idx,
            "ee_translation": _vec3(robot.ee_transform().translation),
        }
        _save_rgb(
            scenario_dir / "before_head.png",
            tb._capture_rgb(obs),
        )
        _save_rgb(
            scenario_dir / "before_wrist.png",
            tb._capture_wrist_rgb(obs),
        )

        # ── THE INDEPENDENT VARIABLE ──────────────────────────────────────────
        if dynamic_before_release:
            _make_dynamic(obj)

        release_ok, release_summary = tb._release()
        obs = tb._last_obs or env.step(tb._null_step_action())
        obj = rom.get_object_by_id(obj_id)
        # Snapshot immediately after release, before any physics has run: snap_idx should now
        # be None (the constraint is gone) while the object has not yet had time to move.
        after_release = {
            "translation": _vec3(obj.translation),
            "motion_type": _motion_name(obj),
            "collidable": bool(obj.collidable),
            "snap_idx": sim.grasp_mgr.snap_idx,
        }

        # Drive physics by hand — the benchmark config disables automatic stepping, so
        # nothing falls unless we tick it. 180 ticks at 1/60 s = 3 s of simulated time, ample
        # for an object to fall from gripper height and settle.
        samples = []
        for i in range(180):
            sim.step_physics(1.0 / 60.0)
            try:
                sim.maybe_update_articulated_agent()
            except Exception:
                pass
            # Sample on a roughly logarithmic schedule: dense early (where the fall happens)
            # and sparse later (confirming it has settled).
            if i in (0, 5, 15, 30, 60, 120, 179):
                obj = rom.get_object_by_id(obj_id)
                samples.append({
                    "step": i + 1,
                    "translation": _vec3(obj.translation),
                })
        obs = env.step(tb._null_step_action())
        tb._last_obs = obs
        obj = rom.get_object_by_id(obj_id)
        final = {
            "translation": _vec3(obj.translation),
            "motion_type": _motion_name(obj),
            "collidable": bool(obj.collidable),
            "snap_idx": sim.grasp_mgr.snap_idx,
        }
        _save_rgb(
            scenario_dir / "after_head.png",
            tb._capture_rgb(obs),
        )
        _save_rgb(
            scenario_dir / "after_wrist.png",
            tb._capture_wrist_rgb(obs),
        )

        return {
            "name": name,
            "object_id": obj_id,
            "object_handle": handle,
            "dynamic_before_release": dynamic_before_release,
            "release_ok": bool(release_ok),
            "release_summary": release_summary,
            "before": before,
            "after_release": after_release,
            "samples": samples,
            "final": final,
            # THE RESULT. [1] is world Y (up in Habitat), so before - final is how far the
            # object fell: ~0 means it froze in mid-air, a positive value means it dropped.
            "drop_m": before["translation"][1] - final["translation"][1],
            "paths": {
                "before_head": str(scenario_dir / "before_head.png"),
                "before_wrist": str(scenario_dir / "before_wrist.png"),
                "after_head": str(scenario_dir / "after_head.png"),
                "after_wrist": str(scenario_dir / "after_wrist.png"),
            },
        }
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="minival")
    parser.add_argument("--episode-id", type=int, default=0)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--out-dir", default="/tmp/ovmm_release_drop_debug")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The two arms of the experiment. Compare their drop_m: if kinematic ≈ 0 and dynamic > 0,
    # the motion-type conversion in _enable_released_object_physics is what makes release work.
    results = [
        _run_scenario(
            name="kinematic_release",
            split=args.split,
            episode_id=args.episode_id,
            gpu_id=args.gpu_id,
            out_dir=out_dir,
            dynamic_before_release=False,
        ),
        _run_scenario(
            name="dynamic_release",
            split=args.split,
            episode_id=args.episode_id,
            gpu_id=args.gpu_id,
            out_dir=out_dir,
            dynamic_before_release=True,
        ),
    ]
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
