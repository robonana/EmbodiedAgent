#!/usr/bin/env python3
"""run_ovmm_embodied.py — Run PromptEmbodiedAgent on the ai-habitat OVMM benchmark.

OVMM (Open-Vocabulary Mobile Manipulation) episodes are loaded from
ai-habitat/OVMM_episodes (converted to a rearrange-compatible json.gz by
tools/convert_ovmm_episodes.py). We reuse the OWMM runner's per-episode
machinery (HabitatToolbox + PromptEmbodiedAgent loop + PDDL metrics) and only
swap the dataset/episode/task/config hooks.

Prereqs (one-time):
    python tools/convert_ovmm_episodes.py minival        # -> data/datasets/ovmm/minival.json.gz
    hf download ai-habitat/OVMM_objects --repo-type dataset --local-dir data/objects
    # data/hssd-hab symlink -> versioned_data/hssd-hab  (scenes)

Usage (from EmbodiedAgent/ root, habitat conda env active):
    python run_ovmm_embodied.py --split minival --episode_ids 0
    python run_ovmm_embodied.py --split minival --max_episodes 3 --max_agent_steps 40
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

print("[run_ovmm_embodied] starting …", flush=True)

import run_owmm_embodied as base  # reuse _import_habitat, _run_episode, path setup

_HERE = Path(__file__).resolve().parent
_OVMM_DIR = _HERE / "data" / "datasets" / "ovmm"
_SINGLE_DIR = _OVMM_DIR / "_single"
_RENDER = os.environ.get("HABITAT_RENDER", "0") == "1"

# task map (episode_id -> categories), loaded per split
_TASKMAP: Dict[str, Dict[str, str]] = {}


# ── OVMM-specific dataset / task / config hooks ──────────────────────────────

def _load_taskmap(split: str) -> None:
    global _TASKMAP
    p = _OVMM_DIR / f"{split}_taskmap.json"
    _TASKMAP = json.loads(p.read_text()) if p.exists() else {}


def _all_episode_ids(split: str) -> List[int]:
    d = json.load(gzip.open(_OVMM_DIR / f"{split}.json.gz"))
    return [int(e["episode_id"]) for e in d["episodes"]]


def _is_transform4x4(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(row, list) and len(row) == 4 for row in value)
    )


def _validate_inline_transforms(split: str, episode_ids: List[int]) -> None:
    """Catch stale OVMM conversions before Habitat crashes in _add_objs."""
    wanted = {int(ep_id) for ep_id in episode_ids}
    d = json.load(gzip.open(_OVMM_DIR / f"{split}.json.gz"))
    bad: list[tuple[int, str]] = []
    for ep in d["episodes"]:
        ep_id = int(ep["episode_id"])
        if ep_id not in wanted:
            continue
        for obj_name, transform in ep.get("rigid_objs", []):
            if not _is_transform4x4(transform):
                bad.append((ep_id, obj_name))
                break
    if bad:
        examples = ", ".join(f"ep{ep_id}:{obj_name}" for ep_id, obj_name in bad[:3])
        raise RuntimeError(
            f"{_OVMM_DIR / (split + '.json.gz')} has stale OVMM rigid object "
            f"transform indices instead of inline 4x4 matrices ({examples}). "
            f"Regenerate it with: python tools/convert_ovmm_episodes.py {split}"
        )


def _present_configs(obj_dirs: List[str]) -> set:
    """Object-config basenames actually loadable from the sim's config dirs
    (NOT the per-mesh stub configs under assets/)."""
    import glob
    present = set()
    for d in obj_dirs:
        present |= {os.path.basename(p)
                    for p in glob.glob(os.path.join(_HERE, d) + "*.object_config.json")}
    return present


def _write_single_episode(split: str, ep_id: int, drop_missing: bool = True):
    """Write a one-episode json.gz; return (data_path, episode_dict).

    When drop_missing is set, rigid_objs whose object config has not been
    downloaded are removed (the target object is always kept) so the episode
    can instantiate even with an incomplete OVMM_objects mirror. Returns the
    count of dropped clutter for transparency.
    """
    d = json.load(gzip.open(_OVMM_DIR / f"{split}.json.gz"))
    match = [e for e in d["episodes"] if int(e["episode_id"]) == ep_id]
    if not match:
        raise ValueError(f"episode {ep_id} not in split {split}")
    ep = match[0]
    for obj_name, transform in ep.get("rigid_objs", []):
        if not _is_transform4x4(transform):
            raise RuntimeError(
                f"episode {ep_id} object {obj_name} has stale transform index "
                f"{transform!r}; regenerate with: "
                f"python tools/convert_ovmm_episodes.py {split}"
            )
    n_dropped = 0
    if drop_missing:
        present = _present_configs(ep["additional_obj_config_paths"])
        tgt_keys = {k.split("_:")[0] for k in ep["targets"].keys()}
        kept = []
        for ro in ep["rigid_objs"]:
            base = os.path.basename(ro[0])
            is_target = any(t in ro[0] for t in tgt_keys)
            if base in present or is_target:
                kept.append(ro)
            else:
                n_dropped += 1
        ep = dict(ep); ep["rigid_objs"] = kept
    out = dict(d); out["episodes"] = [ep]
    _SINGLE_DIR.mkdir(parents=True, exist_ok=True)
    fp = _SINGLE_DIR / f"{split}_{ep_id}.json.gz"
    with gzip.open(fp, "wt") as f:
        json.dump(out, f)
    if n_dropped:
        print(f"[ep {ep_id}] dropped {n_dropped} clutter objects not yet downloaded "
              f"(target kept); benchmark fidelity reduced", flush=True)
    # data_path is relative to the fork root (cwd at config-build time),
    # whose data/ is a symlink to EmbodiedAgent/data.
    return f"data/datasets/ovmm/_single/{split}_{ep_id}.json.gz", ep


def _ovmm_task_prompt(ep_id: int) -> str:
    t = _TASKMAP.get(str(ep_id))
    if not t:
        return ""
    obj = t["object_category"].replace("_", " ")
    src = t["start_recep_category"].replace("_", " ")
    dst = t["goal_recep_category"].replace("_", " ")
    return (f"Find the {obj} on the {src}, pick it up, and place it "
            f"on the {dst}.")


def _ovmm_build_episode_config(ep_id: int, args, get_config):
    """fetch_vlm.yaml config pointed at this OVMM episode's converted json.gz.

    OVMM-specific overrides:
      - additional_object_paths -> the episode's OVMM object config dirs (the
        default fetch_vlm dirs don't contain HSSD/amazon/google OVMM assets).
      - remove the RL polar target sensors, which mis-size for OVMM and aren't
        consumed by the EmbodiedAgent (it reads RGB/depth/localization).
    """
    data_path, ep = _write_single_episode(args.split, ep_id,
                                           drop_missing=not args.no_drop_missing)
    obj_dirs = "[" + ",".join(ep["additional_obj_config_paths"]) + "]"
    overrides = [
        f"habitat.dataset.data_path={data_path}",
        f"habitat.simulator.additional_object_paths={obj_dirs}",
        "~habitat.task.lab_sensors.target_start_gps_compass_sensor",
        "~habitat.task.lab_sensors.target_goal_gps_compass_sensor",
        "~habitat.task.lab_sensors.object_to_goal_distance_sensor",
        f"habitat.seed={ep_id}",
        "habitat.simulator.kinematic_mode=False",
        "habitat.simulator.step_physics=True",
        f"habitat.simulator.habitat_sim_v0.gpu_device_id={args.gpu_id}",
    ]
    cfg = get_config("benchmark/single_agent/fetch_vlm.yaml", overrides=overrides)
    if _RENDER or args.display:
        from run_habitat import _add_third_person_sensor
        _add_third_person_sensor(cfg)
    return cfg


def _no_prepopulate(ep_id, embedding_worker, episodic_memory):
    pass  # OVMM has no pre-collected scene-graph images; always auto-scan


# ── Patch base runner hooks so base._run_episode does OVMM work ──────────────
base._task_prompt = _ovmm_task_prompt
base._build_episode_config = _ovmm_build_episode_config
base._prepopulate_memory = _no_prepopulate


def _build_args():
    p = argparse.ArgumentParser(description="Run EmbodiedAgent on the OVMM benchmark")
    p.add_argument("--split", default="minival", choices=["minival", "train", "val"])
    p.add_argument("--episode_ids", nargs="*", type=int, metavar="ID")
    p.add_argument("--max_episodes", type=int, default=None)
    p.add_argument("--all_episodes", action="store_true")
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--vlm_model", default="models/gemini-3.5-flash")
    p.add_argument("--gemini_api_key",
                   default=os.environ.get("GEMINI_API_KEY", ""))
    p.add_argument("--retrieval_model", default="siglip_base")
    p.add_argument("--scan_points", type=int, default=8)
    p.add_argument("--no_scan", action="store_true")
    p.add_argument("--explore", action="store_true",
                   help="Use frontier-based exploration to build scene memory "
                        "instead of the random-point scan")
    p.add_argument("--explore_iters", type=int, default=40,
                   help="Max frontier viewpoints to visit (default: 40)")
    p.add_argument("--explore_lambda", type=float, default=0.5,
                   help="Travel-cost weight in score=cluster_size-λ·cost (default: 0.5). "
                        "Lower ⇒ distant frontiers stay worth visiting, so the robot "
                        "keeps exploring to full coverage instead of stopping early.")
    p.add_argument("--explore_range", type=float, default=1.5,
                   help="Max depth range fused into the occupancy map, m (default: 1.5). "
                        "Smaller ⇒ the robot must drive closer to objects to map them; "
                        "larger ⇒ it maps the scene from farther away.")
    p.add_argument("--explore_min_gain", type=int, default=1,
                   help="Stop when best frontier-cluster score < this (default: 1). "
                        "Lower ⇒ keep visiting small remaining frontiers for fuller "
                        "coverage.")
    p.add_argument("--explore_teleport", action="store_true",
                   help="Teleport between frontier viewpoints (fast) instead of "
                        "the default continuous oracle-nav driving")
    p.add_argument("--explore_video", action="store_true",
                   help="Save a side-by-side [front RGB | depth | occupancy map] "
                        "MP4 of the exploration to <log>/captures/explore.mp4")
    p.add_argument("--no_drop_missing", action="store_true",
                   help="Fail instead of dropping clutter objects whose assets "
                        "are not downloaded (use once OVMM_objects is complete)")
    p.add_argument("--task", default=None)
    p.add_argument("--max_agent_steps", type=int, default=40)
    p.add_argument("--agent_history_steps", type=int, default=8)
    p.add_argument("--max_monitor_cycles", type=int, default=3)
    p.add_argument("--log_dir", default=str(_HERE / "runs" / "ovmm_embodied"))
    p.add_argument("--display", action="store_true")
    p.add_argument("--obs_camera", default="head",
                   choices=["head", "arm_workspace"],
                   help="Agent observation camera (default: head = forward view)")
    p.add_argument("--no_gdino", action="store_true",
                   help="Disable GroundingDINO object localization (use Gemini "
                        "inspect for bounding boxes instead)")
    base._add_vlm_args(p)
    return base._normalize_vlm_args(p.parse_args())


def main():
    args = _build_args()
    _load_taskmap(args.split)

    split_file = _OVMM_DIR / f"{args.split}.json.gz"
    if not split_file.exists():
        print(f"[run_ovmm_embodied] ERROR: {split_file} missing.\n"
              f"  Run: python tools/convert_ovmm_episodes.py {args.split}", flush=True)
        sys.exit(1)

    if args.episode_ids:
        episode_ids = list(args.episode_ids)
    else:
        episode_ids = _all_episode_ids(args.split)
        if args.max_episodes:
            episode_ids = episode_ids[: args.max_episodes]
    if not episode_ids:
        print("[run_ovmm_embodied] no episodes selected", flush=True)
        sys.exit(1)
    try:
        _validate_inline_transforms(args.split, episode_ids)
    except RuntimeError as exc:
        print(f"[run_ovmm_embodied] ERROR: {exc}", flush=True)
        sys.exit(1)

    print(f"[run_ovmm_embodied] split={args.split} {len(episode_ids)} episodes "
          f"model={args.vlm_model}", flush=True)

    if _RENDER:
        from sim.habitat_toolbox import start_display_process
        start_display_process()

    print("[run_ovmm_embodied] importing Habitat …", flush=True)
    get_config, HabEnv = base._import_habitat()

    log_root = Path(args.log_dir)
    log_root.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    for ep_id in episode_ids:
        try:
            record = base._run_episode(ep_id, args, get_config, HabEnv)
        except KeyboardInterrupt:
            print("\n[run_ovmm_embodied] interrupted", flush=True)
            break
        except Exception as exc:
            print(f"[ep {ep_id}] fatal: {exc}", flush=True)
            traceback.print_exc()
            record = {"episode_id": ep_id, "error": str(exc), "framework": "embodied-ovmm"}
        records.append(record)

    if not records:
        return
    n = len(records)
    successes = sum(1 for r in records if r.get("hab_pddl_success", False))
    summary = {
        "framework": "embodied-ovmm", "split": args.split,
        "vlm_model": args.vlm_model, "total_episodes": n,
        "pddl_success_rate": round(successes / n, 4),
    }
    print("\n── EmbodiedAgent on OVMM Benchmark ─────────────────────────────")
    print(f"  split          : {args.split}")
    print(f"  episodes       : {n}")
    print(f"  pddl_success   : {successes}/{n}  ({100*successes/n:.1f}%)")
    print("─────────────────────────────────────────────────────────────────")
    (log_root / "summary.json").write_text(json.dumps(summary, indent=2))
    csv_path = log_root / "results.csv"
    fieldnames = sorted({k for r in records for k in r})
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(records)
    if _RENDER:
        from sim.habitat_toolbox import stop_display_process
        stop_display_process()
    print(f"[run_ovmm_embodied] results → {log_root}/", flush=True)


if __name__ == "__main__":
    main()
