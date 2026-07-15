#!/usr/bin/env python3
"""
run_habitat.py — PromptEmbodiedAgent evaluation on Habitat/ReplicaCAD.

Mirrors OWMM-Agent's evaluation setup (FetchRobot + ReplicaCAD + oracle_nav)
so results are directly comparable.

Pipeline per episode
--------------------
1. Habitat resets the environment (loads scene, places robot + objects)
2. Derive a natural-language task description from the PDDL episode spec
3. Run PromptEmbodiedAgent (Gemini loop) until "finish" action or max steps
4. Collect Habitat task metrics (pddl_success, num_steps, …)
5. Write per-episode JSON and summary CSV

Usage
-----
    # from the project root (EmbodiedAgent/)
    python run_habitat.py \\
        --gpu_id 0 \\
        --gemini_api_key YOUR_KEY \\
        --max_episodes 10 \\
        --log_dir runs/habitat_eval

    # override dataset path:
    python run_habitat.py \\
        --dataset data/datasets/replica_cad/single_agent_eval.json.gz

The plain Habitat runner: no benchmark harness, no pre-collected scene images, just "load a
ReplicaCAD rearrange episode and let the agent at it". The benchmark runners
(run_owmm_embodied / run_ovmm_embodied) supersede it for evaluation, but two of its functions
are imported by them and are the reason it still matters:

    _derive_task_text()          builds a task SENTENCE from a PDDL episode spec — the
                                 fallback used whenever a dataset ships no task_prompt.json
    _add_third_person_sensor()   injects the over-the-shoulder camera used for rendering/video
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

# Force line-buffered stdout so progress prints immediately even when piped.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

print("[run_habitat] starting up (loading habitat-sim may take 20-30s) …", flush=True)

import numpy as np

# ── Path injection for habitat-lab / habitat-baselines ───────────────────────
_HERE         = Path(__file__).resolve().parent
_OWMM_ROOT    = _HERE / "OWMM-Agent" / "sim" / "habitat-lab"
_HAB_LAB      = str(_OWMM_ROOT / "habitat-lab")
_HAB_BASE     = str(_OWMM_ROOT / "habitat-baselines")
_HAB_MAS      = str(_OWMM_ROOT / "habitat-mas")

# Ensure the package root (agent/, sim/, memory/) is importable regardless of
# whether this script is run directly or as part of a parent package (e.g.
# `python -m sceneagent.run_habitat` from /home/chen/Projects/).
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

for _p in (_HAB_LAB, _HAB_BASE, _HAB_MAS):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)


# ── Deferred heavy imports (after path injection) ────────────────────────────

def _import_habitat():
    import habitat                              # noqa: F401 — triggers registration
    from habitat.config.default import get_config
    from habitat.core.env import Env
    # register OWMM-Agent's custom sensors/actions
    try:
        import habitat.tasks.rearrange          # noqa
        import habitat.tasks.rearrange.vlm      # noqa
    except ImportError:
        pass
    return get_config, Env


# ── Task description derivation ───────────────────────────────────────────────

def _derive_task_text(episode) -> str:
    """
    Build a natural-language task string from a RearrangeEpisode.

    Uses episode.targets (object → goal transform) and episode.name_to_receptacle
    (object → receptacle).  Falls back to a generic pick-and-place description.

    The LAST-RESORT task source across the whole project — every runner calls this when its
    dataset has no authored task sentence. It reverse-engineers prose from the episode's
    structural spec, which is why the output reads a little stilted ("on/in the"): the spec
    says *what* must end up *where*, not how a person would phrase it.

    Every field is fetched with getattr(..., default) because RearrangeEpisode's shape varies
    across dataset versions and a missing attribute must degrade the sentence, not raise.
    """
    targets          = getattr(episode, "targets",          {}) or {}
    name_to_recep    = getattr(episode, "name_to_receptacle", {}) or {}
    goal_receptacles = getattr(episode, "goal_receptacles",  []) or []

    if not targets:
        return "Pick up the object and place it at the goal location."

    # One sentence per target — multi-object episodes exist, though they are rare.
    parts = []
    for obj_key in targets:
        # obj_key is typically "handle:index", e.g. "kitchen_counter_01:0000"
        # Strip the instance index and de-snake_case it into something readable.
        obj_name = obj_key.split(":")[0].replace("_", " ").strip()
        recep    = name_to_recep.get(obj_key, "")
        goal_loc = (
            goal_receptacles[0][0].replace("_", " ").strip()
            if goal_receptacles
            else "the goal location"
        )
        # Naming the SOURCE receptacle when we know it makes the task far easier to ground —
        # "from the kitchen counter" tells the agent where to look first.
        if recep:
            src = recep.replace("_", " ").strip()
            parts.append(
                f"Pick up the {obj_name} from the {src} and place it on/in the {goal_loc}."
            )
        else:
            parts.append(
                f"Pick up the {obj_name} and place it at the {goal_loc}."
            )

    return "  ".join(parts) if parts else "Pick up the object and place it at the goal location."


# ── Config helpers ────────────────────────────────────────────────────────────

def _build_hab_config(args, get_config):
    """Load habitat config with CLI overrides."""
    cfg_path = (
        args.hab_config
        if os.path.isabs(args.hab_config)
        else str(_OWMM_ROOT / "habitat-lab" / "habitat" / "config" / args.hab_config)
    )
    if not os.path.exists(cfg_path):
        # try relative to habitat-lab package config dir
        cfg_path = args.hab_config

    overrides = [
        f"habitat.simulator.habitat_sim_v0.gpu_device_id={args.gpu_id}",
        # ReplicaCAD scenes — override the default hssd-hab path
        "habitat.dataset.scenes_dir=data/scene_datasets",
    ]
    if args.dataset:
        overrides.append(f"habitat.dataset.data_path={args.dataset}")
    if args.record_replay:
        overrides += [
            "habitat.task.measurements.gfx_replay_measure.gfx_replay_dir="
            + str(Path(args.log_dir) / "replays"),
            "habitat.simulator.habitat_sim_v0.enable_gfx_replay_save=True",
        ]
    print(f"[run_habitat] loading config: {cfg_path}")
    cfg = get_config(cfg_path, overrides=overrides)

    if args.display:
        _add_third_person_sensor(cfg)

    return cfg


def _add_third_person_sensor(cfg) -> None:
    """Inject a third-person tracking camera into an already-loaded Habitat config.

    Imported and used by BOTH benchmark runners — this is the over-the-shoulder view the
    pygame window and the recorded videos show. Added at runtime rather than in the YAML so it
    only costs a render pass when someone is actually watching.

    The OmegaConf dance is the interesting part: a loaded Habitat config is STRUCT-mode
    (unknown keys rejected) and READONLY (no mutation). Both flags have to be lifted on the
    config *and* on the nested sensors node before a new key can be inserted, then struct mode
    is restored so the rest of the program still gets typo protection.
    """
    from habitat.config.default_structured_configs import HabitatSimRGBSensorConfig
    from omegaconf import OmegaConf

    sensor = HabitatSimRGBSensorConfig(
        uuid="third_person_sensor",
        width=640, height=480, hfov=90,
        # 2 m above + 2 m behind the robot root; pitched ~25° down (yaw π = face forward).
        # The camera is a CHILD of the robot's root node, so it tracks the robot automatically —
        # these are offsets in the robot's frame, not world coordinates.
        position=[0.0, 2.0, 2.0],
        orientation=[-0.45, 3.14159, 0.0],
        noise_model="None",       # clean image: this is for humans, not for the agent
        noise_model_kwargs={},
    )
    agent_key = list(cfg.habitat.simulator.agents.keys())[0]   # single-agent setup
    sensors_node = cfg.habitat.simulator.agents[agent_key].sim_sensors
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_readonly(cfg, False)
    OmegaConf.set_struct(sensors_node, False)
    OmegaConf.set_readonly(sensors_node, False)
    sensors_node["third_person_sensor"] = OmegaConf.structured(sensor)
    OmegaConf.set_struct(cfg, True)    # re-lock; readonly is intentionally left off
    print("[run_habitat] third-person sensor added to config")


# ── Per-episode runner ────────────────────────────────────────────────────────

def _run_episode(
    hab_env,
    episode_idx: int,
    args,
    ep_log_dir: Path,
) -> Dict[str, Any]:
    """
    Reset the env, run PromptEmbodiedAgent, return metrics dict.

    Calls hab_env.reset() to load the next episode, then reads
    hab_env.current_episode for episode metadata.
    """
    from agent.gemini_client import GeminiClient
    from agent.prompt_agent import PromptEmbodiedAgent
    from sim.habitat_toolbox import HabitatToolbox

    ep_log_dir.mkdir(parents=True, exist_ok=True)
    capture_dir = ep_log_dir / "captures"
    capture_dir.mkdir(parents=True, exist_ok=True)

    # Reset env — advances episode iterator and loads the scene
    obs     = hab_env.reset()
    episode = hab_env.current_episode

    # Task text
    task_text = args.task or _derive_task_text(episode)
    ep_id     = getattr(episode, "episode_id", str(episode_idx))
    print(f"\n[episode {episode_idx}] {ep_id}  task: {task_text!r}")

    scene_id = getattr(episode, "scene_id", f"ep{episode_idx}")

    # event_pump=None: this runner is headless (no pygame window to keep alive during the
    # blocking VLM call). run.py passes one.
    gemini_client = GeminiClient(
        api_key    = args.gemini_api_key,
        model_name = args.vlm_model,
        log_dir    = str(ep_log_dir),
        event_pump = None,
    )

    # NOTE no embedding_worker / episodic_memory / grounding_dino here — unlike the benchmark
    # runners, this one gives the agent NO retrieval stack and NO detector. `retrieve_memory`
    # will report no index and `detect` is unavailable, so the agent works from the live camera
    # and `inspect` alone. That is the deliberate baseline this script measures.
    toolbox = HabitatToolbox(
        hab_env        = hab_env,
        gemini_client  = gemini_client,
        log_dir        = str(ep_log_dir),
        capture_out_dir= str(capture_dir),
        scene_id       = scene_id,
        initial_obs    = obs,
        display        = args.display,
    )

    prompt_agent = PromptEmbodiedAgent(
        toolbox          = toolbox,
        gemini_client    = gemini_client,
        log_dir          = str(ep_log_dir),
        max_agent_steps  = args.max_agent_steps,
        history_window   = args.agent_history_steps,
        max_monitor_cycles = args.max_monitor_cycles,
    )

    t0    = time.time()
    result: Dict[str, Any] = {}
    try:
        result = prompt_agent.run(task=task_text) or {}
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"[episode {episode_idx}] agent error: {exc}")
        traceback.print_exc()
        result = {"error": str(exc)}

    elapsed = time.time() - t0

    # Habitat metrics — the objective verdict (PDDL predicates), not the agent's own opinion
    # of whether it finished. Read before teardown.
    try:
        hab_metrics = toolbox.get_metrics()
    except Exception:
        hab_metrics = {}

    record = {
        "episode_idx"   : episode_idx,
        "episode_id"    : ep_id,
        "scene_id"      : scene_id,
        "task"          : task_text,
        "elapsed_s"     : round(elapsed, 2),
        "agent_result"  : result,        # what the AGENT thought happened
        # hab_-prefixed: what HABITAT says happened. Keeping the two namespaces separate is
        # what makes it possible to compare the agent's self-assessment against ground truth.
        **{f"hab_{k}": v for k, v in hab_metrics.items()},
    }

    # Write per-episode JSON
    (ep_log_dir / "result.json").write_text(
        json.dumps(record, indent=2, default=str)
    )
    print(
        f"[episode {episode_idx}] "
        f"pddl_success={hab_metrics.get('pddl_success', '?')}  "
        f"num_steps={hab_metrics.get('num_steps', '?')}  "
        f"elapsed={elapsed:.1f}s"
    )
    return record


# ── Main ──────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run PromptEmbodiedAgent on Habitat/ReplicaCAD (OWMM comparison)"
    )

    # ── Habitat ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--hab_config",
        default="benchmark/single_agent/fetch_vlm.yaml",
        help="Path or relative-to-habitat-config path for the Habitat config YAML",
    )
    p.add_argument(
        "--dataset",
        default="data/datasets/replica_cad/single_agent_eval.json.gz",
        metavar="PATH",
        help="Dataset path relative to cwd (default: single_agent_eval.json.gz; "
             "use data/versioned_data/rearrange_dataset_v1_1.0/v1/val/tidy_house_10k_1k.json.gz "
             "for the full tidy-house val set)",
    )
    p.add_argument("--gpu_id", type=int, default=0, metavar="N")
    p.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N episodes (default: run all)",
    )

    # ── Agent ──────────────────────────────────────────────────────────────────
    p.add_argument("--vlm_model",  default="gemini-2.5-pro")
    p.add_argument(
        "--gemini_api_key",
        default=os.environ.get("GOOGLE_API_KEY", ""),
        metavar="KEY",
    )
    p.add_argument(
        "--task",
        default=None,
        metavar="TEXT",
        help="Override task text for all episodes (useful for debugging)",
    )
    p.add_argument("--max_agent_steps",     type=int, default=40)
    p.add_argument("--agent_history_steps", type=int, default=8)
    p.add_argument("--max_monitor_cycles",  type=int, default=5)

    # ── Logging ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--log_dir",
        default=str(_HERE / "runs" / "habitat_eval"),
        metavar="DIR",
    )
    p.add_argument(
        "--display",
        action="store_true",
        default=False,
        help="Show live robot-view window via cv2.imshow() (requires a display)",
    )
    p.add_argument(
        "--record_replay",
        action="store_true",
        default=False,
        help="Record GFX replay files to log_dir/replays/ for playback in habitat-viewer",
    )

    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    if not args.gemini_api_key:
        print(
            "[run_habitat] WARNING: --gemini_api_key is empty and GOOGLE_API_KEY "
            "is not set.  Gemini calls will fail."
        )

    # ── Load Habitat ──────────────────────────────────────────────────────────
    print("[run_habitat] importing Habitat …")
    get_config, HabEnv = _import_habitat()

    cfg = _build_hab_config(args, get_config)

    print("[run_habitat] creating Habitat Env …")
    hab_env = HabEnv(config=cfg)

    num_episodes = len(hab_env.episodes)
    max_eps      = args.max_episodes or num_episodes
    max_eps      = min(max_eps, num_episodes)
    print(f"[run_habitat] {num_episodes} episodes in dataset; will run {max_eps}")

    log_root = Path(args.log_dir)
    log_root.mkdir(parents=True, exist_ok=True)

    # ── Episode loop ──────────────────────────────────────────────────────────
    records: list[Dict[str, Any]] = []

    for ep_idx in range(max_eps):
        ep_log_dir = log_root / f"ep{ep_idx:04d}"

        try:
            record = _run_episode(
                hab_env     = hab_env,
                episode_idx = ep_idx,
                args        = args,
                ep_log_dir  = ep_log_dir,
            )
        except KeyboardInterrupt:
            print("\n[run_habitat] interrupted — saving partial results")
            break
        except Exception as exc:
            print(f"[run_habitat] episode {ep_idx} fatal error: {exc}")
            traceback.print_exc()
            record = {
                "episode_idx": ep_idx,
                "error"      : str(exc),
            }

        records.append(record)

    hab_env.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    if not records:
        print("[run_habitat] no episodes completed.")
        return

    n         = len(records)
    successes = sum(1 for r in records if r.get("hab_pddl_success", False))
    avg_steps = (
        sum(r.get("hab_num_steps", 0) for r in records) / n
        if n else 0.0
    )

    summary = {
        "total_episodes"   : n,
        "pddl_success_rate": successes / n,
        "avg_num_steps"    : avg_steps,
        "config"           : args.hab_config,
        "vlm_model"        : args.vlm_model,
    }
    print("\n── Summary ──────────────────────────────────────────────────────")
    print(f"  episodes       : {n}")
    print(f"  pddl_success   : {successes}/{n}  ({100*successes/n:.1f}%)")
    print(f"  avg_num_steps  : {avg_steps:.1f}")
    print("─────────────────────────────────────────────────────────────────")

    (log_root / "summary.json").write_text(json.dumps(summary, indent=2))

    # CSV for easy comparison with OWMM-Agent results tables
    csv_path = log_root / "results.csv"
    fieldnames = sorted({k for r in records for k in r})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"[run_habitat] results written to {log_root}/")


if __name__ == "__main__":
    main()
