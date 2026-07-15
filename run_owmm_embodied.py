#!/usr/bin/env python3
"""
run_owmm_embodied.py — Run PromptEmbodiedAgent on the OWMM-Agent benchmark dataset.

Loads episodes from sat_TEST_YCB_30scene_head_rgb (same as episodic_eval_owmmvlm.py)
and runs EmbodiedAgent's full reasoning loop (Gemini + detect/inspect/navigate/
base_move/manipulate tools), reporting the same PDDL metrics for direct comparison.

Key differences from OWMM-Agent:
  - VLM: Gemini with rich chain-of-thought tool loop (vs OWMM-VLM single action call)
  - Scene context: pre-collected scene graphs pre-populated into FAISS memory
  - Grasping: grasp_mgr.snap_to_obj() (vs IK pixel pick)
  - Navigation: oracle_nav_coord_action (same)

Usage (from EmbodiedAgent/ root, with habitat conda env active):
    python run_owmm_embodied.py --episode_ids 1043 1245 2116 --max_agent_steps 40
    python run_owmm_embodied.py --all_episodes --max_agent_steps 40
    python run_owmm_embodied.py --max_episodes 10   # first N from test_episode_id.txt

THE BASE RUNNER. Besides serving the OWMM benchmark itself, this module owns the per-episode
pipeline that run_ovmm_embodied.py and train/habitat_env_server.py both reuse:

    _run_episode()  build env → build memory (scan / explore / pre-populate) → build toolbox
                    → run PromptEmbodiedAgent → collect PDDL metrics

Three functions are treated as HOOKS — module-level names that other runners REBIND to change
the benchmark without touching the pipeline:

    _task_prompt(ep_id)                     the task sentence
    _build_episode_config(ep_id, ...)       the Habitat config
    _prepopulate_memory(ep_id, ...)         pre-collected scene memory, if any

They must therefore be looked up through the module (`base._task_prompt(...)`) rather than
captured at import; see run_ovmm_embodied.py for the patching side of the contract.
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
from typing import Any, Dict, List, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

print("[run_owmm_embodied] starting …", flush=True)

import numpy as np

# ── Path injection ────────────────────────────────────────────────────────────
# The habitat-lab FORK vendored under OWMM-Agent/ is not pip-installed, and it must SHADOW any
# installed habitat — it carries the OWMM task/sensor registrations and the ArmAction changes
# this agent depends on. Hence sys.path.insert(0, …) rather than append.
_HERE      = Path(__file__).resolve().parent
_OWMM_ROOT = _HERE / "OWMM-Agent" / "sim" / "habitat-lab"
_HAB_LAB   = str(_OWMM_ROOT / "habitat-lab")
_HAB_BASE  = str(_OWMM_ROOT / "habitat-baselines")
_HAB_MAS   = str(_OWMM_ROOT / "habitat-mas")

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
for _p in (_HAB_LAB, _HAB_BASE, _HAB_MAS):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

# OWMM dataset name (under habitat-lab/data/datasets/). Override with --dataset.
_DATASET_NAME = "sat_TEST_YCB_30scene_head_rgb"
_OWMM_DATASET_ROOT = _OWMM_ROOT / "data" / "datasets" / _DATASET_NAME
_RENDER = os.environ.get("HABITAT_RENDER", "0") == "1"


def _set_dataset(name: str) -> None:
    """Point the runner at a different dataset dir (e.g. the train set)."""
    global _DATASET_NAME, _OWMM_DATASET_ROOT
    _DATASET_NAME = name
    _OWMM_DATASET_ROOT = _OWMM_ROOT / "data" / "datasets" / name


# ── Dataset helpers ───────────────────────────────────────────────────────────

def _load_episode_ids(args) -> List[int]:
    """Which episodes to run: explicit list, else the dataset's id file, else infer from disk."""
    if args.episode_ids:
        return [int(x) for x in args.episode_ids]
    # Prefer an explicit id list file (test_/train_episode_id.txt); otherwise
    # (e.g. the train set ships none) derive ids from the extracted image/ dirs.
    # The directory scan is the fallback because episode data is laid out one directory per
    # episode, NAMED for its id — so the filesystem is itself an index.
    ids: List[int] = []
    for fname in ("test_episode_id.txt", "train_episode_id.txt", "episode_id.txt"):
        txt = _OWMM_DATASET_ROOT / fname
        if txt.exists():
            ids = [int(l.strip()) for l in txt.read_text().splitlines()
                   if l.strip().isdigit()]
            break
    if not ids:
        img_dir = _OWMM_DATASET_ROOT / "image"
        if img_dir.is_dir():
            ids = sorted(int(p.name) for p in img_dir.iterdir()
                         if p.is_dir() and p.name.isdigit())
    if args.max_episodes:
        ids = ids[: args.max_episodes]
    return ids


def _task_prompt(ep_id: int) -> str:
    """HOOK. Read natural-language task description from the dataset's task_prompt.json.

    Returns "" when there is no entry (e.g. the train set ships no
    task_prompt.json) — the caller then derives the task from the loaded
    Habitat PDDL episode instead.

    Rebound by run_ovmm_embodied.py to build the sentence from OVMM's category triple.
    """
    # task_prompt.json lives at .../<dataset>/image/task_prompt.json
    task_json = _OWMM_DATASET_ROOT / "image" / "task_prompt.json"
    if task_json.exists():
        for entry in json.loads(task_json.read_text()):
            if str(entry.get("image_number", "")) == str(ep_id):
                return entry.get("task_description", "")
    return ""


def _scene_graph_images(ep_id: int) -> List[Path]:
    """Return ordered list of pre-collected scene graph image paths for this episode.

    The 8 images OWMM-Agent hands its VLM as a fixed prompt: two views that matter (the
    target's receptacle and the goal's) plus six random scene views. The order is fixed and
    meaningful, so it is spelled out rather than globbed. Missing files are skipped.
    """
    ep_dir = _OWMM_DATASET_ROOT / "image" / str(ep_id)
    names = [
        "target_rec.png", "goal_rec.png",
        "random_scene_graph_2.png", "random_scene_graph_3.png",
        "random_scene_graph_4.png", "random_scene_graph_5.png",
        "random_scene_graph_6.png", "random_scene_graph_7.png",
    ]
    return [ep_dir / n for n in names if (ep_dir / n).exists()]


# ── Habitat helpers ───────────────────────────────────────────────────────────

def _import_habitat():
    """Import habitat and return (get_config, Env).

    Deferred into a function rather than done at module import for two reasons: it takes tens
    of seconds, and it must not happen until AFTER the pygame display subprocess has been
    forked (habitat-sim's EGL init is incompatible with forking a GLX child afterwards).

    The inner imports are purely for their REGISTRATION side effects — they populate habitat's
    global task/sensor/action registries with the rearrange task and OWMM's custom sensors, and
    are never referenced by name. Wrapped in try/except so a config that needs none of them
    still runs.
    """
    import habitat  # noqa — triggers registration
    from habitat.config.default import get_config
    from habitat.core.env import Env
    try:
        import habitat.tasks.rearrange        # noqa
        import habitat.tasks.rearrange.vlm    # noqa
        import habitat_mas.tasks              # noqa — register OWMM sensors
    except ImportError:
        pass
    return get_config, Env


def _build_episode_config(ep_id: int, args, get_config):
    """HOOK. Create a Habitat config pointing to this episode's scene_graph.gz.

    Must be called with cwd = OWMM-Agent/sim/habitat-lab so that Hydra's
    relative config search paths resolve correctly.

    Rebound by run_ovmm_embodied.py, which additionally points the object paths at OVMM's
    assets and deletes the RL sensors.
    """
    # data_path is relative to cwd (OWMM-Agent/sim/habitat-lab)
    data_path = (
        f"data/datasets/{_DATASET_NAME}"
        f"/image/{ep_id}/scene_graph.gz"
    )
    overrides = [
        f"habitat.dataset.data_path={data_path}",
        f"habitat.seed={ep_id}",     # deterministic per episode
        # Real physics: objects must fall when released and be held by contact. The benchmark
        # config defaults to kinematic for speed, which would make manipulation cosmetic.
        "habitat.simulator.kinematic_mode=False",
        "habitat.simulator.step_physics=True",
        f"habitat.simulator.habitat_sim_v0.gpu_device_id={args.gpu_id}",
    ]
    # fetch_vlm.yaml lives in habitat-lab's benchmark config dir
    cfg = get_config("benchmark/single_agent/fetch_vlm.yaml", overrides=overrides)

    # Inject a third-person camera when rendering is enabled (it costs a render pass, so only
    # add it when a human is actually going to look at it).
    if _RENDER or args.display:
        from run_habitat import _add_third_person_sensor
        _add_third_person_sensor(cfg)

    return cfg


# ── Memory pre-population ─────────────────────────────────────────────────────

def _prepopulate_memory(ep_id: int, embedding_worker, episodic_memory):
    """
    HOOK. Pre-index the 8 OWMM pre-collected scene graph images into FAISS.

    This gives EmbodiedAgent the same scene-level knowledge that OWMM-Agent
    encodes in its 8-image prompt, but accessible via retrieve_memory() tool.

    That difference is the fair-comparison argument: OWMM force-feeds all 8 images into every
    prompt, while here they go into MEMORY and the agent must decide to retrieve them. Same
    information, actively rather than passively supplied.

    Rebound to a no-op by run_ovmm_embodied.py — OVMM ships no such images, so that agent
    scans the scene for itself.
    """
    from PIL import Image
    from agent.schemas import MemoryEntry, MemorySource, SensorData, EmbeddingRefs
    import datetime

    images = _scene_graph_images(ep_id)
    for i, img_path in enumerate(images):
        if not img_path.exists():
            continue
        try:
            rgb = np.array(Image.open(img_path).convert("RGB"))
            # Pose is ZERO for all of these — the dataset ships the images without the camera
            # poses they were taken from. So they are retrievable and inspectable, but NOT
            # navigable: `navigate` to one of these ids would drive to the origin. That is a
            # real limitation of the OWMM data, not an oversight here.
            embedding_worker.enqueue(
                rgb        = rgb,
                frame_path = str(img_path),
                robot_xy   = np.array([0.0, 0.0]),
                robot_yaw  = 0.0,
            )
            entry = MemoryEntry(
                # "sg_" prefix (not "mem_"): these are scene-graph images, not captured
                # frames, and the distinct id space keeps them from colliding with the
                # frame-index-derived ids the capture pipeline mints.
                memory_id = f"sg_{ep_id}_{i}",
                sensor    = SensorData(
                    image_path = str(img_path),
                    robot_pose = [0.0, 0.0, 0.0],
                    timestamp  = datetime.datetime.now().isoformat(),
                ),
                embeddings = EmbeddingRefs(),
                source     = MemorySource(source_type="scene_graph", episode_id=str(ep_id)),
            )
            episodic_memory.add_entry(entry)
        except Exception as e:
            print(f"[prepopulate] skipping {img_path.name}: {e}", flush=True)

    embedding_worker.flush()
    print(f"[prepopulate] indexed {len(images)} scene graph images for ep {ep_id}", flush=True)


# ── VLM CLI args ──────────────────────────────────────────────────────────────

# Default gemini policy model (overridden to a local OpenAI model when needed).
_GEMINI_DEFAULT_MODEL = "models/gemini-3.5-flash"
_OPENAI_DEFAULT_MODEL = "Qwen3.5-9B"


def _add_vlm_args(p):
    """Add --vlm_service / --vlm_base_url / --vlm_api_key (env-overridable)."""
    p.add_argument("--vlm_service",
                   default=os.environ.get("VLM_SERVICE", "gemini"),
                   choices=["gemini", "openai"],
                   help="Policy VLM backend: 'gemini' (google.generativeai) or "
                        "'openai' (OpenAI /v1/chat/completions, e.g. a local "
                        "Qwen server). Env: VLM_SERVICE")
    p.add_argument("--vlm_base_url",
                   default=os.environ.get("VLM_BASE_URL", "http://localhost:23333/v1"),
                   help="Base URL for --vlm_service openai (default: "
                        "http://localhost:23333/v1). Env: VLM_BASE_URL")
    p.add_argument("--vlm_api_key",
                   default=os.environ.get("VLM_API_KEY"),
                   help="Bearer token for --vlm_service openai (optional). "
                        "Env: VLM_API_KEY")


def _normalize_vlm_args(args):
    """When using the openai backend, swap the gemini default model for the
    local default unless the user explicitly passed --vlm_model.

    Detecting "the user didn't pass one" by comparing against the Gemini default is a little
    sly, but argparse offers no cleaner way to distinguish an unset flag from one explicitly
    set to its default. The failure mode it prevents is real: --vlm_service openai with a
    Gemini model name would be sent verbatim to the local server, which 404s.
    """
    if getattr(args, "vlm_service", "gemini") == "openai" \
            and args.vlm_model == _GEMINI_DEFAULT_MODEL:
        args.vlm_model = _OPENAI_DEFAULT_MODEL
    return args


# ── VLM client factory ────────────────────────────────────────────────────────

def _build_vlm_client(args, log_dir: str):
    """Construct the policy VLM client based on --vlm_service.

    gemini  → agent.gemini_client.GeminiClient (google.generativeai)
    openai  → agent.openai_client.OpenAIClient (OpenAI /v1/chat/completions,
              e.g. a locally-served Qwen at http://localhost:23333/v1)

    ONE client serves both roles: it is the policy (the agent's decisions) AND the tool-internal
    VLM (inspect / rerank). The banner spells that out because it surprises people — switching
    to a local model swaps out the reasoning behind the tools too, not just the action choice.
    (train/habitat_env_server.py deliberately separates them: there the policy is the model
    under training while the tools keep a frozen VLM, so the environment doesn't drift.)
    """
    service = getattr(args, "vlm_service", "gemini")
    if service == "openai":
        from agent.openai_client import OpenAIClient
        base_url = getattr(args, "vlm_base_url", "http://localhost:23333/v1")
        print(f"[VLM] backend=openai  model={args.vlm_model}  endpoint={base_url}  "
              f"→ ALL policy/inspect/rerank calls use this local VLM", flush=True)
        return OpenAIClient(
            base_url   = base_url,
            model_name = args.vlm_model,
            api_key    = getattr(args, "vlm_api_key", None),
            log_dir    = log_dir,
        )
    print(f"[VLM] backend=gemini  model={args.vlm_model}  "
          f"→ ALL policy/inspect/rerank calls use Gemini", flush=True)
    from agent.gemini_client import GeminiClient
    return GeminiClient(
        api_key    = args.gemini_api_key,
        model_name = args.vlm_model,
        log_dir    = log_dir,
    )


def _warm_protobuf_main_thread():
    """Initialise protobuf's C extension on the MAIN thread.

    The EmbeddingWorker loads its SigLIP tokenizer on a background thread, which
    is the first code to import protobuf (via sentencepiece_model_pb2). Loading
    protobuf's C extension for the first time off the main thread — concurrently
    with PIL/pygame — segfaults (pygame_parachute). With the Gemini backend this
    was masked because google.generativeai imported protobuf on the main thread
    first; the OpenAI backend never imports it, so do it explicitly here. Cheap
    and idempotent, so we always run it regardless of --vlm_service.
    """
    try:
        import google.protobuf.descriptor_pool  # noqa: F401  (init C ext)
        import sentencepiece  # noqa: F401
    except Exception as exc:  # pragma: no cover - best-effort warmup
        print(f"[warmup] protobuf/sentencepiece preload skipped: {exc}", flush=True)


# ── Per-episode runner ────────────────────────────────────────────────────────

def _run_episode(ep_id: int, args, get_config, HabEnv) -> Dict[str, Any]:
    """THE SHARED PIPELINE. One episode, end to end, for every Habitat-based runner.

    Order of operations, and none of it is arbitrary:
      1. clean the episode dir (stale index + fresh frames = memory_ids pointing at old images)
      2. start the embedding worker and WAIT for it — its CUDA context must be created before
         habitat-sim's EGL context, or the driver segfaults (see the comment inline)
      3. build the Habitat env (needs cwd = the OWMM root for hydra)
      4. build memory: pre-populate / frontier-explore / random scan — whichever the args ask
      5. run PromptEmbodiedAgent to completion
      6. collect Habitat's PDDL metrics as the objective score

    Steps 1, 4 and the task text all go through the module-level hooks, which is what lets
    run_ovmm_embodied.py reuse this wholesale.
    """
    from agent.prompt_agent import PromptEmbodiedAgent
    from agent.episodic_memory import EpisodicMemory
    from memory.embedding import EmbeddingWorker
    from sim.habitat_toolbox import HabitatToolbox

    # task_prompt.json (test set) takes priority; for datasets without it
    # (train set) we derive the task from the PDDL episode after reset below.
    # NOTE: bare `_task_prompt` — resolved through module globals at CALL time, so a runner
    # that rebound it (run_ovmm_embodied) gets its own version here.
    task_text = args.task or _task_prompt(ep_id)

    log_dir     = Path(args.log_dir) / f"ep{ep_id:04d}"
    capture_dir = log_dir / "captures"
    memory_dir  = log_dir / "memory"
    # index_dir MUST match what retrieve_memory's gate checks:
    #   retrieval_data_root / scene_id / retrieval_index_<model>
    # (retrieval_data_root=capture_dir, scene_id=str(ep_id))
    index_dir   = capture_dir / str(ep_id) / f"retrieval_index_{args.retrieval_model}"
    # Start each episode from a fully clean slate: remove the ENTIRE episode
    # directory left over from any previous run of this episode — captures,
    # FAISS index, episodic memory, observation images, localization crops,
    # explore video, trajectory logs, result.json. This runs before anything
    # is written for this episode, so it only deletes prior-run artifacts.
    import shutil
    if log_dir.exists():
        shutil.rmtree(log_dir, ignore_errors=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    capture_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    # ── Workers ───────────────────────────────────────────────────────────────
    # Must run BEFORE EmbeddingWorker starts its background extractor thread.
    _warm_protobuf_main_thread()
    embedding_worker = EmbeddingWorker(
        index_dir  = str(index_dir),
        model_name = args.retrieval_model,
        device     = "auto",
    )
    episodic_memory = EpisodicMemory(memory_dir=str(memory_dir))

    # Block until the embedding worker has finished loading its extractor (which
    # initialises a CUDA context on its background thread) BEFORE habitat-sim
    # creates its GL/EGL context on the main thread. Two threads initialising GPU
    # contexts simultaneously segfaults the NVIDIA driver (manifests as a crash
    # inside habitat_sim _config_backend). Serialising them here removes the
    # race. Bounded wait: on extractor-load failure _ready is never set, so we
    # fall through after the timeout rather than hang.
    _t0 = time.time()
    while not embedding_worker.is_ready and (time.time() - _t0) < 180.0:
        time.sleep(0.1)
    if not embedding_worker.is_ready:
        print("[run_owmm_embodied] WARNING: embedding extractor not ready after "
              "180s; continuing (retrieval may be degraded)", flush=True)

    # ── Habitat env ───────────────────────────────────────────────────────────
    # Hydra config search paths are relative to habitat-lab's working dir
    _prev_cwd = os.getcwd()
    os.chdir(str(_OWMM_ROOT))
    try:
        cfg = _build_episode_config(ep_id, args, get_config)
        hab_env = HabEnv(config=cfg)
    finally:
        os.chdir(_prev_cwd)
    obs = hab_env.reset()

    # Derive the task from the PDDL episode when no task_prompt.json entry exists
    # (e.g. the train set). Reuses run_habitat's RearrangeEpisode-based builder.
    if not task_text:
        try:
            from run_habitat import _derive_task_text
            task_text = _derive_task_text(hab_env.current_episode)
        except Exception as exc:
            print(f"[ep {ep_id}] task-derive failed: {exc}", flush=True)
            task_text = "Pick up the object and place it at the goal location."
    print(f"\n[ep {ep_id}] task: {task_text!r}", flush=True)

    # ── Agent ─────────────────────────────────────────────────────────────────
    gemini_client = _build_vlm_client(args, str(log_dir))

    # GroundingDINO for sharp open-set bounding boxes (used by object
    # localization); falls back to Gemini `inspect` if unavailable.
    gdino = None
    if not getattr(args, "no_gdino", False):
        try:
            from agent.grounding import GroundingDINODetector
            gdino = GroundingDINODetector()
            print("[gdino] GroundingDINO enabled for object localization", flush=True)
        except Exception as exc:
            print(f"[gdino] init failed, using Gemini inspect fallback: {exc}", flush=True)

    toolbox = HabitatToolbox(
        hab_env          = hab_env,
        gemini_client    = gemini_client,
        grounding_dino   = gdino,
        log_dir          = str(log_dir),
        capture_out_dir  = str(capture_dir),
        scene_id         = str(ep_id),
        embedding_worker = embedding_worker,
        episodic_memory  = episodic_memory,
        retrieval_model  = args.retrieval_model,
        retrieval_data_root = str(capture_dir),
        initial_obs      = obs,
        display          = args.display or (os.environ.get("HABITAT_RENDER", "0") == "1"),
        primary_camera   = getattr(args, "obs_camera", "head"),
    )

    # ── Scene knowledge: frontier-explore / auto-scan / OWMM pre-coded images ──
    # Three mutually exclusive ways to give the agent something to retrieve from, in
    # descending order of how "earned" the knowledge is:
    #   explore   — the robot actively maps the scene and drives to informative viewpoints
    #   (default) — random navigable points, 4 yaws each; cheap, decent coverage
    #   no_scan   — no exploration at all; use OWMM's 8 hand-picked images (the hook)
    # All three end up in the same FAISS index + episodic memory, so the agent's `retrieve_memory`
    # is identical regardless — only the CONTENT of its memory differs.
    if getattr(args, "explore", False):
        toolbox.explore_frontier(
            capture_dir      = str(capture_dir),
            embedding_worker = embedding_worker,
            episodic_memory  = episodic_memory,
            episode_id       = str(ep_id),
            max_iters        = getattr(args, "explore_iters", 40),
            lam              = getattr(args, "explore_lambda", 0.5),
            max_range        = getattr(args, "explore_range", 1.5),
            min_gain         = getattr(args, "explore_min_gain", 1),
            drive            = not getattr(args, "explore_teleport", False),
            video_path       = (str(capture_dir / "explore.mp4")
                                 if getattr(args, "explore_video", False) else None),
        )
    elif args.no_scan:
        _prepopulate_memory(ep_id, embedding_worker, episodic_memory)
    else:
        toolbox.scan_scene(
            n_points         = args.scan_points,
            capture_dir      = str(capture_dir),
            embedding_worker = embedding_worker,
            episodic_memory  = episodic_memory,
            episode_id       = str(ep_id),
        )

    prompt_agent = PromptEmbodiedAgent(
        toolbox            = toolbox,
        gemini_client      = gemini_client,
        log_dir            = str(log_dir),
        max_agent_steps    = args.max_agent_steps,
        history_window     = args.agent_history_steps,
        max_monitor_cycles = args.max_monitor_cycles,
    )

    t0     = time.time()
    result: Dict[str, Any] = {}
    try:
        result = prompt_agent.run(task=task_text) or {}
    except KeyboardInterrupt:
        raise   # let the caller stop the whole sweep — this is the user, not a bug
    except Exception as exc:
        # A crashed agent still gets metrics collected below, so it scores as a failed episode
        # rather than disappearing from the results.
        print(f"[ep {ep_id}] agent error: {exc}", flush=True)
        traceback.print_exc()
        result = {"error": str(exc)}

    elapsed = time.time() - t0

    # ── Metrics ───────────────────────────────────────────────────────────────
    # THE OBJECTIVE SCORE. Habitat's own PDDL predicates decide success — not the agent's
    # opinion of whether it finished. Read BEFORE the env is closed.
    try:
        hab_metrics = toolbox.get_metrics()
    except Exception:
        hab_metrics = {}

    # Teardown order matters: the toolbox must flush its queued frames to disk before the
    # embedding worker stops, and both before the env (which owns the sim) goes away.
    toolbox.close()          # flush live-memory frames captured during the task
    embedding_worker.stop()
    hab_env.close()

    record = {
        "episode_id"     : ep_id,
        "task"           : task_text,
        "elapsed_s"      : round(elapsed, 2),
        "framework"      : "embodied",
        "vlm_model"      : args.vlm_model,
        # hab_ prefix namespaces Habitat's metrics away from our own fields. This is the
        # prefix compare_results.py has to reconcile against OWMM's unprefixed names.
        **{f"hab_{k}": v for k, v in hab_metrics.items()},
    }
    (log_dir / "result.json").write_text(json.dumps(record, indent=2, default=str))

    print(
        f"[ep {ep_id}] pddl_success={hab_metrics.get('pddl_success', '?')}  "
        f"stage1={hab_metrics.get('pddl_stage_goals.stage_1_success', '?')}  "
        f"stage2={hab_metrics.get('pddl_stage_goals.stage_2_success', '?')}  "
        f"steps={hab_metrics.get('num_steps', '?')}  elapsed={elapsed:.1f}s",
        flush=True,
    )
    return record


# ── Main ──────────────────────────────────────────────────────────────────────

def _build_args():
    p = argparse.ArgumentParser(
        description="Run EmbodiedAgent on the OWMM-Agent benchmark for comparison"
    )
    p.add_argument("--dataset", default="sat_TEST_YCB_30scene_head_rgb",
                   metavar="NAME",
                   help="Dataset dir under habitat-lab/data/datasets/ "
                        "(default: sat_TEST_YCB_30scene_head_rgb; "
                        "use sat_TRAIN_YCB_30scene_head_rgb for the train set)")
    p.add_argument("--episode_ids", nargs="*", type=int, metavar="ID",
                   help="Specific episode IDs to evaluate (space-separated)")
    p.add_argument("--all_episodes", action="store_true",
                   help="Run all episodes in the dataset's episode-id list")
    p.add_argument("--max_episodes", type=int, default=None, metavar="N",
                   help="Run first N episodes from the dataset's episode-id list")
    p.add_argument("--gpu_id",    type=int, default=0)
    p.add_argument("--vlm_model", default="models/gemini-3.5-flash",
                   help="Gemini model ID (default: models/gemini-3.5-flash)")
    p.add_argument("--gemini_api_key",
                   default=os.environ.get("GEMINI_API_KEY", ""))
    p.add_argument("--retrieval_model", default="siglip_base",
                   help="Vision model for FAISS retrieval (siglip_base, dinov2_base)")
    p.add_argument("--scan_points", type=int, default=8, metavar="N",
                   help="Random navigable points to scan; 4 yaws each (default: 8)")
    p.add_argument("--no_scan", action="store_true",
                   help="Skip auto-scan; use OWMM's 8 pre-coded scene-graph images")
    p.add_argument("--task", default=None, metavar="TEXT",
                   help="Override task text (useful for debugging a single episode)")
    p.add_argument("--max_agent_steps",     type=int, default=40)
    p.add_argument("--agent_history_steps", type=int, default=8)
    p.add_argument("--max_monitor_cycles",  type=int, default=3)
    p.add_argument("--log_dir",  default=str(_HERE / "runs" / "owmm_embodied"))
    p.add_argument("--display",  action="store_true")
    p.add_argument("--obs_camera", default="arm_workspace",
                   choices=["head", "arm_workspace"],
                   help="Agent observation camera. Default arm_workspace = the "
                        "OWMM baseline's eval camera (head view + reachability "
                        "overlay); use 'head' for a plain forward view.")
    p.add_argument("--no_gdino", action="store_true",
                   help="Disable GroundingDINO object localization (use Gemini "
                        "inspect for bounding boxes instead)")
    _add_vlm_args(p)
    return _normalize_vlm_args(p.parse_args())


def main():
    args = _build_args()
    _set_dataset(args.dataset)
    print(f"[run_owmm_embodied] dataset = {_DATASET_NAME}", flush=True)

    # Validate dataset exists
    if not _OWMM_DATASET_ROOT.exists():
        print(f"[run_owmm_embodied] ERROR: dataset not found at {_OWMM_DATASET_ROOT}", flush=True)
        print("  Download it first: see OWMM-Agent HuggingFace dataset hhyhrhy/OWMM-Agent-data", flush=True)
        sys.exit(1)

    episode_ids = _load_episode_ids(args)
    if not episode_ids:
        print("[run_owmm_embodied] no episodes selected (use --episode_ids, --max_episodes, or --all_episodes)")
        sys.exit(1)

    print(f"[run_owmm_embodied] {len(episode_ids)} episodes  model={args.vlm_model}", flush=True)

    # Start pygame display process BEFORE habitat loads (avoids EGL/GLX conflict)
    if _RENDER:
        from sim.habitat_toolbox import start_display_process
        start_display_process()

    print("[run_owmm_embodied] importing Habitat …", flush=True)
    get_config, HabEnv = _import_habitat()

    log_root = Path(args.log_dir)
    log_root.mkdir(parents=True, exist_ok=True)

    records: list[Dict[str, Any]] = []
    for ep_id in episode_ids:
        try:
            record = _run_episode(ep_id, args, get_config, HabEnv)
        except KeyboardInterrupt:
            print("\n[run_owmm_embodied] interrupted", flush=True)
            break
        except Exception as exc:
            print(f"[ep {ep_id}] fatal: {exc}", flush=True)
            traceback.print_exc()
            record = {"episode_id": ep_id, "error": str(exc), "framework": "embodied"}
        records.append(record)

    if not records:
        return

    n         = len(records)
    successes = sum(1 for r in records if r.get("hab_pddl_success", False))
    avg_steps = sum(r.get("hab_num_steps", 0) for r in records) / n

    summary = {
        "framework"        : "embodied",
        "vlm_model"        : args.vlm_model,
        "total_episodes"   : n,
        "pddl_success_rate": round(successes / n, 4),
        "avg_num_steps"    : round(avg_steps, 1),
    }

    print("\n── EmbodiedAgent on OWMM Benchmark ─────────────────────────────")
    print(f"  episodes       : {n}")
    print(f"  pddl_success   : {successes}/{n}  ({100*successes/n:.1f}%)")
    print(f"  avg_num_steps  : {avg_steps:.1f}")
    print("─────────────────────────────────────────────────────────────────")

    (log_root / "summary.json").write_text(json.dumps(summary, indent=2))

    csv_path = log_root / "results.csv"
    fieldnames = sorted({k for r in records for k in r})
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    if _RENDER:
        from sim.habitat_toolbox import stop_display_process
        stop_display_process()
    print(f"[run_owmm_embodied] results → {log_root}/", flush=True)


if __name__ == "__main__":
    main()
