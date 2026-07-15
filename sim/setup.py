"""
sim/setup.py — ManiSkill simulator setup helpers for ReplicaNav.

Two bring-up functions, split by what they own:
  setup_sim      the simulator itself — scene, robot, spawn, physics warm-up
  setup_workers  the memory stack — frame capture, episodic store, embedding index

They are separate because the memory workers need the scene_id (which only exists once the
scene is built), and because a run can legitimately want one without the other.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import sapien
import gymnasium as gym

from .capture import NavCaptureWorker
# NB: importing .env has side effects (sys.path injection, gym registration, ReplicaCAD
# monkey-patches). It must happen before gym.make() below can resolve the env id.
from .env import (
    WARMUP_STEPS,
    _load_replicacad_nav_positions,
    find_clear_spawn,
)
from agent.episodic_memory import EpisodicMemory
from memory.embedding import EmbeddingWorker


def setup_sim(args) -> dict:
    """
    Build the ManiSkill env, spawn the robot, and warm up physics.

    Returns a dict with keys:
        env, agent, scene_builder, uenv, action_shape, nav_verts, root_pos
    """
    print(f"Building ReplicaCAD scene idx={args.scene_idx} …")
    shader_cfg = dict(shader_pack=args.shader)

    env = gym.make(
        "ReplicaCAD_SceneManipulation-v1",
        # sensor_data (not "rgbd" or "state"): the toolbox wants the raw camera dict,
        # including depth and the camera extrinsics it needs to back-project pixels.
        obs_mode="sensor_data",
        control_mode="pd_joint_delta_pos_dual_arm",
        render_mode="human",
        robot_uids="xlerobot",
        build_config_idxs=[args.scene_idx],
        num_envs=1,
        # CPU backend: a single env with one agent gains nothing from GPU sim, and the CPU
        # path is the better-supported one for ReplicaCAD scene loading.
        sim_backend="cpu",
        # 512×512 head camera — matches the resolution the VLM prompts assume, and the
        # bbox coordinates the model is told to work in.
        sensor_configs=dict(**shader_cfg, fetch_head=dict(width=512, height=512)),
        human_render_camera_configs=shader_cfg,
        parallel_in_single_scene=False,
    )
    # Seed by scene index so a given scene always builds identically across runs.
    # reconfigure=True forces a full scene rebuild rather than a cheap state reset.
    env.reset(seed=args.scene_idx, options=dict(reconfigure=True))
    uenv          = env.unwrapped   # gym wrappers hide agent/scene_builder
    agent         = uenv.agent
    scene_builder = uenv.scene_builder
    action_shape  = env.action_space.shape[0]

    print(f"  Objects: {len(scene_builder.scene_objects)}  |  Action dim: {action_shape}")

    nav_verts = _load_replicacad_nav_positions(uenv)

    # ── Place the robot ───────────────────────────────────────────────────────
    # The patched scene builder deliberately parks the robot outside the scene (see
    # sim/env.py); this is where it actually gets put somewhere useful.
    spawn_xy = find_clear_spawn(uenv, nav_verts)
    print(f"  Spawn: ({spawn_xy[0]:.2f}, {spawn_xy[1]:.2f})")
    agent.robot.set_pose(sapien.Pose(p=[float(spawn_xy[0]), float(spawn_xy[1]), 0.0]))

    qpos = agent.robot.get_qpos().cpu().numpy().flatten()
    # Zero the three base joints. The root POSE now carries the spawn location, so the base
    # joints must start at zero — otherwise the two would compound and the robot would
    # believe it is somewhere it is not (get_robot_xy adds root_pos + qpos[:2]).
    qpos[0] = 0.0; qpos[1] = 0.0; qpos[2] = 0.0
    qpos[8] = 0.3   # head tilt ~17° below horizontal
                    # — aims the head camera at the floor/table band where objects actually
                    # are. Level, it mostly frames walls and ceiling.
    agent.robot.set_qpos(qpos)
    # Remembered separately because qpos[:2] is a displacement from this origin, not a world
    # position; every pose read goes through get_robot_xy(agent, root_pos).
    root_pos = np.array([float(spawn_xy[0]), float(spawn_xy[1])])

    # Physics warm-up: the robot was teleported into place and objects were snapped to their
    # authored poses, so there is interpenetration and unresolved contact to settle. Stepping
    # with a zero action lets gravity and the solver sort it out before the episode's first
    # observation — otherwise frame 0 catches objects mid-jitter.
    print(f"  Warming up physics ({WARMUP_STEPS} steps) …")
    for _ in range(WARMUP_STEPS):
        env.step(np.zeros(action_shape, dtype=np.float32))
    env.render()

    return dict(
        env=env,
        agent=agent,
        scene_builder=scene_builder,
        uenv=uenv,
        action_shape=action_shape,
        nav_verts=nav_verts,
        root_pos=root_pos,
    )


def setup_workers(args, capture_out_dir: str, scene_id: str) -> dict:
    """
    Create and start the capture worker, episodic memory, and embedding worker.

    Clears existing scan data unless --no_clear_scan is set.

    Returns a dict with keys:
        capture_worker, episodic_memory, embedding_worker, index_dir

    The clear-by-default is important for correctness, not just tidiness: frames and their
    embeddings are keyed by frame index, so leaving a stale index next to a fresh scan would
    let retrieval return memory_ids that map to images from a *previous* run of a possibly
    different scene. --no_clear_scan is for the case where you genuinely want to accumulate
    across runs, and it must clear both or neither.
    """
    if not args.no_clear_scan:
        for sub in ("color", "robot_xy"):
            sub_dir = os.path.join(capture_out_dir, sub)
            if os.path.isdir(sub_dir):
                shutil.rmtree(sub_dir)
                print(f"  [scan] cleared {sub_dir}")

    # capture_interval == 0 disables frame capture entirely (a run that only replays
    # existing memory doesn't need to write new frames).
    capture_worker: NavCaptureWorker | None = None
    if args.capture_interval > 0:
        capture_worker = NavCaptureWorker(capture_out_dir)

    episodic_memory = EpisodicMemory(
        memory_dir=os.path.join(capture_out_dir, "memory"))

    # Index path must match exactly what BaseToolbox.retrieve_memory reconstructs from
    # (retrieval_data_root, scene_id, retrieval_model) — the model name is in the directory
    # so that switching backbones doesn't silently reuse incompatible vectors.
    index_dir = os.path.join(
        args.retrieval_data_root, scene_id,
        f"retrieval_index_{args.retrieval_model}",
    )
    if not args.no_clear_scan and os.path.isdir(index_dir):
        shutil.rmtree(index_dir)
        print(f"  [scan] cleared retrieval index {index_dir}")

    # Returns immediately; the extractor loads on a background thread, so the sim can start
    # stepping while the (multi-second) model load happens. Queries before it is ready
    # return [] rather than blocking.
    embedding_worker = EmbeddingWorker(
        index_dir  = index_dir,
        model_name = args.retrieval_model,
        device     = "auto",
    )
    print(f"[EmbeddingWorker] loading {args.retrieval_model} extractor in background …")

    return dict(
        capture_worker=capture_worker,
        episodic_memory=episodic_memory,
        embedding_worker=embedding_worker,
        index_dir=index_dir,
    )
