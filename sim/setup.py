"""
sim/setup.py — ManiSkill simulator setup helpers for ReplicaNav.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import sapien
import gymnasium as gym

from .capture import NavCaptureWorker
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
        obs_mode="sensor_data",
        control_mode="pd_joint_delta_pos_dual_arm",
        render_mode="human",
        robot_uids="xlerobot",
        build_config_idxs=[args.scene_idx],
        num_envs=1,
        sim_backend="cpu",
        sensor_configs=dict(**shader_cfg, fetch_head=dict(width=512, height=512)),
        human_render_camera_configs=shader_cfg,
        parallel_in_single_scene=False,
    )
    env.reset(seed=args.scene_idx, options=dict(reconfigure=True))
    uenv          = env.unwrapped
    agent         = uenv.agent
    scene_builder = uenv.scene_builder
    action_shape  = env.action_space.shape[0]

    print(f"  Objects: {len(scene_builder.scene_objects)}  |  Action dim: {action_shape}")

    nav_verts = _load_replicacad_nav_positions(uenv)

    spawn_xy = find_clear_spawn(uenv, nav_verts)
    print(f"  Spawn: ({spawn_xy[0]:.2f}, {spawn_xy[1]:.2f})")
    agent.robot.set_pose(sapien.Pose(p=[float(spawn_xy[0]), float(spawn_xy[1]), 0.0]))
    qpos = agent.robot.get_qpos().cpu().numpy().flatten()
    qpos[0] = 0.0; qpos[1] = 0.0; qpos[2] = 0.0
    qpos[8] = 0.3   # head tilt ~17° below horizontal
    agent.robot.set_qpos(qpos)
    root_pos = np.array([float(spawn_xy[0]), float(spawn_xy[1])])

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
    """
    if not args.no_clear_scan:
        for sub in ("color", "robot_xy"):
            sub_dir = os.path.join(capture_out_dir, sub)
            if os.path.isdir(sub_dir):
                shutil.rmtree(sub_dir)
                print(f"  [scan] cleared {sub_dir}")

    capture_worker: NavCaptureWorker | None = None
    if args.capture_interval > 0:
        capture_worker = NavCaptureWorker(capture_out_dir)

    episodic_memory = EpisodicMemory(
        memory_dir=os.path.join(capture_out_dir, "memory"))

    index_dir = os.path.join(
        args.retrieval_data_root, scene_id,
        f"retrieval_index_{args.retrieval_model}",
    )
    if not args.no_clear_scan and os.path.isdir(index_dir):
        shutil.rmtree(index_dir)
        print(f"  [scan] cleared retrieval index {index_dir}")

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
