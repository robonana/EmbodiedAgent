"""
scan.py — Human-in-the-loop scene scan for ReplicaNav.

Extracted from navigate.py.  Call scan_scene() before an agent episode
to let the operator WASD-drive the robot while frames are captured and
indexed into EpisodicMemory.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import pygame

from .capture import _capture_nav_frame
from .env import NAV_FWD_M_PER_STEP, NAV_ROT_RAD_PER_STEP, get_robot_xy, get_robot_yaw
from .memory import frame_to_memory_id


def scan_scene(
    env,
    agent,
    root_pos: np.ndarray,
    action_shape: int,
    screen,
    font,
    small_font,
    draw_embed_status,      # callable(y: int) → int
    capture_worker,         # NavCaptureWorker or None
    embedding_worker,       # EmbeddingWorker
    episodic_memory,        # EpisodicMemory
    capture_out_dir: str,
    capture_interval: float,
    retrieval_model: str,
    rerun_log=None,
) -> int:
    """
    Human-in-the-loop scene scan: WASD teleop while building the frame index.

    Captures a frame every `capture_interval` seconds, embeds it via EmbeddingWorker,
    and writes an EpisodicMemory entry for each frame.

    Press N to finish, Q to abort.  Returns frames captured, or -1 on abort.
    """
    frames_captured = 0
    last_capture_t: float = -1e9

    while True:
        # ── HUD ───────────────────────────────────────────────────────────────
        screen.fill((20, 20, 20))
        screen.blit(font.render(
            "Scan scene — WASD to drive, N when done, Q to quit",
            True, (100, 255, 100)), (10, 12))
        screen.blit(small_font.render(
            f"Frames captured: {frames_captured}",
            True, (200, 200, 100)), (10, 44))
        screen.blit(small_font.render(
            "WASD=drive  N=finish scan  Q=quit",
            True, (150, 150, 150)), (10, 68))
        draw_embed_status(90)
        pygame.display.flip()

        # ── WASD input ────────────────────────────────────────────────────────
        keys = pygame.key.get_pressed()
        if any([keys[pygame.K_w], keys[pygame.K_s],
                keys[pygame.K_a], keys[pygame.K_d]]):
            qp  = agent.robot.get_qpos().cpu().numpy().flatten().copy()
            yaw = qp[2]
            if keys[pygame.K_w]:
                qp[0] += NAV_FWD_M_PER_STEP * math.cos(yaw)
                qp[1] += NAV_FWD_M_PER_STEP * math.sin(yaw)
            elif keys[pygame.K_s]:
                qp[0] -= NAV_FWD_M_PER_STEP * math.cos(yaw)
                qp[1] -= NAV_FWD_M_PER_STEP * math.sin(yaw)
            if keys[pygame.K_a]:
                qp[2] += NAV_ROT_RAD_PER_STEP * 2.5
            elif keys[pygame.K_d]:
                qp[2] -= NAV_ROT_RAD_PER_STEP * 2.5
            agent.robot.set_qpos(qp)

        # ── Events ────────────────────────────────────────────────────────────
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return -1
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_q:
                    return -1
                if ev.key == pygame.K_n:
                    print(f"[scan_scene] Done — {frames_captured} frames captured.")
                    return frames_captured

        # ── Step sim + capture ────────────────────────────────────────────────
        obs, _, _, _, _ = env.step(np.zeros(action_shape, dtype=np.float32))
        env.render()
        if rerun_log is not None:
            rerun_log.log_cameras_rerun(obs)

        rgb = _capture_nav_frame(obs)
        if rgb is None:
            continue

        now = time.time()
        if capture_worker is not None and now - last_capture_t >= capture_interval:
            curr_xy   = get_robot_xy(agent, root_pos)
            curr_yaw  = get_robot_yaw(agent)
            frame_idx = capture_worker.enqueue(rgb, curr_xy, curr_yaw)
            frame_path = os.path.join(
                capture_out_dir, "color", f"{frame_idx:06d}.png")
            embedding_worker.enqueue(rgb, frame_path, curr_xy, curr_yaw)
            last_capture_t = now
            frames_captured += 1

            episodic_memory.add_entry(episodic_memory.create_entry(
                memory_id=frame_to_memory_id(frame_idx),
                image_path=frame_path,
                robot_pose=[float(curr_xy[0]), float(curr_xy[1]), float(curr_yaw)],
                embedding_model=retrieval_model,
                source_type="scan_wasd",
            ))
