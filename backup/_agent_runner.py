"""
agent_runner.py — PromptEmbodiedAgent episode setup and execution.

Extracted from navigate.py.  Call run_agent_episode() after the simulator
and workers are ready (and optionally after scan_scene()).
"""

from __future__ import annotations

import os
import time

import numpy as np
import pygame

from .agent.agent_tools import AgentToolbox
from .agent.gemini_client import GeminiClient
from .agent.prompt_agent import PromptEmbodiedAgent
from .scan import scan_scene


def run_agent_episode(
    args,
    # sim
    env,
    agent,
    root_pos: np.ndarray,
    action_shape: int,
    scene_builder,
    nav_verts,
    # workers
    capture_worker,
    episodic_memory,
    embedding_worker,
    # paths
    capture_out_dir: str,
    scene_id: str,
    # UI
    screen,
    font,
    small_font,
    draw_embed_status,      # callable(y: int) → int
) -> dict:
    """
    Optionally run a pre-episode WASD scan, then run one PromptEmbodiedAgent episode.

    Returns the episode result dict from PromptEmbodiedAgent.run().
    """
    if not args.task:
        print("ERROR: --agent_mode prompt requires --task <description>")
        return {"success": False, "error": "no task"}

    # ── Optional pre-episode scan ──────────────────────────────────────────────
    if args.agent_scan_first:
        print("\n[agent] Manual scan enabled — WASD to scan, N to start agent")
        result = scan_scene(
            env=env,
            agent=agent,
            root_pos=root_pos,
            action_shape=action_shape,
            screen=screen,
            font=font,
            small_font=small_font,
            draw_embed_status=draw_embed_status,
            capture_worker=capture_worker,
            embedding_worker=embedding_worker,
            episodic_memory=episodic_memory,
            capture_out_dir=capture_out_dir,
            capture_interval=args.capture_interval,
            retrieval_model=args.retrieval_model,
        )
        if result == -1:
            return {"success": False, "error": "user quit during scan"}

    # ── Pygame quit/status callback used during the agent loop ────────────────
    _quit = [False]

    def _event_callback():
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                _quit[0] = True
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                _quit[0] = True
        screen.fill((10, 10, 30))
        screen.blit(font.render(
            "Agent running — Q to quit", True, (100, 200, 255)), (10, 12))
        screen.blit(small_font.render(
            f"Task: {args.task[:55]}", True, (200, 200, 100)), (10, 44))
        draw_embed_status(68)
        pygame.display.flip()
        if _quit[0]:
            raise KeyboardInterrupt("User quit")

    # ── Build agent components ─────────────────────────────────────────────────
    ep_log_dir = os.path.join(
        args.agent_log_dir,
        f"scene{args.scene_idx:02d}_" + time.strftime("%Y%m%d_%H%M%S"),
    )

    gemini_client = GeminiClient(
        api_key    = args.gemini_api_key,
        model_name = args.vlm_model,
        log_dir    = ep_log_dir,
    )

    toolbox = AgentToolbox(
        env=env,
        agent=agent,
        root_pos=root_pos,
        action_shape=action_shape,
        scene_builder=scene_builder,
        nav_verts=nav_verts,
        capture_out_dir=capture_out_dir,
        gemini_client=gemini_client,
        log_dir=ep_log_dir,
        embedding_worker=embedding_worker,
        episodic_memory=episodic_memory,
        retrieval_model=args.retrieval_model,
        retrieval_data_root=args.retrieval_data_root,
        scene_id=scene_id,
        event_callback=_event_callback,
    )

    prompt_agent = PromptEmbodiedAgent(
        toolbox=toolbox,
        gemini_client=gemini_client,
        log_dir=ep_log_dir,
        max_agent_steps=args.max_agent_steps,
        history_window=args.agent_history_steps,
        max_monitor_cycles=args.max_monitor_cycles,
    )

    print(f"\n[agent] Starting PromptEmbodiedAgent")
    print(f"  task={args.task!r}")
    print(f"  model={args.vlm_model}  max_steps={args.max_agent_steps}")

    try:
        return prompt_agent.run(task=args.task)
    except KeyboardInterrupt:
        print("\n[agent] Interrupted by user.")
        return {"success": False, "answer": None}
    except Exception as e:
        import traceback
        print(f"\n[agent] Episode error: {e}")
        traceback.print_exc()
        return {"success": False, "error": str(e)}
