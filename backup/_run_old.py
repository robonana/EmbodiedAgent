#!/usr/bin/env python3
"""
run.py — Entry point for ReplicaNav.

Usage
-----
    python -m replicanav.run --scene_idx 3 --object_type clock
    python -m replicanav.run --scene_idx 3 --agent_mode prompt --task "bring me the mug" --agent_scan_first
    python -m replicanav.run --scene_idx 3 --list_objects
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pygame

from . import rerun_log
from .agent_runner import run_agent_episode
from .env import get_actor_xy
from .navigate import run_teleop
from .sim_setup import setup_sim, setup_workers


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Navigate XLeRobot to a target object in a ReplicaCAD scene"
    )
    p.add_argument("--scene_idx",    type=int, default=0)
    p.add_argument("--object_type",  type=str, default=None,
                   help="Partial name to navigate to, e.g. 'clock', 'book_01'")
    p.add_argument("--list_objects", action="store_true")
    p.add_argument("--shader",       default="rt-fast",
                   choices=["rt-fast", "rt", "default"])

    # ── Retrieval ──────────────────────────────────────────────────────────────
    p.add_argument("--query",        nargs="+", default=None, metavar="IMAGE",
                   help="Query image path(s) for retrieval-based navigation")
    p.add_argument("--retrieval_scene", type=str, default=None,
                   help="Scene ID for retrieval index "
                        "(default: replicacad_{scene_idx:05d}_maniskill)")
    p.add_argument("--retrieval_model", default="siglip_base",
                   choices=["dinov2", "dinov2_base", "clip", "siglip", "siglip_base"],
                   help="Embedding model (must match the built index)")
    p.add_argument("--retrieval_data_root",
                   default=os.path.dirname(os.path.abspath(__file__)),
                   help="Data root for retrieval index")
    p.add_argument("--retrieval_top_k", type=int, default=5)
    p.add_argument("--retrieval_rerank", action="store_true",
                   help="Rerank retrieval candidates with Gemini")
    p.add_argument("--show_retrieval", action="store_true",
                   help="Open retrieval result grid before navigating")
    p.add_argument("--gemini_model", default="gemini-2.5-flash")
    p.add_argument("--gemini_api_key",
                   default="AIzaSyCdE4FkuAS0h6EtAvLSCXAXpKq6bpo-uB4")

    # ── Capture / memory ───────────────────────────────────────────────────────
    p.add_argument("--capture_interval", type=float, default=3.0, metavar="SECS")
    p.add_argument("--no_clear_scan", action="store_true",
                   help="Keep existing scan data instead of clearing at startup")
    p.add_argument("--rerun", action="store_true",
                   help="Launch Rerun viewer for live debug visualisation")

    # ── Agent mode ─────────────────────────────────────────────────────────────
    p.add_argument("--agent_mode",    default="none", choices=["none", "prompt"])
    p.add_argument("--task",          type=str, default=None,
                   help="Task description for --agent_mode prompt")
    p.add_argument("--vlm_model",     default="gemini-2.5-pro")
    p.add_argument("--max_agent_steps", type=int, default=40)
    p.add_argument("--agent_history_steps", type=int, default=8)
    p.add_argument("--agent_log_dir", default="runs/prompt_agent")
    p.add_argument("--disable_memory_rerank", action="store_true")
    p.add_argument("--max_monitor_cycles", type=int, default=5)
    p.add_argument("--agent_scan_first", action="store_true",
                   help="Allow WASD scanning before agent starts (press N to begin)")

    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    # ── Rerun ──────────────────────────────────────────────────────────────────
    if rerun_log.RERUN_AVAILABLE and args.rerun:
        rerun_log.RERUN_ENABLED = True
        import rerun as rr
        import rerun.blueprint as rrb
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial2DView(name="Head camera",
                                  origin="cameras/fetch_head"),
                rrb.Vertical(
                    rrb.Spatial2DView(name="Nav grid",
                                      origin="nav/grid"),
                    rrb.Spatial2DView(name="Retrieval result",
                                      origin="retrieval/result_grid"),
                ),
            ),
            collapse_panels=False,
        )
        rr.init("replicanav", spawn=True)
        rr.send_blueprint(blueprint)
        print("Rerun viewer launched.")

    # ── Simulator setup ────────────────────────────────────────────────────────
    sim = setup_sim(args)
    env           = sim["env"]
    agent         = sim["agent"]
    scene_builder = sim["scene_builder"]
    action_shape  = sim["action_shape"]
    nav_verts     = sim["nav_verts"]
    root_pos      = sim["root_pos"]

    # ── List objects early exit ────────────────────────────────────────────────
    if args.list_objects:
        print(f"\nObjects in ReplicaCAD scene {args.scene_idx}:")
        for k in sorted(scene_builder.scene_objects):
            actor = scene_builder.scene_objects[k]
            try:
                xy = get_actor_xy(actor)
                print(f"  {k:<70s}  xy=({xy[0]:.2f}, {xy[1]:.2f})")
            except Exception:
                print(f"  {k}")
        env.close()
        return

    # ── Derive scene / capture paths ───────────────────────────────────────────
    scene_id       = (args.retrieval_scene
                      or f"replicacad_{args.scene_idx:05d}_maniskill")
    capture_out_dir = os.path.join(args.retrieval_data_root, scene_id)

    # ── Workers setup ──────────────────────────────────────────────────────────
    workers          = setup_workers(args, capture_out_dir, scene_id)
    capture_worker   = workers["capture_worker"]
    episodic_memory  = workers["episodic_memory"]
    embedding_worker = workers["embedding_worker"]
    index_dir        = workers["index_dir"]

    # ── Pygame window ──────────────────────────────────────────────────────────
    pygame.init()
    WIN_W, WIN_H = 560, 340
    screen     = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption(
        "ReplicaNav  |  WASD=drive  I=image  R=render  T=text  N=go  Q=quit")
    font       = pygame.font.SysFont(None, 26)
    small_font = pygame.font.SysFont(None, 21)
    tiny_font  = pygame.font.SysFont(None, 18)

    def draw_embed_status(y: int) -> int:
        ew = embedding_worker
        if ew.is_ready:
            n_total = ew._faiss_index.ntotal if ew._faiss_index is not None else 0
            text  = (f"Embed [{ew.model_name}]  queued={ew._q.qsize()}  "
                     f"indexed={n_total}v / {ew.embedded}f")
            color = (100, 220, 100)
        else:
            text  = f"Embed [{ew.model_name}]  loading extractor …"
            color = (200, 160, 60)
        screen.blit(tiny_font.render(text[:90], True, color), (10, y))
        return y + 18

    # ── Dispatch ───────────────────────────────────────────────────────────────
    def _shutdown():
        if capture_worker is not None:
            capture_worker.flush()
        embedding_worker.stop()
        pygame.quit()
        env.close()

    try:
        if args.agent_mode == "prompt":
            result = run_agent_episode(
                args=args,
                env=env, agent=agent,
                root_pos=root_pos, action_shape=action_shape,
                scene_builder=scene_builder, nav_verts=nav_verts,
                capture_worker=capture_worker,
                episodic_memory=episodic_memory,
                embedding_worker=embedding_worker,
                capture_out_dir=capture_out_dir,
                scene_id=scene_id,
                screen=screen, font=font, small_font=small_font,
                draw_embed_status=draw_embed_status,
            )
            print(f"\n[agent] Episode result: {result}")
        else:
            run_teleop(
                args=args,
                env=env, agent=agent,
                root_pos=root_pos, action_shape=action_shape,
                scene_builder=scene_builder, nav_verts=nav_verts,
                capture_worker=capture_worker,
                episodic_memory=episodic_memory,
                embedding_worker=embedding_worker,
                capture_out_dir=capture_out_dir,
                scene_id=scene_id,
                index_dir=index_dir,
                screen=screen, font=font, small_font=small_font,
                tiny_font=tiny_font, win_w=WIN_W,
                draw_embed_status=draw_embed_status,
            )
    finally:
        _shutdown()


if __name__ == "__main__":
    main()
