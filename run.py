#!/usr/bin/env python3
"""
run.py — SceneAgent entry point.

Pipeline
--------
1. Open simulator
2. WASD scan: drive robot, frames captured + indexed automatically
3. Set query:
     I  — pick image(s) via file dialog
     R  — render object via render_object.py, images become query
     T  — type text task/description
4. N  — start agent episode with current task
5. After episode ends, back to step 2
6. Q  — quit

Usage
-----
    python -m sceneagent.run --scene_idx 3
    python -m sceneagent.run --scene_idx 3 --task "bring me the mug"
"""

from __future__ import annotations

import argparse
import math
import os
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import pygame

from .agent.episodic_memory import EpisodicMemory, frame_to_memory_id
from .agent.gemini_client import GeminiClient
from .agent.prompt_agent import PromptEmbodiedAgent
from .memory.embedding import EmbeddingWorker
from .sim.capture import NavCaptureWorker, _capture_nav_frame
from .sim.env import (
    NAV_FWD_M_PER_STEP,
    NAV_ROT_RAD_PER_STEP,
    get_robot_xy,
    get_robot_yaw,
)
from .sim.render_object import render_object_async
from .sim.setup import setup_sim
from .sim.tools import AgentToolbox


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ReplicaNav agent pipeline")
    p.add_argument("--scene_idx",    type=int, default=0)
    p.add_argument("--shader",       default="rt-fast",
                   choices=["rt-fast", "rt", "default"])
    p.add_argument("--task",         type=str, default=None,
                   help="Initial task text (skips T dialog on first run)")

    # ── Capture / embedding ────────────────────────────────────────────────────
    p.add_argument("--capture_interval", type=float, default=3.0, metavar="SECS")

    p.add_argument("--retrieval_model", default="siglip_base",
                   choices=["dinov2", "dinov2_base", "clip", "siglip", "siglip_base"])
    p.add_argument("--retrieval_data_root",
                   default=os.path.dirname(os.path.abspath(__file__)))

    # ── Agent ──────────────────────────────────────────────────────────────────
    p.add_argument("--vlm_model",          default="gemini-2.5-pro")
    p.add_argument("--gemini_api_key",
                   default=os.environ.get("GOOGLE_API_KEY", "AIzaSyCdE4FkuAS0h6EtAvLSCXAXpKq6bpo-uB4"))
    p.add_argument("--grounding_dino_model", default=None, metavar="MODEL_ID",
                   help="HuggingFace model ID for GroundingDINO bbox proposals "
                        "(e.g. IDEA-Research/grounding-dino-tiny). Disabled if not set.")
    p.add_argument("--max_agent_steps",    type=int, default=40)
    p.add_argument("--agent_history_steps", type=int, default=8)
    p.add_argument("--max_monitor_cycles",  type=int, default=5)
    p.add_argument("--agent_log_dir",       default=os.path.join(_HERE, "runs", "prompt_agent"))

    return p


def _load_thumbnail(path: str, size=(130, 97)):
    try:
        import PIL.Image
        img = PIL.Image.open(path).convert("RGB")
        img.thumbnail(size, PIL.Image.LANCZOS)
        return pygame.image.fromstring(img.tobytes(), img.size, "RGB")
    except Exception:
        return None


def main() -> None:
    args = _build_arg_parser().parse_args()

    # ── Simulator ──────────────────────────────────────────────────────────────
    sim           = setup_sim(args)
    env           = sim["env"]
    agent         = sim["agent"]
    scene_builder = sim["scene_builder"]
    action_shape  = sim["action_shape"]
    nav_verts     = sim["nav_verts"]
    root_pos      = sim["root_pos"]

    # ── Derive paths ───────────────────────────────────────────────────────────
    scene_id        = f"replicacad_{args.scene_idx:05d}_maniskill"
    capture_out_dir = os.path.join(args.retrieval_data_root, scene_id)
    index_dir       = os.path.join(
        args.retrieval_data_root, scene_id,
        f"retrieval_index_{args.retrieval_model}",
    )

    # ── Workers ────────────────────────────────────────────────────────────────
    import shutil
    memory_dir = os.path.join(capture_out_dir, "memory")
    for d in (os.path.join(capture_out_dir, "color"),
              os.path.join(capture_out_dir, "robot_xy"),
              index_dir, memory_dir):
        if os.path.isdir(d):
            shutil.rmtree(d)
            print(f"  [scan] cleared {d}")

    if os.path.isdir(args.agent_log_dir):
        shutil.rmtree(args.agent_log_dir)
        print(f"  [run] cleared {args.agent_log_dir}")

    capture_worker: NavCaptureWorker | None = None
    if args.capture_interval > 0:
        capture_worker = NavCaptureWorker(capture_out_dir)

    episodic_memory  = EpisodicMemory(memory_dir=memory_dir)
    embedding_worker = EmbeddingWorker(
        index_dir  = index_dir,
        model_name = args.retrieval_model,
        device     = "auto",
    )
    print(f"[EmbeddingWorker] loading {args.retrieval_model} extractor …")

    # ── GroundingDINO detector (optional) ──────────────────────────────────────
    grounding_dino = None
    if args.grounding_dino_model:
        from .agent.grounding import GroundingDINODetector
        grounding_dino = GroundingDINODetector(model_id=args.grounding_dino_model)
        print(f"[GroundingDINO] will lazy-load {args.grounding_dino_model}")

    # ── Pygame window ──────────────────────────────────────────────────────────
    pygame.init()
    WIN_W, WIN_H = 560, 340
    screen     = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption(
        "ReplicaNav  |  WASD=scan  I=image  R=render  T=text  N=run agent  Q=quit")
    font       = pygame.font.SysFont(None, 26)
    small_font = pygame.font.SysFont(None, 21)
    tiny_font  = pygame.font.SysFont(None, 18)

    def draw_embed_status(y: int) -> int:
        ew = embedding_worker
        if ew.is_ready:
            n_vec = ew._faiss_index.ntotal if ew._faiss_index is not None else 0
            text  = (f"Embed [{ew.model_name}]  queued={ew._q.qsize()}  "
                     f"indexed={n_vec}v / {ew.embedded}f")
            color = (100, 220, 100)
        else:
            text  = f"Embed [{ew.model_name}]  loading extractor …"
            color = (200, 160, 60)
        screen.blit(tiny_font.render(text[:90], True, color), (10, y))
        return y + 18

    # ── Shared scan state ──────────────────────────────────────────────────────
    task:          str | None       = args.task
    query_images:  list[str]        = []
    thumb_surf                      = None
    last_capture_t: float           = -1e9
    render_status: list[str]        = [""]
    render_done:   list[bool]       = [True]

    def _ask_string(title: str, prompt_text: str) -> str | None:
        try:
            import tkinter as _tk
            from tkinter.simpledialog import askstring as _ask
            root = _tk.Tk(); root.withdraw()
            root.attributes("-topmost", True)
            result = _ask(title, prompt_text, parent=root)
            root.destroy()
            return result
        except Exception as e:
            print(f"  [dialog] {e}")
            return None

    def _ask_files() -> list[str]:
        try:
            import tkinter as _tk
            from tkinter import filedialog as _fd
            root = _tk.Tk(); root.withdraw()
            root.attributes("-topmost", True)
            paths = _fd.askopenfilenames(
                title="Select query image(s)",
                filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp"),
                           ("All files", "*")],
            )
            root.destroy()
            return list(paths)
        except Exception as e:
            print(f"  [dialog] {e}")
            return []

    def _run_agent_episode(task_text: str, ref_images: list[str]) -> None:
        ep_log_dir = os.path.join(
            args.agent_log_dir,
            f"scene{args.scene_idx:02d}_" + time.strftime("%Y%m%d_%H%M%S"),
        )
        gemini_client = GeminiClient(
            api_key    = args.gemini_api_key,
            model_name = args.vlm_model,
            log_dir    = ep_log_dir,
            event_pump = _event_callback,
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
            capture_worker=capture_worker,
            capture_interval=args.capture_interval,
            retrieval_model=args.retrieval_model,
            retrieval_data_root=args.retrieval_data_root,
            scene_id=scene_id,
            event_callback=_event_callback,
            grounding_dino=grounding_dino,
        )
        prompt_agent = PromptEmbodiedAgent(
            toolbox=toolbox,
            gemini_client=gemini_client,
            log_dir=ep_log_dir,
            max_agent_steps=args.max_agent_steps,
            history_window=args.agent_history_steps,
            max_monitor_cycles=args.max_monitor_cycles,
        )
        try:
            result = prompt_agent.run(task=task_text)
            print(f"\n[run] Episode result: {result}")
        except KeyboardInterrupt:
            print("\n[run] Interrupted by user.")
        except Exception as exc:
            import traceback
            print(f"\n[run] Episode error: {exc}")
            traceback.print_exc()

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
        if task:
            screen.blit(small_font.render(
                f"Task: {task[:55]}", True, (200, 200, 100)), (10, 44))
        draw_embed_status(68)
        pygame.display.flip()
        env.render()
        if _quit[0]:
            raise KeyboardInterrupt("User quit")

    def _draw_scan_hud():
        screen.fill((20, 20, 20))
        screen.blit(font.render(
            "Scan — WASD:drive  I:image  R:render  T:text  N:run agent  Q:quit",
            True, (100, 255, 100)), (10, 12))

        if query_images:
            name   = os.path.basename(query_images[0])
            suffix = f" +{len(query_images)-1}" if len(query_images) > 1 else ""
            screen.blit(small_font.render(
                f"Query images: {name}{suffix}", True, (100, 255, 100)), (10, 44))
        if task:
            screen.blit(small_font.render(
                f"Task: {task[:55]}", True, (200, 200, 100)), (10, 64))
        elif not query_images:
            screen.blit(small_font.render(
                "No task set — press T to type or N to prompt",
                True, (200, 90, 90)), (10, 44))

        if thumb_surf is not None:
            tx = WIN_W - thumb_surf.get_width() - 8
            pygame.draw.rect(screen, (80, 80, 80),
                             (tx - 2, 6, thumb_surf.get_width() + 4,
                              thumb_surf.get_height() + 4))
            screen.blit(thumb_surf, (tx, 8))

        if render_status[0]:
            screen.blit(small_font.render(
                render_status[0], True, (255, 200, 60)), (10, 88))

        draw_embed_status(114)
        pygame.display.flip()

    # ── Main loop ──────────────────────────────────────────────────────────────
    print("[run] Ready. WASD=scan  I=image  R=render  T=text  N=run agent  Q=quit")

    while not _quit[0]:
        _draw_scan_hud()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                _quit[0] = True

            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_q:
                    _quit[0] = True

                elif ev.key == pygame.K_t:
                    text = _ask_string("Task", "Enter task description for the agent:")
                    if text and text.strip():
                        task = text.strip()
                        print(f"  Task set: {task!r}")

                elif ev.key == pygame.K_i:
                    paths = _ask_files()
                    if paths:
                        query_images[:] = paths
                        thumb_surf = _load_thumbnail(paths[0])
                        print(f"  Query images: "
                              + ", ".join(os.path.basename(p) for p in paths))

                elif ev.key == pygame.K_r:
                    if not render_done[0]:
                        print("  Render already in progress …")
                    else:
                        name = _ask_string(
                            "Render object",
                            "Object name or ID (e.g. bowl_01, frl_apartment_bowl_01):")
                        if name and name.strip():
                            name = name.strip()
                            out_dir = os.path.join(
                                os.path.expanduser(
                                    "~/Projects/concept-graphs/query_images"),
                                name.replace("frl_apartment_", ""),
                            )
                            render_status[0] = f"Rendering {name} …"
                            render_done[0]   = False

                            def _on_done(paths, _n=name):
                                query_images[:] = paths
                                render_status[0] = f"Rendered {len(paths)} views: {_n}"
                                render_done[0]   = True

                            def _on_err(msg):
                                render_status[0] = msg
                                render_done[0]   = True

                            render_object_async(name, out_dir, _on_done, _on_err)

                elif ev.key == pygame.K_n:
                    # Determine task text
                    run_task = task
                    if not run_task:
                        if query_images:
                            run_task = ("Navigate to and interact with the object "
                                        "shown in the reference image.")
                        else:
                            run_task = _ask_string(
                                "Task", "Enter task description for the agent:")
                            if run_task:
                                run_task = run_task.strip()
                                task = run_task

                    if not run_task:
                        print("  No task set — press T or type a task when prompted.")
                    else:
                        print(f"\n[run] Starting agent: {run_task!r}")
                        _run_agent_episode(run_task, list(query_images))

        # ── WASD driving + frame capture ───────────────────────────────────────
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

        obs, _, _, _, _ = env.step(np.zeros(action_shape, dtype=np.float32))
        env.render()

        rgb = _capture_nav_frame(obs)
        if rgb is not None:
            now = time.time()
            if capture_worker is not None and now - last_capture_t >= args.capture_interval:
                curr_xy  = get_robot_xy(agent, root_pos)
                curr_yaw = get_robot_yaw(agent)
                fidx     = capture_worker.enqueue(rgb, curr_xy, curr_yaw)
                fpath    = os.path.join(capture_out_dir, "color", f"{fidx:06d}.png")
                embedding_worker.enqueue(rgb, fpath, curr_xy, curr_yaw)
                last_capture_t = now

                episodic_memory.add_entry(episodic_memory.create_entry(
                    memory_id=frame_to_memory_id(fidx),
                    image_path=fpath,
                    robot_pose=[float(curr_xy[0]), float(curr_xy[1]), float(curr_yaw)],
                    embedding_model=args.retrieval_model,
                    source_type="scan_wasd",
                ))

    # ── Shutdown ───────────────────────────────────────────────────────────────
    if capture_worker is not None:
        capture_worker.flush()
    embedding_worker.stop()
    pygame.quit()
    env.close()


if __name__ == "__main__":
    main()
