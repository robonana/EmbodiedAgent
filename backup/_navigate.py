"""
navigate.py — WASD teleop and retrieval-based navigation loop for ReplicaNav.

Provides run_teleop(), which handles:
  - Manual WASD driving and live frame capture
  - Image / text / object-name target selection
  - Retrieval index readiness gate
  - A* path planning + kinematic navigation to the target
  - Post-navigation last-mile and suction controls
"""

from __future__ import annotations

import math
import os
import threading
import time

import numpy as np
import pygame

from . import rerun_log
from .capture import _capture_nav_frame
from .env import (
    ARM_REACH,
    NAV_FWD_M_PER_STEP,
    NAV_MAX_STEPS,
    NAV_ROT_RAD_PER_STEP,
    NAV_STOP_DIST,
    ROT_THRESH,
    SUCTION_RANGE,
    _suction_apply,
    find_objects,
    get_actor_xy,
    get_robot_xy,
    get_robot_yaw,
)
from .memory import frame_to_memory_id
from .nav_grid import NavGrid, kinematic_nav_step
from .retrieval import (
    RENDER_PYTHON,
    RENDER_SCRIPT,
    RETRIEVE_SCRIPT,
    SOIR_PYTHON,
    run_retrieval,
    vlm_decide_frame,
)

# Minimum distance to advance to the next waypoint
_WP_ADVANCE_DIST = max(NAV_FWD_M_PER_STEP * 3, 0.3)


def _load_thumbnail(path: str, size=(130, 97)):
    try:
        import PIL.Image
        img = PIL.Image.open(path).convert("RGB")
        img.thumbnail(size, PIL.Image.LANCZOS)
        return pygame.image.fromstring(img.tobytes(), img.size, "RGB")
    except Exception:
        return None


def run_teleop(
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
    index_dir: str,
    # UI
    screen,
    font,
    small_font,
    tiny_font,
    win_w: int,
    draw_embed_status,      # callable(y: int) → int
) -> None:
    """
    Run the interactive WASD + retrieval navigation loop until the user quits.

    Handles target selection (image / text / object name), retrieval, A* nav,
    and post-nav last-mile / suction controls.
    """
    # ── Mutable UI state ───────────────────────────────────────────────────────
    query_paths: list[str]       = list(args.query) if args.query else []
    query_text:  str | None      = None
    thumb_surf                   = _load_thumbnail(query_paths[0]) if query_paths else None

    # Thread-communication flags for the background render worker
    _render_status:      list[str]  = [""]
    _render_done:        list[bool] = [True]
    _render_needs_thumb: list[bool] = [False]
    _render_clear_text:  list[bool] = [False]

    # Suction gripper state
    suction_active: bool = False
    suction_actor        = None

    last_capture_t: float = -1e9

    # ── HUD helpers ────────────────────────────────────────────────────────────
    def draw_waiting(status_msg: str) -> None:
        screen.fill((20, 20, 20))
        screen.blit(font.render(status_msg, True, (255, 255, 100)), (10, 10))

        if query_paths:
            name   = os.path.basename(query_paths[0])
            suffix = f" +{len(query_paths)-1} more" if len(query_paths) > 1 else ""
            label, color = f"Image: {name}{suffix}", (100, 255, 100)
        elif query_text:
            truncated = query_text if len(query_text) <= 42 else query_text[:42] + "…"
            label, color = f"Text: {truncated}", (100, 200, 255)
        elif args.object_type:
            label, color = f"Object: {args.object_type}", (200, 180, 100)
        else:
            label, color = "No target set — press I or T", (200, 90, 90)
        screen.blit(font.render(label, True, color), (10, 44))

        if thumb_surf is not None:
            thumb_x = win_w - thumb_surf.get_width() - 8
            pygame.draw.rect(screen, (80, 80, 80),
                             (thumb_x - 2, 6,
                              thumb_surf.get_width() + 4,
                              thumb_surf.get_height() + 4))
            screen.blit(thumb_surf, (thumb_x, 8))

        y = 82
        for line in ["WASD  drive robot (scan mode)",
                     "I     pick query image",
                     "R     render object → set query",
                     "T     type text description",
                     "N     start navigation",
                     "Q     quit"]:
            screen.blit(small_font.render(line, True, (150, 150, 150)), (10, y))
            y += 22
        if _render_status[0]:
            screen.blit(small_font.render(
                _render_status[0], True, (255, 200, 60)), (10, y))
            y += 22
        draw_embed_status(y + 8)
        pygame.display.flip()

    # ── Main goal loop ─────────────────────────────────────────────────────────
    print("Spawned. WASD=drive/scan  N=navigate  I=image  R=render  T=text  Q=quit")

    while True:

        # ── Waiting / scan mode ────────────────────────────────────────────────
        waiting = True
        while waiting:
            draw_waiting("Scan mode — WASD:drive  N:navigate  I:image  R:render  T:text")

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        return

                    elif event.key == pygame.K_i:
                        try:
                            import tkinter as _tk
                            from tkinter import filedialog as _fd
                            _root = _tk.Tk(); _root.withdraw()
                            _root.attributes("-topmost", True)
                            paths = _fd.askopenfilenames(
                                title="Select query image(s) [Ctrl/Shift to multi-select]",
                                filetypes=[
                                    ("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tiff"),
                                    ("All files", "*"),
                                ])
                            _root.destroy()
                        except Exception as e:
                            print(f"  [tkinter] file dialog error: {e}")
                            paths = ()
                        if paths:
                            query_paths[:] = list(paths)
                            query_text     = None
                            thumb_surf     = _load_thumbnail(query_paths[0])
                            print(f"  Query images set ({len(query_paths)}): "
                                  + ", ".join(os.path.basename(p) for p in query_paths))

                    elif event.key == pygame.K_t:
                        try:
                            import tkinter as _tk
                            from tkinter.simpledialog import askstring as _ask
                            _root = _tk.Tk(); _root.withdraw()
                            _root.attributes("-topmost", True)
                            text = _ask("Object description",
                                        "Describe the object to navigate to:",
                                        parent=_root)
                            _root.destroy()
                        except Exception as e:
                            print(f"  [tkinter] text dialog error: {e}")
                            text = None
                        if text:
                            query_text     = text
                            query_paths[:] = []
                            thumb_surf     = None
                            print(f"  Query text set: {text!r}")

                    elif event.key == pygame.K_r:
                        if not _render_done[0]:
                            print("  Render already in progress …")
                        else:
                            try:
                                import tkinter as _tk
                                from tkinter.simpledialog import askstring as _ask
                                _root = _tk.Tk(); _root.withdraw()
                                _root.attributes("-topmost", True)
                                _rname = _ask("Render object",
                                              "Object name or ID (e.g. 13, bowl_01, "
                                              "frl_apartment_bowl_01):",
                                              parent=_root)
                                _root.destroy()
                            except Exception as e:
                                print(f"  [tkinter] render dialog error: {e}")
                                _rname = None
                            if _rname and _rname.strip():
                                _rname = _rname.strip()
                                _rout  = os.path.join(
                                    os.path.expanduser(
                                        "~/Projects/concept-graphs/query_images"),
                                    _rname.replace("frl_apartment_", ""))
                                _render_status[0] = f"Rendering {_rname} …"
                                _render_done[0]   = False

                                def _render_worker(_n=_rname, _o=_rout):
                                    import subprocess as _sp
                                    try:
                                        _rc = _sp.run(
                                            [RENDER_PYTHON, RENDER_SCRIPT,
                                             "--object_type", _n,
                                             "--output_dir",  _o],
                                            capture_output=False, text=True)
                                        if _rc.returncode == 0:
                                            _imgs = sorted(
                                                f for f in os.listdir(_o)
                                                if f.lower().endswith(".png"))
                                            if _imgs:
                                                _paths = [os.path.join(_o, f)
                                                          for f in _imgs]
                                                query_paths[:] = _paths
                                                _render_clear_text[0]  = True
                                                _render_needs_thumb[0] = True
                                                _render_status[0] = (
                                                    f"Rendered {len(_imgs)} views: {_n}")
                                                print(f"  [render] done — "
                                                      f"{len(_imgs)} images in {_o}")
                                            else:
                                                _render_status[0] = (
                                                    f"Render OK but no images in {_o}")
                                        else:
                                            _render_status[0] = (
                                                f"Render failed (rc={_rc.returncode})")
                                    except Exception as re:
                                        _render_status[0] = f"Render error: {re}"
                                    _render_done[0] = True

                                threading.Thread(target=_render_worker,
                                                 daemon=True).start()

                    elif event.key == pygame.K_n:
                        if not query_paths and not query_text and not args.object_type:
                            print("  No target set. Press I for image or T for text.")
                        else:
                            waiting = False

            # ── WASD driving ───────────────────────────────────────────────────
            keys = pygame.key.get_pressed()
            drive_moving = False
            if any([keys[pygame.K_w], keys[pygame.K_s],
                    keys[pygame.K_a], keys[pygame.K_d]]):
                qp  = agent.robot.get_qpos().cpu().numpy().flatten().copy()
                yaw = qp[2]
                if keys[pygame.K_w]:
                    qp[0] += NAV_FWD_M_PER_STEP * math.cos(yaw)
                    qp[1] += NAV_FWD_M_PER_STEP * math.sin(yaw)
                    drive_moving = True
                elif keys[pygame.K_s]:
                    qp[0] -= NAV_FWD_M_PER_STEP * math.cos(yaw)
                    qp[1] -= NAV_FWD_M_PER_STEP * math.sin(yaw)
                    drive_moving = True
                if keys[pygame.K_a]:
                    qp[2] += NAV_ROT_RAD_PER_STEP * 2.5
                    drive_moving = True
                elif keys[pygame.K_d]:
                    qp[2] -= NAV_ROT_RAD_PER_STEP * 2.5
                    drive_moving = True
                agent.robot.set_qpos(qp)

            # Reload thumbnail if render worker just finished
            if _render_clear_text[0]:
                query_text = None
                _render_clear_text[0] = False
            if _render_needs_thumb[0] and query_paths:
                thumb_surf = _load_thumbnail(query_paths[0])
                _render_needs_thumb[0] = False

            obs_wait, _, _, _, _ = env.step(np.zeros(action_shape, dtype=np.float32))
            env.render()
            rerun_log.log_cameras_rerun(obs_wait)
            if suction_active and suction_actor is not None:
                _suction_apply(suction_actor, agent, root_pos)

            # Periodic frame capture during scan/wait
            now_scan = time.time()
            rgb_scan = _capture_nav_frame(obs_wait)
            if rgb_scan is not None:
                if (capture_worker is not None
                        and now_scan - last_capture_t >= args.capture_interval):
                    curr_xy  = get_robot_xy(agent, root_pos)
                    curr_yaw = get_robot_yaw(agent)
                    fidx     = capture_worker.enqueue(rgb_scan, curr_xy, curr_yaw)
                    fpath    = os.path.join(
                        capture_out_dir, "color", f"{fidx:06d}.png")
                    embedding_worker.enqueue(rgb_scan, fpath, curr_xy, curr_yaw)
                    last_capture_t = now_scan

                    episodic_memory.add_entry(episodic_memory.create_entry(
                        memory_id=frame_to_memory_id(fidx),
                        image_path=fpath,
                        robot_pose=[float(curr_xy[0]), float(curr_xy[1]),
                                    float(curr_yaw)],
                        embedding_model=args.retrieval_model,
                        source_type="scan_wasd",
                    ))

        # ── Ensure retrieval index is ready ────────────────────────────────────
        if query_paths or query_text:
            _index_bin = os.path.join(index_dir, "index.bin")
            _color_dir = os.path.join(capture_out_dir, "color")
            _n_frames  = (len([f for f in os.listdir(_color_dir)
                               if f.endswith(".png")])
                          if os.path.isdir(_color_dir) else 0)

            if _n_frames == 0:
                print("  No scan frames found. WASD to scan the scene, then press N.")
                _t0 = time.time()
                while time.time() - _t0 < 4.0:
                    screen.fill((20, 20, 20))
                    screen.blit(font.render("No scan data!", True, (255, 80, 80)), (10, 12))
                    screen.blit(small_font.render(
                        "WASD to drive and scan the scene, then press N.",
                        True, (200, 160, 80)), (10, 46))
                    draw_embed_status(75)
                    pygame.display.flip()
                    env.step(np.zeros(action_shape, dtype=np.float32))
                    env.render()
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            return
                        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                            return
                continue

            if capture_worker is not None and capture_worker._q.qsize() > 0:
                draw_waiting("Flushing scan frames to disk …")
                pygame.display.flip()
                capture_worker.flush()

            if not os.path.exists(_index_bin):
                if embedding_worker.is_ready and embedding_worker._q.qsize() > 0:
                    _eq_done = threading.Event()
                    threading.Thread(
                        target=lambda: (embedding_worker.flush(), _eq_done.set()),
                        daemon=True).start()
                    while not _eq_done.wait(timeout=0.033):
                        draw_waiting("Embedding queued frames …")
                        env.step(np.zeros(action_shape, dtype=np.float32))
                        env.render()
                        for ev in pygame.event.get():
                            if ev.type == pygame.QUIT:
                                return
                            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                                return

            if not os.path.exists(_index_bin):
                import subprocess as _sp
                print(f"  [index] building subprocess for {_n_frames} frames …")
                _bdone = threading.Event()
                _brc   = [None]

                def _build_subprocess():
                    r = _sp.run([
                        SOIR_PYTHON, RETRIEVE_SCRIPT,
                        "build",
                        "--scene",     scene_id,
                        "--data_root", args.retrieval_data_root,
                        "--model",     args.retrieval_model,
                    ], capture_output=False, text=True)
                    _brc[0] = r.returncode
                    _bdone.set()

                threading.Thread(target=_build_subprocess, daemon=True).start()
                while not _bdone.wait(timeout=0.033):
                    draw_waiting(f"Building index [{args.retrieval_model}] …  please wait")
                    env.step(np.zeros(action_shape, dtype=np.float32))
                    env.render()
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            return
                        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                            return

                if _brc[0] != 0:
                    print(f"  [index] subprocess build FAILED (rc={_brc[0]}).")
                else:
                    print(f"  [index] built → {_index_bin}")

        # ── Resolve navigation target ──────────────────────────────────────────
        target_xy:    np.ndarray | None = None
        target_yaw:   float | None      = None
        target_actor                    = None

        draw_waiting("Resolving target …")
        pygame.display.flip()

        if query_paths:
            _done, _result = threading.Event(), [None]

            def _retrieval_worker():
                _result[0] = run_retrieval(
                    query_paths=query_paths,
                    scene_id=scene_id,
                    data_root=args.retrieval_data_root,
                    model=args.retrieval_model,
                    top_k=args.retrieval_top_k,
                    rerank=args.retrieval_rerank,
                    show_grid=args.show_retrieval,
                )
                _done.set()

            threading.Thread(target=_retrieval_worker, daemon=True).start()
            while not _done.wait(timeout=0.033):
                env.step(np.zeros(action_shape, dtype=np.float32))
                env.render()
                rerun_log.log_cameras_rerun(
                    env.unwrapped.get_obs() if hasattr(env.unwrapped, "get_obs") else {})
                if suction_active and suction_actor is not None:
                    _suction_apply(suction_actor, agent, root_pos)
                draw_waiting("Retrieving …  (Q to cancel)")
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        return
                    if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                        return

            pose_data = _result[0]
            if pose_data is None:
                print("ERROR: retrieval failed. Press I to pick a different image.")
                query_paths[:] = []
                thumb_surf = None
                continue

            top_k_frames = pose_data.get("top_k_frames", [])
            if len(top_k_frames) > 1:
                _vdone, _vidx, _vreason = threading.Event(), [0], [""]

                def _vlm_worker():
                    _vidx[0], _vreason[0] = vlm_decide_frame(
                        query_paths=query_paths,
                        top_k_frames=top_k_frames,
                        query_text=None,
                        api_key=args.gemini_api_key,
                        model_name="gemini-2.5-pro",
                    )
                    _vdone.set()

                threading.Thread(target=_vlm_worker, daemon=True).start()
                while not _vdone.wait(timeout=0.033):
                    env.step(np.zeros(action_shape, dtype=np.float32))
                    env.render()
                    if suction_active and suction_actor is not None:
                        _suction_apply(suction_actor, agent, root_pos)
                    draw_waiting("VLM deciding …  (Q to cancel)")
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            return
                        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                            return

                chosen = top_k_frames[_vidx[0]]
                if chosen.get("x") is not None:
                    pose_data.update(x=chosen["x"], y=chosen["y"],
                                     yaw=chosen.get("yaw"),
                                     frame_path=chosen["frame_path"],
                                     score=chosen["score"])

            target_xy  = np.array([pose_data["x"], pose_data["y"]], dtype=np.float64)
            target_yaw = pose_data.get("yaw")
            yaw_str    = (f"  yaw={math.degrees(target_yaw):.0f}°"
                          if target_yaw is not None else "")
            print(f"\n  Retrieval goal: ({target_xy[0]:.2f}, {target_xy[1]:.2f})"
                  f"{yaw_str}  score={pose_data.get('score','?'):.4f}"
                  f"  frame={os.path.basename(pose_data.get('frame_path','?'))}")

            import re as _re
            _stem     = os.path.splitext(os.path.basename(query_paths[0]))[0]
            _obj_name = _re.sub(r"_\d+$", "", _stem)
            if _obj_name:
                _matches = find_objects(scene_builder, _obj_name)
                if _matches:
                    target_actor = _matches[0][1]
                    print(f"  Target actor inferred from filename: {_matches[0][0]}")

        elif query_text:
            _txt_model = ("siglip_base"
                          if args.retrieval_model.startswith("dinov2")
                          else args.retrieval_model)
            _done, _result = threading.Event(), [None]

            def _txt_worker():
                _result[0] = run_retrieval(
                    query_paths=[],
                    scene_id=scene_id,
                    data_root=args.retrieval_data_root,
                    model=_txt_model,
                    top_k=args.retrieval_top_k,
                    show_grid=args.show_retrieval,
                    query_text=query_text,
                )
                _done.set()

            threading.Thread(target=_txt_worker, daemon=True).start()
            while not _done.wait(timeout=0.033):
                env.step(np.zeros(action_shape, dtype=np.float32))
                env.render()
                if suction_active and suction_actor is not None:
                    _suction_apply(suction_actor, agent, root_pos)
                draw_waiting("Text retrieval …  (Q to cancel)")
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        return
                    if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                        return

            pose_data = _result[0]
            if pose_data is None:
                print("ERROR: text retrieval failed.")
                query_text = None
                continue

            txt_top_k = pose_data.get("top_k_frames", [])
            if len(txt_top_k) > 1:
                _qt       = query_text
                _vdone_t, _vidx_t = threading.Event(), [0]

                def _vlm_txt_worker():
                    _vidx_t[0], _ = vlm_decide_frame(
                        query_paths=[],
                        top_k_frames=txt_top_k,
                        query_text=_qt,
                        api_key=args.gemini_api_key,
                        model_name="gemini-2.5-pro",
                    )
                    _vdone_t.set()

                threading.Thread(target=_vlm_txt_worker, daemon=True).start()
                while not _vdone_t.wait(timeout=0.033):
                    env.step(np.zeros(action_shape, dtype=np.float32))
                    env.render()
                    if suction_active and suction_actor is not None:
                        _suction_apply(suction_actor, agent, root_pos)
                    draw_waiting("VLM deciding …  (Q to cancel)")
                    for ev in pygame.event.get():
                        if ev.type == pygame.QUIT:
                            return
                        if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                            return

                chosen_t = txt_top_k[_vidx_t[0]]
                if chosen_t.get("x") is not None:
                    pose_data.update(x=chosen_t["x"], y=chosen_t["y"],
                                     yaw=chosen_t.get("yaw"),
                                     frame_path=chosen_t["frame_path"],
                                     score=chosen_t["score"])

            target_xy  = np.array([pose_data["x"], pose_data["y"]], dtype=np.float64)
            target_yaw = pose_data.get("yaw")
            yaw_str    = (f"  yaw={math.degrees(target_yaw):.0f}°"
                          if target_yaw is not None else "")
            print(f"\n  Text retrieval goal: ({target_xy[0]:.2f}, {target_xy[1]:.2f})"
                  f"{yaw_str}  score={pose_data.get('score','?'):.4f}")

            _txt_matches = find_objects(scene_builder, query_text)
            if _txt_matches:
                target_actor = _txt_matches[0][1]
                print(f"  Target actor matched by text: {_txt_matches[0][0]}")

        else:
            matches = find_objects(scene_builder, args.object_type)
            if not matches:
                print(f"ERROR: no objects matching '{args.object_type}'.")
                continue
            target_key, target_actor = matches[0]
            target_xy = get_actor_xy(target_actor)
            print(f"\n  Target: {target_key}  "
                  f"xy=({target_xy[0]:.2f}, {target_xy[1]:.2f})")

        # ── Build nav-grid and plan path ───────────────────────────────────────
        waypoints:   list[np.ndarray] = []
        nav_grid:    NavGrid | None   = None
        nav_goal_xy: np.ndarray       = target_xy.copy()

        if nav_verts is not None and len(nav_verts) >= 4:
            try:
                nav_grid   = NavGrid(nav_verts, robot_radius=0.30)
                start_xy   = get_robot_xy(agent, root_pos)
                dists      = np.linalg.norm(nav_verts - target_xy, axis=1)
                far_mask   = dists >= NAV_STOP_DIST
                d_copy     = dists.copy(); d_copy[~far_mask] = np.inf
                nav_goal_xy = (nav_verts[np.argmin(d_copy)].copy()
                               if far_mask.any()
                               else nav_verts[np.argmin(dists)].copy())
                print(f"  Nav goal: ({nav_goal_xy[0]:.2f}, {nav_goal_xy[1]:.2f})"
                      f"  ({np.linalg.norm(nav_goal_xy - target_xy):.2f}m from goal)")
                waypoints = nav_grid.plan(start_xy, nav_goal_xy)
                print(f"  Path: {len(waypoints)} waypoints")

                debug_png = os.path.expanduser(
                    f"~/Projects/robocasa_data/nav_debug_scene{args.scene_idx}.png")
                nav_grid.log_rerun_grid(waypoints, start_xy, nav_goal_xy,
                                        target_xy=target_xy, save_path=debug_png)
            except Exception as e:
                print(f"  [NavGrid] failed ({e}), falling back to direct navigation.")

        if not waypoints:
            waypoints = [target_xy.copy()]

        # ── Nav-phase HUD ──────────────────────────────────────────────────────
        def draw_status(msg: str, dist: float = 0.0, wp_idx: int = 0) -> None:
            screen.fill((20, 20, 20))
            screen.blit(font.render(msg, True, (255, 255, 100)), (10, 12))
            if query_paths:
                lbl = f"query:{os.path.basename(query_paths[0])}"
            elif query_text:
                lbl = f"text:{query_text[:30]}"
            else:
                lbl = str(args.object_type)
            screen.blit(font.render(
                f"Target: {lbl}   dist={dist:.2f}m   wp {wp_idx}/{len(waypoints)-1}",
                True, (180, 180, 180)), (10, 46))
            draw_embed_status(80)
            pygame.display.flip()

        # ── Coarse navigation loop ─────────────────────────────────────────────
        wp_idx           = 0
        nav_frames_saved = 0
        print(f"\nNavigation started → {len(waypoints)} waypoints → "
              f"nav_goal=({nav_goal_xy[0]:.2f}, {nav_goal_xy[1]:.2f})")

        for step in range(NAV_MAX_STEPS):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    return

            curr_xy    = get_robot_xy(agent, root_pos)
            curr_yaw   = get_robot_yaw(agent)
            dist_final = float(np.linalg.norm(target_xy - curr_xy))
            draw_status(f"Navigating  step={step}", dist_final, wp_idx)

            if dist_final < NAV_STOP_DIST:
                print(f"  Reached target (dist={dist_final:.2f}m) in {step} steps.")
                break

            if (wp_idx == len(waypoints) - 1
                    and np.linalg.norm(waypoints[-1] - curr_xy) < _WP_ADVANCE_DIST):
                print(f"  Best-effort stop; target {dist_final:.2f}m away.")
                break

            while wp_idx < len(waypoints) - 1:
                if np.linalg.norm(waypoints[wp_idx] - curr_xy) < _WP_ADVANCE_DIST:
                    wp_idx += 1
                else:
                    break

            wp_xy   = waypoints[wp_idx]
            delta   = wp_xy - curr_xy
            bearing = (math.atan2(delta[1], delta[0]) - curr_yaw)
            bearing = (bearing + math.pi) % (2 * math.pi) - math.pi

            kinematic_nav_step(agent, bearing)
            obs, _, _, _, _ = env.step(np.zeros(action_shape, dtype=np.float32))
            env.render()
            rerun_log.log_cameras_rerun(obs)
            if suction_active and suction_actor is not None:
                _suction_apply(suction_actor, agent, root_pos)

            now = time.time()
            rgb_nav = _capture_nav_frame(obs)
            if rgb_nav is not None:
                if (capture_worker is not None
                        and args.capture_interval > 0
                        and now - last_capture_t >= args.capture_interval):
                    fidx_n = capture_worker.enqueue(rgb_nav, curr_xy, curr_yaw)
                    fpath_n = os.path.join(
                        capture_out_dir, "color", f"{fidx_n:06d}.png")
                    embedding_worker.enqueue(rgb_nav, fpath_n, curr_xy, curr_yaw)
                    nav_frames_saved += 1
                    last_capture_t = now

            if nav_grid is not None and step % 10 == 0:
                nav_grid.log_rerun_grid(waypoints, root_pos, nav_goal_xy,
                                        target_xy=target_xy, robot_xy=curr_xy)

            if step % 50 == 0:
                print(f"  step={step:4d}  xy=({curr_xy[0]:.2f},{curr_xy[1]:.2f})"
                      f"  yaw={math.degrees(curr_yaw):.1f}°"
                      f"  wp[{wp_idx}]=({wp_xy[0]:.2f},{wp_xy[1]:.2f})"
                      f"  dist={dist_final:.2f}m")

        # ── Final orientation ──────────────────────────────────────────────────
        if target_yaw is not None:
            print(f"\nAligning to yaw {math.degrees(target_yaw):.0f}° …")
            for _ in range(NAV_MAX_STEPS):
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                        return
                current_yaw = get_robot_yaw(agent)
                bearing = ((target_yaw - current_yaw + math.pi)
                           % (2 * math.pi) - math.pi)
                if abs(bearing) < ROT_THRESH:
                    print(f"  Aligned (yaw={math.degrees(current_yaw):.1f}°)")
                    break
                qp = agent.robot.get_qpos().cpu().numpy().flatten().copy()
                qp[2] += np.clip(bearing, -NAV_ROT_RAD_PER_STEP, NAV_ROT_RAD_PER_STEP)
                agent.robot.set_qpos(qp)
                _obs_a, _, _, _, _ = env.step(np.zeros(action_shape, dtype=np.float32))
                env.render()
                rerun_log.log_cameras_rerun(_obs_a)

        # ── Flush nav captures ─────────────────────────────────────────────────
        if capture_worker is not None and nav_frames_saved > 0:
            print(f"\n  [NavCapture] flushing {capture_worker._q.qsize()} frames …")
            capture_worker.flush()
            print(f"  [NavCapture] {nav_frames_saved} new frames  "
                  f"({capture_worker.saved} total)")

        # ── Post-navigation: last mile / suction / release ─────────────────────
        print("\nNavigation done.  L=last mile   G=suction on/off   R=release   "
              "N=new target   Q=quit")

        post_nav = True
        while post_nav:
            screen.fill((20, 20, 20))
            suct_str = "HOLDING" if suction_active else "idle"
            screen.blit(font.render(
                f"Post-nav  [{suct_str}]", True, (255, 255, 100)), (10, 12))
            screen.blit(small_font.render(
                "L=last mile   G=suction   R=release   N=new target   Q=quit",
                True, (150, 150, 150)), (10, 46))
            draw_embed_status(75)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:

                    if event.key == pygame.K_q:
                        return

                    elif event.key == pygame.K_n:
                        post_nav = False

                    elif event.key == pygame.K_l:
                        # ── Last-mile navigation ───────────────────────────────
                        curr_xy = get_robot_xy(agent, root_pos)
                        live_xy = (get_actor_xy(target_actor)
                                   if target_actor is not None else target_xy.copy())
                        dir_vec = curr_xy - live_xy
                        d = float(np.linalg.norm(dir_vec))
                        dir_vec = dir_vec / d if d > 1e-6 else np.array([1.0, 0.0])
                        approach_xy = live_xy + ARM_REACH * dir_vec
                        if nav_verts is not None and len(nav_verts) >= 4:
                            approach_xy = nav_verts[
                                np.argmin(np.linalg.norm(nav_verts - approach_xy,
                                                         axis=1))].copy()

                        lm_wps    = (nav_grid.plan(curr_xy, approach_xy)
                                     if nav_grid is not None else [approach_xy])
                        lm_wp_idx = 0
                        print(f"\nLast mile: approach=({approach_xy[0]:.2f},"
                              f"{approach_xy[1]:.2f})  object=({live_xy[0]:.2f},"
                              f"{live_xy[1]:.2f})  {len(lm_wps)} waypoints")

                        for lm_step in range(NAV_MAX_STEPS):
                            for ev in pygame.event.get():
                                if ev.type == pygame.QUIT:
                                    return
                                if ev.type == pygame.KEYDOWN and ev.key == pygame.K_q:
                                    return

                            lm_xy   = get_robot_xy(agent, root_pos)
                            lm_dist = float(np.linalg.norm(live_xy - lm_xy))

                            screen.fill((20, 20, 20))
                            screen.blit(font.render(
                                f"Last mile  step={lm_step}  dist={lm_dist:.2f}m",
                                True, (255, 200, 50)), (10, 12))
                            screen.blit(small_font.render(
                                f"ARM_REACH={ARM_REACH}m   Q=quit",
                                True, (150, 150, 150)), (10, 46))
                            draw_embed_status(75)
                            pygame.display.flip()

                            if lm_dist <= ARM_REACH:
                                print(f"  Last mile done (dist={lm_dist:.2f}m).")
                                break
                            if (lm_wp_idx == len(lm_wps) - 1
                                    and np.linalg.norm(lm_wps[-1] - lm_xy)
                                    < _WP_ADVANCE_DIST):
                                print(f"  Last mile: closest navmesh point reached.")
                                break

                            while lm_wp_idx < len(lm_wps) - 1:
                                if (np.linalg.norm(lm_wps[lm_wp_idx] - lm_xy)
                                        < _WP_ADVANCE_DIST):
                                    lm_wp_idx += 1
                                else:
                                    break

                            lm_delta   = lm_wps[lm_wp_idx] - lm_xy
                            lm_yaw     = get_robot_yaw(agent)
                            lm_bearing = (math.atan2(lm_delta[1], lm_delta[0])
                                          - lm_yaw)
                            lm_bearing = ((lm_bearing + math.pi)
                                          % (2 * math.pi) - math.pi)
                            kinematic_nav_step(agent, lm_bearing)
                            _obs_lm, _, _, _, _ = env.step(
                                np.zeros(action_shape, dtype=np.float32))
                            env.render()
                            rerun_log.log_cameras_rerun(_obs_lm)
                            if suction_active and suction_actor is not None:
                                _suction_apply(suction_actor, agent, root_pos)

                    elif event.key == pygame.K_g:
                        # ── Suction toggle ─────────────────────────────────────
                        if suction_active:
                            suction_active = False
                            suction_actor  = None
                            print("  Suction OFF.")
                        else:
                            curr_xy = get_robot_xy(agent, root_pos)
                            candidates: list[tuple[float, object]] = []
                            if target_actor is not None:
                                d = float(np.linalg.norm(
                                    get_actor_xy(target_actor) - curr_xy))
                                candidates.append((d, target_actor))
                            else:
                                for actor in scene_builder.scene_objects.values():
                                    try:
                                        d = float(np.linalg.norm(
                                            get_actor_xy(actor) - curr_xy))
                                        candidates.append((d, actor))
                                    except Exception:
                                        pass
                            candidates.sort(key=lambda x: x[0])
                            if candidates and candidates[0][0] <= SUCTION_RANGE:
                                suction_active = True
                                suction_actor  = candidates[0][1]
                                _suction_apply(suction_actor, agent, root_pos)
                                print(f"  Suction ON (dist={candidates[0][0]:.2f}m)")
                            else:
                                closest = candidates[0][0] if candidates else 999.0
                                print(f"  Nothing in suction range "
                                      f"(closest={closest:.2f}m, range={SUCTION_RANGE}m).")

                    elif event.key == pygame.K_r:
                        suction_active = False
                        suction_actor  = None
                        print("  Object released.")

            # Step simulator during post-nav idle
            obs_pn, _, _, _, _ = env.step(np.zeros(action_shape, dtype=np.float32))
            env.render()
            rerun_log.log_cameras_rerun(obs_pn)

            _capture_nav_frame(obs_pn)  # keep sim rendering; frame not indexed here

            if suction_active and suction_actor is not None:
                if not _suction_apply(suction_actor, agent, root_pos):
                    print("  [suction] set_pose failed; releasing.")
                    suction_active = False
                    suction_actor  = None

        # Reset query state for next goal
        query_paths[:] = []
        query_text     = None
        thumb_surf     = None
