#!/usr/bin/env python3
"""train/habitat_env_server.py — serve one Habitat OVMM episode as a step-able
HTTP environment.

Habitat's conda env is Python 3.9 and verl requires >= 3.10, so the simulator
and the trainer cannot share an interpreter.  This server owns the simulator and
every tool implementation; verl's HabitatAgentLoop owns only the policy.

Stepping mirrors agent/prompt_agent.py::PromptEmbodiedAgent.run with the VLM
call replaced by an HTTP round-trip: the server renders the same policy prompt
(agent/prompts.py::build_policy_prompt) and returns it with the images the
policy would have seen.

One server drives one episode at a time — habitat-sim is not thread-safe and
holds a GL context. The blocking simulator work (reset/step) runs on a dedicated
single worker thread (state.sim_pool) so the GL context never migrates, while the
event loop stays free to answer /health and reject concurrent /reset with a fast
409. A per-session `busy` flag (with an idle lease) is the actual mutual-exclusion
primitive; the agent loop acquires a free server by trying /reset across the pool.

The `gemini_client` handed to HabitatToolbox is used ONLY for tool-internal VLM
calls (inspect / detect / rerank).  It is part of the environment, not the
policy under training — point it at a frozen model so the environment does not
drift as the policy learns.

Run (inside the habitat env, from the EmbodiedAgent root):
    python -m train.habitat_env_server --port 8100 --gpu_id 0 --split minival
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import shutil
import sys
import time
import traceback
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Importing run_ovmm_embodied patches the OWMM runner's dataset/task/config
# hooks to their OVMM variants; both modules are then used through `base`.
import run_ovmm_embodied as _ovmm  # noqa: F401  (import for its side effects)
import run_owmm_embodied as base

from agent.prompts import SYSTEM_PROMPT, build_policy_prompt
from agent.schemas import ToolAction, ToolResult, VALID_TOOLS

_SCAN_SNAPSHOT = "_post_scan"


# ── Observation encoding ──────────────────────────────────────────────────────

def _encode_image(path: str, max_side: int) -> Optional[str]:
    """Read an image, optionally downscale it, return base64 PNG.

    Multi-turn VLM RL puts one observation image into the prompt *per step*, and
    those tokens are never reclaimed.  Downscaling here is the cheapest lever on
    total context length; see README for the max_steps / image_max_side budget.
    """
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if max_side and max(img.size) > max_side:
            scale = max_side / float(max(img.size))
            img = img.resize((max(1, int(img.width * scale)),
                              max(1, int(img.height * scale))), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception as exc:  # a missing frame must not kill the episode
        print(f"[env_server] image encode failed {path}: {exc}", flush=True)
        return None


# ── Episode session ───────────────────────────────────────────────────────────

class _EpisodeSession:
    """Owns one Habitat env + toolbox and steps a single episode at a time."""

    def __init__(self, cfg, get_config, hab_env_cls):
        self.cfg = cfg
        self._get_config = get_config
        self._HabEnv = hab_env_cls

        self.ep_id: Optional[int] = None
        self.session_id: Optional[str] = None
        self.task: str = ""
        # Claimed by /reset, released on episode end or /release. verl runs several
        # AgentLoopWorker processes, so the client cannot arbitrate access to a
        # shared server pool — the server has to reject concurrent claims itself.
        self.busy = False
        # A client that dies (or whose /reset times out mid-scene-load) never learns
        # its session_id and so can never /release. Without a lease the server stays
        # busy forever and the whole pool starves. Refreshed on reset and on step.
        self.last_activity = time.time()

        self.hab_env = None
        self.toolbox = None
        self.embedding_worker = None
        self.episodic_memory = None
        self.vlm = None

        # Per-rollout state (mirrors PromptEmbodiedAgent locals)
        self.step_idx = 0
        self.transient_memory: List[str] = []
        self.current_obs: Dict[str, Any] = {}
        self.last_action: Optional[ToolAction] = None
        self.last_result: Optional[ToolResult] = None
        self.last_action_key: Optional[str] = None
        self.repeat_count = 0
        self.done = False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def _episode_args(self) -> SimpleNamespace:
        c = self.cfg
        return SimpleNamespace(
            split=c.split,
            gpu_id=c.gpu_id,
            no_drop_missing=False,
            display=False,
            task=None,
            retrieval_model=c.retrieval_model,
            scan_points=c.scan_points,
            no_scan=False,
            explore=False,
            no_gdino=c.no_gdino,
            max_agent_steps=c.max_steps,
            agent_history_steps=8,
            max_monitor_cycles=3,
            vlm_service=c.tool_vlm_service,
            vlm_base_url=c.tool_vlm_base_url,
            vlm_api_key=c.tool_vlm_api_key,
            vlm_model=c.tool_vlm_model,
            gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
        )

    def _close_agent_side(self) -> None:
        """Drop the per-rollout objects, leaving the simulator alive."""
        for closer in (
            lambda: self.toolbox and self.toolbox.close(),
            lambda: self.embedding_worker and self.embedding_worker.stop(),
        ):
            try:
                closer()
            except Exception:
                pass
        self.toolbox = self.embedding_worker = None
        self.episodic_memory = self.vlm = None

    def _close_sim(self) -> None:
        try:
            if self.hab_env:
                self.hab_env.close()
        except Exception:
            pass
        self.hab_env = None
        self.ep_id = None

    def _build_sim(self, ep_id: int) -> None:
        """Instantiate habitat-sim for `ep_id`. Expensive — scene load."""
        args = self._episode_args()
        prev_cwd = os.getcwd()
        os.chdir(str(base._OWMM_ROOT))  # hydra search paths are relative to it
        try:
            hab_cfg = base._build_episode_config(ep_id, args, self._get_config)
            self.hab_env = self._HabEnv(config=hab_cfg)
        finally:
            os.chdir(prev_cwd)
        self.ep_id = ep_id

    def _paths(self, ep_id: int) -> Dict[str, Path]:
        work = Path(self.cfg.work_root) / f"ep{ep_id:04d}"
        capture_dir = work / "captures"
        return {
            "work": work,
            "capture": capture_dir,
            "memory": work / "memory",
            "index": capture_dir / str(ep_id) / f"retrieval_index_{self.cfg.retrieval_model}",
            "snapshot": work / _SCAN_SNAPSHOT,
        }

    def _start_workers(self, ep_id: int) -> bool:
        """Restore post-scan memory (if any) and start the embedding worker.

        Returns True when this is the episode's first build (a scan is needed).

        The scene scan is by far the most expensive part of a first reset, so its
        products (episodic memory + FAISS index) are snapshotted afterwards and
        restored on every later rollout of that episode, giving each rollout
        identical starting scene knowledge.  Capture images live outside the
        snapshot and are never deleted, which is what keeps the absolute paths
        inside the restored memory entries valid.
        """
        from agent.episodic_memory import EpisodicMemory
        from memory.embedding import EmbeddingWorker

        p = self._paths(ep_id)
        first_build = not p["snapshot"].exists()
        if not first_build:
            for src, dst in ((p["snapshot"] / "memory", p["memory"]),
                             (p["snapshot"] / "index", p["index"])):
                shutil.rmtree(dst, ignore_errors=True)
                if src.exists():
                    shutil.copytree(src, dst)
        for key in ("capture", "index", "memory"):
            p[key].mkdir(parents=True, exist_ok=True)

        self.embedding_worker = EmbeddingWorker(
            index_dir=str(p["index"]), model_name=self.cfg.retrieval_model, device="auto")
        self.episodic_memory = EpisodicMemory(memory_dir=str(p["memory"]))

        t0 = time.time()
        while not self.embedding_worker.is_ready and (time.time() - t0) < 180.0:
            time.sleep(0.1)
        if not self.embedding_worker.is_ready:
            print("[env_server] WARNING: embedding extractor not ready after 180s",
                  flush=True)
        return first_build

    def _build_toolbox(self, ep_id: int, obs, first_build: bool) -> None:
        from sim.habitat_toolbox import HabitatToolbox

        args = self._episode_args()
        p = self._paths(ep_id)

        self.task = base._task_prompt(ep_id) or ""
        if not self.task:
            try:
                from run_habitat import _derive_task_text
                self.task = _derive_task_text(self.hab_env.current_episode)
            except Exception:
                self.task = "Pick up the object and place it at the goal location."

        self.vlm = base._build_vlm_client(args, str(p["work"]))

        gdino = None
        if not args.no_gdino:
            try:
                from agent.grounding import GroundingDINODetector
                gdino = GroundingDINODetector()
            except Exception as exc:
                print(f"[env_server] gdino init failed: {exc}", flush=True)

        self.toolbox = HabitatToolbox(
            hab_env=self.hab_env,
            gemini_client=self.vlm,
            grounding_dino=gdino,
            log_dir=str(p["work"]),
            capture_out_dir=str(p["capture"]),
            scene_id=str(ep_id),
            embedding_worker=self.embedding_worker,
            episodic_memory=self.episodic_memory,
            retrieval_model=args.retrieval_model,
            retrieval_data_root=str(p["capture"]),
            initial_obs=obs,
            display=False,
            primary_camera=self.cfg.obs_camera,
        )

        if first_build:
            self.toolbox.scan_scene(
                n_points=args.scan_points,
                capture_dir=str(p["capture"]),
                embedding_worker=self.embedding_worker,
                episodic_memory=self.episodic_memory,
                episode_id=str(ep_id),
            )
            p["snapshot"].mkdir(parents=True, exist_ok=True)
            for src, dst in ((p["memory"], p["snapshot"] / "memory"),
                             (p["index"], p["snapshot"] / "index")):
                shutil.rmtree(dst, ignore_errors=True)
                if src.exists():
                    shutil.copytree(src, dst)

    def reset(self, ep_id: int) -> Dict[str, Any]:
        base._warm_protobuf_main_thread()
        self._close_agent_side()

        needs_sim = self.ep_id != ep_id or self.hab_env is None
        if needs_sim:
            self._close_sim()
            if not self._paths(ep_id)["snapshot"].exists():
                shutil.rmtree(self._paths(ep_id)["work"], ignore_errors=True)

        # The embedding worker initialises a CUDA context on a background thread.
        # Let it finish BEFORE habitat-sim creates its GL/EGL context on this
        # thread: two threads initialising GPU contexts at once segfaults the
        # NVIDIA driver (surfacing as a crash inside habitat_sim _config_backend).
        first_build = self._start_workers(ep_id)

        if needs_sim:
            self._build_sim(ep_id)
        obs = self.hab_env.reset()
        self._build_toolbox(ep_id, obs, first_build)

        self.session_id = uuid.uuid4().hex[:12]
        self.last_activity = time.time()
        self.step_idx = 0
        self.transient_memory = []
        self.last_action = None
        self.last_action_key = None
        self.repeat_count = 0
        self.done = False
        self.current_obs = {}

        obs_result = self.toolbox.observe()
        self.last_result = obs_result
        if obs_result.ok:
            self.current_obs = dict(obs_result.data)

        return {
            "session_id": self.session_id,
            "task": self.task,
            "system_prompt": SYSTEM_PROMPT,
            "observation": self._observation(),
            "done": False,
        }

    # ── Observation ──────────────────────────────────────────────────────────

    def _observation(self) -> Dict[str, Any]:
        """Render the exact prompt + images PromptEmbodiedAgent would show."""
        obs_result = self.toolbox.observe()
        if obs_result.ok:
            self.current_obs = dict(obs_result.data)

        repeat_warning = None
        if self.repeat_count >= 3:
            repeat_warning = (f"You repeated the same action {self.repeat_count} "
                              f"times without progress.")

        prompt = build_policy_prompt(
            task=self.task,
            step_idx=self.step_idx + 1,
            timestamp=time.strftime("%H:%M:%S"),
            current_observation=self.current_obs,
            transient_memory=self.transient_memory,
            last_action=self.last_action.to_dict() if self.last_action else None,
            last_result=self.last_result.to_dict() if self.last_result else None,
            repeat_warning=repeat_warning,
        )

        images: List[Dict[str, str]] = []
        attached = set()
        obs_path = self.toolbox._last_image_path
        if obs_path:
            b64 = _encode_image(obs_path, self.cfg.image_max_side)
            if b64:
                images.append({
                    "label": "Current head-camera observation (forward-facing):",
                    "b64": b64,
                })
                attached.add(obs_path)

        if self.last_result and self.last_result.image_paths:
            path_to_mem = {}
            for c in (self.last_result.data or {}).get("candidates", []):
                cp, mid = c.get("rgb_path"), c.get("memory_id")
                if cp and mid:
                    path_to_mem[cp] = mid
            for i, p in enumerate(self.last_result.image_paths):
                if p in attached or not os.path.exists(p):
                    continue
                if len(images) >= self.cfg.max_images_per_turn:
                    break
                mem_id = path_to_mem.get(p)
                if mem_id:
                    label = (f"Last tool result image {i + 1} "
                             f"(= memory_id {mem_id}; navigate here with "
                             f'target={{"memory_id":"{mem_id}"}}):')
                else:
                    label = f"Last tool result image {i + 1} ({os.path.basename(p)}):"
                b64 = _encode_image(p, self.cfg.image_max_side)
                if b64:
                    images.append({"label": label, "b64": b64})
                    attached.add(p)

        return {"prompt": prompt, "images": images, "step_idx": self.step_idx}

    # ── Reward ───────────────────────────────────────────────────────────────

    def _terminal_reward(self) -> Tuple[float, Dict[str, Any]]:
        try:
            metrics = self.toolbox.get_metrics()
        except Exception:
            metrics = {}
        success = float(metrics.get("pddl_success", 0.0) or 0.0)
        reward = success * self.cfg.success_reward
        if not success and self.cfg.stage_reward:
            stage1 = float(metrics.get("pddl_stage_goals.stage_1_success", 0.0) or 0.0)
            reward += stage1 * self.cfg.stage_reward
        return reward, metrics

    # ── Step ─────────────────────────────────────────────────────────────────

    def step(self, action_dict: Dict[str, Any]) -> Dict[str, Any]:
        if self.done:
            raise RuntimeError("episode already finished; call /reset")
        self.last_activity = time.time()

        invalid = False
        if not action_dict or "tool" not in action_dict:
            invalid = True
            action = ToolAction(tool="wait", arguments={"seconds": 1},
                                rationale="No valid JSON action produced.")
        else:
            action = ToolAction.from_dict(action_dict)
            if action.tool not in VALID_TOOLS:
                invalid = True
                action = ToolAction(tool="wait", arguments={"seconds": 1},
                                    rationale=f"Tool {action.tool!r} is invalid.")

        action_key = f"{action.tool}:{json.dumps(action.arguments, sort_keys=True, default=str)}"
        self.repeat_count = self.repeat_count + 1 if action_key == self.last_action_key else 0
        self.last_action_key = action_key

        reward = -self.cfg.step_penalty
        if invalid:
            reward -= self.cfg.invalid_penalty

        done = False
        reason = ""

        if self.repeat_count >= 10:
            result = ToolResult(ok=False, tool="finish",
                                summary="Halted: repeated same action 10 times.")
            done, reason = True, "repeat_halt"
        else:
            result = self.toolbox.execute(action)

        self.last_result = result
        self.last_action = action
        if action.progress_analysis:
            self.transient_memory.append(action.progress_analysis)
        if result.ok and result.data.get("image_path"):
            self.current_obs["summary"] = result.summary
            if result.data.get("robot_pose"):
                self.current_obs["robot_pose"] = result.data["robot_pose"]

        self.step_idx += 1

        if not done:
            if action.tool == "finish":
                done, reason = True, "finish"
            elif result.data.get("task_done"):
                done, reason = True, "task_done"
            elif self.step_idx >= self.cfg.max_steps:
                done, reason = True, "max_steps"

        metrics: Dict[str, Any] = {}
        if done:
            terminal, metrics = self._terminal_reward()
            reward += terminal
            self.done = True
            self.busy = False  # free the server for the next rollout

        payload = {
            "reward": reward,
            "done": done,
            "reason": reason,
            "invalid_action": invalid,
            "tool": action.tool,
            "result_ok": bool(result.ok),
            "result_summary": result.summary,
            "metrics": {k: (float(v) if isinstance(v, (int, float)) else str(v))
                        for k, v in metrics.items()},
            "step_idx": self.step_idx,
        }
        payload["observation"] = None if done else self._observation()
        return payload


# ── HTTP app ──────────────────────────────────────────────────────────────────

from pydantic import BaseModel


# Request models must live at module scope: `from __future__ import annotations`
# stringifies the handler signatures, and FastAPI resolves those strings against
# module globals. Nested inside build_app they would be invisible, and FastAPI
# would silently downgrade the body param to a query param (HTTP 422).
class ResetReq(BaseModel):
    episode_id: int


class StepReq(BaseModel):
    session_id: str
    action: Dict[str, Any]


class ReleaseReq(BaseModel):
    session_id: str


def build_app(cfg):
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="habitat-env-server")
    # habitat-sim binds its GL/EGL context to the thread that created it, so every
    # simulator call (reset AND step) must run on ONE consistent worker thread — not
    # whichever thread uvicorn's shared pool hands out. A dedicated single-thread
    # executor guarantees that while still freeing the event loop for /health.
    from concurrent.futures import ThreadPoolExecutor
    state = SimpleNamespace(session=None, sim_pool=ThreadPoolExecutor(max_workers=1))

    def _ensure_session():
        if state.session is None:
            print("[env_server] importing Habitat …", flush=True)
            get_config, hab_env_cls = base._import_habitat()
            _ovmm._load_taskmap(cfg.split)
            state.session = _EpisodeSession(cfg, get_config, hab_env_cls)
        return state.session

    @app.get("/health")
    async def health():
        s = state.session
        return {"ok": True, "gpu_id": cfg.gpu_id, "split": cfg.split,
                "episode_id": s.ep_id if s else None,
                "busy": bool(s and s.busy),
                "idle_s": round(time.time() - s.last_activity, 1) if s else None}

    @app.post("/reset")
    async def reset(req: ResetReq):
        s = _ensure_session()
        # Reclaim a lease abandoned by a dead client before testing `busy`.
        if s.busy and (time.time() - s.last_activity) > cfg.session_lease_s:
            print(f"[env_server] reclaiming stale session {s.session_id} "
                  f"(idle {time.time() - s.last_activity:.0f}s)", flush=True)
            s.busy = False
        # Claim before the first await: handlers run on the single event-loop
        # thread, so test-and-set here is atomic and two concurrent /reset calls
        # cannot both win. A busy server must reject *fast*, before the (minutes
        # long) scene load.
        if s.busy:
            raise HTTPException(status_code=409, detail="busy")
        s.busy = True
        s.last_activity = time.time()
        # s.reset() drives habitat-sim (scene load + scan): seconds to minutes of
        # BLOCKING work. Run it in a worker thread so the event loop stays free to
        # answer /health and hand out fast 409s. `busy` already excludes a second
        # rollout, so the simulator is only ever touched by one thread at a time.
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(state.sim_pool, s.reset, req.episode_id)
        except Exception as exc:
            s.busy = False
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"reset failed: {exc}")

    @app.post("/step")
    async def step(req: StepReq):
        s = state.session
        if s is None or s.session_id != req.session_id:
            raise HTTPException(status_code=409, detail="stale or unknown session_id")
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(state.sim_pool, s.step, req.action)
        except Exception as exc:
            s.busy = False
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"step failed: {exc}")

    @app.post("/release")
    async def release(req: ReleaseReq):
        """Free a server whose rollout ended early (truncation, client error)."""
        s = state.session
        if s is not None and s.session_id == req.session_id:
            s.busy = False
        return {"ok": True}

    return app


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--gpu_id", type=int, default=0,
                   help="Physical GPU for rendering. On zp-nc12 only GPU 0 has a "
                        "usable EGL device, so anything else fails; see main().")
    p.add_argument("--split", default="minival", choices=["minival", "train", "val"])
    p.add_argument("--work_root", default=str(_ROOT / "runs" / "rl_env"))
    p.add_argument("--max_steps", type=int, default=16,
                   help="Episode step cap. Each step adds an image to the policy "
                        "context permanently, so keep this well below the "
                        "eval-time 40 (see README context budget).")
    p.add_argument("--scan_points", type=int, default=8)
    p.add_argument("--retrieval_model", default="siglip_base")
    p.add_argument("--obs_camera", default="head", choices=["head", "arm_workspace"])
    p.add_argument("--no_gdino", action="store_true")
    p.add_argument("--image_max_side", type=int, default=512,
                   help="Downscale observation images to this longest side.")
    p.add_argument("--max_images_per_turn", type=int, default=3)
    p.add_argument("--session_lease_s", type=float, default=1800.0,
                   help="Reclaim a claimed session after this many seconds without "
                        "a /step. Guards against clients that die mid-rollout and "
                        "would otherwise wedge the server forever.")
    # Reward shaping
    p.add_argument("--success_reward", type=float, default=1.0)
    p.add_argument("--stage_reward", type=float, default=0.25,
                   help="Partial credit for pddl stage-1 when the episode fails.")
    p.add_argument("--step_penalty", type=float, default=0.0)
    p.add_argument("--invalid_penalty", type=float, default=0.05)
    # Tool-internal VLM (inspect/detect/rerank) — NOT the policy under training.
    p.add_argument("--tool_vlm_service", default=os.environ.get("TOOL_VLM_SERVICE", "gemini"),
                   choices=["gemini", "openai"])
    p.add_argument("--tool_vlm_base_url", default=os.environ.get("TOOL_VLM_BASE_URL",
                                                                 "http://localhost:23333/v1"))
    p.add_argument("--tool_vlm_api_key", default=os.environ.get("TOOL_VLM_API_KEY"))
    p.add_argument("--tool_vlm_model", default=os.environ.get("TOOL_VLM_MODEL",
                                                              "models/gemini-3.5-flash"))
    return p.parse_args()


def main():
    cfg = _parse_args()
    # habitat-sim resolves its EGL device by matching the physical CUDA ordinal
    # against the EGL device list, so it can only render on a GPU the CUDA runtime
    # can actually open. On zp-nc12 that is GPU 0 alone — the other seven are in a
    # "GPU requires reset" state and cudaGetDeviceCount() returns 1 (see
    # train/README.md). Rendering elsewhere fails with
    #   "unable to find CUDA device N among 1 EGL devices in total".
    # Pin CUDA_VISIBLE_DEVICES to the chosen GPU (which also confines torch) and
    # hand habitat the remapped local index 0. Overwrite rather than setdefault: an
    # inherited CUDA_VISIBLE_DEVICES would otherwise silently break EGL.
    if cfg.gpu_id != 0:
        print(f"[env_server] WARNING: --gpu_id {cfg.gpu_id} — only GPU 0 is usable "
              f"on this host until the GPUs are reset; rendering will likely fail.",
              flush=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.gpu_id)
    cfg.gpu_id = 0
    # Servers run concurrently against the same episodes. Give each its own work
    # tree: capture frames, the FAISS index and the post-scan snapshot are all
    # written per episode, and two servers sharing them would race on
    # rmtree/copytree and interleave capture writes.
    cfg.work_root = os.path.join(cfg.work_root, f"srv{cfg.port}")
    import uvicorn
    print(f"[env_server] listening on {cfg.host}:{cfg.port} split={cfg.split}", flush=True)
    uvicorn.run(build_app(cfg), host=cfg.host, port=cfg.port, log_level="warning")


if __name__ == "__main__":
    main()
