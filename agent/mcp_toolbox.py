"""agent/mcp_toolbox.py — a ToolboxProtocol implementation backed by an MCP
server (the Humanoid_Simulation h12_mcp_server), so PromptEmbodiedAgent can drive
the H1-2 ROS stack unchanged.

The agent loop is synchronous and expects local image FILE PATHS; the MCP server
returns images as base64. This class:
  * holds one persistent MCP streamable-HTTP session on a background asyncio loop,
  * exposes sync observe()/execute() that the agent calls,
  * writes returned base64 images to <log_dir>/images/ and sets _last_image_path,
    so the existing path-based contract (prompt_agent.py) just works.

Nothing in the agent framework changes — this is an additive backend, selected by
the run_h12_mcp.py runner.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import threading
from typing import Any, Optional

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from .schemas import ToolAction, ToolResult


class MCPToolbox:
    """Synchronous facade over an async MCP client session."""

    def __init__(self, server_url: str, log_dir: str, call_timeout: float = 180.0):
        self.server_url = server_url
        self.log_dir = log_dir
        self.call_timeout = call_timeout
        self._img_dir = os.path.join(log_dir, "images")
        os.makedirs(self._img_dir, exist_ok=True)
        self._counter = 0

        self._last_image_path: Optional[str] = None

        # Background event loop holding the persistent MCP session.
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._session: Optional[ClientSession] = None
        self._http_cm = None
        self._sess_cm = None
        self._run(self._connect())
        print(f"[MCPToolbox] connected to {server_url}")

    # ---------------------------------------------------------------- async glue
    def _run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(self.call_timeout)

    async def _connect(self):
        self._http_cm = streamablehttp_client(self.server_url)
        read, write, _ = await self._http_cm.__aenter__()
        self._sess_cm = ClientSession(read, write)
        self._session = await self._sess_cm.__aenter__()
        await self._session.initialize()

    async def _acall(self, name: str, args: dict) -> dict:
        result = await self._session.call_tool(name, args or {})
        payload = getattr(result, "structuredContent", None)
        if not payload and result.content:
            for c in result.content:
                if getattr(c, "type", None) == "text":
                    try:
                        payload = json.loads(c.text)
                    except Exception:
                        payload = {"summary": c.text}
                    break
        payload = payload or {}
        # FastMCP wraps non-dict returns as {"result": ...}; our tools return dicts.
        if set(payload.keys()) == {"result"} and isinstance(payload["result"], dict):
            payload = payload["result"]
        return payload

    def close(self):
        async def _shutdown():
            try:
                if self._sess_cm:
                    await self._sess_cm.__aexit__(None, None, None)
                if self._http_cm:
                    await self._http_cm.__aexit__(None, None, None)
            except Exception:
                pass
        try:
            self._run(_shutdown())
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)

    # ------------------------------------------------------------------ helpers
    def _save_images(self, images_b64: list, tag: str) -> list[str]:
        """Write base64 images to disk; return local paths."""
        paths = []
        for j, b64 in enumerate(images_b64 or []):
            if not b64:
                continue
            self._counter += 1
            p = os.path.join(self._img_dir, f"{self._counter:04d}_{tag}{('_%d' % j) if j else ''}.jpg")
            try:
                with open(p, "wb") as f:
                    f.write(base64.b64decode(b64))
                paths.append(p)
            except Exception as e:
                print(f"[MCPToolbox] image save failed: {e}")
        return paths

    def _to_result(self, tool: str, payload: dict, set_last_image: bool) -> ToolResult:
        images = payload.get("images") or []
        paths = self._save_images(images, tool)
        data = dict(payload.get("data") or {})
        if paths and set_last_image:
            self._last_image_path = paths[0]
            data.setdefault("image_path", paths[0])
        # retrieve_memory: each candidate ships its frame as rgb_b64 — write it to
        # disk, expose it as rgb_path (the loop maps rgb_path -> memory_id), and
        # drop the base64 so it doesn't bloat the next prompt.
        cands = data.get("candidates")
        if isinstance(cands, list):
            for c in cands:
                b64 = c.pop("rgb_b64", None)
                if b64:
                    saved = self._save_images([b64], "mem")
                    if saved:
                        c["rgb_path"] = saved[0]
                        paths.append(saved[0])
        return ToolResult(
            ok=bool(payload.get("ok", False)),
            tool=payload.get("tool", tool),
            summary=str(payload.get("summary", "")),
            data=data,
            image_paths=paths,
        )

    # ---------------------------------------------------- ToolboxProtocol surface
    def observe(self) -> ToolResult:
        payload = self._run(self._acall("observe", {}))
        return self._to_result("observe", payload, set_last_image=True)

    def execute(self, action: ToolAction) -> ToolResult:
        tool = action.tool
        args = dict(action.arguments or {})
        try:
            payload = self._run(self._acall(tool, args))
        except Exception as e:
            return ToolResult(ok=False, tool=tool, summary=f"MCP call failed: {e}")
        # Tools that return a fresh post-action head image refresh the view too.
        refresh = tool in ("manipulate", "base_move", "navigate")
        return self._to_result(tool, payload, set_last_image=refresh)
