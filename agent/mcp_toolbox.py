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

Note this does NOT subclass BaseToolbox. BaseToolbox exists to share *tool logic* across
backends that give us raw primitives (step, pose, grasp). Here the remote MCP server
already implements the tools themselves, so there is no logic left to share — this class
is purely an adapter: async→sync, and base64→file paths.
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
        self._counter = 0   # names saved images; monotonic across the episode

        # Part of the ToolboxProtocol contract — the agent loop reads this directly.
        self._last_image_path: Optional[str] = None

        # Background event loop holding the persistent MCP session.
        #
        # The session must live on one loop for its whole lifetime (the streamable-HTTP
        # transport keeps an open connection), but the agent is synchronous top to bottom.
        # So we park a loop on a daemon thread and marshal every call onto it. The
        # alternative — asyncio.run() per call — would tear down and re-establish the MCP
        # session on every single tool call.
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._session: Optional[ClientSession] = None
        # The two async context managers are entered here and exited in close(); we hold
        # them as attributes because their __aenter__/__aexit__ straddle several calls
        # and cannot be expressed as a `with` block.
        self._http_cm = None
        self._sess_cm = None
        self._run(self._connect())
        print(f"[MCPToolbox] connected to {server_url}")

    # ---------------------------------------------------------------- async glue
    def _run(self, coro):
        """Run a coroutine on the background loop and block until it returns.

        This is the sync↔async boundary. The timeout matters: a robot skill can legitimately
        take minutes, but a hung server must not wedge the episode forever.
        """
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(self.call_timeout)

    async def _connect(self):
        """Open the HTTP transport, wrap it in an MCP session, and handshake."""
        self._http_cm = streamablehttp_client(self.server_url)
        read, write, _ = await self._http_cm.__aenter__()
        self._sess_cm = ClientSession(read, write)
        self._session = await self._sess_cm.__aenter__()
        await self._session.initialize()   # MCP capability handshake

    async def _acall(self, name: str, args: dict) -> dict:
        """Invoke one remote MCP tool and normalise its reply down to a plain dict.

        MCP returns results in several shapes and we have to handle all of them:
          * `structuredContent` — the modern typed payload; use it when present.
          * a text content block — older/simpler servers JSON-encode the payload as text;
            if it isn't valid JSON, treat the raw text as a summary rather than losing it.
        """
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
        # Unwrap that envelope, but only when it is *exactly* the envelope — a genuine
        # payload that happens to contain a "result" key alongside others is left alone.
        if set(payload.keys()) == {"result"} and isinstance(payload["result"], dict):
            payload = payload["result"]
        return payload

    def close(self):
        """Tear down the session, the transport, and the background loop, in that order.

        Exceptions during shutdown are swallowed (the connection may already be dead), but
        the `finally` guarantees the loop is stopped regardless — otherwise the daemon
        thread would keep the process alive.
        """
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
        """Write base64 images to disk; return local paths.

        The core of the adapter: the rest of the agent (prompt building, logging, the
        viewer) is written against file paths, so every image the server sends has to land
        on disk before it can be used. Filenames are counter_tag[_j] — the counter keeps
        them ordered and unique, the suffix disambiguates several images from one call.
        """
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
                # A lost image degrades the step; it should not kill the episode.
                print(f"[MCPToolbox] image save failed: {e}")
        return paths

    def _to_result(self, tool: str, payload: dict, set_last_image: bool) -> ToolResult:
        """Convert an MCP payload into the ToolResult the agent loop expects."""
        images = payload.get("images") or []
        paths = self._save_images(images, tool)
        data = dict(payload.get("data") or {})
        if paths and set_last_image:
            # By convention the first image is the head camera view.
            self._last_image_path = paths[0]
            data.setdefault("image_path", paths[0])

        # retrieve_memory: each candidate ships its frame as rgb_b64 — write it to
        # disk, expose it as rgb_path (the loop maps rgb_path -> memory_id), and
        # drop the base64 so it doesn't bloat the next prompt.
        #
        # That last point is load-bearing: `data` is serialised straight into the next
        # prompt, so leaving a dozen base64-encoded JPEGs in it would add megabytes of
        # unreadable text to the context. `pop` removes them as it goes.
        cands = data.get("candidates")
        if isinstance(cands, list):
            for c in cands:
                b64 = c.pop("rgb_b64", None)
                if b64:
                    saved = self._save_images([b64], "mem")
                    if saved:
                        c["rgb_path"] = saved[0]
                        paths.append(saved[0])   # also attach it as an image to the VLM
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
        """Forward the action to the remote server by tool name.

        Validation lives on the server side here (unlike BaseToolbox, which validates
        locally), so this is a thin pass-through. A transport failure becomes an ok=False
        ToolResult rather than an exception, keeping the agent loop's contract intact.
        """
        tool = action.tool
        args = dict(action.arguments or {})
        try:
            payload = self._run(self._acall(tool, args))
        except Exception as e:
            return ToolResult(ok=False, tool=tool, summary=f"MCP call failed: {e}")
        # Tools that return a fresh post-action head image refresh the view too.
        # The perception tools (detect/inspect/retrieve_*) return crops and memory frames
        # instead, and must NOT overwrite _last_image_path — the agent would then reason
        # about a crop as though it were the robot's live view.
        refresh = tool in ("manipulate", "base_move", "navigate")
        return self._to_result(tool, payload, set_last_image=refresh)
