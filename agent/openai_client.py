"""
agent/openai_client.py — OpenAI-compatible chat-completions client.

Drop-in replacement for GeminiClient that talks to any server exposing the
OpenAI `/v1/chat/completions` API (e.g. a locally-served vLLM / SGLang model
such as Qwen3.5-9B at http://localhost:23333/v1).

It subclasses GeminiClient so all the high-level prompt builders
(call_policy / generate_json / inspect_image / rerank_memory_candidates) and
the JSON-parsing / image-loading / logging helpers are reused verbatim. Only
the transport (`__init__` and `_call_with_retry`) is overridden: instead of
google.generativeai it converts the (str | PIL.Image) `parts` list into an
OpenAI messages array (text + base64 data-URL image blocks) and POSTs it.
"""

from __future__ import annotations

import base64
import concurrent.futures
import io
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from .gemini_client import GeminiClient


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


class OpenAIClient(GeminiClient):
    """OpenAI chat-completions transport with the GeminiClient interface."""

    def __init__(
        self,
        base_url: str = "http://localhost:23333/v1",
        model_name: str = "Qwen3.5-9B",
        api_key: Optional[str] = None,
        log_dir: Optional[str] = None,
        max_retries: int = 5,
        event_pump=None,
        timeout: Optional[float] = None,
        max_tokens: Optional[int] = None,
        enable_thinking: Optional[bool] = None,
    ):
        # Deliberately do NOT call GeminiClient.__init__ (it configures genai).
        self.base_url    = base_url.rstrip("/")
        self.model_name  = model_name
        self.api_key     = api_key or "EMPTY"
        # Retries. Connection errors (a flaky SSH tunnel dropping mid-request)
        # get extra patience: wait VLM_RECONNECT_WAIT seconds so an autossh /
        # reconnect-loop tunnel can come back before we give up and fall to wait.
        self.max_retries = int(os.environ.get("VLM_MAX_RETRIES", max_retries))
        self.reconnect_wait = float(os.environ.get("VLM_RECONNECT_WAIT", 5.0))
        self.log_dir     = Path(log_dir) if log_dir else None
        self.event_pump  = event_pump
        # Tunables (env-overridable). Defaults assume Qwen served by vLLM with
        # `--reasoning-parser qwen3` (Option B): thinking is ON so the model
        # reasons, and the server routes the <think>…</think> out of `content`
        # (into reasoning_content), so the agent sees clean JSON — mirroring
        # Gemini's hidden-thinking behaviour. timeout is generous because
        # thinking + multi-image prefill is slow on the first call of each shape.
        self.timeout     = float(timeout if timeout is not None
                                 else os.environ.get("VLM_TIMEOUT", 600.0))
        # enable_thinking: None → env VLM_ENABLE_THINKING (default True). Sent to
        # vLLM/Qwen via chat_template_kwargs; set VLM_THINKING_KWARG=0 to omit
        # that field for non-vLLM backends. If your server is NOT started with a
        # reasoning parser, set VLM_ENABLE_THINKING=0 so the model emits the JSON
        # directly (otherwise the <think> block lands inline in `content`; the
        # client's _strip_reasoning fallback handles that, but it's wasteful).
        self.enable_thinking = (enable_thinking if enable_thinking is not None
                                else _env_bool("VLM_ENABLE_THINKING", True))
        self.send_thinking_kwarg = _env_bool("VLM_THINKING_KWARG", True)
        # Sampling. Greedy (temp=0) is deterministic and ideal for short JSON,
        # BUT the small Qwen3.5-9B loops endlessly in its reasoning under greedy
        # when thinking is ON (generates to the 32k context, never emits the
        # answer). So default to the model's recommended sampling whenever
        # thinking is on, and stay greedy when it's off. top_p/top_k are only
        # sent when sampling (temperature>0). All env-overridable.
        _default_temp = 0.6 if self.enable_thinking else 0.0
        self.temperature = float(os.environ.get("VLM_TEMPERATURE", _default_temp))
        self.top_p       = float(os.environ.get("VLM_TOP_P", 0.95))
        self.top_k       = int(os.environ.get("VLM_TOP_K", 20))
        # max_tokens: safety bound so a reasoning loop can't run to the full
        # context. 0 means uncapped (vLLM uses remaining context, à la Gemini).
        # Default 16384 leaves room for thinking + even a 30-candidate rerank
        # (~5k-token answer) while halving the worst-case runaway.
        self.max_tokens  = int(max_tokens if max_tokens is not None
                               else os.environ.get("VLM_MAX_TOKENS", 16384))
        self._call_count = 0
        self._last_raw_response: Optional[str] = None

        if self.log_dir:
            # Reuse the same raw-response folder the agent already looks at.
            (self.log_dir / "raw_gemini").mkdir(parents=True, exist_ok=True)

        try:
            import requests  # noqa: F401
        except ImportError as e:
            raise ImportError("requests not installed. pip install requests") from e

        _cap = self.max_tokens if self.max_tokens and self.max_tokens > 0 else "uncapped"
        _samp = (f"temp={self.temperature} top_p={self.top_p} top_k={self.top_k}"
                 if self.temperature > 0 else "greedy")
        print(f"[OpenAIClient] ready — base_url={self.base_url} model={model_name} "
              f"think={self.enable_thinking} {_samp} max_tokens={_cap} "
              f"timeout={self.timeout:.0f}s")

    # ── Transport ──────────────────────────────────────────────────────────────

    @staticmethod
    def _img_to_data_url(pil) -> str:
        buf = io.BytesIO()
        pil.convert("RGB").save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def _parts_to_messages(self, parts: list) -> list[dict]:
        """Convert the GeminiClient (str | PIL.Image) parts list into messages.

        The first string part becomes the system message; everything else is
        packed into a single user message whose content is an array of text and
        image_url blocks (in original order so labels stay attached to images).
        """
        from PIL import Image as _PIL

        system_text: Optional[str] = None
        start = 0
        if parts and isinstance(parts[0], str):
            system_text = parts[0]
            start = 1

        content: list[dict] = []
        for part in parts[start:]:
            if isinstance(part, str):
                content.append({"type": "text", "text": part})
            elif isinstance(part, _PIL.Image):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": self._img_to_data_url(part)},
                })

        messages: list[dict] = []
        if system_text is not None:
            messages.append({"role": "system", "content": system_text})
        if content:
            messages.append({"role": "user", "content": content})
        elif system_text is not None:
            # Degenerate single-string prompt: send it as the user turn.
            messages = [{"role": "user", "content": system_text}]
        return messages

    def _post(self, messages: list[dict]) -> str:
        import requests
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False,
        }
        # Only send top_p/top_k when actually sampling; greedy ignores them.
        if self.temperature > 0:
            payload["top_p"] = self.top_p
            payload["top_k"] = self.top_k   # vLLM extension (accepted in body)
        # Omit max_tokens entirely when uncapped (== Gemini) so vLLM defaults to
        # remaining context and the model stops at EOS.
        if self.max_tokens and self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens
        # Toggle the model's reasoning block via the chat template (vLLM/Qwen).
        # Omit for backends that reject unknown fields (VLM_THINKING_KWARG=0).
        if self.send_thinking_kwarg:
            payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        try:
            um = data.get("usage", {})
            if um:
                print(f"[tokens] in={um.get('prompt_tokens')}  "
                      f"out={um.get('completion_tokens')}  "
                      f"total={um.get('total_tokens')}")
        except Exception:
            pass
        msg = data["choices"][0]["message"]
        return self._strip_reasoning(msg.get("content") or "")

    @staticmethod
    def _strip_reasoning(text: str) -> str:
        """Remove a Qwen/<think> reasoning block, returning only the final answer.

        Two shapes are handled so reasoning can be ON yet filtered out, à la
        Gemini (whose reasoning is hidden server-side):
          • inline block:  '<think> … </think>\\n\\n{json}'
          • lone closer:   '… reasoning … </think>\\n\\n{json}'  — the Qwen chat
            template injects the OPENING <think> into the prompt, so the returned
            content carries only the closing tag.
        If the server runs a --reasoning-parser, thinking is already in a separate
        field and `content` is clean, so this is a harmless no-op.
        Note: if thinking is truncated by max_tokens (no closing tag emitted),
        nothing is stripped and JSON parsing will fail upstream — give thinking a
        large enough VLM_MAX_TOKENS budget to finish.
        """
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[1]
        return text.strip()

    def _call_with_retry(
        self,
        parts: list,
        tag: str,
        max_retries: Optional[int] = None,
        skip_print_first: bool = False,
        print_parts: bool = True,
    ) -> dict[str, Any]:
        retries = max_retries if max_retries is not None else self.max_retries
        repair_parts = None

        from PIL import Image as _PIL
        print(f"\n{'='*60}")
        print(f"[OpenAI] {tag}  model={self.model_name}")
        if print_parts:
            for i, part in enumerate(parts):
                if skip_print_first and i == 0:
                    continue
                if isinstance(part, str):
                    print(part)
                elif isinstance(part, _PIL.Image):
                    print(f"<image {part.size}>")
        print(f"{'='*60}\n")

        for attempt in range(retries):
            try:
                active   = repair_parts if repair_parts else parts
                messages = self._parts_to_messages(active)
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
                    _fut = _pool.submit(self._post, messages)
                    while not _fut.done():
                        if self.event_pump:
                            try:
                                self.event_pump()
                            except Exception:
                                pass
                        time.sleep(0.033)
                    raw = _fut.result().strip()

                self._log_raw(raw, tag)
                parsed = self._parse_json(raw)
                if parsed is not None:
                    self._last_raw_response = raw
                    return parsed

                repair_parts = [
                    parts[0] if parts else "",
                    f"Your previous response was not valid JSON:\n{raw}\n\n"
                    f"Output ONLY the corrected JSON object. "
                    f"No markdown fences, no extra text.",
                ]
                print(f"[OpenAIClient] {tag}: parse failed, retry {attempt + 1}/{retries}")
                time.sleep(0.5)

            except Exception as e:
                import requests
                is_conn = isinstance(e, requests.exceptions.ConnectionError)
                kind = "connection error (tunnel down?)" if is_conn else "API error"
                print(f"[OpenAIClient] {tag}: {kind} attempt {attempt + 1}/{retries}: {e}")
                if attempt < retries - 1:
                    # Wait longer for connection errors so a reconnecting tunnel
                    # has time to come back; short backoff for other errors.
                    time.sleep(self.reconnect_wait if is_conn else 1.5)

        print(f"[OpenAIClient] {tag}: all {retries} attempts failed, returning {{}}")
        return {}
