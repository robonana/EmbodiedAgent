"""
agent/gemini_client.py — Gemini API wrapper for PromptEmbodiedAgent.

Defaults to gemini-2.5-pro.  Strips markdown fences, retries on parse failure,
logs every raw response to disk for debugging / future SFT data collection.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional


class GeminiClient:
    """
    Wrapper around google.generativeai for the agent pipeline.

    All policy calls use self.model_name (default: gemini-2.5-pro).
    Raw responses are written to log_dir/raw_gemini/ for debugging.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-pro",
        log_dir: Optional[str] = None,
        max_retries: int = 3,
        event_pump: Optional[Callable] = None,
    ):
        self.model_name  = model_name
        self.max_retries = max_retries
        self.log_dir     = Path(log_dir) if log_dir else None
        self.event_pump  = event_pump
        self._call_count = 0
        self._last_raw_response: Optional[str] = None

        if self.log_dir:
            (self.log_dir / "raw_gemini").mkdir(parents=True, exist_ok=True)

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._genai = genai
            self._model = genai.GenerativeModel(model_name)
            print(f"[GeminiClient] ready — model={model_name}")
        except ImportError as e:
            raise ImportError(
                "google-generativeai not installed. "
                "pip install google-generativeai"
            ) from e

    # ── Public API ────────────────────────────────────────────────────────────

    def call_policy(
        self,
        system_prompt: str,
        user_prompt: str,
        labeled_images: Optional[list[tuple[str, str]]] = None,
    ) -> dict[str, Any]:
        """
        Main policy call: system + user + optional labeled images → parsed JSON dict.

        labeled_images: list of (label, path) pairs.  Each pair becomes a text label
        followed immediately by the PIL image in the parts list so the model can
        correlate each image to its description.

        Returns {} on total failure (never raises).
        """
        self._call_count += 1
        tag = f"policy_{self._call_count:04d}"
        parts: list = [system_prompt, user_prompt]
        for label, path in (labeled_images or []):
            pil = self._load_images([path])
            if pil:
                parts.append(label)
                parts.extend(pil)

        return self._call_with_retry(parts, tag, skip_print_first=True)

    def generate_json(
        self,
        prompt: str,
        images: Optional[list] = None,
        system_prompt: Optional[str] = None,
        max_retries: Optional[int] = None,
    ) -> dict[str, Any]:
        """Generic JSON generation call (for inspect, rerank, etc.)."""
        self._call_count += 1
        tag   = f"gen_{self._call_count:04d}"
        parts: list = []
        if system_prompt:
            parts.append(system_prompt)
        parts.append(prompt)
        if images:
            parts.extend(self._coerce_images(images))

        return self._call_with_retry(parts, tag, max_retries=max_retries)

    def inspect_image(
        self,
        image_paths: list[str],
        question: str,
        labels: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Ask a visual question about one or more images/crops.  Returns structured JSON.

        labels: optional list of label strings, one per path.  Each label is inserted
        as a text part immediately before its image so the model knows what it's seeing.
        """
        from .prompts import build_visual_inspection_prompt
        prompt = build_visual_inspection_prompt(question)
        parts: list = [prompt]
        any_loaded = False
        for i, path in enumerate(image_paths):
            pil = self._load_images([path])
            if not pil:
                continue
            label = labels[i] if labels and i < len(labels) else f"Image {i + 1}:"
            parts.append(label)
            parts.extend(pil)
            any_loaded = True
        if not any_loaded:
            return {"answer": "image_load_error", "evidence": "",
                    "confidence": 0.0, "candidate_bboxes": []}
        return self._call_with_retry(parts, f"inspect_{self._call_count+1:04d}")

    def rerank_memory_candidates(
        self,
        query: str,
        candidates: list[dict],
        image_paths: Optional[list[str]] = None,
        query_image_path: Optional[str] = None,
    ) -> dict[str, Any]:
        """Ask Gemini to rerank all candidates by relevance.  Returns {ranked_ids, reason}.

        Each candidate image is passed as a separate part so Gemini sees full resolution.
        The prompt numbers each candidate; images follow in the same order.
        If query_image_path is given it is prepended so Gemini sees the query image first.
        """
        from .prompts import build_memory_rerank_prompt
        prompt = build_memory_rerank_prompt(
            query, candidates, is_image_query=bool(query_image_path))

        parts: list = [prompt]
        if query_image_path:
            q_imgs = self._load_images([query_image_path])
            if q_imgs:
                parts.append("\nQuery image:")
                parts.append(q_imgs[0])

        images = self._load_images(image_paths or [])
        for i, img in enumerate(images):
            parts.append(f"\nCandidate {i + 1} image:")
            parts.append(img)

        result = self._call_with_retry(parts, f"rerank_{self._call_count+1:04d}",
                                       print_parts=False)
        print(f"[rerank] ranked_ids={result.get('ranked_ids')}  "
              f"reason={result.get('reason', '')}")
        return result

    # ── Internals ─────────────────────────────────────────────────────────────

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

        # Print text parts of the prompt (skip PIL images and system prompt)
        from PIL import Image as _PIL
        print(f"\n{'='*60}")
        print(f"[Gemini] {tag}  model={self.model_name}")
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
                active = repair_parts if repair_parts else parts
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _pool:
                    _fut = _pool.submit(self._model.generate_content, active)
                    while not _fut.done():
                        if self.event_pump:
                            try:
                                self.event_pump()
                            except Exception:
                                pass
                        time.sleep(0.033)
                    response = _fut.result()
                raw = response.text.strip()
                self._log_raw(raw, tag)
                try:
                    um = response.usage_metadata
                    print(f"[tokens] in={um.prompt_token_count}  "
                          f"out={um.candidates_token_count}  "
                          f"total={um.total_token_count}")
                except Exception:
                    pass

                parsed = self._parse_json(raw)
                if parsed is not None:
                    self._last_raw_response = raw
                    return parsed

                # Build repair prompt for next attempt
                repair_parts = [
                    parts[0] if parts else "",
                    f"Your previous response was not valid JSON:\n{raw}\n\n"
                    f"Output ONLY the corrected JSON object. "
                    f"No markdown fences, no extra text.",
                ]
                print(f"[GeminiClient] {tag}: parse failed, retry {attempt + 1}/{retries}")
                time.sleep(0.5)

            except Exception as e:
                print(f"[GeminiClient] {tag}: API error attempt {attempt + 1}: {e}")
                if attempt < retries - 1:
                    time.sleep(1.5)

        print(f"[GeminiClient] {tag}: all {retries} attempts failed, returning {{}}")
        return {}

    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
        """Strip <think>, <tool_call> blocks and markdown fences, then parse JSON.  Returns None on failure."""
        cleaned = text.strip()

        # Strip <think>...</think> reasoning block
        if "<think>" in cleaned:
            end = cleaned.find("</think>")
            if end != -1:
                cleaned = cleaned[end + len("</think>"):].strip()
            else:
                cleaned = cleaned[cleaned.find("<think>") + len("<think>"):].strip()

        # Extract content from <tool_call>...</tool_call> if present
        if "<tool_call>" in cleaned:
            start = cleaned.find("<tool_call>") + len("<tool_call>")
            end   = cleaned.find("</tool_call>")
            if end != -1:
                cleaned = cleaned[start:end].strip()
            else:
                cleaned = cleaned[start:].strip()

        # Strip ``` fences
        if cleaned.startswith("```"):
            lines  = cleaned.split("\n")
            start  = 1
            if lines[0].strip("` ").startswith("json"):
                start = 1
            end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
            cleaned = "\n".join(lines[start:end]).strip()

        # Direct parse
        try:
            result = json.loads(cleaned)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

        # Extract first JSON object with brace matching
        depth = 0
        start_i = cleaned.find("{")
        if start_i >= 0:
            for i, ch in enumerate(cleaned[start_i:], start_i):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            result = json.loads(cleaned[start_i: i + 1])
                            if isinstance(result, dict):
                                return result
                        except json.JSONDecodeError:
                            pass
                        break

        return None

    def _load_images(self, paths: list[str]) -> list:
        """Load PIL images from file paths, skipping failures."""
        from PIL import Image as _PIL
        images = []
        for p in paths:
            if p and os.path.exists(p):
                try:
                    images.append(_PIL.open(p).convert("RGB"))
                except Exception as e:
                    print(f"[GeminiClient] image load error ({p}): {e}")
        return images

    def _coerce_images(self, items: list) -> list:
        """Accept PIL images or file paths."""
        from PIL import Image as _PIL
        out = []
        for item in items:
            if isinstance(item, str):
                out.extend(self._load_images([item]))
            elif isinstance(item, _PIL.Image):
                out.append(item)
        return out

    def _log_raw(self, raw: str, tag: str) -> Optional[str]:
        if not self.log_dir:
            return None
        try:
            h    = hashlib.md5(raw.encode()).hexdigest()[:8]
            ts   = time.strftime("%H%M%S")
            path = self.log_dir / "raw_gemini" / f"{ts}_{tag}_{h}.txt"
            path.write_text(raw, encoding="utf-8")
            return str(path)
        except Exception:
            return None
