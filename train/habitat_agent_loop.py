"""train/habitat_agent_loop.py — verl AgentLoop that drives the Habitat env server.

Runs inside the verl env (Python 3.12).  The simulator lives behind HTTP in
train/habitat_env_server.py (Python 3.9); this module owns only the policy:
tokenise the observation, sample an action from the rollout server, POST it to
the environment, repeat.

Token bookkeeping follows verl's ToolAgentLoop:
  * assistant tokens  -> appended to prompt_ids, response_mask 1
  * observation tokens-> appended to prompt_ids, response_mask 0
so the final response_ids are exactly prompt_ids[-len(response_mask):].

Wire it up with rollout.agent.agent_loop_config_path pointing at
train/agent_loop_config.yaml, which also supplies `env_server_urls`.

Why the two-process split: habitat-sim pins an old Python and its own CUDA/EGL stack, while
verl wants 3.12 and a completely different torch build. Rather than fight that, the two live
in separate interpreters and talk over HTTP. That boundary is also why observations arrive as
base64 JPEGs rather than tensors.

The rest of the file is essentially one long exercise in keeping verl's token bookkeeping
exactly consistent — the response_mask, the logprobs, and the image blocks all have to line
up to the token, or training dies deep inside the trainer with an opaque shape error. Most
of the comments below are about those invariants.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import random
import re
import uuid
from typing import Any, Optional

import aiohttp
from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    register,
)
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__file__)

# NOTE: do NOT arbitrate server access with a process-local pool. verl runs several
# AgentLoopWorker processes (Ray actors); each would get its own pool holding every
# URL, and they would drive the same single-episode server concurrently — the second
# claimant silently steals the session and the first gets
# "HTTP 409: stale or unknown session_id". The server itself is the arbiter: /reset
# claims it (409 if busy) and /release or episode-end frees it.


def parse_action(text: str) -> Optional[dict]:
    """Extract the policy's JSON tool call.

    Mirrors agent/gemini_client.py::_parse_json — strip a <think> block, a
    <tool_call> wrapper and markdown fences, then take the outermost JSON object.
    Returns None when nothing parses, which the env server scores as an invalid
    action rather than crashing the rollout.
    """
    cleaned = text.strip()

    # Strip the reasoning block. Two shapes: the model emitted both tags, or the chat
    # template already injected the opening <think> into the prompt so only the closer
    # comes back.
    if "<think>" in cleaned:
        end = cleaned.find("</think>")
        cleaned = (cleaned[end + len("</think>"):] if end != -1
                   else cleaned[cleaned.find("<think>") + len("<think>"):]).strip()
    elif "</think>" in cleaned:  # template injected the opening tag
        cleaned = cleaned.rsplit("</think>", 1)[1].strip()

    if "<tool_call>" in cleaned:
        start = cleaned.find("<tool_call>") + len("<tool_call>")
        end = cleaned.find("</tool_call>")
        # Unterminated tag (truncated generation): take everything to the end and hope the
        # JSON is complete.
        cleaned = cleaned[start:end if end != -1 else len(cleaned)].strip()

    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()

    # Outermost braces. Unlike the gemini_client version (which brace-matches), this takes
    # first '{' to last '}' — cruder, but the training policy's output is not worth a
    # careful parse, and a mis-parse is scored as an invalid action rather than crashing.
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _decode_images(obs: dict) -> list[Image.Image]:
    """base64 JPEG (over the wire) → PIL. A failed decode drops that one image."""
    out = []
    for item in obs.get("images", []):
        try:
            out.append(Image.open(io.BytesIO(base64.b64decode(item["b64"]))).convert("RGB"))
        except Exception as exc:
            # WARNING, not raise: losing an image de-tunes the turn but a dead rollout
            # poisons the whole training batch.
            logger.warning("failed to decode observation image: %s", exc)
    return out


def _user_message(obs: dict) -> dict:
    """Interleave each image with its label, then the prompt text.

    The image entries are bare {"type": "image"} placeholders; the PIL objects
    travel separately to the processor, positionally matched to these slots.
    """
    content: list[dict] = []
    for item in obs.get("images", []):
        content.append({"type": "text", "text": item["label"]})
        content.append({"type": "image"})
    content.append({"type": "text", "text": obs["prompt"]})
    return {"role": "user", "content": content}


@register("habitat")
class HabitatAgentLoop(AgentLoopBase):
    """Multi-turn, multimodal rollout against a Habitat OVMM episode."""

    def __init__(
        self,
        trainer_config,
        server_manager,
        tokenizer,
        processor,
        dataset_cls,
        data_config,
        env_server_urls: Any = None,
        max_turns: int = 16,
        max_turn_tokens: int = 2048,
        request_timeout: float = 1800.0,
        acquire_timeout: float = 3600.0,
        acquire_poll_s: float = 5.0,
        **kwargs,
    ):
        super().__init__(trainer_config, server_manager, tokenizer, processor,
                         dataset_cls, data_config, **kwargs)
        # HABITAT_ENV_SERVERS wins over the YAML so the launcher that starts the
        # servers is the single source of truth for how many exist.
        from_env = os.environ.get("HABITAT_ENV_SERVERS", "").strip()
        if from_env:
            env_server_urls = from_env.split(",")
        elif isinstance(env_server_urls, str):
            env_server_urls = env_server_urls.split(",")
        self.env_server_urls = [str(u).strip() for u in (env_server_urls or []) if str(u).strip()]
        self.max_turns = int(max_turns)
        # Without a per-turn cap one generation can swallow the whole response
        # budget: a thinking model that never closes </think> reasons to the limit,
        # the loop truncates before it ever steps the environment, and every rollout
        # scores 0 -> zero advantage -> zero gradient. Cap each turn instead.
        self.max_turn_tokens = int(max_turn_tokens)
        self.request_timeout = float(request_timeout)
        self.acquire_timeout = float(acquire_timeout)
        self.acquire_poll_s = float(acquire_poll_s)
        self.response_length = self.rollout_config.response_length

        # Each image is a token block: <|vision_start|> <|image_pad|>*k <|vision_end|>.
        # Capping response_ids at response_length can slice a block in half, leaving
        # more images than surviving placeholder runs. verl then builds a nested
        # multimodal tensor whose per-sample length exceeds its ragged dim and dies
        # in _compute_old_log_prob ("Expected cond ... _lengths[i] <= ragged_dim_size").
        # We reconcile the cap to a block boundary; these ids drive that (None -> skip).
        def _tid(tok: str) -> Optional[int]:
            i = self.tokenizer.convert_tokens_to_ids(tok)
            unk = getattr(self.tokenizer, "unk_token_id", None)
            return i if (isinstance(i, int) and i >= 0 and i != unk) else None
        self._vision_start_id = _tid("<|vision_start|>")
        self._vision_end_id = _tid("<|vision_end|>")

    # ── Multimodal cap reconciliation ─────────────────────────────────────────

    def _cap_at_image_boundary(self, prompt_ids, response_ids, response_mask,
                               response_logprobs, images):
        """Cap the response to response_length WITHOUT splitting an image block.

        Returns (response_ids, response_mask, response_logprobs, images) where the
        number of complete <|vision_start|>..<|vision_end|> blocks in
        prompt_ids+response_ids equals len(images), and no trailing partial block
        remains. The prompt is never sliced (apply_chat_template fails loudly rather
        than truncate a multimodal prompt), so its blocks are always whole and its
        images are the earliest entries in `images`.
        """
        resp = response_ids[: self.response_length]
        mask = response_mask[: self.response_length]
        lp = response_logprobs[: self.response_length] if response_logprobs else None

        if self._vision_start_id is None or self._vision_end_id is None or not images:
            return resp, mask, lp, images

        seq = prompt_ids + resp
        n_start = seq.count(self._vision_start_id)
        n_end = seq.count(self._vision_end_id)
        if n_start > n_end:
            # The cap landed inside the last block: drop it (start + partial pads).
            for i in range(len(resp) - 1, -1, -1):
                if resp[i] == self._vision_start_id:
                    resp, mask = resp[:i], mask[:i]
                    if lp is not None:
                        lp = lp[:i]
                    break
            n_complete = n_end
        else:
            n_complete = n_start

        # Capping can only drop trailing blocks, so n_complete <= len(images).
        if n_complete < len(images):
            images = images[:n_complete]
        elif n_complete > len(images):  # invariant broken upstream — never expected
            logger.error("placeholder/image mismatch: %d blocks > %d images",
                         n_complete, len(images))
        return resp, mask, lp, images

    # ── HTTP ─────────────────────────────────────────────────────────────────

    async def _post(self, session: aiohttp.ClientSession, url: str, payload: dict) -> dict:
        """POST and unwrap, turning any non-200 into an exception with the server's body.

        The body matters: the env server puts the actual Habitat error in it, and without it
        a failed step is just an opaque 500.
        """
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                raise RuntimeError(f"{url} -> HTTP {resp.status}: {await resp.text()}")
            return await resp.json()

    async def _acquire(self, session: aiohttp.ClientSession,
                       episode_id: int) -> tuple[str, dict]:
        """Claim a free env server by resetting it. A busy server 409s immediately.

        Returns (url, reset_payload). Raises if none frees up within the timeout.
        """
        urls = list(self.env_server_urls)
        if not urls:
            raise RuntimeError("No Habitat env servers configured. Set "
                               "HABITAT_ENV_SERVERS or env_server_urls in "
                               "train/agent_loop_config.yaml.")
        random.shuffle(urls)  # spread load; avoids every worker stampeding url[0]
        deadline = asyncio.get_event_loop().time() + self.acquire_timeout
        while asyncio.get_event_loop().time() < deadline:
            for url in urls:
                async with session.post(f"{url}/reset",
                                        json={"episode_id": episode_id}) as resp:
                    if resp.status == 409:      # busy, try the next one
                        continue
                    if resp.status != 200:
                        raise RuntimeError(
                            f"{url}/reset -> HTTP {resp.status}: {await resp.text()}")
                    return url, await resp.json()
            await asyncio.sleep(self.acquire_poll_s)
        raise RuntimeError(
            f"no free Habitat env server after {self.acquire_timeout}s "
            f"({len(urls)} configured); rollout concurrency exceeds server count")

    async def _release(self, session: aiohttp.ClientSession, url: str,
                       session_id: str) -> None:
        try:
            async with session.post(f"{url}/release", json={"session_id": session_id}):
                pass
        except Exception as exc:  # never let cleanup mask the real error
            logger.warning("release %s failed: %s", url, exc)

    # ── Rollout ──────────────────────────────────────────────────────────────

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra = kwargs.get("extra_info") or {}
        if "episode_id" not in extra:
            raise KeyError("dataset row is missing extra_info.episode_id; "
                           "regenerate the parquet with train/prepare_habitat_dataset.py")
        episode_id = int(extra["episode_id"])
        return await self._rollout(episode_id, sampling_params)

    async def _rollout(self, episode_id: int,
                       sampling_params: dict[str, Any]) -> AgentLoopOutput:
        request_id = uuid.uuid4().hex
        metrics: dict[str, float] = {}
        timeout = aiohttp.ClientTimeout(total=self.request_timeout)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            url, reset = await self._acquire(session, episode_id)
            session_id = reset["session_id"]
            obs = reset["observation"]
            released = False

            images = _decode_images(obs)
            messages = [{"role": "system", "content": reset["system_prompt"]},
                        _user_message(obs)]
            prompt_ids: list[int] = await self.apply_chat_template(
                messages, images=images or None)
            prompt_len = len(prompt_ids)

            response_mask: list[int] = []
            # rollout.calculate_log_probs defaults to True, and the trainer then
            # expects a rollout_log_probs tensor. Mirror ToolAgentLoop: real
            # logprobs for LLM tokens, 0.0 padding for observation tokens.
            response_logprobs: list[float] = []
            # The rollout server tags each generate() with bookkeeping the trainer
            # later reads off batch.tags (min/max_global_steps, spec-decode counts).
            # Drop them and _compute_metrics dies on int(None) — so carry them
            # through, refreshing max_global_steps on every turn as ToolAgentLoop does.
            extra_fields: dict[str, Any] = {}
            total_reward = 0.0
            assistant_turns = 0
            episode_metrics: dict[str, Any] = {}
            truncated = False

            try:
                for _ in range(self.max_turns):
                    # ── 1. Generate one assistant turn ────────────────────────
                    remaining = self.response_length - len(response_mask)
                    if remaining <= 0:
                        truncated = True
                        break
                    # Two caps at once: never exceed the total response budget, and never
                    # let a single turn run away (see max_turn_tokens in __init__).
                    turn_params = {**sampling_params,
                                   "max_tokens": min(remaining, self.max_turn_tokens)}
                    output = await self.server_manager.generate(
                        request_id=request_id,
                        prompt_ids=prompt_ids,
                        sampling_params=turn_params,
                        image_data=images or None,
                    )
                    # prompt_ids is the running FULL sequence; response_ids is recovered at
                    # the end by slicing off the original prompt. mask=1 ⇒ these are the
                    # policy's own tokens, i.e. the ones that get gradient.
                    prompt_ids += output.token_ids
                    response_mask += [1] * len(output.token_ids)
                    if output.log_probs:
                        response_logprobs += output.log_probs
                    if not extra_fields:
                        extra_fields.update(output.extra_fields or {})
                    elif (output.extra_fields or {}).get("max_global_steps"):
                        extra_fields["max_global_steps"] = output.extra_fields["max_global_steps"]
                    assistant_turns += 1

                    # Budget exhausted by the generation itself — stop before stepping the
                    # env, since we could not represent the resulting observation anyway.
                    if len(response_mask) >= self.response_length:
                        truncated = True
                        break

                    # ── 2. Parse the action out of the generated text ─────────
                    text = self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
                    action = parse_action(text)

                    if os.environ.get("HABITAT_AL_DEBUG"):
                        raw = self.tokenizer.decode(output.token_ids, skip_special_tokens=False)
                        with open(os.environ["HABITAT_AL_DEBUG"], "a") as fh:
                            fh.write(
                                f"\n=== ep{episode_id} turn{assistant_turns} "
                                f"ntok={len(output.token_ids)} cap={turn_params['max_tokens']} "
                                f"parsed={action is not None}\n"
                                f"--- head ---\n{raw[:600]}\n--- tail ---\n{raw[-400:]}\n")

                    # ── 3. Step the environment ───────────────────────────────
                    # `action or {}` — an unparseable generation still steps the env, which
                    # scores it as an invalid action. That is deliberate: the policy must
                    # feel a reward penalty for emitting garbage, and skipping the step would
                    # make malformed output free.
                    step = await self._post(session, f"{url}/step",
                                            {"session_id": session_id, "action": action or {}})
                    total_reward += float(step["reward"])

                    if step["done"]:
                        episode_metrics = step.get("metrics", {})
                        released = True   # the server frees itself on episode end
                        break

                    # ── 4. Append the observation as *non-gradient* tokens ────
                    next_obs = step["observation"]
                    new_images = _decode_images(next_obs)
                    env_ids = await self.apply_chat_template(
                        [_user_message(next_obs)],
                        images=new_images or None,
                        remove_system_prompt=True,   # the system turn is already in prompt_ids
                    )
                    env_ids = self.turn_separator + env_ids

                    # Check BEFORE appending. Appending an observation that overflows would
                    # get sliced by the cap, potentially through the middle of an image
                    # block — the exact corruption _cap_at_image_boundary exists to undo.
                    if len(response_mask) + len(env_ids) >= self.response_length:
                        truncated = True
                        break

                    prompt_ids += env_ids
                    # mask=0 ⇒ environment tokens. They condition the next generation but
                    # must NOT receive gradient — the policy didn't write them.
                    response_mask += [0] * len(env_ids)
                    # Padding logprobs keeps this array the same length as the mask, which
                    # the trainer requires.
                    if response_logprobs:
                        response_logprobs += [0.0] * len(env_ids)
                    images.extend(new_images)
            finally:
                # Truncated, max_turns exhausted, or an exception: the episode never
                # reached `done`, so the server is still claimed. Free it or the pool
                # bleeds a server per aborted rollout and later rollouts time out.
                if not released:
                    await self._release(session, url, session_id)

        # Recover the response by slicing the original prompt off the running sequence. The
        # assert pins the loop's core invariant: every token appended to prompt_ids after
        # prompt_len got exactly one mask entry (1 for assistant, 0 for observation).
        response_ids = prompt_ids[prompt_len:]
        assert len(response_ids) == len(response_mask), (
            f"token/mask mismatch: {len(response_ids)} vs {len(response_mask)}")

        capped_prompt = prompt_ids[:prompt_len]
        response_ids, response_mask, response_logprobs, images = self._cap_at_image_boundary(
            capped_prompt, response_ids, response_mask, response_logprobs, images)

        return AgentLoopOutput(
            prompt_ids=capped_prompt,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs if response_logprobs else None,
            multi_modal_data={"images": images} if images else None,
            reward_score=total_reward,
            # verl counts every message: 1 system + (assistant + observation) per turn.
            num_turns=assistant_turns * 2 + 1,
            metrics=AgentLoopMetrics(**metrics),
            extra_fields={
                **extra_fields,
                "episode_id": episode_id,
                "pddl_success": float(episode_metrics.get("pddl_success", 0.0) or 0.0),
                # Watch these two: a high truncation rate means the image budget is
                # too big for max_response_length (see README), not that the length
                # should be raised.
                "truncated": truncated,
                "num_images": len(images),
            },
        )
