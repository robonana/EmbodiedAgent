# verl RL training for the Habitat EmbodiedAgent

GRPO fine-tuning of **Qwen3.5-9B (full fine-tune, multimodal)** as the EmbodiedAgent policy,
with **Habitat OVMM episodes in the loop** and reward from Habitat's `pddl_success`.

## Why there are two processes

Habitat's conda env is **Python 3.9**; verl requires **Python ≥ 3.10**. They cannot
share an interpreter, so the simulator sits behind HTTP:

```
verl (py3.12, GPUs 1-6)                    habitat (py3.9, GPU 0)
┌───────────────────────────┐              ┌──────────────────────────────┐
│ HabitatAgentLoop          │  POST /reset │ habitat_env_server           │
│  - tokenise observation   │─────────────>│  - HabitatToolbox (9 tools)  │
│  - sample action (vLLM)   │  POST /step  │  - build_policy_prompt       │
│  - accumulate tokens/mask │<─────────────│  - pddl_success -> reward    │
└───────────────────────────┘   obs+reward └──────────────────────────────┘
```

The env server re-uses `agent/prompts.py::build_policy_prompt` and
`sim/habitat_toolbox.py`, so the prompt the policy trains on is byte-for-byte the
prompt it sees at eval time under `run_ovmm_embodied.py`.

## Files

| file | env | purpose |
|---|---|---|
| `habitat_env_server.py` | habitat | HTTP env: `/reset`, `/step`, `/health` |
| `habitat_agent_loop.py` | verl | `AgentLoopBase` subclass, registered as `habitat` |
| `agent_loop_config.yaml` | verl | maps `habitat` → the class, lists env-server URLs |
| `prepare_habitat_dataset.py` | verl | OVMM episodes → verl parquet |
| `launch_env_servers.sh` | habitat | starts N env servers in tmux `habenv` |
| `run_grpo_qwen9b.sh` | verl | the GRPO launcher |

## Environments

| | path | python | key packages |
|---|---|---|---|
| trainer | `/data1/chen/conda/envs/verl` | 3.12 | torch 2.11+cu130, vllm 0.23.0, verl 0.9.0.dev (source: `/data1/chen/verl`), peft, ninja |
| simulator | `/data1/chen/conda/envs/habitat` | 3.9 | habitat-sim 0.3.1, habitat-lab, fastapi, uvicorn |

## Quickstart

```bash
# 1. env servers (GPU 0). Rollout concurrency == number of servers.
GEMINI_API_KEY=... bash train/launch_env_servers.sh 4

# 2. dataset (10 minival episodes)
/data1/chen/conda/envs/verl/bin/python train/prepare_habitat_dataset.py --split minival

# 3. train  (single-GPU until the box's GPUs are reset — see below)
CUDA_VISIBLE_DEVICES=0 N_GPUS=1 ROLLOUT_TP=1 \
  HABITAT_ENV_SERVERS=http://127.0.0.1:8100 \
  bash train/run_grpo_qwen9b.sh
```

`HABITAT_ENV_SERVERS` overrides the URL list in `agent_loop_config.yaml`, so whatever
launched the servers stays the single source of truth for how many exist.

## ⚠ Host state (zp-nc12): 7 of 8 GPUs are wedged

**As of 2026-07-09 only physical GPU 0 is usable.** `nvidia-smi` lists 8 H20-3e and
`torch.cuda.device_count()` returns 8 (both read NVML), but the CUDA runtime
disagrees:

```
$ cudaGetDeviceCount()                      -> 1
$ nvidia-smi --query-remapped-rows=...      -> "[GPU requires reset]" on 7 of 8
$ dmesg | grep NVRM                         -> NV_ERR_GPU_IN_FULLCHIP_RESET
$ CUDA_VISIBLE_DEVICES=1 cudaGetDeviceCount -> 0
```

There are no remap *failures*, so this is a recoverable driver state, not dead
silicon. Clearing it needs `sudo nvidia-smi -r` with no processes attached, or a
reboot. Until then:

* multi-GPU training cannot run — use `N_GPUS=1 ROLLOUT_TP=1 CUDA_VISIBLE_DEVICES=0`,
* habitat renders on GPU 0 because it is the only CUDA/EGL device that exists.

This single fault produces several misleading symptoms. Do not "fix" them in code:

| symptom | real cause |
|---|---|
| `unable to find CUDA device N among 1 EGL devices in total` | only one CUDA device exists |
| verl worker `set_device(6)` → `device >= 0 && device < num_gpus` | Ray reports 8 GPUs from NVML, CUDA has 1 |
| `ProcessGroupNCCL ... no GPUs found` under `CUDA_VISIBLE_DEVICES=1,...,6` | none of those GPUs are usable |

Once the GPUs are reset, re-check whether habitat can render off GPU 0; if it can,
give the env servers their own GPU and hand the trainer the rest.

Separately, `flash-attn` is **not** installed (no system `nvcc`, and no prebuilt wheel
for torch 2.11/cu130). The launcher therefore defaults to `ATTN_IMPL=sdpa` and
`REMOVE_PADDING=False` — verl's padding-free path calls `flash_attn_varlen_func`.
Install flash-attn to get the faster path back.

### ⚠ LoRA is broken with Qwen3.5-9B on this verl/vLLM build

Any `lora_rank > 0` makes verl's LoRA→vLLM adapter sync emit **NaN logits**. Every
generation comes back as `max_tokens` × `"!"` (token id 0), `rollout_probs_diff_max`
is `nan`, and rewards / `pg_loss` / `grad_norm` are all `0.0`. **The training steps
still complete and look green.** That is the dangerous part: a NaN policy produces a
perfectly plausible-looking run that optimises nothing.

Established by bisection — each row an actual run, same GPU, same toolchain:

| configuration | result |
|---|---|
| base model, plain vLLM, 5 → 5466 prompt tokens | SANE |
| `enable_lora=True`, no adapter loaded | SANE |
| 1–3 real capture images through the processor | SANE |
| `enforce_eager=True` (no CUDA graphs) | still NaN |
| `layered_summon=False`, `load_format=auto` | still NaN |
| `lora_rank=64`, `target_modules=all-linear` | NaN |
| `lora_rank=64`, `target_modules=[q,k,v,o,gate,up,down]` | NaN |
| **`lora_rank=0`** (full fine-tune) | **SANE — valid tool call on turn 1** |

So it is LoRA itself, not the target-module set, and not the gated-delta-net layers.
The launcher therefore defaults to `LORA_RANK=0`.

**Before trusting any metric, look at a generation.** Set
`HABITAT_AL_DEBUG=/tmp/gen.txt` and confirm the first turn is real JSON, not `!!!!`.

### FlashInfer JIT — four traps, all fixed, all fatal to the vLLM EngineCore

Qwen3.5's gated-delta-net attention kernel is **JIT-compiled by FlashInfer on the first
rollout**, shelling out to `ninja` and nvcc. Every failure below surfaces the same
unhelpful way — a `RuntimeError` logged as a *warning* by `qwen_gdn_linear_attn.py`,
then `EngineDeadError` and a vLLM scheduler `KeyError` — so check the ninja output.

1. `ninja` was missing, and the env's `bin/` isn't on `PATH` because we call python by
   absolute path → `[Errno 2] No such file or directory: 'ninja'`.
   *Fixed:* `pip install ninja`; launcher prepends `$(dirname $VERL_PY)` to `PATH`.
2. The system `/usr/local/cuda` is **CUDA 11.8** → `nvcc fatal: Value 'c++20' is not
   defined for option 'std'`.
   *Fixed:* launcher sets `CUDA_HOME` to the CUDA 13 wheel in `site-packages/nvidia/cu13`
   (`nvidia` is a namespace package — use `__path__[0]`, `__file__` is `None`).
3. The CUDA 13 wheels were mutually inconsistent. `nvidia-cuda-nvcc` was 13.2.78 while
   `nvidia-cuda-runtime` headers were 13.0.96, and CCCL hard-errors on that:
   `"CUDA compiler and CUDA toolkit headers are incompatible"`. Note `cicc` comes from a
   *third* wheel, `nvidia-nvvm` — pinning only nvcc leaves it mismatched and it emits PTX
   the older `ptxas` rejects (`Unsupported .version 9.2; current version is '9.0'`).
   *Fixed:* pin all three to the version flashinfer 0.6.12 expects —
   `pip install "nvidia-cuda-nvcc==13.2.*" "nvidia-nvvm==13.2.*" "nvidia-cuda-runtime==13.2.*"`.
   (13.0 also self-consistently *compiles*, but flashinfer's sources then fail on
   `'__cudaLaunch' was not declared` — it wants the 13.2 runtime.)
4. All 67 CUDA compiles then succeed and the **link** fails:
   `/usr/bin/ld: cannot find -lcudart`. FlashInfer passes `-L$CUDA_HOME/lib64 -lcudart`,
   but the wheel lays out `lib/` (no `lib64`) and ships only `libcudart.so.13`, with no
   bare `libcudart.so` for `-lcudart` to resolve.
   *Fixed:* the launcher idempotently creates `lib64 -> lib` and `libcudart.so ->
   libcudart.so.13`, and exports `LIBRARY_PATH`. A pip reinstall wipes these; rerunning
   the launcher restores them.

After changing nvcc, delete the poisoned cache: `rm -rf ~/.cache/flashinfer`.

When the JIT fails, vLLM logs `GDN prefill kernel warmup ... failed ... Falling back to
torch implementation` and keeps going — but a later failure *during inference* kills the
EngineCore. So a green warmup is not proof: check `ninjafail` and the cached-ops dir.

## The context budget — the main thing that will bite you

Every turn appends a fresh observation image to the prompt, and **those tokens are
never reclaimed**: verl's incremental token building means turn *k*'s image is still
in context at turn *k+n*. Total response tokens grow roughly as

```
max_turns × (image_tokens × images_per_turn + text_tokens)
```

so the knobs that matter are:

| knob | where | default |
|---|---|---|
| `--max_steps` | env server | 16 (eval uses 40) |
| `--image_max_side` | env server | 512 |
| `--max_images_per_turn` | env server | 3 |
| `max_response_length` | GRPO script | 16384 |

`max_turns` in `agent_loop_config.yaml` must not exceed the server's `--max_steps`.
If a rollout hits `max_response_length` the agent loop stops early and marks
`extra_fields.truncated`; a high truncation rate means shrink images or steps, not
raise the length.

## Reward

Sparse, from Habitat's PDDL measure at episode end:

* `pddl_success` → `+success_reward` (default 1.0)
* failed but stage-1 satisfied → `+stage_reward` (default 0.25)
* unparseable / unknown-tool action → `-invalid_penalty` (default 0.05)
* optional `-step_penalty` per step (default 0.0)

The agent loop sums per-step rewards into `AgentLoopOutput.reward_score`, which verl
turns into the `rm_scores` tensor. No reward model, no `custom_reward_function`.

## Cost model

One rollout is one simulated episode. A GRPO step costs
`train_batch_size × rollout_n` episodes (default 8 × 8 = 64), run at a concurrency of
`len(env_server_urls)`. Two mitigations are built in:

* the simulator is **kept alive** across rollouts of the same episode,
* the scene scan runs **once per episode** and is snapshotted; later rollouts restore
  the post-scan episodic memory + FAISS index instead of rescanning.

The snapshot is only reused when its FAISS index is **valid and non-empty**
(`_snapshot_valid`): a run that crashes mid-scan (e.g. an OOM) used to leave an
empty `_post_scan/index`, and because the directory existed, `first_build` stayed
False forever — the scan never re-ran and every `retrieve_memory` returned ~1 stale
frame regardless of `top_k`. An invalid snapshot is now discarded and re-scanned.

A cold episode reset (scene load + scan) is minutes; a warm one is seconds. Ordering
the batch so rollouts of the same episode land on the same server is not implemented
— the pool hands out whichever server is free.

## Known gaps

* **The env's own VLM.** `HabitatToolbox` calls a VLM for `inspect`/`detect`/rerank.
  That client is part of the *environment*, not the policy: it defaults to Gemini
  (`--tool_vlm_service`). Point it at a frozen model. If you point it at the policy's
  own rollout server the environment will drift as the policy trains.
* **Cold-start exploration.** GRPO from a cold policy on a 9-tool JSON action space
  with a sparse terminal reward may produce all-zero-advantage groups (every rollout
  in the group fails). Watch the fraction of groups with non-zero reward variance; if
  it stays ~0, an SFT/rejection-sampling bootstrap is the fix.
* **Server affinity.** No episode→server pinning, so the scan snapshot is only reused
  when the same server happens to draw the same episode again.
