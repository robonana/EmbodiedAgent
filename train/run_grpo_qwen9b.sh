#!/usr/bin/env bash
# train/run_grpo_qwen9b.sh — GRPO | Qwen3.5-9B | full fine-tune | multimodal | Habitat OVMM
#
# Prereqs, in order:
#   1) bash train/launch_env_servers.sh 4          # env servers on the render GPU
#   2) python train/prepare_habitat_dataset.py --split train     # -> data/rl/train.parquet
#      python train/prepare_habitat_dataset.py --split minival   # -> held-out val set
#   3) bash train/run_grpo_qwen9b.sh
# Trains on the TRAIN split, validates on minival. Override with TRAIN_FILES / VAL_FILES.
#
# The reward comes from the agent loop (AgentLoopOutput.reward_score, driven by
# Habitat's pddl_success), so no reward model and no custom reward function are
# configured: verl reads the rm_scores tensor the agent loop produces.
#
# GPUs (zp-nc35, 8x healthy H20-3e): trainer takes 0-5, env servers take 6-7.
# Keep them disjoint — habitat-sim and vLLM will OOM each other.
#
# CUDA_VISIBLE_DEVICES must stay **0-based and contiguous**. verl's worker calls
# set_device(int(ray_accelerator_id)) (single_controller/base/worker.py), and Ray
# reports accelerator ids as the *entries* of CUDA_VISIBLE_DEVICES, not indices
# into it. With CVD=1,2,3,4,5,6 the last worker does set_device(6) on a 6-device
# process and asserts, while the others silently bind the wrong physical GPU.
# So: give the trainer 0..N-1 and push the env servers to the high GPUs.
set -xeuo pipefail

REPO=${REPO:-/data1/chen/EmbodiedAgent}
VERL_PY=${VERL_PY:-/data1/chen/conda/envs/verl/bin/python}
MODEL_PATH=${MODEL_PATH:-/data1/chen/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}

export HF_HOME=${HF_HOME:-/data1/chen/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TMPDIR=${TMPDIR:-/data1/chen/tmp}

# Qwen3.5 uses a gated-delta-net attention whose kernel FlashInfer JIT-compiles on
# first use: it shells out to `ninja` and to nvcc. Two traps:
#   1. we invoke python by absolute path, so the env's bin/ (holding ninja) is not
#      on PATH -> "[Errno 2] No such file or directory: 'ninja'", which kills the
#      vLLM EngineCore and surfaces as EngineDeadError + a scheduler KeyError;
#   2. the system nvcc at /usr/local/cuda is CUDA 11.8 and rejects -std=c++20
#      ("nvcc fatal: Value 'c++20' is not defined"). The CUDA 13.2 nvcc that vLLM
#      pulls in as a wheel does support it, so point CUDA_HOME at that instead.
# `nvidia` is a namespace package (__file__ is None), hence __path__ below.
export PATH="$(dirname "$VERL_PY"):$PATH"
if [ -z "${CUDA_HOME:-}" ]; then
  _cu=$("$VERL_PY" -c 'import nvidia,pathlib;print(pathlib.Path(list(nvidia.__path__)[0])/"cu13")' 2>/dev/null || true)
  if [ -x "${_cu:-}/bin/nvcc" ]; then
    export CUDA_HOME="$_cu"
    export PATH="$CUDA_HOME/bin:$PATH"
  else
    echo "WARNING: CUDA 13 nvcc not found; FlashInfer JIT will fall back to $(command -v nvcc)" >&2
  fi
fi
# FlashInfer links its JIT kernels with `-L$CUDA_HOME/lib64 -lcudart`, but the pip
# CUDA wheel lays out `lib/` (no lib64) and ships only `libcudart.so.13` — there is
# no bare `libcudart.so` for -lcudart to resolve, so the link step dies with
# "/usr/bin/ld: cannot find -lcudart" *after* all 67 CUDA compiles succeed.
# Recreate both names (idempotent; a pip reinstall wipes them).
if [ -n "${CUDA_HOME:-}" ] && [ -d "$CUDA_HOME/lib" ]; then
  [ -e "$CUDA_HOME/lib64" ] || ln -sfn "$CUDA_HOME/lib" "$CUDA_HOME/lib64"
  for _l in cudart cudart_static; do
    _real=$(ls "$CUDA_HOME/lib/lib${_l}.so."* 2>/dev/null | head -1 || true)
    [ -n "$_real" ] && [ ! -e "$CUDA_HOME/lib/lib${_l}.so" ] && ln -sfn "$_real" "$CUDA_HOME/lib/lib${_l}.so"
  done
  export LIBRARY_PATH="$CUDA_HOME/lib:${LIBRARY_PATH:-}"
fi
# ray workers import train.habitat_agent_loop by FQDN
export PYTHONPATH="$REPO:${PYTHONPATH:-}"
# Let each Ray actor see every GPU and pick its own by local rank; this is the
# only arrangement in which verl's set_device(ray_accelerator_id) is correct.
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}
export HABITAT_ENV_SERVERS=${HABITAT_ENV_SERVERS:-http://127.0.0.1:8100,http://127.0.0.1:8101,http://127.0.0.1:8102,http://127.0.0.1:8103}

n_gpus=${N_GPUS:-6}
rollout_tp=${ROLLOUT_TP:-2}
rollout_n=${ROLLOUT_N:-8}                 # GRPO group size (episodes per task)
# Leaves ~70GB on each 140GB H20 — GPU 0 must also host habitat-sim + SigLIP.
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.5}

# One rollout == one simulated episode, so batches stay small on purpose.
train_batch_size=${TRAIN_BATCH_SIZE:-8}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-8}

# Train on the TRAIN split (37477 OVMM episodes), hold out minival for validation.
# Do NOT train on minival — it is the eval set. (Both overridable.)
train_files=${TRAIN_FILES:-$REPO/data/rl/train.parquet}
val_files=${VAL_FILES:-$REPO/data/rl/minival.parquet}

# An episode accumulates one observation image per turn and never drops it, so
# the response, not the prompt, is where the context goes. See README.
max_prompt_length=${MAX_PROMPT_LENGTH:-8192}
max_response_length=${MAX_RESPONSE_LENGTH:-16384}
# verl leaves rollout.max_model_len null, so vLLM sizes its KV cache for Qwen3.5's
# full 262144-token context window even though a rollout never exceeds
# prompt+response. That reservation is pure waste and it is what makes the engine
# fail with "To serve at least one request with the model's max seq len (262144),
# N GiB KV cache is needed". Pin it to what we actually use.
max_model_len=${MAX_MODEL_LEN:-$((max_prompt_length + max_response_length))}
# Must be >= the longest single sequence: verl cannot split one sequence across
# micro-batches (seqlen_balancing asserts max_token_len >= max_seq_len).
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}

# Qwen3.5-9B's own recommended sampling. verl defaults (temp 1.0 / top_p 1.0 /
# top_k -1) are wrong for this model. (ROLLOUT_TEMPERATURE/TOP_P/TOP_K override.)
#
# ENABLE_THINKING=False is the important one. Qwen3.5's chat template ends the
# generation prompt with an *open* `<think>`, so the policy starts inside a
# reasoning block; on this long multimodal prompt it never closes it within the
# per-turn cap, emits no JSON action, and every turn burns max_turn_tokens.
# enable_thinking=False renders `<think></think>` pre-closed, so the model answers
# directly. Turning thinking back on needs a much larger max_turn_tokens.

# ── LoRA IS BROKEN WITH Qwen3.5 ON THIS verl/vLLM BUILD ──────────────────────
# Any lora_rank > 0 makes verl's LoRA->vLLM adapter sync emit NaN logits: every
# generation comes back as max_tokens x "!" (token id 0), with
# training/rollout_probs_diff_max:nan, and rewards/pg_loss/grad_norm all 0.
#
# Established by bisection (each tested standalone, same GPU, same toolchain):
#   base model, plain vLLM ............... SANE (5 -> 5466 prompt tokens)
#   enable_lora=True, no adapter ......... SANE
#   1-3 real images via the processor .... SANE
#   enforce_eager=True ................... still NaN
#   layered_summon=False ................. still NaN
#   lora_rank=64, target_modules=all-linear ......... NaN
#   lora_rank=64, target_modules=[q,k,v,o,gate,up,down] ... NaN  <- not all-linear
#   lora_rank=0 (full fine-tune) ......... SANE, valid tool call on turn 1
#
# So it is LoRA itself, not the target-module set, and not the GDN layers.
# Default to full fine-tune. Set LORA_RANK=64 only to reproduce the bug, and
# ALWAYS check the first generation (HABITAT_AL_DEBUG=/path) before trusting a
# single metric — a NaN policy still produces green, meaningless training steps.
lora_rank=${LORA_RANK:-0}
lora_alpha=${LORA_ALPHA:-32}
actor_lr=${ACTOR_LR:-1e-6}
target_modules=${TARGET_MODULES:-'[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]'}

# Qwen3.5 is a hybrid model: 3 of every 4 layers use gated-delta-net (GDN) linear
# attention. vLLM's FlashInfer GDN prefill kernel `fi_chunk_gated_delta_rule` has a
# known illegal-memory-access bug (vllm-project/vllm#34948, #35945) that crashes the
# engine mid-rollout (~gen 18-20 here) with an *async* "CUDA error: invalid argument"
# that surfaces downstream in the sampler. enable_chunked_prefill=False did NOT avoid
# it. The real fix: force the Triton/FLA GDN prefill kernel via
#   additional_config.gdn_prefill_backend=triton
# On H20 (SM90) vLLM's default "auto" picks FlashInfer (the buggy path); "triton"
# selects vllm/model_executor/layers/mamba/gdn ... fla_chunk_gated_delta_rule, a
# first-class supported backend. (On zp-nc12 the FlashInfer GDN kernel failed to
# JIT-build and fell back anyway, which is why the crash only appeared once the
# toolchain was fixed on nc35.) GDN_BACKEND=flashinfer to reproduce the crash.

# limit_images MUST be set for multi-image prompts. verl only passes
# limit_mm_per_prompt to vLLM when rollout.limit_images is set, and vLLM defaults to
# ONE image per prompt. This agent sends 2 images on turn 1 and accumulates more
# every turn, so the extra images are dropped, the <|image_pad|> placeholders no
# longer line up with the vision features, and the engine emits NaN logits —
# surfacing as generations of pure "!" (token id 0) and rollout_probs_diff_max:nan.
# Keep this >= max_turns * max_images_per_turn (agent_loop_config.yaml).
#
# layered_summon (which forces load_format=safetensors) is a memory optimisation for
# very large models; the 9B does not need it. Left off by default.

# flash-attn is not installed in the verl env (no nvcc, and no prebuilt wheel for
# torch 2.11/cu130), so fall back to SDPA. verl's padding-free path calls
# flash_attn_varlen_func, hence use_remove_padding must be off too. Install
# flash-attn and set ATTN_IMPL=flash_attention_2 REMOVE_PADDING=True to get the
# faster path back.
attn_impl=${ATTN_IMPL:-sdpa}
remove_padding=${REMOVE_PADDING:-False}

project_name=${PROJECT_NAME:-embodied_habitat_grpo}
experiment_name=${EXPERIMENT_NAME:-qwen3_5_9b_lora_ovmm}

cd "$REPO"

# Batching mode. verl's dynamic-batch path (use_dynamic_bsz=True) calls
# rearrange_micro_batches -> index_select_tensor_dict, which does a torch.nested
# unbind over the per-sample multimodal image tensors (variable image counts =>
# ragged/nested). That unbind fails ("Expected cond to be True") for our multimodal
# batches. Fixed-size micro-batching skips rearrange_micro_batches entirely, so it
# is the default here. USE_DYNAMIC_BSZ=True to opt back into the (broken-for-mm) path.
if [ "${USE_DYNAMIC_BSZ:-False}" = "True" ]; then
  BSZ_ARGS=(
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
  )
else
  mbs=${MICRO_BATCH_SIZE:-1}    # per-GPU; 1 is safest for long multimodal contexts
  BSZ_ARGS=(
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${mbs}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${mbs}
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${mbs}
  )
fi

# ── The verl invocation ───────────────────────────────────────────────────────
# Hydra overrides for verl's PPO entrypoint. A leading `+` ADDS a key that is not in
# verl's config schema; without it hydra rejects the override as unknown.
#
# Guide to the arg groups below (they are not otherwise annotatable — a comment cannot
# be placed inside a backslash-continued command):
#
#   algorithm.*
#     GRPO: advantages come from the reward spread within each group of `rollout.n`
#     episodes sampled for the same task, so there is no value network to train.
#     use_kl_in_reward=False together with actor.use_kl_loss=True puts the KL penalty in
#     the LOSS rather than folding it into the reward — that split is GRPO's formulation.
#
#   data.*
#     filter_overlong_prompts=False because our parquet `prompt` is a stub (the real
#     prompt is rendered by the env server at rollout time), so there is nothing
#     meaningful to length-filter. truncation='error' makes a genuine overflow loud
#     instead of silently slicing a multimodal sequence — which corrupts image blocks.
#
#   actor_rollout_ref.model.*
#     gradient checkpointing is what lets the 9B fit alongside long multimodal contexts.
#     See the LoRA note above: lora_rank defaults to 0 (full fine-tune) on purpose.
#
#   actor_rollout_ref.actor.*
#     kl_loss_type=low_var_kl is the k3 estimator — unbiased, and far lower variance than
#     naive KL. Offload is off: there is headroom once the rollout is capped at 0.5.
#
#   actor_rollout_ref.rollout.*
#     mode=async is REQUIRED for a custom agent loop — the loop awaits HTTP round-trips to
#     the env server between turns, which the sync path cannot express.
#     n = the GRPO group size.
#     free_cache_engine returns vLLM's KV cache to FSDP between rollout and training.
#     limit_images and gdn_prefill_backend are both load-bearing bug workarounds; see the
#     long notes above before touching either.
#     agent.* is what wires in HabitatAgentLoop and makes the rollout embodied at all.
#
#   actor_rollout_ref.ref.*
#     The frozen KL anchor. Offloaded because it only ever produces log-probs — no grads.
#
#   reward_model.enable=False
#     There is no reward model. Reward is AgentLoopOutput.reward_score, which the env
#     server computes from Habitat's PDDL success predicate.
#
#   trainer.test_freq=-1
#     Disables periodic validation: a val pass runs real Habitat episodes and is slow.
#     Run it deliberately instead.
$VERL_PY -m verl.trainer.main_ppo \
    "${BSZ_ARGS[@]}" \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$train_files" \
    data.val_files="$val_files" \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING:-False} \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    +actor_rollout_ref.model.override_config.attn_implementation=${attn_impl} \
    actor_rollout_ref.model.use_remove_padding=${remove_padding} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=${lora_rank} \
    actor_rollout_ref.model.lora_alpha=${lora_alpha} \
    actor_rollout_ref.model.target_modules="${target_modules}" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util} \
    actor_rollout_ref.rollout.max_model_len=${max_model_len} \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE:-0.6} \
    actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P:-0.95} \
    actor_rollout_ref.rollout.top_k=${ROLLOUT_TOP_K:-20} \
    actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER:-True} \
    actor_rollout_ref.rollout.enable_chunked_prefill=${ENABLE_CHUNKED_PREFILL:-False} \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.gdn_prefill_backend=${GDN_BACKEND:-triton} \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.layered_summon=${LAYERED_SUMMON:-False} \
    actor_rollout_ref.rollout.load_format=${ROLLOUT_LOAD_FORMAT:-auto} \
    +actor_rollout_ref.rollout.limit_images=${LIMIT_IMAGES:-32} \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$REPO/train/agent_loop_config.yaml" \
    actor_rollout_ref.rollout.agent.default_agent_loop=habitat \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=False \
    trainer.balance_batch=True \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.n_gpus_per_node=${n_gpus} \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=-1 \
    trainer.val_before_train=${VAL_BEFORE_TRAIN:-False} \
    trainer.total_epochs=${TOTAL_EPOCHS:-20} \
    "$@"
