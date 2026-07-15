#!/usr/bin/env bash
# train/launch_env_servers.sh — start N Habitat env servers.
#
# Each server owns a habitat-sim GL context and drives one episode at a time, so
# rollout concurrency == number of servers.
#
# Servers render on RENDER_GPU, which must be disjoint from the trainer's GPUs
# (see train/run_grpo_qwen9b.sh) — habitat-sim and vLLM will OOM each other.
# Default: GPU 7, trainer takes 0-5.
#
# habitat-sim picks its EGL device by physical CUDA ordinal, so the server pins
# CUDA_VISIBLE_DEVICES=$RENDER_GPU itself and hands habitat the remapped index 0.
# (On the old zp-nc12 only GPU 0 could render — that was a symptom of 7 wedged
# GPUs, not an EGL limit. On healthy hardware any GPU works.)
#
#   bash train/launch_env_servers.sh 4            # 4 servers on GPU 7, ports 8100..8103
#   RENDER_GPU=6 N_SERVERS=2 bash train/launch_env_servers.sh
#
# Stop them with:  tmux kill-session -t habenv
set -euo pipefail   # fail fast; an unset var here means a silently misconfigured server

# Every knob is ${VAR:-default}, so anything can be overridden from the environment
# without editing this file. The positional $1 is a convenience alias for N_SERVERS.
N_SERVERS=${1:-${N_SERVERS:-4}}
RENDER_GPU=${RENDER_GPU:-7}               # keep disjoint from the trainer's GPUs
BASE_PORT=${BASE_PORT:-8100}              # servers listen on BASE_PORT .. BASE_PORT+N-1
SPLIT=${SPLIT:-minival}
MAX_STEPS=${MAX_STEPS:-16}
# The HABITAT interpreter (3.9) — deliberately NOT the verl one. See the env server's
# module docstring for why the two cannot share a Python.
HAB_PY=${HAB_PY:-/data1/chen/conda/envs/habitat/bin/python}
REPO=${REPO:-/data1/chen/EmbodiedAgent}
LOG_DIR=${LOG_DIR:-$REPO/runs/rl_env/logs}

# The env server must reach two things zp-nc35 firewalls off: the HF hub (for the
# SigLIP retrieval extractor) and the tool-VLM used by inspect/detect/rerank.
#
#   · SigLIP is cached locally, so force offline HF — otherwise every server spends
#     minutes on huggingface.co retries, then runs with retrieval DISABLED.
#
#   · Tool-VLM = Gemini, but reached via its OpenAI-COMPATIBLE endpoint, not the
#     google.generativeai SDK. The SDK's default gRPC transport ignores HTTPS_PROXY
#     and hangs on this proxy-only box; the OpenAI endpoint is plain HTTPS and
#     honours the proxy. Two payload fixes for that endpoint (proven necessary):
#       VLM_THINKING_KWARG=0  drops chat_template_kwargs (a vLLM-only field → 400)
#       VLM_ENABLE_THINKING=0 forces greedy, which drops top_k (vLLM ext → 400)
#     TOOL_VLM_MODEL must be a live model (gemini-2.5-flash; -3.5-flash also exists).
#     For a local OpenAI server instead, set TOOL_VLM_BASE_URL to it and VLM_PROXY=''.
export HF_HOME=${HF_HOME:-/data1/chen/huggingface}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

TOOL_VLM_SERVICE=${TOOL_VLM_SERVICE:-openai}    # Gemini via its OpenAI-compat surface
TOOL_VLM_BASE_URL=${TOOL_VLM_BASE_URL:-https://generativelanguage.googleapis.com/v1beta/openai}
TOOL_VLM_MODEL=${TOOL_VLM_MODEL:-gemini-2.5-flash}
TOOL_VLM_API_KEY=${TOOL_VLM_API_KEY:-${GEMINI_API_KEY:-}}
export VLM_THINKING_KWARG=${VLM_THINKING_KWARG:-0}
export VLM_ENABLE_THINKING=${VLM_ENABLE_THINKING:-0}
export VLM_TEMPERATURE=${VLM_TEMPERATURE:-0}

VLM_PROXY=${VLM_PROXY:-http://127.0.0.1:7890}   # empty for a local tool-VLM
if [ -n "$VLM_PROXY" ]; then
  export HTTPS_PROXY="$VLM_PROXY" HTTP_PROXY="$VLM_PROXY"
  # keep localhost (vLLM rollout server, /health, sibling servers) off the proxy.
  export NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost"
fi

mkdir -p "$LOG_DIR"
# tmux (rather than plain background jobs) so the servers survive the launching shell and
# can be inspected live. Kill any previous session first — otherwise the old servers still
# hold the ports and every new one dies on bind.
tmux kill-session -t habenv 2>/dev/null || true
tmux new-session -d -s habenv -n main "echo habenv; sleep infinity"

# One tmux window per server, each on its own port.
for i in $(seq 0 $((N_SERVERS - 1))); do
  port=$((BASE_PORT + i))
  log="$LOG_DIR/env_${port}.log"
  echo "[launch] port=$port gpu=$RENDER_GPU log=$log"
  # The server pins CUDA_VISIBLE_DEVICES itself from --gpu_id; do not set it here.
  tmux new-window -t habenv -n "env$port" \
    "cd $REPO && MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet \
     HF_HOME=$HF_HOME HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE \
     HTTPS_PROXY=${HTTPS_PROXY:-} HTTP_PROXY=${HTTP_PROXY:-} \
     NO_PROXY=${NO_PROXY:-} no_proxy=${no_proxy:-} \
     VLM_THINKING_KWARG=$VLM_THINKING_KWARG VLM_ENABLE_THINKING=$VLM_ENABLE_THINKING VLM_TEMPERATURE=$VLM_TEMPERATURE \
     GEMINI_API_KEY=${GEMINI_API_KEY:-} \
     $HAB_PY -m train.habitat_env_server \
       --port $port --gpu_id $RENDER_GPU --split $SPLIT --max_steps $MAX_STEPS \
       --tool_vlm_service $TOOL_VLM_SERVICE \
       --tool_vlm_base_url $TOOL_VLM_BASE_URL \
       --tool_vlm_model $TOOL_VLM_MODEL \
       --tool_vlm_api_key $TOOL_VLM_API_KEY \
       > $log 2>&1"
done

# Block until every server answers /health, so the caller can start training immediately
# after this returns rather than racing a still-importing habitat-sim.
# --noproxy '*' is essential: HTTPS_PROXY is exported above for the tool-VLM, and without
# this curl would dutifully try to reach 127.0.0.1 *through the proxy* and fail.
# 60 × 2 s = up to two minutes, which is about how long the first habitat import takes.
echo "[launch] waiting for /health ..."
for i in $(seq 0 $((N_SERVERS - 1))); do
  port=$((BASE_PORT + i))
  for _ in $(seq 1 60); do
    if curl -sf --noproxy '*' "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      echo "  port $port up"; break
    fi
    sleep 2
  done
done
echo "[launch] done — tmux attach -t habenv"
