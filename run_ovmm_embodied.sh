#!/bin/bash
# Run EmbodiedAgent on the ai-habitat OVMM benchmark with live pygame rendering.
#
# HABITAT_RENDER=1  show live pygame window (robot camera feed)
# HABITAT_STEP=1    pause after each VLM decision (SPACE to execute, Q to quit)
# VLM_SERVICE       policy backend: gemini (default) | openai (local OpenAI-API server)
# VLM_BASE_URL      base URL for VLM_SERVICE=openai (default: http://localhost:23333/v1)
# VLM_MODEL         override policy model (gemini default: models/gemini-3.5-flash;
#                   openai default: Qwen3.5-9B)
# VLM_API_KEY       bearer token for VLM_SERVICE=openai (optional)
# SPLIT             OVMM split: minival (default) | train | val
# EPISODES          space-separated episode ids (default: 0)
#
# Examples:
#   bash run_ovmm_embodied.sh                      # minival ep0, step-through window
#   EPISODES="0 1 2" bash run_ovmm_embodied.sh     # three episodes
#   HABITAT_STEP=0 bash run_ovmm_embodied.sh       # continuous (don't wait for SPACE)
#   bash run_ovmm_embodied.sh --scan_points 8      # pass extra runner args through
#   # Use a local OpenAI-compatible VLM (e.g. Qwen3.5-9B) instead of Gemini:
#   VLM_SERVICE=openai bash run_ovmm_embodied.sh
#   VLM_SERVICE=openai VLM_MODEL=Qwen3.5-9B \
#       VLM_BASE_URL=http://localhost:23333/v1 bash run_ovmm_embodied.sh

# Run from the repo root regardless of where the script was invoked from — every path below
# (and the runner's own data/ lookups) is relative to it.
cd "$(dirname "$0")"

SPLIT=${SPLIT:-train}
EPISODES=${EPISODES:-0}

# VLM backend selection (passed through to the runner, which also honours the
# VLM_SERVICE / VLM_BASE_URL / VLM_API_KEY env vars directly).
# Built as an ARRAY and expanded quoted ("${VLM_ARGS[@]}") so a value containing spaces
# survives intact. Each flag is only added when its env var is non-empty, letting the runner's
# own defaults apply otherwise — passing an empty --vlm_model would override them with "".
VLM_ARGS=()
if [ -n "$VLM_SERVICE" ];  then VLM_ARGS+=(--vlm_service "$VLM_SERVICE"); fi
if [ -n "$VLM_BASE_URL" ]; then VLM_ARGS+=(--vlm_base_url "$VLM_BASE_URL"); fi
if [ -n "$VLM_MODEL" ];    then VLM_ARGS+=(--vlm_model "$VLM_MODEL"); fi
if [ -n "$VLM_API_KEY" ];  then VLM_ARGS+=(--vlm_api_key "$VLM_API_KEY"); fi

# Prereq check: converted episodes must exist (see tools/convert_ovmm_episodes.py).
# Auto-convert rather than erroring — it is a pure function of the downloaded dataset, so
# there is never a reason to make the user run it by hand.
if [ ! -f "data/datasets/ovmm/${SPLIT}.json.gz" ]; then
    echo "Converting OVMM '${SPLIT}' split first…"
    python tools/convert_ovmm_episodes.py "${SPLIT}"
fi

# -u: unbuffered, so the agent's reasoning streams to the terminal live rather than appearing
# in bursts when the pipe buffer fills.
# $EPISODES is intentionally UNQUOTED: it must word-split into several ids for --episode_ids.
# ${@} forwards any extra flags the caller passed straight through to the runner.
HABITAT_RENDER=${HABITAT_RENDER:-1} HABITAT_STEP=${HABITAT_STEP:-1} \
python -u run_ovmm_embodied.py \
    --split "$SPLIT" \
    --episode_ids $EPISODES \
    --no_drop_missing \
    --explore \
    --explore_iters 40 \
    --explore_range 1.5 \
    --explore_lambda 0.5 \
    --explore_min_gain 1 \
    --explore_video \
    --max_agent_steps 40 \
    --log_dir runs/ovmm_embodied \
    "${VLM_ARGS[@]}" \
    ${@}
