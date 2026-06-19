#!/bin/bash
# Run EmbodiedAgent on the ai-habitat OVMM benchmark with live pygame rendering.
#
# HABITAT_RENDER=1  show live pygame window (robot camera feed)
# HABITAT_STEP=1    pause after each Gemini decision (SPACE to execute, Q to quit)
# VLM_MODEL         override Gemini model (default: models/gemini-3.5-flash)
# SPLIT             OVMM split: minival (default) | train | val
# EPISODES          space-separated episode ids (default: 0)
#
# Examples:
#   bash run_ovmm_embodied.sh                      # minival ep0, step-through window
#   EPISODES="0 1 2" bash run_ovmm_embodied.sh     # three episodes
#   HABITAT_STEP=0 bash run_ovmm_embodied.sh       # continuous (don't wait for SPACE)
#   bash run_ovmm_embodied.sh --scan_points 8      # pass extra runner args through

cd "$(dirname "$0")"

SPLIT=${SPLIT:-train}
EPISODES=${EPISODES:-0}

# Prereq check: converted episodes must exist (see tools/convert_ovmm_episodes.py)
if [ ! -f "data/datasets/ovmm/${SPLIT}.json.gz" ]; then
    echo "Converting OVMM '${SPLIT}' split first…"
    python tools/convert_ovmm_episodes.py "${SPLIT}"
fi

HABITAT_RENDER=${HABITAT_RENDER:-1} HABITAT_STEP=${HABITAT_STEP:-1} \
python -u run_ovmm_embodied.py \
    --split "$SPLIT" \
    --episode_ids $EPISODES \
    --no_drop_missing \
    --explore \
    --explore_iters 10 \
    --explore_range 6 \
    --explore_min_gain 2 \
    --explore_video \
    --max_agent_steps 40 \
    --log_dir runs/ovmm_embodied \
    ${@}
