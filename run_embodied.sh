#!/bin/bash
# Run EmbodiedAgent on the OWMM benchmark with live pygame rendering.
#
# HABITAT_RENDER=1  show live pygame window
# HABITAT_STEP=1    pause after each Gemini decision (SPACE to execute, Q to quit)
# VLM_MODEL         override Gemini model (default: models/gemini-3.5-flash)
# DATASET           dataset dir under data/datasets/ (default: train set; use
#                   sat_TEST_YCB_30scene_head_rgb for the test set)

cd "$(dirname "$0")"

DATASET=${DATASET:-sat_TRAIN_YCB_30scene_head_rgb}

# Default episode selection per dataset. The train set only has the 10 episodes
# we extracted (0-9), and clean human task strings for 8 of them — eps 2 and 7
# lack a task_prompt.json entry (would fall back to PDDL hash names), so the
# train default is restricted to the 8 episodes with proper instructions.
# Override by passing --episode_ids ... as extra args.
if [ "$DATASET" = "sat_TRAIN_YCB_30scene_head_rgb" ]; then
    EP_ARGS="--episode_ids 0 1 3 4 5 6 8 9"
else
    EP_ARGS="--max_episodes 1"
fi

HABITAT_RENDER=1 HABITAT_STEP=1 \
python -u run_owmm_embodied.py \
    --dataset "$DATASET" \
    $EP_ARGS \
    --max_agent_steps 40 \
    --log_dir runs/owmm_embodied \
    ${@}
