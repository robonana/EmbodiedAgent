"""
agent/ — Prompt-engineering embodied AI baseline (PromptEmbodiedAgent).

Gemini-2.5-Pro controls the XLeRobot in a closed observe→act loop,
including previous-action verification at the start of each VLM response.
No training, no RL, no task-specific tools.

Entry point:  navigate.py --agent_mode prompt --task "Bring me the water bottle"
"""
