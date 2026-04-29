"""
agent/ — Prompt-engineering embodied AI baseline (PromptEmbodiedAgent).

Gemini-2.5-Pro controls the XLeRobot in a closed observe→act→verify loop
using a fixed generic tool grammar.  No training, no RL, no task-specific tools.

Entry point:  navigate.py --agent_mode prompt --task "Bring me the water bottle"
"""
