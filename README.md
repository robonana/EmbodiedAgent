# EmbodiedAgent (SceneAgent)

A closed-loop embodied agent for the [Habitat](https://aihabitat.org/) simulator. A
vision-language model acts as the policy: at each step it receives an egocentric
observation plus episodic memory and emits one structured tool call
(`navigate`, `manipulate`, `detect`, `inspect`, …). The same agent runs on the
**OVMM** and **OWMM** mobile-manipulation benchmarks and can be
**reinforcement-fine-tuned** with GRPO against Habitat episodes (see [`train/`](train/)).

## Repository layout

| path | what it is |
|---|---|
| `agent/` | the policy agent — the multi-turn loop, tool schemas, VLM clients, episodic memory, verifier |
| `sim/` | Habitat integration: `HabitatToolbox` (the 9 tools), capture, navigation grid, frontier exploration |
| `memory/` | visual retrieval — SigLIP/DINOv2 feature extractors + a FAISS index over observations |
| `tools/` | one-off utilities: OVMM episode conversion, target-view rendering, camera tuning |
| `train/` | **verl GRPO reinforcement-learning pipeline** — Habitat behind an HTTP env server, Qwen3.5-9B policy. See [`train/README.md`](train/README.md) |
| `run_ovmm_embodied.py` | run the agent on the ai-habitat **OVMM** benchmark |
| `run_owmm_embodied.py` | run the agent on the **OWMM** benchmark |
| `run_habitat.py`, `run.py` | lower-level / legacy runners |

### The agent, briefly

`agent/prompt_agent.py::PromptEmbodiedAgent` runs the loop: `observe → VLM picks a
tool → execute → update memory → repeat`, until `finish`, a step cap, or a repeat
guard. The policy sees only what a real robot would (RGB + pose + retrieved
memories) — simulator ground truth never enters a prompt. Tools are defined in
`agent/toolbox*.py` / `sim/habitat_toolbox.py`; observations are stored and
retrieved via `memory/`.

## VLM backends

The policy VLM is pluggable via `--vlm_service` (or `VLM_SERVICE`):

- `gemini` — `google.generativeai` (set `GEMINI_API_KEY` / `GOOGLE_API_KEY`)
- `openai` — any OpenAI-compatible `/v1/chat/completions` server, e.g. a local vLLM
  serving Qwen, or Gemini's OpenAI-compatible endpoint
  (`--vlm_base_url`, `VLM_API_KEY`, plus `VLM_TEMPERATURE` / `VLM_TOP_P` / `VLM_TOP_K` / …)

**No API keys are committed to this repo — pass them via the environment.**

## Setup

Habitat and its datasets are large and installed separately (see the
[Habitat-Lab docs](https://github.com/facebookresearch/habitat-lab)). Roughly:

```bash
pip install -e .                       # this package ("sceneagent")
# + a working habitat-sim / habitat-lab install, and OVMM/OWMM data under data/
```

`data/`, `runs/`, and the vendored `OWMM-Agent/` are git-ignored (large / generated).

## Running the benchmark

```bash
# OVMM, one episode, Gemini policy
GEMINI_API_KEY=... python run_ovmm_embodied.py --split minival --episode_ids 0

# OVMM with a local OpenAI-compatible (e.g. Qwen) policy server
VLM_SERVICE=openai VLM_BASE_URL=http://localhost:8000/v1 \
  python run_ovmm_embodied.py --split minival --max_episodes 3
```

Per-episode trajectories (prompts, raw VLM output, images, final result) are logged
under `runs/` for later SFT / rejection-sampling / RL.

## Reinforcement learning

`train/` fine-tunes the policy with **GRPO**, driving real Habitat episodes and
rewarding task success (`pddl_success`). Because Habitat needs Python 3.9 and verl
needs ≥3.10, the simulator runs as an HTTP **env server** that the trainer calls.
Full details, setup, and the (many) environment gotchas are in
**[`train/README.md`](train/README.md)**.
