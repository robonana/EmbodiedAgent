"""
sceneagent — VLM-powered embodied agent pipeline for XLeRobot.

The organising principle is the horizontal line between agent/ + memory/ (which know nothing
about any simulator and talk only to a ToolboxProtocol) and sim/ (which implements that
protocol for one particular world). Swapping ManiSkill for Habitat, or for a real robot, means
writing a new toolbox and changing nothing above the line. sim/habitat_toolbox.py and
agent/mcp_toolbox.py are the other two implementations that exist today.

Package layout
--------------
agent/      Core agent pipeline — fully simulator-independent.
              prompt_agent.py    PromptEmbodiedAgent (main loop)
              gemini_client.py   Gemini API wrapper
              prompts.py         System + per-step prompts
              schemas.py         ToolAction, ToolResult, MemoryCandidate, …
              toolbox.py         ToolboxProtocol (interface for any robot backend)
              episodic_memory.py Append-only observation store
              trajectory_logger  Per-episode JSONL logging
              tasks.py           Task config registry
              verifier.py        Argument validation helpers

memory/     Frame indexing and retrieval — simulator-independent, works on files.
              embedding.py       EmbeddingWorker (real-time FAISS indexing + query)
              retrieval.py       retrieve_memory_candidates()

sim/        ManiSkill / ReplicaCAD implementation — swap out for real robot.
              env.py             ManiSkill env patches, robot-state helpers
              capture.py         Frame capture from ManiSkill sensor observations
              nav_grid.py        NavGrid (2-D A*) + kinematic_nav_step
              setup.py           setup_sim() — creates the ManiSkill gym env
              tools.py           AgentToolbox for ManiSkill (implements ToolboxProtocol)
              render_object.py   Render ReplicaCAD object views as retrieval queries
              habitat_toolbox.py Habitat/OVMM backend (the one the benchmarks use)
              frontier.py        Occupancy mapping + frontier exploration

run.py      Entry point — wires sim + agent + memory together.

Other entry points, all of which assemble the same three pieces (a toolbox, a VLM client, and
PromptEmbodiedAgent) against a different world:
    run_habitat.py         Habitat, free-form task
    run_ovmm_embodied.py   the ai-habitat OVMM benchmark
    run_owmm_embodied.py   the OWMM benchmark
    run_h12_mcp.py         a real H1-2 robot, over MCP
    train/                 the same env again, but wrapped in an HTTP server for RL
"""
