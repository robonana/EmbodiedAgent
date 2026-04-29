"""
sceneagent — VLM-powered embodied agent pipeline for XLeRobot.

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

run.py      Entry point — wires sim + agent + memory together.
"""
