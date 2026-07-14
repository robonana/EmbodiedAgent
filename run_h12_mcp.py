"""run_h12_mcp.py — drive PromptEmbodiedAgent against the Humanoid_Simulation
ROS stack over MCP (h12_mcp_server).

Prereq: the MCP server is running (inside the ROS container) and reachable, e.g.
    http://127.0.0.1:8000/mcp
and GEMINI_API_KEY (or GOOGLE_API_KEY) is set.

Usage:
    python run_h12_mcp.py --task "open the drawer"
    python run_h12_mcp.py --server_url http://127.0.0.1:8000/mcp --max_agent_steps 20
"""
import argparse
import json
import os
import shutil
from pathlib import Path

from agent.gemini_client import GeminiClient
from agent.mcp_toolbox import MCPToolbox
from agent.prompt_agent import PromptEmbodiedAgent


def main():
    p = argparse.ArgumentParser(description="Run the embodied agent over MCP")
    p.add_argument("--task", default="open the drawer", help="task instruction")
    p.add_argument("--server_url", default=os.environ.get("H12_MCP_URL", "http://127.0.0.1:8000/mcp"))
    p.add_argument("--gemini_model", default="gemini-2.5-pro")
    p.add_argument("--max_agent_steps", type=int, default=20)
    p.add_argument("--log_dir", default="runs/h12_mcp")
    args = p.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("Set GEMINI_API_KEY (or GOOGLE_API_KEY).")

    log_dir = Path(args.log_dir)
    if log_dir.exists():
        shutil.rmtree(log_dir, ignore_errors=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    gemini = GeminiClient(api_key=api_key, model_name=args.gemini_model, log_dir=str(log_dir))
    toolbox = MCPToolbox(server_url=args.server_url, log_dir=str(log_dir))
    agent = PromptEmbodiedAgent(
        toolbox=toolbox,
        gemini_client=gemini,
        log_dir=str(log_dir),
        max_agent_steps=args.max_agent_steps,
    )

    try:
        result = agent.run(task=args.task) or {}
    finally:
        toolbox.close()

    print("\n=== RESULT ===")
    print(json.dumps(result, indent=2, default=str))
    (log_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
