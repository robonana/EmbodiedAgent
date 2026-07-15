"""
agent/tests/test_agent.py — Lightweight smoke tests for the agent pipeline.

Run from the repo root:
    python -m sceneagent.agent.tests.test_agent

Tests do NOT require ManiSkill, SAPIEN, or Gemini API.
Tests 1-4 are fully offline.  Test 5 requires --real_gemini flag + API key.

The offline constraint drives the whole design here. Booting Habitat takes tens of
seconds and needs a GPU, and calling Gemini costs money and is non-deterministic, so
neither may appear in the default suite. Instead we test the parts that are pure logic —
JSON coercion, schema round-trips, argument validation, the agent's control flow (via a
mock VLM that returns a scripted action sequence), and the on-disk memory/logging stores.

Imports are inside each test method rather than at module level so that a broken or
missing optional dependency in one module cannot prevent the entire suite from loading.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock, patch


# ── Test 1: JSON parsing ──────────────────────────────────────────────────────

class TestJSONParsing(unittest.TestCase):
    """GeminiClient._parse_json handles valid, fenced, and invalid inputs.

    One case per wrapping the model has actually been observed to produce. The two
    negative cases matter as much as the positive ones: _parse_json must return None
    (triggering the repair-prompt retry) rather than raising or silently returning
    something half-parsed.

    _parse_json is a staticmethod, so these run without constructing a client — no API key,
    no network.
    """

    def _parse(self, text: str):
        from sceneagent.agent.gemini_client import GeminiClient
        return GeminiClient._parse_json(text)

    def test_valid_plain_json(self):
        raw = '{"tool": "observe", "arguments": {}, "rationale": "start"}'
        result = self._parse(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result["tool"], "observe")

    def test_json_in_markdown_fence(self):
        raw = '```json\n{"tool": "navigate", "arguments": {"target": {}}}\n```'
        result = self._parse(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result["tool"], "navigate")

    def test_json_in_plain_fence(self):
        raw = '```\n{"tool": "finish", "arguments": {}}\n```'
        result = self._parse(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result["tool"], "finish")

    def test_json_embedded_in_prose(self):
        # Exercises the brace-matching fallback: the model chatted around its JSON.
        raw = 'Here is my action:\n{"tool": "inspect", "arguments": {"question": "test"}}\nDone.'
        result = self._parse(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result["tool"], "inspect")

    def test_invalid_json_returns_none(self):
        raw = "This is not JSON at all."
        result = self._parse(raw)
        self.assertIsNone(result)

    def test_partial_json_returns_none(self):
        # Truncated output (hit the token cap mid-object). Brace matching must not
        # "helpfully" close it — an incomplete action is worse than no action.
        raw = '{"tool": "observe", "arguments":'
        result = self._parse(raw)
        self.assertIsNone(result)


# ── Test 2: Schema validation ─────────────────────────────────────────────────

class TestSchemas(unittest.TestCase):
    """ToolAction and ToolResult round-trip correctly."""

    def test_tool_action_from_dict(self):
        from sceneagent.agent.schemas import ToolAction
        d = {
            "tool": "observe",
            "arguments": {},
            "previous_action_verification": "Previous action verification: no previous action yet.",
            "rationale": "test",
            "expected_progress": "get image",
        }
        action = ToolAction.from_dict(d)
        self.assertEqual(action.tool, "observe")
        self.assertEqual(action.rationale, "test")
        self.assertEqual(
            action.previous_action_verification,
            "Previous action verification: no previous action yet.",
        )

    def test_tool_result_to_dict(self):
        from sceneagent.agent.schemas import ToolResult
        result = ToolResult(
            ok=True, tool="observe", summary="test",
            data={"image_path": "/tmp/test.png"},
        )
        d = result.to_dict()
        self.assertTrue(d["ok"])
        self.assertEqual(d["tool"], "observe")

    def test_memory_candidate_to_dict(self):
        from sceneagent.agent.schemas import MemoryCandidate
        mc = MemoryCandidate(
            memory_id="mem_000042",
            image_path="/tmp/000042.png",
            robot_pose=[1.0, 2.0, 0.5],
            retrieval_score=0.87,
        )
        d = mc.to_dict()
        self.assertEqual(d["memory_id"], "mem_000042")
        self.assertAlmostEqual(d["retrieval_score"], 0.87)

    def test_normalize_memory_id(self):
        from sceneagent.agent.schemas import normalize_memory_id
        self.assertEqual(normalize_memory_id("mem42"), "mem_000042")
        self.assertEqual(normalize_memory_id("MEM_42"), "mem_000042")
        self.assertEqual(normalize_memory_id("mem_000042"), "mem_000042")
        self.assertIsNone(normalize_memory_id("kitchen"))





# ── Test 3: Verifier ──────────────────────────────────────────────────────────

class TestVerifier(unittest.TestCase):
    """check_tool_argument_validity — the gate between VLM output and the simulator.

    Two families of test here:
      * Tools that USED to exist and must now be rejected (observe/verify/approach). The
        model, trained on or prompted with older tool lists, still tries to call them;
        these tests pin the rejection so a regression can't silently re-admit them.
      * Argument-shape checks per tool, especially the bbox forms, where a malformed box
        would otherwise reach PIL and crop garbage.
    """

    def test_check_tool_argument_validity_observe_is_invalid(self):
        # observe() is called by the agent loop, not by the VLM — it is not in VALID_TOOLS.
        from sceneagent.agent.verifier import check_tool_argument_validity
        valid, reason = check_tool_argument_validity("observe", {})
        self.assertFalse(valid)
        self.assertEqual(reason, "invalid_tool")

    def test_check_tool_argument_validity_inspect_missing_image_path(self):
        from sceneagent.agent.verifier import check_tool_argument_validity
        valid, reason = check_tool_argument_validity("inspect", {"question": "what?"})
        self.assertFalse(valid)
        self.assertEqual(reason, "invalid_arguments")

    def test_check_tool_argument_validity_inspect_missing_question(self):
        from sceneagent.agent.verifier import check_tool_argument_validity
        valid, reason = check_tool_argument_validity("inspect", {"image_path": "/tmp/x.png"})
        self.assertFalse(valid)
        self.assertEqual(reason, "invalid_arguments")

    def test_navigate_no_pose_rejected(self):
        # Empty target dict — no memory_id under any of its accepted key names.
        from sceneagent.agent.verifier import check_tool_argument_validity
        valid, reason = check_tool_argument_validity(
            "navigate", {"target": {}})
        self.assertFalse(valid)
        self.assertEqual(reason, "invalid_arguments")

    # STALE: this test predates the "memory_id only" restriction on navigate. The verifier
    # now deliberately REJECTS raw coordinate/pose goals (see the comment in
    # verifier.check_tool_argument_validity), so this assertion no longer matches the
    # intended behaviour and will fail. It should be inverted to assertFalse, or deleted.
    def test_navigate_pose_accepted(self):
        from sceneagent.agent.verifier import check_tool_argument_validity
        valid, _ = check_tool_argument_validity(
            "navigate",
            {"target": {"pose": [1.0, 2.0, 0.0]}})
        self.assertTrue(valid)

    def test_navigate_memory_id_accepted(self):
        # The one accepted form.
        from sceneagent.agent.verifier import check_tool_argument_validity
        valid, _ = check_tool_argument_validity(
            "navigate",
            {"target": {"memory_id": "mem_000042"}})
        self.assertTrue(valid)

    def test_unsupported_skill(self):
        # A coherent-but-unimplemented skill gets its own reason code, distinct from
        # "invalid_arguments", so the model is told *why* it can't wipe the board.
        from sceneagent.agent.verifier import check_tool_argument_validity
        valid, reason = check_tool_argument_validity(
            "manipulate", {"skill": "wipe", "target": "board"})
        self.assertFalse(valid)
        self.assertEqual(reason, "unsupported_skill")

    def test_invalid_tool(self):
        from sceneagent.agent.verifier import check_tool_argument_validity
        valid, reason = check_tool_argument_validity("fly_robot", {})
        self.assertFalse(valid)
        self.assertEqual(reason, "invalid_tool")

    def test_verify_tool_is_invalid(self):
        from sceneagent.agent.verifier import check_tool_argument_validity
        valid, reason = check_tool_argument_validity(
            "verify", {"condition": "object visible"})
        self.assertFalse(valid)
        self.assertEqual(reason, "invalid_tool")

    def test_approach_tool_is_invalid(self):
        from sceneagent.agent.verifier import check_tool_argument_validity
        valid, reason = check_tool_argument_validity(
            "approach", {"target": "cup", "desired_distance": 0.5})
        self.assertFalse(valid)
        self.assertEqual(reason, "invalid_tool")

    def test_base_move_valid(self):
        from sceneagent.agent.verifier import check_tool_argument_validity
        valid, reason = check_tool_argument_validity(
            "base_move", {"motion": "rotate -30 degrees"})
        self.assertTrue(valid)
        self.assertIsNone(reason)

    def test_base_move_invalid_motion(self):
        from sceneagent.agent.verifier import check_tool_argument_validity
        valid, reason = check_tool_argument_validity(
            "base_move", {"motion": "diagonal"})
        self.assertFalse(valid)
        self.assertEqual(reason, "invalid_arguments")

    def test_inspect_bad_bbox(self):
        # Single flat box with only 2 coords — must be 4.
        from sceneagent.agent.verifier import check_tool_argument_validity
        valid, _ = check_tool_argument_validity("inspect", {"bbox": [10, 20], "question": "?", "image_path": "/tmp/x.png"})
        self.assertFalse(valid)

    def test_inspect_multi_bbox_valid(self):
        # The list-of-boxes form, which the verifier distinguishes from a single flat box
        # by looking at whether bbox[0] is a number.
        from sceneagent.agent.verifier import check_tool_argument_validity
        valid, reason = check_tool_argument_validity(
            "inspect",
            {"image_path": "/tmp/x.png", "question": "?",
             "bbox": [[0, 0, 100, 100], [200, 200, 300, 300]]})
        self.assertTrue(valid)
        self.assertIsNone(reason)

    def test_inspect_multi_bbox_bad_inner(self):
        # Every box must be well-formed — one bad box invalidates the call, rather than
        # being silently dropped while the others proceed.
        from sceneagent.agent.verifier import check_tool_argument_validity
        valid, reason = check_tool_argument_validity(
            "inspect",
            {"image_path": "/tmp/x.png", "question": "?",
             "bbox": [[0, 0, 100], [200, 200, 300, 300]]})
        self.assertFalse(valid)
        self.assertEqual(reason, "invalid_arguments")


# ── Test 4: Prompt loop dry-run with mock Gemini ──────────────────────────────

class TestPromptLoopMock(unittest.TestCase):
    """Run a short 5-step episode with a deterministic mock Gemini client.

    The closest thing to an end-to-end test that stays offline. Both ends of the agent are
    replaced by mocks — a toolbox that says "ok" to everything, and a VLM that returns a
    scripted list of actions — leaving the real PromptEmbodiedAgent control flow in the
    middle as the thing under test: does it step, log, terminate on finish, and halt on a
    repeat loop?
    """

    def _make_mock_sequence(self) -> list[dict]:
        # A plausible episode arc — search memory, go there, close the gap, declare done —
        # so the loop is exercised across four different tool types, not just one.
        return [
            {"tool": "retrieve_memory", "arguments": {"query": "water bottle", "top_k": 3},
             "previous_action_verification": "Previous action verification: no previous action yet.",
             "progress_analysis": "Need to find the water bottle.",
             "rationale": "not visible","expected_progress": "find memory"},
            {"tool": "navigate",
             "arguments": {"target": {"memory_id": "mem_000001"}},
             "previous_action_verification": "Previous action verification: succeeded; memory candidates were returned.",
             "progress_analysis": "A likely water bottle memory pose is available.",
             "rationale": "go there",   "expected_progress": "move to memory candidate"},
            {"tool": "base_move", "arguments": {"motion": "forward"},
             "previous_action_verification": "Previous action verification: succeeded; the mock navigation completed.",
             "progress_analysis": "At the likely pose; move slightly closer for the mock handoff.",
             "rationale": "close distance", "expected_progress": "adjust base"},
            {"tool": "finish",          "arguments": {"answer": "done"},
             "previous_action_verification": "Previous action verification: succeeded; the mock base_move completed.",
             "progress_analysis": "The mock task is complete.",
             "rationale": "success",    "expected_progress": "end"},
        ]

    def test_dry_run(self):
        """Happy path: the loop runs the scripted actions, terminates, and writes a log."""
        # Build a minimal mock toolbox that returns ok=True for everything
        mock_toolbox = MagicMock()

        def _mock_execute(action):
            from sceneagent.agent.schemas import ToolResult
            is_finish = action.tool == "finish"
            return ToolResult(
                ok=True, tool=action.tool,
                summary=f"mock:{action.tool}",
                data={"summary": f"mock {action.tool}", "task_done": is_finish,
                      "image_path": "/tmp/mock.png",
                      "robot_pose": [0.0, 0.0, 0.0],
"visual_place_hint": {"phrase": "mock area", "confidence": "low",
                                            "evidence": "mock"}},
                image_paths=["/tmp/mock.png"],
            )

        mock_toolbox.execute.side_effect = _mock_execute
        mock_toolbox._last_image_path = "/tmp/mock.png"
        mock_toolbox.observe.return_value = _mock_execute(
            MagicMock(tool="wait", arguments={"seconds": 0}))

        # Mock GeminiClient: walk the scripted sequence, one action per call. The list in
        # `call_idx` is a mutable cell — the closure needs to mutate the counter, and this
        # predates/avoids `nonlocal`.
        sequence = self._make_mock_sequence()
        call_idx = [0]

        mock_gemini = MagicMock()
        def _mock_policy(*args, **kwargs):
            idx = call_idx[0]
            call_idx[0] += 1
            if idx < len(sequence):
                return sequence[idx]
            # Backstop: if the loop somehow asks for more actions than the script has,
            # finish rather than running to max_steps and slowing the suite down.
            return {"tool": "finish", "arguments": {}, "rationale": "done",
                    "expected_progress": "end"}

        mock_gemini.call_policy.side_effect = _mock_policy
        mock_gemini.model_name = "mock-model"   # the loop reads this for logging

        with tempfile.TemporaryDirectory() as tmpdir:
            from sceneagent.agent.prompt_agent import PromptEmbodiedAgent
            agent = PromptEmbodiedAgent(
                toolbox=mock_toolbox,
                gemini_client=mock_gemini,
                log_dir=tmpdir,
                max_agent_steps=10,
                history_window=4,
            )
            result = agent.run("Bring me the water bottle")

            self.assertIn("success", result)
            self.assertIn("total_steps", result)
            self.assertIn("episode_dir", result)
            self.assertGreater(result["total_steps"], 0)
            # Verify trajectory JSONL was written (check while tmpdir still exists)
            # — the assertions must happen inside the `with`, or the directory is gone.
            import glob
            jsonl_files = glob.glob(
                os.path.join(result["episode_dir"], "trajectory.jsonl"))
            self.assertEqual(len(jsonl_files), 1)
            with open(jsonl_files[0]) as f:
                lines = f.readlines()
            self.assertGreater(len(lines), 0)
            first_step = json.loads(lines[0])
            self.assertIn("step_idx", first_step)
            self.assertIn("action", first_step)

    def test_repeat_action_halts(self):
        """Same action 5× should trigger halt before max_steps.

        The anti-livelock guard. A VLM that gets stuck will happily emit the same action
        until the step budget runs out, burning API calls on a wedged state. Here the mock
        always returns the same `wait`, and we assert the loop cuts it short: the halt fires
        at 10 consecutive repeats (~11 steps), well under max_agent_steps=20.
        """
        mock_toolbox = MagicMock()

        def _ok_result(action):
            from sceneagent.agent.schemas import ToolResult
            return ToolResult(
                ok=True, tool=action.tool, summary=f"mock:{action.tool}",
                data={"summary": "mock", "image_path": "/tmp/m.png",
                      "robot_pose": [0.0, 0.0, 0.0],
"visual_place_hint": {"phrase": "mock area", "confidence": "low",
                                            "evidence": "mock"}},
                image_paths=["/tmp/m.png"],
            )

        mock_toolbox.execute.side_effect = _ok_result
        mock_toolbox._last_image_path = "/tmp/m.png"
        mock_toolbox.observe.return_value = _ok_result(MagicMock(tool="wait", arguments={"seconds": 0}))

        mock_gemini = MagicMock()
        # Always return same wait action (observe is no longer a valid tool)
        mock_gemini.call_policy.return_value = {
            "tool": "wait", "arguments": {"seconds": 1},
            "previous_action_verification": "Previous action verification: uncertain in mock.",
            "progress_analysis": "Still waiting in mock.",
            "rationale": "stuck", "expected_progress": "help"}
        mock_gemini.model_name = "mock"

        with tempfile.TemporaryDirectory() as tmpdir:
            from sceneagent.agent.prompt_agent import PromptEmbodiedAgent
            agent = PromptEmbodiedAgent(
                toolbox=mock_toolbox,
                gemini_client=mock_gemini,
                log_dir=tmpdir,
                max_agent_steps=20,
                history_window=4,
            )
            result = agent.run("Test repeat halt")

        # Should halt before max_steps=20 (halts at 10 repeats → 11 steps)
        self.assertLessEqual(result["total_steps"], 15)


# ── Test 5: Memory candidate schema ──────────────────────────────────────────

class TestMemoryCandidateSchema(unittest.TestCase):
    """MemoryCandidate fields and memory_id format."""

    def test_memory_id_format(self):
        from sceneagent.agent.schemas import MemoryCandidate
        mc = MemoryCandidate(
            memory_id="mem_000042",
            image_path="/data/color/000042.png",
            robot_pose=[1.5, -2.3, 0.78],
            retrieval_score=0.92,
        )
        self.assertTrue(mc.memory_id.startswith("mem_"))
        self.assertAlmostEqual(mc.robot_pose[0], 1.5)

    def test_memory_id_from_module(self):
        from sceneagent.agent.episodic_memory import frame_to_memory_id
        self.assertEqual(frame_to_memory_id(0),   "mem_000000")
        self.assertEqual(frame_to_memory_id(42),  "mem_000042")
        self.assertEqual(frame_to_memory_id(999), "mem_000999")

    def test_pose_str(self):
        from sceneagent.agent.schemas import MemoryCandidate
        mc = MemoryCandidate(
            memory_id="mem_000001",
            image_path="/tmp/x.png",
            robot_pose=[2.0, 3.0, 1.57],
            retrieval_score=0.5,
        )
        s = mc.pose_str()
        self.assertIn("2.00", s)
        self.assertIn("3.00", s)


# ── Test 6: Trajectory logger ─────────────────────────────────────────────────

class TestTrajectoryLogger(unittest.TestCase):

    def test_logger_creates_files(self):
        from sceneagent.agent.schemas import ToolAction, ToolResult
        from sceneagent.agent.trajectory_logger import TrajectoryLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrajectoryLogger(
                log_root=tmpdir,
                episode_id="test123",
                task="Test task",
                config={"model": "test", "max_steps": 5},
            )

            action = ToolAction(tool="observe", arguments={}, rationale="test")
            result = ToolResult(
                ok=True, tool="observe", summary="test obs",
                data={}, image_paths=[],
            )

            step = logger.log_step(
                action=action,
                result=result,
                current_obs={"summary": "test"},
                prompt_text="test prompt",
                raw_gemini_text='{"tool": "observe"}',
            )

            final_path = logger.save_final_result(
                success=True, answer=None, total_steps=1)
            logger.close()

            # Verify files
            self.assertTrue(os.path.exists(logger.episode_dir))
            traj_path = os.path.join(logger.episode_dir, "trajectory.jsonl")
            self.assertTrue(os.path.exists(traj_path))
            self.assertTrue(os.path.exists(final_path))

            with open(traj_path) as f:
                step_data = json.loads(f.readline())
            self.assertEqual(step_data["task"], "Test task")
            self.assertEqual(step_data["step_idx"], 0)


# ── Test 7: EpisodicMemory ───────────────────────────────────────────────────

class TestEpisodicMemory(unittest.TestCase):

    def test_add_and_get_entry(self):
        from sceneagent.agent.schemas import (
            MemoryCandidate,
        )
        from sceneagent.agent.episodic_memory import EpisodicMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            mem = EpisodicMemory(memory_dir=tmpdir)

            entry = mem.create_entry(
                memory_id="mem_000007",
                image_path="/tmp/007.png",
                robot_pose=[1.0, 2.0, 0.5],
            )
            mem.add_entry(entry)
            self.assertEqual(len(mem), 1)

            retrieved = mem.get_entry("mem_000007")
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved.memory_id, "mem_000007")
            self.assertEqual(retrieved.sensor.image_path, "/tmp/007.png")
            self.assertEqual(retrieved.sensor.robot_pose, [1.0, 2.0, 0.5])

    def test_get_pose(self):
        from sceneagent.agent.episodic_memory import EpisodicMemory
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = EpisodicMemory(memory_dir=tmpdir)
            entry = mem.create_entry(
                memory_id="mem_000010",
                image_path="/tmp/010.png",
                robot_pose=[3.5, -1.2, 1.57],
            )
            mem.add_entry(entry)
            result = mem.get_pose("mem_000010")
            self.assertIsNotNone(result)
            xy, yaw = result
            self.assertAlmostEqual(xy[0], 3.5)
            self.assertAlmostEqual(xy[1], -1.2)
            self.assertAlmostEqual(yaw, 1.57)

    def test_get_pose_missing(self):
        from sceneagent.agent.episodic_memory import EpisodicMemory
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = EpisodicMemory(memory_dir=tmpdir)
            self.assertIsNone(mem.get_pose("mem_999999"))

    def test_memory_id_navigation_prefers_robot_xy_file(self):
        """_pose_from_memory_id must prefer the on-disk scan pose over EpisodicMemory.

        The two sources are seeded with *deliberately different* values here (the entry's
        robot_pose has y and yaw scrambled relative to the file), so the assertions can only
        pass if the file won. That precedence matters: the robot_xy sidecar is the
        authoritative pose recorded at capture time.

        DummyToolbox exists solely to make BaseToolbox instantiable — every abstract
        primitive is stubbed, because this test touches none of them.
        """
        from sceneagent.agent.episodic_memory import EpisodicMemory
        from sceneagent.agent.toolbox_base import BaseToolbox

        class DummyToolbox(BaseToolbox):
            def _step(self): return {}
            def _capture_rgb(self, obs): return None
            def _get_robot_pose(self): return [0.0, 0.0, 0.0]
            def _navigate_step(self, bearing): pass
            def _base_move_step(self, motion): pass
            def _plan_path(self, start_xy, goal_xy): return [start_xy, goal_xy]
            def _grasp(self, target): return False, target, 0.0
            def _release(self, target="", destination=None, target_region=None):
                return True, "released"
            def _forward_step(self): pass
            def _get_depth_and_intrinsics(self, obs): return None

        with tempfile.TemporaryDirectory() as tmpdir:
            cap = os.path.join(tmpdir, "captures")
            os.makedirs(os.path.join(cap, "robot_xy"))
            with open(os.path.join(cap, "robot_xy", "000065.txt"), "w") as f:
                f.write("-2.8894329 -0.9700761 0.0125347\n")

            mem = EpisodicMemory(memory_dir=os.path.join(tmpdir, "memory"))
            mem.add_entry(mem.create_entry(
                "mem_000065",
                "/tmp/000065.png",
                [-2.8894329, 0.0, -0.9700761],
            ))
            tb = DummyToolbox(
                gemini_client=None,
                log_dir=os.path.join(tmpdir, "log"),
                capture_out_dir=cap,
                episodic_memory=mem,
            )
            xy, yaw = tb._pose_from_memory_id("mem_000065")
            self.assertAlmostEqual(float(xy[0]), -2.8894329)
            self.assertAlmostEqual(float(xy[1]), -0.9700761)
            self.assertAlmostEqual(float(yaw), 0.0125347)

    def test_index_persists(self):
        """Index survives reload from disk.

        Opening a second EpisodicMemory on the same directory simulates a fresh process:
        it must rebuild its id→line map from memory_index.json and still resolve entries.
        """
        from sceneagent.agent.episodic_memory import EpisodicMemory
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = EpisodicMemory(memory_dir=tmpdir)
            entry = mem.create_entry("mem_000001", "/tmp/1.png", [0.0, 0.0, 0.0])
            mem.add_entry(entry)

            mem2 = EpisodicMemory(memory_dir=tmpdir)
            self.assertEqual(len(mem2), 1)
            e = mem2.get_entry("mem_000001")
            self.assertIsNotNone(e)

    def test_enrich_candidates_fills_timestamp(self):
        from sceneagent.agent.schemas import MemoryCandidate
        from sceneagent.agent.episodic_memory import EpisodicMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            mem = EpisodicMemory(memory_dir=tmpdir)
            entry = mem.create_entry("mem_000005", "/tmp/5.png", [0.0, 0.0, 0.0],
                                     timestamp="10:00:00")
            mem.add_entry(entry)

            candidates = [
                MemoryCandidate(memory_id="mem_000005", image_path="/tmp/5.png",
                                robot_pose=[0.0, 0.0, 0.0], retrieval_score=0.9),
                MemoryCandidate(memory_id="mem_000099", image_path="/tmp/99.png",
                                robot_pose=[1.0, 1.0, 0.0], retrieval_score=0.5),
            ]
            mem.enrich_candidates(candidates)

            # The known id gets its timestamp joined in; the unknown one is left alone
            # rather than failing the whole enrichment pass.
            self.assertEqual(candidates[0].timestamp, "10:00:00")
            self.assertIsNone(candidates[1].timestamp)

    def test_memory_entry_round_trip(self):
        """MemoryEntry.to_dict() → from_dict() is lossless."""
        from sceneagent.agent.schemas import (
            EmbeddingRefs, MemoryEntry, MemorySource, SensorData,
        )
        entry = MemoryEntry(
            memory_id="mem_000042",
            sensor=SensorData(
                image_path="/tmp/042.png",
                robot_pose=[1.0, 2.0, 0.78],
                timestamp="12:34:56",
            ),
            embeddings=EmbeddingRefs(embedding_model="siglip_base"),
            source=MemorySource(source_type="scan_wasd"),
        )
        d = entry.to_dict()
        restored = MemoryEntry.from_dict(d)
        self.assertEqual(restored.memory_id, "mem_000042")
        self.assertEqual(restored.sensor.image_path, "/tmp/042.png")
        self.assertEqual(restored.sensor.robot_pose, [1.0, 2.0, 0.78])
        self.assertEqual(restored.source.source_type, "scan_wasd")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Hand-rolled runner rather than plain `unittest.main()`, so that the one test which
    # costs money and needs the network (the live Gemini call) is opt-in behind a flag and
    # never runs by accident in CI or during a routine `python -m ...` invocation.
    import argparse

    ap = argparse.ArgumentParser(description="Agent pipeline smoke tests")
    ap.add_argument("--real_gemini", action="store_true",
                    help="Run Test 5 against the real Gemini API (requires GOOGLE_API_KEY)")
    ap.add_argument("--api_key", default=None,
                    help="Gemini API key (or set GOOGLE_API_KEY env var)")
    test_args, remaining = ap.parse_known_args()

    print("Running agent pipeline smoke tests …")
    print("Tests 1-6: offline (no simulator, no Gemini)")

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestJSONParsing,
        TestSchemas,
        TestVerifier,
        TestPromptLoopMock,
        TestMemoryCandidateSchema,
        TestTrajectoryLogger,
        TestEpisodicMemory,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Opt-in live test: confirms the API key works and that a real round-trip comes back as
    # parseable JSON. Kept outside the suite (and its failures kept out of the exit code)
    # because a network blip should not turn the offline suite red.
    if test_args.real_gemini:
        key = test_args.api_key or os.environ.get("GOOGLE_API_KEY", "")
        if not key:
            print("\n[Test 5] SKIP: no API key provided (--api_key or GOOGLE_API_KEY)")
        else:
            print("\n[Test 5] Real Gemini smoke test …")
            try:
                from sceneagent.agent.gemini_client import GeminiClient
                with tempfile.TemporaryDirectory() as td:
                    client = GeminiClient(api_key=key, log_dir=td)
                    resp = client.generate_json(
                        'Respond with exactly: {"status": "ok", "test": true}'
                    )
                    assert resp.get("status") == "ok", f"Unexpected: {resp}"
                    print(f"[Test 5] PASSED — response: {resp}")
            except Exception as e:
                print(f"[Test 5] FAILED: {e}")

    sys.exit(0 if result.wasSuccessful() else 1)
