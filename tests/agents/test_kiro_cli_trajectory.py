"""Tests for kiro-cli ATIF trajectory extraction from stream-json output.

kiro-cli headless runs do not persist a conversation to SQLite, so the ATIF
trajectory is built from ``--output-format stream-json`` (JSON Lines). These
tests exercise the parser against a real captured fixture and against
hand-written event sequences covering edge cases.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aws_bench.agents.kiro_cli import _STREAM_JSON_FILENAME, KiroCli

FIXTURE = Path(__file__).parent / "fixtures" / "kiro_stream_json_rich.jsonl"


@pytest.fixture
def logs_dir(tmp_path: Path) -> Path:
    d = tmp_path / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def agent(logs_dir: Path) -> KiroCli:
    return KiroCli(logs_dir=logs_dir, model_name="claude-opus-5")


def _write_stream(logs_dir: Path, events: list[dict]) -> Path:
    path = logs_dir / _STREAM_JSON_FILENAME
    with open(path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    return path


def _su(update: dict) -> dict:
    return {"type": "sessionUpdate", "data": {"sessionId": "sess-1", "update": update}}


class TestSupportsAtif:
    def test_supports_atif_flag(self):
        assert KiroCli.SUPPORTS_ATIF is True


class TestReadStreamJson:
    def test_missing_file_returns_empty(self, agent: KiroCli, logs_dir: Path):
        assert agent._read_stream_json(logs_dir / "nope.jsonl") == []

    def test_skips_malformed_lines(self, agent: KiroCli, logs_dir: Path):
        path = logs_dir / _STREAM_JSON_FILENAME
        lines = [
            '{"type":"runStarted","data":{"engine":"v2"}}',
            "NOT JSON",
            '{"type":"runFinished","data":{}}',
        ]
        path.write_text("\n".join(lines) + "\n")
        events = agent._read_stream_json(path)
        assert [e["type"] for e in events] == ["runStarted", "runFinished"]


class TestStreamJsonToTrajectory:
    def test_real_fixture_produces_tool_steps_and_cost(self, agent: KiroCli):
        events = agent._read_stream_json(FIXTURE)
        assert events, "fixture should parse"
        traj = agent._stream_json_to_trajectory(events)
        assert traj is not None
        # The rich fixture ran three shell tools (echo one / ls bad / echo three).
        tool_steps = [s for s in traj.steps if s.tool_calls]
        assert len(tool_steps) == 3
        all_tool_calls = [tc for s in tool_steps if s.tool_calls for tc in s.tool_calls]
        # Function name comes from _meta.kiro.toolName.
        assert all(tc.function_name == "shell" for tc in all_tool_calls)
        # __tool_use_purpose is lifted out of arguments into the step message.
        assert any("Step 1" in (s.message or "") for s in tool_steps)
        assert all("__tool_use_purpose" not in tc.arguments for tc in all_tool_calls)
        # Final assistant message step present.
        assert traj.steps[-1].tool_calls is None
        assert traj.steps[-1].message
        # Model pinned.
        assert traj.agent.model_name == "claude-opus-5"
        assert all(s.model_name == "claude-opus-5" for s in tool_steps)
        # Cost from meteringUsage credits.
        metrics = traj.final_metrics
        assert metrics is not None
        assert metrics.total_cost_usd is not None
        assert metrics.total_cost_usd > 0
        # The middle tool (ls of a missing dir) is recorded as an error.
        assert metrics.extra is not None
        assert metrics.extra["total_tool_calls"] == 3
        assert metrics.extra["total_tool_calls_errors"] == 1

    def test_observation_captured_from_raw_output(self, agent: KiroCli, logs_dir: Path):
        events = [
            {"type": "runStarted", "data": {"engine": "v2"}},
            _su(
                {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "t1",
                    "title": "Running: echo hi",
                    "rawInput": {"command": "echo hi", "__tool_use_purpose": "say hi"},
                    "_meta": {"kiro": {"toolName": "shell"}},
                }
            ),
            _su(
                {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "t1",
                    "status": "completed",
                    "rawOutput": {
                        "items": [
                            {
                                "Json": {
                                    "exit_status": "exit status: 0",
                                    "stdout": "hi\n",
                                    "stderr": "",
                                }
                            }
                        ]
                    },
                }
            ),
            _su(
                {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "done"},
                }
            ),
            {"type": "runFinished", "data": {"status": "success", "finalText": "done"}},
        ]
        _write_stream(logs_dir, events)
        traj = agent._stream_json_to_trajectory(
            agent._read_stream_json(logs_dir / _STREAM_JSON_FILENAME)
        )
        assert traj is not None
        tool_step = traj.steps[0]
        assert tool_step.observation is not None
        assert "hi" in (tool_step.observation.results[0].content or "")
        assert tool_step.observation.results[0].source_call_id == "t1"

    def test_error_exit_status_flagged(self, agent: KiroCli, logs_dir: Path):
        events = [
            _su(
                {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "t1",
                    "rawInput": {"command": "ls /nope"},
                    "_meta": {"kiro": {"toolName": "shell"}},
                }
            ),
            _su(
                {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "t1",
                    "status": "completed",
                    "rawOutput": {
                        "items": [
                            {
                                "Json": {
                                    "exit_status": "exit status: 2",
                                    "stdout": "",
                                    "stderr": "no",
                                }
                            }
                        ]
                    },
                }
            ),
        ]
        _write_stream(logs_dir, events)
        traj = agent._stream_json_to_trajectory(
            agent._read_stream_json(logs_dir / _STREAM_JSON_FILENAME)
        )
        assert traj is not None
        assert traj.final_metrics is not None
        assert traj.final_metrics.extra is not None
        assert traj.final_metrics.extra["total_tool_calls_errors"] == 1

    def test_agent_message_only_produces_single_step(self, agent: KiroCli, logs_dir: Path):
        events = [
            {"type": "runStarted", "data": {"engine": "v2"}},
            _su(
                {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "Hello "},
                }
            ),
            _su(
                {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "world"},
                }
            ),
            {"type": "runFinished", "data": {"status": "success", "finalText": "Hello world"}},
        ]
        _write_stream(logs_dir, events)
        traj = agent._stream_json_to_trajectory(
            agent._read_stream_json(logs_dir / _STREAM_JSON_FILENAME)
        )
        assert traj is not None
        assert len(traj.steps) == 1
        assert traj.steps[0].message == "Hello world"
        assert traj.steps[0].tool_calls is None

    def test_empty_events_returns_none(self, agent: KiroCli):
        assert agent._stream_json_to_trajectory([]) is None

    def test_no_cost_when_no_metering(self, agent: KiroCli, logs_dir: Path):
        events = [
            _su({"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "x"}}),
        ]
        _write_stream(logs_dir, events)
        traj = agent._stream_json_to_trajectory(
            agent._read_stream_json(logs_dir / _STREAM_JSON_FILENAME)
        )
        assert traj is not None
        assert traj.final_metrics is not None
        assert traj.final_metrics.total_cost_usd is None

    def test_sequential_step_ids(self, agent: KiroCli):
        events = agent._read_stream_json(FIXTURE)
        traj = agent._stream_json_to_trajectory(events)
        assert traj is not None
        assert [s.step_id for s in traj.steps] == list(range(1, len(traj.steps) + 1))


class TestPopulateContextPostRun:
    def test_no_stream_file_is_noop(self, agent: KiroCli):
        context = MagicMock()
        context.cost_usd = None
        agent.populate_context_post_run(context)
        assert not (agent.logs_dir / "trajectory.json").exists()

    def test_writes_trajectory_json_from_fixture(self, agent: KiroCli, logs_dir: Path):
        # Copy the real fixture into the expected stream path.
        (logs_dir / _STREAM_JSON_FILENAME).write_text(FIXTURE.read_text())
        context = MagicMock()
        context.cost_usd = None
        agent.populate_context_post_run(context)
        out = logs_dir / "trajectory.json"
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["agent"]["model_name"] == "claude-opus-5"
        assert data["steps"]
        # cost propagated to context
        assert context.cost_usd is not None and context.cost_usd > 0
