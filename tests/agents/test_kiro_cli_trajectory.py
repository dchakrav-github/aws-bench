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


class TestInterleavedReasoning:
    """Reasoning attaches to the tool step it precedes.

    The final answer is a separate last step (not glued to earlier reasoning).
    """

    def _events_two_rounds(self) -> list[dict]:
        # reasoning A -> tool t1 ; reasoning B -> tool t2 ; final answer C
        return [
            {"type": "runStarted", "data": {"engine": "v2"}},
            _su(
                {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "First "},
                }
            ),
            _su(
                {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "I list regions."},
                }
            ),
            _su(
                {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "t1",
                    "title": "Running: aws ec2 describe-regions",
                    "rawInput": {"command": "aws ec2 describe-regions"},
                    "_meta": {"kiro": {"toolName": "shell"}},
                }
            ),
            _su(
                {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "t1",
                    "status": "completed",
                    "rawOutput": {
                        "items": [{"Json": {"exit_status": "exit status: 0", "stdout": "ok"}}]
                    },
                }
            ),
            _su(
                {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "Now the "},
                }
            ),
            _su(
                {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "compact view."},
                }
            ),
            _su(
                {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "t2",
                    "title": "Running: aws ec2 describe-instances",
                    "rawInput": {"command": "aws ec2 describe-instances"},
                    "_meta": {"kiro": {"toolName": "shell"}},
                }
            ),
            _su(
                {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "t2",
                    "status": "completed",
                    "rawOutput": {
                        "items": [{"Json": {"exit_status": "exit status: 0", "stdout": "ok"}}]
                    },
                }
            ),
            _su(
                {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "Done: "},
                }
            ),
            _su(
                {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "2 instances."},
                }
            ),
            {
                "type": "runFinished",
                "data": {"status": "success", "finalText": "Done: 2 instances."},
            },
        ]

    def test_reasoning_attaches_to_next_tool_step(self, agent: KiroCli, logs_dir: Path):
        _write_stream(logs_dir, self._events_two_rounds())
        traj = agent._stream_json_to_trajectory(
            agent._read_stream_json(logs_dir / _STREAM_JSON_FILENAME)
        )
        assert traj is not None
        assert len(traj.steps) == 3  # two tool steps + one final answer
        # Pre-call reasoning lands on the tool step it preceded.
        assert traj.steps[0].message == "First I list regions."
        assert traj.steps[0].tool_calls is not None
        assert traj.steps[1].message == "Now the compact view."
        assert traj.steps[1].tool_calls is not None

    def test_final_answer_is_separate_last_step(self, agent: KiroCli, logs_dir: Path):
        _write_stream(logs_dir, self._events_two_rounds())
        traj = agent._stream_json_to_trajectory(
            agent._read_stream_json(logs_dir / _STREAM_JSON_FILENAME)
        )
        assert traj is not None
        last = traj.steps[-1]
        assert last.tool_calls is None
        assert last.message == "Done: 2 instances."
        # The earlier reasoning is NOT glued into the final answer.
        assert "list regions" not in (last.message or "")
        assert "compact view" not in (last.message or "")

    def test_reasoning_falls_back_to_purpose_then_title(self, agent: KiroCli, logs_dir: Path):
        # No agent_message_chunk before the tool -> use __tool_use_purpose.
        events = [
            _su(
                {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "t1",
                    "title": "Running: echo hi",
                    "rawInput": {"command": "echo hi", "__tool_use_purpose": "greet"},
                    "_meta": {"kiro": {"toolName": "shell"}},
                }
            ),
        ]
        _write_stream(logs_dir, events)
        traj = agent._stream_json_to_trajectory(
            agent._read_stream_json(logs_dir / _STREAM_JSON_FILENAME)
        )
        assert traj is not None
        assert traj.steps[0].message == "greet"


class TestThoughtCapture:
    """agent_thought_chunk reasoning is captured in Step.reasoning_content.

    It attaches to the same step boundary as message text, but stays separate
    from the user-facing ``message``.
    """

    def test_thought_before_tool_sets_reasoning_content(self, agent: KiroCli, logs_dir: Path):
        events = [
            _su(
                {
                    "sessionUpdate": "agent_thought_chunk",
                    "content": {"type": "text", "text": "I need to "},
                }
            ),
            _su(
                {
                    "sessionUpdate": "agent_thought_chunk",
                    "content": {"type": "text", "text": "check route tables."},
                }
            ),
            _su(
                {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "Checking routes."},
                }
            ),
            _su(
                {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "t1",
                    "title": "Running: aws ec2 describe-route-tables",
                    "rawInput": {"command": "aws ec2 describe-route-tables"},
                    "_meta": {"kiro": {"toolName": "use_aws"}},
                }
            ),
            _su(
                {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "t1",
                    "status": "completed",
                    "rawOutput": {
                        "items": [{"Json": {"exit_status": "exit status: 0", "stdout": "ok"}}]
                    },
                }
            ),
        ]
        _write_stream(logs_dir, events)
        traj = agent._stream_json_to_trajectory(
            agent._read_stream_json(logs_dir / _STREAM_JSON_FILENAME)
        )
        assert traj is not None
        step = traj.steps[0]
        # Reasoning captured in the structured field...
        assert step.reasoning_content == "I need to check route tables."
        # ...and the user-facing message is untouched (not polluted with thought).
        assert step.message == "Checking routes."
        assert "check route tables" not in (step.message or "")

    def test_trailing_thought_on_final_step(self, agent: KiroCli, logs_dir: Path):
        events = [
            _su(
                {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "t1",
                    "title": "Running: echo hi",
                    "rawInput": {"command": "echo hi"},
                    "_meta": {"kiro": {"toolName": "shell"}},
                }
            ),
            _su(
                {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": "t1",
                    "status": "completed",
                    "rawOutput": {
                        "items": [{"Json": {"exit_status": "exit status: 0", "stdout": "hi"}}]
                    },
                }
            ),
            _su(
                {
                    "sessionUpdate": "agent_thought_chunk",
                    "content": {"type": "text", "text": "All done "},
                }
            ),
            _su(
                {
                    "sessionUpdate": "agent_thought_chunk",
                    "content": {"type": "text", "text": "thinking."},
                }
            ),
            _su(
                {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "Result: hi"},
                }
            ),
            {"type": "runFinished", "data": {"status": "success", "finalText": "Result: hi"}},
        ]
        _write_stream(logs_dir, events)
        traj = agent._stream_json_to_trajectory(
            agent._read_stream_json(logs_dir / _STREAM_JSON_FILENAME)
        )
        assert traj is not None
        last = traj.steps[-1]
        assert last.tool_calls is None
        assert last.message == "Result: hi"
        assert last.reasoning_content == "All done thinking."

    def test_no_thought_leaves_reasoning_content_none(self, agent: KiroCli, logs_dir: Path):
        events = [
            _su(
                {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "t1",
                    "title": "Running: echo hi",
                    "rawInput": {"command": "echo hi"},
                    "_meta": {"kiro": {"toolName": "shell"}},
                }
            ),
        ]
        _write_stream(logs_dir, events)
        traj = agent._stream_json_to_trajectory(
            agent._read_stream_json(logs_dir / _STREAM_JSON_FILENAME)
        )
        assert traj is not None
        assert traj.steps[0].reasoning_content is None

    def test_real_fixture_populates_reasoning(self, agent: KiroCli):
        # The rich fixture has a thought run before the tool batch.
        traj = agent._stream_json_to_trajectory(agent._read_stream_json(FIXTURE))
        assert traj is not None
        assert any(s.reasoning_content for s in traj.steps)


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
