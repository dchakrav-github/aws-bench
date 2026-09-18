"""Tests for the Kiro CLI agent."""

import sqlite3
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aws_bench.agents.kiro_cli import (
    KiroCli,
    KiroSessionError,
    _build_stripped_session_db,
)
from aws_bench.cli.preflight import PreflightError


def _make_host_session_db(path: Path, *, with_auth: bool = True, with_history: bool = True) -> None:
    """Build a fake host kiro-cli data.sqlite3 resembling the real schema."""
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE migrations (id INTEGER PRIMARY KEY, name TEXT)")
        conn.execute("INSERT INTO migrations VALUES (1, 'init')")
        conn.execute("CREATE TABLE state (key TEXT, value TEXT)")
        conn.execute("INSERT INTO state VALUES ('k', 'v')")
        if with_auth:
            conn.execute("CREATE TABLE auth_kv (key TEXT PRIMARY KEY, value BLOB)")
            conn.execute("INSERT INTO auth_kv VALUES ('token', 'secret-session')")
        if with_history:
            # Sensitive host tables that must NOT be copied into the trial.
            conn.execute("CREATE TABLE history (id INTEGER, prompt TEXT)")
            conn.execute("INSERT INTO history VALUES (1, 'private host prompt')")
            conn.execute("CREATE TABLE conversations_v2 (id INTEGER, value TEXT)")
            conn.execute("INSERT INTO conversations_v2 VALUES (1, 'private convo')")
            conn.execute("CREATE TABLE knowledge_bases (id INTEGER, doc TEXT)")
            conn.execute("INSERT INTO knowledge_bases VALUES (1, 'private kb')")
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def logs_dir(tmp_path: Path) -> Path:
    return tmp_path / "logs"


@pytest.fixture
def agent(logs_dir: Path) -> KiroCli:
    return KiroCli(logs_dir=logs_dir)


class TestKiroCliName:
    def test_name(self):
        assert KiroCli.name() == "kiro-cli"


class TestKiroCliInit:
    def test_default_init(self, agent: KiroCli):
        assert agent.model_name is None

    def test_init_with_model(self, logs_dir: Path):
        agent = KiroCli(logs_dir=logs_dir, model_name="auto")
        assert agent.model_name == "auto"


class TestKiroCliVersionCommand:
    def test_get_version_command(self, agent: KiroCli):
        cmd = agent.get_version_command()
        assert cmd is not None
        assert "kiro-cli" in cmd
        assert "--version" in cmd


class TestKiroCliInstall:
    @pytest.mark.asyncio
    async def test_install_calls_exec(self, agent: KiroCli):
        environment = MagicMock()
        environment.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))

        await agent.install(environment)

        assert environment.exec.call_count >= 2

    @pytest.mark.asyncio
    async def test_install_disables_greeting(self, agent: KiroCli):
        environment = MagicMock()
        environment.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))

        await agent.install(environment)

        calls = environment.exec.call_args_list
        install_call = calls[1]
        command = install_call.kwargs.get("command", "")
        assert "chat.greeting.enabled false" in command


class TestKiroCliSetup:
    @pytest.mark.asyncio
    async def test_setup_raises_when_no_api_key_and_no_session_mount(self, agent: KiroCli):
        environment = MagicMock()

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(PreflightError, match="KIRO_API_KEY"):
                await agent.setup(environment)

        environment.exec.assert_not_called()

    @pytest.mark.asyncio
    async def test_setup_session_mount_missing_db_raises(self, agent: KiroCli, tmp_path: Path):
        environment = MagicMock()
        missing = tmp_path / "nope" / "data.sqlite3"

        with patch.dict(
            "os.environ",
            {"AWS_BENCH_KIRO_SESSION_MOUNT": "1", "KIRO_CLI_SESSION_DB": str(missing)},
            clear=True,
        ):
            with pytest.raises(PreflightError, match="no host Kiro session DB"):
                await agent.setup(environment)

        environment.exec.assert_not_called()

    @pytest.mark.asyncio
    async def test_setup_session_mount_db_without_auth_raises(self, agent: KiroCli, tmp_path: Path):
        environment = MagicMock()
        db = tmp_path / "data.sqlite3"
        _make_host_session_db(db, with_auth=False)

        with patch.dict(
            "os.environ",
            {"AWS_BENCH_KIRO_SESSION_MOUNT": "1", "KIRO_CLI_SESSION_DB": str(db)},
            clear=True,
        ):
            with pytest.raises(PreflightError, match="Could not extract a Kiro login session"):
                await agent.setup(environment)

    @pytest.mark.asyncio
    async def test_setup_session_mount_valid_session_passes_preflight(
        self, agent: KiroCli, tmp_path: Path
    ):
        environment = MagicMock()
        environment.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))
        db = tmp_path / "data.sqlite3"
        _make_host_session_db(db)

        with patch.dict(
            "os.environ",
            {"AWS_BENCH_KIRO_SESSION_MOUNT": "1", "KIRO_CLI_SESSION_DB": str(db)},
            clear=True,
        ):
            # install() runs after preflight passes; stub it to isolate setup().
            with patch.object(agent, "install", new=AsyncMock()):
                await agent.setup(environment)


class TestStrippedSessionDb:
    def test_copies_only_auth_tables(self, tmp_path: Path):
        host = tmp_path / "host.sqlite3"
        dest = tmp_path / "stripped.sqlite3"
        _make_host_session_db(host)

        _build_stripped_session_db(host, dest)

        conn = sqlite3.connect(dest)
        try:
            tables = {
                r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
        finally:
            conn.close()
        assert tables == {"auth_kv", "state", "migrations"}
        # Sensitive host tables must be absent from the stripped copy.
        assert "history" not in tables
        assert "conversations_v2" not in tables
        assert "knowledge_bases" not in tables

    def test_preserves_auth_rows(self, tmp_path: Path):
        host = tmp_path / "host.sqlite3"
        dest = tmp_path / "stripped.sqlite3"
        _make_host_session_db(host)

        _build_stripped_session_db(host, dest)

        conn = sqlite3.connect(dest)
        try:
            value = conn.execute("SELECT value FROM auth_kv WHERE key='token'").fetchone()
        finally:
            conn.close()
        assert value is not None and value[0] == "secret-session"

    def test_raises_without_auth_table(self, tmp_path: Path):
        host = tmp_path / "host.sqlite3"
        dest = tmp_path / "stripped.sqlite3"
        _make_host_session_db(host, with_auth=False)

        with pytest.raises(KiroSessionError, match="auth_kv"):
            _build_stripped_session_db(host, dest)

    def test_raises_when_host_db_missing(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            _build_stripped_session_db(tmp_path / "absent.sqlite3", tmp_path / "out.sqlite3")


class TestKiroCliRun:
    @staticmethod
    def _run_cmds(calls):
        return [
            c.kwargs.get("command", "")
            for c in calls
            if "kiro-cli chat" in c.kwargs.get("command", "")
        ]

    @pytest.mark.asyncio
    async def test_run_basic(self, agent: KiroCli):
        environment = MagicMock()
        environment.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))
        context = MagicMock()

        with patch.dict("os.environ", {"KIRO_API_KEY": "ksk_test"}, clear=True):
            await agent.run("Do the task", environment, context)

        calls = environment.exec.call_args_list
        run_cmds = self._run_cmds(calls)
        assert run_cmds
        command = run_cmds[0]
        assert "--no-interactive" in command
        assert "--trust-all-tools" in command
        assert "Do the task" in command
        assert "/logs/agent/kiro-cli.txt" in command

    @pytest.mark.asyncio
    async def test_run_passes_kiro_api_key(self, agent: KiroCli):
        environment = MagicMock()
        environment.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))
        context = MagicMock()

        with patch.dict("os.environ", {"KIRO_API_KEY": "ksk_test123"}, clear=True):
            await agent.run("Do the task", environment, context)

        calls = environment.exec.call_args_list
        # The main run call should have the env
        run_call = [c for c in calls if "kiro-cli chat" in c.kwargs.get("command", "")][0]
        env = run_call.kwargs.get("env", {})
        assert env.get("KIRO_API_KEY") == "ksk_test123"

    @pytest.mark.asyncio
    async def test_run_with_model_flag(self, logs_dir: Path):
        agent = KiroCli(logs_dir=logs_dir, model_name="sonnet-4")
        environment = MagicMock()
        environment.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))
        context = MagicMock()

        with patch.dict("os.environ", {"KIRO_API_KEY": "ksk_test"}, clear=True):
            await agent.run("Do the task", environment, context)

        run_cmds = self._run_cmds(environment.exec.call_args_list)
        assert "--model" in run_cmds[0]
        assert "sonnet-4" in run_cmds[0]

    @pytest.mark.asyncio
    async def test_run_with_effort(self, logs_dir: Path):
        agent = KiroCli(logs_dir=logs_dir, effort="high")
        environment = MagicMock()
        environment.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))
        context = MagicMock()

        with patch.dict("os.environ", {"KIRO_API_KEY": "ksk_test"}, clear=True):
            await agent.run("Do the task", environment, context)

        run_cmds = self._run_cmds(environment.exec.call_args_list)
        assert "--effort high" in run_cmds[0]

    @pytest.mark.asyncio
    async def test_run_with_mcp_servers(self, logs_dir: Path):
        agent = KiroCli(logs_dir=logs_dir)
        mock_server = MagicMock()
        mock_server.name = "test-server"
        mock_server.transport = "stdio"
        mock_server.command = "node"
        mock_server.args = ["server.js"]
        agent.mcp_servers = cast(list, [mock_server])

        environment = MagicMock()
        environment.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))
        context = MagicMock()

        with patch.dict("os.environ", {"KIRO_API_KEY": "ksk_test"}, clear=True):
            await agent.run("Do the task", environment, context)

        calls = environment.exec.call_args_list
        mcp_call = calls[0]
        mcp_command = mcp_call.kwargs.get("command", "")
        assert "mcp.json" in mcp_command
        assert "test-server" in mcp_command

    @pytest.mark.asyncio
    async def test_run_with_skills_dir(self, logs_dir: Path):
        agent = KiroCli(logs_dir=logs_dir, skills_dir="/harbor/skills")
        environment = MagicMock()
        environment.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))
        context = MagicMock()

        with patch.dict("os.environ", {"KIRO_API_KEY": "ksk_test"}, clear=True):
            await agent.run("Do the task", environment, context)

        calls = environment.exec.call_args_list
        skills_cmds = [
            c.kwargs.get("command", "") for c in calls if "cp -r" in c.kwargs.get("command", "")
        ]
        assert skills_cmds
        assert "/harbor/skills" in skills_cmds[0]
        assert "~/.kiro/skills/" in skills_cmds[0]

    @pytest.mark.asyncio
    async def test_session_mount_injects_stripped_db_and_clean_config(
        self, logs_dir: Path, tmp_path: Path
    ):
        agent = KiroCli(logs_dir=logs_dir)
        db = tmp_path / "data.sqlite3"
        _make_host_session_db(db)
        environment = MagicMock()
        environment.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))
        context = MagicMock()

        with patch.dict(
            "os.environ",
            {"AWS_BENCH_KIRO_SESSION_MOUNT": "1", "KIRO_CLI_SESSION_DB": str(db)},
            clear=True,
        ):
            await agent.run("Do the task", environment, context)

        commands = [c.kwargs.get("command", "") for c in environment.exec.call_args_list]
        mount_cmds = [c for c in commands if "base64 -d" in c]
        assert mount_cmds, "expected a session-mount command that base64-decodes the DB"
        mount_cmd = mount_cmds[0]
        assert "~/.local/share/kiro-cli/data.sqlite3" in mount_cmd
        # Clean config: empty cli.json so no host defaultAgent is inherited.
        assert "~/.kiro/settings/cli.json" in mount_cmd
        # The run command must not forward a KIRO_API_KEY in session-mount mode.
        run_call = [
            c
            for c in environment.exec.call_args_list
            if "kiro-cli chat" in c.kwargs.get("command", "")
        ][0]
        assert "KIRO_API_KEY" not in (run_call.kwargs.get("env") or {})

    @pytest.mark.asyncio
    async def test_session_mount_skips_host_skill_staging(self, logs_dir: Path, tmp_path: Path):
        agent = KiroCli(logs_dir=logs_dir, skills_dir="/harbor/skills")
        db = tmp_path / "data.sqlite3"
        _make_host_session_db(db)
        environment = MagicMock()
        environment.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))
        context = MagicMock()

        with patch.dict(
            "os.environ",
            {"AWS_BENCH_KIRO_SESSION_MOUNT": "1", "KIRO_CLI_SESSION_DB": str(db)},
            clear=True,
        ):
            await agent.run("Do the task", environment, context)

        commands = [c.kwargs.get("command", "") for c in environment.exec.call_args_list]
        # No host skills copied in session-mount mode, even though skills_dir is set.
        assert not any("cp -r" in c and "~/.kiro/skills/" in c for c in commands)

    @pytest.mark.asyncio
    async def test_api_key_takes_precedence_over_session_mount(
        self, logs_dir: Path, tmp_path: Path
    ):
        agent = KiroCli(logs_dir=logs_dir)
        db = tmp_path / "data.sqlite3"
        _make_host_session_db(db)
        environment = MagicMock()
        environment.exec = AsyncMock(return_value=MagicMock(return_code=0, stdout="", stderr=""))
        context = MagicMock()

        with patch.dict(
            "os.environ",
            {
                "KIRO_API_KEY": "ksk_real",
                "AWS_BENCH_KIRO_SESSION_MOUNT": "1",
                "KIRO_CLI_SESSION_DB": str(db),
            },
            clear=True,
        ):
            await agent.run("Do the task", environment, context)

        commands = [c.kwargs.get("command", "") for c in environment.exec.call_args_list]
        # With an API key present, no session DB is mounted.
        assert not any("base64 -d" in c for c in commands)
        run_call = [
            c
            for c in environment.exec.call_args_list
            if "kiro-cli chat" in c.kwargs.get("command", "")
        ][0]
        assert (run_call.kwargs.get("env") or {}).get("KIRO_API_KEY") == "ksk_real"


class TestKiroCliBuildMcpJson:
    def test_returns_none_for_empty(self):
        assert KiroCli._build_mcp_json([]) is None

    def test_builds_stdio_server(self):
        server = MagicMock()
        server.name = "my-server"
        server.transport = "stdio"
        server.command = "node"
        server.args = ["index.js"]

        result = KiroCli._build_mcp_json([server])
        assert result == {"my-server": {"command": "node", "args": ["index.js"]}}

    def test_builds_http_server(self):
        server = MagicMock()
        server.name = "remote"
        server.transport = "streamable-http"
        server.url = "http://localhost:3000"

        result = KiroCli._build_mcp_json([server])
        assert result == {"remote": {"url": "http://localhost:3000"}}
