"""Kiro CLI agent for aws-bench.

Installs Kiro CLI into the trial environment, configures auth and MCP servers,
then runs it non-interactively against the task instruction.

Auth has two modes:

1. **API key (default):** ``$KIRO_API_KEY`` from the host shell is forwarded
   into the container. This is the headless credential Kiro documents for CI.
2. **Session mount (opt-in):** when ``$KIRO_API_KEY`` is unset and
   ``$AWS_BENCH_KIRO_SESSION_MOUNT`` is truthy, the agent reuses an existing
   host ``kiro-cli login`` session. Only the auth-bearing rows of the host
   ``data.sqlite3`` (the ``auth_kv``/``state``/``migrations`` tables) are
   copied into the container — never the host's conversation history,
   knowledge bases, or any ``~/.kiro`` config, steering, agents, or skills.
   This keeps enterprise/host configuration out of the trial so the agent runs
   with its built-in default system prompt only.

Trajectory data is extracted from Kiro CLI's native SQLite database after the
run completes.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shlex
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from harbor.agents.installed.base import (
    BaseInstalledAgent,
    CliFlag,
    with_prompt_template,
)
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.task.config import MCPServerConfig
from harbor.models.trajectories import (
    Agent,
    FinalMetrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)

from aws_bench.cli.preflight import PreflightError

_STREAM_JSON_FILENAME = "kiro-cli-stream.jsonl"
_STREAM_ERR_FILENAME = "kiro-cli-stream.err"
_CONTAINER_DB_PATH = "~/.local/share/kiro-cli/data.sqlite3"
# Name of the per-trial agent config written to pin the model. kiro-cli's
# `--model` flag is a no-op in headless v2 (it logs "failed to set model:
# Method not found" and silently falls back to the default `auto` model). The
# supported way to pin a model is an agent config carrying a `model` field,
# launched with `--agent`. This agent is otherwise minimal (built-in default
# behavior) so only the model is pinned.
_MODEL_PIN_AGENT = "aws-bench"
_CONTAINER_AGENT_PATH = f"~/.kiro/agents/{_MODEL_PIN_AGENT}.json"
# kiro-cli installs into $HOME/.local/bin which is not on PATH for non-login
# shells spawned by environment.exec.
_PATH_PREFIX = 'export PATH="$HOME/.local/bin:$PATH"; '

# Opt-in flag: reuse a host `kiro-cli login` session instead of KIRO_API_KEY.
_SESSION_MOUNT_ENV = "AWS_BENCH_KIRO_SESSION_MOUNT"
# Host data.sqlite3 that holds the login session (override for testing / custom
# installs). Defaults to Kiro CLI's standard per-user data path on the host.
_SESSION_DB_ENV = "KIRO_CLI_SESSION_DB"
_DEFAULT_HOST_SESSION_DB = "~/.local/share/kiro-cli/data.sqlite3"
# Optional host env pinning the kiro-cli release downloaded into the trial
# (e.g. "2.27.1"); default "latest".
_KIRO_VERSION_ENV = "AWS_BENCH_KIRO_VERSION"
# The only tables copied from the host DB. `auth_kv` carries the login session;
# `state`/`migrations` let kiro-cli open the DB at the expected schema version.
# Everything else on the host DB (history, conversations*, knowledge_bases, …)
# is deliberately excluded so no host data enters the trial.
_SESSION_AUTH_TABLES = ("auth_kv", "state", "migrations")


def _truthy(value: str | None) -> bool:
    """Return True for common truthy string flag values."""
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _build_stripped_session_db(host_db: Path, dest_db: Path) -> None:
    """Copy only the auth-bearing tables from ``host_db`` into a fresh ``dest_db``.

    Copies ``auth_kv``/``state``/``migrations`` and nothing else, so the login
    session is reused without dragging the host's conversation history or
    knowledge bases into the trial.

    Args:
        host_db: Path to the host ``data.sqlite3`` written by ``kiro-cli login``.
        dest_db: Path to write the stripped copy to (overwritten if present).

    Raises:
        FileNotFoundError: If ``host_db`` does not exist.
        KiroSessionError: If the host DB has no ``auth_kv`` table (not logged in).
    """
    if not host_db.exists():
        raise FileNotFoundError(f"Kiro session DB not found: {host_db}")
    if dest_db.exists():
        dest_db.unlink()

    src = sqlite3.connect(f"file:{host_db}?mode=ro", uri=True)
    try:
        present = {
            row[0] for row in src.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "auth_kv" not in present:
            raise KiroSessionError(
                f"Host Kiro DB {host_db} has no 'auth_kv' table; run `kiro-cli login` first."
            )
        dst = sqlite3.connect(dest_db)
        try:
            for table in _SESSION_AUTH_TABLES:
                if table not in present:
                    continue
                ddl = src.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                ).fetchone()
                if not ddl or not ddl[0]:
                    continue
                dst.execute(ddl[0])
                rows = src.execute(f"SELECT * FROM {table}").fetchall()  # noqa: S608
                if rows:
                    placeholders = ",".join("?" * len(rows[0]))
                    dst.executemany(
                        f"INSERT INTO {table} VALUES ({placeholders})",  # noqa: S608
                        rows,
                    )
            dst.commit()
        finally:
            dst.close()
    finally:
        src.close()


class KiroSessionError(Exception):
    """Raised when a host Kiro login session cannot be extracted for mounting."""


class KiroCli(BaseInstalledAgent):
    """Kiro CLI agent — runs tasks in headless mode via kiro-cli chat."""

    SUPPORTS_ATIF: bool = True

    CLI_FLAGS = [
        CliFlag(
            "effort",
            cli="--effort",
            type="enum",
            choices=["low", "medium", "high", "xhigh", "max"],
            env_fallback="KIRO_CLI_EFFORT_LEVEL",
        ),
    ]

    @staticmethod
    def name() -> str:
        """Return the agent name identifier."""
        return "kiro-cli"

    def get_version_command(self) -> str | None:
        """Return the shell command to detect the installed kiro-cli version."""
        return f"{_PATH_PREFIX}kiro-cli --version"

    def parse_version(self, stdout: str) -> str:
        """Parse semver from kiro-cli --version output."""
        import re

        text = stdout.strip()
        match = re.search(r"(\d+\.\d+\.\d+)", text)
        return match.group(1) if match else text

    @staticmethod
    def _build_mcp_json(
        servers: list[MCPServerConfig],
    ) -> dict[str, dict[str, Any]] | None:
        """Build Kiro CLI MCP server config dict from Harbor's MCPServerConfig list."""
        if not servers:
            return None
        mcp_servers: dict[str, dict[str, Any]] = {}
        for server in servers:
            if server.transport == "stdio":
                entry: dict[str, Any] = {
                    "command": server.command,
                    "args": server.args,
                }
            else:
                entry = {"url": server.url}
            mcp_servers[server.name] = entry
        return mcp_servers

    def _session_mount_enabled(self) -> bool:
        """Whether opt-in host-session mounting is requested."""
        return _truthy(os.environ.get(_SESSION_MOUNT_ENV))

    def _host_session_db(self) -> Path:
        """Resolve the host ``data.sqlite3`` that holds the login session."""
        raw = os.environ.get(_SESSION_DB_ENV) or _DEFAULT_HOST_SESSION_DB
        return Path(os.path.expanduser(raw))

    def _use_session_mount(self) -> bool:
        """True when auth should come from a mounted host session, not an API key.

        Session mount applies only when no ``KIRO_API_KEY`` is set and the opt-in
        flag is on. The API key always takes precedence when present.
        """
        return not os.environ.get("KIRO_API_KEY", "") and self._session_mount_enabled()

    def _kiro_env(self) -> dict[str, str]:
        """Collect Kiro CLI env vars from the host.

        Returns ``{"KIRO_API_KEY": ...}`` in API-key mode. In session-mount mode
        no auth env is needed (the login session is mounted into the container),
        so an empty dict is returned.
        """
        api_key = os.environ.get("KIRO_API_KEY", "")
        if api_key:
            return {"KIRO_API_KEY": api_key}
        return {}

    async def setup(self, environment: BaseEnvironment) -> None:
        """Validate auth before spending time on install.

        Accepts either ``KIRO_API_KEY`` (default) or, when
        ``AWS_BENCH_KIRO_SESSION_MOUNT`` is set and no API key is present, a host
        ``kiro-cli login`` session. Fails preflight if neither is usable.
        """
        if not os.environ.get("KIRO_API_KEY", ""):
            if not self._session_mount_enabled():
                raise PreflightError(
                    "KIRO_API_KEY is not set (or is empty), but the agent is kiro-cli. "
                    "Export it before running: export KIRO_API_KEY=ksk_xxxxxxxx\n"
                    f"Alternatively, set {_SESSION_MOUNT_ENV}=1 to reuse a host "
                    "`kiro-cli login` session."
                )
            host_db = self._host_session_db()
            if not host_db.exists():
                raise PreflightError(
                    f"{_SESSION_MOUNT_ENV} is set but no host Kiro session DB was found at "
                    f"{host_db}. Run `kiro-cli login` on the host first, or point "
                    f"{_SESSION_DB_ENV} at the correct data.sqlite3."
                )
            # Fail fast if the DB exists but carries no login session.
            probe_path = Path(tempfile.mkstemp(suffix=".sqlite3", prefix="kiro-session-probe-")[1])
            try:
                _build_stripped_session_db(host_db, probe_path)
            except (KiroSessionError, sqlite3.Error, OSError) as exc:
                raise PreflightError(
                    f"Could not extract a Kiro login session from {host_db}: {exc}"
                ) from exc
            finally:
                probe_path.unlink(missing_ok=True)
        await super().setup(environment)

    async def install(self, environment: BaseEnvironment) -> None:
        """Install kiro-cli binary in the agent environment.

        ``$AWS_BENCH_KIRO_VERSION`` (host env) pins the release, e.g. ``2.27.1``;
        default ``latest``. The 2.x installer forwards its args to
        ``kiro-cli setup`` (which takes ``--force --no-confirm``); the 3.x
        installer rejects unknown args, so those flags are passed only for 2.x.
        """
        version = (os.environ.get(_KIRO_VERSION_ENV) or "latest").strip()
        if not re.fullmatch(r"latest|\d+\.\d+\.\d+", version):
            raise PreflightError(
                f"{_KIRO_VERSION_ENV}={version!r} is invalid; use 'latest' or X.Y.Z."
            )
        setup_args = " --force --no-confirm" if version.startswith("2.") else ""
        await self.exec_as_root(
            environment,
            command=(
                "if command -v apt-get &> /dev/null; then"
                "  apt-get update && apt-get install -y curl unzip libasound2;"
                " elif command -v yum &> /dev/null; then"
                "  yum install -y curl unzip alsa-lib;"
                " elif command -v apk &> /dev/null; then"
                "  apk add --no-cache curl bash unzip alsa-lib;"
                " else"
                '  echo "Warning: no known package manager found" >&2;'
                " fi"
            ),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail; "
                "ARCH=$(uname -m); "
                "case $ARCH in x86_64) KIRO_ARCH=x86_64;; aarch64|arm64) KIRO_ARCH=aarch64;; "
                '*) echo "Unsupported architecture: $ARCH"; exit 1;; esac; '
                "curl --proto '=https' --tlsv1.2 -sSf "
                '"https://desktop-release.q.us-east-1.amazonaws.com/'
                f'{version}/kirocli-${{KIRO_ARCH}}-linux.zip" '
                "-o /tmp/kirocli.zip && "
                f"cd /tmp && unzip -q kirocli.zip && ./kirocli/install.sh{setup_args} && "
                "rm -rf /tmp/kirocli.zip /tmp/kirocli && "
                f"{_PATH_PREFIX}"
                "kiro-cli --version && "
                "mkdir -p ~/.kiro/settings && "
                "kiro-cli settings chat.greeting.enabled false"
            ),
        )

    async def _mount_host_session(self, environment: BaseEnvironment) -> None:
        """Copy a stripped host login session into the container and force clean config.

        Builds a stripped ``data.sqlite3`` (auth tables only) on the host, base64
        streams it into the container's ``~/.local/share/kiro-cli/data.sqlite3``,
        and writes an empty ``~/.kiro/settings/cli.json`` so no host default agent
        (e.g. ``chat.defaultAgent``) is inherited — the built-in default agent and
        its native system prompt are used.
        """
        host_db = self._host_session_db()
        stripped = self.logs_dir / "kiro-session-auth.sqlite3"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        _build_stripped_session_db(host_db, stripped)

        payload = base64.b64encode(stripped.read_bytes()).decode("ascii")
        container_db = _CONTAINER_DB_PATH
        container_dir = container_db.rsplit("/", 1)[0]
        # Write a guaranteed-clean cli.json (no defaultAgent) and decode the DB.
        await self.exec_as_agent(
            environment,
            command=(
                f"mkdir -p {container_dir} ~/.kiro/settings && "
                f"printf '%s' {shlex.quote(payload)} | base64 -d > {container_db} && "
                f"printf '%s' '{{}}' > ~/.kiro/settings/cli.json"
            ),
        )

    async def _write_model_pin_agent(self, environment: BaseEnvironment) -> None:
        """Write a minimal agent config that pins ``self.model_name`` and launch via ``--agent``.

        kiro-cli's ``--model`` flag is not honored in headless mode (it logs
        "failed to set model: Method not found" and silently runs the default
        ``auto`` model). The supported mechanism is an agent config with a
        ``model`` field. This writes an otherwise-minimal agent — full tool
        access, no extra prompt/skills/MCP — so only the model differs from the
        built-in default behavior. Verified that the resulting run records the
        pinned model in the session DB's ``model_info``.
        """
        agent_config = {
            "name": _MODEL_PIN_AGENT,
            "description": "aws-bench trial agent (model pinned).",
            "model": self.model_name,
            "tools": ["*"],
            "allowedTools": ["*"],
        }
        escaped = shlex.quote(json.dumps(agent_config))
        await self.exec_as_agent(
            environment,
            command=(f"mkdir -p ~/.kiro/agents && echo {escaped} > {_CONTAINER_AGENT_PATH}"),
        )

    @with_prompt_template
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        """Execute a task using kiro-cli in headless mode."""
        env = self._kiro_env()
        use_session_mount = self._use_session_mount()

        # In session-mount mode, reuse the host login (auth tables only) and pin
        # a clean config so the trial runs with the built-in default agent.
        if use_session_mount:
            await self._mount_host_session(environment)

        # Write MCP server config if any servers are configured
        mcp_servers = self._build_mcp_json(self.mcp_servers)
        if mcp_servers:
            mcp_json_str = json.dumps({"mcpServers": mcp_servers}, indent=2)
            escaped_mcp = shlex.quote(mcp_json_str)
            await self.exec_as_agent(
                environment,
                command=(
                    f"mkdir -p ~/.kiro/settings && echo {escaped_mcp} > ~/.kiro/settings/mcp.json"
                ),
                env=env or None,
            )

        escaped_instruction = shlex.quote(instruction)
        # Pin the model via an agent config (--model is a no-op in headless v2).
        agent_flag = ""
        if self.model_name:
            await self._write_model_pin_agent(environment)
            agent_flag = f"--agent {shlex.quote(_MODEL_PIN_AGENT)} "
        cli_flags = self.build_cli_flags()
        extra_flags = (cli_flags + " ") if cli_flags else ""

        # Copy skills into ~/.kiro/skills/. kiro-cli's default agent loads skills
        # from ~/.kiro/skills/ (global) at chat start, including in headless mode:
        # it lists each skill's name/description and reads the full SKILL.md when
        # a request matches. Verified with kiro-cli 2.21.2 that a skill present
        # only in ~/.kiro/skills/ is discovered and used.
        #
        # Skipped entirely in session-mount mode: that path deliberately keeps all
        # host ~/.kiro content (including skills) out of the trial for a clean slate.
        #
        # Write the resolved home and copied-skill count to /logs/agent/ (which is
        # collected with the run) rather than stdout: environment.exec discards
        # command stdout, so an echoed marker would never surface in the logs.
        # This makes a run that staged no skills easy to spot (global=0 means the
        # copy found nothing at skills_dir).
        if self.skills_dir and not use_session_mount:
            src = shlex.quote(self.skills_dir)
            await self.exec_as_agent(
                environment,
                command=(
                    f"mkdir -p ~/.kiro/skills /logs/agent && "
                    f"cp -r {src}/* ~/.kiro/skills/ 2>/dev/null || true; "
                    f'{{ echo "kiro-skills: HOME=$HOME src={self.skills_dir}"; '
                    f'echo "kiro-skills: global=$(ls ~/.kiro/skills 2>/dev/null | wc -l)"; }} '
                    f"| tee /logs/agent/kiro-skills.log"
                ),
                env=env or None,
            )

        # Emit structured stream-json (JSON Lines) as the single source for the
        # ATIF trajectory. kiro-cli headless does NOT persist conversations to its
        # SQLite DB (that only happens for interactive sessions), so stream-json is
        # the only reliable record of steps/tool-calls/cost for a headless run.
        run_command = (
            f"{_PATH_PREFIX}"
            f"kiro-cli chat --trust-all-tools --no-interactive "
            f"--output-format stream-json "
            f"{agent_flag}"
            f"{extra_flags}"
            f"{escaped_instruction} </dev/null "
            f"> /logs/agent/{_STREAM_JSON_FILENAME} 2> /logs/agent/{_STREAM_ERR_FILENAME}"
        )
        await self.exec_as_agent(environment, command=run_command, env=env or None)

    def populate_context_post_run(self, context: AgentContext) -> None:
        """Build the ATIF trajectory from kiro-cli's stream-json output.

        stream-json (JSON Lines) is the single trajectory source: headless runs
        do not persist a conversation to SQLite, so this is the only reliable
        record of the run's steps, tool calls, and cost.
        """
        stream_path = self.logs_dir / _STREAM_JSON_FILENAME
        if not stream_path.exists():
            self.logger.debug("No Kiro CLI stream-json found at %s", stream_path)
            return

        events = self._read_stream_json(stream_path)
        if not events:
            self.logger.debug("No stream-json events parsed from %s", stream_path)
            return

        try:
            trajectory = self._stream_json_to_trajectory(events)
        except Exception as exc:
            self.logger.debug("Failed to convert stream-json to trajectory: %s", exc)
            return
        if not trajectory:
            self.logger.debug("Failed to convert stream-json to trajectory")
            return

        trajectory_path = self.logs_dir / "trajectory.json"
        try:
            with open(trajectory_path, "w", encoding="utf-8") as f:
                json.dump(trajectory.to_json_dict(), f, indent=2, ensure_ascii=False)
            self.logger.debug("Wrote Kiro CLI trajectory to %s", trajectory_path)
        except OSError as exc:
            self.logger.debug("Failed to write trajectory: %s", exc)
            return

        if trajectory.final_metrics:
            context.cost_usd = trajectory.final_metrics.total_cost_usd

    def _read_stream_json(self, path: Path) -> list[dict[str, Any]]:
        """Parse a stream-json (JSON Lines) file into a list of event dicts.

        Malformed lines are skipped so a single truncated/partial line (e.g. an
        interrupted run) never discards the whole trajectory.
        """
        events: list[dict[str, Any]] = []
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(obj, dict):
                        events.append(obj)
        except OSError as exc:
            self.logger.debug("Failed to read stream-json %s: %s", path, exc)
            return []
        return events

    def _stream_json_to_trajectory(self, events: list[dict[str, Any]]) -> Trajectory | None:
        """Convert kiro-cli stream-json events into an ATIF Trajectory.

        Event model (kiro-cli ``--output-format stream-json``):

        - ``runStarted``  — ``data.engine``.
        - ``metadata``    — repeated; the final one carries ``meteringUsage``
          (credits), ``turnDurationMs``, ``sessionId``.
        - ``sessionUpdate`` with ``data.update.sessionUpdate`` one of:
          ``tool_call`` (new call: ``toolCallId``, ``rawInput`` args,
          ``_meta.kiro.toolName``), ``tool_call_update`` (``status``,
          ``rawOutput.items[].Json`` with exit_status/stdout/stderr),
          ``agent_message_chunk`` (streamed user-facing text), and
          ``agent_thought_chunk`` (the model's reasoning/thinking stream).
        - ``runFinished`` — ``status``, ``stopReason``, ``finalText``.

        Steps are emitted **in event order** so assistant text is interleaved
        with the tool calls it precedes, rather than collapsed into one final
        blob. Concretely: a run of ``agent_message_chunk`` text that is
        immediately followed by tool calls is the model's narration *before*
        those calls and becomes the ``message`` of the next tool step; a run of
        text with no following tool call is the final answer and becomes the last
        step. Each tool call is one Step (tool_call + its observation).

        ``agent_thought_chunk`` (reasoning) is accumulated the same way and
        attached to the same step boundary, but stored in the structured
        ``reasoning_content`` field rather than ``message`` — keeping the
        user-facing text and the private reasoning distinct.
        """
        session_id = "unknown"
        engine: str | None = None
        credits_total = 0.0
        credits_seen = False
        duration_ms: int | None = None
        final_status: str | None = None
        final_text: str | None = None

        # Ordered scaffolding: tool-call entries in first-seen order, plus the
        # assistant text/thought that preceded each. `pending_*` accumulate
        # agent_message_chunk / agent_thought_chunk until the next tool call
        # consumes them (or the run ends, making them the final answer step).
        tool_calls: dict[str, dict[str, Any]] = {}
        tool_order: list[str] = []
        pending_text = ""
        pending_thought = ""
        trailing_text = ""
        trailing_thought = ""

        for event in events:
            etype = event.get("type")
            data = event.get("data") or {}
            if data.get("sessionId"):
                session_id = data["sessionId"]

            if etype == "runStarted":
                engine = data.get("engine")
            elif etype == "metadata":
                metering = data.get("meteringUsage")
                if metering:
                    credits_seen = True
                    credits_total = sum(float(m.get("value", 0) or 0) for m in metering)
                if data.get("turnDurationMs") is not None:
                    duration_ms = data.get("turnDurationMs")
            elif etype == "runFinished":
                final_status = data.get("status")
                final_text = data.get("finalText")
            elif etype == "sessionUpdate":
                update = data.get("update") or {}
                kind = update.get("sessionUpdate")
                if kind == "agent_message_chunk":
                    content = update.get("content") or {}
                    text = content.get("text") if isinstance(content, dict) else None
                    if text:
                        pending_text += text
                        trailing_text += text
                elif kind == "agent_thought_chunk":
                    content = update.get("content") or {}
                    text = content.get("text") if isinstance(content, dict) else None
                    if text:
                        pending_thought += text
                        trailing_thought += text
                elif kind == "tool_call":
                    call_id = update.get("toolCallId") or ""
                    if not call_id:
                        continue
                    if call_id not in tool_calls:
                        tool_order.append(call_id)
                        # The text/thought accumulated since the last tool call
                        # is this call's pre-call narration and reasoning. Only
                        # the first call in a parallel batch claims the run.
                        tool_calls[call_id] = {
                            "reasoning": pending_text.strip(),
                            "thought": pending_thought.strip(),
                        }
                        pending_text = ""
                        pending_thought = ""
                        trailing_text = ""
                        trailing_thought = ""
                    entry = tool_calls[call_id]
                    entry["name"] = ((update.get("_meta") or {}).get("kiro") or {}).get("toolName")
                    entry["args"] = update.get("rawInput") or {}
                    entry["title"] = update.get("title")
                elif kind == "tool_call_update":
                    call_id = update.get("toolCallId") or ""
                    if not call_id:
                        continue
                    if call_id not in tool_calls:
                        tool_order.append(call_id)
                        tool_calls[call_id] = {
                            "reasoning": pending_text.strip(),
                            "thought": pending_thought.strip(),
                        }
                        pending_text = ""
                        pending_thought = ""
                        trailing_text = ""
                        trailing_thought = ""
                    entry = tool_calls[call_id]
                    if update.get("rawInput") and not entry.get("args"):
                        entry["args"] = update.get("rawInput")
                    if update.get("title") and not entry.get("title"):
                        entry["title"] = update.get("title")
                    text, is_error = self._extract_tool_output(update)
                    if text is not None:
                        entry["observation"] = text
                    if is_error:
                        entry["is_error"] = True

        steps: list[Step] = []
        n_tool_errors = 0
        for call_id in tool_order:
            call = tool_calls[call_id]
            if call.get("is_error"):
                n_tool_errors += 1
            args = dict(call.get("args") or {})
            purpose = args.pop("__tool_use_purpose", None)
            observation = None
            if call.get("observation") is not None:
                observation = Observation(
                    results=[ObservationResult(source_call_id=call_id, content=call["observation"])]
                )
            # Prefer the model's own pre-call narration as the step message;
            # fall back to the tool's declared purpose, then its title.
            message = call.get("reasoning") or purpose or call.get("title") or ""
            steps.append(
                Step(
                    step_id=len(steps) + 1,
                    source="agent",
                    message=message,
                    reasoning_content=call.get("thought") or None,
                    tool_calls=[
                        ToolCall(
                            tool_call_id=call_id,
                            function_name=call.get("name") or "unknown",
                            arguments=args,
                        )
                    ],
                    observation=observation,
                    model_name=self.model_name,
                )
            )

        # Text left after the last tool call is the final answer.
        final_answer = trailing_text.strip() or (final_text or "").strip()
        final_reasoning = trailing_thought.strip() or None
        if final_answer or final_reasoning:
            steps.append(
                Step(
                    step_id=len(steps) + 1,
                    source="agent",
                    message=final_answer,
                    reasoning_content=final_reasoning,
                    model_name=self.model_name,
                )
            )

        if not steps:
            return None

        metrics = FinalMetrics(
            total_steps=len(steps),
            total_cost_usd=credits_total if credits_seen else None,
            extra={
                "total_tool_calls": len(tool_order),
                "total_tool_calls_errors": n_tool_errors,
                "engine": engine,
                "turn_duration_ms": duration_ms,
                "stop_status": final_status,
            },
        )

        return Trajectory(
            schema_version="ATIF-v1.7",
            session_id=session_id,
            agent=Agent(
                name="kiro-cli",
                version=self._version or "unknown",
                model_name=self.model_name,
            ),
            steps=steps,
            final_metrics=metrics,
        )

    @staticmethod
    def _extract_tool_output(update: dict[str, Any]) -> tuple[str | None, bool]:
        """Return (observation_text, is_error) from a ``tool_call_update`` payload.

        Prefers the structured ``rawOutput.items[].Json`` (exit_status/stdout/
        stderr) emitted on completion; falls back to the streamed ``content``
        text. ``is_error`` is inferred from a non-zero exit status.
        """
        raw_output = update.get("rawOutput") or {}
        items = raw_output.get("items") if isinstance(raw_output, dict) else None
        if items:
            parts: list[str] = []
            is_error = False
            for item in items:
                if not isinstance(item, dict):
                    parts.append(str(item))
                    continue
                payload = item.get("Json")
                if isinstance(payload, dict):
                    exit_status = str(payload.get("exit_status", ""))
                    if exit_status and exit_status.strip() not in ("exit status: 0", "0"):
                        is_error = True
                    parts.append(json.dumps(payload))
                elif "Text" in item:
                    parts.append(str(item.get("Text")))
                else:
                    parts.append(json.dumps(item))
            return (" ".join(parts) if parts else None), is_error

        content = update.get("content")
        if isinstance(content, list):
            texts: list[str] = []
            for c in content:
                if isinstance(c, dict):
                    inner = c.get("content")
                    if isinstance(inner, dict) and inner.get("text"):
                        texts.append(inner["text"])
            if texts:
                return " ".join(texts), False
        return None, False
