"""CRM access through the lacrm-mcp server over stdio (langchain-mcp-adapters).

The client lookup starts the server with --read-only, so write tools do not even exist in that session.
Only the execute node opens a write session, and it may call only add_note and create_task.
"""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Protocol

from langchain_mcp_adapters.client import MultiServerMCPClient

from .config import Settings, lacrm_api_key
from .security import redact

WRITE_TOOLS = ("add_note", "create_task")
_CHILD_ENV = {"PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "TEMP", "TMP", "USERPROFILE",
              "HOMEDRIVE", "HOMEPATH", "HOME", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA", "PROGRAMFILES",
              "UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "PYTHONPYCACHEPREFIX", "LACRM_TIMEOUT_SECONDS",
              "LACRM_MAX_RETRIES", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "SSL_CERT_FILE", "LANG"}


class CrmError(Exception):
    pass


class Writer(Protocol):
    async def call(self, tool: str, args: dict[str, Any]) -> dict: ...


class Crm(Protocol):
    async def search(self, query: str) -> list[dict]: ...
    def writer(self): ...  # async context manager yielding a Writer


def _result_to_dict(result: Any, secret: str) -> dict:
    if getattr(result, "isError", False):
        text = " ".join(getattr(c, "text", "") for c in (result.content or []))
        raise CrmError(redact(text or "tool error", [secret])[:500])
    sc = getattr(result, "structuredContent", None)
    if isinstance(sc, dict):
        return sc["result"] if set(sc) == {"result"} and isinstance(sc["result"], dict) else sc
    for c in result.content or []:
        if getattr(c, "type", "") == "text":
            try:
                return json.loads(c.text)
            except json.JSONDecodeError:
                return {"text": c.text}
    return {}


class _McpWriter:
    def __init__(self, session, secret: str):
        self._s, self._secret = session, secret
        self.calls: list[tuple[str, dict]] = []

    async def call(self, tool: str, args: dict[str, Any]) -> dict:
        if tool not in WRITE_TOOLS:
            raise CrmError(f"tool {tool!r} is not allowed")
        self.calls.append((tool, args))
        return _result_to_dict(await self._s.call_tool(tool, args), self._secret)


class McpCrm:
    def __init__(self, settings: Settings):
        self.settings = settings

    def _client(self, read_only: bool) -> tuple[MultiServerMCPClient, str]:
        key = lacrm_api_key()
        if not key:
            raise CrmError("LACRM_API_KEY is not set (environment variable).")
        # Allowlisted environment: no PYTHONHOME/VIRTUAL_ENV from our own venv, no unrelated secrets.
        env = {k: v for k, v in os.environ.items() if k.upper() in _CHILD_ENV}
        own_venv = os.environ.get("VIRTUAL_ENV", "")
        if own_venv and "PATH" in {k.upper() for k in env}:
            pk = next(k for k in env if k.upper() == "PATH")
            env[pk] = os.pathsep.join(p for p in env[pk].split(os.pathsep) if not p.lower().startswith(own_venv.lower()))
        if not self.settings.lacrm_mcp_dir:
            raise CrmError("LACRM_MCP_DIR is not set: point it at your lacrm-mcp checkout.")
        if self.settings.lacrm_mcp_env:
            env["UV_PROJECT_ENVIRONMENT"] = self.settings.lacrm_mcp_env  # never our own venv
        else:
            env.pop("UV_PROJECT_ENVIRONMENT", None)
        env["LACRM_API_KEY"] = key
        args = ["run", "--quiet", "--directory", self.settings.lacrm_mcp_dir, "lacrm-mcp"]
        if read_only:
            args.append("--read-only")
            env["LACRM_READ_ONLY"] = "1"
        else:
            env.pop("LACRM_READ_ONLY", None)
        return MultiServerMCPClient({"lacrm": {"transport": "stdio", "command": "uv", "args": args, "env": env}}), key

    async def search(self, query: str) -> list[dict]:
        client, key = self._client(read_only=True)
        async with client.session("lacrm") as s:
            data = _result_to_dict(await s.call_tool("search_contacts", {"query": query[:200], "max_results": 10}), key)
        return list(data.get("results") or [])

    @asynccontextmanager
    async def writer(self) -> AsyncIterator[_McpWriter]:
        client, key = self._client(read_only=False)
        async with client.session("lacrm") as s:
            yield _McpWriter(s, key)


class OfflineCrm:
    """No CRM at all (evals): the client record is a fixed fake id, writes are refused."""

    def __init__(self, client_name: str = "", contact_id: str = "1"):
        self.client_name, self.contact_id = client_name, contact_id

    async def search(self, query: str) -> list[dict]:
        return [{"contact_id": self.contact_id, "name": self.client_name or query, "is_company": True}]

    @asynccontextmanager
    async def writer(self):
        raise CrmError("offline CRM: writes are not possible")
        yield  # pragma: no cover
