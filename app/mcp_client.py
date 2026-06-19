"""MCP server integration.

Reads mcp.json, connects each declared server (stdio for `command`, streamable
HTTP for `url`), lists its tools, and exposes them as registry Tools namespaced
`mcp__<server>__<tool>`. Sessions are kept open for the app's lifetime via a
shared AsyncExitStack and closed on shutdown.
"""
from __future__ import annotations

import json
import os
from contextlib import AsyncExitStack
from typing import Any

from .config import MCP_CONFIG_PATH
from .tools.registry import Tool


def _load_config() -> dict[str, Any]:
    if not MCP_CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(MCP_CONFIG_PATH.read_text()).get("servers", {})
    except Exception:
        return {}


class MCPManager:
    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self.sessions: dict[str, Any] = {}
        self.tools: list[Tool] = []

    async def start(self) -> list[str]:
        """Connect all configured servers. Returns a list of status strings."""
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        status: list[str] = []
        servers = _load_config()
        for name, cfg in servers.items():
            try:
                if "command" in cfg:
                    params = StdioServerParameters(
                        command=cfg["command"],
                        args=cfg.get("args", []),
                        env={**os.environ, **cfg.get("env", {})},
                    )
                    read, write = await self._stack.enter_async_context(stdio_client(params))
                elif "url" in cfg:
                    from mcp.client.streamable_http import streamablehttp_client

                    read, write, _ = await self._stack.enter_async_context(
                        streamablehttp_client(cfg["url"])
                    )
                else:
                    status.append(f"{name}: skipped (need 'command' or 'url')")
                    continue

                session = await self._stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                self.sessions[name] = session

                listed = await session.list_tools()
                for t in listed.tools:
                    self.tools.append(self._wrap(name, session, t))
                status.append(f"{name}: {len(listed.tools)} tool(s)")
            except Exception as exc:  # one bad server shouldn't kill startup
                status.append(f"{name}: failed ({type(exc).__name__}: {exc})")
        return status

    def _wrap(self, server: str, session: Any, mcp_tool: Any) -> Tool:
        async def handler(**kwargs: Any) -> str:
            result = await session.call_tool(mcp_tool.name, kwargs)
            parts: list[str] = []
            for block in getattr(result, "content", []) or []:
                text = getattr(block, "text", None)
                parts.append(text if text is not None else str(block))
            return "\n".join(parts) or "(no content)"

        schema = getattr(mcp_tool, "inputSchema", None) or {"type": "object", "properties": {}}
        return Tool(
            name=f"mcp__{server}__{mcp_tool.name}",
            description=(getattr(mcp_tool, "description", "") or f"MCP tool {mcp_tool.name}")[:1024],
            parameters=schema,
            handler=handler,
        )

    async def stop(self) -> None:
        try:
            await self._stack.aclose()
        except Exception:
            pass
