"""Uniform tool abstraction shared by builtin, web, skill, and MCP tools."""
from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema for the arguments object
    handler: Callable[..., Any]  # sync or async; receives **kwargs

    def schema(self) -> dict[str, Any]:
        """OpenAI `tools` array entry."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def run(self, args: dict[str, Any]) -> str:
        if inspect.iscoroutinefunction(self.handler):
            result = await self.handler(**args)
        else:
            # Offload blocking handlers (bash, file IO, httpx) off the event loop.
            result = await asyncio.to_thread(self.handler, **args)
        return result if isinstance(result, str) else str(result)


class Registry:
    """Collects tools and dispatches calls by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def add(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def extend(self, tools: list[Tool]) -> None:
        for t in tools:
            self.add(t)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    async def call(self, name: str, args: dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'."
        try:
            return await tool.run(args)
        except Exception as exc:  # surface to the model, don't crash the loop
            return f"Error running tool '{name}': {type(exc).__name__}: {exc}"
