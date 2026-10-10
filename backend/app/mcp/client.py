"""Small, read-only Streamable HTTP MCP client wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass(frozen=True)
class MCPServer:
    name: str
    url: str
    headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 30.0


class MCPClientError(RuntimeError):
    """Raised when an MCP server cannot complete a tool call."""


class MCPHttpClient:
    """Open a short-lived MCP session for each tool invocation.

    Short-lived sessions avoid sharing mutable session state between concurrent
    FastAPI requests. The official MCP SDK handles initialization, protocol
    negotiation and Streamable HTTP framing.
    """

    async def call_tool(
        self,
        server: MCPServer,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
        except ImportError as exc:  # pragma: no cover - depends on deployment
            raise MCPClientError(
                "MCP client dependency is missing; run pip install -r requirements.txt"
            ) from exc

        try:
            timeout = httpx.Timeout(server.timeout_seconds)
            async with httpx.AsyncClient(headers=server.headers, timeout=timeout) as http_client:
                async with streamable_http_client(
                    server.url,
                    http_client=http_client,
                ) as (read_stream, write_stream, _):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        result = await session.call_tool(tool_name, arguments=arguments)
        except Exception as exc:
            raise MCPClientError(
                f"MCP server {server.name!r} failed while calling {tool_name!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if getattr(result, "isError", False):
            raise MCPClientError(
                f"MCP server {server.name!r} returned an error for {tool_name!r}"
            )

        parts: list[str] = []
        for item in getattr(result, "content", []) or []:
            text = getattr(item, "text", None)
            if text:
                parts.append(str(text))
        structured = getattr(result, "structuredContent", None)
        if structured and not parts:
            import json

            parts.append(json.dumps(structured, ensure_ascii=False))
        return "\n\n".join(parts).strip()

    async def list_tools(self, server: MCPServer) -> list[str]:
        """List tool names without invoking any provider operation."""
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
        except ImportError as exc:  # pragma: no cover - depends on deployment
            raise MCPClientError(
                "MCP client dependency is missing; run pip install -r requirements.txt"
            ) from exc

        try:
            timeout = httpx.Timeout(server.timeout_seconds)
            async with httpx.AsyncClient(headers=server.headers, timeout=timeout) as http_client:
                async with streamable_http_client(
                    server.url,
                    http_client=http_client,
                ) as (read_stream, write_stream, _):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        result = await session.list_tools()
        except Exception as exc:
            raise MCPClientError(
                f"MCP server {server.name!r} failed during tools/list: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return [tool.name for tool in result.tools]
