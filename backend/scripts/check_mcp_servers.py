"""Check MCP protocol connectivity without invoking any search tool."""

from __future__ import annotations

import asyncio

from app.config import settings
from app.mcp import ExternalLegalMCPService


async def main() -> None:
    service = ExternalLegalMCPService()
    servers = [service.flk, service.cases]
    if settings.TAVILY_MCP_ENABLED and settings.TAVILY_API_KEY:
        servers.append(service.tavily)
    for server in servers:
        tools = await service.client.list_tools(server)
        print(f"{server.name}: connected ({len(tools)} tools): {', '.join(tools)}")


if __name__ == "__main__":
    asyncio.run(main())
