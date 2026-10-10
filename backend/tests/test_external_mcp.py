"""Offline tests for curated external legal MCP adapters."""

import asyncio


class FakeMCPClient:
    def __init__(self):
        self.calls = []

    async def call_tool(self, server, tool_name, arguments):
        self.calls.append((server.name, tool_name, arguments))
        return f"result from {server.name}/{tool_name}"


def test_statute_and_case_tools_map_to_provider_schemas(monkeypatch):
    from app.config import settings
    from app.mcp.legal_tools import ExternalLegalMCPService

    monkeypatch.setattr(settings, "EXTERNAL_MCP_ENABLED", True)
    monkeypatch.setattr(settings, "FLK_MCP_ENABLED", True)
    monkeypatch.setattr(settings, "RMFYALK_MCP_ENABLED", True)
    fake = FakeMCPClient()
    service = ExternalLegalMCPService(fake)

    statute = asyncio.run(service.execute("search_statutes", {
        "query": "民法典 合同效力 13800138000",
        "effective_only": True,
        "max_results": 50,
    }))
    case = asyncio.run(service.execute("search_court_cases", {
        "query": "格式条款",
        "max_results": 5,
    }))

    assert "国家法律法规数据库" == statute.metadata["title"]
    assert fake.calls[0][1] == "flk_search"
    assert fake.calls[0][2]["params"]["sxx"] == 3
    assert fake.calls[0][2]["params"]["page_size"] == 10
    assert "13800138000" not in fake.calls[0][2]["params"]["search_content"]
    assert fake.calls[1][1] == "rmfyalk_search"
    assert fake.calls[1][2]["params"]["search_field"] == "qw"
    assert case.metadata["mcp_server"] == "rmfyalk"


def test_tavily_registers_only_with_key_and_enforces_domain_allowlist(monkeypatch):
    from app.config import settings
    from app.mcp.legal_tools import ExternalLegalMCPService

    monkeypatch.setattr(settings, "EXTERNAL_MCP_ENABLED", True)
    monkeypatch.setattr(settings, "TAVILY_MCP_ENABLED", True)
    monkeypatch.setattr(settings, "TAVILY_API_KEY", "test-key")
    fake = FakeMCPClient()
    service = ExternalLegalMCPService(fake)

    assert "search_official_legal_web" in service.enabled_tools
    result = asyncio.run(service.execute("search_official_legal_web", {
        "query": "最高人民法院 最新司法解释",
        "max_results": 99,
    }))
    assert fake.calls[0][1] == "tavily-search"
    assert fake.calls[0][2]["max_results"] == 8
    assert fake.calls[0][2]["include_domains"] == settings.TAVILY_LEGAL_DOMAINS
    assert result.metadata["mcp_server"] == "tavily"

    try:
        asyncio.run(service.execute("fetch_official_legal_page", {
            "url": "https://example.com/untrusted",
        }))
    except ValueError as exc:
        assert "allow-listed" in str(exc)
    else:
        raise AssertionError("non-official domains must be rejected")


def test_agent_registry_exposes_only_enabled_external_tools():
    from app.agentic.tools import AgentToolRegistry, ToolExecution

    async def retrieve(query, collection, top_k):
        return ToolExecution(content="local")

    async def one_arg(value):
        return ToolExecution(content=value)

    async def external(name, arguments):
        return ToolExecution(content=f"{name}:{arguments}")

    registry = AgentToolRegistry(
        retrieve=retrieve,
        crime=one_arg,
        source=one_arg,
        external=external,
        external_tool_names={"search_statutes", "search_court_cases"},
    )
    names = {tool.name for tool in registry.tools}
    assert "search_statutes" in names
    assert "search_court_cases" in names
    assert "search_official_legal_web" not in names
